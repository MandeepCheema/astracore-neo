"""Conv2d → matmul lowering via im2col (F1-C4).

Takes a single conv layer (INT8 weights + INT8 input) and emits a
Program that computes it on the weight-stationary NPU tile as a
sequence of `compile_matmul_chained`-style tiles, chained via
`cfg_acc_init_mode` across K-chunks.

## Mapping conv → tile operations

Given:
  input  (1, C_in, H_in, W_in)   INT8
  weight (C_out, C_in, k_h, k_w) INT8
  stride s, pad p (pad_top, pad_left, pad_bottom, pad_right)
  output (1, C_out, H_out, W_out) INT32

im2col-style reshape:
  M       = H_out * W_out                       (spatial positions)
  K_total = C_in * k_h * k_w                    (reduction dimension)
  N       = C_out                               (output channels)
  W_mat   = weight reshaped to (K_total, N)     (row-major over C_in,kh,kw)

Per-output:
  out[m, n] = sum_{k in 0..K_total} input_im2col[m, k] * W_mat[k, n]

The NPU holds one (N_rows × N_cols) weight matrix per tile. The tile's
natural GEMM primitive is a reduction of length N_rows with
`cfg_k=1`; longer reductions fold through `cfg_acc_init_mode` chaining.

Loop structure emitted by compile_conv2d:

    for n_chunk in range(ceil(N / N_cols)):
        for k_chunk in range(ceil(K_total / N_rows)):
            LoadWeight x (N_rows * N_cols)   # fresh slice of W_mat
            for m in range(M):
                if k_chunk > 0:
                    ReadAO(m)                 # seed = this cell's
                                              # partial from prev
                                              # k_chunk (goes into
                                              # last_ao_read)
                LoadActivation(m, im2col_vec_for_this_m_and_k_chunk)
                RunTile(k=1, ai_base=m, ao_base=m,
                        acc_init_mode=(0 if k_chunk==0 else 1),
                        acc_init_data=(0 if k_chunk==0
                                       else ACC_INIT_FROM_PREV_AO))
        for m in range(M):
            ReadAO(m)                         # output capture

Output captures are the final M reads of each n_chunk block. The
ConvCompileResult records their positions in `read_log` so reassembly
is unambiguous even though intermediate seed-reads share the log.

## Scope in v1 (F1-C4)

- Batch size 1.
- groups = 1 (no depthwise; YOLOv8 doesn't need it).
- No dilation.
- M ≤ AO_DEPTH (16 in current silicon). Larger M needs multi-round
  output readout — straightforward extension, not in v1.
- INT8 weights and activations, INT32 accumulator output (no AFU
  here; F1-C5 adds activation + bias + dequant post-compile).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .compiler import (
    ACC_INIT_FROM_PREV_AO,
    AI_SRAM_DEPTH,
    LoadActivation,
    LoadWeight,
    Program,
    ReadAO,
    RunTile,
    _pack_bytes,
)

AO_DEPTH = 16  # hardware AO SRAM depth on current silicon


# ---------------------------------------------------------------------------
# Direct numpy INT8 conv reference — the bit-exact oracle
# ---------------------------------------------------------------------------
def _as_int8_scalar(v) -> int:
    v = int(v) & 0xFF
    return v - 0x100 if v & 0x80 else v


def reference_conv2d_int8(x: np.ndarray,
                           w: np.ndarray,
                           *,
                           stride: Tuple[int, int] = (1, 1),
                           pad: Tuple[int, int, int, int] = (0, 0, 0, 0),
                           ) -> np.ndarray:
    """Direct-loop INT8 conv producing the exact INT32 accumulator
    tensor the NPU would. Used as the bit-exact oracle for F1-C4.

    x     : (1, C_in, H_in, W_in) int8 (or int-castable)
    w     : (C_out, C_in, k_h, k_w) int8 (or int-castable)
    stride: (s_h, s_w)
    pad   : (pad_top, pad_left, pad_bottom, pad_right)
    returns (1, C_out, H_out, W_out) int32
    """
    assert x.ndim == 4 and x.shape[0] == 1, f"x shape {x.shape}"
    assert w.ndim == 4, f"w shape {w.shape}"
    _, C_in, H_in, W_in = x.shape
    C_out, C_in_w, k_h, k_w = w.shape
    assert C_in == C_in_w, f"C_in mismatch x={C_in} w={C_in_w}"
    s_h, s_w = stride
    pt, pl, pb, pr = pad

    H_out = (H_in + pt + pb - k_h) // s_h + 1
    W_out = (W_in + pl + pr - k_w) // s_w + 1

    # Signed int8 view on the element values (without changing dtype).
    xi = x.astype(np.int32)
    wi = w.astype(np.int32)

    out = np.zeros((1, C_out, H_out, W_out), dtype=np.int32)
    for h_out in range(H_out):
        for w_out in range(W_out):
            for co in range(C_out):
                acc = 0
                for ci in range(C_in):
                    for dh in range(k_h):
                        for dw in range(k_w):
                            h_in = h_out * s_h + dh - pt
                            w_in = w_out * s_w + dw - pl
                            if 0 <= h_in < H_in and 0 <= w_in < W_in:
                                acc += int(wi[co, ci, dh, dw]) * int(
                                    xi[0, ci, h_in, w_in])
                out[0, co, h_out, w_out] = acc
    return out


# ---------------------------------------------------------------------------
# im2col helpers
# ---------------------------------------------------------------------------
def _im2col_index_list(C_in: int, k_h: int, k_w: int) -> List[Tuple[int, int, int]]:
    """Row-major enumeration of the K_total dimension as (ci, dh, dw) triples.
    Matches the order the weight tensor is reshaped to (K_total, C_out)."""
    return [(ci, dh, dw)
            for ci in range(C_in)
            for dh in range(k_h)
            for dw in range(k_w)]


def _gather_input_vector(x_nchw: np.ndarray,
                          h_out: int, w_out: int,
                          k_slice_indices: List[Tuple[int, int, int]],
                          stride_h: int, stride_w: int,
                          pad_top: int, pad_left: int,
                          n_rows: int) -> List[int]:
    """Produce one N_rows-wide activation vector for spatial position
    (h_out, w_out), selecting K values from the receptive field at the
    given im2col k-slice indices. Zero-pads when the slice is shorter
    than N_rows (happens at the tail of a non-divisible K)."""
    _, C_in, H_in, W_in = x_nchw.shape
    vec: List[int] = [0] * n_rows
    for i, (ci, dh, dw) in enumerate(k_slice_indices):
        if i >= n_rows:
            break
        h_in = h_out * stride_h + dh - pad_top
        w_in = w_out * stride_w + dw - pad_left
        if 0 <= h_in < H_in and 0 <= w_in < W_in:
            vec[i] = _as_int8_scalar(x_nchw[0, ci, h_in, w_in])
        # else: zero-pad (out-of-bounds, counts as 0).
    return vec


# ---------------------------------------------------------------------------
# Compile result
# ---------------------------------------------------------------------------
@dataclass
class ConvCompileResult:
    """Everything the reassembler needs to turn simulate_program's
    read_log into an (1, C_out, H_out, W_out) INT32 tensor."""
    program: Program
    output_shape: Tuple[int, int, int, int]  # (1, C_out, H_out, W_out)
    n_rows: int
    n_cols: int
    K_total: int
    M: int                                    # = H_out * W_out
    n_chunks: int                             # ceil(C_out / n_cols)
    k_chunks: int                             # ceil(K_total / n_rows)
    # Indices into simulate_program's read_log that hold the final
    # output captures (one entry per (n_chunk, m) cell, in order).
    output_read_indices: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------
def compile_conv2d(w: np.ndarray,
                    x: np.ndarray,
                    *,
                    n_rows: int,
                    n_cols: int,
                    stride: Tuple[int, int] = (1, 1),
                    pad: Tuple[int, int, int, int] = (0, 0, 0, 0),
                    ai_base: int = 0,
                    ao_base: int = 0,
                    ) -> ConvCompileResult:
    """Lower a conv2d to a Program via im2col.

    Args:
        w: (C_out, C_in, k_h, k_w) INT8 weights.
        x: (1, C_in, H_in, W_in) INT8 activations.
        n_rows, n_cols: NPU systolic array dimensions.
        stride, pad: standard conv params; pad is (top, left, bottom,
            right).
        ai_base: start address in AI SRAM. One slot per M position
            (so M ≤ AI_SRAM_DEPTH - ai_base is required).
        ao_base: start address in AO SRAM. One slot per M position.

    Returns:
        ConvCompileResult holding the Program plus reassembly metadata.
    """
    assert x.ndim == 4 and x.shape[0] == 1, "v1 assumes batch=1"
    _, C_in, H_in, W_in = x.shape
    C_out, C_in_w, k_h, k_w = w.shape
    assert C_in == C_in_w
    s_h, s_w = stride
    pt, pl, pb, pr = pad
    H_out = (H_in + pt + pb - k_h) // s_h + 1
    W_out = (W_in + pl + pr - k_w) // s_w + 1

    K_total = C_in * k_h * k_w
    M = H_out * W_out
    k_chunks = (K_total + n_rows - 1) // n_rows
    n_chunks = (C_out + n_cols - 1) // n_cols

    # Multi-round M tiling: split M into rounds that fit in the
    # AI / AO SRAM. ai_base and ao_base are the *per-round* starts;
    # the same slot addresses get overwritten each round. A round
    # processes up to `M_per_round` spatial positions.
    ai_room = max(0, AI_SRAM_DEPTH - ai_base)
    ao_room = max(0, AO_DEPTH - ao_base)
    M_per_round = min(ai_room, ao_room, M) if M > 0 else 0
    if M_per_round <= 0:
        raise ValueError(
            f"no SRAM room for any M positions: ai_base={ai_base}, "
            f"ao_base={ao_base}, AI_SRAM_DEPTH={AI_SRAM_DEPTH}, "
            f"AO_DEPTH={AO_DEPTH}"
        )
    m_rounds = (M + M_per_round - 1) // M_per_round

    k_indices = _im2col_index_list(C_in, k_h, k_w)
    prog = Program()
    output_read_indices: List[int] = []
    read_counter = 0  # tracks position in the eventual read_log

    # Row-major output positions.
    m_to_hw: List[Tuple[int, int]] = [(m // W_out, m % W_out) for m in range(M)]

    for n_chunk in range(n_chunks):
        n_start = n_chunk * n_cols
        n_end = min(n_start + n_cols, C_out)

        for m_round in range(m_rounds):
            m_round_start = m_round * M_per_round
            m_round_end = min(m_round_start + M_per_round, M)
            m_in_round = m_round_end - m_round_start

            for kc in range(k_chunks):
                k_start = kc * n_rows
                k_end = min(k_start + n_rows, K_total)
                k_slice = k_indices[k_start:k_end]
                k_len = len(k_slice)

                # LoadWeight: N_rows × N_cols matrix for (kc, n_chunk).
                # Reloaded per round — the weights are stationary for
                # every M position *inside* a round.
                for r in range(n_rows):
                    for c in range(n_cols):
                        if r < k_len and (n_start + c) < n_end:
                            ci, dh, dw = k_slice[r]
                            co = n_start + c
                            val = _as_int8_scalar(w[co, ci, dh, dw])
                        else:
                            val = 0  # zero-pad the unused weight slots
                        prog.instructions.append(
                            LoadWeight(r * n_cols + c, val)
                        )

                # Per-M inner loop within this round: seed + activation + run.
                for m_local in range(m_in_round):
                    m_global = m_round_start + m_local
                    h_out, w_out = m_to_hw[m_global]
                    # Seed setup for chained k-chunks — ReadAO populates
                    # `last_ao_read` so the RunTile's
                    # ACC_INIT_FROM_PREV_AO can pick it up. k_chunk 0
                    # of a round always starts fresh (acc_init_mode=0),
                    # because AO slot contents from the previous round
                    # are for different M positions.
                    if kc > 0:
                        prog.instructions.append(ReadAO(ao_base + m_local))
                        read_counter += 1

                    vec = _gather_input_vector(
                        x, h_out, w_out, k_slice, s_h, s_w, pt, pl, n_rows)
                    prog.instructions.append(
                        LoadActivation(ai_base + m_local, _pack_bytes(vec))
                    )
                    prog.instructions.append(
                        RunTile(
                            k=1,
                            ai_base=ai_base + m_local,
                            ao_base=ao_base + m_local,
                            afu_mode=0,   # F1-C4/C5 emit raw INT32;
                                          # AFU / bias / dequant are
                                          # applied post-compile.
                            acc_init_mode=0 if kc == 0 else 1,
                            acc_init_data=0 if kc == 0
                                            else ACC_INIT_FROM_PREV_AO,
                        )
                    )

            # After all k_chunks for this round: capture this round's
            # M outputs. Each round produces m_in_round output cells.
            for m_local in range(m_in_round):
                prog.instructions.append(ReadAO(ao_base + m_local))
                output_read_indices.append(read_counter)
                read_counter += 1

    return ConvCompileResult(
        program=prog,
        output_shape=(1, C_out, H_out, W_out),
        n_rows=n_rows,
        n_cols=n_cols,
        K_total=K_total,
        M=M,
        n_chunks=n_chunks,
        k_chunks=k_chunks,
        output_read_indices=output_read_indices,
    )


# ---------------------------------------------------------------------------
# Reassembler
# ---------------------------------------------------------------------------
def reassemble_conv_output(read_log: List[Tuple[int, List[int]]],
                            result: ConvCompileResult) -> np.ndarray:
    """Turn the chronological read_log produced by
    simulate_program(..., return_read_log=True) into a (1, C_out,
    H_out, W_out) INT32 tensor.

    Uses `result.output_read_indices` to pick only the output captures
    — intermediate ReadAOs used as K-chunk seeds are ignored.
    """
    _, C_out, H_out, W_out = result.output_shape
    M = result.M
    out = np.zeros(result.output_shape, dtype=np.int64)

    # output_read_indices is flat: n_chunk × M entries, in that order.
    idx = 0
    for n_chunk in range(result.n_chunks):
        n_start = n_chunk * result.n_cols
        n_end = min(n_start + result.n_cols, C_out)
        width = n_end - n_start
        for m in range(M):
            h_out = m // W_out
            w_out = m % W_out
            _addr, vec = read_log[result.output_read_indices[idx]]
            idx += 1
            for c in range(width):
                out[0, n_start + c, h_out, w_out] = vec[c]
    return out.astype(np.int32)
