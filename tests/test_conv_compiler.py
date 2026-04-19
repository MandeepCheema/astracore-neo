"""Unit tests for tools/npu_ref/conv_compiler.py (F1-C4).

Every test compiles a conv via `compile_conv2d` → simulates the
Program → reassembles the output tensor → asserts it is bit-exact
against `reference_conv2d_int8` (our direct-loop oracle).

Covers:
  - 1×1 conv (simplest: no spatial reduction, K = C_in).
  - 3×3 stride 1 pad 1 (YOLO backbone pattern).
  - 3×3 stride 2 pad 1 (YOLO downsample).
  - 3×3 stride 1 pad 0 (no padding; checks out-of-bounds handling).
  - Asymmetric pad (e.g. pad=(1,0,1,0)).
  - C_out > N_COLS (N-split required).
  - K_total > N_ROWS (K-chunking required).
  - K_total > N_ROWS × 2 (three-way K chain).
  - Multi-chunk in both K and N simultaneously.
  - Negative-value edge cases (INT8 extremes).

Reference oracle is `reference_conv2d_int8`. A separate one-off
cross-check against torch.nn.functional.conv2d is available in the
side export venv (kept out of this test file to avoid the torch dep
in the core .venv).
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.npu_ref.compiler import simulate_program
from tools.npu_ref.conv_compiler import (
    compile_conv2d,
    reassemble_conv_output,
    reference_conv2d_int8,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _roundtrip(w, x, *, n_rows, n_cols, stride=(1, 1), pad=(0, 0, 0, 0)):
    res = compile_conv2d(w, x, n_rows=n_rows, n_cols=n_cols,
                          stride=stride, pad=pad)
    _, read_log = simulate_program(
        res.program, n_rows=n_rows, n_cols=n_cols, return_read_log=True)
    out = reassemble_conv_output(read_log, res)
    ref = reference_conv2d_int8(x, w, stride=stride, pad=pad)
    return out, ref, res


def _rand_int8(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(-50, 50, size=shape, dtype=np.int8)


# ---------------------------------------------------------------------------
# 1×1 conv: K_total = C_in, no spatial reduction
# ---------------------------------------------------------------------------
def test_1x1_conv_small_cin():
    w = _rand_int8((3, 2, 1, 1), seed=0)
    x = _rand_int8((1, 2, 3, 3), seed=1)
    out, ref, res = _roundtrip(w, x, n_rows=4, n_cols=4)
    assert out.shape == ref.shape == (1, 3, 3, 3)
    assert np.array_equal(out, ref)
    assert res.k_chunks == 1          # C_in=2 ≤ N_rows=4
    assert res.n_chunks == 1          # C_out=3 ≤ N_cols=4


def test_1x1_conv_cin_exceeds_n_rows():
    """C_in = 10 needs 3 K-chunks on a 4-row array."""
    w = _rand_int8((2, 10, 1, 1), seed=2)
    x = _rand_int8((1, 10, 3, 3), seed=3)
    out, ref, res = _roundtrip(w, x, n_rows=4, n_cols=4)
    assert np.array_equal(out, ref)
    assert res.k_chunks == 3          # ceil(10 / 4)


# ---------------------------------------------------------------------------
# 3×3 conv variants — the plan's acceptance target
# ---------------------------------------------------------------------------
def test_3x3_conv_stride1_pad1():
    """YOLO backbone pattern: H_out == H_in."""
    w = _rand_int8((2, 1, 3, 3), seed=4)
    x = _rand_int8((1, 1, 4, 4), seed=5)
    out, ref, _ = _roundtrip(w, x, n_rows=4, n_cols=4,
                              stride=(1, 1), pad=(1, 1, 1, 1))
    assert out.shape == (1, 2, 4, 4)
    assert np.array_equal(out, ref)


def test_3x3_conv_stride1_pad0():
    """Valid-pad conv: every spatial position has an in-bounds receptive
    field (tests the no-padding path of _gather_input_vector)."""
    w = _rand_int8((2, 1, 3, 3), seed=6)
    x = _rand_int8((1, 1, 4, 4), seed=7)
    out, ref, _ = _roundtrip(w, x, n_rows=4, n_cols=4,
                              stride=(1, 1), pad=(0, 0, 0, 0))
    assert out.shape == (1, 2, 2, 2)
    assert np.array_equal(out, ref)


def test_3x3_conv_stride2_pad1():
    """Downsample pattern: output is ceil(H_in/2)."""
    w = _rand_int8((2, 1, 3, 3), seed=8)
    x = _rand_int8((1, 1, 4, 4), seed=9)
    out, ref, _ = _roundtrip(w, x, n_rows=4, n_cols=4,
                              stride=(2, 2), pad=(1, 1, 1, 1))
    assert out.shape == (1, 2, 2, 2)
    assert np.array_equal(out, ref)


def test_3x3_conv_asymmetric_pad():
    """Asymmetric pad (top-left only) to catch index-math bugs in
    _gather_input_vector and the reference alike."""
    w = _rand_int8((2, 1, 3, 3), seed=10)
    x = _rand_int8((1, 1, 3, 3), seed=11)
    out, ref, _ = _roundtrip(w, x, n_rows=4, n_cols=4,
                              stride=(1, 1), pad=(1, 1, 0, 0))
    assert np.array_equal(out, ref)


# ---------------------------------------------------------------------------
# N-splitting — C_out > N_COLS
# ---------------------------------------------------------------------------
def test_conv_n_split_cout_exceeds_ncols():
    """C_out = 8 on 4-column array → 2 N-chunks."""
    w = _rand_int8((8, 1, 3, 3), seed=12)
    x = _rand_int8((1, 1, 4, 4), seed=13)
    out, ref, res = _roundtrip(w, x, n_rows=4, n_cols=4,
                                stride=(1, 1), pad=(1, 1, 1, 1))
    assert out.shape == (1, 8, 4, 4)
    assert res.n_chunks == 2
    assert np.array_equal(out, ref)


def test_conv_n_split_uneven_cout():
    """C_out = 6 on 4-column array: last chunk is narrower (2 cols used
    out of 4). Tests zero-pad on the trailing output columns."""
    w = _rand_int8((6, 2, 1, 1), seed=14)
    x = _rand_int8((1, 2, 3, 3), seed=15)
    out, ref, res = _roundtrip(w, x, n_rows=4, n_cols=4)
    assert out.shape == (1, 6, 3, 3)
    assert res.n_chunks == 2
    assert np.array_equal(out, ref)


# ---------------------------------------------------------------------------
# K-chunking — K_total > N_ROWS
# ---------------------------------------------------------------------------
def test_conv_k_chunked_two_way():
    """3×3 conv with C_in=1 gives K_total=9. On a 4-row array: 3 K-chunks.
    The third chunk has only 1 valid row (zero-padded)."""
    w = _rand_int8((2, 1, 3, 3), seed=16)
    x = _rand_int8((1, 1, 4, 4), seed=17)
    out, ref, res = _roundtrip(w, x, n_rows=4, n_cols=4,
                                stride=(1, 1), pad=(1, 1, 1, 1))
    assert res.k_chunks == 3          # ceil(9 / 4)
    assert np.array_equal(out, ref)


def test_conv_k_chunked_many():
    """3×3 conv, C_in=5 → K_total=45 → 12 K-chunks on 4-row array."""
    w = _rand_int8((2, 5, 3, 3), seed=18)
    x = _rand_int8((1, 5, 4, 4), seed=19)
    out, ref, res = _roundtrip(w, x, n_rows=4, n_cols=4,
                                stride=(1, 1), pad=(1, 1, 1, 1))
    assert res.k_chunks == 12
    assert np.array_equal(out, ref)


# ---------------------------------------------------------------------------
# Combined K + N splitting
# ---------------------------------------------------------------------------
def test_conv_k_and_n_split():
    """Both K-chunking (K=18, 5 chunks) and N-splitting (C_out=6, 2
    chunks). Exercises the full nested-loop structure."""
    w = _rand_int8((6, 2, 3, 3), seed=20)
    x = _rand_int8((1, 2, 3, 3), seed=21)
    out, ref, res = _roundtrip(w, x, n_rows=4, n_cols=4,
                                stride=(1, 1), pad=(1, 1, 1, 1))
    assert res.k_chunks == 5          # ceil(18 / 4)
    assert res.n_chunks == 2
    assert np.array_equal(out, ref)


# ---------------------------------------------------------------------------
# Numerical edge cases
# ---------------------------------------------------------------------------
def test_conv_int8_extreme_values():
    """Weights + activations at INT8 extremes: products up to 127²
    ≈ 16k per K-element. For K=9 with all extremes, accumulator up
    to ~145k which fits INT32 comfortably."""
    w = np.full((2, 1, 3, 3), 127, dtype=np.int8)
    # Use -127 not -128 to match the compiler's symmetric recipe
    # (the compiler treats -128 as signed-int correctly, but the test
    # intent is to cover the extremes the quantiser would produce).
    x = np.full((1, 1, 3, 3), -127, dtype=np.int8)
    out, ref, _ = _roundtrip(w, x, n_rows=4, n_cols=4,
                              stride=(1, 1), pad=(1, 1, 1, 1))
    assert np.array_equal(out, ref)


def test_conv_all_zeros_produces_zero_output():
    w = np.zeros((2, 1, 3, 3), dtype=np.int8)
    x = np.zeros((1, 1, 3, 3), dtype=np.int8)
    out, ref, _ = _roundtrip(w, x, n_rows=4, n_cols=4,
                              stride=(1, 1), pad=(1, 1, 1, 1))
    assert np.array_equal(out, ref)
    assert np.all(out == 0)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------
def test_m_exceeds_ai_sram_depth_uses_multi_round():
    """5×5 output = M=25, exceeds AI_SRAM_DEPTH=16. F1-C5 multi-round
    M tiling should handle this — emit 2 rounds of 16 + 9 and produce
    bit-exact output."""
    w = _rand_int8((2, 1, 1, 1), seed=22)
    x = _rand_int8((1, 1, 5, 5), seed=23)
    out, ref, res = _roundtrip(w, x, n_rows=4, n_cols=4,
                                stride=(1, 1), pad=(0, 0, 0, 0))
    assert res.M == 25
    assert np.array_equal(out, ref)


def test_multi_round_m_larger_shape():
    """M=64 forces 4 rounds on AI_SRAM_DEPTH=16. Exercises the
    round-edge arithmetic."""
    w = _rand_int8((2, 1, 1, 1), seed=24)
    x = _rand_int8((1, 1, 8, 8), seed=25)
    out, ref, res = _roundtrip(w, x, n_rows=4, n_cols=4,
                                stride=(1, 1), pad=(0, 0, 0, 0))
    assert res.M == 64
    assert np.array_equal(out, ref)


def test_multi_round_m_plus_k_and_n_split():
    """M=36, plus K-chunking (K=18) and N-splitting (C_out=6). Tests
    the full nested-loop structure under multi-round M."""
    w = _rand_int8((6, 2, 3, 3), seed=26)
    x = _rand_int8((1, 2, 6, 6), seed=27)
    out, ref, res = _roundtrip(w, x, n_rows=4, n_cols=4,
                                stride=(1, 1), pad=(1, 1, 1, 1))
    assert res.M == 36
    assert res.k_chunks == 5
    assert res.n_chunks == 2
    assert np.array_equal(out, ref)


# ---------------------------------------------------------------------------
# Reference oracle self-check
# ---------------------------------------------------------------------------
def test_compiler_scales_to_8x8_array():
    """GAP-3: compiler correctness at 8×8 array parameters (production
    AWS-F1 target is 64×64; intermediate 8×8 and 16×16 are the most
    we can afford to cocotb in WSL). The 4×4 unit tests prove the
    compiler's core logic; these check the n_rows / n_cols
    parameters are plumbed through every path (K-chunking cap,
    N-splitting boundary, multi-round M thresholds)."""
    rng = np.random.default_rng(0)
    x = rng.integers(-40, 40, size=(1, 4, 3, 3), dtype=np.int8)
    w = rng.integers(-10, 10, size=(16, 4, 3, 3), dtype=np.int8)
    res = compile_conv2d(w, x, n_rows=8, n_cols=8,
                          stride=(1, 1), pad=(1, 1, 1, 1))
    _, log = simulate_program(
        res.program, n_rows=8, n_cols=8, return_read_log=True)
    out = reassemble_conv_output(log, res)
    ref = reference_conv2d_int8(x, w, stride=(1, 1), pad=(1, 1, 1, 1))
    assert np.array_equal(out, ref)
    # Structural check: K_total=4×3×3=36, n_rows=8 → 5 K-chunks.
    assert res.k_chunks == 5
    # C_out=16, n_cols=8 → 2 N-chunks.
    assert res.n_chunks == 2


def test_compiler_scales_to_16x16_array():
    """Same as 8×8 test but at 16×16 — one step closer to the 64×64
    AWS-F1 target. Exercises K-chunking past N_ROWS=16."""
    rng = np.random.default_rng(7)
    x = rng.integers(-40, 40, size=(1, 8, 3, 3), dtype=np.int8)
    w = rng.integers(-10, 10, size=(32, 8, 3, 3), dtype=np.int8)
    res = compile_conv2d(w, x, n_rows=16, n_cols=16,
                          stride=(1, 1), pad=(1, 1, 1, 1))
    _, log = simulate_program(
        res.program, n_rows=16, n_cols=16, return_read_log=True)
    out = reassemble_conv_output(log, res)
    ref = reference_conv2d_int8(x, w, stride=(1, 1), pad=(1, 1, 1, 1))
    assert np.array_equal(out, ref)
    assert res.k_chunks == 5          # K_total=72, n_rows=16
    assert res.n_chunks == 2          # C_out=32, n_cols=16


def test_reference_matches_manual_1x1():
    """Hand-computed case to catch reference-side bugs."""
    x = np.array([[[[1, 2], [3, 4]]]], dtype=np.int8)           # (1,1,2,2)
    w = np.array([[[[2]]], [[[3]]]], dtype=np.int8)             # (2,1,1,1)
    ref = reference_conv2d_int8(x, w)
    # Output: channel 0 = 2*input, channel 1 = 3*input.
    expected = np.array([[[[2, 4], [6, 8]],
                          [[3, 6], [9, 12]]]], dtype=np.int32)
    assert np.array_equal(ref, expected)


def test_reference_padding_correct():
    """3×3 conv with pad=1 on a single pixel: output at center should
    include only the center weight × the center pixel."""
    x = np.array([[[[5]]]], dtype=np.int8)                       # (1,1,1,1)
    w = np.array([[[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]]], dtype=np.int8)                 # (1,1,3,3)
    ref = reference_conv2d_int8(x, w, pad=(1, 1, 1, 1))
    # H_in=1, H_out = 1+2-3 +1 = 1. Single output.
    # Center weight is 5; rest of the receptive field is out of bounds.
    # So out[0,0,0,0] = 5 * 5 = 25.
    assert ref.shape == (1, 1, 1, 1)
    assert ref[0, 0, 0, 0] == 25
