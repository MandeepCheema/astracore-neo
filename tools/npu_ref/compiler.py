"""AstraCore NPU compiler skeleton (WP-3).

Minimal viable compiler that takes a simple model description and emits
a linear sequence of NPU instructions that a runtime can replay through
the npu_top external interface. The goal at this revision is to prove
the compiler-NPU contract end-to-end for single-tile matmul; multi-tile
tiling, conv2d, quantisation, and optimisation passes are future work.

Model representation:
    Model(layers=[Matmul(M, K, N, weights_MxN)])
    - M <= N_ROWS, N <= N_COLS (single-tile limit, V1)
    - K arbitrary (via tile_ctrl k-loop, already supported in hardware)

Instruction set (opaque to the hardware — just a way to sequence AXI /
external-port commands in Python):
    LoadWeight(addr, data)   : ext_w_we=1, ext_w_waddr=addr, ext_w_wdata=data
    LoadActivation(addr, packed) : ext_ai_we=1, ext_ai_waddr=addr, ext_ai_wdata=packed
    RunTile(k, ai_base, ao_base, afu_mode=0, acc_init_mode=0, acc_init_data=0)
                             : start=1, cfg_* fields set, wait for done
    ReadAO(addr)             : ext_ao_re=1, ext_ao_raddr=addr, return rdata

Hardware references: npu_top.v ports, tile_ctrl.v FSM, test_npu_top.py
end-to-end driver functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Instruction set
# ---------------------------------------------------------------------------
@dataclass
class LoadWeight:
    addr: int
    data: int   # 8-bit, interpreted as signed INT8 downstream


@dataclass
class LoadActivation:
    addr: int
    packed: int   # N_ROWS * DATA_W bits; caller packs


# Sentinel for `RunTile.acc_init_data` on a chained continuation tile:
# the runtime should substitute the most recent ReadAO result (packed
# via _pack_cvec_signed) before issuing the tile. Lets the Program stay
# self-contained — the compiler doesn't need to know partial-sum values
# at compile time, and the executor doesn't need side-channel state.
ACC_INIT_FROM_PREV_AO = "FROM_PREV_AO"


@dataclass
class RunTile:
    k: int
    ai_base: int
    ao_base: int
    afu_mode: int = 0
    acc_init_mode: int = 0
    # Either an integer literal (the seed packed as cfg_acc_init_data),
    # or the sentinel ACC_INIT_FROM_PREV_AO. The latter is meaningful
    # only when acc_init_mode=1 and signals a K-chained tile.
    acc_init_data: "int | str" = 0


@dataclass
class ReadAO:
    addr: int


Instruction = LoadWeight | LoadActivation | RunTile | ReadAO


# ---------------------------------------------------------------------------
# Model description
# ---------------------------------------------------------------------------
@dataclass
class Matmul:
    """Weight-stationary tile matmul matching the NPU hardware semantics:

        out[1, N_cols] = sum over k of (act[k] @ W)

    where W is an N_rows × N_cols matrix loaded once per tile, and
    `act[k]` are cfg_k activation vectors each of length N_rows.
    After cfg_k cycles the accumulator holds the element-wise sum of
    cfg_k partial dot products.

    V1: a single Matmul models one tile. Multi-tile tilings (for real
    K > N_rows, M > 1) are a future compiler pass.

    Fields:
        N_rows, N_cols: dimensions of the weight matrix
        cfg_k: number of activation vectors to stream per tile
        weights: [N_rows][N_cols] INT8 weight matrix
        afu_mode: activation mode applied at AO writeback
    """
    N_rows: int
    N_cols: int
    cfg_k: int
    weights: Sequence[Sequence[int]]
    afu_mode: int = 0


@dataclass
class Model:
    layers: List[Matmul] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Program (linear sequence of instructions)
# ---------------------------------------------------------------------------
@dataclass
class Program:
    instructions: List[Instruction] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.instructions)

    def __iter__(self):
        return iter(self.instructions)


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------
def _pack_bytes(vals: Sequence[int], elem_w: int = 8) -> int:
    out = 0
    for i, v in enumerate(vals):
        out |= (v & ((1 << elem_w) - 1)) << (i * elem_w)
    return out


def compile_matmul(layer: Matmul, *,
                   ai_base: int, ao_base: int,
                   activation_vectors: Sequence[Sequence[int]]) -> Program:
    """Compile a single Matmul tile into a weight-load + activation-load
    + RunTile + ReadAO sequence.

    The caller owns the ai_base / ao_base addressing — a real compiler
    would track SRAM allocations as a liveness pass.
    """
    assert len(layer.weights) == layer.N_rows, "weights shape mismatch (N_rows)"
    assert all(len(row) == layer.N_cols for row in layer.weights), \
        "weights shape mismatch (N_cols)"
    assert len(activation_vectors) == layer.cfg_k, \
        f"need cfg_k={layer.cfg_k} activation vectors, got {len(activation_vectors)}"
    for v in activation_vectors:
        assert len(v) == layer.N_rows, \
            f"each activation vector must be {layer.N_rows} elements"

    prog = Program()

    # Weight upload: linear INT8 addr = row*N_cols + col
    for row in range(layer.N_rows):
        for col in range(layer.N_cols):
            linear = row * layer.N_cols + col
            prog.instructions.append(LoadWeight(linear, layer.weights[row][col]))

    # Activation upload: one packed vector per K-step
    for k_idx, vec in enumerate(activation_vectors):
        prog.instructions.append(
            LoadActivation(ai_base + k_idx, _pack_bytes(vec))
        )

    # Run tile
    prog.instructions.append(
        RunTile(
            k=layer.cfg_k,
            ai_base=ai_base,
            ao_base=ao_base,
            afu_mode=layer.afu_mode,
        )
    )

    # Read output
    prog.instructions.append(ReadAO(ao_base))

    return prog


# ---------------------------------------------------------------------------
# Multi-tile K chaining (F1-C3).
#
# When a matmul's cfg_k exceeds the hardware AI SRAM depth (AI SRAM
# holds ACT_IN_DEPTH=16 activation vectors in current silicon) the
# compiler must split K across multiple tiles and chain them via
# cfg_acc_init_mode=1 + cfg_acc_init_data. The first tile runs with
# acc_init_mode=0 (clear accumulator). Every subsequent tile runs
# with acc_init_mode=1 and its seed is the previous tile's partial
# sum (read back from AO SRAM).
#
# Constraint (F1-C2 audit H1): AFU / bias must only be applied on the
# final tile. Intermediate tiles run afu_mode=PASS so the partial
# sum stays as a raw INT32 accumulator. Applying RELU on an
# intermediate partial sum would zero out negatives that the next
# tile's additions would have lifted positive — wrong math.
# ---------------------------------------------------------------------------
# ACT_IN_DEPTH — max activation vectors per tile on current RTL. If
# the RTL parameter changes, update this and any cocotb test that
# mirrors it.
AI_SRAM_DEPTH = 16


def compile_matmul_chained(layer: Matmul, *,
                            ai_base: int,
                            ao_base_per_tile: Sequence[int],
                            activation_vectors: Sequence[Sequence[int]],
                            k_per_tile: int = AI_SRAM_DEPTH) -> Program:
    """Compile a Matmul whose reduction `cfg_k` may exceed one tile's
    capacity, by chaining tiles via `cfg_acc_init_mode`.

    Args:
        layer: the logical Matmul (weights, full cfg_k, afu_mode).
        ai_base: starting AI SRAM address; each tile reuses it
            (activations are overwritten per tile).
        ao_base_per_tile: distinct AO address per tile so the
            executor can read back the partial sum of tile i and
            feed it as seed to tile i+1.
        activation_vectors: exactly `layer.cfg_k` vectors, each of
            length `layer.N_rows`.
        k_per_tile: cap on cfg_k per tile; defaults to AI_SRAM_DEPTH
            (hardware limit). Lower values are useful for testing.

    Returns:
        Program whose final ReadAO reads the completed result at
        `ao_base_per_tile[-1]`.

    Emission pattern:
        LoadWeight × N_rows*N_cols       (once, weights are stationary)
        For each tile i in [0 .. n_tiles):
            LoadActivation × cfg_k_i     (at ai_base, overwriting prev)
            RunTile(k=cfg_k_i,
                    afu_mode = user's if last tile else PASS,
                    acc_init_mode = 0 if i==0 else 1,
                    acc_init_data = 0 if i==0 else ACC_INIT_FROM_PREV_AO)
            ReadAO(ao_base_per_tile[i])  (final + intermediate)

    The intermediate ReadAOs are emitted so the executor can capture
    each partial for seeding the next tile. A runtime with a larger
    AO bank could skip them, but the Program contract stays consistent.
    """
    if layer.cfg_k <= k_per_tile:
        # Degenerate: one tile is enough; route to the non-chaining
        # helper so callers don't get a weirdly-structured Program
        # for the single-tile case.
        return compile_matmul(layer,
                              ai_base=ai_base,
                              ao_base=ao_base_per_tile[0],
                              activation_vectors=activation_vectors)

    assert len(activation_vectors) == layer.cfg_k, (
        f"chained matmul: need cfg_k={layer.cfg_k} activation vectors, "
        f"got {len(activation_vectors)}"
    )
    for v in activation_vectors:
        assert len(v) == layer.N_rows, \
            f"each activation vector must be {layer.N_rows} elements"
    if k_per_tile <= 0 or k_per_tile > AI_SRAM_DEPTH:
        raise ValueError(
            f"k_per_tile={k_per_tile} outside [1, {AI_SRAM_DEPTH}]"
        )

    # Split activation_vectors into chunks of ≤ k_per_tile.
    chunks: List[Sequence[Sequence[int]]] = []
    for start in range(0, layer.cfg_k, k_per_tile):
        chunks.append(activation_vectors[start:start + k_per_tile])
    n_tiles = len(chunks)
    if len(ao_base_per_tile) != n_tiles:
        raise ValueError(
            f"need {n_tiles} ao_base_per_tile entries (one per tile), "
            f"got {len(ao_base_per_tile)}"
        )

    prog = Program()

    # Weight upload — once; weights are stationary across the chain.
    for row in range(layer.N_rows):
        for col in range(layer.N_cols):
            linear = row * layer.N_cols + col
            prog.instructions.append(
                LoadWeight(linear, layer.weights[row][col])
            )

    for i, chunk in enumerate(chunks):
        # Activation upload for this tile's K-chunk. ai_base is reused
        # across tiles since the AI SRAM is rewritten each iteration.
        for k_idx, vec in enumerate(chunk):
            prog.instructions.append(
                LoadActivation(ai_base + k_idx, _pack_bytes(vec))
            )

        is_first = (i == 0)
        is_last = (i == n_tiles - 1)
        prog.instructions.append(
            RunTile(
                k=len(chunk),
                ai_base=ai_base,
                ao_base=ao_base_per_tile[i],
                # AFU applies only on the final tile (F1-C2 audit H1).
                afu_mode=layer.afu_mode if is_last else 0,
                acc_init_mode=0 if is_first else 1,
                acc_init_data=0 if is_first else ACC_INIT_FROM_PREV_AO,
            )
        )
        # ReadAO: intermediate reads feed the chain; the last read
        # is the layer's output.
        prog.instructions.append(ReadAO(ao_base_per_tile[i]))

    return prog


def compile_model(model: Model, *,
                  activations_per_layer: Sequence[Sequence[Sequence[int]]],
                  ai_base_per_layer: Sequence[int] | None = None,
                  ao_base_per_layer: Sequence[int] | None = None) -> Program:
    """Compile a multi-layer model to a linear program.

    V1: each layer gets a new AI / AO base. No inter-layer feed-through
    (caller must arrange activation propagation manually for now).

    A future compiler pass would fold outputs of layer N into AI of
    layer N+1 automatically, and hoist shared weights into a single
    upload where possible.
    """
    if ai_base_per_layer is None:
        ai_base_per_layer = [0] * len(model.layers)
    if ao_base_per_layer is None:
        ao_base_per_layer = [i for i in range(len(model.layers))]
    assert len(activations_per_layer) == len(model.layers)
    assert len(ai_base_per_layer) == len(model.layers)
    assert len(ao_base_per_layer) == len(model.layers)

    full = Program()
    for layer, acts, ai, ao in zip(model.layers, activations_per_layer,
                                    ai_base_per_layer, ao_base_per_layer):
        sub = compile_matmul(layer, ai_base=ai, ao_base=ao,
                             activation_vectors=acts)
        full.instructions.extend(sub.instructions)
    return full


# ---------------------------------------------------------------------------
# Software reference: apply the same math a hardware run would produce.
# Used for bit-exact comparison against RTL output.
# ---------------------------------------------------------------------------
def _as_int8(v: int) -> int:
    v &= 0xFF
    return v - 0x100 if v & 0x80 else v


def _apply_afu(v: int, afu_mode: int) -> int:
    """Mirror of npu_activation.v modes 0..4 (LUT modes 5..7 not handled here)."""
    if afu_mode == 0:
        return v
    if afu_mode == 1:   # RELU
        return 0 if v < 0 else v
    if afu_mode == 2:   # LEAKY_RELU
        return (v >> 3) if v < 0 else v
    if afu_mode == 3:   # CLIP_INT8
        return max(-128, min(127, v))
    if afu_mode == 4:   # RELU_CLIP_INT8
        return 0 if v < 0 else min(127, v)
    raise ValueError(f"afu_mode {afu_mode} has LUT backend; use activation_ref for ref")


def emulate_matmul(layer: Matmul,
                   activation_vectors: Sequence[Sequence[int]]) -> List[int]:
    """Python ref for one tile, mirroring NPU hardware semantics exactly.

    Same math as test_npu_top._expected_matmul_accumulated.
    """
    acc = [0] * layer.N_cols
    for vec in activation_vectors:
        assert len(vec) == layer.N_rows
        for n in range(layer.N_cols):
            for r in range(layer.N_rows):
                acc[n] += _as_int8(layer.weights[r][n]) * _as_int8(vec[r])
    return [_apply_afu(v, layer.afu_mode) for v in acc]


# ---------------------------------------------------------------------------
# Program interpreter (F1-C3). Walks a compiled Program and simulates
# what the RTL would do using the emulator's math. Used to validate
# Program correctness without needing Verilator in the loop.
#
# This is NOT cycle-accurate — it's value-accurate. The value-accuracy
# contract matches `emulate_matmul`, which is already proven bit-exact
# against the RTL in sim/npu_top/test_npu_compiled.py.
# ---------------------------------------------------------------------------
def _unpack_bytes(packed: int, n: int, elem_w: int = 8) -> List[int]:
    mask = (1 << elem_w) - 1
    return [(packed >> (i * elem_w)) & mask for i in range(n)]


def simulate_program(program: Program, *,
                      n_rows: int,
                      n_cols: int,
                      return_read_log: bool = False):
    """Execute a Program against a Python model of the NPU.

    Default return: final AO SRAM contents as `dict[addr -> list[int]]`
    (matches the F1-C3 callers).

    When `return_read_log=True`: returns `(ao_sram, read_log)` where
    `read_log` is a chronological list of `(addr, value)` tuples —
    one entry per `ReadAO` instruction executed. Required by F1-C4
    conv compilation because M-loops reuse a small pool of AO slots;
    the dict alone loses the per-position history.

    Models:
        weight_sram : dict addr -> int8 (signed).
        ai_sram     : dict addr -> packed int (N_ROWS bytes).
        acc         : list of length N_COLS (accumulator snapshot —
                      valid only during a tile's execution).
        ao_sram     : dict addr -> list[int32] of length N_COLS.
        last_ao_read: list[int32] | None — the most recent ReadAO
                      result, used to resolve ACC_INIT_FROM_PREV_AO.

    Behaviour mirrors the RTL:
        LoadWeight(addr, data)     : weight_sram[addr] = data
        LoadActivation(addr, p)    : ai_sram[addr] = p
        RunTile(...)               : acc = (acc_init) + sum over k of
                                      (W @ act[k]); AFU applied; result
                                      stored at ao_sram[ao_base].
        ReadAO(addr)               : last_ao_read = ao_sram[addr];
                                      appended to read_log.

    The ACC_INIT_FROM_PREV_AO sentinel on a chained RunTile resolves
    to `last_ao_read` — exactly what a hardware runtime would do by
    reading the previous AO over the external port.
    """
    weight_sram: Dict[int, int] = {}
    ai_sram: Dict[int, int] = {}
    ao_sram: Dict[int, List[int]] = {}
    last_ao_read: Optional[List[int]] = None
    read_log: List[tuple] = []

    for instr in program:
        if isinstance(instr, LoadWeight):
            weight_sram[instr.addr] = instr.data & 0xFF
        elif isinstance(instr, LoadActivation):
            ai_sram[instr.addr] = instr.packed
        elif isinstance(instr, RunTile):
            # Reconstruct the weight matrix from the weight SRAM.
            W = [[_as_int8(weight_sram.get(r * n_cols + c, 0))
                  for c in range(n_cols)] for r in range(n_rows)]
            # Gather the cfg_k activation vectors for this tile.
            acts: List[List[int]] = []
            for k_idx in range(instr.k):
                packed = ai_sram.get(instr.ai_base + k_idx, 0)
                vec_unsigned = _unpack_bytes(packed, n_rows)
                acts.append([_as_int8(v) for v in vec_unsigned])
            # Compute acc[n] = sum_k sum_r W[r,n] * act[k,r]
            acc = [0] * n_cols
            for vec in acts:
                for n in range(n_cols):
                    for r in range(n_rows):
                        acc[n] += W[r][n] * vec[r]
            # Apply acc_init seed.
            if instr.acc_init_mode == 1:
                seed = instr.acc_init_data
                if seed == ACC_INIT_FROM_PREV_AO:
                    if last_ao_read is None:
                        raise RuntimeError(
                            "RunTile with ACC_INIT_FROM_PREV_AO but no "
                            "prior ReadAO to source the seed from"
                        )
                    acc = [a + s for a, s in zip(acc, last_ao_read)]
                else:
                    # Integer literal: unpack it as N_COLS × 32-bit
                    # signed values (same packing the RTL uses).
                    mask = (1 << 32) - 1
                    half = 1 << 31
                    for n in range(n_cols):
                        chunk = (seed >> (n * 32)) & mask
                        if chunk & half:
                            chunk -= (1 << 32)
                        acc[n] += chunk
            # Apply AFU last (PASS on non-final chained tiles).
            acc = [_apply_afu(v, instr.afu_mode) for v in acc]
            ao_sram[instr.ao_base] = acc
        elif isinstance(instr, ReadAO):
            last_ao_read = ao_sram.get(instr.addr, [0] * n_cols)
            read_log.append((instr.addr, list(last_ao_read)))
        else:
            raise TypeError(f"unknown instruction {type(instr).__name__}")

    if return_read_log:
        return ao_sram, read_log
    return ao_sram


# ---------------------------------------------------------------------------
# Self-check: compile + emulate a small matmul, verify the program
# structure and emulation math are self-consistent.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 4×4 identity, cfg_k=1 tile
    W = [[1 if i == j else 0 for j in range(4)] for i in range(4)]
    layer = Matmul(N_rows=4, N_cols=4, cfg_k=1, weights=W, afu_mode=0)
    prog = compile_matmul(layer, ai_base=0, ao_base=0,
                          activation_vectors=[[7, -3, 11, -5]])
    instr_types = [type(i).__name__ for i in prog]
    # 16 LoadWeights (4x4) + 1 LoadActivation + 1 RunTile + 1 ReadAO = 19
    assert len(prog) == 19, f"expected 19, got {len(prog)}: {instr_types}"
    assert instr_types[:16] == ["LoadWeight"] * 16
    assert instr_types[16] == "LoadActivation"
    assert instr_types[17] == "RunTile"
    assert instr_types[18] == "ReadAO"
    # Emulation: identity W * [7,-3,11,-5] = [7,-3,11,-5]
    emu = emulate_matmul(layer, [[7, -3, 11, -5]])
    assert emu == [7, -3, 11, -5], emu

    # Multi-step cfg_k=3 test
    W2 = [[1, 0, 2, 0],
          [0, 1, 0, 2],
          [1, 1, 1, 1],
          [-1, -1, -1, -1]]
    layer2 = Matmul(N_rows=4, N_cols=4, cfg_k=3, weights=W2, afu_mode=0)
    acts2 = [[1, 2, 3, 4], [-1, 0, 1, 2], [2, -1, 3, 1]]
    emu2 = emulate_matmul(layer2, acts2)
    # Match test_k_equals_3_accumulation expected value [2, 1, 4, 2]
    assert emu2 == [2, 1, 4, 2], f"cfg_k=3 emulation got {emu2}"
    prog2 = compile_matmul(layer2, ai_base=0, ao_base=0,
                           activation_vectors=acts2)
    # 16 weights + 3 activations + 1 runtile + 1 read = 21
    assert len(prog2) == 21, len(prog2)

    # RELU mode
    layer3 = Matmul(N_rows=4, N_cols=4, cfg_k=1,
                    weights=[[-1, 0, 1, 2]] * 4,
                    afu_mode=1)  # RELU
    emu3 = emulate_matmul(layer3, [[1, 1, 1, 1]])
    assert emu3 == [0, 0, 4, 8], f"RELU: got {emu3}"

    print(f"compiler self-check PASS (19-instr single-tile, "
          f"21-instr cfg_k=3 tile, RELU applied at writeback)")
