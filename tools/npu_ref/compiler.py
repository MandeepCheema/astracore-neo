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
from typing import Iterable, List, Sequence


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


@dataclass
class RunTile:
    k: int
    ai_base: int
    ao_base: int
    afu_mode: int = 0
    acc_init_mode: int = 0
    acc_init_data: int = 0


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
