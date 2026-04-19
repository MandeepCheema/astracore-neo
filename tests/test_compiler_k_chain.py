"""Unit tests for compile_matmul_chained + simulate_program (F1-C3).

Verifies:
  - Single-tile K ≤ k_per_tile routes to the non-chained compile_matmul
    helper (no awkward chain-of-one).
  - 2-tile chain (even K split) is bit-exact vs single-tile emulation.
  - 3-tile chain (uneven K split with a short final chunk) is bit-exact.
  - AFU is applied ONLY on the final tile of a chain (intermediate
    tiles must run with afu_mode=0 so partial sums stay as raw INT32).
  - Weights are loaded once, not re-loaded per tile (weight-stationary).
  - Activation re-use across tiles (ai_base reused, not appended).
  - acc_init_mode pattern: first=0, rest=1 with ACC_INIT_FROM_PREV_AO.
  - ReadAO emitted after each tile so the executor can seed the chain.

The bit-exact gate is expressed against `emulate_matmul`, which is
already proven bit-exact vs the RTL by the existing
sim/npu_top/test_npu_compiled.py tests. A cocotb test against real
RTL is a separate F1-C3 acceptance (needs Verilator) — see
sim/npu_top/test_compiled_k_chain.py.
"""

from __future__ import annotations

from typing import List

import pytest

from tools.npu_ref.compiler import (
    ACC_INIT_FROM_PREV_AO,
    LoadActivation,
    LoadWeight,
    Matmul,
    ReadAO,
    RunTile,
    compile_matmul,
    compile_matmul_chained,
    emulate_matmul,
    simulate_program,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
def _identity_4x4() -> Matmul:
    W = [[1 if i == j else 0 for j in range(4)] for i in range(4)]
    return Matmul(N_rows=4, N_cols=4, cfg_k=1, weights=W)


def _random_4x4(cfg_k: int, afu_mode: int = 0, seed: int = 0) -> Matmul:
    import random
    r = random.Random(seed)
    W = [[r.randint(-10, 10) for _ in range(4)] for _ in range(4)]
    return Matmul(N_rows=4, N_cols=4, cfg_k=cfg_k, weights=W, afu_mode=afu_mode)


def _random_activations(cfg_k: int, n_rows: int, seed: int = 1) -> List[List[int]]:
    import random
    r = random.Random(seed)
    return [[r.randint(-10, 10) for _ in range(n_rows)] for _ in range(cfg_k)]


# ---------------------------------------------------------------------------
# Routing: short K falls back to compile_matmul (no degenerate chain)
# ---------------------------------------------------------------------------
def test_short_k_routes_to_single_tile():
    layer = _identity_4x4()
    acts = [[7, -3, 11, -5]]
    prog = compile_matmul_chained(
        layer, ai_base=0, ao_base_per_tile=[0], activation_vectors=acts,
        k_per_tile=8,
    )
    # compile_matmul emits exactly 1 RunTile and 1 ReadAO.
    assert sum(isinstance(i, RunTile) for i in prog) == 1
    assert sum(isinstance(i, ReadAO) for i in prog) == 1
    rt = next(i for i in prog if isinstance(i, RunTile))
    assert rt.acc_init_mode == 0  # no chaining
    assert rt.acc_init_data == 0


# ---------------------------------------------------------------------------
# Bit-exact numerical gate — the core correctness check
# ---------------------------------------------------------------------------
def test_two_tile_chain_matches_emulator():
    """K=32 matmul split into 2 tiles of K=16 each."""
    layer = _random_4x4(cfg_k=32, seed=3)
    acts = _random_activations(32, 4, seed=4)
    expected = emulate_matmul(layer, acts)
    prog = compile_matmul_chained(
        layer, ai_base=0, ao_base_per_tile=[0, 1],
        activation_vectors=acts, k_per_tile=16,
    )
    sim = simulate_program(prog, n_rows=4, n_cols=4)
    # Final tile writes to ao_base_per_tile[-1].
    assert sim[1] == expected


def test_three_tile_uneven_chain_matches_emulator():
    """K=40 split 16 + 16 + 8 — final chunk is short to exercise the
    fact that k_per_tile is a cap, not a fixed step."""
    layer = _random_4x4(cfg_k=40, seed=5)
    acts = _random_activations(40, 4, seed=6)
    expected = emulate_matmul(layer, acts)
    prog = compile_matmul_chained(
        layer, ai_base=0, ao_base_per_tile=[0, 1, 2],
        activation_vectors=acts, k_per_tile=16,
    )
    sim = simulate_program(prog, n_rows=4, n_cols=4)
    assert sim[2] == expected


def test_k16_on_4x4_array_bit_exact():
    """Plan-specified acceptance: 4×4 array runs K=16 matmul bit-
    exact. We exercise it both as a single-tile K=16 run (fits AI
    SRAM depth) and split across 2 tiles of K=8 via a forced
    k_per_tile=8. Both must match the emulator."""
    layer = _random_4x4(cfg_k=16, seed=7)
    acts = _random_activations(16, 4, seed=8)
    expected = emulate_matmul(layer, acts)

    # Single-tile path (K_per_tile default = 16 = cfg_k → falls back
    # to compile_matmul).
    prog_single = compile_matmul_chained(
        layer, ai_base=0, ao_base_per_tile=[0],
        activation_vectors=acts,
    )
    sim_single = simulate_program(prog_single, n_rows=4, n_cols=4)
    assert sim_single[0] == expected

    # Forced 2-tile chain (K=8+8).
    prog_chain = compile_matmul_chained(
        layer, ai_base=0, ao_base_per_tile=[0, 1],
        activation_vectors=acts, k_per_tile=8,
    )
    sim_chain = simulate_program(prog_chain, n_rows=4, n_cols=4)
    assert sim_chain[1] == expected


# ---------------------------------------------------------------------------
# Structural invariants of the emitted Program
# ---------------------------------------------------------------------------
def test_weights_loaded_once_even_across_chain():
    layer = _random_4x4(cfg_k=32, seed=9)
    acts = _random_activations(32, 4, seed=10)
    prog = compile_matmul_chained(
        layer, ai_base=0, ao_base_per_tile=[0, 1],
        activation_vectors=acts, k_per_tile=16,
    )
    n_weight_loads = sum(isinstance(i, LoadWeight) for i in prog)
    # Exactly N_rows × N_cols = 16.
    assert n_weight_loads == 16


def test_activation_loads_reuse_ai_base():
    """Each tile's activations are loaded starting at ai_base (the
    previous tile's activations are overwritten). Two tiles of K=16
    each should issue 32 LoadActivation, all in the range
    [ai_base, ai_base+15]."""
    layer = _random_4x4(cfg_k=32, seed=11)
    acts = _random_activations(32, 4, seed=12)
    prog = compile_matmul_chained(
        layer, ai_base=5, ao_base_per_tile=[0, 1],
        activation_vectors=acts, k_per_tile=16,
    )
    la_addrs = [i.addr for i in prog if isinstance(i, LoadActivation)]
    assert len(la_addrs) == 32
    assert all(5 <= a <= 20 for a in la_addrs)


def test_afu_only_on_final_tile():
    """AFU mode must be 0 on every non-final RunTile even if the user
    asked for RELU. Otherwise intermediate partial sums get their
    negatives zeroed before the chain finishes summing."""
    layer = _random_4x4(cfg_k=32, afu_mode=1, seed=13)  # RELU
    acts = _random_activations(32, 4, seed=14)
    prog = compile_matmul_chained(
        layer, ai_base=0, ao_base_per_tile=[0, 1],
        activation_vectors=acts, k_per_tile=16,
    )
    run_tiles = [i for i in prog if isinstance(i, RunTile)]
    assert len(run_tiles) == 2
    assert run_tiles[0].afu_mode == 0        # non-final: PASS
    assert run_tiles[1].afu_mode == 1        # final: user's RELU


def test_acc_init_mode_pattern():
    layer = _random_4x4(cfg_k=48, seed=15)
    acts = _random_activations(48, 4, seed=16)
    prog = compile_matmul_chained(
        layer, ai_base=0, ao_base_per_tile=[0, 1, 2],
        activation_vectors=acts, k_per_tile=16,
    )
    run_tiles = [i for i in prog if isinstance(i, RunTile)]
    assert len(run_tiles) == 3
    assert run_tiles[0].acc_init_mode == 0
    assert run_tiles[0].acc_init_data == 0
    for rt in run_tiles[1:]:
        assert rt.acc_init_mode == 1
        assert rt.acc_init_data == ACC_INIT_FROM_PREV_AO


def test_readao_between_every_tile():
    layer = _random_4x4(cfg_k=48, seed=17)
    acts = _random_activations(48, 4, seed=18)
    prog = compile_matmul_chained(
        layer, ai_base=0, ao_base_per_tile=[0, 1, 2],
        activation_vectors=acts, k_per_tile=16,
    )
    # Count RunTile → ReadAO interleave.
    sequence = [type(i).__name__ for i in prog
                if isinstance(i, (RunTile, ReadAO))]
    assert sequence == ["RunTile", "ReadAO"] * 3


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------
def test_wrong_ao_base_count_raises():
    layer = _random_4x4(cfg_k=32, seed=19)
    acts = _random_activations(32, 4, seed=20)
    # Needs 2 ao bases (K=32, k_per_tile=16) but only 1 given.
    with pytest.raises(ValueError, match="ao_base_per_tile"):
        compile_matmul_chained(
            layer, ai_base=0, ao_base_per_tile=[0],
            activation_vectors=acts, k_per_tile=16,
        )


def test_k_per_tile_out_of_range_raises():
    layer = _random_4x4(cfg_k=32, seed=21)
    acts = _random_activations(32, 4, seed=22)
    with pytest.raises(ValueError, match="k_per_tile"):
        compile_matmul_chained(
            layer, ai_base=0, ao_base_per_tile=[0],
            activation_vectors=acts, k_per_tile=0,
        )
    with pytest.raises(ValueError, match="k_per_tile"):
        compile_matmul_chained(
            layer, ai_base=0, ao_base_per_tile=[0],
            activation_vectors=acts, k_per_tile=17,  # > AI_SRAM_DEPTH
        )


# ---------------------------------------------------------------------------
# Interpreter — reused in F1-C3 as the value-accuracy gate for chained
# programs. Lightweight coverage here to prove the interpreter itself
# is self-consistent with emulate_matmul.
# ---------------------------------------------------------------------------
def test_interpreter_matches_emulator_on_single_tile():
    layer = _random_4x4(cfg_k=3, afu_mode=1, seed=23)  # RELU
    acts = _random_activations(3, 4, seed=24)
    prog = compile_matmul(layer, ai_base=0, ao_base=0, activation_vectors=acts)
    sim = simulate_program(prog, n_rows=4, n_cols=4)
    expected = emulate_matmul(layer, acts)
    assert sim[0] == expected


def test_interpreter_raises_on_unseeded_chain():
    """A ChainedRunTile (acc_init_mode=1, data=FROM_PREV_AO) with no
    prior ReadAO is a programmer error — we want a loud failure, not
    a silent zero seed."""
    layer = _identity_4x4()
    acts = [[1, 2, 3, 4]]
    prog = compile_matmul(layer, ai_base=0, ao_base=0,
                           activation_vectors=acts)
    # Mutate the tile to claim it's chained without a prior ReadAO.
    run_tile = next(i for i in prog if isinstance(i, RunTile))
    run_tile.acc_init_mode = 1
    run_tile.acc_init_data = ACC_INIT_FROM_PREV_AO
    # Wipe the ReadAO that compile_matmul emits so there's no prior
    # read to source from.
    prog.instructions = [i for i in prog
                          if not isinstance(i, ReadAO)] + [
        i for i in prog if isinstance(i, ReadAO)]
    # Actually the ReadAO sits after the RunTile; to force the error
    # we need the RunTile first with no prior ReadAO.
    prog2 = type(prog)([i for i in prog if not isinstance(i, ReadAO)])
    with pytest.raises(RuntimeError, match="no prior ReadAO"):
        simulate_program(prog2, n_rows=4, n_cols=4)
