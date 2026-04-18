"""cocotb bit-exact regression: rtl/npu_systolic_array vs Python golden ref.

Every test drives the RTL and the Python reference with identical per-cycle
stimulus, then compares c_valid and every element of c_vec.  Fixtures cover:

  - Directed: identity weights, per-column basis, max-positive extremes,
    clear-wins-over-accumulate, long accumulation to exercise ACC_W.
  - Random: 4x4 and 16x8 bit-exact stress over 200+ cycles.

The harness parametrizes N_ROWS and N_COLS on the command line so a single
test file exercises multiple array shapes without duplication.
"""

import os
import random
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
from tools.npu_ref.systolic_ref import SystolicArray  # noqa: E402


DATA_W = 8
ACC_W = 32
N_ROWS = int(os.environ.get("N_ROWS", "4"))
N_COLS = int(os.environ.get("N_COLS", "4"))

DATA_MIN = -(1 << (DATA_W - 1))
DATA_MAX = (1 << (DATA_W - 1)) - 1


def _to_signed(val: int, width: int) -> int:
    mask = (1 << width) - 1
    val &= mask
    if val & (1 << (width - 1)):
        return val - (1 << width)
    return val


def _pack_vec(vec):
    """Pack a list of signed DATA_W ints into a single wide int (LSB=index 0)."""
    out = 0
    for i, v in enumerate(vec):
        out |= (v & ((1 << DATA_W) - 1)) << (i * DATA_W)
    return out


def _unpack_cvec(val, n_cols):
    return [_to_signed((val >> (i * ACC_W)) & ((1 << ACC_W) - 1), ACC_W)
            for i in range(n_cols)]


async def _reset(dut):
    dut.rst_n.value          = 0
    dut.w_load.value         = 0
    dut.w_addr.value         = 0
    dut.w_data.value         = 0
    dut.clear_acc.value      = 0
    dut.acc_load_valid.value = 0
    dut.acc_load_data.value  = 0
    dut.a_valid.value        = 0
    dut.a_vec.value          = 0
    dut.precision_mode.value = 0
    dut.sparse_skip_vec.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def _step(dut, ref: SystolicArray, **stim):
    defaults = dict(w_load=0, w_addr=0, w_data=0, clear_acc=0,
                    acc_load_valid=0, acc_load_data=0,
                    a_valid=0, a_vec=None, precision_mode=0,
                    sparse_skip_vec=0)
    defaults.update(stim)
    a_vec = defaults["a_vec"] if defaults["a_vec"] is not None else [0] * N_ROWS

    dut.w_load.value         = defaults["w_load"]
    dut.w_addr.value         = defaults["w_addr"]
    dut.w_data.value         = defaults["w_data"] & ((1 << (N_COLS * DATA_W)) - 1)
    dut.clear_acc.value      = defaults["clear_acc"]
    dut.acc_load_valid.value = defaults["acc_load_valid"]
    dut.acc_load_data.value  = defaults["acc_load_data"] & ((1 << (N_COLS * ACC_W)) - 1)
    dut.a_valid.value        = defaults["a_valid"]
    dut.a_vec.value          = _pack_vec(a_vec)
    dut.precision_mode.value = defaults["precision_mode"]
    dut.sparse_skip_vec.value = defaults["sparse_skip_vec"]

    ref.tick(w_load=defaults["w_load"], w_addr=defaults["w_addr"],
             w_data=defaults["w_data"], clear_acc=defaults["clear_acc"],
             acc_load_valid=defaults["acc_load_valid"],
             acc_load_data=defaults["acc_load_data"],
             precision_mode=defaults["precision_mode"],
             sparse_skip_vec=defaults["sparse_skip_vec"],
             a_valid=defaults["a_valid"], a_vec=a_vec)

    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


def _check(dut, ref: SystolicArray):
    rtl_valid = int(dut.c_valid.value)
    rtl_vec = _unpack_cvec(int(dut.c_vec.value), N_COLS)
    ref_vec = ref.c_vec_signed
    assert rtl_valid == ref.c_valid, (
        f"c_valid mismatch: rtl={rtl_valid}, ref={ref.c_valid}")
    assert rtl_vec == ref_vec, (
        f"c_vec mismatch:\n  rtl = {rtl_vec}\n  ref = {ref_vec}")


def _pack_row(row):
    """Pack N_COLS weights (LSB-first) into a wide w_data value."""
    out = 0
    for i, v in enumerate(row):
        out |= (v & ((1 << DATA_W) - 1)) << (i * DATA_W)
    return out


async def _load_weights(dut, ref, W):
    """Load a full weight tile one ROW per cycle.  W is [N_ROWS][N_COLS]."""
    for k in range(N_ROWS):
        await _step(dut, ref,
                    w_load=1,
                    w_addr=k,
                    w_data=_pack_row(W[k]))


# ---------------------------------------------------------------------------
# Directed tests
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_reset_state(dut):
    """After reset, c_valid=0, c_vec all zeros, accumulator cleared."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = SystolicArray(N_ROWS, N_COLS, DATA_W, ACC_W)
    ref.reset()
    await _reset(dut)
    _check(dut, ref)


@cocotb.test()
async def test_identity_weights(dut):
    """Identity weights (W[k][k]=1, else 0): output = activation vector as int32."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = SystolicArray(N_ROWS, N_COLS, DATA_W, ACC_W)
    ref.reset()
    await _reset(dut)
    n = min(N_ROWS, N_COLS)
    W = [[1 if (k < n and k == kn) else 0
          for kn in range(N_COLS)] for k in range(N_ROWS)]
    await _load_weights(dut, ref, W)
    # Build an N_ROWS-long activation vector with mixed signs so the test
    # scales to any array size without truncating the stimulus.
    a_vec = [((k + 1) * (1 if k % 2 == 0 else -1)) & 0xFF for k in range(N_ROWS)]
    await _step(dut, ref, a_valid=1, a_vec=a_vec)
    _check(dut, ref)


@cocotb.test()
async def test_clear_wins_over_accumulate(dut):
    """clear_acc + a_valid same cycle → accumulator reset."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = SystolicArray(N_ROWS, N_COLS, DATA_W, ACC_W)
    ref.reset()
    await _reset(dut)
    # Build a trivial weight set: W = all 1s
    W = [[1] * N_COLS for _ in range(N_ROWS)]
    await _load_weights(dut, ref, W)
    # Accumulate a few cycles
    for _ in range(3):
        await _step(dut, ref, a_valid=1, a_vec=[2] * N_ROWS)
    _check(dut, ref)
    # Clear in same cycle as a_valid → clear wins
    await _step(dut, ref, clear_acc=1, a_valid=1, a_vec=[99] * N_ROWS)
    _check(dut, ref)


@cocotb.test()
async def test_long_accumulation_within_acc_width(dut):
    """Accumulate 200 cycles at max INT8 values — must not overflow ACC_W."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = SystolicArray(N_ROWS, N_COLS, DATA_W, ACC_W)
    ref.reset()
    await _reset(dut)
    # Worst case per column per cycle: N_ROWS * (-128)*(-128) = N_ROWS * 16384
    # Over 200 cycles: N_ROWS * 16384 * 200 ≤ 2^31 when N_ROWS ≤ 655 → fine.
    W = [[127] * N_COLS for _ in range(N_ROWS)]
    await _load_weights(dut, ref, W)
    for _ in range(200):
        await _step(dut, ref, a_valid=1, a_vec=[127] * N_ROWS)
    _check(dut, ref)
    expected = 200 * N_ROWS * 127 * 127
    assert ref.c_vec_signed[0] == expected, (ref.c_vec_signed[0], expected)


@cocotb.test()
async def test_negative_operands(dut):
    """Mix of signed weights and activations — sign handling must be correct."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = SystolicArray(N_ROWS, N_COLS, DATA_W, ACC_W)
    ref.reset()
    await _reset(dut)
    W = [[-3 if (k + n) % 2 == 0 else 5 for n in range(N_COLS)]
         for k in range(N_ROWS)]
    await _load_weights(dut, ref, W)
    a_vec = [(((k + 1) * (-1 if k % 2 == 0 else 1))) & 0xFF for k in range(N_ROWS)]
    await _step(dut, ref, a_valid=1, a_vec=a_vec)
    _check(dut, ref)


# ---------------------------------------------------------------------------
# Random stress
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_random_bit_exact_500_cycles(dut):
    """Random weights + random activations, 500 cycles, bit-exact every cycle."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = SystolicArray(N_ROWS, N_COLS, DATA_W, ACC_W)
    ref.reset()
    await _reset(dut)
    rng = random.Random(0xACD5A57A)
    W = [[rng.randint(DATA_MIN, DATA_MAX) for _ in range(N_COLS)]
         for _ in range(N_ROWS)]
    await _load_weights(dut, ref, W)

    mismatches = 0
    for cycle in range(500):
        stim = dict(
            clear_acc=1 if rng.random() < 0.03 else 0,
            a_valid=1 if rng.random() < 0.85 else 0,
            a_vec=[rng.randint(DATA_MIN, DATA_MAX) for _ in range(N_ROWS)],
        )
        await _step(dut, ref, **stim)
        try:
            _check(dut, ref)
        except AssertionError as exc:
            mismatches += 1
            if mismatches <= 3:
                dut._log.error(f"cycle {cycle}: {exc}")
    assert mismatches == 0, f"{mismatches} cycles failed"
    dut._log.info(f"500 random cycles PASS (N_ROWS={N_ROWS} N_COLS={N_COLS})")


@cocotb.test()
async def test_weight_reload_mid_stream(dut):
    """Reload weights between activation cycles; subsequent accumulations
    use the new weights (RTL latches W on the same edge as load)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = SystolicArray(N_ROWS, N_COLS, DATA_W, ACC_W)
    ref.reset()
    await _reset(dut)
    W1 = [[1] * N_COLS for _ in range(N_ROWS)]
    await _load_weights(dut, ref, W1)
    await _step(dut, ref, a_valid=1, a_vec=[2] * N_ROWS)
    _check(dut, ref)
    W2 = [[2] * N_COLS for _ in range(N_ROWS)]
    await _load_weights(dut, ref, W2)
    await _step(dut, ref, a_valid=1, a_vec=[3] * N_ROWS)
    _check(dut, ref)


@cocotb.test()
async def test_acc_load_primes_accumulator(dut):
    """Gap #2 Phase 1: acc_load_valid primes the accumulator from external
    data; next EXECUTE cycle accumulates on top.  Enables scratch-based
    k-tile accumulation across weight tile boundaries."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = SystolicArray(N_ROWS, N_COLS, DATA_W, ACC_W)
    ref.reset()
    await _reset(dut)
    # Load identity-ish weights: W[k][k]=1 for k<min(N_ROWS,N_COLS), else 0
    W = [[1 if (k < min(N_ROWS, N_COLS) and k == n) else 0
          for n in range(N_COLS)] for k in range(N_ROWS)]
    await _load_weights(dut, ref, W)
    # Prime accumulator with distinct pattern per column
    primed = [100 + 50 * n for n in range(N_COLS)]
    packed = 0
    for n, v in enumerate(primed):
        packed |= (v & ((1 << ACC_W) - 1)) << (n * ACC_W)
    await _step(dut, ref, acc_load_valid=1, acc_load_data=packed)
    _check(dut, ref)
    # Execute with identity-input → first N activations pass through
    a = [k + 1 for k in range(N_ROWS)]
    await _step(dut, ref, a_valid=1, a_vec=a)
    _check(dut, ref)
    # Expected per col: primed[n] + (a[n] if n<min else 0)
    for n in range(N_COLS):
        want = primed[n] + (a[n] if n < min(N_ROWS, N_COLS) else 0)
        rtl_col = _to_signed((int(dut.c_vec.value) >> (n * ACC_W))
                              & ((1 << ACC_W) - 1), ACC_W)
        assert rtl_col == want, f"col {n}: rtl={rtl_col} want={want}"


@cocotb.test()
async def test_clear_wins_over_acc_load(dut):
    """When clear_acc and acc_load_valid fire the same cycle, clear wins."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = SystolicArray(N_ROWS, N_COLS, DATA_W, ACC_W)
    ref.reset()
    await _reset(dut)
    primed = [99] * N_COLS
    packed = 0
    for n, v in enumerate(primed):
        packed |= (v & ((1 << ACC_W) - 1)) << (n * ACC_W)
    # clear_acc + acc_load_valid same cycle
    await _step(dut, ref, clear_acc=1, acc_load_valid=1, acc_load_data=packed)
    _check(dut, ref)
    # All zero after a clear
    for n in range(N_COLS):
        rtl_col = _to_signed((int(dut.c_vec.value) >> (n * ACC_W))
                              & ((1 << ACC_W) - 1), ACC_W)
        assert rtl_col == 0, f"col {n}: expected 0, got {rtl_col}"
