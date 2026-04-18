"""cocotb bit-exact regression: rtl/npu_pe vs tools/npu_ref/pe_ref.PE.

Every test drives the RTL and the Python golden reference with the same
cycle-by-cycle stimulus, then asserts the visible outputs match exactly.
The Python model is the oracle; any mismatch is an RTL bug.

The RTL uses NBAs for every output; we read the values AFTER RisingEdge
followed by a 1-ns settle Timer (same pattern that fixed the object_tracker
iverilog sampling quirk).
"""

import os
import random
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# Make tools/npu_ref importable without installing the package.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
from tools.npu_ref.pe_ref import PE  # noqa: E402


DATA_W = 8
ACC_W = 32
DATA_MIN = -(1 << (DATA_W - 1))
DATA_MAX = (1 << (DATA_W - 1)) - 1


def _to_signed(val: int, width: int) -> int:
    mask = (1 << width) - 1
    val &= mask
    if val & (1 << (width - 1)):
        return val - (1 << width)
    return val


async def _reset(dut):
    dut.rst_n.value          = 0
    dut.precision_mode.value = 0
    dut.sparse_en.value      = 0
    dut.load_w.value         = 0
    dut.clear_acc.value      = 0
    dut.weight_in.value      = 0
    dut.a_valid.value        = 0
    dut.a_in.value           = 0
    dut.sparse_skip.value    = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


def _apply(dut, **kwargs) -> None:
    """Drive one cycle's stimulus onto the RTL inputs."""
    defaults = dict(load_w=0, clear_acc=0, weight_in=0, a_valid=0, a_in=0,
                    sparse_skip=0, precision_mode=0, sparse_en=0)
    defaults.update(kwargs)
    dut.load_w.value         = defaults["load_w"]
    dut.clear_acc.value      = defaults["clear_acc"]
    dut.weight_in.value      = defaults["weight_in"] & ((1 << DATA_W) - 1)
    dut.a_valid.value        = defaults["a_valid"]
    dut.a_in.value           = defaults["a_in"] & ((1 << DATA_W) - 1)
    dut.sparse_skip.value    = defaults["sparse_skip"]
    dut.precision_mode.value = defaults["precision_mode"]
    dut.sparse_en.value      = defaults["sparse_en"]


async def _step(dut, pe: PE, **stim) -> None:
    """Advance RTL and reference by one cycle under identical stimulus."""
    _apply(dut, **stim)
    pe.tick(**stim)
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


def _check(dut, pe: PE) -> None:
    """Compare RTL outputs to the reference after a cycle."""
    rtl_psum  = _to_signed(int(dut.psum_out.value), ACC_W)
    rtl_a_out = _to_signed(int(dut.a_out.value), DATA_W)
    rtl_avld  = int(dut.a_valid_out.value)
    rtl_skip  = int(dut.sparse_skip_out.value)
    assert rtl_psum  == pe.psum_out,       f"psum mismatch: rtl={rtl_psum}, ref={pe.psum_out}"
    assert rtl_a_out == pe.a_out_signed,   f"a_out mismatch: rtl={rtl_a_out}, ref={pe.a_out_signed}"
    assert rtl_avld  == pe.a_valid_out,    f"a_valid_out mismatch: rtl={rtl_avld}, ref={pe.a_valid_out}"
    assert rtl_skip  == pe.sparse_skip_out, f"sparse_skip_out mismatch: rtl={rtl_skip}, ref={pe.sparse_skip_out}"


# ---------------------------------------------------------------------------
# Directed tests: exercise the known-hard cases before the randoms.
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_reset_state(dut):
    """After reset: accumulator=0, pass-through flags=0, weight=0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    pe = PE(DATA_W, ACC_W)
    pe.reset()
    await _reset(dut)
    _check(dut, pe)
    dut._log.info("reset_state PASS")


@cocotb.test()
async def test_load_weight_and_accumulate(dut):
    """Load weight=3, stream activations 5 and 2 → psum=21."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    pe = PE(DATA_W, ACC_W)
    pe.reset()
    await _reset(dut)
    await _step(dut, pe, load_w=1, weight_in=3)
    _check(dut, pe)
    await _step(dut, pe, a_valid=1, a_in=5)
    _check(dut, pe)
    await _step(dut, pe, a_valid=1, a_in=2)
    _check(dut, pe)
    assert pe.psum_out == 21
    dut._log.info(f"psum = {pe.psum_out}")


@cocotb.test()
async def test_negative_operands(dut):
    """Signed arithmetic: (-3)*(-7) + (-3)*10 = 21 - 30 = -9."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    pe = PE(DATA_W, ACC_W)
    pe.reset()
    await _reset(dut)
    await _step(dut, pe, load_w=1, weight_in=-3 & 0xFF)
    await _step(dut, pe, a_valid=1, a_in=-7 & 0xFF)
    _check(dut, pe)
    await _step(dut, pe, a_valid=1, a_in=10)
    _check(dut, pe)
    assert pe.psum_out == -9
    dut._log.info(f"psum = {pe.psum_out}")


@cocotb.test()
async def test_sparse_skip_does_not_accumulate(dut):
    """sparse_skip=1 must NOT change the accumulator but MUST pass activation."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    pe = PE(DATA_W, ACC_W)
    pe.reset()
    await _reset(dut)
    await _step(dut, pe, load_w=1, weight_in=4)
    await _step(dut, pe, a_valid=1, a_in=5)          # acc += 20
    _check(dut, pe)
    await _step(dut, pe, a_valid=1, a_in=3, sparse_skip=1)  # skipped
    _check(dut, pe)
    await _step(dut, pe, a_valid=1, a_in=2)          # acc += 8 → 28
    _check(dut, pe)
    assert pe.psum_out == 28


@cocotb.test()
async def test_clear_wins_over_accumulate(dut):
    """clear_acc and a_valid on the same cycle → acc resets to 0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    pe = PE(DATA_W, ACC_W)
    pe.reset()
    await _reset(dut)
    await _step(dut, pe, load_w=1, weight_in=10)
    await _step(dut, pe, a_valid=1, a_in=5)          # acc = 50
    _check(dut, pe)
    await _step(dut, pe, clear_acc=1, a_valid=1, a_in=5)  # clear wins
    _check(dut, pe)
    assert pe.psum_out == 0


@cocotb.test()
async def test_max_operand_extremes(dut):
    """INT8 extremes: (-128) * (-128) = 16384; 127 * 127 = 16129."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    pe = PE(DATA_W, ACC_W)
    pe.reset()
    await _reset(dut)
    await _step(dut, pe, load_w=1, weight_in=-128 & 0xFF)
    await _step(dut, pe, a_valid=1, a_in=-128 & 0xFF)
    _check(dut, pe)
    assert pe.psum_out == 16384
    await _step(dut, pe, clear_acc=1)
    await _step(dut, pe, load_w=1, weight_in=127)
    await _step(dut, pe, a_valid=1, a_in=127)
    _check(dut, pe)
    assert pe.psum_out == 16129


@cocotb.test()
async def test_weight_reload_mid_accumulate(dut):
    """Re-loading weight mid-tile changes subsequent products immediately."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    pe = PE(DATA_W, ACC_W)
    pe.reset()
    await _reset(dut)
    await _step(dut, pe, load_w=1, weight_in=2)
    await _step(dut, pe, a_valid=1, a_in=10)         # acc = 20
    _check(dut, pe)
    # New weight; v1 RTL uses pre-edge weight_reg on this cycle's multiply,
    # so the first activation after load_w still uses the OLD weight.
    await _step(dut, pe, load_w=1, weight_in=5, a_valid=1, a_in=3)  # acc += 2*3=6
    _check(dut, pe)
    await _step(dut, pe, a_valid=1, a_in=4)          # acc += 5*4 = 20
    _check(dut, pe)
    assert pe.psum_out == 46, pe.psum_out


# ---------------------------------------------------------------------------
# Randomised stress — 2000 cycles of mixed stimulus, every cycle checked.
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_random_bit_exact_2000_cycles(dut):
    """2000 random cycles of mixed stimulus; every output compared to reference."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    pe = PE(DATA_W, ACC_W)
    pe.reset()
    await _reset(dut)

    rng = random.Random(0xA57ACD8E)  # deterministic seed
    mismatches = 0
    for cycle in range(2000):
        # Choose stimulus with a realistic distribution of control strobes
        stim = dict(
            load_w=1 if rng.random() < 0.05 else 0,
            clear_acc=1 if rng.random() < 0.02 else 0,
            weight_in=rng.randint(DATA_MIN, DATA_MAX) & 0xFF,
            a_valid=1 if rng.random() < 0.80 else 0,
            a_in=rng.randint(DATA_MIN, DATA_MAX) & 0xFF,
            sparse_skip=1 if rng.random() < 0.25 else 0,
            precision_mode=0,  # keep INT8 for v1
            sparse_en=1 if rng.random() < 0.3 else 0,
        )
        await _step(dut, pe, **stim)
        try:
            _check(dut, pe)
        except AssertionError as exc:
            mismatches += 1
            if mismatches <= 5:
                dut._log.error(f"cycle {cycle}: {exc}")
    assert mismatches == 0, f"{mismatches} cycles failed bit-exact check"
    dut._log.info(f"2000-cycle random stress PASS; final psum = {pe.psum_out}")


@cocotb.test()
async def test_random_bit_exact_10k_all_valid(dut):
    """10,000 cycles of always-valid activations — stresses accumulator bound."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    pe = PE(DATA_W, ACC_W)
    pe.reset()
    await _reset(dut)

    rng = random.Random(0xDEADBEEF)
    # Load a non-trivial weight and then stream random activations.
    await _step(dut, pe, load_w=1, weight_in=rng.randint(1, DATA_MAX))
    for _ in range(10_000):
        await _step(dut, pe,
                    a_valid=1,
                    a_in=rng.randint(DATA_MIN, DATA_MAX) & 0xFF)
        _check(dut, pe)
    dut._log.info(f"10k always-valid PASS; final psum = {pe.psum_out}")


@cocotb.test()
async def test_int4_packed_mode(dut):
    """WP-1: INT4 mode — weight and activation each pack 2 INT4 values.
    Each cycle accumulates (w_hi*a_hi + w_lo*a_lo). Bit-exact vs pe_ref."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    pe = PE(DATA_W, ACC_W)
    pe.reset()
    await _reset(dut)

    # weight = {w_hi=+3, w_lo=-2} packed = 0x3E (3<<4 | 0xE two's-complement -2)
    w_packed = ((3 & 0xF) << 4) | (-2 & 0xF)
    await _step(dut, pe, load_w=1, weight_in=w_packed, precision_mode=0b01)

    # act = {a_hi=+5, a_lo=+6} packed = 0x56. Expected: 3*5 + (-2)*6 = 15-12 = 3
    a_packed = ((5 & 0xF) << 4) | (6 & 0xF)
    await _step(dut, pe, a_valid=1, a_in=a_packed, precision_mode=0b01)
    _check(dut, pe)
    assert pe.psum_out == 3, f"INT4 expected 3, got {pe.psum_out}"

    # act = {a_hi=-1, a_lo=-4} packed.  Expected: 3*(-1) + (-2)*(-4) = -3+8 = 5
    a_packed = ((-1 & 0xF) << 4) | (-4 & 0xF)
    await _step(dut, pe, a_valid=1, a_in=a_packed, precision_mode=0b01)
    _check(dut, pe)
    assert pe.psum_out == 8, f"INT4 expected 3+5=8, got {pe.psum_out}"
    dut._log.info(f"INT4 packed mode PASS; psum={pe.psum_out}")


@cocotb.test()
async def test_int2_packed_mode(dut):
    """WP-1: INT2 mode — weight and activation each pack 4 INT2 values.
    Each cycle accumulates sum of 4 INT2×INT2 products."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    pe = PE(DATA_W, ACC_W)
    pe.reset()
    await _reset(dut)

    # INT2 encoding (2's complement): 00=0, 01=1, 10=-2, 11=-1
    # weight = {w3=1, w2=-1, w1=-2, w0=0} packed = 01_11_10_00 = 0x78
    w_packed = (1 << 6) | (3 << 4) | (2 << 2) | 0
    await _step(dut, pe, load_w=1, weight_in=w_packed, precision_mode=0b10)

    # act = {a3=-1, a2=1, a1=1, a0=-2} packed = 11_01_01_10 = 0xD6
    a_packed = (3 << 6) | (1 << 4) | (1 << 2) | 2
    # Expected: 1*(-1) + (-1)*1 + (-2)*1 + 0*(-2) = -1 + -1 + -2 + 0 = -4
    await _step(dut, pe, a_valid=1, a_in=a_packed, precision_mode=0b10)
    _check(dut, pe)
    assert pe.psum_out == -4, f"INT2 expected -4, got {pe.psum_out}"
    dut._log.info(f"INT2 packed mode PASS; psum={pe.psum_out}")


@cocotb.test()
async def test_mixed_precision_random(dut):
    """WP-1: 500-cycle random stress across INT8/INT4/INT2 modes, bit-exact
    vs pe_ref. Validates the multi-precision datapath is functionally
    consistent at every mode and every random mix."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    pe = PE(DATA_W, ACC_W)
    pe.reset()
    await _reset(dut)

    rng = random.Random(0x1337BEEF)
    for _ in range(500):
        mode = rng.choice([0b00, 0b01, 0b10])
        op = rng.choice(["load_w", "acc", "idle", "clear"])
        if op == "load_w":
            await _step(dut, pe, load_w=1, weight_in=rng.randint(0, 0xFF),
                        precision_mode=mode)
        elif op == "acc":
            await _step(dut, pe, a_valid=1, a_in=rng.randint(0, 0xFF),
                        precision_mode=mode)
        elif op == "clear":
            await _step(dut, pe, clear_acc=1, precision_mode=mode)
        else:
            await _step(dut, pe, precision_mode=mode)
        _check(dut, pe)
    dut._log.info(f"500-cycle mixed-precision stress PASS; psum={pe.psum_out}")
