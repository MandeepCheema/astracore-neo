"""cocotb gate for rtl/npu_layernorm/npu_layernorm.v  (F1-A4).

Covers both LayerNorm and RMSNorm modes.  Comparison points:

  - `layernorm_rtl_mirror` : bit-exact Python fixed-point replay.  RTL
                             must match every output exactly.
  - `layernorm_fp32`       : FP32 oracle.  RTL rescaled to FP must meet
                             SNR >= 30 dB averaged over random batches.
"""

import os
import random
import sys
from pathlib import Path

import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from tools.npu_ref.layernorm_ref import layernorm_fp32, rmsnorm_fp32  # noqa: E402
from tools.npu_ref.layernorm_luts import (                            # noqa: E402
    IN_FRAC_BITS, layernorm_rtl_mirror, make_rsqrt_lut,
)


VEC_LEN = 256
CLK_NS  = 10


def _mask(v, w): return v & ((1 << w) - 1)


def _to_signed(val, width):
    val &= (1 << width) - 1
    if val & (1 << (width - 1)):
        return val - (1 << width)
    return val


async def _reset(dut):
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.mode.value = 0
    dut.in_valid.value = 0
    dut.in_data.value = 0
    dut.in_scale.value = 0
    dut.in_bias.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def _run_one(dut, x_int, s_int, b_int, *, rmsnorm: bool = False):
    """Run one LN/RMSNorm over `x_int, s_int, b_int`; return INT32 output."""
    assert x_int.shape == (VEC_LEN,)
    assert s_int.shape == (VEC_LEN,)
    assert b_int.shape == (VEC_LEN,)

    # Pulse start
    dut.mode.value = 1 if rmsnorm else 0
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await Timer(1, unit="ns")

    # Pass 1: stream VEC_LEN values (in_scale/in_bias ignored by RTL here,
    # but to match the mirror we drive them through — the RTL's pass-1
    # accumulate path only reads in_data).
    for i in range(VEC_LEN):
        dut.in_valid.value = 1
        dut.in_data.value  = _mask(int(x_int[i]), 32)
        dut.in_scale.value = _mask(int(s_int[i]), 32)
        dut.in_bias.value  = _mask(int(b_int[i]), 32)
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
    dut.in_valid.value = 0

    # Wait for S_RS0..S_RS3 to complete (4 cycles) + one cycle to land in S_PASS2.
    for _ in range(5):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")

    # Pass 2: stream again, collect outputs.
    y = np.zeros(VEC_LEN, dtype=np.int64)
    collected = 0
    for i in range(VEC_LEN):
        dut.in_valid.value = 1
        dut.in_data.value  = _mask(int(x_int[i]), 32)
        dut.in_scale.value = _mask(int(s_int[i]), 32)
        dut.in_bias.value  = _mask(int(b_int[i]), 32)
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.out_valid.value) == 1:
            y[collected] = _to_signed(int(dut.out_data.value), 32)
            collected += 1
    dut.in_valid.value = 0

    # Wait for done pulse.
    for _ in range(5):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.done.value) == 1:
            break

    assert collected == VEC_LEN, f"got {collected}/{VEC_LEN} outputs"
    return y.astype(np.int32)


# ---------------------------------------------------------------------------
# Directed tests
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_zero_mean_unit_var_ln(dut):
    """x=[1,-1,1,-1,...] should produce y≈x (scale=1, bias=0)."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await _reset(dut)
    sign = np.where(np.arange(VEC_LEN) % 2 == 0, 1, -1)
    x = (sign * (1 << 16)).astype(np.int32)
    s = np.full(VEC_LEN, 1 << 16, dtype=np.int32)
    b = np.zeros(VEC_LEN, dtype=np.int32)
    y = await _run_one(dut, x, s, b, rmsnorm=False)
    # Debug: read internal regs after run.
    try:
        dut._log.info(f"sum_x=0x{int(dut.sum_x.value):x}  sum_x2=0x{int(dut.sum_x2.value):x}")
        dut._log.info(f"mu=0x{int(dut.mu.value):x}  var_eps=0x{int(dut.var_eps.value):x}")
        dut._log.info(f"lzc={int(dut.lzc.value)}  rs_idx={int(dut.rs_idx.value)}  lut_val=0x{int(dut.lut_val.value):x}  inv_sigma=0x{int(dut.inv_sigma.value):x}")
    except Exception:
        pass
    mirror = layernorm_rtl_mirror(x, s, b, mode="layernorm")
    assert np.array_equal(y, mirror), (
        f"LN unit-var: RTL {y[:4]} vs mirror {mirror[:4]}")
    y_fp = y.astype(np.float64) / (1 << 16)
    assert abs(y_fp[0] - 1.0) < 0.05 and abs(y_fp[1] + 1.0) < 0.05, (
        f"expected y~[1,-1,...], got {y_fp[:4]}")


@cocotb.test()
async def test_zero_mean_unit_var_rmsnorm(dut):
    """RMSNorm on same pattern: RMS=1 so y ≈ x."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await _reset(dut)
    sign = np.where(np.arange(VEC_LEN) % 2 == 0, 1, -1)
    x = (sign * (1 << 16)).astype(np.int32)
    s = np.full(VEC_LEN, 1 << 16, dtype=np.int32)
    b = np.zeros(VEC_LEN, dtype=np.int32)
    y = await _run_one(dut, x, s, b, rmsnorm=True)
    mirror = layernorm_rtl_mirror(x, s, b, mode="rmsnorm")
    assert np.array_equal(y, mirror), (
        f"RMS: RTL {y[:4]} vs mirror {mirror[:4]}")


# ---------------------------------------------------------------------------
# Bit-exact mirror regression
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_bit_exact_ln_16_trials(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await _reset(dut)
    rng = np.random.default_rng(0xA4C0)
    mismatches = 0
    for trial in range(16):
        x_fp = rng.standard_normal(VEC_LEN).astype(np.float32)
        s_fp = rng.uniform(0.5, 2.0, VEC_LEN).astype(np.float32)
        b_fp = rng.uniform(-0.5, 0.5, VEC_LEN).astype(np.float32)
        to_i = lambda a: np.clip(np.round(a * (1 << IN_FRAC_BITS)),
                                 -(1 << 31), (1 << 31) - 1).astype(np.int32)
        x, s, b = to_i(x_fp), to_i(s_fp), to_i(b_fp)
        y_rtl = await _run_one(dut, x, s, b, rmsnorm=False)
        y_mirror = layernorm_rtl_mirror(x, s, b, mode="layernorm")
        if not np.array_equal(y_rtl, y_mirror):
            mismatches += 1
            if mismatches <= 2:
                diffs = np.where(y_rtl != y_mirror)[0][:5]
                dut._log.error(
                    f"LN trial {trial} mismatch @{diffs.tolist()}: "
                    f"rtl={y_rtl[diffs].tolist()} mirror={y_mirror[diffs].tolist()}")
    assert mismatches == 0, f"{mismatches}/16 LN mismatches"
    dut._log.info("bit-exact LN mirror: 16 trials PASS")


@cocotb.test()
async def test_bit_exact_rmsnorm_8_trials(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await _reset(dut)
    rng = np.random.default_rng(0xA4C1)
    mismatches = 0
    for trial in range(8):
        x_fp = rng.standard_normal(VEC_LEN).astype(np.float32)
        s_fp = rng.uniform(0.5, 2.0, VEC_LEN).astype(np.float32)
        to_i = lambda a: np.clip(np.round(a * (1 << IN_FRAC_BITS)),
                                 -(1 << 31), (1 << 31) - 1).astype(np.int32)
        x, s = to_i(x_fp), to_i(s_fp)
        b = np.zeros(VEC_LEN, dtype=np.int32)
        y_rtl = await _run_one(dut, x, s, b, rmsnorm=True)
        y_mirror = layernorm_rtl_mirror(x, s, b, mode="rmsnorm")
        if not np.array_equal(y_rtl, y_mirror):
            mismatches += 1
    assert mismatches == 0, f"{mismatches}/8 RMS mismatches"


# ---------------------------------------------------------------------------
# SNR vs FP32 oracle (spec gate)
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_snr_vs_fp32_ln(dut):
    """Aggregate SNR >= 30 dB over 16 LN trials."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await _reset(dut)
    rng = np.random.default_rng(0xBADC001)
    total_sig = 0.0
    total_noise = 0.0
    for trial in range(16):
        x_fp = rng.standard_normal(VEC_LEN).astype(np.float32)
        s_fp = rng.uniform(0.5, 2.0, VEC_LEN).astype(np.float32)
        b_fp = rng.uniform(-0.5, 0.5, VEC_LEN).astype(np.float32)
        to_i = lambda a: np.clip(np.round(a * (1 << IN_FRAC_BITS)),
                                 -(1 << 31), (1 << 31) - 1).astype(np.int32)
        x, s, b = to_i(x_fp), to_i(s_fp), to_i(b_fp)
        y_rtl = (await _run_one(dut, x, s, b, rmsnorm=False)).astype(np.float64) / (1 << 16)
        y_fp  = layernorm_fp32(x_fp, s_fp, b_fp, axis=-1)
        noise = y_rtl - y_fp
        total_sig   += float(np.sum(y_fp   ** 2))
        total_noise += float(np.sum(noise  ** 2))
    snr = 10.0 * np.log10(total_sig / max(total_noise, 1e-30))
    dut._log.info(f"LN SNR vs FP32 oracle: {snr:.2f} dB")
    assert snr >= 30.0, f"LN SNR {snr:.2f} dB below 30 dB spec"
