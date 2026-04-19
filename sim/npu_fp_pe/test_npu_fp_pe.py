"""cocotb gate for rtl/npu_fp/npu_fp_pe.v  (F1-A1).

Weight-stationary FP PE.  Exercises:
  1. Weight latches on load_w pulse and is held across activations.
  2. Activation pass-through (a_in -> a_out with 1-cycle delay) is
     unchanged in FP mode (needed for systolic wavefront).
  3. sparse_skip is respected (skipped element does not accumulate).
  4. clear_acc zeros the accumulator.
  5. Dot-product accuracy matches `fp_mac` oracle.
"""

import os
import struct
import sys
from pathlib import Path

import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from tools.npu_ref.fp_ref import (     # noqa: E402
    fp32_to_fp16, fp_mac,
)

# reuse FP encoders from the npu_fp_mac test
sys.path.insert(0, str(_HERE.parent / "npu_fp_mac"))
from test_npu_fp_mac import (           # noqa: E402
    _bits_fp16, _bits_e4m3, _bits_e5m2, _fp64_from_bits,
    MODE_FP8_E4M3, MODE_FP8_E5M2, MODE_FP16,
)

CLK_NS = 10


async def _reset(dut):
    dut.rst_n.value = 0
    dut.precision_mode.value = 0
    dut.sparse_en.value = 0
    dut.load_w.value = 0
    dut.clear_acc.value = 0
    dut.weight_in.value = 0
    dut.a_valid.value = 0
    dut.a_in.value = 0
    dut.sparse_skip.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def _load_weight(dut, w_bits):
    dut.load_w.value = 1
    dut.weight_in.value = w_bits
    await RisingEdge(dut.clk)
    dut.load_w.value = 0
    await Timer(1, unit="ns")


async def _clear(dut, mode):
    dut.precision_mode.value = mode
    dut.clear_acc.value = 1
    dut.a_valid.value = 0
    await RisingEdge(dut.clk)
    dut.clear_acc.value = 0
    await Timer(1, unit="ns")


async def _stream(dut, a_bits, *, mode, sparse_skip=0):
    dut.precision_mode.value = mode
    dut.a_in.value = a_bits
    dut.a_valid.value = 1
    dut.sparse_skip.value = sparse_skip
    await RisingEdge(dut.clk)
    dut.a_valid.value = 0
    dut.sparse_skip.value = 0
    await Timer(1, unit="ns")


def _read_acc(dut) -> float:
    return _fp64_from_bits(int(dut.psum_out.value))


# ---------------------------------------------------------------------------
# Directed
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_fp16_weight_stationary_dot(dut):
    """Load w=2.0, stream a=[1,2,3], expect 2*(1+2+3) = 12."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await _reset(dut)
    await _clear(dut, MODE_FP16)
    await _load_weight(dut, _bits_fp16(2.0))
    for v in (1.0, 2.0, 3.0):
        await _stream(dut, _bits_fp16(v), mode=MODE_FP16)
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    got = _read_acc(dut)
    dut._log.info(f"FP16 weight-stat dot (w=2, a=[1,2,3]) = {got:.3f}  expect 12.0")
    assert abs(got - 12.0) < 0.01


@cocotb.test()
async def test_activation_passthrough(dut):
    """a_in should appear on a_out one cycle later, regardless of FP mode."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await _reset(dut)
    dut.precision_mode.value = MODE_FP16
    dut.a_in.value = 0xABCD
    dut.a_valid.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    assert int(dut.a_out.value) == 0xABCD, (
        f"a_out={int(dut.a_out.value):#x} expected 0xABCD")
    assert int(dut.a_valid_out.value) == 1


@cocotb.test()
async def test_sparse_skip_blocks_accumulate(dut):
    """sparse_skip=1 must skip the MAC even though a_valid=1."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await _reset(dut)
    await _clear(dut, MODE_FP16)
    await _load_weight(dut, _bits_fp16(1.0))
    # Stream a=5 with sparse_skip=1 — accumulator stays at 0.
    await _stream(dut, _bits_fp16(5.0), mode=MODE_FP16, sparse_skip=1)
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    got = _read_acc(dut)
    assert abs(got) < 1e-3, f"skipped mac accumulated: got {got}"
    # Now without skip — should produce 5.0
    await _stream(dut, _bits_fp16(5.0), mode=MODE_FP16)
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    got = _read_acc(dut)
    assert abs(got - 5.0) < 0.01, f"unskipped mac got {got}, expected 5"


@cocotb.test()
async def test_weight_stationary_dot_vs_oracle_all_modes(dut):
    """64-length weight-stationary dot product in each mode; RTL vs fp_mac."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await _reset(dut)
    rng = np.random.default_rng(0xABCDEF)
    # For FP8 the range is limited, scale inputs down.
    spec = [
        ("fp16",     MODE_FP16,     _bits_fp16, 64, 1.0, 0.05),
        ("fp8_e4m3", MODE_FP8_E4M3, _bits_e4m3, 32, 1.5, 5.0),
        ("fp8_e5m2", MODE_FP8_E5M2, _bits_e5m2, 32, 1.5, 5.0),
    ]
    for precision, mode_code, encoder, N, sigma, tol in spec:
        await _clear(dut, mode_code)
        # Sample one weight + N activations.
        w_fp = float(rng.standard_normal() * sigma)
        a_fp = rng.standard_normal(N).astype(np.float32) * sigma
        await _load_weight(dut, encoder(w_fp))
        acc_ref = np.float32(0.0)
        for a in a_fp:
            acc_ref = fp_mac(np.float32(a), np.float32(w_fp),
                             precision=precision, acc=acc_ref)
            await _stream(dut, encoder(a), mode=mode_code)
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        got = _read_acc(dut)
        err = abs(got - float(acc_ref))
        dut._log.info(f"{precision:8s}: rtl={got:+.4f} ref={float(acc_ref):+.4f} err={err:.2e}")
        assert err < tol, f"{precision} err {err} > tol {tol}"
