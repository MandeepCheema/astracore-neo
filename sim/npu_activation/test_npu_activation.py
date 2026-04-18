"""cocotb bit-exact regression for rtl/npu_activation/npu_activation.v.

Each test drives RTL and the Python reference with identical stimulus and
compares out_data + out_saturated every cycle.  Covers every implemented
mode plus boundary conditions (saturation boundaries, reserved modes).
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
from tools.npu_ref.activation_ref import (  # noqa: E402
    Activation,
    MODE_PASS, MODE_RELU, MODE_LEAKY_RELU,
    MODE_CLIP_INT8, MODE_RELU_CLIP_INT8,
)


ACC_W = 32
OUT_W = 32


def _mask(v, w): return v & ((1 << w) - 1)


def _to_signed(val, width):
    val &= (1 << width) - 1
    if val & (1 << (width - 1)):
        return val - (1 << width)
    return val


async def _reset(dut):
    dut.rst_n.value = 0
    dut.mode.value = 0
    dut.in_valid.value = 0
    dut.in_data.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def _step(dut, ref: Activation, *, in_valid=0, in_data=0, mode=0):
    dut.in_valid.value = in_valid
    dut.in_data.value = _mask(in_data, ACC_W)
    dut.mode.value = mode
    ref.tick(in_valid=in_valid, in_data=in_data, mode=mode)
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


def _check(dut, ref: Activation):
    rtl_valid = int(dut.out_valid.value)
    rtl_data  = _to_signed(int(dut.out_data.value), OUT_W)
    rtl_sat   = int(dut.out_saturated.value)
    assert rtl_valid == ref.out_valid, (
        f"out_valid mismatch: rtl={rtl_valid} ref={ref.out_valid}")
    assert rtl_data  == ref.out_data_signed, (
        f"out_data mismatch: rtl={rtl_data} ref={ref.out_data_signed}")
    assert rtl_sat   == ref.out_saturated, (
        f"out_saturated mismatch: rtl={rtl_sat} ref={ref.out_saturated}")


# ---------------------------------------------------------------------------
# Directed tests
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_reset(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = Activation(acc_w=ACC_W, out_w=OUT_W)
    ref.reset()
    await _reset(dut)
    _check(dut, ref)


@cocotb.test()
async def test_pass_mode(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = Activation(); ref.reset()
    await _reset(dut)
    for v in (0, 1, -1, 127, -128, 0x7FFFFFFF, -0x80000000):
        await _step(dut, ref, in_valid=1, in_data=v, mode=MODE_PASS)
        _check(dut, ref)


@cocotb.test()
async def test_relu_mode(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = Activation(); ref.reset()
    await _reset(dut)
    for v in (0, 1, 50, -1, -50, 127, -128, 1000, -1000):
        await _step(dut, ref, in_valid=1, in_data=v, mode=MODE_RELU)
        _check(dut, ref)


@cocotb.test()
async def test_leaky_relu_mode(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = Activation(); ref.reset()
    await _reset(dut)
    # Positive passes through unchanged
    await _step(dut, ref, in_valid=1, in_data=80, mode=MODE_LEAKY_RELU)
    _check(dut, ref)
    # -64 >> 3 = -8 exactly
    await _step(dut, ref, in_valid=1, in_data=-64, mode=MODE_LEAKY_RELU)
    _check(dut, ref)
    # -63 >> 3 = -8 (floor, toward -inf)
    await _step(dut, ref, in_valid=1, in_data=-63, mode=MODE_LEAKY_RELU)
    _check(dut, ref)
    # -1 >> 3 = -1 (floor)
    await _step(dut, ref, in_valid=1, in_data=-1, mode=MODE_LEAKY_RELU)
    _check(dut, ref)


@cocotb.test()
async def test_clip_int8_mode(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = Activation(); ref.reset()
    await _reset(dut)
    # Boundary cases
    for v in (127, -128, 128, -129, 500, -500, 0, 1, -1):
        await _step(dut, ref, in_valid=1, in_data=v, mode=MODE_CLIP_INT8)
        _check(dut, ref)


@cocotb.test()
async def test_relu_clip_int8_mode(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = Activation(); ref.reset()
    await _reset(dut)
    for v in (-1000, -1, 0, 1, 127, 128, 1000):
        await _step(dut, ref, in_valid=1, in_data=v, mode=MODE_RELU_CLIP_INT8)
        _check(dut, ref)


@cocotb.test()
async def test_lut_modes_silu_gelu_sigmoid(dut):
    """WP-7: MODE_SILU / MODE_GELU / MODE_SIGMOID produce LUT-backed values
    bit-exact against the Python reference's LUT dictionary. Sweep both
    in-range and out-of-range (saturating) inputs."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = Activation(); ref.reset()
    await _reset(dut)
    test_values = [0, 16, -16, 32, -32, 64, -64, 127, -128,
                   200, -200, 1000, -1000]   # last four test saturation
    for mode_code, name in ((0b101, "SILU"), (0b110, "GELU"), (0b111, "SIGMOID")):
        for v in test_values:
            await _step(dut, ref, in_valid=1, in_data=v, mode=mode_code)
            _check(dut, ref)
        dut._log.info(f"LUT mode {name} exact vs ref on {len(test_values)} values")


@cocotb.test()
async def test_in_valid_gates_output(dut):
    """When in_valid=0, out_valid falls; when in_valid=1, it rises next cycle."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = Activation(); ref.reset()
    await _reset(dut)
    # Drive a value then go idle
    await _step(dut, ref, in_valid=1, in_data=42, mode=MODE_RELU)
    _check(dut, ref)
    await _step(dut, ref, in_valid=0)
    _check(dut, ref)
    # Several idle cycles
    for _ in range(3):
        await _step(dut, ref, in_valid=0)
        _check(dut, ref)
    # Re-assert
    await _step(dut, ref, in_valid=1, in_data=-77, mode=MODE_RELU)
    _check(dut, ref)


# ---------------------------------------------------------------------------
# Random stress
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_random_bit_exact_2000_cycles(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = Activation(); ref.reset()
    await _reset(dut)
    rng = random.Random(0xAC71A7E)
    modes = [MODE_PASS, MODE_RELU, MODE_LEAKY_RELU, MODE_CLIP_INT8,
             MODE_RELU_CLIP_INT8]
    mismatches = 0
    for cycle in range(2000):
        stim = dict(
            in_valid=1 if rng.random() < 0.85 else 0,
            in_data=rng.randint(-(1 << 20), (1 << 20) - 1),
            mode=rng.choice(modes),
        )
        await _step(dut, ref, **stim)
        try:
            _check(dut, ref)
        except AssertionError as exc:
            mismatches += 1
            if mismatches <= 3:
                dut._log.error(f"cycle {cycle}: {exc}")
    assert mismatches == 0, f"{mismatches} cycles failed"
    dut._log.info(f"2000-cycle random stress PASS")
