"""
AstraCore Neo — MAC Array cocotb testbench.

Validates the signed INT8 multiply-accumulate unit.
The Python numpy INT8 computation is the GOLDEN REFERENCE.

Each valid cycle: result = result + (a * b)
With clear=1:      result = (a * b)   [start of new dot product]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import numpy as np


def s8(v: int) -> int:
    """Convert to signed 8-bit (Python int) and back to unsigned for Verilog."""
    v = v & 0xFF
    return v


def expected_s32(acc: int, a: int, b: int) -> int:
    """Compute expected 32-bit signed accumulation: acc + int8(a)*int8(b)."""
    # Interpret as signed
    sa = a if a < 128 else a - 256
    sb = b if b < 128 else b - 256
    result = (acc + sa * sb) & 0xFFFFFFFF
    return result


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.valid.value = 0
    dut.clear.value = 0
    dut.a.value     = 0
    dut.b.value     = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def mac_op(dut, a: int, b: int, clear: int = 0):
    """Perform one MAC operation. Returns result (sampled one cycle after valid)."""
    dut.a.value     = a & 0xFF
    dut.b.value     = b & 0xFF
    dut.clear.value = clear
    dut.valid.value = 1
    await RisingEdge(dut.clk)
    dut.valid.value = 0
    await RisingEdge(dut.clk)   # result registered on next cycle
    return int(dut.result.value)


@cocotb.test()
async def test_single_positive_multiply(dut):
    """3 * 4 = 12 (with clear)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    result = await mac_op(dut, a=3, b=4, clear=1)
    assert result == 12, f"3*4 should be 12, got {result}"
    dut._log.info("single_positive_multiply passed")


@cocotb.test()
async def test_accumulate(dut):
    """3*4 then 5*6 accumulate: result = 12 + 30 = 42."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await mac_op(dut, a=3, b=4, clear=1)   # 12
    result = await mac_op(dut, a=5, b=6, clear=0)  # 12 + 30 = 42
    assert result == 42, f"Expected 42, got {result}"
    dut._log.info("accumulate test passed")


@cocotb.test()
async def test_clear_resets_accumulator(dut):
    """After accumulating, clear should restart from a*b (not 0)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await mac_op(dut, a=10, b=10, clear=1)   # 100
    await mac_op(dut, a=5, b=5, clear=0)     # 100 + 25 = 125
    result = await mac_op(dut, a=2, b=3, clear=1)  # clear: just 2*3 = 6
    assert result == 6, f"After clear, expected 6, got {result}"
    dut._log.info("clear_resets_accumulator test passed")


@cocotb.test()
async def test_signed_negative_multiply(dut):
    """(-2) * 3 = -6 (signed)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    a = s8(-2)   # 0xFE
    b = s8(3)    # 0x03
    result = await mac_op(dut, a=a, b=b, clear=1)

    # Interpret 32-bit signed
    expected = (-2 * 3) & 0xFFFFFFFF   # = 0xFFFFFFFA = -6 in two's complement
    assert result == expected, f"(-2)*3 should be {expected:#010x}, got {result:#010x}"
    dut._log.info(f"signed_negative_multiply passed: result=0x{result:08X}")


@cocotb.test()
async def test_signed_both_negative(dut):
    """(-5) * (-7) = 35."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    result = await mac_op(dut, a=s8(-5), b=s8(-7), clear=1)
    assert result == 35, f"(-5)*(-7) should be 35, got {result}"
    dut._log.info("signed_both_negative test passed")


@cocotb.test()
async def test_max_positive_multiply(dut):
    """127 * 127 = 16129."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    result = await mac_op(dut, a=127, b=127, clear=1)
    assert result == 16129, f"127*127 should be 16129, got {result}"
    dut._log.info("max_positive_multiply test passed")


@cocotb.test()
async def test_max_negative_magnitude(dut):
    """(-128) * 127 = -16256."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    result = await mac_op(dut, a=s8(-128), b=127, clear=1)
    expected = (-128 * 127) & 0xFFFFFFFF
    assert result == expected, f"(-128)*127 expected 0x{expected:08X}, got 0x{result:08X}"
    dut._log.info("max_negative_magnitude test passed")


@cocotb.test()
async def test_dot_product_matches_numpy(dut):
    """Compute dot product of two 8-element vectors; compare with numpy."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    a_vec = [10, -20, 30, -5, 127, -128, 50, -50]
    b_vec = [3,   7, -2, 15, -1,    1, 100,  -80]

    # Numpy reference (use int32 to avoid overflow)
    np_a = np.array(a_vec, dtype=np.int8)
    np_b = np.array(b_vec, dtype=np.int8)
    np_result = int(np.dot(np_a.astype(np.int32), np_b.astype(np.int32)))
    expected = np_result & 0xFFFFFFFF

    # Drive DUT
    for i, (a, b) in enumerate(zip(a_vec, b_vec)):
        clear = 1 if i == 0 else 0
        result = await mac_op(dut, a=s8(a), b=s8(b), clear=clear)

    assert result == expected, (
        f"Dot product mismatch: DUT=0x{result:08X} ({result}) "
        f"NumPy=0x{expected:08X} ({np_result})"
    )
    dut._log.info(f"dot_product_matches_numpy passed: result={np_result}")


@cocotb.test()
async def test_ready_follows_valid(dut):
    """ready should pulse one cycle after valid."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    dut.a.value     = 5
    dut.b.value     = 5
    dut.clear.value = 1
    dut.valid.value = 1
    await RisingEdge(dut.clk)
    dut.valid.value = 0

    # On the very next rising edge, ready should be 1
    await RisingEdge(dut.clk)
    assert int(dut.ready.value) == 1, "ready should be 1 one cycle after valid"

    # Then drop back
    await RisingEdge(dut.clk)
    assert int(dut.ready.value) == 0, "ready should drop when valid is 0"
    dut._log.info("ready_follows_valid test passed")


@cocotb.test()
async def test_zero_inputs(dut):
    """0 * anything = 0; accumulate stays at previous."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await mac_op(dut, a=10, b=10, clear=1)   # acc = 100
    result = await mac_op(dut, a=0, b=127, clear=0)  # acc = 100 + 0 = 100
    assert result == 100, f"0*127 should not change accumulator, got {result}"
    dut._log.info("zero_inputs test passed")


@cocotb.test()
async def test_reset_clears_result(dut):
    """After reset, result should be 0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await mac_op(dut, a=50, b=50, clear=1)

    dut.rst_n.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    assert int(dut.result.value) == 0, f"result should be 0 after reset"
    dut._log.info("reset_clears_result test passed")
