"""
AstraCore Neo — ThermalZone cocotb testbench.

Python ThermalZone is the GOLDEN REFERENCE. The Verilog DUT must produce
matching state outputs for every temperature stimulus.

State encoding:
  Python ThermalState  →  3-bit Verilog state
  NOMINAL   (1)        →  3'd0
  WARNING   (2)        →  3'd1
  THROTTLED (3)        →  3'd2
  CRITICAL  (4)        →  3'd3
  SHUTDOWN  (5)        →  3'd4
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from telemetry.thermal import ThermalZone, ThermalZoneConfig, ThermalState
from telemetry.exceptions import ThermalShutdownError

STATE_MAP = {
    ThermalState.NOMINAL:   0,
    ThermalState.WARNING:   1,
    ThermalState.THROTTLED: 2,
    ThermalState.CRITICAL:  3,
    ThermalState.SHUTDOWN:  4,
}


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.valid.value = 0
    dut.temp_in.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def drive_temp(dut, ref: ThermalZone, temp_c: int):
    """Drive one temperature sample into the DUT and Python reference."""
    dut.temp_in.value = temp_c
    dut.valid.value = 1
    await RisingEdge(dut.clk)
    dut.valid.value = 0
    await RisingEdge(dut.clk)

    # Advance Python reference (may raise ThermalShutdownError at >= shutdown_c)
    reading = None
    try:
        reading = ref.update(float(temp_c))
    except ThermalShutdownError:
        pass
    return reading


@cocotb.test()
async def test_nominal_state(dut):
    """Temperature below warning → NOMINAL (0)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = ThermalZone("test")
    await drive_temp(dut, ref, 50)

    assert dut.state.value == 0, f"Expected NOMINAL(0), got {dut.state.value}"
    assert dut.throttle_en.value == 0
    assert dut.shutdown_req.value == 0
    dut._log.info("nominal test passed")


@cocotb.test()
async def test_warning_state(dut):
    """Temperature >= 75 and < 85 → WARNING (1)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = ThermalZone("test")
    await drive_temp(dut, ref, 80)

    assert dut.state.value == 1, f"Expected WARNING(1), got {dut.state.value}"
    assert dut.throttle_en.value == 0
    dut._log.info("warning test passed")


@cocotb.test()
async def test_throttled_state(dut):
    """Temperature >= 85 and < 95 → THROTTLED (2)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = ThermalZone("test")
    await drive_temp(dut, ref, 90)

    assert dut.state.value == 2, f"Expected THROTTLED(2), got {dut.state.value}"
    assert dut.throttle_en.value == 1, "throttle_en should be asserted"
    dut._log.info("throttled test passed")


@cocotb.test()
async def test_critical_state(dut):
    """Temperature >= 95 and < 105 → CRITICAL (3)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = ThermalZone("test")
    await drive_temp(dut, ref, 100)

    assert dut.state.value == 3, f"Expected CRITICAL(3), got {dut.state.value}"
    dut._log.info("critical test passed")


@cocotb.test()
async def test_shutdown_state(dut):
    """Temperature >= 105 → SHUTDOWN (4)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = ThermalZone("test")
    await drive_temp(dut, ref, 110)

    assert dut.state.value == 4, f"Expected SHUTDOWN(4), got {dut.state.value}"
    assert dut.shutdown_req.value == 1, "shutdown_req should be asserted"
    dut._log.info("shutdown test passed")


@cocotb.test()
async def test_threshold_boundaries(dut):
    """Test exact threshold boundaries: 74, 75, 84, 85, 94, 95, 104, 105."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    cases = [
        (74,  0),   # NOMINAL
        (75,  1),   # WARNING (exactly at threshold)
        (84,  1),   # WARNING
        (85,  2),   # THROTTLED
        (94,  2),   # THROTTLED
        (95,  3),   # CRITICAL
        (104, 3),   # CRITICAL
        (105, 4),   # SHUTDOWN
    ]

    ref = ThermalZone("test")
    for temp, expected_state in cases:
        await drive_temp(dut, ref, temp)
        actual = int(dut.state.value)
        assert actual == expected_state, (
            f"Temp={temp}: expected state={expected_state}, got {actual}"
        )

    dut._log.info("boundary test passed")


@cocotb.test()
async def test_state_sequence_reference_match(dut):
    """Drive a mixed temperature sequence and verify DUT matches Python ref."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    temps = [30, 50, 75, 80, 85, 88, 95, 98, 105, 60, 40]
    ref = ThermalZone("test")

    for temp in temps:
        if temp < 105:   # skip shutdown (raises exception in Python model)
            reading = await drive_temp(dut, ref, temp)
            expected = STATE_MAP[reading.state]
            actual = int(dut.state.value)
            assert actual == expected, (
                f"Temp={temp}: DUT state={actual} REF state={expected} ({reading.state})"
            )

    dut._log.info("sequence match test passed")


@cocotb.test()
async def test_throttle_en_follows_state(dut):
    """throttle_en must be 1 exactly when state == THROTTLED (2)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = ThermalZone("test")
    for temp in [50, 80, 90, 100, 40]:
        await drive_temp(dut, ref, temp)
        expected_throttle = (int(dut.state.value) == 2)
        assert int(dut.throttle_en.value) == expected_throttle, (
            f"Temp={temp}: throttle_en={dut.throttle_en.value} "
            f"but state={dut.state.value}"
        )

    dut._log.info("throttle_en signal test passed")


@cocotb.test()
async def test_reset_clears_state(dut):
    """After reset, state must return to NOMINAL (0)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = ThermalZone("test")
    await drive_temp(dut, ref, 90)   # drive to THROTTLED

    dut.rst_n.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    assert dut.state.value == 0, "State should be NOMINAL after reset"
    assert dut.throttle_en.value == 0
    assert dut.shutdown_req.value == 0
    dut._log.info("reset test passed")


@cocotb.test()
async def test_no_change_without_valid(dut):
    """State should not change when valid is low."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = ThermalZone("test")
    # Drive to THROTTLED
    await drive_temp(dut, ref, 90)
    assert dut.state.value == 2

    # Now change temp_in but keep valid=0 for 5 cycles
    dut.temp_in.value = 20   # would be NOMINAL
    dut.valid.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)

    # State should still be THROTTLED
    assert dut.state.value == 2, "State should not change without valid"
    dut._log.info("no_change_without_valid test passed")
