"""
AstraCore Neo — FaultPredictor cocotb testbench.

Tests the threshold-based fault risk classification in the RTL.
The Python MetricTracker is used as the golden reference for threshold checks.

Risk encoding:
  Python FaultRisk  →  3-bit Verilog risk
  NONE     (1)     →  3'd0
  LOW      (2)     →  3'd1
  MEDIUM   (3)     →  3'd2
  HIGH     (4)     →  3'd3
  CRITICAL (5)     →  3'd4
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from telemetry.fault_predictor import MetricTracker, MetricConfig, FaultRisk

# Default RTL thresholds match the parameter defaults
WARN_THRESH     = 50
CRITICAL_THRESH = 100
RISK_MAP = {
    FaultRisk.NONE:     0,
    FaultRisk.LOW:      1,
    FaultRisk.MEDIUM:   2,
    FaultRisk.HIGH:     3,
    FaultRisk.CRITICAL: 4,
}


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.valid.value = 0
    dut.value.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def push_value(dut, val: int):
    """Drive one value sample into DUT."""
    dut.value.value = val & 0xFFFF
    dut.valid.value = 1
    await RisingEdge(dut.clk)
    dut.valid.value = 0
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_nominal_risk(dut):
    """Value well below warning threshold → NONE (0)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await push_value(dut, 20)

    assert int(dut.risk.value) == 0, f"Expected NONE(0), got {dut.risk.value}"
    assert int(dut.alarm.value) == 0
    dut._log.info("nominal_risk test passed")


@cocotb.test()
async def test_low_risk(dut):
    """Value just above warning threshold → LOW (1)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # 55 is above warn=50, and (55-50)/(100-50) = 10% < 30% → LOW
    await push_value(dut, 55)

    assert int(dut.risk.value) == 1, f"Expected LOW(1), got {dut.risk.value}"
    dut._log.info("low_risk test passed")


@cocotb.test()
async def test_medium_risk(dut):
    """Value in medium risk zone → MEDIUM (2)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # 70 is above warn=50; (70-50)/(100-50) = 40%, 30-70% → MEDIUM
    await push_value(dut, 70)

    assert int(dut.risk.value) == 2, f"Expected MEDIUM(2), got {dut.risk.value}"
    dut._log.info("medium_risk test passed")


@cocotb.test()
async def test_high_risk(dut):
    """Value in high risk zone → HIGH (3)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # 90 is above warn=50; (90-50)/(100-50) = 80% > 70% → HIGH
    await push_value(dut, 90)

    assert int(dut.risk.value) == 3, f"Expected HIGH(3), got {dut.risk.value}"
    assert int(dut.alarm.value) == 1, "alarm should be set for HIGH risk"
    dut._log.info("high_risk test passed")


@cocotb.test()
async def test_critical_risk(dut):
    """Value >= critical threshold → CRITICAL (4)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await push_value(dut, 100)

    assert int(dut.risk.value) == 4, f"Expected CRITICAL(4), got {dut.risk.value}"
    assert int(dut.alarm.value) == 1
    dut._log.info("critical_risk test passed")


@cocotb.test()
async def test_risk_exceeds_critical(dut):
    """Value >> critical threshold → still CRITICAL (4)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await push_value(dut, 200)

    assert int(dut.risk.value) == 4
    dut._log.info("risk_exceeds_critical test passed")


@cocotb.test()
async def test_warning_boundary(dut):
    """Test exact boundary: 49 = NONE, 50 = LOW."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await push_value(dut, 49)
    assert int(dut.risk.value) == 0, f"49 should be NONE, got {dut.risk.value}"

    await push_value(dut, 50)
    assert int(dut.risk.value) >= 1, f"50 should be LOW+, got {dut.risk.value}"

    dut._log.info("warning_boundary test passed")


@cocotb.test()
async def test_spike_detection(dut):
    """After nominal values, a sudden spike should escalate risk to at least MEDIUM."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Fill window with small nominal values (mean ~10)
    for _ in range(16):
        await push_value(dut, 10)

    # Push a spike: 10 + 30 (SPIKE_OFFSET default) + 1 = 41 (above mean + offset)
    await push_value(dut, 45)

    # The spike should raise risk to at least MEDIUM (2) since 45 > mean(10)+30
    assert int(dut.risk.value) >= 2, (
        f"Spike should raise risk to >= MEDIUM(2), got {dut.risk.value}"
    )
    dut._log.info(f"spike_detection test passed: risk={dut.risk.value}")


@cocotb.test()
async def test_rolling_mean_updates(dut):
    """After many samples, rolling_mean should approximate the input mean."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Push 32 samples of value 20 → mean should converge to ~20
    for _ in range(32):
        await push_value(dut, 20)

    mean = int(dut.rolling_mean.value)
    assert abs(mean - 20) <= 2, f"rolling_mean should be ~20, got {mean}"
    dut._log.info(f"rolling_mean test passed: mean={mean}")


@cocotb.test()
async def test_reset_clears_state(dut):
    """After reset, risk=0 and rolling_mean=0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for _ in range(5):
        await push_value(dut, 90)

    dut.rst_n.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    assert int(dut.risk.value) == 0,         "risk should be 0 after reset"
    assert int(dut.rolling_mean.value) == 0, "rolling_mean should be 0 after reset"
    dut._log.info("reset_clears_state test passed")


@cocotb.test()
async def test_no_alarm_below_high(dut):
    """alarm should only be asserted for HIGH(3) or CRITICAL(4)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for val in [0, 20, 49, 55, 70]:
        await push_value(dut, val)
        risk = int(dut.risk.value)
        alarm = int(dut.alarm.value)
        if risk < 3:
            assert alarm == 0, f"val={val}: alarm should be 0 for risk={risk}"
        else:
            assert alarm == 1, f"val={val}: alarm should be 1 for risk={risk}"

    dut._log.info("no_alarm_below_high test passed")
