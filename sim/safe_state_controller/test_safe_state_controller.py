"""
AstraCore Neo — Safe-State Controller cocotb testbench

State ladder: NORMAL(0) → ALERT(1) → DEGRADE(2) → MRC(3)
Critical fault escalates up the ladder over time; warnings only force ALERT.
MRC is absorbing (only operator_reset clears it).

Timer values here use defaults: ALERT=2000ms, DEGRADE=3000ms, RECOVER=5000ms.
Each tick_1ms pulse advances the internal timer by 1 ms.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


ALERT_TIME_MS   = 2000
DEGRADE_TIME_MS = 3000
RECOVER_TIME_MS = 5000

STATE_NORMAL  = 0
STATE_ALERT   = 1
STATE_DEGRADE = 2
STATE_MRC     = 3


async def reset_dut(dut):
    dut.rst_n.value           = 0
    dut.critical_faults.value = 0
    dut.warning_faults.value  = 0
    dut.tick_1ms.value        = 0
    dut.operator_reset.value  = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def advance_ms(dut, n):
    """Fire n consecutive tick_1ms pulses (each = 1 simulated ms)."""
    for _ in range(n):
        dut.tick_1ms.value = 1
        await RisingEdge(dut.clk)
        dut.tick_1ms.value = 0
        await RisingEdge(dut.clk)


async def set_fault(dut, critical=0, warning=0):
    """Set fault inputs and advance one clock so they register."""
    dut.critical_faults.value = critical
    dut.warning_faults.value  = warning
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


# ===========================================================================
# Tests
# ===========================================================================

@cocotb.test()
async def test_initial_state(dut):
    """After reset: NORMAL, full speed, no alerts."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    assert int(dut.safe_state.value)    == STATE_NORMAL
    assert int(dut.max_speed_kmh.value) == 130
    assert int(dut.alert_driver.value)  == 0
    assert int(dut.limit_speed.value)   == 0
    assert int(dut.mrc_pull_over.value) == 0
    dut._log.info("initial_state passed")


@cocotb.test()
async def test_critical_fault_immediate_alert(dut):
    """Critical fault → ALERT immediately (no timer wait)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await set_fault(dut, critical=0x01)
    assert int(dut.safe_state.value)   == STATE_ALERT
    assert int(dut.alert_driver.value) == 1
    assert int(dut.limit_speed.value)  == 0
    dut._log.info("critical_fault_immediate_alert passed")


@cocotb.test()
async def test_warning_only_goes_to_alert(dut):
    """Warning fault → ALERT, but never escalates."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await set_fault(dut, warning=0x04)
    assert int(dut.safe_state.value) == STATE_ALERT

    # Sustain the warning long enough that a critical would escalate
    await advance_ms(dut, ALERT_TIME_MS + 100)
    assert int(dut.safe_state.value) == STATE_ALERT, \
        f"warning alone should not escalate, got state {int(dut.safe_state.value)}"
    dut._log.info("warning_only_goes_to_alert passed")


@cocotb.test()
async def test_critical_escalates_alert_to_degrade(dut):
    """Sustained critical fault for ALERT_TIME_MS → DEGRADE."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await set_fault(dut, critical=0x02)
    assert int(dut.safe_state.value) == STATE_ALERT

    await advance_ms(dut, ALERT_TIME_MS)
    assert int(dut.safe_state.value)    == STATE_DEGRADE
    assert int(dut.limit_speed.value)   == 1
    assert int(dut.max_speed_kmh.value) == 60
    dut._log.info("alert_to_degrade passed")


@cocotb.test()
async def test_critical_escalates_degrade_to_mrc(dut):
    """After reaching DEGRADE, sustained critical fault → MRC."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await set_fault(dut, critical=0x01)
    await advance_ms(dut, ALERT_TIME_MS)     # → DEGRADE
    await advance_ms(dut, DEGRADE_TIME_MS)   # → MRC

    assert int(dut.safe_state.value)    == STATE_MRC
    assert int(dut.max_speed_kmh.value) == 5
    assert int(dut.mrc_pull_over.value) == 1
    dut._log.info("degrade_to_mrc passed")


@cocotb.test()
async def test_mrc_is_absorbing(dut):
    """Once in MRC, clearing faults doesn't recover."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await set_fault(dut, critical=0x01)
    await advance_ms(dut, ALERT_TIME_MS)
    await advance_ms(dut, DEGRADE_TIME_MS)
    assert int(dut.safe_state.value) == STATE_MRC

    # Clear faults and wait a long time
    await set_fault(dut, critical=0, warning=0)
    await advance_ms(dut, RECOVER_TIME_MS * 2)
    assert int(dut.safe_state.value) == STATE_MRC, \
        "MRC must not auto-recover"
    dut._log.info("mrc_is_absorbing passed")


@cocotb.test()
async def test_operator_reset_clears_mrc(dut):
    """operator_reset pulse while in MRC returns to NORMAL."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await set_fault(dut, critical=0x08)
    await advance_ms(dut, ALERT_TIME_MS)
    await advance_ms(dut, DEGRADE_TIME_MS)
    assert int(dut.safe_state.value) == STATE_MRC

    # Clear faults first, then pulse operator_reset
    await set_fault(dut, critical=0, warning=0)
    dut.operator_reset.value = 1
    await RisingEdge(dut.clk)
    dut.operator_reset.value = 0
    await RisingEdge(dut.clk)

    assert int(dut.safe_state.value)    == STATE_NORMAL
    assert int(dut.max_speed_kmh.value) == 130
    assert int(dut.latched_faults.value) == 0
    dut._log.info("operator_reset_clears_mrc passed")


@cocotb.test()
async def test_fault_clear_auto_recovers_from_degrade(dut):
    """Sustained critical → DEGRADE, clear fault, RECOVER_TIME_MS → ALERT."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await set_fault(dut, critical=0x01)
    await advance_ms(dut, ALERT_TIME_MS)
    assert int(dut.safe_state.value) == STATE_DEGRADE

    await set_fault(dut, critical=0, warning=0)
    await advance_ms(dut, RECOVER_TIME_MS)
    assert int(dut.safe_state.value) == STATE_ALERT

    await advance_ms(dut, RECOVER_TIME_MS)
    assert int(dut.safe_state.value) == STATE_NORMAL
    dut._log.info("fault_clear_auto_recovers passed")


@cocotb.test()
async def test_latched_faults_sticky(dut):
    """latched_faults accumulate all observed fault bits until operator_reset."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await set_fault(dut, critical=0x03, warning=0x40)
    latched = int(dut.latched_faults.value)
    assert (latched & 0x03) == 0x03, f"criticals not latched: {latched:04x}"
    assert (latched & (0x40 << 8)) == (0x40 << 8), f"warnings not latched: {latched:04x}"

    # Clear inputs; latches should hold
    await set_fault(dut, critical=0, warning=0)
    latched2 = int(dut.latched_faults.value)
    assert latched2 == latched, "latched_faults should be sticky"
    dut._log.info(f"latched_faults_sticky passed: 0x{latched:04x}")


@cocotb.test()
async def test_critical_priority_over_warning(dut):
    """Both critical and warning present: treated as critical (escalates)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await set_fault(dut, critical=0x01, warning=0xFF)
    await advance_ms(dut, ALERT_TIME_MS)
    assert int(dut.safe_state.value) == STATE_DEGRADE, \
        "critical+warning should escalate like critical alone"
    dut._log.info("critical_priority passed")


@cocotb.test()
async def test_speed_limits_per_state(dut):
    """Verify max_speed_kmh decreases with each escalation."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    assert int(dut.max_speed_kmh.value) == 130   # NORMAL

    await set_fault(dut, critical=0x01)          # ALERT
    assert int(dut.max_speed_kmh.value) == 130

    await advance_ms(dut, ALERT_TIME_MS)         # DEGRADE
    assert int(dut.max_speed_kmh.value) == 60

    await advance_ms(dut, DEGRADE_TIME_MS)       # MRC
    assert int(dut.max_speed_kmh.value) == 5
    dut._log.info("speed_limits_per_state passed")
