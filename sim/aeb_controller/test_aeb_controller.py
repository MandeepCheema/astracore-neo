"""
AstraCore Neo — AEB Controller cocotb testbench

4-level FSM on TTC flags: 0=OFF / 1=WARN / 2=PRECHARGE / 3=EMERGENCY
Escalation immediate; de-escalation requires CLEAR_TICKS consecutive clear
events AND (if at EMERGENCY) brake_hold_ms timer expired.

Timing: registered FSM — read results after 2 RisingEdges from ttc pulse.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


CLEAR_TICKS  = 5
MIN_BRAKE_MS = 500


async def reset_dut(dut):
    dut.rst_n.value        = 0
    dut.ttc_valid.value    = 0
    dut.ttc_track_id.value = 0
    dut.ttc_warning.value  = 0
    dut.ttc_prepare.value  = 0
    dut.ttc_brake.value    = 0
    dut.tick_1ms.value     = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def fire_ttc(dut, warn=0, prep=0, brake=0, track_id=1):
    """Fire a 1-cycle TTC pulse; result visible after 2 RisingEdges."""
    dut.ttc_track_id.value = track_id
    dut.ttc_warning.value  = warn
    dut.ttc_prepare.value  = prep
    dut.ttc_brake.value    = brake
    dut.ttc_valid.value    = 1
    await RisingEdge(dut.clk)
    dut.ttc_valid.value    = 0
    dut.ttc_warning.value  = 0
    dut.ttc_prepare.value  = 0
    dut.ttc_brake.value    = 0
    await RisingEdge(dut.clk)


async def fire_tick_ms(dut, n=1):
    """Fire n consecutive 1ms ticks (each advances brake_hold_ms by 1)."""
    for _ in range(n):
        dut.tick_1ms.value = 1
        await RisingEdge(dut.clk)
        dut.tick_1ms.value = 0
        await RisingEdge(dut.clk)


# ===========================================================================
# Tests
# ===========================================================================

@cocotb.test()
async def test_initial_state(dut):
    """After reset: level=0, no brake, no alert."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    assert int(dut.brake_level.value)  == 0
    assert int(dut.brake_active.value) == 0
    assert int(dut.alert_driver.value) == 0
    assert int(dut.target_decel_mms2.value) == 0
    dut._log.info("initial_state passed")


@cocotb.test()
async def test_warning_activates_level_1(dut):
    """ttc_warning → level 1, alert on, no brake."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_ttc(dut, warn=1, track_id=42)
    assert int(dut.brake_level.value)  == 1
    assert int(dut.alert_driver.value) == 1
    assert int(dut.brake_active.value) == 0
    assert int(dut.target_decel_mms2.value) == 0
    assert int(dut.active_threat_id.value) == 42
    dut._log.info("warning_activates_level_1 passed")


@cocotb.test()
async def test_prepare_activates_level_2(dut):
    """ttc_prepare → level 2, brake active at precharge decel."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_ttc(dut, prep=1, track_id=5)
    assert int(dut.brake_level.value)       == 2
    assert int(dut.brake_active.value)      == 1
    assert int(dut.alert_driver.value)      == 1
    assert int(dut.target_decel_mms2.value) == 2000
    dut._log.info("prepare_activates_level_2 passed")


@cocotb.test()
async def test_brake_activates_level_3_with_hold(dut):
    """ttc_brake → level 3, emergency decel, brake_hold_ms loaded."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_ttc(dut, brake=1, track_id=7)
    assert int(dut.brake_level.value)       == 3
    assert int(dut.target_decel_mms2.value) == 10000
    assert int(dut.brake_hold_ms.value)     == MIN_BRAKE_MS
    assert int(dut.brake_active.value)      == 1
    dut._log.info("brake_activates_level_3 passed")


@cocotb.test()
async def test_escalation_warn_to_prep_to_emerg(dut):
    """Progressive escalation jumps each level immediately."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_ttc(dut, warn=1)
    assert int(dut.brake_level.value) == 1

    await fire_ttc(dut, prep=1)
    assert int(dut.brake_level.value) == 2

    await fire_ttc(dut, brake=1)
    assert int(dut.brake_level.value) == 3
    dut._log.info("escalation_warn_to_prep_to_emerg passed")


@cocotb.test()
async def test_same_level_holds(dut):
    """Repeat same TTC level → stays at that level, no downgrade."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for _ in range(10):
        await fire_ttc(dut, warn=1)
    assert int(dut.brake_level.value) == 1
    dut._log.info("same_level_holds passed")


@cocotb.test()
async def test_downgrade_requires_clear_ticks(dut):
    """From level 2, CLEAR_TICKS clear events required to drop to level 1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_ttc(dut, prep=1)
    assert int(dut.brake_level.value) == 2

    # First CLEAR_TICKS-1 clear events: still at level 2
    for i in range(CLEAR_TICKS - 1):
        await fire_ttc(dut)   # no flags
        assert int(dut.brake_level.value) == 2, \
            f"should hold at 2 after {i+1} clears"

    # The CLEAR_TICKS-th clear event triggers downgrade
    await fire_ttc(dut)
    assert int(dut.brake_level.value) == 1, "should downgrade to 1"
    dut._log.info("downgrade_requires_clear_ticks passed")


@cocotb.test()
async def test_full_release_to_idle(dut):
    """From level 2, 2 * CLEAR_TICKS clears → level 0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_ttc(dut, prep=1)
    # downgrade 2→1
    for _ in range(CLEAR_TICKS):
        await fire_ttc(dut)
    assert int(dut.brake_level.value) == 1
    # downgrade 1→0
    for _ in range(CLEAR_TICKS):
        await fire_ttc(dut)
    assert int(dut.brake_level.value) == 0
    assert int(dut.brake_active.value) == 0
    assert int(dut.alert_driver.value) == 0
    dut._log.info("full_release_to_idle passed")


@cocotb.test()
async def test_emergency_hold_blocks_downgrade(dut):
    """While brake_hold_ms > 0, no downgrade from level 3."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_ttc(dut, brake=1)
    assert int(dut.brake_level.value) == 3
    assert int(dut.brake_hold_ms.value) == MIN_BRAKE_MS

    # Fire many clear ttc events — level should NOT drop (hold still active)
    for _ in range(CLEAR_TICKS * 3):
        await fire_ttc(dut)
    assert int(dut.brake_level.value) == 3, \
        "level 3 should be locked by brake_hold_ms"
    dut._log.info("emergency_hold_blocks_downgrade passed")


@cocotb.test()
async def test_hold_expires_then_downgrade(dut):
    """After MIN_BRAKE_MS tick_1ms pulses, hold expires and downgrade allowed."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_ttc(dut, brake=1)
    # Drive MIN_BRAKE_MS 1ms ticks to expire the hold
    await fire_tick_ms(dut, n=MIN_BRAKE_MS)
    assert int(dut.brake_hold_ms.value) == 0

    # Now CLEAR_TICKS clear events should downgrade to 2
    for _ in range(CLEAR_TICKS):
        await fire_ttc(dut)
    assert int(dut.brake_level.value) == 2, \
        f"should downgrade to 2 after hold expiry, got {int(dut.brake_level.value)}"
    dut._log.info("hold_expires_then_downgrade passed")


@cocotb.test()
async def test_reescalation_resets_clear_counter(dut):
    """Partial clear then re-threat → clear_cnt resets, no premature downgrade."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_ttc(dut, prep=1)
    # 3 clear events (below CLEAR_TICKS threshold)
    for _ in range(3):
        await fire_ttc(dut)
    assert int(dut.brake_level.value) == 2

    # Threat returns at same level → resets clear counter
    await fire_ttc(dut, prep=1)

    # Now we need a FULL CLEAR_TICKS to downgrade
    for _ in range(CLEAR_TICKS - 1):
        await fire_ttc(dut)
    assert int(dut.brake_level.value) == 2, "should still be at 2"

    await fire_ttc(dut)   # CLEAR_TICKS-th clear
    assert int(dut.brake_level.value) == 1
    dut._log.info("reescalation_resets_clear_counter passed")


@cocotb.test()
async def test_target_decel_tracks_level(dut):
    """target_decel_mms2 updates as level changes."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    assert int(dut.target_decel_mms2.value) == 0

    await fire_ttc(dut, warn=1)
    assert int(dut.target_decel_mms2.value) == 0, "level 1 has no brake torque"

    await fire_ttc(dut, prep=1)
    assert int(dut.target_decel_mms2.value) == 2000

    await fire_ttc(dut, brake=1)
    assert int(dut.target_decel_mms2.value) == 10000
    dut._log.info("target_decel_tracks_level passed")
