"""
AstraCore Neo — LDW/LKA Controller cocotb testbench

Consumes fused lane estimate, raises LDW warning at |offset| > 600 mm,
engages LKA torque at |offset| > 900 mm.
torque = K_TORQUE * center_offset_mm, clamped to ±MAX_TORQUE_MNM.

Coordinate convention:
  offset > 0 → ego drifted LEFT  (lane center is to the right)
  offset < 0 → ego drifted RIGHT (lane center is to the left)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


WARN_THRESH_MM = 600
ACT_THRESH_MM  = 900
K_TORQUE       = 5
MAX_TORQUE_MNM = 5000

FS_NONE  = 0
FS_MAP   = 1
FS_CAM   = 2
FS_BLEND = 3

DIR_NONE  = 0
DIR_LEFT  = 1
DIR_RIGHT = 2


def to_s16(v):
    v = int(v)
    return v if v < (1 << 15) else v - (1 << 16)


async def reset_dut(dut):
    dut.rst_n.value            = 0
    dut.lane_valid.value       = 0
    dut.center_offset_mm.value = 0
    dut.lane_width_mm.value    = 0
    dut.fusion_source.value    = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def fire_lane(dut, offset_mm, width_mm=3600, source=FS_BLEND):
    dut.center_offset_mm.value = offset_mm & 0xFFFFFFFF
    dut.lane_width_mm.value    = width_mm & 0xFFFFFFFF
    dut.fusion_source.value    = source
    dut.lane_valid.value       = 1
    await RisingEdge(dut.clk)
    dut.lane_valid.value       = 0
    await RisingEdge(dut.clk)


# ===========================================================================
# Tests
# ===========================================================================

@cocotb.test()
async def test_initial_state(dut):
    """After reset: LDW off, LKA off, torque 0, direction none."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    assert int(dut.ldw_warning.value)            == 0
    assert int(dut.lka_active.value)             == 0
    assert to_s16(dut.steering_torque_mnm.value) == 0
    assert int(dut.departure_direction.value)    == DIR_NONE
    dut._log.info("initial_state passed")


@cocotb.test()
async def test_centered_no_warning(dut):
    """offset = 0: no LDW, no LKA."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_lane(dut, offset_mm=0)
    assert int(dut.ldw_warning.value) == 0
    assert int(dut.lka_active.value)  == 0
    assert int(dut.departure_direction.value) == DIR_NONE
    dut._log.info("centered_no_warning passed")


@cocotb.test()
async def test_small_drift_no_warning(dut):
    """|offset| just below warn threshold: no LDW."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_lane(dut, offset_mm=WARN_THRESH_MM)   # exactly at threshold
    assert int(dut.ldw_warning.value) == 0, "should not fire at exact threshold"
    dut._log.info("small_drift_no_warning passed")


@cocotb.test()
async def test_ldw_warning_left_drift(dut):
    """offset = +700 mm (drifted left): LDW fires, direction = LEFT, no LKA."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_lane(dut, offset_mm=700)
    assert int(dut.ldw_warning.value)         == 1
    assert int(dut.lka_active.value)          == 0
    assert int(dut.departure_direction.value) == DIR_LEFT
    assert to_s16(dut.steering_torque_mnm.value) == 0, "no torque below LKA thresh"
    dut._log.info("ldw_warning_left_drift passed")


@cocotb.test()
async def test_ldw_warning_right_drift(dut):
    """offset = -700 mm (drifted right): LDW fires, direction = RIGHT."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_lane(dut, offset_mm=-700)
    assert int(dut.ldw_warning.value)         == 1
    assert int(dut.departure_direction.value) == DIR_RIGHT
    dut._log.info("ldw_warning_right_drift passed")


@cocotb.test()
async def test_lka_engages_and_torque_sign(dut):
    """offset = +1000 mm (drifted left): LKA active, positive torque (steer right)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_lane(dut, offset_mm=1000)
    assert int(dut.ldw_warning.value) == 1
    assert int(dut.lka_active.value)  == 1
    expected_torque = 1000 * K_TORQUE  # 5000 mNm, exactly at clamp
    assert to_s16(dut.steering_torque_mnm.value) == expected_torque, \
        f"torque: got {to_s16(dut.steering_torque_mnm.value)}, expected {expected_torque}"
    dut._log.info(f"lka_engages positive: torque={expected_torque}")


@cocotb.test()
async def test_lka_engages_negative_torque(dut):
    """offset = -1000 mm: LKA negative torque (steer left)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_lane(dut, offset_mm=-1000)
    assert int(dut.lka_active.value) == 1
    expected_torque = -1000 * K_TORQUE
    assert to_s16(dut.steering_torque_mnm.value) == expected_torque
    dut._log.info(f"lka negative: torque={expected_torque}")


@cocotb.test()
async def test_torque_clamp_positive(dut):
    """Very large positive drift → torque saturates at +MAX_TORQUE_MNM."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_lane(dut, offset_mm=5000)   # K*5000 = 25000 > MAX
    assert to_s16(dut.steering_torque_mnm.value) == MAX_TORQUE_MNM
    dut._log.info("torque_clamp_positive passed")


@cocotb.test()
async def test_torque_clamp_negative(dut):
    """Very large negative drift → torque saturates at -MAX_TORQUE_MNM."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_lane(dut, offset_mm=-5000)
    assert to_s16(dut.steering_torque_mnm.value) == -MAX_TORQUE_MNM
    dut._log.info("torque_clamp_negative passed")


@cocotb.test()
async def test_no_data_disables_ldw_lka(dut):
    """fusion_source=00 (no data): outputs hold at zero regardless of offset."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_lane(dut, offset_mm=2000, source=FS_NONE)
    assert int(dut.ldw_warning.value) == 0
    assert int(dut.lka_active.value)  == 0
    assert to_s16(dut.steering_torque_mnm.value) == 0
    dut._log.info("no_data_disables passed")


@cocotb.test()
async def test_recovery_clears_lka(dut):
    """After LKA engages, return to centered → outputs clear."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_lane(dut, offset_mm=1200)
    assert int(dut.lka_active.value) == 1

    await fire_lane(dut, offset_mm=0)
    assert int(dut.ldw_warning.value) == 0
    assert int(dut.lka_active.value)  == 0
    assert to_s16(dut.steering_torque_mnm.value) == 0
    assert int(dut.departure_direction.value)    == DIR_NONE
    dut._log.info("recovery_clears_lka passed")


@cocotb.test()
async def test_cam_only_source_still_works(dut):
    """fusion_source=cam-only (10): LDW/LKA still operates."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_lane(dut, offset_mm=800, source=FS_CAM)
    assert int(dut.ldw_warning.value) == 1
    dut._log.info("cam_only_source passed")
