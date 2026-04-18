"""
AstraCore Neo — Plausibility Checker cocotb testbench

Enforces ISO 26262 ASIL-D cross-sensor redundancy rules.
1-cycle pipeline: check_valid → check_done one clock later.

Sensor mask bits: [0]=CAM, [1]=RAD, [2]=LID, [3]=US
Classes: 1=VEHICLE, 2=PEDESTRIAN, 3=PROXIMITY, 4=LANE
Violations: 0=none, 1=no_redundancy, 2=low_conf, 3=unknown_class
ASIL: 0x00=kept, 0x01=degraded, 0xFF=rejected
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


MIN_CONF = 64

S_CAM = 0b0001
S_RAD = 0b0010
S_LID = 0b0100
S_US  = 0b1000

CLASS_VEHICLE    = 1
CLASS_PEDESTRIAN = 2
CLASS_PROXIMITY  = 3
CLASS_LANE       = 4

VIO_NONE          = 0
VIO_NO_REDUNDANCY = 1
VIO_LOW_CONF      = 2
VIO_UNKNOWN_CLASS = 3

ASIL_D_KEEP = 0x00
ASIL_B_DEG  = 0x01
ASIL_REJECT = 0xFF


async def reset_dut(dut):
    dut.rst_n.value             = 0
    dut.check_valid.value       = 0
    dut.check_class_id.value    = 0
    dut.check_sensor_mask.value = 0
    dut.check_confidence.value  = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def do_check(dut, class_id, sensor_mask, confidence=200):
    """
    Drive a check request and wait for the registered result.

    Timing: EDGE A samples check_valid, EDGE B makes the NBA result visible.
    """
    dut.check_class_id.value    = class_id
    dut.check_sensor_mask.value = sensor_mask
    dut.check_confidence.value  = confidence
    dut.check_valid.value       = 1
    await RisingEdge(dut.clk)   # EDGE A: sample, schedule result NBAs
    dut.check_valid.value       = 0
    await RisingEdge(dut.clk)   # EDGE B: result now visible


# ===========================================================================
# Tests
# ===========================================================================

@cocotb.test()
async def test_initial_state(dut):
    """After reset: no done pulse, counters zero."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    assert int(dut.check_done.value)       == 0
    assert int(dut.total_checks.value)     == 0
    assert int(dut.total_violations.value) == 0
    dut._log.info("initial_state passed")


@cocotb.test()
async def test_vehicle_cam_and_radar_passes(dut):
    """VEHICLE with camera+radar → ok, ASIL-D kept."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await do_check(dut, CLASS_VEHICLE, S_CAM | S_RAD)

    assert int(dut.check_done.value)      == 1
    assert int(dut.check_ok.value)        == 1
    assert int(dut.check_violation.value) == VIO_NONE
    assert int(dut.asil_degrade.value)    == ASIL_D_KEEP
    dut._log.info("vehicle_cam_and_radar passed")


@cocotb.test()
async def test_vehicle_cam_only_fails(dut):
    """VEHICLE with camera only → no redundancy, degrade to ASIL-B."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await do_check(dut, CLASS_VEHICLE, S_CAM)

    assert int(dut.check_ok.value)        == 0
    assert int(dut.check_violation.value) == VIO_NO_REDUNDANCY
    assert int(dut.asil_degrade.value)    == ASIL_B_DEG
    dut._log.info("vehicle_cam_only passed")


@cocotb.test()
async def test_vehicle_radar_only_fails(dut):
    """VEHICLE with radar only → no redundancy (missing camera)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await do_check(dut, CLASS_VEHICLE, S_RAD)
    assert int(dut.check_ok.value) == 0
    assert int(dut.check_violation.value) == VIO_NO_REDUNDANCY
    dut._log.info("vehicle_radar_only passed")


@cocotb.test()
async def test_pedestrian_cam_and_radar_passes(dut):
    """PEDESTRIAN with cam+radar (Camera AND (Radar OR LiDAR)) → ok."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await do_check(dut, CLASS_PEDESTRIAN, S_CAM | S_RAD)
    assert int(dut.check_ok.value) == 1
    assert int(dut.asil_degrade.value) == ASIL_D_KEEP
    dut._log.info("pedestrian_cam_radar passed")


@cocotb.test()
async def test_pedestrian_cam_and_lidar_passes(dut):
    """PEDESTRIAN with cam+LiDAR → ok (LiDAR satisfies redundancy)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await do_check(dut, CLASS_PEDESTRIAN, S_CAM | S_LID)
    assert int(dut.check_ok.value) == 1
    dut._log.info("pedestrian_cam_lidar passed")


@cocotb.test()
async def test_pedestrian_cam_only_fails(dut):
    """PEDESTRIAN with cam only → no redundancy."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await do_check(dut, CLASS_PEDESTRIAN, S_CAM)
    assert int(dut.check_ok.value) == 0
    assert int(dut.check_violation.value) == VIO_NO_REDUNDANCY
    assert int(dut.asil_degrade.value) == ASIL_B_DEG
    dut._log.info("pedestrian_cam_only passed")


@cocotb.test()
async def test_proximity_us_and_cam_passes(dut):
    """PROXIMITY with ultrasonic+camera → ok."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await do_check(dut, CLASS_PROXIMITY, S_US | S_CAM)
    assert int(dut.check_ok.value) == 1
    dut._log.info("proximity_us_cam passed")


@cocotb.test()
async def test_proximity_cam_only_fails(dut):
    """PROXIMITY with camera only → missing ultrasonic → degrade."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await do_check(dut, CLASS_PROXIMITY, S_CAM)
    assert int(dut.check_ok.value) == 0
    assert int(dut.asil_degrade.value) == ASIL_B_DEG
    dut._log.info("proximity_cam_only passed")


@cocotb.test()
async def test_lane_cam_only_passes(dut):
    """LANE with camera only → ok (no redundancy required)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await do_check(dut, CLASS_LANE, S_CAM)
    assert int(dut.check_ok.value) == 1
    assert int(dut.asil_degrade.value) == ASIL_D_KEEP
    dut._log.info("lane_cam_only passed")


@cocotb.test()
async def test_lane_no_camera_fails(dut):
    """LANE without camera → fails (camera required for lane)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await do_check(dut, CLASS_LANE, S_RAD | S_LID)
    assert int(dut.check_ok.value) == 0
    assert int(dut.check_violation.value) == VIO_NO_REDUNDANCY
    dut._log.info("lane_no_camera passed")


@cocotb.test()
async def test_unknown_class_rejected(dut):
    """Unknown class id → reject outright (ASIL 0xFF)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await do_check(dut, 99, S_CAM | S_RAD | S_LID | S_US)   # all sensors but unknown class
    assert int(dut.check_ok.value) == 0
    assert int(dut.check_violation.value) == VIO_UNKNOWN_CLASS
    assert int(dut.asil_degrade.value) == ASIL_REJECT
    dut._log.info("unknown_class_rejected passed")


@cocotb.test()
async def test_low_confidence_fails(dut):
    """Confidence below MIN_CONFIDENCE → low-conf violation, degrade."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await do_check(dut, CLASS_VEHICLE, S_CAM | S_RAD, confidence=MIN_CONF - 1)
    assert int(dut.check_ok.value) == 0
    assert int(dut.check_violation.value) == VIO_LOW_CONF
    assert int(dut.asil_degrade.value) == ASIL_B_DEG
    dut._log.info("low_confidence_fails passed")


@cocotb.test()
async def test_statistics_counters(dut):
    """total_checks and total_violations count correctly across multiple checks."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # 3 passing checks
    await do_check(dut, CLASS_VEHICLE, S_CAM | S_RAD)
    await do_check(dut, CLASS_PEDESTRIAN, S_CAM | S_LID)
    await do_check(dut, CLASS_LANE, S_CAM)
    # 2 failing checks
    await do_check(dut, CLASS_VEHICLE, S_CAM)              # no redundancy
    await do_check(dut, CLASS_PROXIMITY, S_CAM, confidence=10)  # low conf

    assert int(dut.total_checks.value)     == 5, f"checks: {int(dut.total_checks.value)}"
    assert int(dut.total_violations.value) == 2, f"violations: {int(dut.total_violations.value)}"
    dut._log.info(f"statistics passed: checks=5, violations=2")
