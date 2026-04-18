"""
AstraCore Neo — Ego Motion Estimator cocotb testbench

Fuses IMU (gyro/accel) + wheel odometry into ego vx, vy, yaw_rate.
Fusion = 50/50 complementary filter on yaw rate once both sources present.
vx comes directly from wheel odometry; vy is 0 (v1).

Timing note:
  The output is a single registered stage.  After driving an input and one
  RisingEdge, the NBA has not yet settled in cocotb's active region.  Read
  one additional RisingEdge later.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


WATCHDOG_CYCLES = 500   # matches RTL default


async def reset_dut(dut):
    dut.rst_n.value            = 0
    dut.imu_valid.value        = 0
    dut.accel_x_mg.value       = 0
    dut.accel_y_mg.value       = 0
    dut.gyro_z_mdps.value      = 0
    dut.odo_valid.value        = 0
    dut.wheel_speed_mmps.value = 0
    dut.steer_mdeg.value       = 0
    dut.odo_yaw_rate_mdps.value= 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


def s16(v):
    """Convert Python int to 16-bit unsigned repr (for signed inputs)."""
    return v & 0xFFFF


def to_s32(v):
    v = int(v)
    return v if v < (1 << 31) else v - (1 << 32)


async def fire_imu(dut, gyro_mdps, accel_x_mg=0, accel_y_mg=0):
    """Fire a 1-cycle IMU valid pulse.  After returning, output NBAs are visible."""
    dut.imu_valid.value   = 1
    dut.gyro_z_mdps.value = s16(gyro_mdps)
    dut.accel_x_mg.value  = s16(accel_x_mg)
    dut.accel_y_mg.value  = s16(accel_y_mg)
    await RisingEdge(dut.clk)   # EDGE A: sample + schedule NBAs
    dut.imu_valid.value   = 0
    await RisingEdge(dut.clk)   # EDGE B: EDGE A's NBAs now visible


async def fire_odo(dut, wheel_speed_mmps, odo_yaw_mdps=0, steer_mdeg=0):
    dut.odo_valid.value         = 1
    dut.wheel_speed_mmps.value  = wheel_speed_mmps & 0xFFFF
    dut.odo_yaw_rate_mdps.value = s16(odo_yaw_mdps)
    dut.steer_mdeg.value        = s16(steer_mdeg)
    await RisingEdge(dut.clk)
    dut.odo_valid.value         = 0
    await RisingEdge(dut.clk)


async def fire_both(dut, gyro_mdps, wheel_speed_mmps, odo_yaw_mdps=0):
    """Fire IMU and odometry simultaneously on the same clock cycle."""
    dut.imu_valid.value         = 1
    dut.gyro_z_mdps.value       = s16(gyro_mdps)
    dut.odo_valid.value         = 1
    dut.wheel_speed_mmps.value  = wheel_speed_mmps & 0xFFFF
    dut.odo_yaw_rate_mdps.value = s16(odo_yaw_mdps)
    await RisingEdge(dut.clk)
    dut.imu_valid.value = 0
    dut.odo_valid.value = 0
    await RisingEdge(dut.clk)


# ===========================================================================
# Tests
# ===========================================================================

@cocotb.test()
async def test_initial_state(dut):
    """After reset: all outputs zero, no stale."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    assert int(dut.ego_valid.value)          == 0
    assert int(dut.ego_vx_mmps.value)        == 0
    assert int(dut.ego_vy_mmps.value)        == 0
    assert to_s32(dut.ego_yaw_rate_mdps.value) == 0
    assert int(dut.sensor_stale.value)       == 0
    dut._log.info("initial_state passed")


@cocotb.test()
async def test_imu_only_update(dut):
    """First-ever IMU pulse: yaw_rate=gyro, vx=0 (no odo yet), ego_valid=1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_imu(dut, gyro_mdps=1500)   # 1.5 deg/s yaw rate

    assert int(dut.ego_valid.value) == 1, "ego_valid should pulse"
    assert to_s32(dut.ego_yaw_rate_mdps.value) == 1500, \
        f"yaw_rate should be 1500 (raw IMU), got {to_s32(dut.ego_yaw_rate_mdps.value)}"
    assert to_s32(dut.ego_vx_mmps.value) == 0, "vx should be 0 (no odo)"
    dut._log.info("imu_only_update passed")


@cocotb.test()
async def test_odo_only_update(dut):
    """First-ever odo pulse: vx=wheel_speed, yaw_rate=odo_yaw, ego_valid=1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_odo(dut, wheel_speed_mmps=20000, odo_yaw_mdps=800)

    assert int(dut.ego_valid.value) == 1
    assert to_s32(dut.ego_vx_mmps.value) == 20000, \
        f"vx should be 20000, got {to_s32(dut.ego_vx_mmps.value)}"
    assert to_s32(dut.ego_yaw_rate_mdps.value) == 800, \
        f"yaw_rate should be 800 (raw odo), got {to_s32(dut.ego_yaw_rate_mdps.value)}"
    dut._log.info("odo_only_update passed")


@cocotb.test()
async def test_ego_valid_deasserts_after_one_cycle(dut):
    """ego_valid stays high for exactly 1 cycle per update."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Drive imu_valid for exactly 1 cycle
    dut.imu_valid.value   = 1
    dut.gyro_z_mdps.value = s16(100)
    await RisingEdge(dut.clk)   # EDGE A: sampled
    dut.imu_valid.value   = 0
    # At EDGE A: ego_valid still 0 (NBA not applied yet)
    assert int(dut.ego_valid.value) == 0, "ego_valid should be 0 at EDGE A"

    await RisingEdge(dut.clk)   # EDGE B: EDGE A's NBA visible → ego_valid=1
    assert int(dut.ego_valid.value) == 1, "ego_valid should be 1 at EDGE B"

    await RisingEdge(dut.clk)   # EDGE C: EDGE B's default NBA → ego_valid=0
    assert int(dut.ego_valid.value) == 0, "ego_valid should de-assert after 1 cycle"
    dut._log.info("ego_valid_deasserts passed")


@cocotb.test()
async def test_odo_after_imu_blends_yaw(dut):
    """IMU first (1000 mdps), then odo (600 mdps): fused yaw = (1000+600)>>1 = 800."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_imu(dut, gyro_mdps=1000)         # imu_has_data=1, yaw=1000 (raw)
    await fire_odo(dut, wheel_speed_mmps=15000, odo_yaw_mdps=600)
    # Now both sources have contributed — odo update blends with last IMU
    assert to_s32(dut.ego_yaw_rate_mdps.value) == ((1000 + 600) >> 1), \
        f"blended yaw should be 800, got {to_s32(dut.ego_yaw_rate_mdps.value)}"
    assert to_s32(dut.ego_vx_mmps.value) == 15000, \
        f"vx should track odo=15000, got {to_s32(dut.ego_vx_mmps.value)}"
    dut._log.info("odo_after_imu_blends_yaw passed")


@cocotb.test()
async def test_imu_after_odo_blends_yaw(dut):
    """Odo first (400 mdps), then IMU (1200 mdps): fused yaw = (1200+400)>>1 = 800."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_odo(dut, wheel_speed_mmps=12000, odo_yaw_mdps=400)
    await fire_imu(dut, gyro_mdps=1200)
    # IMU update now blends with last odo yaw
    assert to_s32(dut.ego_yaw_rate_mdps.value) == ((1200 + 400) >> 1), \
        f"blended yaw should be 800, got {to_s32(dut.ego_yaw_rate_mdps.value)}"
    # vx stays at last odo value (IMU doesn't touch vx in v1)
    assert to_s32(dut.ego_vx_mmps.value) == 12000, \
        f"vx should still be 12000, got {to_s32(dut.ego_vx_mmps.value)}"
    dut._log.info("imu_after_odo_blends_yaw passed")


@cocotb.test()
async def test_simultaneous_imu_and_odo(dut):
    """Both sources fire on same cycle: fused yaw, vx from odo."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_both(dut, gyro_mdps=500, wheel_speed_mmps=25000, odo_yaw_mdps=300)
    # Blend from raw inputs (not latches): (500 + 300) >> 1 = 400
    assert to_s32(dut.ego_yaw_rate_mdps.value) == 400, \
        f"simultaneous blend yaw should be 400, got {to_s32(dut.ego_yaw_rate_mdps.value)}"
    assert to_s32(dut.ego_vx_mmps.value) == 25000
    assert int(dut.ego_valid.value) == 1
    dut._log.info("simultaneous_imu_and_odo passed")


@cocotb.test()
async def test_vx_tracks_odo_across_updates(dut):
    """Multiple odometry updates: vx updates each time."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for speed in [1000, 5000, 12000, 8000, 0]:
        await fire_odo(dut, wheel_speed_mmps=speed, odo_yaw_mdps=0)
        assert to_s32(dut.ego_vx_mmps.value) == speed, \
            f"vx should track {speed}, got {to_s32(dut.ego_vx_mmps.value)}"
    dut._log.info("vx_tracks_odo_across_updates passed")


@cocotb.test()
async def test_negative_yaw_rate(dut):
    """Negative gyro reading produces negative ego_yaw_rate (right turn)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_imu(dut, gyro_mdps=-2500)
    assert to_s32(dut.ego_yaw_rate_mdps.value) == -2500, \
        f"yaw_rate should be -2500, got {to_s32(dut.ego_yaw_rate_mdps.value)}"
    dut._log.info("negative_yaw_rate passed")


@cocotb.test()
async def test_imu_watchdog_stale(dut):
    """No IMU for WATCHDOG_CYCLES cycles → sensor_stale[0]=1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Fire one odo pulse periodically to keep odo watchdog alive
    # For simplicity, just wait without odo — both will go stale
    for _ in range(WATCHDOG_CYCLES + 20):
        await RisingEdge(dut.clk)

    stale = int(dut.sensor_stale.value)
    assert (stale & 0x1) != 0, f"IMU stale bit should be set, got {stale:b}"
    assert (stale & 0x2) != 0, f"odo stale bit should be set, got {stale:b}"
    dut._log.info(f"imu_watchdog_stale passed: stale={stale:b}")


@cocotb.test()
async def test_stale_clears_on_new_data(dut):
    """Stale flag clears on next valid pulse from that source."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Let both go stale
    for _ in range(WATCHDOG_CYCLES + 10):
        await RisingEdge(dut.clk)
    assert int(dut.sensor_stale.value) == 0x3, "both should be stale"

    # Fire IMU pulse → IMU stale clears
    await fire_imu(dut, gyro_mdps=100)
    stale = int(dut.sensor_stale.value)
    assert (stale & 0x1) == 0, f"IMU stale should clear, got {stale:b}"
    assert (stale & 0x2) != 0, f"odo should still be stale, got {stale:b}"

    # Fire odo pulse → odo stale clears too
    await fire_odo(dut, wheel_speed_mmps=5000, odo_yaw_mdps=0)
    stale = int(dut.sensor_stale.value)
    assert stale == 0x0, f"both should be clear, got {stale:b}"
    dut._log.info("stale_clears_on_new_data passed")
