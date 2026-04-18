"""
AstraCore Neo — Detection Arbiter cocotb testbench

Round-robin arbitration over 3 detection sources (camera / radar / lidar).
Registered output: 1-cycle latency from *_valid to out_valid.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


def to_s32(v):
    v = int(v)
    return v if v < (1 << 31) else v - (1 << 32)


async def reset_dut(dut):
    dut.rst_n.value          = 0
    dut.cam_valid.value      = 0
    dut.rad_valid.value      = 0
    dut.lid_valid.value      = 0
    dut.cam_x_mm.value       = 0
    dut.cam_y_mm.value       = 0
    dut.cam_z_mm.value       = 0
    dut.cam_class_id.value   = 0
    dut.cam_confidence.value = 0
    dut.rad_x_mm.value       = 0
    dut.rad_y_mm.value       = 0
    dut.rad_z_mm.value       = 0
    dut.rad_class_id.value   = 0
    dut.rad_confidence.value = 0
    dut.lid_x_mm.value       = 0
    dut.lid_y_mm.value       = 0
    dut.lid_z_mm.value       = 0
    dut.lid_class_id.value   = 0
    dut.lid_confidence.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def drive_src(dut, src, x, y, z, cls=1, conf=100):
    """Drive one *_valid pulse for 1 clock; read output after the NBA settles."""
    if src == "cam":
        dut.cam_valid.value      = 1
        dut.cam_x_mm.value       = x & 0xFFFFFFFF
        dut.cam_y_mm.value       = y & 0xFFFFFFFF
        dut.cam_z_mm.value       = z & 0xFFFFFFFF
        dut.cam_class_id.value   = cls
        dut.cam_confidence.value = conf
    elif src == "rad":
        dut.rad_valid.value      = 1
        dut.rad_x_mm.value       = x & 0xFFFFFFFF
        dut.rad_y_mm.value       = y & 0xFFFFFFFF
        dut.rad_z_mm.value       = z & 0xFFFFFFFF
        dut.rad_class_id.value   = cls
        dut.rad_confidence.value = conf
    elif src == "lid":
        dut.lid_valid.value      = 1
        dut.lid_x_mm.value       = x & 0xFFFFFFFF
        dut.lid_y_mm.value       = y & 0xFFFFFFFF
        dut.lid_z_mm.value       = z & 0xFFFFFFFF
        dut.lid_class_id.value   = cls
        dut.lid_confidence.value = conf
    await RisingEdge(dut.clk)   # EDGE A: sampled
    dut.cam_valid.value = 0
    dut.rad_valid.value = 0
    dut.lid_valid.value = 0
    await RisingEdge(dut.clk)   # EDGE B: output visible


# ===========================================================================
# Tests
# ===========================================================================

@cocotb.test()
async def test_initial_state(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    assert int(dut.out_valid.value) == 0
    assert int(dut.cam_ack.value)   == 0
    assert int(dut.rad_ack.value)   == 0
    assert int(dut.lid_ack.value)   == 0
    dut._log.info("initial_state passed")


@cocotb.test()
async def test_camera_alone_passes_through(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await drive_src(dut, "cam", 1000, 500, 0, cls=2, conf=200)
    assert int(dut.out_valid.value)    == 1
    assert int(dut.out_sensor_id.value) == 0
    assert to_s32(dut.out_x_mm.value)  == 1000
    assert to_s32(dut.out_y_mm.value)  == 500
    assert int(dut.out_class_id.value) == 2
    dut._log.info("camera_alone_passes_through passed")


@cocotb.test()
async def test_radar_alone_passes_through(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await drive_src(dut, "rad", -2000, 1000, 0)
    assert int(dut.out_sensor_id.value) == 1
    assert to_s32(dut.out_x_mm.value)   == -2000
    dut._log.info("radar_alone_passes_through passed")


@cocotb.test()
async def test_lidar_alone_passes_through(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await drive_src(dut, "lid", 3000, -1500, 200)
    assert int(dut.out_sensor_id.value) == 2
    assert to_s32(dut.out_x_mm.value)   == 3000
    assert to_s32(dut.out_y_mm.value)   == -1500
    dut._log.info("lidar_alone_passes_through passed")


@cocotb.test()
async def test_round_robin_fairness(dut):
    """All 3 sources asserted simultaneously across consecutive cycles
    should take turns in round-robin order."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Hold all three valid continuously; sample the winner ID each cycle
    dut.cam_valid.value = 1
    dut.rad_valid.value = 1
    dut.lid_valid.value = 1
    dut.cam_x_mm.value = 100
    dut.rad_x_mm.value = 200
    dut.lid_x_mm.value = 300

    winners = []
    for _ in range(9):
        await RisingEdge(dut.clk)
        # Output is registered — sampled here reflects the previous edge
        if int(dut.out_valid.value) == 1:
            winners.append(int(dut.out_sensor_id.value))
    dut.cam_valid.value = 0
    dut.rad_valid.value = 0
    dut.lid_valid.value = 0

    # After the 1-cycle pipeline delay, we expect a round-robin pattern
    # starting at camera (priority_idx=0 initially): cam, rad, lid, cam, rad, lid, ...
    dut._log.info(f"winners observed: {winners}")
    assert len(winners) >= 6, f"not enough winners captured: {len(winners)}"
    # Drop the first (pipeline priming) and check 6 consecutive are rotating
    tail = winners[-6:]
    # Each unique ID must appear exactly twice
    assert tail.count(0) == 2
    assert tail.count(1) == 2
    assert tail.count(2) == 2
    dut._log.info("round_robin_fairness passed")


@cocotb.test()
async def test_ack_matches_winner(dut):
    """When a source wins, only its *_ack fires for that cycle."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Camera alone
    await drive_src(dut, "cam", 1, 2, 3)
    assert int(dut.cam_ack.value) == 1
    assert int(dut.rad_ack.value) == 0
    assert int(dut.lid_ack.value) == 0

    # Radar alone
    await drive_src(dut, "rad", 4, 5, 6)
    assert int(dut.cam_ack.value) == 0
    assert int(dut.rad_ack.value) == 1
    assert int(dut.lid_ack.value) == 0

    # LiDAR alone
    await drive_src(dut, "lid", 7, 8, 9)
    assert int(dut.cam_ack.value) == 0
    assert int(dut.rad_ack.value) == 0
    assert int(dut.lid_ack.value) == 1
    dut._log.info("ack_matches_winner passed")


@cocotb.test()
async def test_out_valid_low_when_no_source(dut):
    """With no *_valid asserted, out_valid stays low."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    for _ in range(5):
        await RisingEdge(dut.clk)
    assert int(dut.out_valid.value) == 0
    dut._log.info("out_valid_low_when_no_source passed")


@cocotb.test()
async def test_no_starvation_cam_when_rad_always_high(dut):
    """If radar is always asserted, camera still wins at least once in 3 cycles."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    dut.rad_valid.value = 1
    dut.cam_valid.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)
    # Within 3 edges cam should have been picked at least once
    # Continue checking over 6 cycles
    cam_wins = 0
    rad_wins = 0
    for _ in range(6):
        await RisingEdge(dut.clk)
        if int(dut.out_valid.value) == 1:
            if int(dut.out_sensor_id.value) == 0:
                cam_wins += 1
            elif int(dut.out_sensor_id.value) == 1:
                rad_wins += 1

    dut.cam_valid.value = 0
    dut.rad_valid.value = 0

    assert cam_wins >= 2, f"camera should get fair share, got {cam_wins} wins"
    assert rad_wins >= 2, f"radar should also win, got {rad_wins} wins"
    dut._log.info(f"no_starvation passed: cam={cam_wins} rad={rad_wins}")
