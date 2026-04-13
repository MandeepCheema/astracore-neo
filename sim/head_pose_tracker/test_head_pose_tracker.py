"""
AstraCore Neo — HeadPoseTracker cocotb testbench.

Python HeadPoseTracker is the GOLDEN REFERENCE.
Angles are represented as signed 8-bit integers (degrees).

Encoding:
  Python float angle (degrees) → signed 8-bit integer (truncated to int)
  Valid range: -128 to +127 degrees
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from dms.head_pose import HeadPoseTracker, AttentionZone


def angle_to_s8(deg: float) -> int:
    """Convert float angle to signed 8-bit integer for DUT. Clamp to [-128, 127]."""
    v = int(deg)
    return max(-128, min(127, v))


def s8_to_verilog(v: int) -> int:
    """Convert signed Python int to unsigned 8-bit Verilog value."""
    return v & 0xFF


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.valid.value = 0
    dut.yaw.value   = 0
    dut.pitch.value = 0
    dut.roll.value  = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def drive_pose(dut, ref: HeadPoseTracker, yaw: float, pitch: float, roll: float = 0.0):
    """Drive one pose frame into the DUT and Python reference."""
    dut.yaw.value   = s8_to_verilog(angle_to_s8(yaw))
    dut.pitch.value = s8_to_verilog(angle_to_s8(pitch))
    dut.roll.value  = s8_to_verilog(angle_to_s8(roll))
    dut.valid.value = 1
    await RisingEdge(dut.clk)
    dut.valid.value = 0
    await RisingEdge(dut.clk)

    pose = ref.update(yaw=yaw, pitch=pitch, roll=roll)
    return pose


@cocotb.test()
async def test_in_zone_forward_gaze(dut):
    """Yaw=0, pitch=0, roll=0 → in_zone = 1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = HeadPoseTracker()
    pose = await drive_pose(dut, ref, yaw=0.0, pitch=0.0, roll=0.0)

    assert ref.attention_zone.in_zone(pose)
    assert int(dut.in_zone.value) == 1, f"Expected in_zone=1, got {dut.in_zone.value}"
    dut._log.info("in_zone_forward test passed")


@cocotb.test()
async def test_out_of_zone_yaw(dut):
    """Yaw = 45° (> 30° threshold) → in_zone = 0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = HeadPoseTracker()
    pose = await drive_pose(dut, ref, yaw=45.0, pitch=0.0, roll=0.0)

    assert not ref.attention_zone.in_zone(pose)
    assert int(dut.in_zone.value) == 0, f"Expected in_zone=0, got {dut.in_zone.value}"
    dut._log.info("out_of_zone_yaw test passed")


@cocotb.test()
async def test_out_of_zone_negative_yaw(dut):
    """Yaw = -45° → in_zone = 0 (absolute value check)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = HeadPoseTracker()
    pose = await drive_pose(dut, ref, yaw=-45.0, pitch=0.0, roll=0.0)

    assert int(dut.in_zone.value) == 0
    dut._log.info("out_of_zone_negative_yaw test passed")


@cocotb.test()
async def test_out_of_zone_pitch(dut):
    """Pitch = 25° (> 20° threshold) → in_zone = 0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = HeadPoseTracker()
    pose = await drive_pose(dut, ref, yaw=0.0, pitch=25.0, roll=0.0)

    assert int(dut.in_zone.value) == 0
    dut._log.info("out_of_zone_pitch test passed")


@cocotb.test()
async def test_yaw_boundary_in(dut):
    """Yaw = 30° (exactly at boundary) → in_zone = 1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = HeadPoseTracker()
    pose = await drive_pose(dut, ref, yaw=30.0, pitch=0.0, roll=0.0)

    assert ref.attention_zone.in_zone(pose)
    assert int(dut.in_zone.value) == 1
    dut._log.info("yaw_boundary_in test passed")


@cocotb.test()
async def test_yaw_boundary_out(dut):
    """Yaw = 31° (just outside boundary) → in_zone = 0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = HeadPoseTracker()
    pose = await drive_pose(dut, ref, yaw=31.0, pitch=0.0, roll=0.0)

    assert not ref.attention_zone.in_zone(pose)
    assert int(dut.in_zone.value) == 0
    dut._log.info("yaw_boundary_out test passed")


@cocotb.test()
async def test_distraction_count_all_out(dut):
    """15 consecutive out-of-zone frames → distracted_count = 15."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Reset initializes window to all-in-zone. We need to fill the 15-slot window.
    ref = HeadPoseTracker(window=15)
    for _ in range(15):
        await drive_pose(dut, ref, yaw=60.0, pitch=0.0, roll=0.0)

    assert int(dut.distracted_count.value) == 15, (
        f"Expected 15, got {dut.distracted_count.value}"
    )
    dut._log.info("distraction_count_all_out test passed")


@cocotb.test()
async def test_distraction_count_all_in(dut):
    """15 consecutive in-zone frames → distracted_count = 0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = HeadPoseTracker(window=15)
    for _ in range(15):
        await drive_pose(dut, ref, yaw=5.0, pitch=5.0, roll=5.0)

    assert int(dut.distracted_count.value) == 0, (
        f"Expected 0, got {dut.distracted_count.value}"
    )
    dut._log.info("distraction_count_all_in test passed")


@cocotb.test()
async def test_distraction_window_rolls(dut):
    """Fill with out-of-zone, then push in-zone; distracted_count drops to 0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = HeadPoseTracker(window=15)

    # Fill with distracted frames
    for _ in range(15):
        await drive_pose(dut, ref, yaw=60.0, pitch=0.0)
    assert int(dut.distracted_count.value) == 15

    # Replace all with in-zone frames
    for _ in range(15):
        await drive_pose(dut, ref, yaw=0.0, pitch=0.0)
    assert int(dut.distracted_count.value) == 0, (
        f"After rolling, distracted_count should be 0, got {dut.distracted_count.value}"
    )
    dut._log.info("distraction_window_rolls test passed")


@cocotb.test()
async def test_reference_match_sequence(dut):
    """Mixed sequence: verify DUT in_zone matches Python reference every frame."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = HeadPoseTracker(window=15)
    sequence = [
        (0, 0, 0),     # in
        (15, 10, 5),   # in
        (31, 0, 0),    # out (yaw > 30)
        (-35, 0, 0),   # out (|yaw| > 30)
        (0, 21, 0),    # out (pitch > 20)
        (0, -22, 0),   # out
        (29, 19, 19),  # in (all at threshold)
        (0, 0, 21),    # out (roll > 20)
        (5, -5, 10),   # in
        (30, 20, 20),  # in (all exactly at threshold)
    ]

    for frame_idx, (yaw, pitch, roll) in enumerate(sequence):
        pose = await drive_pose(dut, ref, float(yaw), float(pitch), float(roll))
        ref_in_zone = ref.attention_zone.in_zone(pose)
        dut_in_zone = int(dut.in_zone.value)
        assert dut_in_zone == int(ref_in_zone), (
            f"Frame {frame_idx} ({yaw},{pitch},{roll}): "
            f"DUT in_zone={dut_in_zone} REF in_zone={ref_in_zone}"
        )

    dut._log.info("reference_match_sequence passed")


@cocotb.test()
async def test_reset_clears_window(dut):
    """After reset, distracted_count returns to 0 (window reset to in-zone)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = HeadPoseTracker(window=15)
    for _ in range(15):
        await drive_pose(dut, ref, yaw=60.0, pitch=0.0)

    dut.rst_n.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    assert int(dut.distracted_count.value) == 0, (
        f"distracted_count should be 0 after reset, got {dut.distracted_count.value}"
    )
    dut._log.info("reset_clears_window test passed")
