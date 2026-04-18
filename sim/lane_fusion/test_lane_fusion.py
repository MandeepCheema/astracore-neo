"""
AstraCore Neo — Lane Fusion cocotb testbench

Confidence-weighted blend of camera + HD map lane estimates.
  w_cam = cam_conf >> 1  (0..127)
  w_map = 128 - w_cam    (1..128)
  fused = (cam*w_cam + map*w_map) >> 7

Stale fallback: cam stale → map-only; map stale → cam-only; both stale → hold.

Timing: 2-stage pipeline.  After driving cam_valid or map_valid, the result
is visible 2 clocks later (stage 1 registers multiplier outputs, stage 2
registers sums + derived quantities).  The fire_* helpers include the
extra await needed for NBA settlement.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


STALE_CYCLES = 500


def to_s32(v):
    v = int(v)
    return v if v < (1 << 31) else v - (1 << 32)


def ref_blend(cam_left, cam_right, cam_conf, map_left, map_right, cam_only=False, map_only=False):
    """Python reference model — matches RTL integer arithmetic exactly."""
    if cam_only:
        w_cam, w_map = 128, 0
    elif map_only:
        w_cam, w_map = 0, 128
    else:
        w_cam = cam_conf >> 1
        w_map = 128 - w_cam
    fused_left  = (cam_left  * w_cam + map_left  * w_map) >> 7
    fused_right = (cam_right * w_cam + map_right * w_map) >> 7
    width       = fused_right - fused_left
    center      = (fused_left + fused_right) >> 1
    return fused_left, fused_right, width, center


async def reset_dut(dut):
    dut.rst_n.value           = 0
    dut.cam_valid.value       = 0
    dut.cam_left_mm.value     = 0
    dut.cam_right_mm.value    = 0
    dut.cam_confidence.value  = 0
    dut.map_valid.value       = 0
    dut.map_left_mm.value     = 0
    dut.map_right_mm.value    = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def fire_cam(dut, left, right, conf):
    dut.cam_left_mm.value    = left  & 0xFFFFFFFF
    dut.cam_right_mm.value   = right & 0xFFFFFFFF
    dut.cam_confidence.value = conf & 0xFF
    dut.cam_valid.value      = 1
    await RisingEdge(dut.clk)   # EDGE A: input sampled
    dut.cam_valid.value      = 0
    await RisingEdge(dut.clk)   # EDGE B: stage 1 visible
    await RisingEdge(dut.clk)   # EDGE C: stage 2 visible (fused_* readable)


async def fire_map(dut, left, right):
    dut.map_left_mm.value  = left  & 0xFFFFFFFF
    dut.map_right_mm.value = right & 0xFFFFFFFF
    dut.map_valid.value    = 1
    await RisingEdge(dut.clk)
    dut.map_valid.value    = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


async def fire_both(dut, cam_left, cam_right, cam_conf, map_left, map_right):
    dut.cam_left_mm.value    = cam_left  & 0xFFFFFFFF
    dut.cam_right_mm.value   = cam_right & 0xFFFFFFFF
    dut.cam_confidence.value = cam_conf & 0xFF
    dut.cam_valid.value      = 1
    dut.map_left_mm.value    = map_left  & 0xFFFFFFFF
    dut.map_right_mm.value   = map_right & 0xFFFFFFFF
    dut.map_valid.value      = 1
    await RisingEdge(dut.clk)
    dut.cam_valid.value      = 0
    dut.map_valid.value      = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


# ===========================================================================
# Tests
# ===========================================================================

@cocotb.test()
async def test_initial_state(dut):
    """After reset: no valid output, source=00, no stale."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    assert int(dut.fused_valid.value)   == 0
    assert int(dut.fusion_source.value) == 0
    assert int(dut.sensor_stale.value)  == 0
    dut._log.info("initial_state passed")


@cocotb.test()
async def test_cam_only_first_pulse(dut):
    """First-ever camera pulse: map has no data → cam-only output."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_cam(dut, left=-1800, right=1800, conf=200)

    assert int(dut.fused_valid.value) == 1
    assert int(dut.fusion_source.value) == 0b10, \
        f"source should be cam-only (0b10), got {int(dut.fusion_source.value):02b}"
    # With map missing, w_cam forced to 128 → fused = cam
    assert to_s32(dut.fused_left_mm.value)  == -1800
    assert to_s32(dut.fused_right_mm.value) == 1800
    assert to_s32(dut.fused_lane_width_mm.value) == 3600
    assert to_s32(dut.fused_center_offset_mm.value) == 0
    dut._log.info("cam_only_first_pulse passed")


@cocotb.test()
async def test_map_only_first_pulse(dut):
    """First-ever map pulse: cam has no data → map-only output."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_map(dut, left=-1750, right=1750)

    assert int(dut.fused_valid.value) == 1
    assert int(dut.fusion_source.value) == 0b01, \
        f"source should be map-only (0b01), got {int(dut.fusion_source.value):02b}"
    assert to_s32(dut.fused_left_mm.value)  == -1750
    assert to_s32(dut.fused_right_mm.value) == 1750
    dut._log.info("map_only_first_pulse passed")


@cocotb.test()
async def test_blended_output(dut):
    """Both sources present: fused matches reference blend model."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Seed map first, then fire camera
    await fire_map(dut, left=-1800, right=1800)
    await fire_cam(dut, left=-1600, right=1900, conf=128)   # cam_conf 128 → w_cam=64, w_map=64

    assert int(dut.fusion_source.value) == 0b11, "source should be blended"
    ex_l, ex_r, ex_w, ex_c = ref_blend(-1600, 1900, 128, -1800, 1800)
    assert to_s32(dut.fused_left_mm.value)          == ex_l
    assert to_s32(dut.fused_right_mm.value)         == ex_r
    assert to_s32(dut.fused_lane_width_mm.value)    == ex_w
    assert to_s32(dut.fused_center_offset_mm.value) == ex_c
    dut._log.info(f"blended_output passed: L={ex_l} R={ex_r} W={ex_w} C={ex_c}")


@cocotb.test()
async def test_high_cam_conf_favors_camera(dut):
    """At maximum cam_conf (255): blend is ~cam dominant (127/128)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_map(dut, left=0, right=0)            # map lane at origin
    await fire_cam(dut, left=-2000, right=2000, conf=255)

    # w_cam=127, w_map=1 → fused_left = (-2000*127 + 0*1) >> 7 = -1984
    ex_l, ex_r, _, _ = ref_blend(-2000, 2000, 255, 0, 0)
    assert to_s32(dut.fused_left_mm.value)  == ex_l
    assert to_s32(dut.fused_right_mm.value) == ex_r
    assert abs(ex_l - (-2000)) <= 20, "should be close to cam value"
    dut._log.info(f"high_cam_conf_favors_camera passed: L={ex_l}")


@cocotb.test()
async def test_low_cam_conf_favors_map(dut):
    """At cam_conf=0: w_cam=0, blend = pure map."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_map(dut, left=-1900, right=1900)
    await fire_cam(dut, left=0, right=0, conf=0)

    ex_l, ex_r, _, _ = ref_blend(0, 0, 0, -1900, 1900)
    assert to_s32(dut.fused_left_mm.value)  == ex_l == -1900
    assert to_s32(dut.fused_right_mm.value) == ex_r ==  1900
    dut._log.info("low_cam_conf_favors_map passed")


@cocotb.test()
async def test_simultaneous_cam_and_map(dut):
    """Both pulses on same cycle: single fused_valid pulse with blended output."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_both(dut, cam_left=-1700, cam_right=1700, cam_conf=128,
                          map_left=-1900, map_right=1900)

    assert int(dut.fused_valid.value) == 1
    assert int(dut.fusion_source.value) == 0b11
    ex_l, ex_r, _, _ = ref_blend(-1700, 1700, 128, -1900, 1900)
    assert to_s32(dut.fused_left_mm.value)  == ex_l
    assert to_s32(dut.fused_right_mm.value) == ex_r
    dut._log.info("simultaneous_cam_and_map passed")


@cocotb.test()
async def test_cam_stale_falls_back_to_map(dut):
    """Cam goes stale while map continues → fusion_source becomes map-only."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_both(dut, -1700, 1700, 200, -1800, 1800)
    assert int(dut.fusion_source.value) == 0b11

    # Let camera go stale (no cam_valid for > STALE_CYCLES)
    for _ in range(STALE_CYCLES + 10):
        await RisingEdge(dut.clk)
    stale = int(dut.sensor_stale.value)
    assert (stale & 0x1) != 0, f"cam should be stale, got {stale:02b}"

    # Fire map-only; fusion should use map exclusively
    await fire_map(dut, -1850, 1850)
    assert int(dut.fusion_source.value) == 0b01, "should be map-only after cam stale"
    assert to_s32(dut.fused_left_mm.value)  == -1850
    assert to_s32(dut.fused_right_mm.value) == 1850
    dut._log.info("cam_stale_falls_back_to_map passed")


@cocotb.test()
async def test_map_stale_falls_back_to_cam(dut):
    """Map goes stale while cam continues → fusion_source becomes cam-only."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_both(dut, -1700, 1700, 150, -1800, 1800)

    for _ in range(STALE_CYCLES + 10):
        await RisingEdge(dut.clk)
    stale = int(dut.sensor_stale.value)
    assert (stale & 0x2) != 0, f"map should be stale, got {stale:02b}"

    await fire_cam(dut, -1650, 1750, 200)
    assert int(dut.fusion_source.value) == 0b10, "should be cam-only after map stale"
    assert to_s32(dut.fused_left_mm.value)  == -1650
    assert to_s32(dut.fused_right_mm.value) ==  1750
    dut._log.info("map_stale_falls_back_to_cam passed")


@cocotb.test()
async def test_stale_clears_on_new_data(dut):
    """Stale flag clears on next valid pulse from that source."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for _ in range(STALE_CYCLES + 10):
        await RisingEdge(dut.clk)
    assert int(dut.sensor_stale.value) == 0x3, "both should be stale"

    await fire_cam(dut, -1500, 1500, 200)
    stale = int(dut.sensor_stale.value)
    assert (stale & 0x1) == 0, "cam stale should clear"

    await fire_map(dut, -1700, 1700)
    stale = int(dut.sensor_stale.value)
    assert stale == 0x0, f"both should be clear, got {stale:02b}"
    dut._log.info("stale_clears_on_new_data passed")


@cocotb.test()
async def test_center_offset_computed(dut):
    """center_offset_mm = (left + right) >> 1 tracks ego offset from lane center."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Ego drifted 200mm right of lane center: left=-2000, right=1600
    await fire_cam(dut, left=-2000, right=1600, conf=255)

    # cam-only: ex_l=-2000, ex_r=1600, center = (-2000+1600)>>1 = -200
    ex_l, ex_r, ex_w, ex_c = ref_blend(-2000, 1600, 255, 0, 0, cam_only=True)
    assert to_s32(dut.fused_center_offset_mm.value) == ex_c == -200
    assert to_s32(dut.fused_lane_width_mm.value)    == ex_w == 3600
    dut._log.info(f"center_offset_computed passed: center={ex_c}, width={ex_w}")
