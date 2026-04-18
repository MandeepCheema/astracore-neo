"""
AstraCore Neo — Object Tracker cocotb testbench

Fixed 8-entry track table with bounding-box association gate:
  det_valid → combinatorial search → match or allocate or drop
  tick_valid → age every valid track, prune at MAX_AGE

Timing:
  Event pulses (det_matched / det_allocated / det_dropped) are registered,
  so after driving det_valid and one RisingEdge, the NBAs have not settled.
  Use two RisingEdges after driving to observe the pulse.

  num_active_tracks is a combinatorial popcount → visible immediately after
  the track NBAs settle (i.e. at the second RisingEdge).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


NUM_TRACKS = 8
GATE_MM    = 2000
MAX_AGE    = 10


def to_s32(v):
    v = int(v)
    return v if v < (1 << 31) else v - (1 << 32)


async def reset_dut(dut):
    dut.rst_n.value          = 0
    dut.det_valid.value      = 0
    dut.det_sensor_id.value  = 0
    dut.det_x_mm.value       = 0
    dut.det_y_mm.value       = 0
    dut.det_class_id.value   = 0
    dut.det_confidence.value = 0
    dut.tick_valid.value     = 0
    dut.query_idx.value      = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def fire_detection(dut, x, y, sensor_id=0, class_id=1, confidence=128):
    """
    Drive a 1-cycle detection pulse. After returning, the event flags
    (det_matched/allocated/dropped) and num_active_tracks reflect this detection.
    """
    dut.det_x_mm.value       = x & 0xFFFFFFFF
    dut.det_y_mm.value       = y & 0xFFFFFFFF
    dut.det_sensor_id.value  = sensor_id
    dut.det_class_id.value   = class_id
    dut.det_confidence.value = confidence
    dut.det_valid.value      = 1
    await RisingEdge(dut.clk)   # EDGE A: det_valid sampled, NBAs scheduled
    dut.det_valid.value      = 0
    await RisingEdge(dut.clk)   # EDGE B: EDGE A's NBAs visible — flags set
    await Timer(1, unit="ns")   # settle NBAs across simulators (iverilog VPI
                                # reads can lag the post-NBA value otherwise)


async def fire_tick(dut):
    """Drive a 1-cycle tick pulse."""
    dut.tick_valid.value = 1
    await RisingEdge(dut.clk)
    dut.tick_valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")   # see fire_detection()


async def query_track(dut, idx):
    """Read the track at the given index via the combinatorial query port."""
    dut.query_idx.value = idx
    await Timer(1, unit="ns")   # let combinatorial settle
    return {
        "valid":       int(dut.query_valid.value),
        "track_id":    int(dut.query_track_id.value),
        "x":           to_s32(dut.query_x_mm.value),
        "y":           to_s32(dut.query_y_mm.value),
        "vx":          to_s32(dut.query_vx_mm_per_update.value),
        "vy":          to_s32(dut.query_vy_mm_per_update.value),
        "age":         int(dut.query_age.value),
        "sensor_mask": int(dut.query_sensor_mask.value),
        "class_id":    int(dut.query_class_id.value),
        "confidence":  int(dut.query_confidence.value),
    }


# ===========================================================================
# Tests
# ===========================================================================

@cocotb.test()
async def test_initial_state(dut):
    """After reset: no tracks valid, num_active=0, no event pulses."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    assert int(dut.num_active_tracks.value) == 0
    assert int(dut.det_matched.value)       == 0
    assert int(dut.det_allocated.value)     == 0
    assert int(dut.det_dropped.value)       == 0
    for i in range(NUM_TRACKS):
        t = await query_track(dut, i)
        assert t["valid"] == 0, f"track {i} should be invalid"
    dut._log.info("initial_state passed")


@cocotb.test()
async def test_first_detection_allocates(dut):
    """First detection → allocated into track[0], id=1, state populated."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_detection(dut, x=5000, y=3000, sensor_id=1, class_id=7, confidence=200)

    assert int(dut.det_allocated.value) == 1, "det_allocated should pulse"
    assert int(dut.det_matched.value)   == 0
    assert int(dut.det_dropped.value)   == 0
    assert int(dut.num_active_tracks.value) == 1

    t = await query_track(dut, 0)
    assert t["valid"] == 1
    assert t["track_id"] == 1, f"track_id should be 1, got {t['track_id']}"
    assert t["x"] == 5000
    assert t["y"] == 3000
    assert t["sensor_mask"] == 0b0010
    assert t["class_id"] == 7
    assert t["confidence"] == 200
    assert t["age"] == 0
    dut._log.info(f"first_detection_allocates passed: {t}")


@cocotb.test()
async def test_same_position_matches(dut):
    """Second detection at same position → matched, not allocated."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_detection(dut, x=1000, y=2000, sensor_id=0)
    await fire_detection(dut, x=1000, y=2000, sensor_id=0)

    assert int(dut.det_matched.value)   == 1, "second det should match"
    assert int(dut.det_allocated.value) == 0
    assert int(dut.num_active_tracks.value) == 1
    dut._log.info("same_position_matches passed")


@cocotb.test()
async def test_far_detection_allocates_new(dut):
    """Detection far from existing track (> GATE_MM) → new allocation."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_detection(dut, x=0, y=0)
    await fire_detection(dut, x=10000, y=10000)   # far outside gate

    assert int(dut.det_allocated.value) == 1, "far detection should allocate new"
    assert int(dut.num_active_tracks.value) == 2

    t0 = await query_track(dut, 0)
    t1 = await query_track(dut, 1)
    assert t0["valid"] == 1 and t0["x"] == 0
    assert t1["valid"] == 1 and t1["x"] == 10000
    assert t0["track_id"] == 1 and t1["track_id"] == 2
    dut._log.info("far_detection_allocates_new passed")


@cocotb.test()
async def test_position_blend_on_update(dut):
    """Update blends position: (old + new) >> 1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_detection(dut, x=1000, y=2000)
    await fire_detection(dut, x=1200, y=2200)   # inside gate, slightly offset

    t = await query_track(dut, 0)
    # Expected: x = (1000+1200)>>1 = 1100, y = (2000+2200)>>1 = 2100
    assert t["x"] == 1100, f"x blend: got {t['x']}, expected 1100"
    assert t["y"] == 2100, f"y blend: got {t['y']}, expected 2100"
    dut._log.info(f"position_blend passed: ({t['x']},{t['y']})")


@cocotb.test()
async def test_track_id_monotonic(dut):
    """Each newly allocated track gets a unique monotonically increasing id."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    positions = [(0, 0), (5000, 5000), (10000, 10000), (-5000, -5000)]
    for i, (x, y) in enumerate(positions):
        await fire_detection(dut, x, y)
        t = await query_track(dut, i)
        assert t["track_id"] == i + 1, \
            f"track {i}: id={t['track_id']}, expected {i+1}"
    dut._log.info("track_id_monotonic passed")


@cocotb.test()
async def test_sensor_mask_accumulates(dut):
    """Multiple sensors updating same track accumulate into sensor_mask."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_detection(dut, x=1000, y=1000, sensor_id=0)   # mask = 0001
    await fire_detection(dut, x=1000, y=1000, sensor_id=2)   # mask = 0101
    await fire_detection(dut, x=1000, y=1000, sensor_id=3)   # mask = 1101

    t = await query_track(dut, 0)
    assert t["sensor_mask"] == 0b1101, \
        f"sensor_mask should be 0b1101, got 0b{t['sensor_mask']:04b}"
    dut._log.info(f"sensor_mask_accumulates passed: 0b{t['sensor_mask']:04b}")


@cocotb.test()
async def test_table_full_drops_detection(dut):
    """Fill all 8 slots then fire a 9th detection → det_dropped."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Fill all 8 slots with positions far apart
    for i in range(NUM_TRACKS):
        await fire_detection(dut, x=i * 10000, y=0)
    assert int(dut.num_active_tracks.value) == NUM_TRACKS

    # 9th detection at yet another far position → must drop
    await fire_detection(dut, x=999999, y=999999)
    assert int(dut.det_dropped.value)   == 1, "9th detection should drop"
    assert int(dut.det_allocated.value) == 0
    assert int(dut.num_active_tracks.value) == NUM_TRACKS
    dut._log.info("table_full_drops_detection passed")


@cocotb.test()
async def test_tick_advances_age(dut):
    """tick_valid increments age of every valid track."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_detection(dut, x=0, y=0)
    # age should start at 0
    t = await query_track(dut, 0)
    assert t["age"] == 0

    await fire_tick(dut)
    t = await query_track(dut, 0)
    assert t["age"] == 1, f"after 1 tick age should be 1, got {t['age']}"

    await fire_tick(dut)
    await fire_tick(dut)
    t = await query_track(dut, 0)
    assert t["age"] == 3, f"after 3 ticks age should be 3, got {t['age']}"
    dut._log.info("tick_advances_age passed")


@cocotb.test()
async def test_track_expires_at_max_age(dut):
    """After MAX_AGE ticks, the track is invalidated and num_active decrements."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_detection(dut, x=500, y=500)
    assert int(dut.num_active_tracks.value) == 1

    # Fire MAX_AGE ticks: ages 0→1→...→9→invalid (on the tick when age would be 10)
    for _ in range(MAX_AGE):
        await fire_tick(dut)

    t = await query_track(dut, 0)
    assert t["valid"] == 0, f"track should be expired, got valid={t['valid']}"
    assert int(dut.num_active_tracks.value) == 0
    dut._log.info("track_expires_at_max_age passed")


@cocotb.test()
async def test_velocity_estimate_first_update(dut):
    """
    On the first match after allocation, vx should equal
        (0 + (det_x - track_x)) >> 1
    where track_x is still the raw position from allocation.
    Fire allocate at x=0, then match at x=400 → vx = (0+400)>>1 = 200.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_detection(dut, x=0, y=0)
    t = await query_track(dut, 0)
    assert t["vx"] == 0
    assert t["x"]  == 0

    # Match with delta_x = 400
    await fire_detection(dut, x=400, y=0)
    t = await query_track(dut, 0)
    assert t["x"]  == 200, f"position blend: got {t['x']}, expected 200"
    assert t["vx"] == 200, f"vx: got {t['vx']}, expected 200"
    assert t["vy"] == 0,   f"vy should be 0, got {t['vy']}"
    dut._log.info(f"velocity_estimate_first_update passed: x={t['x']} vx={t['vx']}")


@cocotb.test()
async def test_velocity_handles_negative_motion(dut):
    """Object moving -y (signed negative delta) produces negative vy."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_detection(dut, x=0, y=500)
    await fire_detection(dut, x=0, y=-500)   # delta_y = -1000
    t = await query_track(dut, 0)
    # track_y blends from 500 → (500 + -500) >> 1 = 0
    # delta_y at match = -500 - 500 = -1000
    # vy = (0 + -1000) >> 1 = -500
    assert t["vy"] == -500, f"vy: got {t['vy']}, expected -500"
    assert t["vx"] == 0,    "vx should be 0 with no x motion"
    dut._log.info(f"velocity_handles_negative_motion passed: vy={t['vy']}")


@cocotb.test()
async def test_velocity_reset_on_allocation(dut):
    """New tracks start with vx=vy=0, not carrying over from prior slots."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Build up vx on track 0
    await fire_detection(dut, x=0, y=0)
    await fire_detection(dut, x=300, y=0)
    t0 = await query_track(dut, 0)
    assert t0["vx"] != 0, "track 0 should have non-zero vx"

    # Allocate track 1 far away — should start with vx=vy=0
    await fire_detection(dut, x=20000, y=20000)
    t1 = await query_track(dut, 1)
    assert t1["valid"] == 1
    assert t1["vx"] == 0 and t1["vy"] == 0, \
        f"new track should start with vx=vy=0, got {t1['vx']},{t1['vy']}"
    dut._log.info("velocity_reset_on_allocation passed")


@cocotb.test()
async def test_expired_slot_reused(dut):
    """After expiration, a new detection reuses the freed slot."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_detection(dut, x=0, y=0)        # track 0, id=1
    await fire_detection(dut, x=20000, y=0)    # track 1, id=2

    # Expire only track 0 by feeding ticks and keeping track 1 refreshed
    for _ in range(MAX_AGE):
        # Refresh track 1 between ticks so it stays young
        await fire_detection(dut, x=20000, y=0)
        await fire_tick(dut)

    t0 = await query_track(dut, 0)
    t1 = await query_track(dut, 1)
    # track 0 should have expired
    assert t0["valid"] == 0, f"track 0 should be expired, got {t0}"
    assert t1["valid"] == 1, f"track 1 should still be alive, got {t1}"

    # New detection should reuse slot 0 with a fresh id
    await fire_detection(dut, x=-5000, y=-5000)
    t0 = await query_track(dut, 0)
    assert t0["valid"] == 1
    assert t0["x"] == -5000
    # track_id should be new (not 1); next_id_ctr started at 1, incremented
    # each allocation: 1 (track0), 2 (track1), then 3 for this new one.
    assert t0["track_id"] == 3, f"new track in reused slot should have id=3, got {t0['track_id']}"
    dut._log.info(f"expired_slot_reused passed: new id={t0['track_id']}")
