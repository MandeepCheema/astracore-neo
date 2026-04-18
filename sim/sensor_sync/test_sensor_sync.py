"""
AstraCore Neo — Sensor Sync cocotb testbench

Tests fusion window lifecycle, timestamp alignment, stale watchdog.

Default parameters (simulation-friendly, set in RTL):
  WINDOW_US     = 50   — half-width tolerance in μs
  WINDOW_CYCLES = 100  — window timeout (clock cycles)
  STALE_CYCLES  = 200  — stale threshold (clock cycles)

Tests use timestamps in range 1000–1200 μs.  In-window = |delta| ≤ 50.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


# RTL parameter defaults (match sensor_sync.v)
WINDOW_US     = 50
WINDOW_CYCLES = 100
STALE_CYCLES  = 200


async def reset_dut(dut):
    dut.rst_n.value       = 0
    dut.sensor_valid.value = 0
    dut.s0_time_us.value  = 0
    dut.s1_time_us.value  = 0
    dut.s2_time_us.value  = 0
    dut.s3_time_us.value  = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def fire_sensor(dut, sensor_idx: int, timestamp_us: int):
    """Fire sensor_idx for 1 cycle with given timestamp."""
    times = [0, 0, 0, 0]
    times[sensor_idx] = timestamp_us
    dut.s0_time_us.value  = times[0]
    dut.s1_time_us.value  = times[1]
    dut.s2_time_us.value  = times[2]
    dut.s3_time_us.value  = times[3]
    dut.sensor_valid.value = (1 << sensor_idx)
    await RisingEdge(dut.clk)
    dut.sensor_valid.value = 0
    await RisingEdge(dut.clk)   # extra cycle so NBA settles


async def fire_all_sensors(dut, t0, t1, t2, t3):
    """Fire all 4 sensors simultaneously with given timestamps."""
    dut.s0_time_us.value   = t0
    dut.s1_time_us.value   = t1
    dut.s2_time_us.value   = t2
    dut.s3_time_us.value   = t3
    dut.sensor_valid.value = 0xF
    await RisingEdge(dut.clk)
    dut.sensor_valid.value = 0
    await RisingEdge(dut.clk)


# ===========================================================================
# Tests
# ===========================================================================

@cocotb.test()
async def test_initial_state(dut):
    """After reset: window_open=0, sensors_ready=0, window_release=0, no stale."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    assert int(dut.window_open.value)    == 0, "window_open should be 0 after reset"
    assert int(dut.sensors_ready.value)  == 0, "sensors_ready should be 0 after reset"
    assert int(dut.window_release.value) == 0, "window_release should be 0 after reset"
    assert int(dut.sensor_stale.value)   == 0, "sensor_stale should be 0 after reset"
    dut._log.info("initial_state passed")


@cocotb.test()
async def test_window_opens_on_first_detection(dut):
    """First sensor_valid fires → window_open=1, window_center=anchor timestamp."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    anchor = 1000

    # Fire sensor 0 at t=1000 μs
    dut.s0_time_us.value   = anchor
    dut.sensor_valid.value = 0x1
    await RisingEdge(dut.clk)   # EDGE A: window_open scheduled, window_center=1000
    dut.sensor_valid.value = 0
    # Read AFTER EDGE A's NBAs settle — need one more clock
    await RisingEdge(dut.clk)   # EDGE B: EDGE A's result visible

    assert int(dut.window_open.value)   == 1,      "window should be open"
    assert int(dut.window_center.value) == anchor, \
        f"window_center should be {anchor}, got {int(dut.window_center.value)}"
    assert int(dut.sensors_ready.value) == 0x1,    "sensor 0 should be marked ready"
    dut._log.info("window_opens_on_first_detection passed")


@cocotb.test()
async def test_all_sensors_aligned_triggers_release(dut):
    """All 4 sensors within window half-width → window_release pulses."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    base = 1000  # anchor timestamp
    # Timestamps within ±50 μs: 1000, 1010, 990, 1005
    await fire_sensor(dut, 0, base)       # opens window, anchor=1000, ready=0x1
    await fire_sensor(dut, 1, base + 10)  # in window (delta=10 ≤ 50), ready=0x3
    await fire_sensor(dut, 2, base - 10)  # in window (delta=-10), ready=0x7
    await fire_sensor(dut, 3, base + 5)   # in window (delta=5), ready=0xF → close!

    # fire_sensor includes 2 extra clocks. After sensor 3 fires and extras settle,
    # window_release should have pulsed within those cycles.
    # Wait one more clock to ensure window_release NBA is visible.
    await RisingEdge(dut.clk)

    # window should now be closed (opened + all 4 sensors = release)
    assert int(dut.window_open.value)    == 0, "window should be closed after all sensors ready"
    assert int(dut.sensors_ready.value)  == 0, "sensors_ready should be cleared after close"
    dut._log.info("all_sensors_aligned_triggers_release passed")


@cocotb.test()
async def test_window_release_pulse_captured(dut):
    """window_release pulses exactly once when all sensors align."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    base = 2000
    # All 4 sensors fire simultaneously with same timestamp
    dut.s0_time_us.value   = base
    dut.s1_time_us.value   = base
    dut.s2_time_us.value   = base
    dut.s3_time_us.value   = base
    dut.sensor_valid.value = 0xF   # all 4 at once
    await RisingEdge(dut.clk)      # EDGE A: window opens, sensors_ready=0xF scheduled
    dut.sensor_valid.value = 0
    # After EDGE A NBA: window_open=1, sensors_ready=0xF
    # EDGE B: all_ready=1 → do_close=1 → window_release scheduled
    await RisingEdge(dut.clk)      # EDGE B: close fires
    # EDGE B's NBA: window_release=1, window_open=0
    # Reading at EDGE C:
    await RisingEdge(dut.clk)      # EDGE C: read EDGE B's result
    assert int(dut.window_release.value) == 1, \
        f"window_release should be 1 at EDGE C, got {int(dut.window_release.value)}"

    # Should de-assert at EDGE D
    await RisingEdge(dut.clk)      # EDGE D: default de-assert
    assert int(dut.window_release.value) == 0, "window_release should de-assert"
    dut._log.info("window_release_pulse_captured passed")


@cocotb.test()
async def test_out_of_window_detection_ignored(dut):
    """Detection with |delta| > WINDOW_US is not added to sensors_ready."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    base = 500
    await fire_sensor(dut, 0, base)          # opens window, anchor=500, ready=0x1
    await fire_sensor(dut, 1, base + 60)     # out of window (delta=60 > 50) → ignored
    await RisingEdge(dut.clk)               # settle

    # sensors_ready should still only have sensor 0
    assert int(dut.sensors_ready.value) == 0x1, \
        f"sensors_ready should be 0x1 (only sensor 0), got 0x{int(dut.sensors_ready.value):X}"
    assert int(dut.window_open.value) == 1, "window should still be open"
    dut._log.info("out_of_window_detection_ignored passed")


@cocotb.test()
async def test_window_timeout_triggers_release(dut):
    """If not all sensors align within WINDOW_CYCLES, window closes via timeout."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    base = 3000
    await fire_sensor(dut, 0, base)     # opens window with only sensor 0

    # Wait for the window to time out (WINDOW_CYCLES = 100, fire_sensor adds 2 clocks)
    # We already consumed 2 clocks in fire_sensor; wait WINDOW_CYCLES more
    release_seen = False
    for _ in range(WINDOW_CYCLES + 10):  # generous margin
        await RisingEdge(dut.clk)
        if int(dut.window_release.value) == 1:
            release_seen = True
            break

    assert release_seen, "window_release should have fired after timeout"
    assert int(dut.window_open.value) == 0, "window should be closed after timeout"
    dut._log.info("window_timeout_triggers_release passed")


@cocotb.test()
async def test_sequential_windows(dut):
    """After a window closes, the next sensor detection opens a new window."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Window 1: all 4 sensors at t=100
    dut.s0_time_us.value   = 100
    dut.s1_time_us.value   = 100
    dut.s2_time_us.value   = 100
    dut.s3_time_us.value   = 100
    dut.sensor_valid.value = 0xF
    await RisingEdge(dut.clk)
    dut.sensor_valid.value = 0
    await RisingEdge(dut.clk)   # window opens with sensors_ready=0xF
    await RisingEdge(dut.clk)   # all_ready → close fires
    await RisingEdge(dut.clk)   # window now closed, window_release pulsed

    # Window 2: new detection opens a fresh window
    dut.s0_time_us.value   = 500
    dut.sensor_valid.value = 0x1
    await RisingEdge(dut.clk)   # opens window 2
    dut.sensor_valid.value = 0
    await RisingEdge(dut.clk)   # settle

    assert int(dut.window_open.value)   == 1,   "new window should be open"
    assert int(dut.window_center.value) == 500, \
        f"window_center should be 500, got {int(dut.window_center.value)}"
    assert int(dut.sensors_ready.value) == 0x1, "only sensor 0 ready in new window"
    dut._log.info("sequential_windows passed")


@cocotb.test()
async def test_stale_sensor_asserts(dut):
    """Sensor silent for STALE_CYCLES cycles → sensor_stale bit asserts."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Wait STALE_CYCLES + a few extra for stale to assert
    for _ in range(STALE_CYCLES + 20):
        await RisingEdge(dut.clk)

    # All sensors should be stale
    stale = int(dut.sensor_stale.value)
    assert stale == 0xF, \
        f"All 4 sensors should be stale after {STALE_CYCLES} cycles, got 0x{stale:X}"
    dut._log.info("stale_sensor_asserts passed")


@cocotb.test()
async def test_stale_clears_on_detection(dut):
    """Stale flag clears when the sensor provides a new detection."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Force sensors 0, 1, 2, 3 stale
    for _ in range(STALE_CYCLES + 10):
        await RisingEdge(dut.clk)
    assert int(dut.sensor_stale.value) == 0xF, "All should be stale"

    # Fire sensor 2 — its stale should clear
    dut.s2_time_us.value   = 9999
    dut.sensor_valid.value = 0x4   # sensor 2
    await RisingEdge(dut.clk)
    dut.sensor_valid.value = 0
    await RisingEdge(dut.clk)   # extra cycle for NBA to settle

    stale = int(dut.sensor_stale.value)
    assert (stale & 0x4) == 0, f"sensor 2 stale should be cleared, got 0x{stale:X}"
    assert (stale & 0xB) == 0xB, f"sensors 0,1,3 should still be stale, got 0x{stale:X}"
    dut._log.info(f"stale_clears_on_detection passed: stale=0x{stale:X}")


@cocotb.test()
async def test_negative_timestamp_delta_in_window(dut):
    """Sensor with timestamp EARLIER than anchor (negative delta) is still in-window."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    anchor = 1000
    await fire_sensor(dut, 0, anchor)          # opens, anchor=1000, ready=0x1
    await fire_sensor(dut, 1, anchor - 40)     # delta=-40, |delta|=40 ≤ 50 → in window

    await RisingEdge(dut.clk)  # settle

    ready = int(dut.sensors_ready.value)
    assert (ready & 0x2) != 0, \
        f"sensor 1 (delta=-40) should be in window, sensors_ready=0x{ready:X}"
    dut._log.info("negative_timestamp_delta_in_window passed")
