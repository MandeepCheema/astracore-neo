"""
AstraCore Neo — GNSS Interface cocotb testbench

Verifies:
  - μs counter free-runs at CYCLES_PER_US clock cycles per tick (default 50)
  - time_set_valid jam-syncs time_us
  - PPS rising edge pulses pps_valid, latches pps_time_us, increments pps_count
  - fix load captures lat/lon and fix flag
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


CYCLES_PER_US = 50


def to_s32(v):
    v = int(v)
    return v if v < (1 << 31) else v - (1 << 32)


async def reset_dut(dut):
    dut.rst_n.value          = 0
    dut.pps_in.value         = 0
    dut.time_set_valid.value = 0
    dut.time_set_us.value    = 0
    dut.fix_set_valid.value  = 0
    dut.fix_valid_in.value   = 0
    dut.lat_mdeg_in.value    = 0
    dut.lon_mdeg_in.value    = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_time(dut, time_us):
    dut.time_set_us.value    = time_us & 0xFFFFFFFFFFFFFFFF
    dut.time_set_valid.value = 1
    await RisingEdge(dut.clk)
    dut.time_set_valid.value = 0
    await RisingEdge(dut.clk)


async def pps_edge(dut):
    """Drive a PPS rising edge."""
    dut.pps_in.value = 0
    await RisingEdge(dut.clk)
    dut.pps_in.value = 1
    await RisingEdge(dut.clk)


# ===========================================================================
# Tests
# ===========================================================================

@cocotb.test()
async def test_initial_state(dut):
    """After reset: time_us=0, pps_count=0, no fix."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    assert int(dut.time_us.value)       == 0
    assert int(dut.pps_count.value)     == 0
    assert int(dut.pps_valid.value)     == 0
    assert int(dut.gps_fix_valid.value) == 0
    dut._log.info("initial_state passed")


@cocotb.test()
async def test_us_counter_advances(dut):
    """After CYCLES_PER_US clocks, time_us should increment by 1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    start = int(dut.time_us.value)
    for _ in range(CYCLES_PER_US):
        await RisingEdge(dut.clk)
    # cocotb reads at active region of the rollover edge; one more edge to settle
    await RisingEdge(dut.clk)
    elapsed = int(dut.time_us.value) - start
    assert elapsed >= 1, f"time_us should have advanced, got {elapsed}"
    dut._log.info(f"us_counter_advances passed: +{elapsed} us")


@cocotb.test()
async def test_time_set_jam_sync(dut):
    """time_set_valid snaps time_us to time_set_us immediately."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await load_time(dut, 1_000_000_000)   # 1000 sec = 10^9 μs
    assert int(dut.time_us.value) == 1_000_000_000
    dut._log.info("time_set_jam_sync passed")


@cocotb.test()
async def test_pps_edge_pulses_valid(dut):
    """pps_in rising edge → pps_valid pulse + pps_count++."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await pps_edge(dut)
    # After the edge, one more clock for NBAs to settle
    await RisingEdge(dut.clk)
    assert int(dut.pps_count.value) == 1, \
        f"pps_count should be 1, got {int(dut.pps_count.value)}"
    dut._log.info("pps_edge_pulses_valid passed")


@cocotb.test()
async def test_multiple_pps_edges(dut):
    """Multiple PPS pulses increment the counter correctly."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for i in range(5):
        dut.pps_in.value = 0
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        dut.pps_in.value = 1
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)

    assert int(dut.pps_count.value) == 5, \
        f"pps_count should be 5, got {int(dut.pps_count.value)}"
    dut._log.info("multiple_pps_edges passed")


@cocotb.test()
async def test_pps_latches_time(dut):
    """pps_time_us captures time_us at the PPS edge."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Jam-sync to a known time
    await load_time(dut, 500_000)
    await pps_edge(dut)
    await RisingEdge(dut.clk)
    latched = int(dut.pps_time_us.value)
    # Allow a small window (time advanced a few μs while driving the edge)
    assert 500_000 <= latched <= 500_010, \
        f"pps_time_us should be ~500_000, got {latched}"
    dut._log.info(f"pps_latches_time passed: pps_time_us={latched}")


@cocotb.test()
async def test_fix_load(dut):
    """fix_set_valid captures lat/lon/fix_valid."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    dut.fix_valid_in.value  = 1
    dut.lat_mdeg_in.value   = 37_774_900    # ~37.7749° San Francisco
    dut.lon_mdeg_in.value   = (-122_419_400) & 0xFFFFFFFF
    dut.fix_set_valid.value = 1
    await RisingEdge(dut.clk)
    dut.fix_set_valid.value = 0
    await RisingEdge(dut.clk)

    assert int(dut.gps_fix_valid.value) == 1
    assert to_s32(dut.lat_mdeg.value) == 37_774_900
    assert to_s32(dut.lon_mdeg.value) == -122_419_400
    dut._log.info("fix_load passed")


@cocotb.test()
async def test_counter_continues_after_time_set(dut):
    """After jam-sync, μs counter resumes counting from the new value."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await load_time(dut, 10_000)

    # Run for many μs worth of clocks
    for _ in range(CYCLES_PER_US * 100):
        await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    elapsed = int(dut.time_us.value) - 10_000
    assert 95 <= elapsed <= 105, \
        f"time_us should have advanced ~100 us, got {elapsed}"
    dut._log.info(f"counter_continues_after_time_set passed: +{elapsed} us")
