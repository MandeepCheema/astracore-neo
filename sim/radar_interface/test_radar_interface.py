"""
AstraCore Neo — Radar Interface cocotb testbench

13-byte radar_object_t big-endian frames over SPI byte stream → FIFO → AXI-S.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


FIFO_DEPTH = 16


def to_s16(v):
    v = int(v)
    return v if v < (1 << 15) else v - (1 << 16)


def pack_s16(v):
    v &= 0xFFFF
    return ((v >> 8) & 0xFF, v & 0xFF)


def pack_u32_be(v):
    return [(v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF]


def build_radar_frame(range_cm, vel_cms, az_mdeg, rcs, conf, ts_us):
    frame = []
    for f in (range_cm, vel_cms, az_mdeg, rcs):
        msb, lsb = pack_s16(f)
        frame.extend([msb, lsb])
    frame.append(conf & 0xFF)
    frame.extend(pack_u32_be(ts_us & 0xFFFFFFFF))
    return frame


async def reset_dut(dut):
    dut.rst_n.value          = 0
    dut.spi_byte_valid.value = 0
    dut.spi_byte.value       = 0
    dut.spi_frame_end.value  = 0
    dut.out_ready.value      = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def send_frame(dut, bytes_list):
    for b in bytes_list:
        dut.spi_byte.value       = b
        dut.spi_byte_valid.value = 1
        await RisingEdge(dut.clk)
    dut.spi_byte_valid.value = 0
    await RisingEdge(dut.clk)
    dut.spi_frame_end.value = 1
    await RisingEdge(dut.clk)
    dut.spi_frame_end.value = 0
    await RisingEdge(dut.clk)


async def pop_one(dut):
    while int(dut.out_valid.value) == 0:
        await RisingEdge(dut.clk)
    captured = {
        "range":  to_s16(dut.out_range_cm.value),
        "vel":    to_s16(dut.out_velocity_cms.value),
        "az":     to_s16(dut.out_azimuth_mdeg.value),
        "rcs":    int(dut.out_rcs_dbsm.value),
        "conf":   int(dut.out_confidence.value),
        "ts":     int(dut.out_timestamp_us.value),
    }
    dut.out_ready.value = 1
    await RisingEdge(dut.clk)
    dut.out_ready.value = 0
    await RisingEdge(dut.clk)
    return captured


# ===========================================================================
# Tests
# ===========================================================================

@cocotb.test()
async def test_initial_state(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    assert int(dut.fifo_empty.value) == 1
    assert int(dut.out_valid.value)  == 0
    assert int(dut.frame_count.value) == 0
    dut._log.info("initial_state passed")


@cocotb.test()
async def test_single_object_parsed(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await send_frame(dut, build_radar_frame(
        range_cm=500, vel_cms=-2000, az_mdeg=1500,
        rcs=0x0321, conf=180, ts_us=0x12345678))

    assert int(dut.fifo_count.value) == 1
    det = await pop_one(dut)
    assert det == {"range": 500, "vel": -2000, "az": 1500,
                   "rcs": 0x0321, "conf": 180, "ts": 0x12345678}
    dut._log.info("single_object_parsed passed")


@cocotb.test()
async def test_multiple_objects_fifo_order(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    objs = [
        (100,   -500,  0,     0x01, 200, 1000),
        (1500, -3000,  500,   0x05, 150, 2000),
        (2000, -1500, -1200,  0x10, 220, 3000),
    ]
    for o in objs:
        await send_frame(dut, build_radar_frame(*o))

    assert int(dut.fifo_count.value) == 3
    assert int(dut.frame_count.value) == 3

    for o in objs:
        det = await pop_one(dut)
        assert det["range"] == o[0] and det["vel"] == o[1] and det["az"] == o[2]
        assert det["rcs"] == o[3] and det["conf"] == o[4] and det["ts"] == o[5]
    dut._log.info("multiple_objects_fifo_order passed")


@cocotb.test()
async def test_short_frame_error(dut):
    """Frame with wrong byte count → error_count++, no enqueue."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Send only 10 bytes then frame_end
    short = build_radar_frame(0, 0, 0, 0, 0, 0)[:10]
    await send_frame(dut, short)
    assert int(dut.frame_count.value) == 0
    assert int(dut.error_count.value) == 1
    assert int(dut.fifo_empty.value) == 1
    dut._log.info("short_frame_error passed")


@cocotb.test()
async def test_fifo_full_drop(dut):
    """Fill FIFO to depth; next valid frame goes to total_dropped."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for i in range(FIFO_DEPTH):
        await send_frame(dut, build_radar_frame(i, 0, 0, 0, 0, i))
    assert int(dut.fifo_full.value)  == 1
    assert int(dut.fifo_count.value) == FIFO_DEPTH

    await send_frame(dut, build_radar_frame(9999, 0, 0, 0, 0, 9999))
    assert int(dut.total_dropped.value) == 1
    assert int(dut.fifo_count.value)    == FIFO_DEPTH
    dut._log.info("fifo_full_drop passed")


@cocotb.test()
async def test_fill_drain_refill(dut):
    """Fill + drain + refill cycle — pointers wrap correctly."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for i in range(FIFO_DEPTH):
        await send_frame(dut, build_radar_frame(i, 0, 0, 0, 0, i))
    for i in range(FIFO_DEPTH):
        det = await pop_one(dut)
        assert det["range"] == i and det["ts"] == i
    assert int(dut.fifo_empty.value) == 1

    for i in range(FIFO_DEPTH):
        await send_frame(dut, build_radar_frame(100 + i, 0, 0, 0, 0, 100 + i))
    for i in range(FIFO_DEPTH):
        det = await pop_one(dut)
        assert det["range"] == 100 + i
    dut._log.info("fill_drain_refill passed")
