"""
AstraCore Neo — LiDAR Interface cocotb testbench

Parses 24-byte packets from the ethernet_controller RX payload stream:
  bytes 0-1:   0xA5A5 magic
  bytes 2-5:   x_mm  (s32 BE)
  bytes 6-9:   y_mm
  bytes 10-13: z_mm
  bytes 14-15: length_mm
  bytes 16-17: width_mm
  bytes 18-19: height_mm
  byte  20:    class_id
  byte  21:    confidence
  bytes 22-23: timestamp_us_lo (u16)

rx_payload_last must fire on byte 23 for a valid commit.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


MAGIC = 0xA5A5
FIFO_DEPTH = 8


def to_s32(v):
    v = int(v)
    return v if v < (1 << 31) else v - (1 << 32)


def pack_s32_be(v):
    v &= 0xFFFFFFFF
    return [(v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF]


def pack_u16_be(v):
    v &= 0xFFFF
    return [(v >> 8) & 0xFF, v & 0xFF]


def build_packet(x, y, z, length, width, height, class_id, confidence, ts_lo):
    pkt = [(MAGIC >> 8) & 0xFF, MAGIC & 0xFF]
    pkt.extend(pack_s32_be(x))
    pkt.extend(pack_s32_be(y))
    pkt.extend(pack_s32_be(z))
    pkt.extend(pack_u16_be(length))
    pkt.extend(pack_u16_be(width))
    pkt.extend(pack_u16_be(height))
    pkt.append(class_id & 0xFF)
    pkt.append(confidence & 0xFF)
    pkt.extend(pack_u16_be(ts_lo))
    assert len(pkt) == 24
    return pkt


async def reset_dut(dut):
    dut.rst_n.value            = 0
    dut.rx_payload_valid.value = 0
    dut.rx_payload_byte.value  = 0
    dut.rx_payload_last.value  = 0
    dut.out_ready.value        = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def send_packet(dut, byte_list):
    """Feed bytes via rx_payload_valid, asserting rx_payload_last on the final byte."""
    n = len(byte_list)
    for i, b in enumerate(byte_list):
        dut.rx_payload_byte.value  = b
        dut.rx_payload_valid.value = 1
        dut.rx_payload_last.value  = 1 if (i == n - 1) else 0
        await RisingEdge(dut.clk)
    dut.rx_payload_valid.value = 0
    dut.rx_payload_last.value  = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


async def pop_one(dut):
    while int(dut.out_valid.value) == 0:
        await RisingEdge(dut.clk)
    captured = {
        "x":     to_s32(dut.out_x_mm.value),
        "y":     to_s32(dut.out_y_mm.value),
        "z":     to_s32(dut.out_z_mm.value),
        "len":   int(dut.out_length_mm.value),
        "wid":   int(dut.out_width_mm.value),
        "hei":   int(dut.out_height_mm.value),
        "class": int(dut.out_class_id.value),
        "conf":  int(dut.out_confidence.value),
        "ts_lo": int(dut.out_timestamp_us_lo.value),
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
async def test_single_packet_parsed(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    pkt = build_packet(
        x=10_000, y=-5000, z=500,
        length=4500, width=1800, height=1600,
        class_id=3, confidence=220, ts_lo=0x1234,
    )
    await send_packet(dut, pkt)

    assert int(dut.fifo_count.value) == 1
    det = await pop_one(dut)
    assert det == {
        "x": 10_000, "y": -5000, "z": 500,
        "len": 4500, "wid": 1800, "hei": 1600,
        "class": 3, "conf": 220, "ts_lo": 0x1234,
    }
    dut._log.info("single_packet_parsed passed")


@cocotb.test()
async def test_negative_coordinates(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    pkt = build_packet(-1, -100, -1_000_000, 0, 0, 0, 0, 0, 0)
    await send_packet(dut, pkt)

    det = await pop_one(dut)
    assert det["x"] == -1
    assert det["y"] == -100
    assert det["z"] == -1_000_000
    dut._log.info("negative_coordinates passed")


@cocotb.test()
async def test_bad_magic_rejected(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    pkt = build_packet(1, 2, 3, 0, 0, 0, 0, 0, 0)
    pkt[0] = 0x00   # corrupt first magic byte
    await send_packet(dut, pkt)

    assert int(dut.frame_count.value) == 0
    assert int(dut.error_count.value) == 1
    assert int(dut.fifo_empty.value)  == 1
    dut._log.info("bad_magic_rejected passed")


@cocotb.test()
async def test_short_packet_rejected(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    pkt = build_packet(1, 2, 3, 0, 0, 0, 0, 0, 0)[:20]   # truncated
    await send_packet(dut, pkt)

    assert int(dut.frame_count.value) == 0
    assert int(dut.error_count.value) == 1
    dut._log.info("short_packet_rejected passed")


@cocotb.test()
async def test_long_packet_rejected(dut):
    """Over-length packet → overrun handling, no garbage commit."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    pkt = build_packet(1, 2, 3, 0, 0, 0, 0, 0, 0) + [0xFF] * 6   # 30 bytes
    await send_packet(dut, pkt)

    assert int(dut.frame_count.value) == 0
    assert int(dut.error_count.value) == 1
    assert int(dut.fifo_empty.value)  == 1
    dut._log.info("long_packet_rejected passed")


@cocotb.test()
async def test_multiple_packets(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    objs = [
        (100,  0,  0,  1000, 500, 500, 1, 50, 0x1111),
        (2000, -1000,  200,  4000, 1800, 1500, 2, 180, 0x2222),
        (-500, 500, 100, 500, 500, 500, 5, 90, 0x3333),
    ]
    for o in objs:
        await send_packet(dut, build_packet(*o))

    assert int(dut.fifo_count.value)  == 3
    assert int(dut.frame_count.value) == 3

    for o in objs:
        det = await pop_one(dut)
        assert det["x"]    == o[0]
        assert det["y"]    == o[1]
        assert det["z"]    == o[2]
        assert det["len"]  == o[3]
        assert det["ts_lo"] == o[8]
    dut._log.info("multiple_packets passed")


@cocotb.test()
async def test_fifo_full_drop(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for i in range(FIFO_DEPTH):
        await send_packet(dut, build_packet(i, 0, 0, 0, 0, 0, 0, 0, i))

    assert int(dut.fifo_full.value)  == 1
    assert int(dut.fifo_count.value) == FIFO_DEPTH

    await send_packet(dut, build_packet(999, 0, 0, 0, 0, 0, 0, 0, 999))
    assert int(dut.total_dropped.value) == 1
    assert int(dut.fifo_count.value)    == FIFO_DEPTH
    dut._log.info("fifo_full_drop passed")
