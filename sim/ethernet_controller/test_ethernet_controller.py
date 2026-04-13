"""
AstraCore Neo — Ethernet Controller cocotb testbench.

Validates the Ethernet frame receiver: length validation, EtherType extraction,
and MAC type classification.

Frame type encoding:
  2'd0  DATA (unknown)
  2'd1  IPv4  (EtherType 0x0800)
  2'd2  ARP   (EtherType 0x0806)
  2'd3  IPv6  (EtherType 0x86DD)

MAC type encoding:
  2'd0  UNICAST
  2'd1  MULTICAST
  2'd2  BROADCAST
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


async def reset_dut(dut):
    dut.rst_n.value    = 0
    dut.rx_valid.value = 0
    dut.rx_byte.value  = 0
    dut.rx_last.value  = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


def build_frame(dst_mac: bytes, src_mac: bytes, ethertype: int, payload: bytes) -> bytes:
    """Build a raw Ethernet II frame bytes."""
    et = bytes([(ethertype >> 8) & 0xFF, ethertype & 0xFF])
    return dst_mac + src_mac + et + payload


async def send_frame(dut, frame: bytes):
    """Drive a byte stream into the DUT. Returns (frame_ok, frame_err, ethertype, frame_type, mac_type)."""
    for i, byte_val in enumerate(frame):
        dut.rx_byte.value  = byte_val
        dut.rx_valid.value = 1
        dut.rx_last.value  = 1 if (i == len(frame) - 1) else 0
        await RisingEdge(dut.clk)

    dut.rx_valid.value = 0
    dut.rx_last.value  = 0
    await RisingEdge(dut.clk)   # outputs registered on cycle after rx_last

    return (
        int(dut.frame_ok.value),
        int(dut.frame_err.value),
        int(dut.ethertype.value),
        int(dut.frame_type.value),
        int(dut.mac_type.value),
        int(dut.byte_count.value),
    )


DST_UNICAST   = bytes([0x00, 0x11, 0x22, 0x33, 0x44, 0x55])
DST_MULTICAST = bytes([0x01, 0x00, 0x5E, 0x00, 0x00, 0x01])
DST_BROADCAST = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])
SRC_MAC       = bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF])
PAYLOAD_MIN   = bytes([0x00] * (64 - 14))    # minimum payload for 64-byte frame
PAYLOAD_DATA  = bytes([0xAB] * 100)


@cocotb.test()
async def test_valid_ipv4_frame(dut):
    """Valid 64-byte IPv4 Ethernet frame → frame_ok=1, frame_type=1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    frame = build_frame(DST_UNICAST, SRC_MAC, 0x0800, PAYLOAD_MIN)
    assert len(frame) == 64

    ok, err, et, ftype, mtype, bcount = await send_frame(dut, frame)

    assert ok    == 1,      f"frame_ok should be 1"
    assert err   == 0,      f"frame_err should be 0"
    assert et    == 0x0800, f"ethertype should be 0x0800, got 0x{et:04X}"
    assert ftype == 1,      f"frame_type should be IPv4(1), got {ftype}"
    assert mtype == 0,      f"mac_type should be UNICAST(0), got {mtype}"
    assert bcount == 64,    f"byte_count should be 64, got {bcount}"
    dut._log.info("valid_ipv4_frame test passed")


@cocotb.test()
async def test_valid_arp_frame(dut):
    """Valid ARP frame → frame_type=2, ethertype=0x0806."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    frame = build_frame(DST_BROADCAST, SRC_MAC, 0x0806, PAYLOAD_MIN)
    ok, err, et, ftype, mtype, _ = await send_frame(dut, frame)

    assert ok    == 1
    assert et    == 0x0806, f"ethertype should be 0x0806"
    assert ftype == 2,      f"frame_type should be ARP(2), got {ftype}"
    assert mtype == 2,      f"mac_type should be BROADCAST(2), got {mtype}"
    dut._log.info("valid_arp_frame test passed")


@cocotb.test()
async def test_valid_ipv6_frame(dut):
    """Valid IPv6 frame → frame_type=3, ethertype=0x86DD."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    frame = build_frame(DST_MULTICAST, SRC_MAC, 0x86DD, PAYLOAD_DATA)
    ok, err, et, ftype, mtype, _ = await send_frame(dut, frame)

    assert ok    == 1
    assert et    == 0x86DD, f"ethertype should be 0x86DD"
    assert ftype == 3,      f"frame_type should be IPv6(3)"
    assert mtype == 1,      f"mac_type should be MULTICAST(1), got {mtype}"
    dut._log.info("valid_ipv6_frame test passed")


@cocotb.test()
async def test_frame_too_short(dut):
    """Frame < 64 bytes → frame_err=1, frame_ok=0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # 40-byte frame (14 header + 26 payload) — too short
    frame = build_frame(DST_UNICAST, SRC_MAC, 0x0800, bytes([0] * 26))
    assert len(frame) == 40

    ok, err, et, ftype, mtype, bcount = await send_frame(dut, frame)

    assert ok  == 0, f"frame_ok should be 0 for short frame"
    assert err == 1, f"frame_err should be 1 for short frame"
    dut._log.info(f"frame_too_short test passed: byte_count={bcount}")


@cocotb.test()
async def test_frame_max_valid_length(dut):
    """Frame of exactly 1518 bytes → frame_ok=1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # 1518 = 14 header + 1504 payload
    frame = build_frame(DST_UNICAST, SRC_MAC, 0x0800, bytes([0xAB] * 1504))
    assert len(frame) == 1518

    ok, err, et, ftype, mtype, bcount = await send_frame(dut, frame)

    assert ok    == 1, f"1518-byte frame should be valid"
    assert bcount == 1518
    dut._log.info("frame_max_valid_length test passed")


@cocotb.test()
async def test_broadcast_mac_detection(dut):
    """Broadcast destination MAC → mac_type=2."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    frame = build_frame(DST_BROADCAST, SRC_MAC, 0x0800, PAYLOAD_MIN)
    ok, err, et, ftype, mtype, _ = await send_frame(dut, frame)

    assert mtype == 2, f"mac_type should be BROADCAST(2), got {mtype}"
    dut._log.info("broadcast_mac_detection test passed")


@cocotb.test()
async def test_multicast_mac_detection(dut):
    """Multicast destination MAC (bit 0 of byte 0 = 1) → mac_type=1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    frame = build_frame(DST_MULTICAST, SRC_MAC, 0x0800, PAYLOAD_MIN)
    ok, err, et, ftype, mtype, _ = await send_frame(dut, frame)

    assert mtype == 1, f"mac_type should be MULTICAST(1), got {mtype}"
    dut._log.info("multicast_mac_detection test passed")


@cocotb.test()
async def test_unknown_ethertype(dut):
    """Unknown EtherType → frame_type=0 (DATA)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    frame = build_frame(DST_UNICAST, SRC_MAC, 0x1234, PAYLOAD_MIN)
    ok, err, et, ftype, mtype, _ = await send_frame(dut, frame)

    assert ok    == 1
    assert ftype == 0, f"Unknown EtherType should give frame_type=0 (DATA)"
    dut._log.info("unknown_ethertype test passed")


@cocotb.test()
async def test_two_consecutive_frames(dut):
    """DUT should reset between frames and handle back-to-back frames correctly."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Frame 1: IPv4
    frame1 = build_frame(DST_UNICAST, SRC_MAC, 0x0800, PAYLOAD_MIN)
    ok1, _, et1, _, _, _ = await send_frame(dut, frame1)
    assert ok1 == 1 and et1 == 0x0800

    # Frame 2: ARP
    frame2 = build_frame(DST_BROADCAST, SRC_MAC, 0x0806, PAYLOAD_MIN)
    ok2, _, et2, ftype2, _, _ = await send_frame(dut, frame2)
    assert ok2 == 1 and et2 == 0x0806 and ftype2 == 2

    dut._log.info("two_consecutive_frames test passed")
