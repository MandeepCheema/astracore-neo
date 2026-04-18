"""
AstraCore Neo — MIPI CSI-2 RX cocotb testbench

Parses the byte stream post-D-PHY:
  Short packet: 4 bytes (DI, WC lo, WC hi, ECC); DT < 0x10
    DT 0x00 = Frame Start, 0x01 = Frame End, 0x02 = Line Start, 0x03 = Line End
  Long packet: 4-byte header + WC payload bytes + 2 CRC bytes
    DT 0x10..0x3F = image data; WC = payload byte count
  DI byte layout: {VC[1:0], DT[5:0]}
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


DT_FS = 0x00
DT_FE = 0x01
DT_LS = 0x02
DT_LE = 0x03
DT_RAW8 = 0x2A


def di(vc, dt):
    return ((vc & 0x3) << 6) | (dt & 0x3F)


def short_packet(vc, dt, wc=0):
    return [di(vc, dt), wc & 0xFF, (wc >> 8) & 0xFF, 0x00]   # ECC=0 dummy


def long_packet(vc, dt, payload):
    wc = len(payload)
    return [di(vc, dt), wc & 0xFF, (wc >> 8) & 0xFF, 0x00] + list(payload) + [0x00, 0x00]


async def reset_dut(dut):
    dut.rst_n.value      = 0
    dut.byte_valid.value = 0
    dut.byte_data.value  = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def send_bytes(dut, byte_list):
    """Feed bytes one per clock and capture any pixel_valid output."""
    captured_pixels = []
    captured_last   = []
    events = {"fs": 0, "fe": 0, "ls": 0, "le": 0}

    for b in byte_list:
        dut.byte_data.value  = b
        dut.byte_valid.value = 1
        await RisingEdge(dut.clk)
        # Sample outputs after rising edge — but the NBA hasn't settled yet.
        # Events and pixel stream are 1-cycle pulses registered from the
        # CURRENT byte, so they appear on the NEXT clock.
    dut.byte_valid.value = 0

    # Drain one additional clock to let the last byte's NBA propagate
    await RisingEdge(dut.clk)


async def drive_and_capture(dut, byte_list):
    """
    Drive bytes and continuously observe outputs on every clock.
    Returns lists of (cycle, pixel_byte), pulse_events.
    """
    pixels  = []
    events  = []
    n = len(byte_list)
    i = 0
    # Start with no byte_valid
    dut.byte_valid.value = 0
    # Drive bytes while also sampling outputs
    while i < n + 8:   # a few extra drain cycles
        if i < n:
            dut.byte_data.value  = byte_list[i]
            dut.byte_valid.value = 1
        else:
            dut.byte_valid.value = 0
        await RisingEdge(dut.clk)
        # After this edge the previous cycle's NBAs are visible
        if int(dut.pixel_valid.value) == 1:
            pixels.append((i, int(dut.pixel_byte.value), int(dut.pixel_last.value)))
        if int(dut.frame_start.value) == 1:
            events.append((i, "fs"))
        if int(dut.frame_end.value) == 1:
            events.append((i, "fe"))
        if int(dut.line_start.value) == 1:
            events.append((i, "ls"))
        if int(dut.line_end.value) == 1:
            events.append((i, "le"))
        i += 1
    return pixels, events


# ===========================================================================
# Tests
# ===========================================================================

@cocotb.test()
async def test_initial_state(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    assert int(dut.frame_start.value) == 0
    assert int(dut.frame_end.value)   == 0
    assert int(dut.pixel_valid.value) == 0
    assert int(dut.frame_count.value) == 0
    dut._log.info("initial_state passed")


@cocotb.test()
async def test_frame_start_short_packet(dut):
    """FS short packet produces frame_start pulse and increments frame_count."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    _, events = await drive_and_capture(dut, short_packet(0, DT_FS))
    fs_events = [e for e in events if e[1] == "fs"]
    assert len(fs_events) == 1, f"expected 1 FS pulse, got {len(fs_events)}"
    assert int(dut.frame_count.value) == 1
    dut._log.info("frame_start_short_packet passed")


@cocotb.test()
async def test_frame_end_short_packet(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    _, events = await drive_and_capture(dut, short_packet(0, DT_FE))
    fe_events = [e for e in events if e[1] == "fe"]
    assert len(fe_events) == 1
    dut._log.info("frame_end_short_packet passed")


@cocotb.test()
async def test_line_start_end(dut):
    """LS and LE short packets fire the right pulses."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    bytes_list = short_packet(0, DT_LS) + short_packet(0, DT_LE)
    _, events = await drive_and_capture(dut, bytes_list)
    assert [e[1] for e in events] == ["ls", "le"], f"got {[e[1] for e in events]}"
    assert int(dut.line_count.value) == 1
    dut._log.info("line_start_end passed")


@cocotb.test()
async def test_long_packet_payload(dut):
    """Long packet streams WC payload bytes out via pixel_valid/pixel_byte."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    payload = [0x11, 0x22, 0x33, 0x44, 0x55]
    pixels, _ = await drive_and_capture(dut, long_packet(0, DT_RAW8, payload))

    pixel_vals = [p[1] for p in pixels]
    assert pixel_vals == payload, f"payload mismatch: got {pixel_vals}"

    # The last byte should have pixel_last=1
    assert pixels[-1][2] == 1, "pixel_last should fire on final payload byte"
    # Earlier bytes should have pixel_last=0
    for p in pixels[:-1]:
        assert p[2] == 0
    dut._log.info(f"long_packet_payload passed: {len(pixel_vals)} bytes")


@cocotb.test()
async def test_latched_header_fields(dut):
    """last_data_type / last_word_count / last_virtual_channel reflect header."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    payload = [0xAA] * 8
    await drive_and_capture(dut, long_packet(vc=2, dt=DT_RAW8, payload=payload))

    assert int(dut.last_data_type.value)       == DT_RAW8
    assert int(dut.last_word_count.value)      == len(payload)
    assert int(dut.last_virtual_channel.value) == 2
    dut._log.info("latched_header_fields passed")


@cocotb.test()
async def test_frame_sequence(dut):
    """Full FS → LS → <data> → LE → FE sequence tracks correctly."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    payload = [i & 0xFF for i in range(16)]
    stream  = short_packet(0, DT_FS) \
            + short_packet(0, DT_LS) \
            + long_packet(0, DT_RAW8, payload) \
            + short_packet(0, DT_LE) \
            + short_packet(0, DT_FE)

    pixels, events = await drive_and_capture(dut, stream)
    event_seq = [e[1] for e in events]
    assert event_seq == ["fs", "ls", "le", "fe"], f"event order: {event_seq}"

    pixel_vals = [p[1] for p in pixels]
    assert pixel_vals == payload

    assert int(dut.frame_count.value) == 1
    assert int(dut.line_count.value)  == 1
    dut._log.info("frame_sequence passed")


@cocotb.test()
async def test_multiple_frames(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    stream = []
    for _ in range(3):
        stream += short_packet(0, DT_FS)
        stream += long_packet(0, DT_RAW8, [0xFF] * 4)
        stream += short_packet(0, DT_FE)

    await drive_and_capture(dut, stream)

    assert int(dut.frame_count.value) == 3
    dut._log.info("multiple_frames passed")
