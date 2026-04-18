"""
AstraCore Neo — Ultrasonic Interface cocotb testbench

Parses 28-byte frame: 0xAA + 24 distance bytes (12 × big-endian u16)
                      + 1 health byte + 1 XOR checksum + 0x55
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


SOF = 0xAA
EOF = 0x55


def build_frame(distances, health, *, bad_cksum=False, bad_eof=False):
    """Assemble a 29-byte frame. distances = list of 12 u16 values.
    health is a 12-bit mask (sent as 2 bytes big-endian, bits 15..12 ignored)."""
    assert len(distances) == 12
    data = []
    for d in distances:
        data.append((d >> 8) & 0xFF)
        data.append(d & 0xFF)
    data.append((health >> 8) & 0xFF)   # health MSB (bits 15..8, only 3..0 used)
    data.append(health & 0xFF)           # health LSB
    cksum = 0
    for b in data:
        cksum ^= b
    if bad_cksum:
        cksum ^= 0xFF
    frame = [SOF] + data + [cksum] + [EOF if not bad_eof else 0x00]
    return frame


def unpack_distances(vec_int):
    """Inverse of RTL packing: return list of 12 channel distances."""
    return [(vec_int >> (16 * c)) & 0xFFFF for c in range(12)]


async def reset_dut(dut):
    dut.rst_n.value    = 0
    dut.rx_valid.value = 0
    dut.rx_byte.value  = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def send_frame(dut, frame_bytes):
    """Feed bytes through rx_valid one per clock."""
    for b in frame_bytes:
        dut.rx_byte.value  = b
        dut.rx_valid.value = 1
        await RisingEdge(dut.clk)
    dut.rx_valid.value = 0
    await RisingEdge(dut.clk)   # let final NBA settle
    await RisingEdge(dut.clk)


# ===========================================================================
# Tests
# ===========================================================================

@cocotb.test()
async def test_initial_state(dut):
    """After reset: no frame_valid, counts zero."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    assert int(dut.frame_valid.value)    == 0
    assert int(dut.frame_count.value)    == 0
    assert int(dut.error_count.value)    == 0
    dut._log.info("initial_state passed")


@cocotb.test()
async def test_single_valid_frame(dut):
    """Well-formed frame → distances and health populated, count incremented."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    dists  = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    health = 0x0FFF  # all channels healthy
    frame  = build_frame(dists, health)

    await send_frame(dut, frame)

    unpacked = unpack_distances(int(dut.distance_mm_vec.value))
    assert unpacked == dists, f"distances mismatch: got {unpacked}"
    assert int(dut.sensor_health.value) == 0xFFF
    assert int(dut.frame_count.value) == 1
    assert int(dut.error_count.value) == 0
    dut._log.info(f"single_valid_frame passed: {unpacked}")


@cocotb.test()
async def test_channel_ordering(dut):
    """Distances pack into channel slots in the expected order."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Each channel = its index * 0x100 + 1
    dists = [(i * 0x100) + 1 for i in range(12)]
    frame = build_frame(dists, 0x0AAA)
    await send_frame(dut, frame)

    unpacked = unpack_distances(int(dut.distance_mm_vec.value))
    assert unpacked == dists
    dut._log.info("channel_ordering passed")


@cocotb.test()
async def test_health_bitmask(dut):
    """Health byte low 12 bits populate sensor_health."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    dists  = [1000] * 12
    health = 0b1010_0101_1100
    frame  = build_frame(dists, health)
    await send_frame(dut, frame)

    assert int(dut.sensor_health.value) == health
    dut._log.info(f"health_bitmask passed: 0x{health:03x}")


@cocotb.test()
async def test_checksum_mismatch_rejected(dut):
    """Bad checksum → error_count++, no frame_valid, no state change."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    frame = build_frame([0] * 12, 0, bad_cksum=True)
    await send_frame(dut, frame)

    assert int(dut.frame_count.value) == 0, "bad frame must not increment count"
    assert int(dut.error_count.value) == 1
    dut._log.info("checksum_mismatch_rejected passed")


@cocotb.test()
async def test_eof_mismatch_rejected(dut):
    """Bad end-of-frame byte → error_count++."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    frame = build_frame([0] * 12, 0, bad_eof=True)
    await send_frame(dut, frame)

    assert int(dut.frame_count.value) == 0
    assert int(dut.error_count.value) == 1
    dut._log.info("eof_mismatch_rejected passed")


@cocotb.test()
async def test_multiple_frames_back_to_back(dut):
    """Several frames in sequence all get parsed correctly."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    test_sets = [
        ([10 * i for i in range(12)], 0x0FFF),
        ([4000 - 50 * i for i in range(12)], 0x0AAA),
        ([0xBEEF] * 12, 0x0001),
    ]
    for dists, health in test_sets:
        await send_frame(dut, build_frame(dists, health))
        unpacked = unpack_distances(int(dut.distance_mm_vec.value))
        assert unpacked == dists, f"expected {dists}, got {unpacked}"
        assert int(dut.sensor_health.value) == (health & 0xFFF)

    assert int(dut.frame_count.value) == len(test_sets)
    assert int(dut.error_count.value) == 0
    dut._log.info("multiple_frames_back_to_back passed")


@cocotb.test()
async def test_garbage_before_sof_ignored(dut):
    """Stray bytes before SOF are silently dropped, frame still parses."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    dists = [0x1234] * 12
    frame = [0x00, 0xFF, 0x12, 0x34] + build_frame(dists, 0x0FFF)
    await send_frame(dut, frame)

    assert int(dut.frame_count.value) == 1
    unpacked = unpack_distances(int(dut.distance_mm_vec.value))
    assert unpacked == dists
    dut._log.info("garbage_before_sof_ignored passed")


@cocotb.test()
async def test_error_then_recovery(dut):
    """Bad frame followed by good frame → both counters advance properly."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await send_frame(dut, build_frame([1] * 12, 0, bad_cksum=True))
    assert int(dut.error_count.value) == 1
    assert int(dut.frame_count.value) == 0

    dists = [42 * i for i in range(12)]
    await send_frame(dut, build_frame(dists, 0x0FFF))
    assert int(dut.frame_count.value) == 1
    assert int(dut.error_count.value) == 1
    unpacked = unpack_distances(int(dut.distance_mm_vec.value))
    assert unpacked == dists
    dut._log.info("error_then_recovery passed")
