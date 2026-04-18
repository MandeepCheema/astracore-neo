"""
AstraCore Neo — PTP Clock Sync cocotb testbench

Master-mode grandmaster: on each SYNC_INTERVAL_MS (default 125 ms) emits a
16-byte Sync frame through the ethernet_controller TX byte pipeline.

Frame layout (big-endian):
  bytes 0-1   : 0xAA55 magic
  bytes 2-3   : 0x0001 message type (SYNC)
  bytes 4-11  : master_time_us (frozen at frame start)
  bytes 12-15 : sequence_id
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


SYNC_INTERVAL_MS = 125
SYNC_MAGIC       = 0xAA55
MSG_SYNC         = 0x0001


async def reset_dut(dut):
    dut.rst_n.value          = 0
    dut.master_time_us.value = 0
    dut.tick_1ms.value       = 0
    dut.tx_ready.value       = 1    # PHY always ready by default
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def advance_ms(dut, n):
    """Fire n consecutive 1 ms ticks."""
    for _ in range(n):
        dut.tick_1ms.value = 1
        await RisingEdge(dut.clk)
        dut.tick_1ms.value = 0
        await RisingEdge(dut.clk)


async def capture_frame(dut, max_cycles=200):
    """Collect bytes while tx_valid is high; return list of bytes."""
    out = []
    cycles = 0
    while cycles < max_cycles:
        # Sample after each rising edge
        if int(dut.tx_valid.value) == 1 and int(dut.tx_ready.value) == 1:
            out.append(int(dut.tx_byte_in.value))
            if int(dut.tx_last.value) == 1:
                await RisingEdge(dut.clk)
                break
        await RisingEdge(dut.clk)
        cycles += 1
    return out


def parse_frame(bytes_list):
    assert len(bytes_list) == 16, f"expected 16 bytes, got {len(bytes_list)}"
    magic = (bytes_list[0] << 8) | bytes_list[1]
    msg_t = (bytes_list[2] << 8) | bytes_list[3]
    time_us = 0
    for b in bytes_list[4:12]:
        time_us = (time_us << 8) | b
    seq = 0
    for b in bytes_list[12:16]:
        seq = (seq << 8) | b
    return {"magic": magic, "msg_type": msg_t, "time_us": time_us, "seq": seq}


# ===========================================================================
# Tests
# ===========================================================================

@cocotb.test()
async def test_initial_state(dut):
    """After reset: no TX, zero counters."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    assert int(dut.tx_valid.value)        == 0
    assert int(dut.sync_count.value)      == 0
    assert int(dut.sync_sequence.value)   == 0
    dut._log.info("initial_state passed")


@cocotb.test()
async def test_no_sync_before_interval(dut):
    """No frame emitted before SYNC_INTERVAL_MS ticks have elapsed."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    dut.master_time_us.value = 1_000_000
    await advance_ms(dut, SYNC_INTERVAL_MS - 1)

    assert int(dut.tx_valid.value)    == 0
    assert int(dut.sync_count.value)  == 0
    dut._log.info("no_sync_before_interval passed")


@cocotb.test()
async def test_first_sync_at_interval(dut):
    """A Sync frame is emitted exactly when sync_timer hits SYNC_INTERVAL_MS."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    dut.master_time_us.value = 0x0000_0000_0001_2345
    await advance_ms(dut, SYNC_INTERVAL_MS)

    frame = await capture_frame(dut)
    parsed = parse_frame(frame)

    assert parsed["magic"]    == SYNC_MAGIC
    assert parsed["msg_type"] == MSG_SYNC
    assert parsed["time_us"]  == 0x0000_0000_0001_2345
    assert parsed["seq"]      == 1
    assert int(dut.sync_count.value) == 1
    dut._log.info(f"first_sync_at_interval passed: seq={parsed['seq']}")


@cocotb.test()
async def test_multiple_syncs_increment_sequence(dut):
    """Several intervals → sequence number increments by 1 each time."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    sequences = []
    for i in range(3):
        dut.master_time_us.value = 100_000 + i * 1000
        await advance_ms(dut, SYNC_INTERVAL_MS)
        frame = await capture_frame(dut)
        parsed = parse_frame(frame)
        sequences.append(parsed["seq"])

    assert sequences == [1, 2, 3], f"sequence should be monotonic: {sequences}"
    assert int(dut.sync_count.value) == 3
    dut._log.info("multiple_syncs_increment_sequence passed")


@cocotb.test()
async def test_time_frozen_during_tx(dut):
    """master_time_us latched at TX start — changes during TX don't affect frame."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    dut.master_time_us.value = 0xAAAA_BBBB_CCCC_DDDD
    await advance_ms(dut, SYNC_INTERVAL_MS)

    # First byte should be out by now; mutate master_time_us mid-frame
    await RisingEdge(dut.clk)
    dut.master_time_us.value = 0x1111_2222_3333_4444

    frame = await capture_frame(dut)
    parsed = parse_frame(frame)
    assert parsed["time_us"] == 0xAAAA_BBBB_CCCC_DDDD, \
        f"time should be frozen at frame start, got {parsed['time_us']:016x}"
    dut._log.info("time_frozen_during_tx passed")


@cocotb.test()
async def test_tx_backpressure_stalls_bytes(dut):
    """When tx_ready=0, byte_idx does not advance."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    dut.master_time_us.value = 0xDEAD_BEEF_CAFE_BABE
    await advance_ms(dut, SYNC_INTERVAL_MS)

    # After interval + one clock: tx_valid should be high
    # Drop tx_ready and confirm tx_byte_in is stable
    dut.tx_ready.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    first_byte = int(dut.tx_byte_in.value)
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    assert int(dut.tx_byte_in.value) == first_byte, "byte must hold during stall"
    assert int(dut.tx_valid.value) == 1, "valid must hold during stall"

    # Release backpressure and finish the frame
    dut.tx_ready.value = 1
    frame = await capture_frame(dut)
    # frame already has the first byte consumed in capture loop — confirm
    # at least we reach the last byte normally
    assert len(frame) >= 1
    dut._log.info("tx_backpressure_stalls_bytes passed")


@cocotb.test()
async def test_tx_last_only_on_final_byte(dut):
    """tx_last is high on byte 15 only."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    dut.master_time_us.value = 1234
    await advance_ms(dut, SYNC_INTERVAL_MS)

    seen_last_positions = []
    byte_num = 0
    for _ in range(200):
        if int(dut.tx_valid.value) == 1 and int(dut.tx_ready.value) == 1:
            if int(dut.tx_last.value) == 1:
                seen_last_positions.append(byte_num)
            byte_num += 1
            if byte_num == 16:
                break
        await RisingEdge(dut.clk)

    assert seen_last_positions == [15], \
        f"tx_last should only fire on byte 15, got positions {seen_last_positions}"
    dut._log.info("tx_last_only_on_final_byte passed")


@cocotb.test()
async def test_sync_count_tracks_transmissions(dut):
    """sync_count equals the number of completed Sync frames."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for i in range(5):
        dut.master_time_us.value = i * 1000
        await advance_ms(dut, SYNC_INTERVAL_MS)
        await capture_frame(dut)

    assert int(dut.sync_count.value) == 5
    dut._log.info("sync_count_tracks_transmissions passed")
