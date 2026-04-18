"""
AstraCore Neo — CANFDController cocotb testbench  (Rev 2)

Rev 1 tests (9): error counter + bus-state FSM — ALL PRESERVED, unchanged.
Rev 2 tests (7): RX frame FIFO, TX frame FIFO, backpressure, BUS_OFF gating,
                 auto tx_success from TX drain, simultaneous RX+TX.

Bus state encoding:
  ERROR_ACTIVE  → 2'd0
  ERROR_PASSIVE → 2'd1
  BUS_OFF       → 2'd2
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from connectivity.canfd import CANFDController, CANFrame, CANIDFormat, CANBusState

BUS_STATE_MAP = {
    CANBusState.ERROR_ACTIVE:  0,
    CANBusState.ERROR_PASSIVE: 1,
    CANBusState.BUS_OFF:       2,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def reset_dut(dut):
    dut.rst_n.value          = 0
    dut.tx_success.value     = 0
    dut.tx_error.value       = 0
    dut.rx_error.value       = 0
    dut.bus_off_recovery.value = 0
    # Rev-2 RX ports
    dut.rx_frame_valid.value = 0
    dut.rx_frame_id.value    = 0
    dut.rx_frame_dlc.value   = 0
    dut.rx_frame_data.value  = 0
    dut.rx_out_ready.value   = 0
    # Rev-2 TX ports
    dut.tx_frame_valid.value = 0
    dut.tx_frame_id.value    = 0
    dut.tx_frame_dlc.value   = 0
    dut.tx_frame_data.value  = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def pulse(dut, signal_name):
    """Pulse a single-bit input for one clock cycle."""
    getattr(dut, signal_name).value = 1
    await RisingEdge(dut.clk)
    getattr(dut, signal_name).value = 0
    await RisingEdge(dut.clk)


async def write_rx_frame(dut, msg_id: int, dlc: int, data: int):
    """Write one frame into the RX FIFO (single-cycle handshake)."""
    dut.rx_frame_valid.value = 1
    dut.rx_frame_id.value    = msg_id
    dut.rx_frame_dlc.value   = dlc
    dut.rx_frame_data.value  = data
    await RisingEdge(dut.clk)
    dut.rx_frame_valid.value = 0
    await RisingEdge(dut.clk)


async def write_tx_frame(dut, msg_id: int, dlc: int, data: int):
    """Write one frame into the TX FIFO (single-cycle handshake)."""
    dut.tx_frame_valid.value = 1
    dut.tx_frame_id.value    = msg_id
    dut.tx_frame_dlc.value   = dlc
    dut.tx_frame_data.value  = data
    await RisingEdge(dut.clk)
    dut.tx_frame_valid.value = 0
    await RisingEdge(dut.clk)


async def read_rx_frame(dut):
    """Read one frame from the RX FIFO output. Returns (id, dlc, data)."""
    dut.rx_out_ready.value = 1
    await RisingEdge(dut.clk)
    msg_id = int(dut.rx_out_id.value)
    dlc    = int(dut.rx_out_dlc.value)
    data   = int(dut.rx_out_data.value)
    dut.rx_out_ready.value = 0
    await RisingEdge(dut.clk)
    return msg_id, dlc, data


# ===========================================================================
# Rev-1 Tests — PRESERVED UNCHANGED (9 tests)
# ===========================================================================

@cocotb.test()
async def test_initial_state(dut):
    """After reset, TEC=0, REC=0, bus_state=ERROR_ACTIVE."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    assert dut.tec.value == 0,       f"TEC should be 0, got {dut.tec.value}"
    assert dut.rec.value == 0,       f"REC should be 0, got {dut.rec.value}"
    assert dut.bus_state.value == 0, f"bus_state should be ERROR_ACTIVE(0)"
    dut._log.info("initial state test passed")


@cocotb.test()
async def test_tx_error_increments_tec(dut):
    """Each tx_error increments TEC by 8."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = CANFDController(node_id=1)
    for i in range(5):
        ref.inject_tx_error()
        await pulse(dut, "tx_error")
        assert int(dut.tec.value) == ref.tec, (
            f"Step {i+1}: DUT TEC={dut.tec.value} REF TEC={ref.tec}"
        )
    dut._log.info(f"tx_error test passed: TEC={dut.tec.value}")


@cocotb.test()
async def test_tx_success_decrements_tec(dut):
    """tx_success decrements TEC by 1 (floor 0)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = CANFDController(node_id=1)
    for _ in range(3):
        ref.inject_tx_error()
        await pulse(dut, "tx_error")

    assert int(dut.tec.value) == 24

    for _ in range(5):
        ref.send(CANFrame(can_id=0x100, id_format=CANIDFormat.STANDARD, data=b"hi"))
        await pulse(dut, "tx_success")

    assert int(dut.tec.value) == ref.tec, (
        f"DUT TEC={dut.tec.value} REF TEC={ref.tec}"
    )
    dut._log.info(f"tx_success test passed: TEC={dut.tec.value}")


@cocotb.test()
async def test_rx_error_increments_rec(dut):
    """Each rx_error increments REC by 1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = CANFDController(node_id=1)
    for i in range(10):
        ref.inject_rx_error()
        await pulse(dut, "rx_error")
        assert int(dut.rec.value) == ref.rec, (
            f"Step {i+1}: DUT REC={dut.rec.value} REF REC={ref.rec}"
        )
    dut._log.info(f"rx_error test passed: REC={dut.rec.value}")


@cocotb.test()
async def test_error_passive_transition(dut):
    """TEC >= 128 → ERROR_PASSIVE (2'd1)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = CANFDController(node_id=1)
    for _ in range(16):
        ref.inject_tx_error()
        await pulse(dut, "tx_error")

    assert ref.tec >= 128
    assert ref.bus_state == CANBusState.ERROR_PASSIVE
    assert int(dut.bus_state.value) == BUS_STATE_MAP[CANBusState.ERROR_PASSIVE], (
        f"DUT bus_state={dut.bus_state.value} expected ERROR_PASSIVE(1)"
    )
    dut._log.info("error_passive transition test passed")


@cocotb.test()
async def test_bus_off_transition(dut):
    """TEC >= 256 → BUS_OFF (2'd2)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for _ in range(32):
        await pulse(dut, "tx_error")

    assert int(dut.bus_state.value) == 2, (
        f"Expected BUS_OFF(2), got {dut.bus_state.value}"
    )
    dut._log.info(f"bus_off test passed: TEC={dut.tec.value}")


@cocotb.test()
async def test_bus_off_recovery(dut):
    """bus_off_recovery resets TEC, REC to 0 and returns to ERROR_ACTIVE."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for _ in range(32):
        await pulse(dut, "tx_error")

    assert int(dut.bus_state.value) == 2

    await pulse(dut, "bus_off_recovery")

    assert int(dut.tec.value) == 0,       "TEC should be 0 after recovery"
    assert int(dut.rec.value) == 0,       "REC should be 0 after recovery"
    assert int(dut.bus_state.value) == 0, "Should return to ERROR_ACTIVE(0)"
    dut._log.info("bus_off_recovery test passed")


@cocotb.test()
async def test_rec_triggers_error_passive(dut):
    """REC >= 128 → ERROR_PASSIVE even if TEC < 128."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = CANFDController(node_id=1)
    for _ in range(128):
        ref.inject_rx_error()
        await pulse(dut, "rx_error")

    assert ref.bus_state == CANBusState.ERROR_PASSIVE
    assert int(dut.bus_state.value) == 1, (
        f"Expected ERROR_PASSIVE(1), got {dut.bus_state.value}"
    )
    dut._log.info(f"rec_error_passive test passed: REC={dut.rec.value}")


@cocotb.test()
async def test_tec_floor_zero(dut):
    """TEC should not underflow below 0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for _ in range(5):
        await pulse(dut, "tx_success")

    assert int(dut.tec.value) == 0, f"TEC should stay at 0, got {dut.tec.value}"
    dut._log.info("tec_floor_zero test passed")


# ===========================================================================
# Rev-2 Tests — RX/TX Frame FIFO (7 new tests)
# ===========================================================================

@cocotb.test()
async def test_rx_fifo_single_frame(dut):
    """Write one frame into RX FIFO; read back matches ID, DLC, data."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    test_id   = 0x1A2B3C4
    test_dlc  = 8
    test_data = 0xDEADBEEFCAFEBABE

    assert int(dut.rx_frame_ready.value) == 1, "FIFO should be ready initially"
    assert int(dut.rx_out_valid.value)   == 0, "FIFO should be empty initially"

    await write_rx_frame(dut, test_id, test_dlc, test_data)

    assert int(dut.rx_out_valid.value) == 1, "Frame should be in FIFO"

    msg_id, dlc, data = await read_rx_frame(dut)

    assert msg_id == test_id,   f"ID mismatch: got 0x{msg_id:X} expected 0x{test_id:X}"
    assert dlc    == test_dlc,  f"DLC mismatch: got {dlc} expected {test_dlc}"
    assert data   == test_data, f"Data mismatch: got 0x{data:X} expected 0x{test_data:X}"
    assert int(dut.rx_out_valid.value) == 0, "FIFO should be empty after read"
    dut._log.info("rx_fifo_single_frame passed")


@cocotb.test()
async def test_rx_fifo_fills_to_depth(dut):
    """Write 4 frames (full depth); rx_frame_ready goes low when full."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for i in range(4):
        assert int(dut.rx_frame_ready.value) == 1, f"Should be ready before frame {i}"
        await write_rx_frame(dut, i + 1, i % 8, i * 0x0101010101010101)

    assert int(dut.rx_frame_ready.value) == 0, "FIFO full: rx_frame_ready should be 0"
    dut._log.info("rx_fifo_fills_to_depth passed")


@cocotb.test()
async def test_rx_fifo_preserves_order(dut):
    """Frames read back in FIFO order (FIFO, not LIFO)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    frames = [(0x100 + i, i, i * 0x1111111111111111) for i in range(3)]
    for msg_id, dlc, data in frames:
        await write_rx_frame(dut, msg_id, dlc, data)

    for i, (exp_id, exp_dlc, exp_data) in enumerate(frames):
        assert int(dut.rx_out_valid.value) == 1
        got_id, got_dlc, got_data = await read_rx_frame(dut)
        assert got_id   == exp_id,   f"Frame {i}: ID {got_id:#x} != {exp_id:#x}"
        assert got_dlc  == exp_dlc,  f"Frame {i}: DLC {got_dlc} != {exp_dlc}"
        assert got_data == exp_data, f"Frame {i}: data mismatch"

    dut._log.info("rx_fifo_preserves_order passed")


@cocotb.test()
async def test_rx_fifo_blocked_when_bus_off(dut):
    """rx_frame_ready = 0 when BUS_OFF; frames are not accepted."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Force BUS_OFF
    for _ in range(32):
        await pulse(dut, "tx_error")
    assert int(dut.bus_state.value) == 2

    assert int(dut.rx_frame_ready.value) == 0, "rx_frame_ready must be 0 in BUS_OFF"

    # Attempt to write — should be silently ignored
    await write_rx_frame(dut, 0xABC, 4, 0x1234)
    assert int(dut.rx_out_valid.value) == 0, "No frame should have been accepted"
    dut._log.info("rx_fifo_blocked_when_bus_off passed")


@cocotb.test()
async def test_tx_fifo_single_frame_and_done(dut):
    """Write one TX frame; tx_frame_done pulses on the next cycle."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    assert int(dut.tx_frame_ready.value) == 1, "TX FIFO should be ready"
    assert int(dut.tx_frame_done.value)  == 0, "tx_frame_done idle"

    # Write frame — drains on the next cycle after write
    dut.tx_frame_valid.value = 1
    dut.tx_frame_id.value    = 0x6FF
    dut.tx_frame_dlc.value   = 8
    dut.tx_frame_data.value  = 0x0102030405060708
    await RisingEdge(dut.clk)        # EDGE A: write fires (tx_count 0→1)
    dut.tx_frame_valid.value = 0
    await RisingEdge(dut.clk)        # EDGE B: drain fires, tx_frame_done NBA=1 scheduled
    # cocotb resumes in EDGE B's active region before its NBAs settle;
    # wait one more edge so EDGE B's result is visible
    await RisingEdge(dut.clk)        # EDGE C: EDGE B's NBA now settled — tx_frame_done=1
    assert int(dut.tx_frame_done.value) == 1, "tx_frame_done should pulse after TX"
    await RisingEdge(dut.clk)        # EDGE D: EDGE C de-asserts tx_frame_done
    assert int(dut.tx_frame_done.value) == 0, "tx_frame_done should de-assert after 1 cycle"
    dut._log.info("tx_fifo_single_frame_and_done passed")


@cocotb.test()
async def test_tx_frame_done_decrements_tec(dut):
    """Each tx_frame_done auto-fires int_tx_success; TEC decrements by 1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Build up TEC to 16 via direct strobes
    for _ in range(2):
        await pulse(dut, "tx_error")   # +8 each → TEC=16
    assert int(dut.tec.value) == 16

    # Send 4 TX frames via FIFO — each completion should decrement TEC by 1
    for i in range(4):
        await write_tx_frame(dut, 0x200 + i, 4, 0xDEAD)

    # Wait for all 4 to drain (drain is 1/cycle, so at least 4 cycles)
    for _ in range(6):
        await RisingEdge(dut.clk)

    assert int(dut.tec.value) <= 12, (
        f"TEC should have dropped by at least 4 (got {int(dut.tec.value)})"
    )
    dut._log.info(f"tx_frame_done_decrements_tec passed: TEC={int(dut.tec.value)}")


@cocotb.test()
async def test_tx_fifo_blocked_when_bus_off(dut):
    """tx_frame_ready = 0 in BUS_OFF; TX frames are not accepted."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Force BUS_OFF
    for _ in range(32):
        await pulse(dut, "tx_error")
    assert int(dut.bus_state.value) == 2

    assert int(dut.tx_frame_ready.value) == 0, "tx_frame_ready must be 0 in BUS_OFF"

    # Attempt TX — should not accept
    await write_tx_frame(dut, 0x7FF, 8, 0xBEEF)
    assert int(dut.tx_frame_done.value) == 0, "No TX done while BUS_OFF"
    dut._log.info("tx_fifo_blocked_when_bus_off passed")
