"""
AstraCore Neo — CANFDController cocotb testbench.

Python CANFDController is the GOLDEN REFERENCE for error counter behaviour
and bus state transitions.

Bus state encoding:
  Python CANBusState  →  2-bit Verilog bus_state
  ERROR_ACTIVE  (1)  →  2'd0
  ERROR_PASSIVE (2)  →  2'd1
  BUS_OFF       (3)  →  2'd2
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


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.tx_success.value = 0
    dut.tx_error.value = 0
    dut.rx_error.value = 0
    dut.bus_off_recovery.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def pulse(dut, signal_name):
    """Pulse a single-bit input for one clock cycle."""
    getattr(dut, signal_name).value = 1
    await RisingEdge(dut.clk)
    getattr(dut, signal_name).value = 0
    await RisingEdge(dut.clk)  # sample outputs


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

    # Build up TEC
    for _ in range(3):
        ref.inject_tx_error()
        await pulse(dut, "tx_error")

    assert int(dut.tec.value) == 24

    # Decrement via tx_success
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

    # Inject enough tx errors to reach ERROR_PASSIVE (need TEC >= 128)
    # Each tx_error adds 8 → need 16 errors
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

    # Need 32+ tx_errors to push TEC >= 256
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

    # Force BUS_OFF
    for _ in range(32):
        await pulse(dut, "tx_error")

    assert int(dut.bus_state.value) == 2

    # Recovery
    await pulse(dut, "bus_off_recovery")

    assert int(dut.tec.value) == 0,       f"TEC should be 0 after recovery"
    assert int(dut.rec.value) == 0,       f"REC should be 0 after recovery"
    assert int(dut.bus_state.value) == 0, f"Should return to ERROR_ACTIVE(0)"
    dut._log.info("bus_off_recovery test passed")


@cocotb.test()
async def test_rec_triggers_error_passive(dut):
    """REC >= 128 → ERROR_PASSIVE even if TEC < 128."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = CANFDController(node_id=1)

    # Inject 128 rx errors
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

    # tx_success when TEC already 0
    for _ in range(5):
        await pulse(dut, "tx_success")

    assert int(dut.tec.value) == 0, f"TEC should stay at 0, got {dut.tec.value}"
    dut._log.info("tec_floor_zero test passed")
