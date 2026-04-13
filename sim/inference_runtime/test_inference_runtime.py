"""
AstraCore Neo — Inference Runtime cocotb testbench.

Validates the inference session state machine.

State encoding:
  3'd0  UNLOADED
  3'd1  LOADED
  3'd2  RUNNING
  3'd3  DONE
  3'd4  ERROR

Python SessionState reference:
  UNLOADED (1) → 3'd0
  LOADED   (2) → 3'd1
  RUNNING  (3) → 3'd2
  DONE     (4) → 3'd3
  ERROR    (5) → 3'd4
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from inference.runtime import SessionState

ST_UNLOADED = 0
ST_LOADED   = 1
ST_RUNNING  = 2
ST_DONE     = 3
ST_ERROR    = 4

STATE_NAMES = {0: "UNLOADED", 1: "LOADED", 2: "RUNNING", 3: "DONE", 4: "ERROR"}


async def reset_dut(dut):
    dut.rst_n.value        = 0
    dut.load_start.value   = 0
    dut.model_valid.value  = 0
    dut.run_start.value    = 0
    dut.abort.value        = 0
    dut.run_done_in.value  = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def pulse_signal(dut, signal_name, model_valid=1):
    """Pulse a control signal for one clock cycle."""
    if signal_name == "load_start":
        dut.load_start.value  = 1
        dut.model_valid.value = model_valid
    else:
        getattr(dut, signal_name).value = 1
    await RisingEdge(dut.clk)
    dut.load_start.value  = 0
    dut.model_valid.value = 0
    getattr(dut, signal_name).value = 0 if signal_name != "load_start" else 0
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_initial_state(dut):
    """After reset, state = UNLOADED (0)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    assert int(dut.state.value) == ST_UNLOADED, (
        f"Expected UNLOADED(0), got {dut.state.value}"
    )
    assert int(dut.busy.value) == 0
    assert int(dut.error.value) == 0
    dut._log.info("initial_state test passed")


@cocotb.test()
async def test_load_valid_model(dut):
    """load_start + model_valid → LOADED (1)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    dut.load_start.value  = 1
    dut.model_valid.value = 1
    await RisingEdge(dut.clk)
    dut.load_start.value  = 0
    dut.model_valid.value = 0
    await RisingEdge(dut.clk)

    assert int(dut.state.value) == ST_LOADED, (
        f"Expected LOADED(1), got {dut.state.value}"
    )
    assert int(dut.busy.value) == 1, "busy should be asserted in LOADED"
    dut._log.info("load_valid_model test passed")


@cocotb.test()
async def test_load_invalid_model(dut):
    """load_start + !model_valid → ERROR (4)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    dut.load_start.value  = 1
    dut.model_valid.value = 0
    await RisingEdge(dut.clk)
    dut.load_start.value  = 0
    await RisingEdge(dut.clk)

    assert int(dut.state.value) == ST_ERROR, (
        f"Expected ERROR(4), got {dut.state.value}"
    )
    assert int(dut.error.value) == 1
    dut._log.info("load_invalid_model test passed")


@cocotb.test()
async def test_loaded_to_running(dut):
    """LOADED + run_start → RUNNING (2)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Load
    dut.load_start.value  = 1
    dut.model_valid.value = 1
    await RisingEdge(dut.clk)
    dut.load_start.value  = 0
    dut.model_valid.value = 0
    await RisingEdge(dut.clk)

    # Run
    dut.run_start.value = 1
    await RisingEdge(dut.clk)
    dut.run_start.value = 0
    await RisingEdge(dut.clk)

    assert int(dut.state.value) == ST_RUNNING, (
        f"Expected RUNNING(2), got {dut.state.value}"
    )
    assert int(dut.busy.value) == 1
    dut._log.info("loaded_to_running test passed")


@cocotb.test()
async def test_running_to_done(dut):
    """RUNNING + run_done_in → DONE (3); session_done pulses."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Load + Run
    dut.load_start.value  = 1
    dut.model_valid.value = 1
    await RisingEdge(dut.clk)
    dut.load_start.value  = 0
    dut.model_valid.value = 0
    await RisingEdge(dut.clk)
    dut.run_start.value = 1
    await RisingEdge(dut.clk)
    dut.run_start.value = 0
    await RisingEdge(dut.clk)

    # Signal done
    dut.run_done_in.value = 1
    await RisingEdge(dut.clk)
    dut.run_done_in.value = 0
    await RisingEdge(dut.clk)

    assert int(dut.state.value) == ST_DONE, (
        f"Expected DONE(3), got {dut.state.value}"
    )
    assert int(dut.busy.value)         == 0
    assert int(dut.session_done.value) == 1, "session_done should pulse"

    # Check session_done is only a pulse
    await RisingEdge(dut.clk)
    assert int(dut.session_done.value) == 0, "session_done should deassert"
    dut._log.info("running_to_done test passed")


@cocotb.test()
async def test_abort_any_state(dut):
    """abort from RUNNING → ERROR (4)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Load + Run
    dut.load_start.value  = 1
    dut.model_valid.value = 1
    await RisingEdge(dut.clk)
    dut.load_start.value  = 0
    dut.model_valid.value = 0
    await RisingEdge(dut.clk)
    dut.run_start.value = 1
    await RisingEdge(dut.clk)
    dut.run_start.value = 0
    await RisingEdge(dut.clk)

    # Abort
    dut.abort.value = 1
    await RisingEdge(dut.clk)
    dut.abort.value = 0
    await RisingEdge(dut.clk)

    assert int(dut.state.value) == ST_ERROR, (
        f"Expected ERROR(4) after abort, got {dut.state.value}"
    )
    assert int(dut.error.value) == 1
    dut._log.info("abort_any_state test passed")


@cocotb.test()
async def test_error_recovery(dut):
    """ERROR + load_start → UNLOADED (recovery path)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Force error
    dut.load_start.value  = 1
    dut.model_valid.value = 0
    await RisingEdge(dut.clk)
    dut.load_start.value  = 0
    await RisingEdge(dut.clk)
    assert int(dut.state.value) == ST_ERROR

    # Recovery
    dut.load_start.value = 1
    await RisingEdge(dut.clk)
    dut.load_start.value = 0
    await RisingEdge(dut.clk)

    assert int(dut.state.value) == ST_UNLOADED, (
        f"Expected UNLOADED after recovery, got {dut.state.value}"
    )
    dut._log.info("error_recovery test passed")


@cocotb.test()
async def test_done_reload_and_rerun(dut):
    """From DONE, load_start → LOADED; then run_start → RUNNING."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Full run to DONE
    dut.load_start.value  = 1
    dut.model_valid.value = 1
    await RisingEdge(dut.clk)
    dut.load_start.value  = 0
    dut.model_valid.value = 0
    await RisingEdge(dut.clk)
    dut.run_start.value = 1
    await RisingEdge(dut.clk)
    dut.run_start.value = 0
    await RisingEdge(dut.clk)
    dut.run_done_in.value = 1
    await RisingEdge(dut.clk)
    dut.run_done_in.value = 0
    await RisingEdge(dut.clk)
    assert int(dut.state.value) == ST_DONE

    # Re-run from DONE
    dut.run_start.value = 1
    await RisingEdge(dut.clk)
    dut.run_start.value = 0
    await RisingEdge(dut.clk)

    assert int(dut.state.value) == ST_RUNNING, (
        f"Expected RUNNING from DONE+run_start, got {dut.state.value}"
    )
    dut._log.info("done_reload_and_rerun test passed")


@cocotb.test()
async def test_busy_signal(dut):
    """busy should be asserted in LOADED and RUNNING, but not UNLOADED/DONE/ERROR."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    assert int(dut.busy.value) == 0   # UNLOADED

    dut.load_start.value  = 1
    dut.model_valid.value = 1
    await RisingEdge(dut.clk)
    dut.load_start.value  = 0
    dut.model_valid.value = 0
    await RisingEdge(dut.clk)
    assert int(dut.busy.value) == 1   # LOADED

    dut.run_start.value = 1
    await RisingEdge(dut.clk)
    dut.run_start.value = 0
    await RisingEdge(dut.clk)
    assert int(dut.busy.value) == 1   # RUNNING

    dut.run_done_in.value = 1
    await RisingEdge(dut.clk)
    dut.run_done_in.value = 0
    await RisingEdge(dut.clk)
    assert int(dut.busy.value) == 0   # DONE

    dut._log.info("busy_signal test passed")


@cocotb.test()
async def test_reset_returns_to_unloaded(dut):
    """Reset from any state returns to UNLOADED."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Get to RUNNING
    dut.load_start.value  = 1
    dut.model_valid.value = 1
    await RisingEdge(dut.clk)
    dut.load_start.value  = 0
    dut.model_valid.value = 0
    await RisingEdge(dut.clk)
    dut.run_start.value = 1
    await RisingEdge(dut.clk)
    dut.run_start.value = 0
    await RisingEdge(dut.clk)

    # Reset
    dut.rst_n.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    assert int(dut.state.value) == ST_UNLOADED
    assert int(dut.busy.value) == 0
    assert int(dut.error.value) == 0
    dut._log.info("reset_returns_to_unloaded test passed")
