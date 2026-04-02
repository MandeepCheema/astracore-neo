"""
AstraCore Neo — GazeTracker cocotb testbench.

This is the bridge between the Python behavioral simulation and the Verilog RTL.
The Python GazeTracker is the GOLDEN REFERENCE; the Verilog DUT must produce
matching outputs for every stimulus.

Encoding convention
-------------------
EAR values in Python are floats in [0.0, 1.0].
In Verilog they are 8-bit integers: verilog_val = round(python_float * 255).

EyeState mapping
----------------
Python EyeState  →  2-bit Verilog eye_state
  OPEN    (0)    →  2'b00
  PARTIAL (1)    →  2'b01
  CLOSED  (2)    →  2'b10
"""

import sys
import os

# Make the project src/ available so we can import the Python reference model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from dms import GazeTracker, EyeState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EYE_STATE_MAP = {
    EyeState.OPEN:    0b00,
    EyeState.PARTIAL: 0b01,
    EyeState.CLOSED:  0b10,
}


def float_to_ear8(v: float) -> int:
    """Convert a Python float EAR [0,1] to 8-bit integer for the DUT."""
    return max(0, min(255, round(v * 255)))


async def reset_dut(dut):
    """Apply synchronous active-low reset for 5 clock cycles."""
    dut.rst_n.value = 0
    dut.valid.value = 0
    dut.left_ear.value = 0
    dut.right_ear.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def drive_frame(dut, ref: GazeTracker, left: float, right: float):
    """
    Drive one frame into the DUT and the Python reference simultaneously.
    Returns the Python GazeReading so the caller can make assertions.
    """
    # Drive DUT inputs
    dut.left_ear.value  = float_to_ear8(left)
    dut.right_ear.value = float_to_ear8(right)
    dut.valid.value = 1
    await RisingEdge(dut.clk)
    dut.valid.value = 0
    await RisingEdge(dut.clk)   # outputs registered on the clock after valid

    # Advance Python reference
    reading = ref.update(left, right)
    return reading


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_eye_state_open(dut):
    """Fully open eyes → DUT must report OPEN (2'b00)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = GazeTracker(perclos_window=30)
    reading = await drive_frame(dut, ref, left=0.40, right=0.38)

    assert dut.eye_state.value == EYE_STATE_MAP[reading.eye_state], (
        f"eye_state mismatch: DUT={dut.eye_state.value} "
        f"REF={EYE_STATE_MAP[reading.eye_state]} ({reading.eye_state})"
    )
    dut._log.info(f"OPEN test passed: DUT eye_state={dut.eye_state.value}")


@cocotb.test()
async def test_eye_state_partial(dut):
    """Partially open eyes → DUT must report PARTIAL (2'b01)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = GazeTracker(perclos_window=30)
    reading = await drive_frame(dut, ref, left=0.25, right=0.25)

    assert dut.eye_state.value == EYE_STATE_MAP[reading.eye_state], (
        f"eye_state mismatch: DUT={dut.eye_state.value} "
        f"REF={EYE_STATE_MAP[reading.eye_state]}"
    )
    dut._log.info(f"PARTIAL test passed: DUT eye_state={dut.eye_state.value}")


@cocotb.test()
async def test_eye_state_closed(dut):
    """Closed eyes → DUT must report CLOSED (2'b10)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = GazeTracker(perclos_window=30)
    reading = await drive_frame(dut, ref, left=0.10, right=0.10)

    assert dut.eye_state.value == EYE_STATE_MAP[reading.eye_state], (
        f"eye_state mismatch: DUT={dut.eye_state.value} "
        f"REF={EYE_STATE_MAP[reading.eye_state]}"
    )
    dut._log.info(f"CLOSED test passed: DUT eye_state={dut.eye_state.value}")


@cocotb.test()
async def test_avg_ear_output(dut):
    """avg_ear_out must match round((left+right)/2 * 255)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    left, right = 0.40, 0.20
    ref = GazeTracker(perclos_window=30)
    await drive_frame(dut, ref, left=left, right=right)

    expected_avg8 = (float_to_ear8(left) + float_to_ear8(right)) >> 1
    assert dut.avg_ear_out.value == expected_avg8, (
        f"avg_ear_out mismatch: DUT={dut.avg_ear_out.value} expected={expected_avg8}"
    )
    dut._log.info(f"avg_ear test passed: {dut.avg_ear_out.value} == {expected_avg8}")


@cocotb.test()
async def test_perclos_zero_all_open(dut):
    """All open frames → perclos_num must be 0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = GazeTracker(perclos_window=30)
    for _ in range(30):
        await drive_frame(dut, ref, left=0.40, right=0.40)

    assert dut.perclos_num.value == 0, (
        f"perclos_num should be 0, got {dut.perclos_num.value}"
    )
    dut._log.info("perclos_zero test passed")


@cocotb.test()
async def test_perclos_all_closed(dut):
    """All closed frames → perclos_num must equal WINDOW_SIZE (30)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = GazeTracker(perclos_window=30)
    for _ in range(30):
        await drive_frame(dut, ref, left=0.10, right=0.10)

    assert dut.perclos_num.value == 30, (
        f"perclos_num should be 30, got {dut.perclos_num.value}"
    )
    dut._log.info("perclos_all_closed test passed")


@cocotb.test()
async def test_perclos_half(dut):
    """15 closed then 15 open → perclos_num must be 15."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = GazeTracker(perclos_window=30)
    for _ in range(15):
        await drive_frame(dut, ref, left=0.10, right=0.10)   # closed
    for _ in range(15):
        await drive_frame(dut, ref, left=0.40, right=0.40)   # open

    assert dut.perclos_num.value == 15, (
        f"perclos_num should be 15, got {dut.perclos_num.value}"
    )
    dut._log.info("perclos_half test passed")


@cocotb.test()
async def test_perclos_window_rolls(dut):
    """
    Fill with closed frames, then push 30 open frames.
    After rolling, perclos_num should drop back to 0.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = GazeTracker(perclos_window=30)
    # Fill window with closed
    for _ in range(30):
        await drive_frame(dut, ref, left=0.10, right=0.10)
    assert dut.perclos_num.value == 30, "Pre-condition: window should be all-closed"

    # Push 30 open frames — should evict all closed entries
    for _ in range(30):
        await drive_frame(dut, ref, left=0.40, right=0.40)
    assert dut.perclos_num.value == 0, (
        f"After rolling, perclos_num should be 0, got {dut.perclos_num.value}"
    )
    dut._log.info("perclos_roll test passed")


@cocotb.test()
async def test_blink_counting(dut):
    """Three CLOSED→OPEN transitions → blink_count must be 3."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = GazeTracker(perclos_window=30)
    for _ in range(3):
        await drive_frame(dut, ref, left=0.40, right=0.40)   # OPEN
        await drive_frame(dut, ref, left=0.10, right=0.10)   # CLOSED
        await drive_frame(dut, ref, left=0.40, right=0.40)   # OPEN → blink

    assert dut.blink_count.value == 3, (
        f"blink_count should be 3, got {dut.blink_count.value}"
    )
    assert ref.blink_count == 3, f"Python ref blink_count should be 3, got {ref.blink_count}"
    dut._log.info(f"blink_count test passed: DUT={dut.blink_count.value} REF={ref.blink_count}")


@cocotb.test()
async def test_reset_clears_state(dut):
    """After reset, all outputs should return to initial state."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = GazeTracker(perclos_window=30)
    # Build up state
    for _ in range(20):
        await drive_frame(dut, ref, left=0.10, right=0.10)

    # Now reset
    dut.rst_n.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    assert dut.perclos_num.value == 0, "perclos_num should be 0 after reset"
    assert dut.blink_count.value == 0, "blink_count should be 0 after reset"
    assert dut.eye_state.value == 0,   "eye_state should be OPEN after reset"
    dut._log.info("reset test passed")


@cocotb.test()
async def test_reference_match_sequence(dut):
    """
    Drive a mixed EAR sequence and verify DUT eye_state matches Python reference
    on every single frame.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = GazeTracker(perclos_window=30)

    sequence = [
        (0.40, 0.38),   # OPEN
        (0.35, 0.33),   # OPEN
        (0.25, 0.24),   # PARTIAL
        (0.10, 0.10),   # CLOSED
        (0.10, 0.10),   # CLOSED
        (0.40, 0.40),   # OPEN  (blink)
        (0.28, 0.26),   # PARTIAL
        (0.05, 0.05),   # CLOSED
        (0.40, 0.40),   # OPEN  (blink)
        (0.40, 0.40),   # OPEN
    ]

    for frame_idx, (l, r) in enumerate(sequence):
        reading = await drive_frame(dut, ref, left=l, right=r)
        expected = EYE_STATE_MAP[reading.eye_state]
        actual   = int(dut.eye_state.value)
        assert actual == expected, (
            f"Frame {frame_idx}: left={l} right={r} — "
            f"DUT eye_state={actual} REF={expected} ({reading.eye_state})"
        )

    dut._log.info(
        f"reference_match_sequence passed — "
        f"DUT perclos={dut.perclos_num.value} "
        f"REF perclos={ref.perclos:.2f} "
        f"DUT blinks={dut.blink_count.value} REF blinks={ref.blink_count}"
    )
