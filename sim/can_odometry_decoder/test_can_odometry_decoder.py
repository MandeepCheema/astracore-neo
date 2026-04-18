"""
AstraCore Neo — CAN Odometry Decoder cocotb testbench

Consumes frames from canfd_controller RX FIFO (rx_out_* AXI-S port).
Decodes:
  WHEEL_SPEED_ID (0x1A0): 4 × u16 wheel speeds (FL/FR/RL/RR, mm/s)
  STEERING_ID    (0x1B0): s16 steering angle + s16 yaw rate

Timing: event-driven — 1 clock after rx_out_valid, odo_valid pulses with
the decoded fields (read at 2 RisingEdges after driving, per NBA pattern).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


WHEEL_SPEED_ID = 0x1A0
STEERING_ID    = 0x1B0


def to_s16(v):
    v = int(v)
    return v if v < (1 << 15) else v - (1 << 16)


def pack_wheels(fl, fr, rl, rr):
    """Pack 4 u16 wheel speeds into 64-bit big-endian data word."""
    return ((fl & 0xFFFF) << 48) | ((fr & 0xFFFF) << 32) \
         | ((rl & 0xFFFF) << 16) | (rr & 0xFFFF)


def pack_steering(steer_mdeg, yaw_rate_mdps):
    """Pack steering angle + yaw rate into 64-bit data word."""
    return ((steer_mdeg & 0xFFFF) << 48) | ((yaw_rate_mdps & 0xFFFF) << 32)


async def reset_dut(dut):
    dut.rst_n.value        = 0
    dut.rx_out_valid.value = 0
    dut.rx_out_id.value    = 0
    dut.rx_out_dlc.value   = 0
    dut.rx_out_data.value  = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def fire_frame(dut, can_id, data64, dlc=8):
    """Drive a 1-cycle rx_out_valid pulse with the given CAN frame."""
    dut.rx_out_id.value    = can_id
    dut.rx_out_dlc.value   = dlc
    dut.rx_out_data.value  = data64 & 0xFFFFFFFFFFFFFFFF
    dut.rx_out_valid.value = 1
    await RisingEdge(dut.clk)   # EDGE A: sampled, NBAs scheduled
    dut.rx_out_valid.value = 0
    await RisingEdge(dut.clk)   # EDGE B: outputs visible


# ===========================================================================
# Tests
# ===========================================================================

@cocotb.test()
async def test_initial_state(dut):
    """After reset: no valid, all fields zero."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    assert int(dut.odo_valid.value)             == 0
    assert int(dut.wheel_speed_mmps.value)      == 0
    assert to_s16(dut.steer_mdeg.value)         == 0
    assert to_s16(dut.odo_yaw_rate_mdps.value)  == 0
    assert int(dut.wheel_frame_count.value)     == 0
    assert int(dut.steering_frame_count.value)  == 0
    dut._log.info("initial_state passed")


@cocotb.test()
async def test_rx_out_ready_always_high(dut):
    """Decoder accepts frames unconditionally."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    assert int(dut.rx_out_ready.value) == 1
    dut._log.info("rx_out_ready_always_high passed")


@cocotb.test()
async def test_wheel_speed_frame_decoded(dut):
    """Wheel-speed CAN frame → per-wheel latches + average."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    fl, fr, rl, rr = 1000, 1100, 1050, 1150
    await fire_frame(dut, WHEEL_SPEED_ID, pack_wheels(fl, fr, rl, rr))

    assert int(dut.odo_valid.value)         == 1
    assert int(dut.wheel_fl_mmps.value)     == fl
    assert int(dut.wheel_fr_mmps.value)     == fr
    assert int(dut.wheel_rl_mmps.value)     == rl
    assert int(dut.wheel_rr_mmps.value)     == rr
    expected_avg = (fl + fr + rl + rr) >> 2
    assert int(dut.wheel_speed_mmps.value)  == expected_avg, \
        f"avg: got {int(dut.wheel_speed_mmps.value)}, expected {expected_avg}"
    assert int(dut.wheel_frame_count.value) == 1
    dut._log.info(f"wheel_speed_frame_decoded passed: avg={expected_avg}")


@cocotb.test()
async def test_average_computation(dut):
    """Known inputs produce exact integer average."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    cases = [
        (0, 0, 0, 0, 0),
        (100, 100, 100, 100, 100),
        (100, 200, 300, 400, 250),     # (100+200+300+400)/4 = 250
        (65535, 65535, 65535, 65535, 65535),  # saturation check
        (1000, 1000, 0, 0, 500),
    ]
    for (fl, fr, rl, rr, expected) in cases:
        await fire_frame(dut, WHEEL_SPEED_ID, pack_wheels(fl, fr, rl, rr))
        assert int(dut.wheel_speed_mmps.value) == expected, \
            f"({fl},{fr},{rl},{rr}): got {int(dut.wheel_speed_mmps.value)}, expected {expected}"
    dut._log.info("average_computation passed")


@cocotb.test()
async def test_steering_frame_decoded(dut):
    """Steering CAN frame → steer_mdeg and yaw_rate latched."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_frame(dut, STEERING_ID, pack_steering(2500, 1200))

    assert int(dut.odo_valid.value) == 1
    assert to_s16(dut.steer_mdeg.value)        == 2500
    assert to_s16(dut.odo_yaw_rate_mdps.value) == 1200
    assert int(dut.steering_frame_count.value) == 1
    dut._log.info("steering_frame_decoded passed")


@cocotb.test()
async def test_negative_steering_and_yaw(dut):
    """Signed fields round-trip with negative values."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_frame(dut, STEERING_ID, pack_steering(-3000, -800))
    assert to_s16(dut.steer_mdeg.value)        == -3000
    assert to_s16(dut.odo_yaw_rate_mdps.value) == -800
    dut._log.info("negative_steering_and_yaw passed")


@cocotb.test()
async def test_unknown_id_ignored(dut):
    """Unrecognised CAN ID increments ignored counter, no odo_valid pulse."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_frame(dut, 0x0777, 0xDEADBEEF_CAFEF00D)

    assert int(dut.odo_valid.value)             == 0
    assert int(dut.wheel_frame_count.value)     == 0
    assert int(dut.steering_frame_count.value)  == 0
    assert int(dut.ignored_frame_count.value)   == 1
    dut._log.info("unknown_id_ignored passed")


@cocotb.test()
async def test_back_to_back_mixed_frames(dut):
    """Interleaved wheel + steering frames update independent latches."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await fire_frame(dut, WHEEL_SPEED_ID, pack_wheels(500, 500, 500, 500))
    assert int(dut.wheel_speed_mmps.value) == 500

    await fire_frame(dut, STEERING_ID, pack_steering(100, 50))
    assert to_s16(dut.steer_mdeg.value)        == 100
    assert to_s16(dut.odo_yaw_rate_mdps.value) == 50
    # Wheel speed should still be latched
    assert int(dut.wheel_speed_mmps.value) == 500

    await fire_frame(dut, WHEEL_SPEED_ID, pack_wheels(1000, 1000, 1000, 1000))
    assert int(dut.wheel_speed_mmps.value) == 1000
    # Steering still latched from previous
    assert to_s16(dut.steer_mdeg.value) == 100

    assert int(dut.wheel_frame_count.value)    == 2
    assert int(dut.steering_frame_count.value) == 1
    dut._log.info("back_to_back_mixed_frames passed")


@cocotb.test()
async def test_odo_valid_pulse_width(dut):
    """odo_valid pulses for exactly 1 cycle."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    dut.rx_out_id.value    = WHEEL_SPEED_ID
    dut.rx_out_data.value  = pack_wheels(1, 2, 3, 4)
    dut.rx_out_valid.value = 1
    await RisingEdge(dut.clk)   # EDGE A: sample
    dut.rx_out_valid.value = 0
    assert int(dut.odo_valid.value) == 0, "should be 0 at EDGE A"

    await RisingEdge(dut.clk)   # EDGE B: NBA visible
    assert int(dut.odo_valid.value) == 1, "should pulse at EDGE B"

    await RisingEdge(dut.clk)   # EDGE C: default de-assert
    assert int(dut.odo_valid.value) == 0, "should de-assert at EDGE C"
    dut._log.info("odo_valid_pulse_width passed")
