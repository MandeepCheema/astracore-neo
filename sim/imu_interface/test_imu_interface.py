"""
AstraCore Neo — IMU Interface cocotb testbench

13-byte frame: header 0x3A + 6 × 2-byte big-endian signed values.
spi_byte_valid pulses feed bytes; spi_frame_end commits the frame.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


HDR = 0x3A


def to_s16(v):
    v = int(v)
    return v if v < (1 << 15) else v - (1 << 16)


def pack_s16(v):
    """Convert a signed int to a (msb, lsb) tuple of bytes."""
    v &= 0xFFFF
    return ((v >> 8) & 0xFF, v & 0xFF)


def build_imu_frame(ax, ay, az, gx, gy, gz, *, bad_header=False):
    """Build a 13-byte IMU frame."""
    hdr = 0x00 if bad_header else HDR
    frame = [hdr]
    for v in (ax, ay, az, gx, gy, gz):
        msb, lsb = pack_s16(v)
        frame.extend([msb, lsb])
    return frame


async def reset_dut(dut):
    dut.rst_n.value          = 0
    dut.spi_byte_valid.value = 0
    dut.spi_byte.value       = 0
    dut.spi_frame_end.value  = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def send_frame(dut, byte_list):
    """Send bytes sequentially, then pulse spi_frame_end."""
    for b in byte_list:
        dut.spi_byte.value       = b
        dut.spi_byte_valid.value = 1
        await RisingEdge(dut.clk)
    dut.spi_byte_valid.value = 0
    await RisingEdge(dut.clk)
    # Frame-end pulse
    dut.spi_frame_end.value = 1
    await RisingEdge(dut.clk)
    dut.spi_frame_end.value = 0
    await RisingEdge(dut.clk)


# ===========================================================================
# Tests
# ===========================================================================

@cocotb.test()
async def test_initial_state(dut):
    """After reset: no valid, all axes zero, no frames."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    assert int(dut.imu_valid.value)   == 0
    assert int(dut.frame_count.value) == 0
    assert int(dut.error_count.value) == 0
    dut._log.info("initial_state passed")


@cocotb.test()
async def test_single_valid_frame(dut):
    """A well-formed frame populates all 6 DOF registers and pulses imu_valid."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ax, ay, az = 100, -200, 1000
    gx, gy, gz = 500, -1500, 50
    await send_frame(dut, build_imu_frame(ax, ay, az, gx, gy, gz))

    assert to_s16(dut.accel_x_mg.value)  == ax
    assert to_s16(dut.accel_y_mg.value)  == ay
    assert to_s16(dut.accel_z_mg.value)  == az
    assert to_s16(dut.gyro_x_mdps.value) == gx
    assert to_s16(dut.gyro_y_mdps.value) == gy
    assert to_s16(dut.gyro_z_mdps.value) == gz
    assert int(dut.frame_count.value)    == 1
    assert int(dut.error_count.value)    == 0
    dut._log.info("single_valid_frame passed")


@cocotb.test()
async def test_negative_values(dut):
    """Negative readings round-trip through signed output registers."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await send_frame(dut, build_imu_frame(-1, -100, -32768, -500, -1000, -16000))

    assert to_s16(dut.accel_x_mg.value)  == -1
    assert to_s16(dut.accel_y_mg.value)  == -100
    assert to_s16(dut.accel_z_mg.value)  == -32768
    assert to_s16(dut.gyro_x_mdps.value) == -500
    assert to_s16(dut.gyro_y_mdps.value) == -1000
    assert to_s16(dut.gyro_z_mdps.value) == -16000
    dut._log.info("negative_values passed")


@cocotb.test()
async def test_bad_header_rejected(dut):
    """Wrong header byte → error_count++, outputs unchanged."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await send_frame(dut, build_imu_frame(1, 2, 3, 4, 5, 6, bad_header=True))

    assert int(dut.frame_count.value) == 0
    assert int(dut.error_count.value) == 1
    assert int(dut.accel_x_mg.value)  == 0, "outputs must not be updated"
    dut._log.info("bad_header_rejected passed")


@cocotb.test()
async def test_short_frame_rejected(dut):
    """Frame with fewer than 13 bytes → rejected on frame_end."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    full = build_imu_frame(1, 2, 3, 4, 5, 6)
    await send_frame(dut, full[:10])   # only 10 bytes

    assert int(dut.frame_count.value) == 0
    assert int(dut.error_count.value) == 1
    dut._log.info("short_frame_rejected passed")


@cocotb.test()
async def test_multiple_frames(dut):
    """Multiple valid frames increment frame_count and refresh data each time."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    frames = [
        (  0,   0, 1000,  0,  0,  0),
        (500, -500,  900, 100, -100, 50),
        (-1000, 200,  800,  -50, 250, -75),
    ]
    for (ax, ay, az, gx, gy, gz) in frames:
        await send_frame(dut, build_imu_frame(ax, ay, az, gx, gy, gz))
        assert to_s16(dut.accel_x_mg.value)  == ax
        assert to_s16(dut.gyro_z_mdps.value) == gz

    assert int(dut.frame_count.value) == len(frames)
    assert int(dut.error_count.value) == 0
    dut._log.info("multiple_frames passed")


@cocotb.test()
async def test_imu_valid_pulse_width(dut):
    """imu_valid stays high for exactly 1 cycle."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Manually drive the frame to observe the pulse
    for b in build_imu_frame(1, 2, 3, 4, 5, 6):
        dut.spi_byte.value       = b
        dut.spi_byte_valid.value = 1
        await RisingEdge(dut.clk)
    dut.spi_byte_valid.value = 0
    await RisingEdge(dut.clk)
    dut.spi_frame_end.value = 1
    await RisingEdge(dut.clk)   # EDGE A: frame_end sampled, NBAs scheduled
    dut.spi_frame_end.value = 0

    await RisingEdge(dut.clk)   # EDGE B: imu_valid visible
    assert int(dut.imu_valid.value) == 1

    await RisingEdge(dut.clk)   # EDGE C: default de-assert visible
    assert int(dut.imu_valid.value) == 0
    dut._log.info("imu_valid_pulse_width passed")


@cocotb.test()
async def test_error_then_recovery(dut):
    """Bad frame followed by good frame — both counters advance correctly."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await send_frame(dut, build_imu_frame(9, 9, 9, 9, 9, 9, bad_header=True))
    assert int(dut.error_count.value) == 1

    await send_frame(dut, build_imu_frame(42, -42, 100, 0, 0, 0))
    assert int(dut.frame_count.value) == 1
    assert int(dut.error_count.value) == 1
    assert to_s16(dut.accel_x_mg.value) == 42
    dut._log.info("error_then_recovery passed")
