"""
AstraCore Neo — Coordinate Transform cocotb testbench

2-stage pipelined transform (Q15 fixed-point rotation + calibration offsets):
  Stage 1: multiply (det_valid → s1_valid, 1-cycle latency)
  Stage 2: accumulate + Q15 descale + offset (s1_valid → out_valid, 1-cycle latency)

Timing:
  EDGE A: det_valid sampled → s1 products scheduled as NBAs
  EDGE B: s1 results visible → out results scheduled as NBAs
  EDGE C: out results visible (cocotb reads here)

Total pipeline latency: 2 clock cycles.

AXI-Lite cal register word address: addr[6:2] = index 0-19.
  Sensor k base index = k*5:  off_x, off_y, off_z, cos_q15, sin_q15
Default (power-on): identity — cos=32767, sin=0, off=0 for all sensors.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


async def reset_dut(dut):
    dut.rst_n.value         = 0
    dut.det_valid.value     = 0
    dut.det_sensor_id.value = 0
    dut.det_x_mm.value      = 0
    dut.det_y_mm.value      = 0
    dut.det_z_mm.value      = 0
    dut.cal_we.value        = 0
    dut.cal_addr.value      = 0
    dut.cal_wdata.value     = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def write_cal(dut, sensor_id, off_x, off_y, off_z, cos_q15, sin_q15):
    """Write all 5 calibration registers for sensor_id (5 sequential clock cycles)."""
    base = sensor_id * 5
    for idx, val in enumerate([off_x, off_y, off_z, cos_q15, sin_q15]):
        dut.cal_addr.value  = (base + idx) << 2   # [6:2] = word index
        dut.cal_wdata.value = val & 0xFFFFFFFF
        dut.cal_we.value    = 1
        await RisingEdge(dut.clk)
    dut.cal_we.value = 0
    await RisingEdge(dut.clk)   # settle: cal_regs and combinatorial wires updated


async def fire_and_read(dut, sensor_id, x, y, z):
    """
    Drive 1-cycle det_valid, then return outputs after 2-cycle pipeline.
    Returns (out_x, out_y, out_z, out_sensor_id) as signed Python ints.

    Timing:
      EDGE A: det_valid=1 sampled; s1 products scheduled
      EDGE B: s1 results visible; out results scheduled
      EDGE C: out results visible — read here
    """
    dut.det_sensor_id.value = sensor_id
    dut.det_x_mm.value      = x & 0xFFFFFFFF
    dut.det_y_mm.value      = y & 0xFFFFFFFF
    dut.det_z_mm.value      = z & 0xFFFFFFFF
    dut.det_valid.value     = 1
    await RisingEdge(dut.clk)   # EDGE A
    dut.det_valid.value     = 0
    await RisingEdge(dut.clk)   # EDGE B: stage-1 visible, stage-2 scheduled
    await RisingEdge(dut.clk)   # EDGE C: stage-2 visible

    def to_s32(v):
        v = int(v)
        return v if v < (1 << 31) else v - (1 << 32)

    return (
        to_s32(dut.out_x_mm.value),
        to_s32(dut.out_y_mm.value),
        to_s32(dut.out_z_mm.value),
        int(dut.out_sensor_id.value),
    )


def ref(det_x, det_y, det_z, cos_q15, sin_q15, off_x, off_y, off_z):
    """
    Python reference model — matches RTL integer arithmetic exactly.
    Q15 descale uses arithmetic right shift (Python >> for signed ints).
    """
    xcos = det_x * cos_q15
    xsin = det_x * sin_q15
    ycos = det_y * cos_q15
    ysin = det_y * sin_q15
    body_x = ((xcos - ysin) >> 15) + off_x
    body_y = ((xsin + ycos) >> 15) + off_y
    body_z = det_z + off_z
    return body_x, body_y, body_z


# ===========================================================================
# Tests
# ===========================================================================

@cocotb.test()
async def test_initial_state(dut):
    """After reset: out_valid=0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    assert int(dut.out_valid.value) == 0, "out_valid should be 0 after reset"
    dut._log.info("initial_state passed")


@cocotb.test()
async def test_pipeline_latency(dut):
    """out_valid appears exactly 2 cycles after det_valid, de-asserts after 1 cycle."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    dut.det_valid.value = 1
    await RisingEdge(dut.clk)   # EDGE A: sampled
    dut.det_valid.value = 0
    assert int(dut.out_valid.value) == 0, "out_valid should be 0 at EDGE A (only stage 0 done)"

    await RisingEdge(dut.clk)   # EDGE B: stage 1 done, stage 2 scheduled
    assert int(dut.out_valid.value) == 0, "out_valid should be 0 at EDGE B (stage 2 not settled)"

    await RisingEdge(dut.clk)   # EDGE C: stage 2 settled
    assert int(dut.out_valid.value) == 1, "out_valid should be 1 at EDGE C (2-cycle latency)"

    await RisingEdge(dut.clk)   # EDGE D: de-assert propagated
    assert int(dut.out_valid.value) == 0, "out_valid should de-assert after 1 cycle"
    dut._log.info("pipeline_latency passed")


@cocotb.test()
async def test_identity_transform_no_offset(dut):
    """Default identity cal (cos=32767, sin=0, off=0): output matches reference model."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    x, y, z = 1000, -500, 200
    ox, oy, oz, _ = await fire_and_read(dut, 0, x, y, z)
    ex, ey, ez = ref(x, y, z, 32767, 0, 0, 0, 0)

    assert ox == ex, f"out_x: got {ox}, expected {ex}"
    assert oy == ey, f"out_y: got {oy}, expected {ey}"
    assert oz == ez, f"out_z: got {oz}, expected {ez}"
    dut._log.info(f"identity_transform_no_offset passed: ({ox},{oy},{oz})")


@cocotb.test()
async def test_x_offset(dut):
    """X calibration offset applied correctly."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await write_cal(dut, 1, 500, 0, 0, 32767, 0)   # sensor 1: x_off=500mm

    x, y, z = 100, 0, 0
    ox, oy, oz, _ = await fire_and_read(dut, 1, x, y, z)
    ex, ey, ez = ref(x, y, z, 32767, 0, 500, 0, 0)

    assert ox == ex, f"out_x: got {ox}, expected {ex}"
    assert oy == ey, f"out_y: got {oy}, expected {ey}"
    dut._log.info(f"x_offset passed: out_x={ox}")


@cocotb.test()
async def test_y_offset(dut):
    """Y calibration offset applied correctly."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await write_cal(dut, 2, 0, -300, 0, 32767, 0)  # sensor 2: y_off=-300mm

    x, y, z = 0, 200, 0
    ox, oy, oz, _ = await fire_and_read(dut, 2, x, y, z)
    ex, ey, ez = ref(x, y, z, 32767, 0, 0, -300, 0)

    assert oy == ey, f"out_y: got {oy}, expected {ey}"
    dut._log.info(f"y_offset passed: out_y={oy}")


@cocotb.test()
async def test_z_offset(dut):
    """Z passes through unchanged, offset added."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await write_cal(dut, 0, 0, 0, 1500, 32767, 0)  # sensor 0: z_off=1500mm

    x, y, z = 0, 0, 800
    ox, oy, oz, _ = await fire_and_read(dut, 0, x, y, z)
    ex, ey, ez = ref(x, y, z, 32767, 0, 0, 0, 1500)

    assert oz == ez, f"out_z: got {oz}, expected {ez}"
    dut._log.info(f"z_offset passed: out_z={oz}")


@cocotb.test()
async def test_90_degree_rotation(dut):
    """90-deg yaw (cos=0, sin=32767): body_x = -(sensor_y * 32767 >> 15)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    cos_q15, sin_q15 = 0, 32767
    await write_cal(dut, 0, 0, 0, 0, cos_q15, sin_q15)

    x, y, z = 1000, 500, 0
    ox, oy, oz, _ = await fire_and_read(dut, 0, x, y, z)
    ex, ey, ez = ref(x, y, z, cos_q15, sin_q15, 0, 0, 0)

    assert ox == ex, f"out_x: got {ox}, expected {ex}"
    assert oy == ey, f"out_y: got {oy}, expected {ey}"
    dut._log.info(f"90deg_rotation passed: ({ox},{oy}) expected ({ex},{ey})")


@cocotb.test()
async def test_negative_90_degree_rotation(dut):
    """Negative 90-deg yaw (cos=0, sin=-32768): body_x = sensor_y, body_y = -sensor_x."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    cos_q15, sin_q15 = 0, -32768   # Q15 -1.0
    await write_cal(dut, 0, 0, 0, 0, cos_q15, sin_q15 & 0xFFFF)

    x, y, z = 1000, 500, 0
    ox, oy, oz, _ = await fire_and_read(dut, 0, x, y, z)
    ex, ey, ez = ref(x, y, z, cos_q15, sin_q15, 0, 0, 0)

    assert ox == ex, f"out_x: got {ox}, expected {ex}"
    assert oy == ey, f"out_y: got {oy}, expected {ey}"
    dut._log.info(f"neg_90deg_rotation passed: ({ox},{oy}) expected ({ex},{ey})")


@cocotb.test()
async def test_sensor_id_echoed(dut):
    """out_sensor_id echoes det_sensor_id through both pipeline stages."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for sid in range(4):
        await write_cal(dut, sid, 0, 0, 0, 32767, 0)
        _, _, _, out_sid = await fire_and_read(dut, sid, 0, 0, 0)
        assert out_sid == sid, f"out_sensor_id: got {out_sid}, expected {sid}"
    dut._log.info("sensor_id_echoed passed")


@cocotb.test()
async def test_calibration_write_takes_effect(dut):
    """Writing cal registers changes output on the following detection."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    x, y, z = 500, 0, 0

    # Step 1: default identity → body_x = (500*32767)>>15
    ox1, _, _, _ = await fire_and_read(dut, 0, x, y, z)
    ex1, _, _    = ref(x, y, z, 32767, 0, 0, 0, 0)
    assert ox1 == ex1, f"step1 out_x: got {ox1}, expected {ex1}"

    # Step 2: write x_offset=1000mm for sensor 0
    await write_cal(dut, 0, 1000, 0, 0, 32767, 0)
    ox2, _, _, _ = await fire_and_read(dut, 0, x, y, z)
    ex2, _, _    = ref(x, y, z, 32767, 0, 1000, 0, 0)
    assert ox2 == ex2, f"step2 out_x: got {ox2}, expected {ex2}"
    assert ox2 == ox1 + 1000, f"offset delta wrong: {ox2} vs {ox1}+1000"
    dut._log.info(f"calibration_write_takes_effect passed: {ox1} -> {ox2}")


@cocotb.test()
async def test_combined_rotation_and_offset(dut):
    """45-degree yaw + all three offsets: output matches reference model."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # 45 degrees: cos = sin = round(32768 * sqrt(0.5)) = 23170 (standard Q15 approximation)
    cos_q15 = 23170
    sin_q15 = 23170
    off_x, off_y, off_z = 200, -100, 50
    await write_cal(dut, 3, off_x, off_y, off_z, cos_q15, sin_q15)

    x, y, z = 300, 400, 100
    ox, oy, oz, _ = await fire_and_read(dut, 3, x, y, z)
    ex, ey, ez     = ref(x, y, z, cos_q15, sin_q15, off_x, off_y, off_z)

    assert ox == ex, f"out_x: got {ox}, expected {ex}"
    assert oy == ey, f"out_y: got {oy}, expected {ey}"
    assert oz == ez, f"out_z: got {oz}, expected {ez}"
    dut._log.info(f"combined_rotation_and_offset passed: ({ox},{oy},{oz})")
