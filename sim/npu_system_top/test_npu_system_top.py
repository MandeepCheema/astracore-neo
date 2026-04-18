"""Smoke test for rtl/npu_system_top/npu_system_top.v.

Verifies that the chip-top wrapper elaborates cleanly and both subsystems
(base+fusion via astracore_system_top, NPU via npu_top) come up in
defined state after reset. A second test exercises the NPU end-to-end
through the new top-level ports to confirm the pass-through wiring
is correct.

Not a functional re-verification of the subsystems — those are tested
in sim/astracore_system_top + sim/npu_top individually. Goal here is
proof of wrapper correctness.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


N_ROWS = 4
N_COLS = 4
DATA_W = 8
ACC_W  = 32
AI_DATA_W = N_ROWS * DATA_W
AO_DATA_W = N_COLS * ACC_W


def _to_signed(val, w):
    val &= (1 << w) - 1
    return val - (1 << w) if val & (1 << (w - 1)) else val


def _unpack_cvec(val):
    return [_to_signed((val >> (i * ACC_W)) & ((1 << ACC_W) - 1), ACC_W)
            for i in range(N_COLS)]


def _as_int8(v: int) -> int:
    v &= 0xFF
    return v - 0x100 if v & 0x80 else v


def _pack_vec(vec, elem_w):
    out = 0
    for i, v in enumerate(vec):
        out |= (v & ((1 << elem_w) - 1)) << (i * elem_w)
    return out


async def _reset(dut):
    dut.rst_n.value = 0
    # Base AXI
    dut.base_awaddr.value  = 0
    dut.base_awvalid.value = 0
    dut.base_wdata.value   = 0
    dut.base_wstrb.value   = 0
    dut.base_wvalid.value  = 0
    dut.base_bready.value  = 1
    dut.base_araddr.value  = 0
    dut.base_arvalid.value = 0
    dut.base_rready.value  = 1
    # Fusion sensors — zero everything
    for sig in ("tick_1ms","operator_reset","cam_byte_valid","cam_byte_data",
                "cam_det_valid","cam_det_class_id","cam_det_confidence",
                "cam_det_bbox_x","cam_det_bbox_y","cam_det_bbox_w",
                "cam_det_bbox_h","cam_det_timestamp_us","cam_det_camera_id",
                "imu_spi_byte_valid","imu_spi_byte","imu_spi_frame_end",
                "gnss_pps","gnss_time_set_valid","gnss_time_set_us",
                "gnss_fix_set_valid","gnss_fix_valid_in","gnss_lat_mdeg_in",
                "gnss_lon_mdeg_in","can_rx_frame_valid","can_rx_frame_id",
                "can_rx_frame_dlc","can_rx_frame_data","radar_spi_byte_valid",
                "radar_spi_byte","radar_spi_frame_end","ultra_rx_valid",
                "ultra_rx_byte","eth_rx_valid","eth_rx_byte","eth_rx_last",
                "map_lane_valid","map_left_mm","map_right_mm",
                "cal_we","cal_addr","cal_wdata"):
        getattr(dut, sig).value = 0
    # NPU
    dut.npu_start.value             = 0
    dut.npu_cfg_k.value             = 0
    dut.npu_cfg_ai_base.value       = 0
    dut.npu_cfg_ao_base.value       = 0
    dut.npu_cfg_afu_mode.value      = 0
    dut.npu_cfg_acc_init_mode.value = 0
    dut.npu_cfg_acc_init_data.value = 0
    dut.npu_cfg_precision_mode.value = 0
    dut.npu_ext_sparse_skip_vec.value = 0
    dut.npu_ext_w_we.value          = 0
    dut.npu_ext_w_waddr.value       = 0
    dut.npu_ext_w_wdata.value       = 0
    dut.npu_ext_ai_we.value         = 0
    dut.npu_ext_ai_waddr.value      = 0
    dut.npu_ext_ai_wdata.value      = 0
    dut.npu_ext_ao_re.value         = 0
    dut.npu_ext_ao_raddr.value      = 0
    # DMA
    dut.npu_dma_start.value          = 0
    dut.npu_dma_cfg_src_addr.value   = 0
    dut.npu_dma_cfg_ai_base.value    = 0
    dut.npu_dma_cfg_tile_h.value     = 0
    dut.npu_dma_cfg_src_stride.value = 0
    dut.npu_mem_rdata.value          = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


@cocotb.test()
async def test_post_reset_state(dut):
    """Both subsystems must come up in defined state after reset."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    # Fusion safe defaults
    assert int(dut.brake_level.value)  == 0, "brake should be off"
    assert int(dut.brake_active.value) == 0
    assert int(dut.safe_state.value)   == 0, "safe_state NORMAL"
    assert int(dut.max_speed_kmh.value) == 130
    # NPU idle
    assert int(dut.npu_busy.value) == 0
    assert int(dut.npu_done.value) == 0
    dut._log.info("post_reset_state passed — both subsystems in defined state")


@cocotb.test()
async def test_npu_identity_matmul_through_top(dut):
    """Drive an NPU identity matmul end-to-end through the npu_system_top
    ports and confirm the pass-through wiring delivers correct results."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)

    # Load identity weights
    W = [1 if (a // N_COLS) == (a % N_COLS) else 0
         for a in range(N_ROWS * N_COLS)]
    for addr, w in enumerate(W):
        dut.npu_ext_w_we.value    = 1
        dut.npu_ext_w_waddr.value = addr
        dut.npu_ext_w_wdata.value = w & ((1 << DATA_W) - 1)
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
    dut.npu_ext_w_we.value = 0

    # Load one activation vector
    act = [7, -3, 11, -5]
    dut.npu_ext_ai_we.value    = 1
    dut.npu_ext_ai_waddr.value = 0
    dut.npu_ext_ai_wdata.value = _pack_vec(act, DATA_W)
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.npu_ext_ai_we.value = 0

    # Trigger tile
    dut.npu_start.value       = 1
    dut.npu_cfg_k.value       = 1
    dut.npu_cfg_ai_base.value = 0
    dut.npu_cfg_ao_base.value = 0
    dut.npu_cfg_afu_mode.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.npu_start.value = 0

    for _ in range(200):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.npu_done.value):
            break
    else:
        raise AssertionError("NPU did not complete tile via npu_system_top")
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    # Read AO[0]
    dut.npu_ext_ao_re.value    = 1
    dut.npu_ext_ao_raddr.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.npu_ext_ao_re.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    result = _unpack_cvec(int(dut.npu_ext_ao_rdata.value))
    expected = [_as_int8(v) for v in act]
    assert result == expected, (
        f"NPU through npu_system_top: rtl={result} expected={expected}")
    dut._log.info(f"npu end-to-end via npu_system_top PASS: {result}")
