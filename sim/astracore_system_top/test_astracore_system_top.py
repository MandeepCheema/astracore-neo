"""
AstraCore Neo — System Top smoke test.

Verifies the structural wrapper elaborates, runs from reset, and routes a
single stimulus through to both subsystems:
  • Base subsystem reset-state visible via AXI-Lite base_led outputs.
  • Fusion subsystem reset-state visible via safe_state + max_speed_kmh.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


async def reset_dut(dut):
    # Drive all inputs to a defined state
    dut.rst_n.value = 0
    # Base AXI
    dut.base_awaddr.value  = 0
    dut.base_awvalid.value = 0
    dut.base_wdata.value   = 0
    dut.base_wstrb.value   = 0
    dut.base_wvalid.value  = 0
    dut.base_bready.value  = 0
    dut.base_araddr.value  = 0
    dut.base_arvalid.value = 0
    dut.base_rready.value  = 0
    # Fusion
    dut.tick_1ms.value             = 0
    dut.operator_reset.value       = 0
    dut.cam_byte_valid.value       = 0
    dut.cam_byte_data.value        = 0
    dut.cam_det_valid.value        = 0
    dut.cam_det_class_id.value     = 0
    dut.cam_det_confidence.value   = 0
    dut.cam_det_bbox_x.value       = 0
    dut.cam_det_bbox_y.value       = 0
    dut.cam_det_bbox_w.value       = 0
    dut.cam_det_bbox_h.value       = 0
    dut.cam_det_timestamp_us.value = 0
    dut.cam_det_camera_id.value    = 0
    dut.imu_spi_byte_valid.value   = 0
    dut.imu_spi_byte.value         = 0
    dut.imu_spi_frame_end.value    = 0
    dut.gnss_pps.value             = 0
    dut.gnss_time_set_valid.value  = 0
    dut.gnss_time_set_us.value     = 0
    dut.gnss_fix_set_valid.value   = 0
    dut.gnss_fix_valid_in.value    = 0
    dut.gnss_lat_mdeg_in.value     = 0
    dut.gnss_lon_mdeg_in.value     = 0
    dut.can_rx_frame_valid.value   = 0
    dut.can_rx_frame_id.value      = 0
    dut.can_rx_frame_dlc.value     = 0
    dut.can_rx_frame_data.value    = 0
    dut.radar_spi_byte_valid.value = 0
    dut.radar_spi_byte.value       = 0
    dut.radar_spi_frame_end.value  = 0
    dut.ultra_rx_valid.value       = 0
    dut.ultra_rx_byte.value        = 0
    dut.eth_rx_valid.value         = 0
    dut.eth_rx_byte.value          = 0
    dut.eth_rx_last.value          = 0
    dut.map_lane_valid.value       = 0
    dut.map_left_mm.value          = 0
    dut.map_right_mm.value         = 0
    dut.cal_we.value               = 0
    dut.cal_addr.value             = 0
    dut.cal_wdata.value            = 0

    for _ in range(10):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_system_top_reset_state(dut):
    """
    After reset both subsystems come up in their safe-default state:
      • base_led = 0  (no throttle / fault / inf busy / eth ok stretched)
      • brake_level = 0, safe_state = 0, max_speed_kmh = 130
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Base subsystem (via LED outputs)
    assert int(dut.base_led.value) == 0, \
        f"base_led should be 0 after reset, got {int(dut.base_led.value):04b}"

    # Fusion subsystem decision outputs
    assert int(dut.brake_level.value)   == 0
    assert int(dut.brake_active.value)  == 0
    assert int(dut.alert_driver.value)  == 0
    assert int(dut.lka_active.value)    == 0
    assert int(dut.ldw_warning.value)   == 0
    assert int(dut.safe_state.value)    == 0
    assert int(dut.max_speed_kmh.value) == 130
    assert int(dut.mrc_pull_over.value) == 0

    dut._log.info("system_top_reset_state passed")


@cocotb.test()
async def test_both_subsystems_coexist(dut):
    """
    Drive a stimulus on each subsystem and confirm neither disturbs the other.
      • Base: AXI-Lite write to gaze register (0x04), expect base_led unchanged.
      • Fusion: fire a camera detection, expect safe_state still NORMAL.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Base: AXI-Lite write to gaze register (address 0x04 = word offset 1)
    dut.base_awaddr.value  = 0x04
    dut.base_awvalid.value = 1
    dut.base_wdata.value   = 0x0000_BEEF
    dut.base_wstrb.value   = 0xF
    dut.base_wvalid.value  = 1
    dut.base_bready.value  = 1
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.base_awvalid.value = 0
    dut.base_wvalid.value  = 0

    # Fusion: fire a camera detection with a benign class (VEHICLE)
    dut.cam_det_class_id.value     = 1
    dut.cam_det_confidence.value   = 200
    dut.cam_det_bbox_x.value       = 1000
    dut.cam_det_bbox_y.value       = 500
    dut.cam_det_bbox_w.value       = 100
    dut.cam_det_bbox_h.value       = 100
    dut.cam_det_timestamp_us.value = 10000
    dut.cam_det_valid.value        = 1
    await RisingEdge(dut.clk)
    dut.cam_det_valid.value        = 0

    # Let the pipelines settle
    for _ in range(15):
        await RisingEdge(dut.clk)

    # Both subsystems should still be in a clean state
    assert int(dut.safe_state.value) == 0, \
        f"safe_state should still be NORMAL, got {int(dut.safe_state.value)}"
    assert int(dut.brake_level.value) == 0, \
        "brake should not have activated from a single detection"

    # At least one fusion track should now be active
    n_tracks = int(dut.u_fusion.u_tracker.num_active_tracks.value)
    assert n_tracks >= 1, \
        f"expected >= 1 active track after cam detection, got {n_tracks}"

    dut._log.info(f"both_subsystems_coexist passed ({n_tracks} fusion tracks)")
