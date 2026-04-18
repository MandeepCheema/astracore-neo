`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — System Top  (astracore_system_top.v)
// =============================================================================
// Unified structural wrapper that instantiates both subsystems of the chip:
//
//   u_base   : astracore_top            — 11 base modules behind AXI4-Lite
//              (gaze, thermal, canfd, ecc, tmr, fault, headpose, pcie,
//              ethernet, mac, inference)
//
//   u_fusion : astracore_fusion_top     — 20-module sensor fusion pipeline
//              (Layer 1 sensor interfaces → Layer 2 fusion → Layer 3 decision)
//
// The two tops share only clk and rst_n; their I/O is physically disjoint.
// A production system ties the signals through an MMCM and reset synchroniser,
// here they pass through directly to keep the wrapper structural.
//
// ── Why not one big AXI-Lite bus? ────────────────────────────────────────────
// The base subsystem uses the AXI-Lite register-bank pattern that is natural
// for software driving individual modules from a host PC.  The fusion pipeline
// is a continuous dataflow where modules feed each other directly and would
// be awkward to stimulate one register at a time.  Merging them into a single
// bus would force the fusion modules to be externally stimulable through
// registers, which works against the point of a dataflow pipeline.
//
// The two subsystems therefore keep their native interfaces and this wrapper
// simply routes everything to top-level pads.  An eventual second-level SoC
// wrapper can place both behind an AXI interconnect once a CPU is added.
// =============================================================================

module astracore_system_top #(
    parameter CLK_FREQ_HZ = 100_000_000
)(
    // ── Clock & Reset ────────────────────────────────────────────────────────
    input  wire        clk,
    input  wire        rst_n,

    // =========================================================================
    // Base subsystem I/O (passthrough to astracore_top)
    // =========================================================================
    // AXI4-Lite slave (write)
    input  wire [7:0]  base_awaddr,
    input  wire        base_awvalid,
    output wire        base_awready,
    input  wire [31:0] base_wdata,
    input  wire [3:0]  base_wstrb,
    input  wire        base_wvalid,
    output wire        base_wready,
    output wire [1:0]  base_bresp,
    output wire        base_bvalid,
    input  wire        base_bready,
    // AXI4-Lite slave (read)
    input  wire [7:0]  base_araddr,
    input  wire        base_arvalid,
    output wire        base_arready,
    output wire [31:0] base_rdata,
    output wire [1:0]  base_rresp,
    output wire        base_rvalid,
    input  wire        base_rready,
    // Board LEDs
    output wire [3:0]  base_led,

    // =========================================================================
    // Fusion subsystem I/O (passthrough to astracore_fusion_top)
    // =========================================================================
    // System tick + operator controls
    input  wire        tick_1ms,
    input  wire        operator_reset,

    // MIPI CSI-2 camera (post-D-PHY)
    input  wire        cam_byte_valid,
    input  wire [7:0]  cam_byte_data,
    output wire        cam_frame_start,
    output wire        cam_frame_end,
    output wire        cam_line_start,
    output wire        cam_line_end,
    output wire        cam_pixel_valid,
    output wire [7:0]  cam_pixel_byte,
    output wire        cam_pixel_last,

    // External CNN detection write port
    input  wire        cam_det_valid,
    input  wire [15:0] cam_det_class_id,
    input  wire [15:0] cam_det_confidence,
    input  wire [15:0] cam_det_bbox_x,
    input  wire [15:0] cam_det_bbox_y,
    input  wire [15:0] cam_det_bbox_w,
    input  wire [15:0] cam_det_bbox_h,
    input  wire [31:0] cam_det_timestamp_us,
    input  wire [7:0]  cam_det_camera_id,

    // IMU SPI
    input  wire        imu_spi_byte_valid,
    input  wire [7:0]  imu_spi_byte,
    input  wire        imu_spi_frame_end,

    // GNSS
    input  wire        gnss_pps,
    input  wire        gnss_time_set_valid,
    input  wire [63:0] gnss_time_set_us,
    input  wire        gnss_fix_set_valid,
    input  wire        gnss_fix_valid_in,
    input  wire signed [31:0] gnss_lat_mdeg_in,
    input  wire signed [31:0] gnss_lon_mdeg_in,

    // CAN-FD RX
    input  wire        can_rx_frame_valid,
    input  wire [28:0] can_rx_frame_id,
    input  wire [3:0]  can_rx_frame_dlc,
    input  wire [63:0] can_rx_frame_data,
    output wire        can_rx_frame_ready,

    // Radar SPI
    input  wire        radar_spi_byte_valid,
    input  wire [7:0]  radar_spi_byte,
    input  wire        radar_spi_frame_end,

    // Ultrasonic UART
    input  wire        ultra_rx_valid,
    input  wire [7:0]  ultra_rx_byte,

    // Ethernet RX
    input  wire        eth_rx_valid,
    input  wire [7:0]  eth_rx_byte,
    input  wire        eth_rx_last,

    // Ethernet TX (PTP Sync output)
    output wire        eth_tx_out_valid,
    output wire [7:0]  eth_tx_out_byte,
    output wire        eth_tx_out_last,

    // HD map lane input
    input  wire        map_lane_valid,
    input  wire signed [31:0] map_left_mm,
    input  wire signed [31:0] map_right_mm,

    // coord_transform calibration interface (software-programmable)
    input  wire        cal_we,
    input  wire [6:0]  cal_addr,
    input  wire [31:0] cal_wdata,

    // Decision outputs
    output wire [1:0]  brake_level,
    output wire        brake_active,
    output wire [15:0] target_decel_mms2,
    output wire        alert_driver,
    output wire        ldw_warning,
    output wire        lka_active,
    output wire signed [15:0] steering_torque_mnm,
    output wire [1:0]  safe_state,
    output wire        limit_speed,
    output wire [7:0]  max_speed_kmh,
    output wire        mrc_pull_over,

    // Status observables
    output wire [63:0] master_time_us,
    output wire        pps_pulse,
    output wire [15:0] ptp_sync_count,
    output wire        window_release,
    output wire [3:0]  sensor_stale_layer1
);

    // =========================================================================
    // Base subsystem instance
    // =========================================================================
    astracore_top #(
        .CLK_FREQ_HZ(CLK_FREQ_HZ)
    ) u_base (
        .clk             (clk),
        .rst_n           (rst_n),

        .s_axil_awaddr   (base_awaddr),
        .s_axil_awvalid  (base_awvalid),
        .s_axil_awready  (base_awready),
        .s_axil_wdata    (base_wdata),
        .s_axil_wstrb    (base_wstrb),
        .s_axil_wvalid   (base_wvalid),
        .s_axil_wready   (base_wready),
        .s_axil_bresp    (base_bresp),
        .s_axil_bvalid   (base_bvalid),
        .s_axil_bready   (base_bready),
        .s_axil_araddr   (base_araddr),
        .s_axil_arvalid  (base_arvalid),
        .s_axil_arready  (base_arready),
        .s_axil_rdata    (base_rdata),
        .s_axil_rresp    (base_rresp),
        .s_axil_rvalid   (base_rvalid),
        .s_axil_rready   (base_rready),
        .led             (base_led)
    );

    // =========================================================================
    // Fusion subsystem instance
    // =========================================================================
    astracore_fusion_top u_fusion (
        .clk                    (clk),
        .rst_n                  (rst_n),
        .tick_1ms               (tick_1ms),
        .operator_reset         (operator_reset),

        .cam_byte_valid         (cam_byte_valid),
        .cam_byte_data          (cam_byte_data),
        .cam_frame_start        (cam_frame_start),
        .cam_frame_end          (cam_frame_end),
        .cam_line_start         (cam_line_start),
        .cam_line_end           (cam_line_end),
        .cam_pixel_valid        (cam_pixel_valid),
        .cam_pixel_byte         (cam_pixel_byte),
        .cam_pixel_last         (cam_pixel_last),

        .cam_det_valid          (cam_det_valid),
        .cam_det_class_id       (cam_det_class_id),
        .cam_det_confidence     (cam_det_confidence),
        .cam_det_bbox_x         (cam_det_bbox_x),
        .cam_det_bbox_y         (cam_det_bbox_y),
        .cam_det_bbox_w         (cam_det_bbox_w),
        .cam_det_bbox_h         (cam_det_bbox_h),
        .cam_det_timestamp_us   (cam_det_timestamp_us),
        .cam_det_camera_id      (cam_det_camera_id),

        .imu_spi_byte_valid     (imu_spi_byte_valid),
        .imu_spi_byte           (imu_spi_byte),
        .imu_spi_frame_end      (imu_spi_frame_end),

        .gnss_pps               (gnss_pps),
        .gnss_time_set_valid    (gnss_time_set_valid),
        .gnss_time_set_us       (gnss_time_set_us),
        .gnss_fix_set_valid     (gnss_fix_set_valid),
        .gnss_fix_valid_in      (gnss_fix_valid_in),
        .gnss_lat_mdeg_in       (gnss_lat_mdeg_in),
        .gnss_lon_mdeg_in       (gnss_lon_mdeg_in),

        .can_rx_frame_valid     (can_rx_frame_valid),
        .can_rx_frame_id        (can_rx_frame_id),
        .can_rx_frame_dlc       (can_rx_frame_dlc),
        .can_rx_frame_data      (can_rx_frame_data),
        .can_rx_frame_ready     (can_rx_frame_ready),

        .radar_spi_byte_valid   (radar_spi_byte_valid),
        .radar_spi_byte         (radar_spi_byte),
        .radar_spi_frame_end    (radar_spi_frame_end),

        .ultra_rx_valid         (ultra_rx_valid),
        .ultra_rx_byte          (ultra_rx_byte),

        .eth_rx_valid           (eth_rx_valid),
        .eth_rx_byte            (eth_rx_byte),
        .eth_rx_last            (eth_rx_last),
        .eth_tx_out_valid       (eth_tx_out_valid),
        .eth_tx_out_byte        (eth_tx_out_byte),
        .eth_tx_out_last        (eth_tx_out_last),

        .map_lane_valid         (map_lane_valid),
        .map_left_mm            (map_left_mm),
        .map_right_mm           (map_right_mm),

        .cal_we                 (cal_we),
        .cal_addr               (cal_addr),
        .cal_wdata              (cal_wdata),

        .brake_level            (brake_level),
        .brake_active           (brake_active),
        .target_decel_mms2      (target_decel_mms2),
        .alert_driver           (alert_driver),
        .ldw_warning            (ldw_warning),
        .lka_active             (lka_active),
        .steering_torque_mnm    (steering_torque_mnm),
        .safe_state             (safe_state),
        .limit_speed            (limit_speed),
        .max_speed_kmh          (max_speed_kmh),
        .mrc_pull_over          (mrc_pull_over),

        .master_time_us         (master_time_us),
        .pps_pulse              (pps_pulse),
        .ptp_sync_count         (ptp_sync_count),
        .window_release         (window_release),
        .sensor_stale_layer1    (sensor_stale_layer1)
    );

endmodule
