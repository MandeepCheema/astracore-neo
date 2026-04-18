`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — NPU-inclusive System Top  (npu_system_top.v)
// =============================================================================
// Final product-level wrapper that instantiates the three subsystems on
// one die:
//
//   u_sys : astracore_system_top   — base 11-module chip + 20-module sensor
//                                    fusion pipeline (already integrated)
//   u_npu : npu_top                 — NPU datapath (pe + systolic array +
//                                    sram controller + dma + activation +
//                                    tile controller + writeback AFUs)
//
// All subsystems share `clk` and `rst_n`. Their I/O is physically disjoint
// at the port level — `base_*` for the base chip's AXI4-Lite, unprefixed
// sensor signals for the fusion pipeline (matches astracore_system_top),
// and `npu_*` for the NPU's external weight-load / activation-load / AO-
// read / start handshake interface.
//
// ── V1 scope ────────────────────────────────────────────────────────────────
// In V1 the NPU and fusion pipelines run side-by-side but do NOT auto-wire.
// Specifically, `cam_det_*` input to the fusion pipeline is still driven
// from an external CNN / camera ECU; the NPU's output is NOT yet post-
// processed into detection bounding-boxes for fusion consumption. That
// handoff requires a `npu_postproc` module (YOLO-NMS + detection packer)
// which is tracked as future work in `memory/open_work_packages.md`.
//
// The V1 wrapper is still useful because:
//   1. Provides the chip-top for timing closure in Phase E
//   2. Lets integration-simulation drive both subsystems concurrently
//   3. Establishes the chip-boundary pad list for the physical design
// =============================================================================

module npu_system_top #(
    parameter CLK_FREQ_HZ  = 100_000_000,

    // NPU parameters — keep at small-test defaults so the wrapper
    // elaborates cleanly for regression; product instantiations override.
    parameter NPU_DATA_W         = 8,
    parameter NPU_ACC_W          = 32,
    parameter NPU_N_ROWS         = 4,
    parameter NPU_N_COLS         = 4,
    parameter NPU_WEIGHT_DEPTH   = 16,
    parameter NPU_ACT_IN_DEPTH   = 16,
    parameter NPU_ACT_OUT_DEPTH  = 16,
    parameter NPU_SCRATCH_DEPTH  = 16,
    parameter NPU_K_W            = 16,
    parameter NPU_DRAIN_CYCLES   = 2,

    // Derived NPU widths (exposed so the caller can size its drivers)
    parameter NPU_W_ADDR_W     = (NPU_WEIGHT_DEPTH  <= 1) ? 1 : $clog2(NPU_WEIGHT_DEPTH),
    parameter NPU_AI_ADDR_W    = (NPU_ACT_IN_DEPTH  <= 1) ? 1 : $clog2(NPU_ACT_IN_DEPTH),
    parameter NPU_AO_ADDR_W    = (NPU_ACT_OUT_DEPTH <= 1) ? 1 : $clog2(NPU_ACT_OUT_DEPTH),
    parameter NPU_AI_DATA_W    = NPU_N_ROWS * NPU_DATA_W,
    parameter NPU_AO_DATA_W    = NPU_N_COLS * NPU_ACC_W
)(
    // ── Clock & Reset (shared) ───────────────────────────────────────────────
    input  wire        clk,
    input  wire        rst_n,

    // =========================================================================
    // Base subsystem I/O (passthrough to astracore_top via astracore_system_top)
    // =========================================================================
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
    input  wire [7:0]  base_araddr,
    input  wire        base_arvalid,
    output wire        base_arready,
    output wire [31:0] base_rdata,
    output wire [1:0]  base_rresp,
    output wire        base_rvalid,
    input  wire        base_rready,
    output wire [3:0]  base_led,

    // =========================================================================
    // Fusion subsystem I/O (passthrough to astracore_fusion_top)
    // =========================================================================
    input  wire        tick_1ms,
    input  wire        operator_reset,

    input  wire        cam_byte_valid,
    input  wire [7:0]  cam_byte_data,
    output wire        cam_frame_start,
    output wire        cam_frame_end,
    output wire        cam_line_start,
    output wire        cam_line_end,
    output wire        cam_pixel_valid,
    output wire [7:0]  cam_pixel_byte,
    output wire        cam_pixel_last,

    input  wire        cam_det_valid,
    input  wire [15:0] cam_det_class_id,
    input  wire [15:0] cam_det_confidence,
    input  wire [15:0] cam_det_bbox_x,
    input  wire [15:0] cam_det_bbox_y,
    input  wire [15:0] cam_det_bbox_w,
    input  wire [15:0] cam_det_bbox_h,
    input  wire [31:0] cam_det_timestamp_us,
    input  wire [7:0]  cam_det_camera_id,

    input  wire        imu_spi_byte_valid,
    input  wire [7:0]  imu_spi_byte,
    input  wire        imu_spi_frame_end,

    input  wire        gnss_pps,
    input  wire        gnss_time_set_valid,
    input  wire [63:0] gnss_time_set_us,
    input  wire        gnss_fix_set_valid,
    input  wire        gnss_fix_valid_in,
    input  wire signed [31:0] gnss_lat_mdeg_in,
    input  wire signed [31:0] gnss_lon_mdeg_in,

    input  wire        can_rx_frame_valid,
    input  wire [28:0] can_rx_frame_id,
    input  wire [3:0]  can_rx_frame_dlc,
    input  wire [63:0] can_rx_frame_data,
    output wire        can_rx_frame_ready,

    input  wire        radar_spi_byte_valid,
    input  wire [7:0]  radar_spi_byte,
    input  wire        radar_spi_frame_end,

    input  wire        ultra_rx_valid,
    input  wire [7:0]  ultra_rx_byte,

    input  wire        eth_rx_valid,
    input  wire [7:0]  eth_rx_byte,
    input  wire        eth_rx_last,
    output wire        eth_tx_out_valid,
    output wire [7:0]  eth_tx_out_byte,
    output wire        eth_tx_out_last,

    input  wire        map_lane_valid,
    input  wire signed [31:0] map_left_mm,
    input  wire signed [31:0] map_right_mm,

    input  wire        cal_we,
    input  wire [6:0]  cal_addr,
    input  wire [31:0] cal_wdata,

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

    output wire [63:0] master_time_us,
    output wire        pps_pulse,
    output wire [15:0] ptp_sync_count,
    output wire        window_release,
    output wire [3:0]  sensor_stale_layer1,

    // =========================================================================
    // NPU subsystem I/O (passthrough to npu_top)
    // =========================================================================
    input  wire                        npu_start,
    input  wire [NPU_K_W-1:0]          npu_cfg_k,
    input  wire [NPU_AI_ADDR_W-1:0]    npu_cfg_ai_base,
    input  wire [NPU_AO_ADDR_W-1:0]    npu_cfg_ao_base,
    input  wire [2:0]                  npu_cfg_afu_mode,
    input  wire                        npu_cfg_acc_init_mode,
    input  wire [NPU_AO_DATA_W-1:0]    npu_cfg_acc_init_data,
    input  wire [1:0]                  npu_cfg_precision_mode,
    output wire                        npu_busy,
    output wire                        npu_done,

    input  wire                        npu_ext_w_we,
    input  wire [NPU_W_ADDR_W-1:0]     npu_ext_w_waddr,
    input  wire [NPU_DATA_W-1:0]       npu_ext_w_wdata,

    input  wire                        npu_ext_ai_we,
    input  wire [NPU_AI_ADDR_W-1:0]    npu_ext_ai_waddr,
    input  wire [NPU_AI_DATA_W-1:0]    npu_ext_ai_wdata,

    input  wire                        npu_ext_ao_re,
    input  wire [NPU_AO_ADDR_W-1:0]    npu_ext_ao_raddr,
    output wire [NPU_AO_DATA_W-1:0]    npu_ext_ao_rdata,

    input  wire [NPU_N_ROWS-1:0]       npu_ext_sparse_skip_vec,

    // DMA path (WP-9): DDR → AI bank via narrow-to-wide packer
    input  wire                        npu_dma_start,
    input  wire [31:0]                 npu_dma_cfg_src_addr,
    input  wire [NPU_AI_ADDR_W-1:0]    npu_dma_cfg_ai_base,
    input  wire [15:0]                 npu_dma_cfg_tile_h,
    input  wire [15:0]                 npu_dma_cfg_src_stride,
    output wire                        npu_dma_busy,
    output wire                        npu_dma_done,
    output wire                        npu_mem_re,
    output wire [31:0]                 npu_mem_raddr,
    input  wire [NPU_DATA_W-1:0]       npu_mem_rdata,

    output wire                        npu_afu_out_valid,
    output wire signed [NPU_ACC_W-1:0] npu_afu_out_data,
    output wire                        npu_afu_out_saturated
);

    // =========================================================================
    // Base + Fusion subsystem (already wrapped by astracore_system_top)
    // =========================================================================
    astracore_system_top #(
        .CLK_FREQ_HZ(CLK_FREQ_HZ)
    ) u_sys (
        .clk                    (clk),
        .rst_n                  (rst_n),

        // base AXI-Lite
        .base_awaddr            (base_awaddr),
        .base_awvalid           (base_awvalid),
        .base_awready           (base_awready),
        .base_wdata             (base_wdata),
        .base_wstrb             (base_wstrb),
        .base_wvalid            (base_wvalid),
        .base_wready            (base_wready),
        .base_bresp             (base_bresp),
        .base_bvalid            (base_bvalid),
        .base_bready            (base_bready),
        .base_araddr            (base_araddr),
        .base_arvalid           (base_arvalid),
        .base_arready           (base_arready),
        .base_rdata             (base_rdata),
        .base_rresp             (base_rresp),
        .base_rvalid            (base_rvalid),
        .base_rready            (base_rready),
        .base_led               (base_led),

        // fusion
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

    // =========================================================================
    // NPU subsystem
    // =========================================================================
    npu_top #(
        .DATA_W        (NPU_DATA_W),
        .ACC_W         (NPU_ACC_W),
        .N_ROWS        (NPU_N_ROWS),
        .N_COLS        (NPU_N_COLS),
        .WEIGHT_DEPTH  (NPU_WEIGHT_DEPTH),
        .ACT_IN_DEPTH  (NPU_ACT_IN_DEPTH),
        .ACT_OUT_DEPTH (NPU_ACT_OUT_DEPTH),
        .SCRATCH_DEPTH (NPU_SCRATCH_DEPTH),
        .K_W           (NPU_K_W),
        .DRAIN_CYCLES  (NPU_DRAIN_CYCLES)
    ) u_npu (
        .clk                (clk),
        .rst_n              (rst_n),

        .start              (npu_start),
        .cfg_k              (npu_cfg_k),
        .cfg_ai_base        (npu_cfg_ai_base),
        .cfg_ao_base        (npu_cfg_ao_base),
        .cfg_afu_mode       (npu_cfg_afu_mode),
        .cfg_acc_init_mode  (npu_cfg_acc_init_mode),
        .cfg_acc_init_data  (npu_cfg_acc_init_data),
        .cfg_precision_mode (npu_cfg_precision_mode),
        .busy               (npu_busy),
        .done               (npu_done),

        .ext_w_we           (npu_ext_w_we),
        .ext_w_waddr        (npu_ext_w_waddr),
        .ext_w_wdata        (npu_ext_w_wdata),

        .ext_ai_we          (npu_ext_ai_we),
        .ext_ai_waddr       (npu_ext_ai_waddr),
        .ext_ai_wdata       (npu_ext_ai_wdata),

        .ext_ao_re          (npu_ext_ao_re),
        .ext_ao_raddr       (npu_ext_ao_raddr),
        .ext_ao_rdata       (npu_ext_ao_rdata),

        .ext_sparse_skip_vec(npu_ext_sparse_skip_vec),

        .dma_start          (npu_dma_start),
        .dma_cfg_src_addr   (npu_dma_cfg_src_addr),
        .dma_cfg_ai_base    (npu_dma_cfg_ai_base),
        .dma_cfg_tile_h     (npu_dma_cfg_tile_h),
        .dma_cfg_src_stride (npu_dma_cfg_src_stride),
        .dma_busy           (npu_dma_busy),
        .dma_done           (npu_dma_done),
        .mem_re             (npu_mem_re),
        .mem_raddr          (npu_mem_raddr),
        .mem_rdata          (npu_mem_rdata),

        .afu_out_valid      (npu_afu_out_valid),
        .afu_out_data       (npu_afu_out_data),
        .afu_out_saturated  (npu_afu_out_saturated)
    );

endmodule
