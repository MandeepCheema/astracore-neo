`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — Sensor Fusion Top-Level  (astracore_fusion_top.v)
// =============================================================================
// Structural integration of the full sensor-fusion RTL stack:
//   Layer 1: 10 sensor-interface modules + the 2 base controllers they extend
//   Layer 2: 6 fusion processing modules
//   Layer 3: 4 decision / output modules
//
// This is a distinct top-level from `astracore_top.v` (which is the FPGA
// bring-up design for the base 11 non-fusion modules behind an AXI4-Lite
// register bank).  The two tops address different purposes and can be
// synthesised independently.  An eventual system top can instantiate both.
//
// ── Design philosophy ────────────────────────────────────────────────────────
// Direct module-to-module wiring (no register bank) — the fusion dataflow is
// an asynchronous pipeline where each stage drives the next on a valid/ready
// or pulse interface.  Where multiple sources could feed one sink (notably
// coord_transform), a simple priority pick is used; the camera-detection
// path is treated as primary and the other sensor FIFOs act as confirming
// presence indicators into sensor_sync and plausibility_checker.
//
// ── Architectural simplifications in v1 ────────────────────────────────────
//   • coord_transform is driven only by cam_detection_receiver drain; radar/
//     lidar/ultrasonic FIFOs feed sensor_sync's sensor_valid mask instead of
//     going through coord_transform.  A future v2 will add a round-robin
//     arbiter so every detection source flows through the transform stage.
//   • ttc_calculator sweeps object_tracker's query port at 1 kHz
//     (one track index per tick_1ms), giving ~125 ms worst-case latency on
//     an 8-entry table.
//   • Closure rate fed to ttc_calculator is approximated as ego_vx_mmps
//     (valid when the object is directly ahead; overestimates risk for
//     side targets, which is conservative for ASIL-D).
//   • plausibility_checker fires once per object_tracker allocation/match,
//     using the current detection class and a sensor_mask bit set from any
//     Layer 1 FIFO that currently has data.
//   • safe_state_controller's fault vector aggregates:
//       critical[0] = canfd BUS_OFF
//       critical[1] = plausibility ASIL_REJECT (last check)
//       critical[2] = sensor_sync double-stale
//       warning [0] = canfd ERROR_PASSIVE
//       warning [1] = any single Layer-1 source stale
//       warning [2] = ego_motion_estimator IMU or odo stale
//
// ── Top-level I/O ────────────────────────────────────────────────────────────
// The top exposes raw sensor byte streams + event outputs that a host would
// stimulate (in simulation) or wire to pads (on silicon).  Parameter names
// mirror the submodule port names to keep the integration obvious.
// =============================================================================

module astracore_fusion_top #(
    parameter integer LIDAR_FIFO_DEPTH  = 8,
    parameter integer RADAR_FIFO_DEPTH  = 16,
    parameter integer CAMDET_FIFO_DEPTH = 16,
    parameter integer OBJTR_NUM_TRACKS  = 8
)(
    // ── Clock / reset / system tick ─────────────────────────────────────────
    input  wire        clk,
    input  wire        rst_n,
    input  wire        tick_1ms,
    input  wire        operator_reset,

    // ── MIPI CSI-2 camera byte stream (post-D-PHY) ───────────────────────────
    input  wire        cam_byte_valid,
    input  wire [7:0]  cam_byte_data,
    output wire        cam_frame_start,
    output wire        cam_frame_end,
    output wire        cam_line_start,
    output wire        cam_line_end,
    output wire        cam_pixel_valid,
    output wire [7:0]  cam_pixel_byte,
    output wire        cam_pixel_last,

    // ── External CNN detection write port ────────────────────────────────────
    input  wire        cam_det_valid,
    input  wire [15:0] cam_det_class_id,
    input  wire [15:0] cam_det_confidence,
    input  wire [15:0] cam_det_bbox_x,
    input  wire [15:0] cam_det_bbox_y,
    input  wire [15:0] cam_det_bbox_w,
    input  wire [15:0] cam_det_bbox_h,
    input  wire [31:0] cam_det_timestamp_us,
    input  wire [7:0]  cam_det_camera_id,

    // ── IMU SPI byte stream ──────────────────────────────────────────────────
    input  wire        imu_spi_byte_valid,
    input  wire [7:0]  imu_spi_byte,
    input  wire        imu_spi_frame_end,

    // ── GNSS control inputs ──────────────────────────────────────────────────
    input  wire        gnss_pps,
    input  wire        gnss_time_set_valid,
    input  wire [63:0] gnss_time_set_us,
    input  wire        gnss_fix_set_valid,
    input  wire        gnss_fix_valid_in,
    input  wire signed [31:0] gnss_lat_mdeg_in,
    input  wire signed [31:0] gnss_lon_mdeg_in,

    // ── CAN-FD RX (from external transceiver) ────────────────────────────────
    input  wire        can_rx_frame_valid,
    input  wire [28:0] can_rx_frame_id,
    input  wire [3:0]  can_rx_frame_dlc,
    input  wire [63:0] can_rx_frame_data,
    output wire        can_rx_frame_ready,

    // ── Radar SPI byte stream ────────────────────────────────────────────────
    input  wire        radar_spi_byte_valid,
    input  wire [7:0]  radar_spi_byte,
    input  wire        radar_spi_frame_end,

    // ── Ultrasonic UART byte stream ──────────────────────────────────────────
    input  wire        ultra_rx_valid,
    input  wire [7:0]  ultra_rx_byte,

    // ── Ethernet RX byte stream (carries LiDAR packets) ─────────────────────
    input  wire        eth_rx_valid,
    input  wire [7:0]  eth_rx_byte,
    input  wire        eth_rx_last,

    // ── Ethernet TX pipeline to PHY (driven by ptp_clock_sync) ──────────────
    output wire        eth_tx_out_valid,
    output wire [7:0]  eth_tx_out_byte,
    output wire        eth_tx_out_last,

    // ── HD map lane input (software-populated) ──────────────────────────────
    input  wire        map_lane_valid,
    input  wire signed [31:0] map_left_mm,
    input  wire signed [31:0] map_right_mm,

    // ── coord_transform calibration interface (software-programmable) ───────
    // Per-vehicle sensor-mounting offsets are written here at install time.
    // A host writes cal_regs[cal_addr[6:2]] = cal_wdata on a 1-cycle cal_we
    // pulse. A production wrapper typically exposes these behind AXI-Lite.
    input  wire        cal_we,
    input  wire [6:0]  cal_addr,
    input  wire [31:0] cal_wdata,

    // ── Decision outputs ─────────────────────────────────────────────────────
    // AEB
    output wire [1:0]  brake_level,
    output wire        brake_active,
    output wire [15:0] target_decel_mms2,
    output wire        alert_driver,
    // LDW/LKA
    output wire        ldw_warning,
    output wire        lka_active,
    output wire signed [15:0] steering_torque_mnm,
    // Safe state
    output wire [1:0]  safe_state,
    output wire        limit_speed,
    output wire [7:0]  max_speed_kmh,
    output wire        mrc_pull_over,

    // ── Time / status observables ───────────────────────────────────────────
    output wire [63:0] master_time_us,
    output wire        pps_pulse,
    output wire [15:0] ptp_sync_count,
    output wire        window_release,
    output wire [3:0]  sensor_stale_layer1
);

    // =========================================================================
    // Internal wires — per Layer 1 module outputs
    // =========================================================================

    // --- imu_interface ---
    wire        w_imu_valid;
    wire signed [15:0] w_accel_x_mg, w_accel_y_mg, w_accel_z_mg;
    wire signed [15:0] w_gyro_x_mdps, w_gyro_y_mdps, w_gyro_z_mdps;

    // --- gnss_interface ---
    wire [63:0] w_master_time_us;
    wire        w_pps_valid;
    wire [63:0] w_pps_time_us;
    wire [15:0] w_pps_count;
    wire        w_gps_fix_valid;
    wire signed [31:0] w_gnss_lat_mdeg, w_gnss_lon_mdeg;

    assign master_time_us = w_master_time_us;
    assign pps_pulse      = w_pps_valid;

    // --- ptp_clock_sync ---
    wire        w_ptp_tx_valid;
    wire [7:0]  w_ptp_tx_byte;
    wire        w_ptp_tx_last;
    wire        w_ptp_tx_ready;
    wire [31:0] w_ptp_seq;
    wire [63:0] w_ptp_last_time;
    wire [15:0] w_ptp_count;
    assign ptp_sync_count = w_ptp_count;

    // --- canfd_controller ---
    wire [8:0]  w_canfd_tec;
    wire [7:0]  w_canfd_rec;
    wire [1:0]  w_canfd_bus_state;
    wire        w_canfd_rx_out_valid;
    wire [28:0] w_canfd_rx_out_id;
    wire [3:0]  w_canfd_rx_out_dlc;
    wire [63:0] w_canfd_rx_out_data;
    wire        w_canfd_rx_out_ready;
    wire        w_canfd_tx_frame_done;

    // --- can_odometry_decoder ---
    wire        w_odo_valid;
    wire [15:0] w_wheel_speed_mmps;
    wire signed [15:0] w_steer_mdeg;
    wire signed [15:0] w_odo_yaw_rate_mdps;
    wire [15:0] w_wheel_fl, w_wheel_fr, w_wheel_rl, w_wheel_rr;

    // --- radar_interface ---
    wire        w_radar_out_valid;
    wire signed [15:0] w_radar_range_cm, w_radar_vel_cms, w_radar_az_mdeg;
    wire [15:0] w_radar_rcs;
    wire [7:0]  w_radar_conf;
    wire [31:0] w_radar_ts_us;
    wire [$clog2(RADAR_FIFO_DEPTH+1)-1:0] w_radar_fifo_count;

    // --- ultrasonic_interface ---
    wire        w_ultra_frame_valid;
    wire [191:0] w_ultra_dist_vec;
    wire [11:0]  w_ultra_health;
    wire [15:0]  w_ultra_frame_count, w_ultra_error_count;

    // --- ethernet_controller ---
    wire        w_eth_rx_payload_valid;
    wire [7:0]  w_eth_rx_payload_byte;
    wire        w_eth_rx_payload_last;
    wire        w_eth_frame_ok, w_eth_frame_err;
    wire [15:0] w_eth_ethertype;
    wire [1:0]  w_eth_frame_type;
    wire [1:0]  w_eth_mac_type;
    wire [10:0] w_eth_byte_count;

    // --- lidar_interface ---
    wire        w_lidar_out_valid;
    wire signed [31:0] w_lidar_x, w_lidar_y, w_lidar_z;
    wire [15:0] w_lidar_length, w_lidar_width, w_lidar_height;
    wire [7:0]  w_lidar_class, w_lidar_conf;
    wire [15:0] w_lidar_ts_lo;

    // --- cam_detection_receiver ---
    wire        w_camdet_out_valid;
    wire [15:0] w_camdet_class_id, w_camdet_confidence;
    wire [15:0] w_camdet_bbox_x, w_camdet_bbox_y;
    wire [15:0] w_camdet_bbox_w, w_camdet_bbox_h;
    wire [31:0] w_camdet_ts_us;
    wire [7:0]  w_camdet_camera_id;
    wire        w_camdet_wr_ready;

    // =========================================================================
    // Internal wires — Layer 2 + Layer 3
    // =========================================================================

    // sensor_sync
    wire        w_window_open, w_window_release;
    wire [31:0] w_window_center;
    wire [3:0]  w_sensors_ready;
    wire [3:0]  w_sensor_sync_stale;
    assign window_release       = w_window_release;
    assign sensor_stale_layer1  = w_sensor_sync_stale;

    // coord_transform
    wire        w_ct_out_valid;
    wire [1:0]  w_ct_out_sensor_id;
    wire signed [31:0] w_ct_body_x, w_ct_body_y, w_ct_body_z;

    // ego_motion_estimator
    wire        w_ego_valid;
    wire signed [31:0] w_ego_vx_mmps, w_ego_vy_mmps, w_ego_yaw_rate_mdps;
    wire [1:0]  w_ego_stale;

    // object_tracker
    wire        w_ot_det_matched, w_ot_det_allocated, w_ot_det_dropped;
    wire [3:0]  w_ot_det_sensor_mask;
    wire [7:0]  w_ot_num_active;
    wire        w_ot_q_valid;
    wire [15:0] w_ot_q_track_id;
    wire signed [31:0] w_ot_q_x_mm, w_ot_q_y_mm;
    wire signed [31:0] w_ot_q_vx_mm_per_update, w_ot_q_vy_mm_per_update;
    wire [7:0]  w_ot_q_age;
    wire [3:0]  w_ot_q_sensor_mask;
    wire [7:0]  w_ot_q_class_id, w_ot_q_confidence;

    // lane_fusion
    wire        w_lf_fused_valid;
    wire signed [31:0] w_lf_left_mm, w_lf_right_mm;
    wire signed [31:0] w_lf_lane_width_mm, w_lf_center_offset_mm;
    wire [1:0]  w_lf_fusion_source;
    wire [1:0]  w_lf_sensor_stale;

    // plausibility_checker
    wire        w_pc_check_done, w_pc_check_ok;
    wire [2:0]  w_pc_violation;
    wire [7:0]  w_pc_asil_degrade;
    wire [15:0] w_pc_total_checks, w_pc_total_violations;

    // ttc_calculator
    wire        w_ttc_valid;
    wire [15:0] w_ttc_track_id;
    wire        w_ttc_approaching, w_ttc_warning, w_ttc_prepare, w_ttc_brake;

    // aeb_controller
    wire [15:0] w_aeb_active_threat_id;
    wire [15:0] w_aeb_brake_hold_ms;

    // ldw_lka_controller
    wire [1:0]  w_lk_departure_direction;

    // safe_state_controller
    wire [15:0] w_ss_latched_faults;

    // =========================================================================
    // 1. mipi_csi2_rx — camera byte stream parser
    // =========================================================================
    mipi_csi2_rx u_mipi (
        .clk(clk),
        .rst_n(rst_n),
        .byte_valid(cam_byte_valid),
        .byte_data(cam_byte_data),
        .frame_start(cam_frame_start),
        .frame_end(cam_frame_end),
        .line_start(cam_line_start),
        .line_end(cam_line_end),
        .last_data_type(),
        .last_word_count(),
        .last_virtual_channel(),
        .pixel_valid(cam_pixel_valid),
        .pixel_byte(cam_pixel_byte),
        .pixel_last(cam_pixel_last),
        .frame_count(),
        .line_count(),
        .error_count()
    );

    // =========================================================================
    // 2. imu_interface — SPI-framed 6-DOF
    // =========================================================================
    imu_interface u_imu (
        .clk(clk),
        .rst_n(rst_n),
        .spi_byte_valid(imu_spi_byte_valid),
        .spi_byte(imu_spi_byte),
        .spi_frame_end(imu_spi_frame_end),
        .imu_valid(w_imu_valid),
        .accel_x_mg(w_accel_x_mg),
        .accel_y_mg(w_accel_y_mg),
        .accel_z_mg(w_accel_z_mg),
        .gyro_x_mdps(w_gyro_x_mdps),
        .gyro_y_mdps(w_gyro_y_mdps),
        .gyro_z_mdps(w_gyro_z_mdps),
        .frame_count(),
        .error_count()
    );

    // =========================================================================
    // 3. gnss_interface — time base + fix
    // =========================================================================
    gnss_interface u_gnss (
        .clk(clk),
        .rst_n(rst_n),
        .pps_in(gnss_pps),
        .time_set_valid(gnss_time_set_valid),
        .time_set_us(gnss_time_set_us),
        .fix_set_valid(gnss_fix_set_valid),
        .fix_valid_in(gnss_fix_valid_in),
        .lat_mdeg_in(gnss_lat_mdeg_in),
        .lon_mdeg_in(gnss_lon_mdeg_in),
        .time_us(w_master_time_us),
        .pps_valid(w_pps_valid),
        .pps_time_us(w_pps_time_us),
        .pps_count(w_pps_count),
        .gps_fix_valid(w_gps_fix_valid),
        .lat_mdeg(w_gnss_lat_mdeg),
        .lon_mdeg(w_gnss_lon_mdeg)
    );

    // =========================================================================
    // 4. ptp_clock_sync — Sync-frame grandmaster
    // =========================================================================
    ptp_clock_sync u_ptp (
        .clk(clk),
        .rst_n(rst_n),
        .master_time_us(w_master_time_us),
        .tick_1ms(tick_1ms),
        .tx_valid(w_ptp_tx_valid),
        .tx_byte_in(w_ptp_tx_byte),
        .tx_last(w_ptp_tx_last),
        .tx_ready(w_ptp_tx_ready),
        .sync_sequence(w_ptp_seq),
        .last_sync_time_us(w_ptp_last_time),
        .sync_count(w_ptp_count)
    );

    // =========================================================================
    // 5. canfd_controller + can_odometry_decoder
    // =========================================================================
    canfd_controller u_canfd (
        .clk(clk),
        .rst_n(rst_n),
        .tx_success(1'b0),
        .tx_error(1'b0),
        .rx_error(1'b0),
        .bus_off_recovery(1'b0),
        .tec(w_canfd_tec),
        .rec(w_canfd_rec),
        .bus_state(w_canfd_bus_state),
        .rx_frame_valid(can_rx_frame_valid),
        .rx_frame_id(can_rx_frame_id),
        .rx_frame_dlc(can_rx_frame_dlc),
        .rx_frame_data(can_rx_frame_data),
        .rx_frame_ready(can_rx_frame_ready),
        .rx_out_valid(w_canfd_rx_out_valid),
        .rx_out_id(w_canfd_rx_out_id),
        .rx_out_dlc(w_canfd_rx_out_dlc),
        .rx_out_data(w_canfd_rx_out_data),
        .rx_out_ready(w_canfd_rx_out_ready),
        .tx_frame_valid(1'b0),
        .tx_frame_id(29'd0),
        .tx_frame_dlc(4'd0),
        .tx_frame_data(64'd0),
        .tx_frame_ready(),
        .tx_frame_done(w_canfd_tx_frame_done)
    );

    can_odometry_decoder u_can_odo (
        .clk(clk),
        .rst_n(rst_n),
        .rx_out_valid(w_canfd_rx_out_valid),
        .rx_out_id(w_canfd_rx_out_id),
        .rx_out_dlc(w_canfd_rx_out_dlc),
        .rx_out_data(w_canfd_rx_out_data),
        .rx_out_ready(w_canfd_rx_out_ready),
        .odo_valid(w_odo_valid),
        .wheel_speed_mmps(w_wheel_speed_mmps),
        .steer_mdeg(w_steer_mdeg),
        .odo_yaw_rate_mdps(w_odo_yaw_rate_mdps),
        .wheel_fl_mmps(w_wheel_fl),
        .wheel_fr_mmps(w_wheel_fr),
        .wheel_rl_mmps(w_wheel_rl),
        .wheel_rr_mmps(w_wheel_rr),
        .wheel_frame_count(),
        .steering_frame_count(),
        .ignored_frame_count()
    );

    // =========================================================================
    // 6. radar_interface — SPI frame decode → FIFO → det_arbiter
    // =========================================================================
    wire w_arb_rad_ack;

    radar_interface #(.FIFO_DEPTH(RADAR_FIFO_DEPTH)) u_radar (
        .clk(clk),
        .rst_n(rst_n),
        .spi_byte_valid(radar_spi_byte_valid),
        .spi_byte(radar_spi_byte),
        .spi_frame_end(radar_spi_frame_end),
        .out_valid(w_radar_out_valid),
        .out_ready(w_arb_rad_ack),             // popped when arbiter wins radar
        .out_range_cm(w_radar_range_cm),
        .out_velocity_cms(w_radar_vel_cms),
        .out_azimuth_mdeg(w_radar_az_mdeg),
        .out_rcs_dbsm(w_radar_rcs),
        .out_confidence(w_radar_conf),
        .out_timestamp_us(w_radar_ts_us),
        .fifo_count(w_radar_fifo_count),
        .fifo_full(),
        .fifo_empty(),
        .frame_count(),
        .error_count(),
        .total_dropped()
    );

    // =========================================================================
    // 7. ultrasonic_interface — UART frame decode
    // =========================================================================
    ultrasonic_interface u_ultra (
        .clk(clk),
        .rst_n(rst_n),
        .rx_valid(ultra_rx_valid),
        .rx_byte(ultra_rx_byte),
        .frame_valid(w_ultra_frame_valid),
        .distance_mm_vec(w_ultra_dist_vec),
        .sensor_health(w_ultra_health),
        .frame_count(w_ultra_frame_count),
        .error_count(w_ultra_error_count)
    );

    // =========================================================================
    // 8. ethernet_controller + lidar_interface
    // =========================================================================
    ethernet_controller u_eth (
        .clk(clk),
        .rst_n(rst_n),
        .rx_valid(eth_rx_valid),
        .rx_byte(eth_rx_byte),
        .rx_last(eth_rx_last),
        .frame_ok(w_eth_frame_ok),
        .frame_err(w_eth_frame_err),
        .ethertype(w_eth_ethertype),
        .frame_type(w_eth_frame_type),
        .mac_type(w_eth_mac_type),
        .byte_count(w_eth_byte_count),
        .rx_payload_valid(w_eth_rx_payload_valid),
        .rx_payload_byte(w_eth_rx_payload_byte),
        .rx_payload_last(w_eth_rx_payload_last),
        .tx_valid(w_ptp_tx_valid),
        .tx_byte_in(w_ptp_tx_byte),
        .tx_last(w_ptp_tx_last),
        .tx_ready(w_ptp_tx_ready),
        .tx_out_valid(eth_tx_out_valid),
        .tx_out_byte(eth_tx_out_byte),
        .tx_out_last(eth_tx_out_last)
    );

    wire w_arb_lid_ack;

    lidar_interface #(.FIFO_DEPTH(LIDAR_FIFO_DEPTH)) u_lidar (
        .clk(clk),
        .rst_n(rst_n),
        .rx_payload_valid(w_eth_rx_payload_valid),
        .rx_payload_byte(w_eth_rx_payload_byte),
        .rx_payload_last(w_eth_rx_payload_last),
        .out_valid(w_lidar_out_valid),
        .out_ready(w_arb_lid_ack),             // popped when arbiter wins lidar
        .out_x_mm(w_lidar_x),
        .out_y_mm(w_lidar_y),
        .out_z_mm(w_lidar_z),
        .out_length_mm(w_lidar_length),
        .out_width_mm(w_lidar_width),
        .out_height_mm(w_lidar_height),
        .out_class_id(w_lidar_class),
        .out_confidence(w_lidar_conf),
        .out_timestamp_us_lo(w_lidar_ts_lo),
        .fifo_count(),
        .fifo_full(),
        .fifo_empty(),
        .frame_count(),
        .error_count(),
        .total_dropped()
    );

    // =========================================================================
    // 9. cam_detection_receiver — drained by det_arbiter
    // =========================================================================
    wire w_arb_cam_ack;
    wire camdet_rd_ready = w_arb_cam_ack;

    cam_detection_receiver #(.FIFO_DEPTH(CAMDET_FIFO_DEPTH)) u_camdet (
        .clk(clk),
        .rst_n(rst_n),
        .wr_valid(cam_det_valid),
        .wr_class_id(cam_det_class_id),
        .wr_confidence(cam_det_confidence),
        .wr_bbox_x(cam_det_bbox_x),
        .wr_bbox_y(cam_det_bbox_y),
        .wr_bbox_w(cam_det_bbox_w),
        .wr_bbox_h(cam_det_bbox_h),
        .wr_timestamp_us(cam_det_timestamp_us),
        .wr_camera_id(cam_det_camera_id),
        .wr_ready(w_camdet_wr_ready),
        .rd_valid(w_camdet_out_valid),
        .rd_ready(camdet_rd_ready),
        .rd_class_id(w_camdet_class_id),
        .rd_confidence(w_camdet_confidence),
        .rd_bbox_x(w_camdet_bbox_x),
        .rd_bbox_y(w_camdet_bbox_y),
        .rd_bbox_w(w_camdet_bbox_w),
        .rd_bbox_h(w_camdet_bbox_h),
        .rd_timestamp_us(w_camdet_ts_us),
        .rd_camera_id(w_camdet_camera_id),
        .fifo_count(),
        .fifo_full(),
        .fifo_empty(),
        .total_received(),
        .total_dropped()
    );

    // =========================================================================
    // 10. sensor_sync — timestamp alignment / presence watchdog
    //    sensor_valid[0]=camera (on cam_det pulse),
    //    sensor_valid[1]=radar FIFO has data,
    //    sensor_valid[2]=lidar FIFO has data,
    //    sensor_valid[3]=ultrasonic frame (on ultra_frame_valid).
    //    Timestamps all use the lower 32 bits of master_time_us.
    // =========================================================================
    wire [3:0]  ss_sensor_valid;
    wire [31:0] ss_time_us_curr = w_master_time_us[31:0];
    assign ss_sensor_valid[0] = cam_det_valid;
    assign ss_sensor_valid[1] = w_radar_out_valid;
    assign ss_sensor_valid[2] = w_lidar_out_valid;
    assign ss_sensor_valid[3] = w_ultra_frame_valid;

    sensor_sync u_sensor_sync (
        .clk(clk),
        .rst_n(rst_n),
        .sensor_valid(ss_sensor_valid),
        .s0_time_us(ss_time_us_curr),
        .s1_time_us(ss_time_us_curr),
        .s2_time_us(ss_time_us_curr),
        .s3_time_us(ss_time_us_curr),
        .window_open(w_window_open),
        .window_center(w_window_center),
        .sensors_ready(w_sensors_ready),
        .window_release(w_window_release),
        .sensor_stale(w_sensor_sync_stale)
    );

    // =========================================================================
    // 10b. det_arbiter — round-robin over camera / radar / lidar detections.
    //      The arbiter is purely combinatorial on its winner-pick; each cycle
    //      exactly one winning source advances through coord_transform.
    //
    //  Camera → (bbox_x, bbox_y, 0, class, conf[7:0])
    //  Radar  → (range_cm * 10, 0, 0, class=VEHICLE, confidence)
    //  LiDAR  → (x_mm, y_mm, z_mm, class, confidence)
    // =========================================================================
    localparam [7:0] CLASS_VEHICLE_V1 = 8'd1;

    // Camera field conversion
    wire signed [31:0] arb_cam_x = { {16{w_camdet_bbox_x[15]}}, w_camdet_bbox_x };
    wire signed [31:0] arb_cam_y = { {16{w_camdet_bbox_y[15]}}, w_camdet_bbox_y };

    // Radar field conversion: range (cm → mm) = range_cm * 10.
    //
    // An earlier revision used *12 (*8 + *4) and called it "≈ *10". That 20 %
    // error pushed a radar echo of a target at 10 m (1000 cm → 12 000 mm) out
    // of the ±2 m association gate against the camera's equivalent 10 000 mm
    // detection, so object_tracker fragmented a single physical target into
    // 3-4 independent tracks. Exact *10 = *8 + *2 (two shifts, one add) keeps
    // cam and radar on the same spatial frame so the tracker can fuse them.
    wire signed [31:0] arb_rad_cm_ext = { {16{w_radar_range_cm[15]}}, w_radar_range_cm };
    wire signed [31:0] arb_rad_range_mm = (arb_rad_cm_ext <<< 3) +
                                          (arb_rad_cm_ext <<< 1);

    // Detection arbiter instance
    wire        w_arb_out_valid;
    wire [1:0]  w_arb_out_sensor_id;
    wire signed [31:0] w_arb_out_x_mm, w_arb_out_y_mm, w_arb_out_z_mm;
    wire [7:0]  w_arb_out_class_id;
    wire [7:0]  w_arb_out_confidence;

    det_arbiter u_arb (
        .clk(clk),
        .rst_n(rst_n),

        // Camera source
        .cam_valid      (w_camdet_out_valid),
        .cam_x_mm       (arb_cam_x),
        .cam_y_mm       (arb_cam_y),
        .cam_z_mm       (32'sd0),
        .cam_class_id   (w_camdet_class_id[7:0]),
        .cam_confidence (w_camdet_confidence[7:0]),
        .cam_ack        (w_arb_cam_ack),

        // Radar source
        .rad_valid      (w_radar_out_valid),
        .rad_x_mm       (arb_rad_range_mm),
        .rad_y_mm       (32'sd0),
        .rad_z_mm       (32'sd0),
        .rad_class_id   (CLASS_VEHICLE_V1),
        .rad_confidence (w_radar_conf),
        .rad_ack        (w_arb_rad_ack),

        // LiDAR source
        .lid_valid      (w_lidar_out_valid),
        .lid_x_mm       (w_lidar_x),
        .lid_y_mm       (w_lidar_y),
        .lid_z_mm       (w_lidar_z),
        .lid_class_id   (w_lidar_class),
        .lid_confidence (w_lidar_conf),
        .lid_ack        (w_arb_lid_ack),

        // Merged output
        .out_valid      (w_arb_out_valid),
        .out_sensor_id  (w_arb_out_sensor_id),
        .out_x_mm       (w_arb_out_x_mm),
        .out_y_mm       (w_arb_out_y_mm),
        .out_z_mm       (w_arb_out_z_mm),
        .out_class_id   (w_arb_out_class_id),
        .out_confidence (w_arb_out_confidence)
    );

    // =========================================================================
    // 11. coord_transform — body-frame conversion fed by det_arbiter
    // =========================================================================
    coord_transform u_coord (
        .clk(clk),
        .rst_n(rst_n),
        .det_valid(w_arb_out_valid),
        .det_sensor_id(w_arb_out_sensor_id),
        .det_x_mm(w_arb_out_x_mm),
        .det_y_mm(w_arb_out_y_mm),
        .det_z_mm(w_arb_out_z_mm),
        .out_valid(w_ct_out_valid),
        .out_sensor_id(w_ct_out_sensor_id),
        .out_x_mm(w_ct_body_x),
        .out_y_mm(w_ct_body_y),
        .out_z_mm(w_ct_body_z),
        .cal_we(cal_we),
        .cal_addr(cal_addr),
        .cal_wdata(cal_wdata)
    );

    // =========================================================================
    // 12. ego_motion_estimator — IMU + wheel odometry
    // =========================================================================
    ego_motion_estimator u_ego (
        .clk(clk),
        .rst_n(rst_n),
        .imu_valid(w_imu_valid),
        .accel_x_mg(w_accel_x_mg),
        .accel_y_mg(w_accel_y_mg),
        .gyro_z_mdps(w_gyro_z_mdps),
        .odo_valid(w_odo_valid),
        .wheel_speed_mmps(w_wheel_speed_mmps),
        .steer_mdeg(w_steer_mdeg),
        .odo_yaw_rate_mdps(w_odo_yaw_rate_mdps),
        .ego_valid(w_ego_valid),
        .ego_vx_mmps(w_ego_vx_mmps),
        .ego_vy_mmps(w_ego_vy_mmps),
        .ego_yaw_rate_mdps(w_ego_yaw_rate_mdps),
        .sensor_stale(w_ego_stale)
    );

    // =========================================================================
    // 13. object_tracker — fed by coord_transform; ticks at 1 kHz
    //     The query interface is swept by a small counter driven by tick_1ms
    //     so ttc_calculator can walk every active slot.
    // =========================================================================
    reg [2:0] ot_query_idx;
    always @(posedge clk) begin
        if (!rst_n)         ot_query_idx <= 3'd0;
        else if (tick_1ms)  ot_query_idx <= ot_query_idx + 3'd1;
    end

    // Class + confidence ride along with the detection through the 2-cycle
    // coord_transform pipeline.  det_arbiter's out_* is already registered
    // once, so we only need one more stage to align with coord_transform.out_*.
    reg [7:0] ct_class_d1, ct_conf_d1;
    always @(posedge clk) begin
        if (!rst_n) begin
            ct_class_d1 <= 8'd0;
            ct_conf_d1  <= 8'd0;
        end else begin
            ct_class_d1 <= w_arb_out_class_id;
            ct_conf_d1  <= w_arb_out_confidence;
        end
    end
    wire [7:0] ct_class_d2 = ct_class_d1;
    wire [7:0] ct_conf_d2  = ct_conf_d1;

    object_tracker #(.NUM_TRACKS(OBJTR_NUM_TRACKS)) u_tracker (
        .clk(clk),
        .rst_n(rst_n),
        .det_valid(w_ct_out_valid),
        .det_sensor_id(w_ct_out_sensor_id),
        .det_x_mm(w_ct_body_x),
        .det_y_mm(w_ct_body_y),
        .det_class_id(ct_class_d2),
        .det_confidence(ct_conf_d2),
        .tick_valid(tick_1ms),
        .det_matched(w_ot_det_matched),
        .det_allocated(w_ot_det_allocated),
        .det_dropped(w_ot_det_dropped),
        .det_sensor_mask(w_ot_det_sensor_mask),
        .num_active_tracks(w_ot_num_active),
        .query_idx(ot_query_idx),
        .query_valid(w_ot_q_valid),
        .query_track_id(w_ot_q_track_id),
        .query_x_mm(w_ot_q_x_mm),
        .query_y_mm(w_ot_q_y_mm),
        .query_vx_mm_per_update(w_ot_q_vx_mm_per_update),
        .query_vy_mm_per_update(w_ot_q_vy_mm_per_update),
        .query_age(w_ot_q_age),
        .query_sensor_mask(w_ot_q_sensor_mask),
        .query_class_id(w_ot_q_class_id),
        .query_confidence(w_ot_q_confidence)
    );

    // =========================================================================
    // 14. lane_fusion — camera lane detections + HD map
    //     A camera detection with class == CLASS_LANE drives cam_valid;
    //     bbox_x / bbox_w are reinterpreted as left_mm / right_mm (stand-in
    //     mapping — a real implementation would use separate lane regs).
    // =========================================================================
    localparam [15:0] CLASS_LANE_VAL = 16'd4;
    wire lf_cam_valid = w_camdet_out_valid &&
                        (w_camdet_class_id == CLASS_LANE_VAL);
    wire signed [31:0] lf_cam_left  = { {16{w_camdet_bbox_x[15]}}, w_camdet_bbox_x };
    wire signed [31:0] lf_cam_right = { {16{w_camdet_bbox_w[15]}}, w_camdet_bbox_w };

    lane_fusion u_lane (
        .clk(clk),
        .rst_n(rst_n),
        .cam_valid(lf_cam_valid),
        .cam_left_mm(lf_cam_left),
        .cam_right_mm(lf_cam_right),
        .cam_confidence(w_camdet_confidence[7:0]),
        .map_valid(map_lane_valid),
        .map_left_mm(map_left_mm),
        .map_right_mm(map_right_mm),
        .fused_valid(w_lf_fused_valid),
        .fused_left_mm(w_lf_left_mm),
        .fused_right_mm(w_lf_right_mm),
        .fused_lane_width_mm(w_lf_lane_width_mm),
        .fused_center_offset_mm(w_lf_center_offset_mm),
        .fusion_source(w_lf_fusion_source),
        .sensor_stale(w_lf_sensor_stale)
    );

    // =========================================================================
    // 15. plausibility_checker — fires on each object_tracker match/alloc
    //
    // pc_sensor_mask must reflect which sensors actively saw the current
    // object.  object_tracker.det_sensor_mask exposes the affected track's
    // accumulated sensor bitmask aligned with det_matched/det_allocated,
    // giving plausibility the true per-object mask (cam/radar/lidar bits).
    //
    // Ultrasonic detections do not flow through det_arbiter/object_tracker
    // (sensor_id encoding is cam/radar/lidar only), so the US bit is still
    // sourced from a short recent-activity window on w_ultra_frame_valid.
    // The window must exceed the arbiter→tracker pipeline depth (~8-10 cyc)
    // but stay shorter than the cadence of a real US frame so an unrelated
    // prior frame cannot bleed into a new check.
    // =========================================================================
    localparam integer US_RECENT_CYCLES = 16;
    reg [4:0] us_recent_cnt;

    always @(posedge clk) begin
        if (!rst_n) begin
            us_recent_cnt <= 5'd0;
        end else begin
            if (w_ultra_frame_valid)     us_recent_cnt <= US_RECENT_CYCLES[4:0];
            else if (us_recent_cnt != 0) us_recent_cnt <= us_recent_cnt - 5'd1;
        end
    end

    wire pc_check_valid = w_ot_det_matched || w_ot_det_allocated;
    // Merge tracker's accumulated mask (bits [2:0] = cam/radar/lidar) with
    // the ultrasonic recent-activity bit ([3]) to cover PROXIMITY rules.
    wire [3:0] pc_sensor_mask =
        {1'b0, w_ot_det_sensor_mask[2:0]} |
        {(us_recent_cnt != 5'd0), 3'b000};

    plausibility_checker u_plaus (
        .clk(clk),
        .rst_n(rst_n),
        .check_valid(pc_check_valid),
        .check_class_id(ct_class_d2),
        .check_sensor_mask(pc_sensor_mask),
        .check_confidence(ct_conf_d2),
        .check_done(w_pc_check_done),
        .check_ok(w_pc_check_ok),
        .check_violation(w_pc_violation),
        .asil_degrade(w_pc_asil_degrade),
        .total_checks(w_pc_total_checks),
        .total_violations(w_pc_total_violations)
    );

    // =========================================================================
    // 16. ttc_calculator — sweeps tracks via query_idx.
    //     Range is the Manhattan approximation |x| + |y| (cheap, no sqrt).
    //
    //     ttc_calculator's contract: obj_closure_mms is the time-derivative of
    //     range (dR/dt). Negative ⇒ range shrinking ⇒ object approaching.
    //     Positive ⇒ range growing ⇒ not approaching (TTC = ∞).
    //
    //     For a Manhattan range R = |x|+|y|:
    //         dR/dt = sign(x)·vx + sign(y)·vy
    //
    //     object_tracker's query_v[xy]_mm_per_update is the detection-to-
    //     detection position delta expressed in ego body-frame, so it already
    //     encodes the relative motion (if ego moves forward toward a static
    //     target, that target's body-frame vx goes negative). No separate ego
    //     contribution is added here — adding ego_vx on top would double-count
    //     the motion that is already baked into the tracked vx. A dedicated
    //     ego-compensation pass (needed when object_tracker gains true
    //     world-frame tracking) belongs inside object_tracker, not here.
    //
    //     Scale: mm/update → mm/s via <<< 7 (×128, ~100 Hz fail-safe).
    // =========================================================================
    wire signed [31:0] q_abs_x = w_ot_q_x_mm[31] ? -w_ot_q_x_mm : w_ot_q_x_mm;
    wire signed [31:0] q_abs_y = w_ot_q_y_mm[31] ? -w_ot_q_y_mm : w_ot_q_y_mm;
    wire signed [31:0] q_range_mm = q_abs_x + q_abs_y;

    // Sign of x/y (+1 if positive/zero, -1 if negative) as signed 2-bit
    wire signed [1:0] q_sign_x = w_ot_q_x_mm[31] ? -2'sd1 : 2'sd1;
    wire signed [1:0] q_sign_y = w_ot_q_y_mm[31] ? -2'sd1 : 2'sd1;

    wire signed [31:0] obj_vx_mmps = w_ot_q_vx_mm_per_update <<< 7;
    wire signed [31:0] obj_vy_mmps = w_ot_q_vy_mm_per_update <<< 7;

    // dR/dt = sign(x)·vx + sign(y)·vy  (matches ttc_calculator's convention:
    // negative ⇒ approaching, positive ⇒ receding)
    wire signed [31:0] q_closure_mms =
        (q_sign_x * obj_vx_mmps) + (q_sign_y * obj_vy_mmps);

    ttc_calculator u_ttc (
        .clk(clk),
        .rst_n(rst_n),
        .obj_valid(w_ot_q_valid && tick_1ms),
        .obj_track_id(w_ot_q_track_id),
        .obj_range_mm(q_range_mm),
        .obj_closure_mms(q_closure_mms),
        .ttc_valid(w_ttc_valid),
        .ttc_track_id(w_ttc_track_id),
        .ttc_approaching(w_ttc_approaching),
        .ttc_warning(w_ttc_warning),
        .ttc_prepare(w_ttc_prepare),
        .ttc_brake(w_ttc_brake)
    );

    // =========================================================================
    // 17. aeb_controller
    // =========================================================================
    aeb_controller u_aeb (
        .clk(clk),
        .rst_n(rst_n),
        .ttc_valid(w_ttc_valid),
        .ttc_track_id(w_ttc_track_id),
        .ttc_warning(w_ttc_warning),
        .ttc_prepare(w_ttc_prepare),
        .ttc_brake(w_ttc_brake),
        .tick_1ms(tick_1ms),
        .brake_level(brake_level),
        .brake_active(brake_active),
        .target_decel_mms2(target_decel_mms2),
        .alert_driver(alert_driver),
        .active_threat_id(w_aeb_active_threat_id),
        .brake_hold_ms(w_aeb_brake_hold_ms)
    );

    // =========================================================================
    // 18. ldw_lka_controller
    // =========================================================================
    ldw_lka_controller u_ldw_lka (
        .clk(clk),
        .rst_n(rst_n),
        .lane_valid(w_lf_fused_valid),
        .center_offset_mm(w_lf_center_offset_mm),
        .lane_width_mm(w_lf_lane_width_mm),
        .fusion_source(w_lf_fusion_source),
        .ldw_warning(ldw_warning),
        .lka_active(lka_active),
        .steering_torque_mnm(steering_torque_mnm),
        .departure_direction(w_lk_departure_direction)
    );

    // =========================================================================
    // 19. safe_state_controller — fault aggregation
    // =========================================================================
    localparam [7:0] ASIL_REJECT_CODE = 8'hFF;

    // Registered view of plausibility reject so the comparator is stable
    reg plaus_rejected;
    always @(posedge clk) begin
        if (!rst_n)            plaus_rejected <= 1'b0;
        else if (w_pc_check_done)
            plaus_rejected <= (w_pc_asil_degrade == ASIL_REJECT_CODE);
    end

    wire [7:0] critical_faults = {
        5'd0,
        (&w_sensor_sync_stale),                  // [2] all 4 sensors stale
        plaus_rejected,                           // [1] plausibility reject
        (w_canfd_bus_state == 2'b10)              // [0] canfd BUS_OFF
    };

    wire [7:0] warning_faults = {
        5'd0,
        (|w_ego_stale),                           // [2] ego IMU or odo stale
        (|w_sensor_sync_stale),                   // [1] any L1 source stale
        (w_canfd_bus_state == 2'b01)              // [0] canfd ERROR_PASSIVE
    };

    safe_state_controller u_safestate (
        .clk(clk),
        .rst_n(rst_n),
        .critical_faults(critical_faults),
        .warning_faults(warning_faults),
        .tick_1ms(tick_1ms),
        .operator_reset(operator_reset),
        .safe_state(safe_state),
        .alert_driver(),                          // separate AEB alert drives top port
        .limit_speed(limit_speed),
        .max_speed_kmh(max_speed_kmh),
        .mrc_pull_over(mrc_pull_over),
        .latched_faults(w_ss_latched_faults)
    );

endmodule
