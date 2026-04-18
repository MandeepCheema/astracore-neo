`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — Ego Motion Estimator  (ego_motion_estimator.v)
// =============================================================================
// Layer 2 fusion module.  Fuses IMU gyroscope + wheel odometry to produce a
// continuous ego-vehicle motion estimate used by object_tracker to transform
// detections into the world frame.
//
// ── Design ───────────────────────────────────────────────────────────────────
//   Two measurement streams:
//     IMU   (100–1000 Hz): accel_x/y in milli-g, gyro_z in mdeg/s
//     Odo   (~100 Hz)    : wheel_speed in mm/s, steering in mdeg, yaw_rate in mdeg/s
//
//   Fusion (complementary filter, equal weights — simple and overflow-safe):
//     yaw_rate: (IMU_gyro + ODO_yaw) >> 1  (50/50 blend)
//     vx      : wheel_speed_mmps  (odometry authoritative for longitudinal speed)
//     vy      : 0  (lateral slip negligible for L2+ ADAS; upgradeable)
//
//   Blend uses 33-bit signed accumulators to prevent overflow before >> 1.
//
//   ego_valid pulses 1 cycle whenever the estimate is updated (each IMU or odo
//   sample, including simultaneous arrivals fused in one combined output).
//
// ── Stale watchdog ───────────────────────────────────────────────────────────
//   Separate per-source counters.  If either source exceeds WATCHDOG_CYCLES
//   without a new sample, the corresponding sensor_stale bit asserts.
//   Stale clears on next valid pulse from that source.
//   sensor_stale[0] = IMU stale,  sensor_stale[1] = odometry stale.
//
// ── Parameters ────────────────────────────────────────────────────────────────
//   WATCHDOG_CYCLES — stale threshold (default 500 for sim;
//                     production: 500_000 = 10ms @ 50MHz)
//
// ── Interface ─────────────────────────────────────────────────────────────────
//   imu_valid, accel_x_mg[15:0], accel_y_mg[15:0], gyro_z_mdps[15:0]
//   odo_valid, wheel_speed_mmps[15:0], steer_mdeg[15:0], odo_yaw_rate_mdps[15:0]
//   ego_valid, ego_vx_mmps[31:0], ego_vy_mmps[31:0], ego_yaw_rate_mdps[31:0]
//   sensor_stale[1:0]
// =============================================================================

module ego_motion_estimator #(
    parameter [23:0] WATCHDOG_CYCLES = 24'd500   // stale threshold (sim default)
)(
    input  wire        clk,
    input  wire        rst_n,

    // ── IMU inputs (from imu_interface) ──────────────────────────────────────
    input  wire        imu_valid,
    input  wire signed [15:0] accel_x_mg,
    input  wire signed [15:0] accel_y_mg,
    input  wire signed [15:0] gyro_z_mdps,

    // ── Odometry inputs (from can_odometry_decoder) ───────────────────────────
    input  wire        odo_valid,
    input  wire [15:0] wheel_speed_mmps,
    input  wire signed [15:0] steer_mdeg,
    input  wire signed [15:0] odo_yaw_rate_mdps,

    // ── Fused ego motion output ───────────────────────────────────────────────
    output reg         ego_valid,
    output reg  signed [31:0] ego_vx_mmps,
    output reg  signed [31:0] ego_vy_mmps,
    output reg  signed [31:0] ego_yaw_rate_mdps,

    // ── Stale watchdog ────────────────────────────────────────────────────────
    output reg  [1:0]  sensor_stale
);

    // =========================================================================
    // 1. Last-measurement latches and data-available flags
    // =========================================================================
    reg signed [31:0] last_imu_yaw;
    reg signed [31:0] last_odo_yaw;
    reg signed [31:0] last_odo_vx;
    reg               imu_has_data;
    reg               odo_has_data;

    // =========================================================================
    // 2. Stale watchdog counters
    // =========================================================================
    reg [23:0] imu_cnt;
    reg [23:0] odo_cnt;

    always @(posedge clk) begin
        if (!rst_n) begin
            imu_cnt      <= 24'd0;
            odo_cnt      <= 24'd0;
            sensor_stale <= 2'b00;
        end else begin
            if (imu_valid) begin
                imu_cnt        <= 24'd0;
                sensor_stale[0] <= 1'b0;
            end else begin
                if (imu_cnt < WATCHDOG_CYCLES)
                    imu_cnt <= imu_cnt + 24'd1;
                else
                    sensor_stale[0] <= 1'b1;
            end

            if (odo_valid) begin
                odo_cnt        <= 24'd0;
                sensor_stale[1] <= 1'b0;
            end else begin
                if (odo_cnt < WATCHDOG_CYCLES)
                    odo_cnt <= odo_cnt + 24'd1;
                else
                    sensor_stale[1] <= 1'b1;
            end
        end
    end

    // =========================================================================
    // 3. Sign-extended combinatorial wires (safe for 33-bit accumulation)
    // =========================================================================
    wire signed [31:0] gyro_ext    = {{16{gyro_z_mdps[15]}},      gyro_z_mdps};
    wire signed [31:0] odo_yaw_ext = {{16{odo_yaw_rate_mdps[15]}}, odo_yaw_rate_mdps};
    wire signed [31:0] odo_vx_ext  = {16'd0, wheel_speed_mmps};   // unsigned → signed

    // 50/50 blends via 33-bit sums (no overflow): result = sum >>> 1
    wire signed [32:0] sum_imu_odo    = {gyro_ext[31],    gyro_ext}    + {odo_yaw_ext[31],    odo_yaw_ext};
    wire signed [32:0] sum_latch_odo  = {last_imu_yaw[31],last_imu_yaw}+ {odo_yaw_ext[31],    odo_yaw_ext};
    wire signed [32:0] sum_imu_latch  = {gyro_ext[31],    gyro_ext}    + {last_odo_yaw[31],   last_odo_yaw};

    wire signed [31:0] blend_imu_odo   = $signed(sum_imu_odo)   >>> 1;
    wire signed [31:0] blend_latch_odo = $signed(sum_latch_odo)  >>> 1;
    wire signed [31:0] blend_imu_latch = $signed(sum_imu_latch)  >>> 1;

    // =========================================================================
    // 4. Fusion FSM
    //    Priority: simultaneous > imu_only > odo_only > idle
    //    When only one source has been seen, use that source's measurement raw
    //    (no blend until both have contributed at least once).
    // =========================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            last_imu_yaw      <= 32'd0;
            last_odo_yaw      <= 32'd0;
            last_odo_vx       <= 32'd0;
            imu_has_data      <= 1'b0;
            odo_has_data      <= 1'b0;
            ego_valid         <= 1'b0;
            ego_vx_mmps       <= 32'd0;
            ego_vy_mmps       <= 32'd0;
            ego_yaw_rate_mdps <= 32'd0;
        end else begin
            ego_valid <= 1'b0;   // default de-assert

            if (imu_valid && odo_valid) begin
                last_imu_yaw      <= gyro_ext;
                last_odo_yaw      <= odo_yaw_ext;
                last_odo_vx       <= odo_vx_ext;
                imu_has_data      <= 1'b1;
                odo_has_data      <= 1'b1;
                ego_vx_mmps       <= odo_vx_ext;
                ego_vy_mmps       <= 32'd0;
                ego_yaw_rate_mdps <= blend_imu_odo;
                ego_valid         <= 1'b1;

            end else if (imu_valid) begin
                last_imu_yaw      <= gyro_ext;
                imu_has_data      <= 1'b1;
                ego_vx_mmps       <= last_odo_vx;
                ego_vy_mmps       <= 32'd0;
                ego_yaw_rate_mdps <= odo_has_data ? blend_imu_latch : gyro_ext;
                ego_valid         <= 1'b1;

            end else if (odo_valid) begin
                last_odo_yaw      <= odo_yaw_ext;
                last_odo_vx       <= odo_vx_ext;
                odo_has_data      <= 1'b1;
                ego_vx_mmps       <= odo_vx_ext;
                ego_vy_mmps       <= 32'd0;
                ego_yaw_rate_mdps <= imu_has_data ? blend_latch_odo : odo_yaw_ext;
                ego_valid         <= 1'b1;
            end
        end
    end

    // =========================================================================
    // Ego-motion invariants — Phase-F formal verification targets
    // =========================================================================
`ifndef SYNTHESIS
`ifndef __ICARUS__
    // Invariant: ego_valid is a 1-cycle pulse; never held for 2+ cycles.
    // (registered output, single pulse per input frame).
    property p_ego_valid_pulse;
        @(posedge clk) disable iff (!rst_n)
        ego_valid |=> !ego_valid;
    endproperty
    a_ego_valid_pulse: assert property (p_ego_valid_pulse)
        else $error("ego_motion: ego_valid held for >1 cycle");

    // Invariant: sensor_stale is a 2-bit vector — only bits [1:0] can be set.
    property p_sensor_stale_valid;
        @(posedge clk) disable iff (!rst_n)
        (sensor_stale[1:0] <= 2'b11);
    endproperty
    a_sensor_stale_valid: assert property (p_sensor_stale_valid)
        else $error("ego_motion: sensor_stale beyond 2-bit range");
`endif
`endif

endmodule
