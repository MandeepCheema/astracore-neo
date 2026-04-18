`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — Plausibility Checker  (plausibility_checker.v)
// =============================================================================
// Layer 2 ASIL-D fusion safety module.  Gate-keeper for detections leaving the
// fusion pipeline and entering Layer 3 decision logic.  Enforces cross-sensor
// redundancy requirements per ISO 26262 ASIL decomposition rules and can
// degrade or reject detections that fail to meet them.
//
// ── Redundancy rules (from architecture spec) ────────────────────────────────
//   Class                 Required sensors              On violation
//   --------------------  ----------------------------  ---------------------
//   VEHICLE (collision)   Camera AND Radar              Degrade to ASIL-B
//   PEDESTRIAN (collis.)  Camera AND (Radar OR LiDAR)   Degrade to ASIL-B
//   PROXIMITY (<0.5m)     Ultrasonic AND Camera         Degrade to ASIL-B
//   LANE (departure)      Camera                        No redundancy required
//   Other classes         — no match — reject
//
//   Additional common check:
//     confidence < MIN_CONFIDENCE → VIO_LOW_CONF, ASIL-B
//
// ── Design ───────────────────────────────────────────────────────────────────
//   1-cycle pipeline: check_valid pulses with {class, sensor_mask, confidence}
//   → next clock, check_done pulses with the check result.
//
//   check_ok          — 1 if all rules satisfied, 0 otherwise
//   check_violation   — violation code (see localparams below)
//   asil_degrade      — 0x00 kept at ASIL-D, 0x01 degraded to ASIL-B, 0xFF reject
//
//   total_checks + total_violations are 16-bit saturating counters for health
//   monitoring / statistics readout.
//
// ── Sensor mask bit assignment ────────────────────────────────────────────────
//   Matches the object_tracker sensor_mask layout:
//     bit 0 = Camera     (cam_detection_receiver)
//     bit 1 = Radar      (radar_interface)
//     bit 2 = LiDAR      (lidar_interface)
//     bit 3 = Ultrasonic (ultrasonic_interface)
// =============================================================================

module plausibility_checker #(
    parameter [7:0] MIN_CONFIDENCE = 8'd64
)(
    input  wire        clk,
    input  wire        rst_n,

    // ── Check request ─────────────────────────────────────────────────────────
    input  wire        check_valid,
    input  wire [7:0]  check_class_id,
    input  wire [3:0]  check_sensor_mask,
    input  wire [7:0]  check_confidence,

    // ── Check result (registered, 1-cycle pulse 1 clk after check_valid) ──────
    output reg         check_done,
    output reg         check_ok,
    output reg  [2:0]  check_violation,
    output reg  [7:0]  asil_degrade,

    // ── Saturating statistics counters ────────────────────────────────────────
    output reg  [15:0] total_checks,
    output reg  [15:0] total_violations
);

    // =========================================================================
    // 1. Class IDs, sensor mask bits, violation codes, ASIL levels
    // =========================================================================
    localparam [7:0] CLASS_VEHICLE    = 8'd1;
    localparam [7:0] CLASS_PEDESTRIAN = 8'd2;
    localparam [7:0] CLASS_PROXIMITY  = 8'd3;
    localparam [7:0] CLASS_LANE       = 8'd4;

    localparam [3:0] S_CAM = 4'b0001;
    localparam [3:0] S_RAD = 4'b0010;
    localparam [3:0] S_LID = 4'b0100;
    localparam [3:0] S_US  = 4'b1000;

    localparam [2:0] VIO_NONE          = 3'd0;
    localparam [2:0] VIO_NO_REDUNDANCY = 3'd1;
    localparam [2:0] VIO_LOW_CONF      = 3'd2;
    localparam [2:0] VIO_UNKNOWN_CLASS = 3'd3;

    localparam [7:0] ASIL_D_KEEP = 8'h00;
    localparam [7:0] ASIL_B_DEG  = 8'h01;
    localparam [7:0] ASIL_REJECT = 8'hFF;

    // =========================================================================
    // 2. Combinatorial rule evaluation on the incoming request
    // =========================================================================
    wire has_cam = |(check_sensor_mask & S_CAM);
    wire has_rad = |(check_sensor_mask & S_RAD);
    wire has_lid = |(check_sensor_mask & S_LID);
    wire has_us  = |(check_sensor_mask & S_US);

    // Per-class redundancy satisfaction
    wire veh_ok = has_cam && has_rad;
    wire ped_ok = has_cam && (has_rad || has_lid);
    wire prox_ok = has_us  && has_cam;
    wire lane_ok = has_cam;

    // Combinatorial decision (evaluated only when check_valid)
    reg        next_ok;
    reg [2:0]  next_violation;
    reg [7:0]  next_asil;

    always @(*) begin
        next_ok        = 1'b0;
        next_violation = VIO_UNKNOWN_CLASS;
        next_asil      = ASIL_REJECT;

        if (check_confidence < MIN_CONFIDENCE) begin
            next_ok        = 1'b0;
            next_violation = VIO_LOW_CONF;
            next_asil      = ASIL_B_DEG;
        end else begin
            case (check_class_id)
                CLASS_VEHICLE: begin
                    if (veh_ok) begin
                        next_ok = 1'b1; next_violation = VIO_NONE;          next_asil = ASIL_D_KEEP;
                    end else begin
                        next_ok = 1'b0; next_violation = VIO_NO_REDUNDANCY; next_asil = ASIL_B_DEG;
                    end
                end
                CLASS_PEDESTRIAN: begin
                    if (ped_ok) begin
                        next_ok = 1'b1; next_violation = VIO_NONE;          next_asil = ASIL_D_KEEP;
                    end else begin
                        next_ok = 1'b0; next_violation = VIO_NO_REDUNDANCY; next_asil = ASIL_B_DEG;
                    end
                end
                CLASS_PROXIMITY: begin
                    if (prox_ok) begin
                        next_ok = 1'b1; next_violation = VIO_NONE;          next_asil = ASIL_D_KEEP;
                    end else begin
                        next_ok = 1'b0; next_violation = VIO_NO_REDUNDANCY; next_asil = ASIL_B_DEG;
                    end
                end
                CLASS_LANE: begin
                    if (lane_ok) begin
                        next_ok = 1'b1; next_violation = VIO_NONE;          next_asil = ASIL_D_KEEP;
                    end else begin
                        next_ok = 1'b0; next_violation = VIO_NO_REDUNDANCY; next_asil = ASIL_B_DEG;
                    end
                end
                default: begin
                    // Unknown class — reject outright
                    next_ok        = 1'b0;
                    next_violation = VIO_UNKNOWN_CLASS;
                    next_asil      = ASIL_REJECT;
                end
            endcase
        end
    end

    // =========================================================================
    // 3. Registered result + saturating statistics counters
    // =========================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            check_done       <= 1'b0;
            check_ok         <= 1'b0;
            check_violation  <= VIO_NONE;
            asil_degrade     <= ASIL_D_KEEP;
            total_checks     <= 16'd0;
            total_violations <= 16'd0;
        end else begin
            check_done <= 1'b0;   // default de-assert

            if (check_valid) begin
                check_done      <= 1'b1;
                check_ok        <= next_ok;
                check_violation <= next_violation;
                asil_degrade    <= next_asil;

                // Saturating increments
                if (total_checks != 16'hFFFF)
                    total_checks <= total_checks + 16'd1;
                if (!next_ok && total_violations != 16'hFFFF)
                    total_violations <= total_violations + 16'd1;
            end
        end
    end

    // =========================================================================
    // ASIL-D safety invariants (SVA) — Phase-F formal verification targets
    // =========================================================================
`ifndef SYNTHESIS
`ifndef __ICARUS__
    // Invariant 1: check_done follows check_valid by exactly 1 cycle.
    property p_check_done_follows_valid;
        @(posedge clk) disable iff (!rst_n)
        check_valid |=> check_done;
    endproperty
    a_check_done_follows_valid: assert property (p_check_done_follows_valid)
        else $error("plausibility: check_done did not follow check_valid by 1 cycle");

    // Invariant 2: check_ok=1 implies no violation code set.
    property p_ok_iff_no_violation;
        @(posedge clk) disable iff (!rst_n)
        (check_done && check_ok) |-> (check_violation == VIO_NONE);
    endproperty
    a_ok_iff_no_violation: assert property (p_ok_iff_no_violation)
        else $error("plausibility: check_ok=1 with non-zero violation code");

    // Invariant 3: asil_degrade is one of {0x00 keep, 0x01 degrade, 0xFF reject}.
    property p_asil_degrade_valid;
        @(posedge clk) disable iff (!rst_n)
        check_done |->
            (asil_degrade == 8'h00 || asil_degrade == 8'h01 || asil_degrade == 8'hFF);
    endproperty
    a_asil_degrade_valid: assert property (p_asil_degrade_valid)
        else $error("plausibility: asil_degrade out of {0x00, 0x01, 0xFF}");

    // Invariant 4: total_violations can only increase or stay the same,
    // never decrease except during reset.
    property p_violations_monotonic;
        @(posedge clk) disable iff (!rst_n)
        (total_violations <= $past(total_violations) + 16'd1) &&
        (total_violations >= $past(total_violations));
    endproperty
    a_violations_monotonic: assert property (p_violations_monotonic)
        else $error("plausibility: total_violations decreased without reset");
`endif
`endif

endmodule
