`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — LDW / LKA Controller  (ldw_lka_controller.v)
// =============================================================================
// Layer 3 decision module.  ASIL-B.  Consumes the fused lane estimate from
// lane_fusion (center_offset_mm indicates how far ego is from the lane center)
// and drives:
//   • Lane-Departure Warning (audible/visual alert to driver)
//   • Lane-Keeping Assist    (actuated steering torque request)
//
// ── Coordinate convention ────────────────────────────────────────────────────
//   center_offset_mm = (fused_left_mm + fused_right_mm) / 2
//     > 0  →  lane center is right of ego origin, i.e. ego drifted LEFT
//     < 0  →  lane center is left  of ego origin, i.e. ego drifted RIGHT
//
//   Corrective steering torque pushes ego back towards lane center:
//     torque_request = K_TORQUE * center_offset_mm
//     (same sign as offset: positive torque = right turn, negative = left turn)
//   Clamped to ±MAX_TORQUE_MNM to bound actuator stress.
//
// ── Thresholds ────────────────────────────────────────────────────────────────
//   WARN_THRESH_MM — |offset| above this raises ldw_warning
//   ACT_THRESH_MM  — |offset| above this enables LKA torque (lka_active)
//   Hierarchy: act implies warn.  Both clear below their thresholds.
//
// ── No-data handling ──────────────────────────────────────────────────────────
//   If lane_fusion reports fusion_source == 00 (no data) or lane_valid has
//   never fired, outputs hold at zero — LDW/LKA is disabled until a valid
//   lane estimate is available.
//
// ── Interface ─────────────────────────────────────────────────────────────────
//   lane_valid, center_offset_mm[31:0], lane_width_mm[31:0], fusion_source[1:0]
//   ldw_warning, lka_active, steering_torque_mnm[15:0] (signed),
//   departure_direction[1:0]  (00=none, 01=left drift, 10=right drift)
// =============================================================================

module ldw_lka_controller #(
    parameter signed [31:0] WARN_THRESH_MM = 32'sd600,
    parameter signed [31:0] ACT_THRESH_MM  = 32'sd900,
    parameter signed [15:0] K_TORQUE       = 16'sd5,      // mNm per mm offset
    parameter signed [15:0] MAX_TORQUE_MNM = 16'sd5000
)(
    input  wire        clk,
    input  wire        rst_n,

    // ── Lane input (from lane_fusion) ─────────────────────────────────────────
    input  wire        lane_valid,
    input  wire signed [31:0] center_offset_mm,
    input  wire signed [31:0] lane_width_mm,
    input  wire [1:0]  fusion_source,

    // ── LDW / LKA outputs ─────────────────────────────────────────────────────
    output reg         ldw_warning,
    output reg         lka_active,
    output reg signed [15:0] steering_torque_mnm,
    output reg  [1:0]  departure_direction
);

    // =========================================================================
    // 1. Absolute offset (combinatorial, safe for non-INT_MIN values)
    // =========================================================================
    wire signed [31:0] abs_offset = center_offset_mm[31] ?
                                      (-center_offset_mm) :
                                      center_offset_mm;

    wire warn_trip = (abs_offset > WARN_THRESH_MM);
    wire act_trip  = (abs_offset > ACT_THRESH_MM);

    // =========================================================================
    // 2. Torque request (combinatorial, clamped to ±MAX_TORQUE_MNM)
    //    Signed 32b * signed 16b → signed 48b; clamp then truncate to 16b.
    // =========================================================================
    wire signed [47:0] raw_torque = center_offset_mm * K_TORQUE;

    wire signed [15:0] clamped_torque =
        (raw_torque >  $signed({32'sd0, MAX_TORQUE_MNM})) ?  MAX_TORQUE_MNM :
        (raw_torque < -$signed({32'sd0, MAX_TORQUE_MNM})) ? -MAX_TORQUE_MNM :
                                                             raw_torque[15:0];

    // =========================================================================
    // 3. Direction code (combinatorial)
    // =========================================================================
    wire [1:0] dir_comb =
        (!warn_trip)              ? 2'b00 :   // no departure
        (center_offset_mm > 32'sd0) ? 2'b01 : // drifted LEFT (lane center is right)
                                    2'b10;    // drifted RIGHT

    // =========================================================================
    // 4. Registered output update
    //    Only re-evaluate when a valid lane estimate arrives with usable data.
    //    Otherwise outputs hold (or are cleared on reset).
    // =========================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            ldw_warning         <= 1'b0;
            lka_active          <= 1'b0;
            steering_torque_mnm <= 16'sd0;
            departure_direction <= 2'b00;
        end else if (lane_valid && fusion_source != 2'b00) begin
            ldw_warning         <= warn_trip;
            lka_active          <= act_trip;
            departure_direction <= dir_comb;
            // Only issue torque when LKA is actually active
            steering_torque_mnm <= act_trip ? clamped_torque : 16'sd0;
        end
    end

    // =========================================================================
    // ASIL-B safety invariants — Phase-F formal verification targets
    // =========================================================================
`ifndef SYNTHESIS
`ifndef __ICARUS__
    // Invariant 1: lka_active=1 implies ldw_warning=1 (LKA implies LDW).
    property p_lka_implies_ldw;
        @(posedge clk) disable iff (!rst_n)
        lka_active |-> ldw_warning;
    endproperty
    a_lka_implies_ldw: assert property (p_lka_implies_ldw)
        else $error("LDW/LKA: lka_active without ldw_warning (hierarchy violated)");

    // Invariant 2: non-zero torque only when LKA is active.
    property p_torque_only_when_active;
        @(posedge clk) disable iff (!rst_n)
        (steering_torque_mnm != 16'sd0) |-> lka_active;
    endproperty
    a_torque_only_when_active: assert property (p_torque_only_when_active)
        else $error("LDW/LKA: torque non-zero without lka_active");
`endif
`endif

endmodule
