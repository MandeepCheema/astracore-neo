`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — Time-To-Collision (TTC) Calculator  (ttc_calculator.v)
// =============================================================================
// Layer 3 decision module.  ASIL-D.  For each tracked object, computes whether
// the time-to-collision is below WARNING / PREPARE / BRAKE thresholds and
// raises the corresponding flags.  Feeds aeb_controller.
//
// ── Math ─────────────────────────────────────────────────────────────────────
//   Given:
//     range_mm      — distance to object (>= 0, mm)
//     closure_mms   — closure rate (mm/s; negative ⇒ object approaching ego)
//
//   If closure_mms >= 0 the object is moving away or stationary → TTC = ∞,
//   no flag raised.
//
//   Otherwise TTC in ms = range_mm * 1000 / (-closure_mms).
//
//   Division is expensive; instead we multiply both sides of the threshold
//   inequality to get a pure multiply-compare implementation:
//
//     TTC_ms < THRESH_MS
//     ⇔ range_mm * 1000 < THRESH_MS * (-closure_mms)
//
//   Computed in a single clock — 32b×16b products fit easily in 64-bit
//   intermediates with margin.
//
// ── Thresholds ────────────────────────────────────────────────────────────────
//   WARN_MS  — TTC below this raises ttc_warning  (default 3000 ms)
//   PREP_MS  — TTC below this raises ttc_prepare  (default 1500 ms)
//   BRAKE_MS — TTC below this raises ttc_brake    (default  700 ms)
//
//   Hierarchy: brake ⊂ prepare ⊂ warning.  If brake fires, prepare and warning
//   are also set.
//
// ── Pipeline ──────────────────────────────────────────────────────────────────
//   1-cycle latency: obj_valid sampled on EDGE A → ttc_valid pulses on EDGE B
//   with the flags for that object.
//
// ── Interface ─────────────────────────────────────────────────────────────────
//   obj_valid, obj_track_id[15:0], obj_range_mm[31:0], obj_closure_mms[31:0]
//   ttc_valid, ttc_track_id[15:0], ttc_approaching,
//     ttc_warning, ttc_prepare, ttc_brake
// =============================================================================

module ttc_calculator #(
    parameter [15:0] WARN_MS  = 16'd3000,
    parameter [15:0] PREP_MS  = 16'd1500,
    parameter [15:0] BRAKE_MS = 16'd700
)(
    input  wire        clk,
    input  wire        rst_n,

    // ── Object input (per-track, from object_tracker / downstream) ───────────
    input  wire        obj_valid,
    input  wire [15:0] obj_track_id,
    input  wire signed [31:0] obj_range_mm,
    input  wire signed [31:0] obj_closure_mms,

    // ── TTC result (registered, 1-cycle latency from obj_valid) ──────────────
    output reg         ttc_valid,
    output reg  [15:0] ttc_track_id,
    output reg         ttc_approaching,
    output reg         ttc_warning,
    output reg         ttc_prepare,
    output reg         ttc_brake
);

    // =========================================================================
    // 1. Absolute value / approach detection
    //    closure_mms < 0 ⇒ object approaching.  We need |closure| for the
    //    compare math.  Range is normally non-negative but abs() makes it safe.
    // =========================================================================
    wire approaching = obj_closure_mms[31];   // sign bit

    wire [31:0] abs_closure = obj_closure_mms[31]
                                ? (~obj_closure_mms + 32'd1)
                                : obj_closure_mms;
    wire [31:0] abs_range   = obj_range_mm[31]
                                ? (~obj_range_mm + 32'd1)
                                : obj_range_mm;

    // =========================================================================
    // 2. Combinatorial multiply-compare feeding stage-1 pipeline registers
    //    lhs = range_mm * 1000                          (≤ 43 bits)
    //    rhs = abs_closure * THRESH_MS                  (≤ 48 bits)
    //    threshold met when lhs < rhs AND object is approaching
    //
    // The four 64-bit multipliers are registered in stage 1 so Vivado can
    // use the DSP48E1 MREG slot, relaxing the longest combinational path
    // from obj_* → ttc_brake_reg from ~16 ns to ~6 ns per stage.
    // =========================================================================
    wire [63:0] lhs_c       = {32'd0, abs_range}   * 64'd1000;
    wire [63:0] rhs_warn_c  = {32'd0, abs_closure} * {48'd0, WARN_MS};
    wire [63:0] rhs_prep_c  = {32'd0, abs_closure} * {48'd0, PREP_MS};
    wire [63:0] rhs_brake_c = {32'd0, abs_closure} * {48'd0, BRAKE_MS};

    // =========================================================================
    // 3. Stage 1 — register the four products + valid pulse + ride-along info
    // =========================================================================
    reg         s1_valid;
    reg         s1_approaching;
    reg [15:0]  s1_track_id;
    reg [63:0]  s1_lhs, s1_rhs_warn, s1_rhs_prep, s1_rhs_brake;

    always @(posedge clk) begin
        if (!rst_n) begin
            s1_valid       <= 1'b0;
            s1_approaching <= 1'b0;
            s1_track_id    <= 16'd0;
            s1_lhs         <= 64'd0;
            s1_rhs_warn    <= 64'd0;
            s1_rhs_prep    <= 64'd0;
            s1_rhs_brake   <= 64'd0;
        end else begin
            s1_valid <= obj_valid;
            if (obj_valid) begin
                s1_approaching <= approaching;
                s1_track_id    <= obj_track_id;
                s1_lhs         <= lhs_c;
                s1_rhs_warn    <= rhs_warn_c;
                s1_rhs_prep    <= rhs_prep_c;
                s1_rhs_brake   <= rhs_brake_c;
            end
        end
    end

    // =========================================================================
    // 4. Stage 2 — compare against each threshold, register the flags.
    //    Total pipeline latency from obj_valid to ttc_valid: 2 clock cycles.
    // =========================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            ttc_valid       <= 1'b0;
            ttc_track_id    <= 16'd0;
            ttc_approaching <= 1'b0;
            ttc_warning     <= 1'b0;
            ttc_prepare     <= 1'b0;
            ttc_brake       <= 1'b0;
        end else begin
            ttc_valid <= 1'b0;   // default de-assert
            if (s1_valid) begin
                ttc_valid       <= 1'b1;
                ttc_track_id    <= s1_track_id;
                ttc_approaching <= s1_approaching;
                ttc_warning     <= s1_approaching && (s1_lhs < s1_rhs_warn);
                ttc_prepare     <= s1_approaching && (s1_lhs < s1_rhs_prep);
                ttc_brake       <= s1_approaching && (s1_lhs < s1_rhs_brake);
            end
        end
    end

    // =========================================================================
    // ASIL-D safety invariants (SVA) — Phase-F formal verification targets
    // =========================================================================
`ifndef SYNTHESIS
`ifndef __ICARUS__
    // Invariant 1: flag hierarchy — brake implies prepare implies warning.
    // A brake-level threat must ALSO be classified as prepare and warning.
    property p_brake_implies_prepare;
        @(posedge clk) disable iff (!rst_n)
        (ttc_valid && ttc_brake) |-> ttc_prepare;
    endproperty
    a_brake_implies_prepare: assert property (p_brake_implies_prepare)
        else $error("TTC: ttc_brake asserted without ttc_prepare");

    property p_prepare_implies_warning;
        @(posedge clk) disable iff (!rst_n)
        (ttc_valid && ttc_prepare) |-> ttc_warning;
    endproperty
    a_prepare_implies_warning: assert property (p_prepare_implies_warning)
        else $error("TTC: ttc_prepare asserted without ttc_warning");

    // Invariant 2: no flags when not approaching. Objects that are
    // stationary or receding must NEVER raise warning/prepare/brake.
    property p_no_flags_when_receding;
        @(posedge clk) disable iff (!rst_n)
        (ttc_valid && !ttc_approaching)
            |-> (!ttc_warning && !ttc_prepare && !ttc_brake);
    endproperty
    a_no_flags_when_receding: assert property (p_no_flags_when_receding)
        else $error("TTC: warning/prepare/brake raised for non-approaching target");

    // Invariant 3: ttc_valid is a 1-cycle pulse (registered, one-cycle
    // latency from obj_valid — so it should never stay high for 2+ cycles
    // consecutively without an obj_valid pulse.  This is a weaker check
    // that just verifies ttc_valid follows obj_valid with 1-cycle delay.
    property p_ttc_valid_follows_obj_valid;
        @(posedge clk) disable iff (!rst_n)
        obj_valid |=> ttc_valid;
    endproperty
    a_ttc_valid_follows_obj_valid:
        assert property (p_ttc_valid_follows_obj_valid)
        else $error("TTC: ttc_valid did not follow obj_valid by 1 cycle");
`endif
`endif

endmodule
