`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — DMS Fusion Engine
// =============================================================================
// Combines gaze_tracker and head_pose_tracker outputs into a unified
// driver_attention_level[2:0] with IIR temporal smoothing and ASIL-D
// sensor-fail detection.
//
// Level encoding (sensor_fusion_architecture.md §A):
//   3'b000 = ATTENTIVE   — in zone, eyes open, PERCLOS < 20%
//   3'b001 = DROWSY      — PERCLOS 20–50% or elevated blink rate
//   3'b010 = DISTRACTED  — head out of zone continuously > 3 s
//   3'b100 = CRITICAL    — PERCLOS > 50% or eyes closed continuously > 2 s
//   3'b111 = SENSOR_FAIL — watchdog: no valid pulse within WATCHDOG_CYCLES
//
// Fusion rules (priority, highest first):
//   1. SENSOR_FAIL  — watchdog fires (camera stuck)
//   2. CRITICAL     — perclos_num >= PERCLOS_CRIT  OR  continuous closed >= 2 s
//   3. DROWSY       — perclos_num >= PERCLOS_DROWSY OR blink rate elevated
//   4. DISTRACTED   — continuous out-of-zone frames >= 3 s
//   5. ATTENTIVE    — none of the above
//
// Temporal smoothing (prevents single-frame false alerts):
//   raw level maps to score 0-100; IIR: score = (3*score + raw) >> 2
//   (~0.75 prev + 0.25 new). Alert upgrades use score >= threshold.
//   SENSOR_FAIL bypasses the IIR (immediate assertion).
//
// Interface:
//   clk, rst_n              — 50 MHz system clock, active-low sync reset
//   gaze_valid              — 1-cycle pulse when gaze_tracker outputs are new
//   eye_state[1:0]          — gaze_tracker: 00=OPEN, 01=PARTIAL, 10=CLOSED
//   perclos_num[PWIDTH-1:0] — gaze_tracker: closed frames in 30-frame window
//   blink_count[15:0]       — gaze_tracker: total blinks since reset
//   pose_valid              — 1-cycle pulse when head_pose_tracker outputs new
//   in_zone                 — head_pose_tracker: 1 = within attention zone
//   distracted_count[DWIDTH-1:0] — head_pose_tracker: out-of-zone frames/window
//   driver_attention_level[2:0]  — fused attention level (see encoding above)
//   dms_confidence[7:0]     — fusion confidence 0–100 (0 = unknown/fail)
//   dms_alert               — asserted on CRITICAL or SENSOR_FAIL
// =============================================================================

module dms_fusion #(
    // Bit widths — must match gaze_tracker / head_pose_tracker parameters
    parameter PWIDTH = 5,   // perclos_num width  ($clog2(30)+1 = 5)
    parameter DWIDTH = 4,   // distracted_count width ($clog2(15)+1 = 4)

    // PERCLOS thresholds (frames within 30-frame window)
    parameter PERCLOS_DROWSY_THRESH  = 5'd6,   // 20%  → DROWSY
    parameter PERCLOS_CRIT_THRESH    = 5'd15,  // 50%  → CRITICAL

    // Continuous-event thresholds (frames @ ~30 fps)
    parameter CLOSED_CRIT_FRAMES     = 7'd60,  // 2 s @ 30 fps → CRITICAL
    parameter DISTRACTED_CRIT_FRAMES = 7'd90,  // 3 s @ 30 fps → DISTRACTED

    // Blink-rate elevation: blinks per 30-frame window
    parameter BLINK_HIGH_THRESH      = 4'd8,

    // Sensor watchdog: no gaze_valid for this many clocks → SENSOR_FAIL
    // Default: 200 ms at 50 MHz = 10_000_000 cycles
    parameter WATCHDOG_CYCLES        = 24'd10_000_000
) (
    input  wire        clk,
    input  wire        rst_n,

    // --- gaze_tracker outputs (connect directly) ---
    input  wire        gaze_valid,
    input  wire [1:0]  eye_state,
    input  wire [PWIDTH-1:0] perclos_num,
    input  wire [15:0] blink_count,

    // --- head_pose_tracker outputs (connect directly) ---
    input  wire        pose_valid,
    input  wire        in_zone,
    input  wire [DWIDTH-1:0] distracted_count,

    // --- Outputs ---
    output wire [2:0]  driver_attention_level,
    output wire [7:0]  dms_confidence,
    output wire        dms_alert,
    output wire        tmr_agreement,
    output wire        tmr_fault
);

    // -------------------------------------------------------------------------
    // Level encodings (local parameters)
    // -------------------------------------------------------------------------
    localparam LEVEL_ATTENTIVE   = 3'b000;
    localparam LEVEL_DROWSY      = 3'b001;
    localparam LEVEL_DISTRACTED  = 3'b010;
    localparam LEVEL_CRITICAL    = 3'b100;
    localparam LEVEL_SENSOR_FAIL = 3'b111;

    // -------------------------------------------------------------------------
    // 1. Watchdog — sensor-fail detection (ASIL-D)
    //    If gaze_valid is not seen within WATCHDOG_CYCLES, camera is stuck.
    // -------------------------------------------------------------------------
    reg [23:0] wdog_cnt;
    reg        sensor_fail;

    always @(posedge clk) begin
        if (!rst_n) begin
            wdog_cnt    <= 24'd0;
            sensor_fail <= 1'b0;
        end else if (gaze_valid) begin
            wdog_cnt    <= 24'd0;
            sensor_fail <= 1'b0;
        end else if (wdog_cnt == WATCHDOG_CYCLES - 1) begin
            wdog_cnt    <= wdog_cnt;   // saturate
            sensor_fail <= 1'b1;
        end else begin
            wdog_cnt    <= wdog_cnt + 24'd1;
        end
    end

    // -------------------------------------------------------------------------
    // 2. Continuous-closed counter (for >2 s CRITICAL path)
    //    Increments each gaze_valid frame where eye_state == CLOSED.
    //    Resets on any non-CLOSED frame. Saturates at 127 to prevent wrap.
    // -------------------------------------------------------------------------
    reg [6:0] cont_closed;

    always @(posedge clk) begin
        if (!rst_n) begin
            cont_closed <= 7'd0;
        end else if (gaze_valid) begin
            if (eye_state == 2'b10)
                cont_closed <= (cont_closed == 7'd127) ? 7'd127 : cont_closed + 7'd1;
            else
                cont_closed <= 7'd0;
        end
    end

    // -------------------------------------------------------------------------
    // 3. Continuous-distracted counter (for >3 s DISTRACTED path)
    //    Increments each pose_valid frame where in_zone == 0.
    //    Resets when back in zone. Saturates at 127.
    // -------------------------------------------------------------------------
    reg [6:0] cont_distracted;

    always @(posedge clk) begin
        if (!rst_n) begin
            cont_distracted <= 7'd0;
        end else if (pose_valid) begin
            if (!in_zone)
                cont_distracted <= (cont_distracted == 7'd127) ? 7'd127 : cont_distracted + 7'd1;
            else
                cont_distracted <= 7'd0;
        end
    end

    // -------------------------------------------------------------------------
    // 4. Blink-rate elevation
    //    Every BLINK_WINDOW gaze_valid frames, snapshot blink_count.
    //    If delta > BLINK_HIGH_THRESH in that window → blink_elevated.
    // -------------------------------------------------------------------------
    localparam BLINK_WINDOW = 6'd30;   // 1 s @ 30 fps

    reg  [5:0]  blink_frame_cnt;
    reg  [15:0] blink_snapshot;
    reg         blink_elevated;

    always @(posedge clk) begin
        if (!rst_n) begin
            blink_frame_cnt <= 6'd0;
            blink_snapshot  <= 16'd0;
            blink_elevated  <= 1'b0;
        end else if (gaze_valid) begin
            if (blink_frame_cnt == BLINK_WINDOW - 1) begin
                blink_frame_cnt <= 6'd0;
                blink_snapshot  <= blink_count;
                // blink_count - blink_snapshot = delta in this window
                blink_elevated  <= ((blink_count - blink_snapshot) > {12'b0, BLINK_HIGH_THRESH});
            end else begin
                blink_frame_cnt <= blink_frame_cnt + 6'd1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // 5. Raw level — priority logic (combinational)
    //    Produces a 0–100 score that the IIR filter will smooth.
    // -------------------------------------------------------------------------
    // Score mapping: ATTENTIVE=0, DISTRACTED=30, DROWSY=55, CRITICAL=90
    // (SENSOR_FAIL bypasses IIR and is handled separately)
    localparam [6:0] SCORE_ATTENTIVE  = 7'd0;
    localparam [6:0] SCORE_DISTRACTED = 7'd30;
    localparam [6:0] SCORE_DROWSY     = 7'd55;
    localparam [6:0] SCORE_CRITICAL   = 7'd90;

    reg [6:0] raw_score;

    always @(*) begin
        if (sensor_fail) begin
            raw_score = SCORE_CRITICAL;   // IIR not used for SENSOR_FAIL path
        end else if ((perclos_num >= PERCLOS_CRIT_THRESH) ||
                     (cont_closed  >= CLOSED_CRIT_FRAMES)) begin
            raw_score = SCORE_CRITICAL;
        end else if ((perclos_num >= PERCLOS_DROWSY_THRESH) || blink_elevated) begin
            raw_score = SCORE_DROWSY;
        end else if (cont_distracted >= DISTRACTED_CRIT_FRAMES) begin
            raw_score = SCORE_DISTRACTED;
        end else begin
            raw_score = SCORE_ATTENTIVE;
        end
    end

    // -------------------------------------------------------------------------
    // 6. IIR temporal smoother
    //    score_filt = (3*score_filt + raw_score) >> 2  (~0.75 prev + 0.25 new)
    //    Stored as 9-bit to hold 3*max_score without overflow before shift.
    //    Max: 3*90 + 90 = 360 → needs 9 bits (512 > 360).
    //    Updated on every gaze_valid or pose_valid pulse.
    // -------------------------------------------------------------------------
    reg [8:0] score_filt_x4;    // Q integer scaled by 4 to retain precision

    wire [8:0] raw_score_x4 = {2'b00, raw_score};  // upscale: multiply by 1 (already integer)

    // iir_numer = 3 * score_filt_x4 + raw_score_x4
    // Both operands ≤ 9 bits; product with literal 3 → 11 bits max; add → 11 bits.
    // After >> 2: 9 bits. Assign to 9-bit score_filt_x4.
    wire [10:0] iir_numer = ({2'b00, score_filt_x4} * 3) + {2'b00, raw_score_x4};

    always @(posedge clk) begin
        if (!rst_n) begin
            score_filt_x4 <= 9'd0;
        end else if (gaze_valid || pose_valid) begin
            score_filt_x4 <= iir_numer[10:2];   // divide by 4 (right-shift 2)
        end
    end

    // Extract integer score (score_filt_x4 already holds the unscaled value
    // since raw_score was not pre-multiplied — the x4 label is the accumulator
    // scale due to not dividing back at each step; see note below).
    //
    // NOTE: raw_score_x4 = raw_score (no actual *4). The "x4" accumulator
    // means the IIR accumulates in fractional space: score_filt_x4 converges
    // toward raw_score. Extract final score directly as score_filt_x4[6:0]
    // (top 7 of 9 bits; max converged value = 90, fits in 7 bits).
    wire [6:0] score_int = score_filt_x4[6:0];

    // -------------------------------------------------------------------------
    // 7. TMR output registers — three independent lanes computing the same
    //    {dms_confidence, driver_attention_level} word, voted by tmr_voter.
    //    Catches single-event upsets on the ASIL-D safety output.
    // -------------------------------------------------------------------------
    reg [2:0] dal_a, dal_b, dal_c;
    reg [7:0] conf_a, conf_b, conf_c;
    reg       tmr_valid_r;

    `define DMS_TMR_OUTPUT_LOGIC(DAL, CONF) \
        if (!rst_n) begin \
            DAL  <= LEVEL_ATTENTIVE; \
            CONF <= 8'd100; \
        end else if (sensor_fail) begin \
            DAL  <= LEVEL_SENSOR_FAIL; \
            CONF <= 8'd0; \
        end else begin \
            if (score_int >= 7'd75) begin \
                DAL  <= LEVEL_CRITICAL; \
                CONF <= 8'd95; \
            end else if (score_int >= 7'd42) begin \
                DAL  <= LEVEL_DROWSY; \
                CONF <= 8'd80; \
            end else if (score_int >= 7'd20) begin \
                DAL  <= LEVEL_DISTRACTED; \
                CONF <= 8'd70; \
            end else begin \
                DAL  <= LEVEL_ATTENTIVE; \
                CONF <= 8'd100; \
            end \
        end

    always @(posedge clk) begin `DMS_TMR_OUTPUT_LOGIC(dal_a, conf_a) end
    always @(posedge clk) begin `DMS_TMR_OUTPUT_LOGIC(dal_b, conf_b) end
    always @(posedge clk) begin `DMS_TMR_OUTPUT_LOGIC(dal_c, conf_c) end

    always @(posedge clk)
        tmr_valid_r <= !rst_n ? 1'b0 : (gaze_valid || pose_valid || sensor_fail);

    `undef DMS_TMR_OUTPUT_LOGIC

    wire [31:0] tmr_lane_a = {21'd0, conf_a, dal_a};
    wire [31:0] tmr_lane_b = {21'd0, conf_b, dal_b};
    wire [31:0] tmr_lane_c = {21'd0, conf_c, dal_c};
    wire [31:0] tmr_voted;
    wire        tmr_fault_a, tmr_fault_b, tmr_fault_c, tmr_triple;

    tmr_voter u_tmr_dal (
        .clk         (clk),
        .rst_n       (rst_n),
        .valid       (tmr_valid_r),
        .lane_a      (tmr_lane_a),
        .lane_b      (tmr_lane_b),
        .lane_c      (tmr_lane_c),
        .voted       (tmr_voted),
        .agreement   (tmr_agreement),
        .fault_a     (tmr_fault_a),
        .fault_b     (tmr_fault_b),
        .fault_c     (tmr_fault_c),
        .triple_fault(tmr_triple),
        .vote_count  ()
    );

    assign driver_attention_level = tmr_voted[2:0];
    assign dms_confidence         = tmr_voted[10:3];
    assign tmr_fault = tmr_fault_a | tmr_fault_b | tmr_fault_c | tmr_triple;

    // -------------------------------------------------------------------------
    // 8. Alert output (combinational from voted level)
    // -------------------------------------------------------------------------
    assign dms_alert = (driver_attention_level == LEVEL_CRITICAL) ||
                       (driver_attention_level == LEVEL_SENSOR_FAIL);

    // =========================================================================
    // DMS-fusion ASIL-D safety invariants — Phase-F formal targets
    // =========================================================================
`ifndef SYNTHESIS
`ifndef __ICARUS__
    // Invariant 1: dms_alert asserted for CRITICAL or SENSOR_FAIL only.
    property p_alert_iff_crit_or_fail;
        @(posedge clk) disable iff (!rst_n)
        dms_alert == ((driver_attention_level == LEVEL_CRITICAL) ||
                      (driver_attention_level == LEVEL_SENSOR_FAIL));
    endproperty
    a_alert_iff_crit_or_fail: assert property (p_alert_iff_crit_or_fail)
        else $error("DMS: dms_alert inconsistent with driver_attention_level");

    // Invariant 2: tmr_agreement=0 implies tmr_fault is non-zero
    // (if no lanes agree, at least one fault signal must be set).
    property p_disagreement_implies_fault;
        @(posedge clk) disable iff (!rst_n)
        !tmr_agreement |-> tmr_fault;
    endproperty
    a_disagreement_implies_fault: assert property (p_disagreement_implies_fault)
        else $error("DMS: tmr_agreement=0 but tmr_fault=0 (missing fault report)");
`endif
`endif

endmodule
