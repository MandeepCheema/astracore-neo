`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — AEB Controller  (aeb_controller.v)
// =============================================================================
// Layer 3 decision module.  ASIL-D.  Consumes TTC flags from ttc_calculator
// and drives a 4-level brake command bundle: off / warn / precharge / emergency.
// Applies escalation-up immediately, de-escalation gated by a clear-ticks
// threshold and a minimum emergency hold time (MIN_BRAKE_MS).
//
// ── Decision levels ──────────────────────────────────────────────────────────
//   Level 0 (OFF)       — no threat, no brake, no alert
//   Level 1 (WARN)      — audible alert only, no brake (ttc_warning fired)
//   Level 2 (PRECHARGE) — low-g pre-fill, target_decel = 2000 mm/s² (~0.2 g)
//   Level 3 (EMERGENCY) — full emergency, target_decel = 10000 mm/s² (~1 g)
//
//   Incoming ttc level per tick:
//     ttc_brake   → 3    ttc_prepare → 2    ttc_warning → 1    else → 0
//
// ── Escalation / de-escalation ───────────────────────────────────────────────
//   On each ttc_valid pulse:
//     if incoming > current → jump to incoming immediately (clear_cnt = 0)
//     if incoming == current → clear_cnt = 0 (threat still present)
//     if incoming <  current → clear_cnt++ ; when clear_cnt == CLEAR_TICKS-1
//                              advance, downgrade level by 1, reset clear_cnt.
//
//   Additional gate: while in EMERGENCY (level 3), any downgrade is blocked
//   until brake_hold_ms reaches zero.  Protects against dropout glitches
//   while the brakes are engaged.  brake_hold_ms decrements on tick_1ms.
//
// ── Output bundle ────────────────────────────────────────────────────────────
//   brake_level[1:0]   — registered FSM state (0..3)
//   brake_active       — combinatorial: brake_level >= 2
//   target_decel_mms2  — combinatorial: per-level LUT
//   alert_driver       — combinatorial: brake_level >= 1
//   active_threat_id   — track id of the object that triggered the current level
//
// ── Parameters ────────────────────────────────────────────────────────────────
//   CLEAR_TICKS  — consecutive non-threat ttc events before a downgrade (5)
//   MIN_BRAKE_MS — emergency-brake minimum hold time in ms (500)
// =============================================================================

module aeb_controller #(
    parameter [7:0]  CLEAR_TICKS  = 8'd5,
    parameter [15:0] MIN_BRAKE_MS = 16'd500
)(
    input  wire        clk,
    input  wire        rst_n,

    // ── TTC input (from ttc_calculator) ──────────────────────────────────────
    input  wire        ttc_valid,
    input  wire [15:0] ttc_track_id,
    input  wire        ttc_warning,
    input  wire        ttc_prepare,
    input  wire        ttc_brake,

    // ── 1 ms tick — drives brake_hold_ms countdown ───────────────────────────
    input  wire        tick_1ms,

    // ── Brake command outputs ─────────────────────────────────────────────────
    output reg  [1:0]  brake_level,
    output wire        brake_active,
    output reg  [15:0] target_decel_mms2,
    output wire        alert_driver,
    output reg  [15:0] active_threat_id,
    output reg  [15:0] brake_hold_ms
);

    // =========================================================================
    // 1. Decode incoming TTC flags into a priority level
    // =========================================================================
    wire [1:0] incoming_level =
        ttc_brake   ? 2'd3 :
        ttc_prepare ? 2'd2 :
        ttc_warning ? 2'd1 :
                      2'd0;

    // =========================================================================
    // 2. Combinatorial derived outputs (track brake_level without 1-cycle lag)
    // =========================================================================
    assign brake_active = (brake_level >= 2'd2);
    assign alert_driver = (brake_level >= 2'd1);

    // =========================================================================
    // 3. Main FSM and hold timer
    // =========================================================================
    reg [7:0] clear_cnt;

    always @(posedge clk) begin
        if (!rst_n) begin
            brake_level       <= 2'd0;
            target_decel_mms2 <= 16'd0;
            active_threat_id  <= 16'd0;
            brake_hold_ms     <= 16'd0;
            clear_cnt         <= 8'd0;
        end else begin
            // Emergency-hold timer counts down on 1ms tick
            if (tick_1ms && brake_hold_ms != 16'd0)
                brake_hold_ms <= brake_hold_ms - 16'd1;

            if (ttc_valid) begin
                if (incoming_level > brake_level) begin
                    // ── Escalation (immediate) ──────────────────────────────
                    brake_level      <= incoming_level;
                    active_threat_id <= ttc_track_id;
                    clear_cnt        <= 8'd0;
                    if (incoming_level == 2'd3)
                        brake_hold_ms <= MIN_BRAKE_MS;

                    // Update decel LUT on the same edge
                    case (incoming_level)
                        2'd0:    target_decel_mms2 <= 16'd0;
                        2'd1:    target_decel_mms2 <= 16'd0;
                        2'd2:    target_decel_mms2 <= 16'd2000;
                        2'd3:    target_decel_mms2 <= 16'd10000;
                        default: target_decel_mms2 <= 16'd0;
                    endcase

                end else if (incoming_level == brake_level) begin
                    // ── Same level: refresh threat id and clear counter ─────
                    active_threat_id <= ttc_track_id;
                    clear_cnt        <= 8'd0;

                end else begin
                    // ── Would de-escalate: gated by clear_cnt and hold timer
                    if (brake_level == 2'd3 && brake_hold_ms != 16'd0) begin
                        // emergency hold: no downgrade allowed
                        clear_cnt <= clear_cnt;
                    end else if (clear_cnt == CLEAR_TICKS - 8'd1) begin
                        brake_level <= brake_level - 2'd1;
                        clear_cnt   <= 8'd0;
                        // Update decel LUT for the downgraded level
                        case (brake_level - 2'd1)
                            2'd0:    target_decel_mms2 <= 16'd0;
                            2'd1:    target_decel_mms2 <= 16'd0;
                            2'd2:    target_decel_mms2 <= 16'd2000;
                            default: target_decel_mms2 <= 16'd0;
                        endcase
                    end else begin
                        clear_cnt <= clear_cnt + 8'd1;
                    end
                end
            end
        end
    end

    // =========================================================================
    // ASIL-D safety invariants (SVA) — Phase-F formal verification targets
    // =========================================================================
`ifndef SYNTHESIS
`ifndef __ICARUS__
    // Invariant 1: brake_level is one of {0,1,2,3}; the reset/else paths
    // must never leave it outside that range.
    property p_brake_level_valid;
        @(posedge clk) disable iff (!rst_n)
        (brake_level <= 2'd3);
    endproperty
    a_brake_level_valid: assert property (p_brake_level_valid)
        else $error("AEB: brake_level out of 0..3 range");

    // Invariant 2: brake_active is set iff brake_level >= EMERGENCY (3).
    property p_brake_active_iff_emergency;
        @(posedge clk) disable iff (!rst_n)
        (brake_active == (brake_level == 2'd3));
    endproperty
    a_brake_active_iff_emergency:
        assert property (p_brake_active_iff_emergency)
        else $error("AEB: brake_active inconsistent with brake_level == EMERGENCY");

    // Invariant 3: target_decel_mms2 > 0 iff brake_level == EMERGENCY.
    // Non-emergency states must not apply deceleration.
    property p_decel_only_when_emergency;
        @(posedge clk) disable iff (!rst_n)
        ((brake_level == 2'd3) || (target_decel_mms2 == 16'd0));
    endproperty
    a_decel_only_when_emergency:
        assert property (p_decel_only_when_emergency)
        else $error("AEB: target_decel_mms2 > 0 outside EMERGENCY state");
`endif
`endif

endmodule
