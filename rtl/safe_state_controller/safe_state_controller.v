`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — Safe-State Controller  (safe_state_controller.v)
// =============================================================================
// Layer 3 decision module.  ASIL-D.  Top-level safety manager.  Aggregates
// fault signals from every subsystem and escalates the vehicle through a
// fixed safe-state ladder until the faults clear.  Protects against driving
// further under a degraded compute / sensor / comm stack.
//
// ── State ladder ─────────────────────────────────────────────────────────────
//   0 NORMAL   — full capability, no alert
//   1 ALERT    — driver alert active; retain full driving envelope
//   2 DEGRADE  — speed limited; limit ADAS features
//   3 MRC      — Minimal Risk Condition: pull over, crawl to stop
//
//   MRC is absorbing — only operator_reset returns to NORMAL from MRC.
//
// ── Transitions ───────────────────────────────────────────────────────────────
//   critical_faults : any bit set
//     NORMAL  → ALERT    (immediate)
//     ALERT   → DEGRADE  (after ALERT_TIME_MS of sustained critical fault)
//     DEGRADE → MRC      (after DEGRADE_TIME_MS of sustained critical fault)
//     MRC     → MRC      (absorbing)
//
//   warning_faults only : goes to ALERT and stays there while faults persist.
//
//   No faults : steps down one level per RECOVER_TIME_MS of clear time.
//     NORMAL ← ALERT ← DEGRADE    (but MRC does not auto-recover)
//
//   The escalation and recovery timers are driven by tick_1ms from the system
//   clock generator and share a single counter that is reset on any state
//   transition or fault-class change.
//
// ── Outputs ──────────────────────────────────────────────────────────────────
//   safe_state[1:0]         — current state (0..3)
//   alert_driver            — 1 for state >= 1
//   limit_speed             — 1 for state >= 2
//   max_speed_kmh[7:0]      — 130 / 130 / 60 / 5 by state
//   mrc_pull_over           — 1 only in MRC
//   latched_faults[15:0]    — {warning_faults, critical_faults} since reset
//
// ── Parameters ────────────────────────────────────────────────────────────────
//   ALERT_TIME_MS   — default 2000 ms   (ALERT→DEGRADE escalation delay)
//   DEGRADE_TIME_MS — default 3000 ms   (DEGRADE→MRC escalation delay)
//   RECOVER_TIME_MS — default 5000 ms   (clear time per auto-recovery step)
// =============================================================================

module safe_state_controller #(
    parameter [15:0] ALERT_TIME_MS   = 16'd2000,
    parameter [15:0] DEGRADE_TIME_MS = 16'd3000,
    parameter [15:0] RECOVER_TIME_MS = 16'd5000
)(
    input  wire        clk,
    input  wire        rst_n,

    // ── Fault vectors from subsystems ─────────────────────────────────────────
    input  wire [7:0]  critical_faults,   // any bit = escalate toward MRC
    input  wire [7:0]  warning_faults,    // any bit = ALERT only

    // ── Tick + operator controls ──────────────────────────────────────────────
    input  wire        tick_1ms,
    input  wire        operator_reset,    // 1-cycle pulse: clear MRC → NORMAL

    // ── Safe-state outputs ────────────────────────────────────────────────────
    output reg  [1:0]  safe_state,
    output wire        alert_driver,
    output wire        limit_speed,
    output reg  [7:0]  max_speed_kmh,
    output wire        mrc_pull_over,
    output reg  [15:0] latched_faults
);

    // =========================================================================
    // 1. Derived outputs (combinatorial on safe_state)
    // =========================================================================
    assign alert_driver  = (safe_state >= 2'd1);
    assign limit_speed   = (safe_state >= 2'd2);
    assign mrc_pull_over = (safe_state == 2'd3);

    // =========================================================================
    // 2. Fault class decode
    // =========================================================================
    wire has_critical = |critical_faults;
    wire has_warning  = |warning_faults;
    wire any_fault    = has_critical || has_warning;

    // =========================================================================
    // 3. Escalation / recovery timer
    //    timer_ms counts ms since the last state change or fault-class change.
    //    When it reaches the current-state threshold, advance / recover one step.
    // =========================================================================
    reg [15:0] timer_ms;

    always @(posedge clk) begin
        if (!rst_n) begin
            safe_state     <= 2'd0;
            max_speed_kmh  <= 8'd130;
            timer_ms       <= 16'd0;
            latched_faults <= 16'd0;
        end else begin
            // Latch any observed faults (sticky until operator_reset)
            latched_faults <= latched_faults |
                              {warning_faults, critical_faults};

            // Operator reset from MRC
            if (operator_reset && safe_state == 2'd3) begin
                safe_state     <= 2'd0;
                max_speed_kmh  <= 8'd130;
                timer_ms       <= 16'd0;
                latched_faults <= 16'd0;
            end
            // Critical-fault escalation
            else if (has_critical) begin
                case (safe_state)
                    2'd0: begin   // NORMAL → ALERT (immediate)
                        safe_state    <= 2'd1;
                        max_speed_kmh <= 8'd130;
                        timer_ms      <= 16'd0;
                    end
                    2'd1: begin   // ALERT → DEGRADE (after ALERT_TIME_MS)
                        if (timer_ms >= ALERT_TIME_MS - 16'd1) begin
                            safe_state    <= 2'd2;
                            max_speed_kmh <= 8'd60;
                            timer_ms      <= 16'd0;
                        end else if (tick_1ms) begin
                            timer_ms <= timer_ms + 16'd1;
                        end
                    end
                    2'd2: begin   // DEGRADE → MRC (after DEGRADE_TIME_MS)
                        if (timer_ms >= DEGRADE_TIME_MS - 16'd1) begin
                            safe_state    <= 2'd3;
                            max_speed_kmh <= 8'd5;
                            timer_ms      <= 16'd0;
                        end else if (tick_1ms) begin
                            timer_ms <= timer_ms + 16'd1;
                        end
                    end
                    2'd3: begin   // MRC (absorbing)
                        max_speed_kmh <= 8'd5;
                    end
                endcase
            end
            // Warning-only: force ALERT, do not advance further
            else if (has_warning) begin
                if (safe_state < 2'd1) begin
                    safe_state    <= 2'd1;
                    max_speed_kmh <= 8'd130;
                    timer_ms      <= 16'd0;
                end else if (safe_state > 2'd1) begin
                    // Came down from DEGRADE with only warnings present:
                    // use the same recovery timer as no-fault case
                    if (timer_ms >= RECOVER_TIME_MS - 16'd1) begin
                        safe_state    <= safe_state - 2'd1;
                        timer_ms      <= 16'd0;
                        max_speed_kmh <= (safe_state - 2'd1 == 2'd1) ? 8'd130 : 8'd60;
                    end else if (tick_1ms) begin
                        timer_ms <= timer_ms + 16'd1;
                    end
                end else begin
                    // Already in ALERT with warnings present: hold
                    timer_ms <= 16'd0;
                end
            end
            // No faults: auto-recovery, one step per RECOVER_TIME_MS
            else begin
                if (safe_state == 2'd0) begin
                    timer_ms <= 16'd0;
                end else if (safe_state == 2'd3) begin
                    // MRC never auto-recovers
                    timer_ms <= 16'd0;
                end else begin
                    if (timer_ms >= RECOVER_TIME_MS - 16'd1) begin
                        safe_state <= safe_state - 2'd1;
                        timer_ms   <= 16'd0;
                        case (safe_state - 2'd1)
                            2'd0: max_speed_kmh <= 8'd130;
                            2'd1: max_speed_kmh <= 8'd130;
                            default: max_speed_kmh <= 8'd60;
                        endcase
                    end else if (tick_1ms) begin
                        timer_ms <= timer_ms + 16'd1;
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
    // Invariant 1 (MRC absorbing): once in MRC, the only exit is
    // operator_reset.  Any other transition out of MRC is a safety bug.
    property p_mrc_absorbing;
        @(posedge clk) disable iff (!rst_n)
        (safe_state == 2'd3 && !operator_reset) |=> (safe_state == 2'd3);
    endproperty
    a_mrc_absorbing: assert property (p_mrc_absorbing)
        else $error("safe_state: MRC exited without operator_reset");

    // Invariant 2: mrc_pull_over is set iff safe_state == MRC.
    property p_mrc_pullover_iff_mrc;
        @(posedge clk) disable iff (!rst_n)
        (mrc_pull_over == (safe_state == 2'd3));
    endproperty
    a_mrc_pullover_iff_mrc: assert property (p_mrc_pullover_iff_mrc)
        else $error("safe_state: mrc_pull_over inconsistent with MRC state");

    // Invariant 3: alert_driver is set iff safe_state >= ALERT.
    property p_alert_iff_nonnormal;
        @(posedge clk) disable iff (!rst_n)
        (alert_driver == (safe_state != 2'd0));
    endproperty
    a_alert_iff_nonnormal: assert property (p_alert_iff_nonnormal)
        else $error("safe_state: alert_driver inconsistent with state >= ALERT");

    // Invariant 4: limit_speed is set iff safe_state >= DEGRADE.
    property p_limit_iff_degrade_or_above;
        @(posedge clk) disable iff (!rst_n)
        (limit_speed == (safe_state >= 2'd2));
    endproperty
    a_limit_iff_degrade: assert property (p_limit_iff_degrade_or_above)
        else $error("safe_state: limit_speed inconsistent with state >= DEGRADE");

    // Invariant 5: max_speed_kmh decreases monotonically with state.
    // NORMAL=130, ALERT=130, DEGRADE=60, MRC=5.
    property p_max_speed_per_state;
        @(posedge clk) disable iff (!rst_n)
        ((safe_state == 2'd0 && max_speed_kmh == 8'd130) ||
         (safe_state == 2'd1 && max_speed_kmh == 8'd130) ||
         (safe_state == 2'd2 && max_speed_kmh == 8'd60)  ||
         (safe_state == 2'd3 && max_speed_kmh == 8'd5));
    endproperty
    a_max_speed_per_state: assert property (p_max_speed_per_state)
        else $error("safe_state: max_speed_kmh incorrect for current state");
`endif
`endif

endmodule
