// =============================================================================
// AstraCore Neo — Fault-injection testbench wrapper for dms_fusion.
//
// Drives a steady "ATTENTIVE driver" stimulus:
//   gaze_valid pulses every 10 cycles (mimics 30 fps frame rate
//   compressed for sim speed)
//   eye_state = 2'b00 (OPEN) — no DROWSY/CRITICAL accumulation
//   in_zone   = 1'b1        — no DISTRACTED accumulation
//   blink_count increments slowly (normal blink rate)
//
// With this baseline, dms_fusion settles to LEVEL_ATTENTIVE on its
// outputs. cocotb runner injects SEUs on internal regs and observes:
//   - tmr_fault (oracle): asserts on TMR-voted disagreement OR (per
//     F4-A-5 fix) on tmr_valid_r vs tmr_valid_r_shadow disagreement.
//   - driver_attention_level (secondary): may transiently drift; the
//     mechanism's expected behaviour (counter-reset, IIR-attenuate)
//     determines whether dl returns to ATTENTIVE within bounds.
// =============================================================================
`timescale 1ns/1ps

module tb_dms_fusion_fi;
    reg            clk;
    reg            rst_n;
    reg            gaze_valid;
    reg  [1:0]     eye_state;
    reg  [4:0]     perclos_num;
    reg  [15:0]    blink_count;
    reg            pose_valid;
    reg            in_zone;
    reg  [3:0]     distracted_count;

    wire [2:0]     driver_attention_level;
    wire [7:0]     dms_confidence;
    wire           dms_alert;
    wire           tmr_agreement;
    wire           tmr_fault;

    dms_fusion u_dut (
        .clk                    (clk),
        .rst_n                  (rst_n),
        .gaze_valid             (gaze_valid),
        .eye_state              (eye_state),
        .perclos_num            (perclos_num),
        .blink_count            (blink_count),
        .pose_valid             (pose_valid),
        .in_zone                (in_zone),
        .distracted_count       (distracted_count),
        .driver_attention_level (driver_attention_level),
        .dms_confidence         (dms_confidence),
        .dms_alert              (dms_alert),
        .tmr_agreement          (tmr_agreement),
        .tmr_fault              (tmr_fault)
    );

    integer tick;

    // clk + rst_n are driven by the cocotb runner (Python side).
    // This initial block sets the steady-state stimulus: OPEN eyes,
    // in zone, no blinks — so dms_fusion settles to ATTENTIVE.
    initial begin
        gaze_valid       = 0;
        eye_state        = 2'b00;   // OPEN
        perclos_num      = 5'd0;
        blink_count      = 16'd0;
        pose_valid       = 0;
        in_zone          = 1'b1;
        distracted_count = 4'd0;
        tick             = 0;
    end

    // 10-cycle "frame tick": pulse gaze_valid + pose_valid every 10
    // cycles to mimic 30 fps frame rate. blink_count increments by 1
    // every 50 ticks to simulate normal 12 blinks/min.
    always @(posedge clk) begin
        if (!rst_n) begin
            tick       <= 0;
            gaze_valid <= 1'b0;
            pose_valid <= 1'b0;
        end else begin
            tick <= tick + 1;
            if ((tick % 10) == 0) begin
                gaze_valid <= 1'b1;
                pose_valid <= 1'b1;
                if ((tick % 50) == 0) blink_count <= blink_count + 1;
            end else begin
                gaze_valid <= 1'b0;
                pose_valid <= 1'b0;
            end
        end
    end

    initial begin
        $dumpfile("dms_fusion_fi.vcd");
        $dumpvars(0, tb_dms_fusion_fi);
    end
endmodule
