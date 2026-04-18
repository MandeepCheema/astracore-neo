`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — Detection Arbiter  (det_arbiter.v)
// =============================================================================
// Layer 2 glue.  Round-robin arbiter that multiplexes 3 detection sources
// (camera / radar / lidar) into a single det_valid + x,y,z,class,conf stream
// for coord_transform + object_tracker.  Resolves the v1 astracore_fusion_top
// caveat that only the camera path fed the transform stage.
//
// ── Arbitration ──────────────────────────────────────────────────────────────
//   On each clock, check the sources in rotating priority order starting from
//   current_priority.  The first source whose *_valid is high wins:
//     • emits out_valid with that source's fields
//     • asserts its corresponding *_ack signal (so a FIFO upstream can pop)
//     • advances current_priority by 1 so the NEXT cycle starts elsewhere
//
//   Fairness: over N cycles every asserted source eventually wins; no source
//   can starve another more than N-1 cycles.
//
//   Single-cycle latency from in_valid to out_valid (registered output).
//
// ── Source ID encoding ───────────────────────────────────────────────────────
//   out_sensor_id[1:0]:
//     2'd0 = camera
//     2'd1 = radar
//     2'd2 = lidar
//     2'd3 = reserved (future ultrasonic when it has per-point detections)
//
// ── Interface ─────────────────────────────────────────────────────────────────
//   Per-source input: *_valid + *_x_mm + *_y_mm + *_z_mm + *_class + *_conf
//   *_ack output pulses for 1 cycle when that source wins arbitration.
//
//   out_* is a standard 1-cycle det_valid pulse with the transformed-ready
//   fields.  Downstream consumer is assumed always-ready (coord_transform is).
// =============================================================================

module det_arbiter (
    input  wire        clk,
    input  wire        rst_n,

    // ── Source 0: Camera ──────────────────────────────────────────────────────
    input  wire        cam_valid,
    input  wire signed [31:0] cam_x_mm,
    input  wire signed [31:0] cam_y_mm,
    input  wire signed [31:0] cam_z_mm,
    input  wire [7:0]  cam_class_id,
    input  wire [7:0]  cam_confidence,
    output reg         cam_ack,

    // ── Source 1: Radar ──────────────────────────────────────────────────────
    input  wire        rad_valid,
    input  wire signed [31:0] rad_x_mm,
    input  wire signed [31:0] rad_y_mm,
    input  wire signed [31:0] rad_z_mm,
    input  wire [7:0]  rad_class_id,
    input  wire [7:0]  rad_confidence,
    output reg         rad_ack,

    // ── Source 2: LiDAR ──────────────────────────────────────────────────────
    input  wire        lid_valid,
    input  wire signed [31:0] lid_x_mm,
    input  wire signed [31:0] lid_y_mm,
    input  wire signed [31:0] lid_z_mm,
    input  wire [7:0]  lid_class_id,
    input  wire [7:0]  lid_confidence,
    output reg         lid_ack,

    // ── Merged detection output ──────────────────────────────────────────────
    output reg         out_valid,
    output reg  [1:0]  out_sensor_id,
    output reg  signed [31:0] out_x_mm,
    output reg  signed [31:0] out_y_mm,
    output reg  signed [31:0] out_z_mm,
    output reg  [7:0]  out_class_id,
    output reg  [7:0]  out_confidence
);

    // Rotating priority pointer: 0=cam first, 1=rad first, 2=lid first
    reg [1:0] priority_idx;

    // Per-source "already fired, waiting for source to release" latches.
    // Without these, the arbiter re-fires on the following cycle because the
    // upstream FIFO's *_out_valid is driven combinatorially from !empty and
    // does not fall until the pop completes at edge N+1 (one cycle after
    // *_ack is already outbound). Each detection therefore produces two
    // out_valid pulses, one of which doubles every downstream allocation.
    // The "fired" latch masks the source from re-winning until its valid
    // input goes low (indicating the upstream FIFO has completed its pop).
    reg cam_fired, rad_fired, lid_fired;

    wire eff_cam_valid = cam_valid && !cam_fired;
    wire eff_rad_valid = rad_valid && !rad_fired;
    wire eff_lid_valid = lid_valid && !lid_fired;

    // Combinatorial winner selection based on priority_idx + eff_*_valid
    reg [1:0] winner_id;
    reg       any_winner;

    always @(*) begin
        winner_id  = 2'd0;
        any_winner = 1'b0;

        case (priority_idx)
            2'd0: begin
                if      (eff_cam_valid) begin winner_id = 2'd0; any_winner = 1'b1; end
                else if (eff_rad_valid) begin winner_id = 2'd1; any_winner = 1'b1; end
                else if (eff_lid_valid) begin winner_id = 2'd2; any_winner = 1'b1; end
            end
            2'd1: begin
                if      (eff_rad_valid) begin winner_id = 2'd1; any_winner = 1'b1; end
                else if (eff_lid_valid) begin winner_id = 2'd2; any_winner = 1'b1; end
                else if (eff_cam_valid) begin winner_id = 2'd0; any_winner = 1'b1; end
            end
            default: begin // 2'd2
                if      (eff_lid_valid) begin winner_id = 2'd2; any_winner = 1'b1; end
                else if (eff_cam_valid) begin winner_id = 2'd0; any_winner = 1'b1; end
                else if (eff_rad_valid) begin winner_id = 2'd1; any_winner = 1'b1; end
            end
        endcase
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            priority_idx  <= 2'd0;
            out_valid     <= 1'b0;
            out_sensor_id <= 2'd0;
            out_x_mm      <= 32'sd0;
            out_y_mm      <= 32'sd0;
            out_z_mm      <= 32'sd0;
            out_class_id  <= 8'd0;
            out_confidence<= 8'd0;
            cam_ack       <= 1'b0;
            rad_ack       <= 1'b0;
            lid_ack       <= 1'b0;
            cam_fired     <= 1'b0;
            rad_fired     <= 1'b0;
            lid_fired     <= 1'b0;
        end else begin
            out_valid <= 1'b0;
            cam_ack   <= 1'b0;
            rad_ack   <= 1'b0;
            lid_ack   <= 1'b0;

            // Clear fired latch whenever the corresponding source has released
            // its valid (FIFO drained the entry). Must run before the win case
            // so that a source which re-arms *this* cycle can also win.
            if (!cam_valid) cam_fired <= 1'b0;
            if (!rad_valid) rad_fired <= 1'b0;
            if (!lid_valid) lid_fired <= 1'b0;

            if (any_winner) begin
                out_valid <= 1'b1;
                out_sensor_id <= winner_id;
                if (priority_idx == 2'd2)
                    priority_idx <= 2'd0;
                else
                    priority_idx <= priority_idx + 2'd1;

                case (winner_id)
                    2'd0: begin
                        cam_ack <= 1'b1;
                        cam_fired <= 1'b1;
                        out_x_mm       <= cam_x_mm;
                        out_y_mm       <= cam_y_mm;
                        out_z_mm       <= cam_z_mm;
                        out_class_id   <= cam_class_id;
                        out_confidence <= cam_confidence;
                    end
                    2'd1: begin
                        rad_ack <= 1'b1;
                        rad_fired <= 1'b1;
                        out_x_mm       <= rad_x_mm;
                        out_y_mm       <= rad_y_mm;
                        out_z_mm       <= rad_z_mm;
                        out_class_id   <= rad_class_id;
                        out_confidence <= rad_confidence;
                    end
                    default: begin // 2'd2
                        lid_ack <= 1'b1;
                        lid_fired <= 1'b1;
                        out_x_mm       <= lid_x_mm;
                        out_y_mm       <= lid_y_mm;
                        out_z_mm       <= lid_z_mm;
                        out_class_id   <= lid_class_id;
                        out_confidence <= lid_confidence;
                    end
                endcase
            end
        end
    end

    // =========================================================================
    // Det-arbiter invariants — Phase-F formal verification targets
    // =========================================================================
`ifndef SYNTHESIS
`ifndef __ICARUS__
    // Invariant: out_sensor_id is one of {0=cam, 1=radar, 2=lidar}. 3 is never
    // legal (only 3 sources multiplexed).
    property p_out_sensor_id_valid;
        @(posedge clk) disable iff (!rst_n)
        out_valid |-> (out_sensor_id <= 2'd2);
    endproperty
    a_out_sensor_id_valid: assert property (p_out_sensor_id_valid)
        else $error("det_arbiter: out_sensor_id out of {0,1,2}");

    // Invariant: at most one of cam_ack / rad_ack / lid_ack asserted per cycle.
    property p_at_most_one_ack;
        @(posedge clk) disable iff (!rst_n)
        $countones({cam_ack, rad_ack, lid_ack}) <= 1;
    endproperty
    a_at_most_one_ack: assert property (p_at_most_one_ack)
        else $error("det_arbiter: multiple acks in one cycle (arbiter violation)");
`endif
`endif

endmodule
