`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — CAN Odometry Decoder  (can_odometry_decoder.v)
// =============================================================================
// Layer 1 sensor interface.  Consumes decoded CAN-FD frames from the RX FIFO
// of canfd_controller (Rev2) and extracts vehicle odometry signals that feed
// ego_motion_estimator and plausibility_checker.
//
// ── Consumed CAN IDs (parameterisable per OEM DBC) ───────────────────────────
//   WHEEL_SPEED_ID — 8-byte payload: 4 × u16 big-endian wheel speeds in mm/s
//       rx_out_data[63:48] = FL (front-left)
//       rx_out_data[47:32] = FR (front-right)
//       rx_out_data[31:16] = RL (rear-left)
//       rx_out_data[15: 0] = RR (rear-right)
//
//   STEERING_ID — 8-byte payload:
//       rx_out_data[63:48] = steering_angle_mdeg  (signed, +right turn)
//       rx_out_data[47:32] = yaw_rate_mdps         (signed, +left turn)
//       rx_out_data[31: 0] = reserved (ignored)
//
//   Unknown IDs are silently ignored (consumed from the FIFO with odo_valid=0).
//
// ── Outputs ──────────────────────────────────────────────────────────────────
//   odo_valid              — 1-cycle pulse whenever odometry state refreshes
//                            (either wheel-speed or steering frame)
//   wheel_speed_mmps[15:0] — latched arithmetic mean of the 4 wheels (mm/s)
//   steer_mdeg[15:0]       — latched signed steering angle
//   odo_yaw_rate_mdps[15:0]— latched signed yaw rate
//
//   Per-wheel diagnostics (latched on every wheel-speed frame):
//     wheel_fl_mmps, wheel_fr_mmps, wheel_rl_mmps, wheel_rr_mmps
//
// The module always sets rx_out_ready = 1 — it always accepts the next frame
// from the canfd_controller FIFO in a single cycle.
//
// ── Parameters ────────────────────────────────────────────────────────────────
//   WHEEL_SPEED_ID / STEERING_ID — 29-bit CAN extended IDs (default 0x1A0/0x1B0)
// =============================================================================

module can_odometry_decoder #(
    parameter [28:0] WHEEL_SPEED_ID = 29'h0000_01A0,
    parameter [28:0] STEERING_ID    = 29'h0000_01B0
)(
    input  wire        clk,
    input  wire        rst_n,

    // ── Downstream read port of canfd_controller RX FIFO ─────────────────────
    input  wire        rx_out_valid,
    input  wire [28:0] rx_out_id,
    input  wire [3:0]  rx_out_dlc,     // unused here — we trust payload width
    input  wire [63:0] rx_out_data,
    output wire        rx_out_ready,

    // ── Odometry outputs to ego_motion_estimator ──────────────────────────────
    output reg         odo_valid,
    output reg  [15:0] wheel_speed_mmps,
    output reg  signed [15:0] steer_mdeg,
    output reg  signed [15:0] odo_yaw_rate_mdps,

    // ── Per-wheel diagnostic outputs ──────────────────────────────────────────
    output reg  [15:0] wheel_fl_mmps,
    output reg  [15:0] wheel_fr_mmps,
    output reg  [15:0] wheel_rl_mmps,
    output reg  [15:0] wheel_rr_mmps,

    // ── Counters ──────────────────────────────────────────────────────────────
    output reg  [15:0] wheel_frame_count,
    output reg  [15:0] steering_frame_count,
    output reg  [15:0] ignored_frame_count
);

    // Always-ready consumer — single-cycle acceptance from the FIFO
    assign rx_out_ready = 1'b1;

    // Per-wheel fields extracted combinatorially from the current payload
    wire [15:0] w_fl = rx_out_data[63:48];
    wire [15:0] w_fr = rx_out_data[47:32];
    wire [15:0] w_rl = rx_out_data[31:16];
    wire [15:0] w_rr = rx_out_data[15: 0];

    // Average wheel speed — 4-way add in 18-bit space, then >>> 2
    // (unsigned, so shift-right = divide)
    wire [17:0] w_sum     = {2'b00, w_fl} + {2'b00, w_fr}
                          + {2'b00, w_rl} + {2'b00, w_rr};
    wire [15:0] w_avg_mmps = w_sum[17:2];   // divide by 4

    // Steering + yaw-rate fields (signed)
    wire signed [15:0] steer_field    = rx_out_data[63:48];
    wire signed [15:0] yaw_rate_field = rx_out_data[47:32];

    always @(posedge clk) begin
        if (!rst_n) begin
            odo_valid            <= 1'b0;
            wheel_speed_mmps     <= 16'd0;
            steer_mdeg           <= 16'sd0;
            odo_yaw_rate_mdps    <= 16'sd0;
            wheel_fl_mmps        <= 16'd0;
            wheel_fr_mmps        <= 16'd0;
            wheel_rl_mmps        <= 16'd0;
            wheel_rr_mmps        <= 16'd0;
            wheel_frame_count    <= 16'd0;
            steering_frame_count <= 16'd0;
            ignored_frame_count  <= 16'd0;
        end else begin
            odo_valid <= 1'b0;   // default 1-cycle pulse

            if (rx_out_valid) begin
                case (rx_out_id)
                    WHEEL_SPEED_ID: begin
                        wheel_fl_mmps    <= w_fl;
                        wheel_fr_mmps    <= w_fr;
                        wheel_rl_mmps    <= w_rl;
                        wheel_rr_mmps    <= w_rr;
                        wheel_speed_mmps <= w_avg_mmps;
                        odo_valid        <= 1'b1;
                        if (wheel_frame_count != 16'hFFFF)
                            wheel_frame_count <= wheel_frame_count + 16'd1;
                    end

                    STEERING_ID: begin
                        steer_mdeg        <= steer_field;
                        odo_yaw_rate_mdps <= yaw_rate_field;
                        odo_valid         <= 1'b1;
                        if (steering_frame_count != 16'hFFFF)
                            steering_frame_count <= steering_frame_count + 16'd1;
                    end

                    default: begin
                        if (ignored_frame_count != 16'hFFFF)
                            ignored_frame_count <= ignored_frame_count + 16'd1;
                    end
                endcase
            end
        end
    end

endmodule
