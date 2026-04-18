`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — Ultrasonic Sensor Interface  (ultrasonic_interface.v)
// =============================================================================
// Layer 1 sensor interface.  Receives a 28-byte framed packet over a UART byte
// stream from an external 12-channel ultrasonic sensor array and parses it
// into 12 × 16-bit distance registers plus a 12-bit sensor-health bitmask.
//
// ── Frame format (29 bytes) ──────────────────────────────────────────────────
//   Byte  0      : 0xAA  start-of-frame marker
//   Bytes 1..24  : 12 × 2-byte distance values, big-endian (mm)
//                  bytes 1-2 = ch 0, bytes 3-4 = ch 1, ... bytes 23-24 = ch 11
//   Bytes 25..26 : sensor_health bitmask (big-endian u16; bits 11..0 used,
//                                         bits 15..12 reserved and ignored)
//   Byte  27     : checksum = XOR of bytes 1..26
//   Byte  28     : 0x55  end-of-frame marker
//
//   Invalid start byte, checksum mismatch, or end-of-frame mismatch increments
//   error_count and discards the frame.  The 12 distance registers hold the
//   last successfully-parsed frame.
//
// ── FSM ──────────────────────────────────────────────────────────────────────
//   S_IDLE  : wait for 0xAA
//   S_DATA  : accumulate 26 payload bytes (24 distance + 2 health)
//   S_CKSUM : consume checksum byte, compare against running XOR
//   S_END   : consume end-of-frame byte; commit distances + frame_valid pulse
//
// ── Interface ─────────────────────────────────────────────────────────────────
//   rx_valid, rx_byte[7:0]       — UART byte stream from external USART
//   frame_valid                   — 1-cycle pulse on successful frame parse
//   distance_mm_vec[191:0]        — 12 × 16-bit distances (ch 0 in [15:0])
//   sensor_health[11:0]           — per-channel health bits
//   frame_count[15:0]             — saturating count of valid frames
//   error_count[15:0]             — saturating count of rejected frames
// =============================================================================

module ultrasonic_interface (
    input  wire         clk,
    input  wire         rst_n,

    // ── UART byte stream input ───────────────────────────────────────────────
    input  wire         rx_valid,
    input  wire [7:0]   rx_byte,

    // ── Parsed outputs ────────────────────────────────────────────────────────
    output reg          frame_valid,
    output reg  [191:0] distance_mm_vec,    // 12 × 16 bits, ch 0 in [15:0]
    output reg  [11:0]  sensor_health,
    output reg  [15:0]  frame_count,
    output reg  [15:0]  error_count
);

    localparam SOF_BYTE = 8'hAA;
    localparam EOF_BYTE = 8'h55;

    // State encoding
    localparam [1:0] S_IDLE  = 2'd0,
                     S_DATA  = 2'd1,
                     S_CKSUM = 2'd2,
                     S_END   = 2'd3;

    reg  [1:0]   state;
    reg  [4:0]   byte_idx;        // 0..24 within S_DATA
    reg  [7:0]   running_xor;
    reg  [191:0] dist_accum;      // accumulator — commits to output on valid EOF
    reg  [11:0]  health_accum;

    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            byte_idx        <= 5'd0;
            running_xor     <= 8'd0;
            dist_accum      <= 192'd0;
            health_accum    <= 12'd0;
            frame_valid     <= 1'b0;
            distance_mm_vec <= 192'd0;
            sensor_health   <= 12'd0;
            frame_count     <= 16'd0;
            error_count     <= 16'd0;
        end else begin
            frame_valid <= 1'b0;   // default de-assert

            if (rx_valid) begin
                case (state)
                    // ── Wait for start-of-frame ─────────────────────────────
                    S_IDLE: begin
                        if (rx_byte == SOF_BYTE) begin
                            state        <= S_DATA;
                            byte_idx     <= 5'd0;
                            running_xor  <= 8'd0;
                            dist_accum   <= 192'd0;
                            health_accum <= 12'd0;
                        end
                        // Non-SOF byte in idle: silently ignored (no error —
                        // it may be bus noise or trailing garbage from a prior
                        // aborted frame)
                    end

                    // ── Collect 25 payload bytes (24 distance + 1 health) ──
                    S_DATA: begin
                        running_xor <= running_xor ^ rx_byte;

                        if (byte_idx < 5'd24) begin
                            // Distance bytes: write into the right 16-bit slot
                            // byte_idx even → MSB of channel (byte_idx >> 1)
                            // byte_idx odd  → LSB of that channel
                            // channel index c = byte_idx >> 1 ;  shift = c * 16
                            // big-endian packing:
                            //   MSB at [c*16 + 15 : c*16 + 8]
                            //   LSB at [c*16 +  7 : c*16 + 0]
                            case (byte_idx)
                                5'd0:  dist_accum[  15:  8] <= rx_byte;
                                5'd1:  dist_accum[   7:  0] <= rx_byte;
                                5'd2:  dist_accum[  31: 24] <= rx_byte;
                                5'd3:  dist_accum[  23: 16] <= rx_byte;
                                5'd4:  dist_accum[  47: 40] <= rx_byte;
                                5'd5:  dist_accum[  39: 32] <= rx_byte;
                                5'd6:  dist_accum[  63: 56] <= rx_byte;
                                5'd7:  dist_accum[  55: 48] <= rx_byte;
                                5'd8:  dist_accum[  79: 72] <= rx_byte;
                                5'd9:  dist_accum[  71: 64] <= rx_byte;
                                5'd10: dist_accum[  95: 88] <= rx_byte;
                                5'd11: dist_accum[  87: 80] <= rx_byte;
                                5'd12: dist_accum[ 111:104] <= rx_byte;
                                5'd13: dist_accum[ 103: 96] <= rx_byte;
                                5'd14: dist_accum[ 127:120] <= rx_byte;
                                5'd15: dist_accum[ 119:112] <= rx_byte;
                                5'd16: dist_accum[ 143:136] <= rx_byte;
                                5'd17: dist_accum[ 135:128] <= rx_byte;
                                5'd18: dist_accum[ 159:152] <= rx_byte;
                                5'd19: dist_accum[ 151:144] <= rx_byte;
                                5'd20: dist_accum[ 175:168] <= rx_byte;
                                5'd21: dist_accum[ 167:160] <= rx_byte;
                                5'd22: dist_accum[ 191:184] <= rx_byte;
                                5'd23: dist_accum[ 183:176] <= rx_byte;
                                default: ;
                            endcase
                            byte_idx <= byte_idx + 5'd1;
                        end else if (byte_idx == 5'd24) begin
                            // Health MSB: rx_byte[3:0] → bits [11:8] of health
                            health_accum[11:8] <= rx_byte[3:0];
                            byte_idx           <= 5'd25;
                        end else begin
                            // byte_idx == 25: health LSB → bits [7:0]
                            health_accum[7:0] <= rx_byte;
                            byte_idx          <= 5'd26;
                            state             <= S_CKSUM;
                        end
                    end

                    // ── Checksum byte ───────────────────────────────────────
                    S_CKSUM: begin
                        if (rx_byte == running_xor) begin
                            state <= S_END;
                        end else begin
                            // Checksum mismatch — abort, count error
                            state <= S_IDLE;
                            if (error_count != 16'hFFFF)
                                error_count <= error_count + 16'd1;
                        end
                    end

                    // ── End-of-frame byte ───────────────────────────────────
                    S_END: begin
                        if (rx_byte == EOF_BYTE) begin
                            // Commit parsed frame to output registers
                            distance_mm_vec <= dist_accum;
                            sensor_health   <= health_accum;
                            frame_valid     <= 1'b1;
                            if (frame_count != 16'hFFFF)
                                frame_count <= frame_count + 16'd1;
                        end else begin
                            if (error_count != 16'hFFFF)
                                error_count <= error_count + 16'd1;
                        end
                        state <= S_IDLE;
                    end

                    default: state <= S_IDLE;
                endcase
            end
        end
    end

endmodule
