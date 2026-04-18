`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — PTP Clock Sync  (ptp_clock_sync.v)
// =============================================================================
// Layer 1 support module.  IEEE 1588 / gPTP grandmaster role.  Consumes the
// 64-bit absolute μs time from gnss_interface and periodically emits a Sync
// frame through ethernet_controller's TX byte pipeline so downstream ECUs
// on the vehicle network can share a common time base.
//
// Scope of this v1 implementation:
//   • Master (grandmaster) role only.  Slave offset/delay computation is a
//     future upgrade; the architecture keeps this module authoritative since
//     GNSS+PPS already anchors us to UTC.
//   • Simplified 16-byte Sync frame (custom internal format — the hook to
//     wrap in a full 802.1AS L2 frame header lives in ethernet_controller).
//   • Back-pressure aware: transmission advances only when tx_ready is high.
//
// ── Sync frame layout (16 bytes, big-endian) ─────────────────────────────────
//   Bytes 0-1  : SYNC_MAGIC                  (default 0xAA55)
//   Bytes 2-3  : message_type                (0x0001 = SYNC)
//   Bytes 4-11 : master_time_us              (64-bit absolute μs, latched at
//                                             TX start so time is atomic)
//   Bytes 12-15: sequence_id                 (monotonic per frame)
//
// ── FSM ──────────────────────────────────────────────────────────────────────
//   S_IDLE : increment sync_timer_ms on each tick_1ms.  When it reaches
//            SYNC_INTERVAL_MS, latch master_time_us into tx_time, advance
//            sequence, reset timer, enter S_TX.
//   S_TX   : drive one byte per cycle onto tx_valid/tx_byte_in, asserting
//            tx_last on the 16th byte.  Stall when tx_ready is low.
//            Return to S_IDLE after the final accepted byte.
//
// ── Parameters ────────────────────────────────────────────────────────────────
//   SYNC_INTERVAL_MS — time between Sync frames (default 125 ms, gPTP typical)
//   SYNC_MAGIC       — 16-bit magic identifier
// =============================================================================

module ptp_clock_sync #(
    parameter [15:0] SYNC_INTERVAL_MS = 16'd125,
    parameter [15:0] SYNC_MAGIC       = 16'hAA55
)(
    input  wire        clk,
    input  wire        rst_n,

    // ── Time source (from gnss_interface) ────────────────────────────────────
    input  wire [63:0] master_time_us,
    input  wire        tick_1ms,

    // ── TX byte pipeline (to ethernet_controller Rev2 tx_* port) ─────────────
    output reg         tx_valid,
    output reg  [7:0]  tx_byte_in,
    output reg         tx_last,
    input  wire        tx_ready,

    // ── Status ────────────────────────────────────────────────────────────────
    output reg  [31:0] sync_sequence,
    output reg  [63:0] last_sync_time_us,
    output reg  [15:0] sync_count
);

    localparam [15:0] MSG_TYPE_SYNC = 16'h0001;
    localparam FRAME_LEN = 5'd16;

    localparam S_IDLE = 1'b0;
    localparam S_TX   = 1'b1;

    reg        state;
    reg [15:0] sync_timer_ms;
    reg [4:0]  byte_idx;
    reg [63:0] tx_time;      // frozen timestamp for the in-flight frame
    reg [31:0] tx_seq;

    // =========================================================================
    // Byte-selection mux — combinatorial next-byte lookup for the frame under
    // transmission.  Indexed by byte_idx in S_TX.
    // =========================================================================
    reg [7:0] frame_byte;
    always @(*) begin
        case (byte_idx)
            5'd0:  frame_byte = SYNC_MAGIC[15:8];
            5'd1:  frame_byte = SYNC_MAGIC[ 7:0];
            5'd2:  frame_byte = MSG_TYPE_SYNC[15:8];
            5'd3:  frame_byte = MSG_TYPE_SYNC[ 7:0];
            5'd4:  frame_byte = tx_time[63:56];
            5'd5:  frame_byte = tx_time[55:48];
            5'd6:  frame_byte = tx_time[47:40];
            5'd7:  frame_byte = tx_time[39:32];
            5'd8:  frame_byte = tx_time[31:24];
            5'd9:  frame_byte = tx_time[23:16];
            5'd10: frame_byte = tx_time[15: 8];
            5'd11: frame_byte = tx_time[ 7: 0];
            5'd12: frame_byte = tx_seq[31:24];
            5'd13: frame_byte = tx_seq[23:16];
            5'd14: frame_byte = tx_seq[15: 8];
            5'd15: frame_byte = tx_seq[ 7: 0];
            default: frame_byte = 8'h00;
        endcase
    end

    wire last_byte = (byte_idx == FRAME_LEN - 5'd1);

    // =========================================================================
    // FSM
    // =========================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            state             <= S_IDLE;
            sync_timer_ms     <= 16'd0;
            byte_idx          <= 5'd0;
            tx_time           <= 64'd0;
            tx_seq            <= 32'd0;
            sync_sequence     <= 32'd0;
            last_sync_time_us <= 64'd0;
            sync_count        <= 16'd0;
            tx_valid          <= 1'b0;
            tx_byte_in        <= 8'h00;
            tx_last           <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    tx_valid <= 1'b0;
                    tx_last  <= 1'b0;

                    if (tick_1ms) begin
                        if (sync_timer_ms >= SYNC_INTERVAL_MS - 16'd1) begin
                            // Interval reached — latch timestamp + seq, start TX
                            sync_timer_ms <= 16'd0;
                            tx_time       <= master_time_us;
                            tx_seq        <= sync_sequence + 32'd1;
                            sync_sequence <= sync_sequence + 32'd1;
                            byte_idx      <= 5'd0;
                            state         <= S_TX;
                        end else begin
                            sync_timer_ms <= sync_timer_ms + 16'd1;
                        end
                    end
                end

                S_TX: begin
                    // Drive the current byte; advance only when consumer accepts
                    tx_valid   <= 1'b1;
                    tx_byte_in <= frame_byte;
                    tx_last    <= last_byte;

                    if (tx_ready) begin
                        if (last_byte) begin
                            // Final byte accepted — commit frame and return to idle
                            last_sync_time_us <= tx_time;
                            if (sync_count != 16'hFFFF)
                                sync_count <= sync_count + 16'd1;
                            state    <= S_IDLE;
                            byte_idx <= 5'd0;
                            // Next-cycle de-assert handled in S_IDLE branch
                        end else begin
                            byte_idx <= byte_idx + 5'd1;
                        end
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
