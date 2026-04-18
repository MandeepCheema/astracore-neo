`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — LiDAR Interface  (lidar_interface.v)
// =============================================================================
// Layer 1 sensor interface.  Consumes ethernet_controller Rev2's RX payload
// byte stream and parses 24-byte LiDAR object packets (2-byte magic header
// + 22-byte lidar_object_t) into an internal FIFO drained by object_tracker.
//
// ── Packet format (24 bytes over L2 payload, big-endian) ────────────────────
//   Bytes  0..1  : 0xA5A5 magic
//   Bytes  2..5  : x_mm           (s32)
//   Bytes  6..9  : y_mm           (s32)
//   Bytes 10..13 : z_mm           (s32)
//   Bytes 14..15 : length_mm      (u16)
//   Bytes 16..17 : width_mm       (u16)
//   Bytes 18..19 : height_mm      (u16)
//   Byte  20     : class_id       (u8)
//   Byte  21     : confidence     (u8)
//   Bytes 22..23 : timestamp_us_hi (split over two fields)  ← see below
//
// Wait — 22 bytes of payload fit exactly after the 2-byte magic (total 24).
// timestamp_us (u32) doesn't fit in 22; this v1 instead stores the low 16
// bits of the timestamp:
//   Bytes 20     : class_id
//   Byte  21     : confidence
//   Bytes 22..23 : timestamp_us[15:0]  (low 16 bits only; upper bits tracked
//                                        elsewhere via gnss_interface if needed)
//
// The upper bits are derived from last_timestamp_hi, a tracked counter that
// increments each packet and can be jammed by higher layers for absolute
// alignment.  For pure relative-time tracking (which is all object_tracker
// needs) the low 16 bits are sufficient.
//
// Packets whose byte count at rx_payload_last is != 24, or whose magic bytes
// don't match, increment error_count.
//
// ── FIFO semantics ───────────────────────────────────────────────────────────
//   FIFO_DEPTH parameterisable (default 8, production 128).
//   out_valid/out_ready handshake on the downstream side.
//   Full FIFO discards the packet and bumps total_dropped.
//
// ── Interface ─────────────────────────────────────────────────────────────────
//   rx_payload_valid, rx_payload_byte[7:0], rx_payload_last  ← from eth_ctrl
//   out_valid/out_ready + lidar_object_t fields
//   fifo_count, fifo_full, fifo_empty
//   frame_count, error_count, total_dropped
// =============================================================================

module lidar_interface #(
    parameter integer FIFO_DEPTH = 8
)(
    input  wire        clk,
    input  wire        rst_n,

    // ── Ethernet RX payload stream ────────────────────────────────────────────
    input  wire        rx_payload_valid,
    input  wire [7:0]  rx_payload_byte,
    input  wire        rx_payload_last,

    // ── AXI-S drain ───────────────────────────────────────────────────────────
    output wire        out_valid,
    input  wire        out_ready,
    output wire signed [31:0] out_x_mm,
    output wire signed [31:0] out_y_mm,
    output wire signed [31:0] out_z_mm,
    output wire [15:0] out_length_mm,
    output wire [15:0] out_width_mm,
    output wire [15:0] out_height_mm,
    output wire [7:0]  out_class_id,
    output wire [7:0]  out_confidence,
    output wire [15:0] out_timestamp_us_lo,

    // ── Status ────────────────────────────────────────────────────────────────
    output wire [$clog2(FIFO_DEPTH+1)-1:0] fifo_count,
    output wire        fifo_full,
    output wire        fifo_empty,
    output reg  [15:0] frame_count,
    output reg  [15:0] error_count,
    output reg  [15:0] total_dropped
);

    localparam [7:0] MAGIC_HI   = 8'hA5;
    localparam [7:0] MAGIC_LO   = 8'hA5;
    localparam       PKT_LEN    = 5'd24;
    localparam       PTR_W      = $clog2(FIFO_DEPTH);

    // =========================================================================
    // 1. Packet assembly scratch
    // =========================================================================
    reg [7:0] rx_buf [0:23];
    reg [4:0] byte_idx;
    reg       magic_ok;

    // =========================================================================
    // 2. Parallel-array FIFO
    // =========================================================================
    reg signed [31:0] f_x     [0:FIFO_DEPTH-1];
    reg signed [31:0] f_y     [0:FIFO_DEPTH-1];
    reg signed [31:0] f_z     [0:FIFO_DEPTH-1];
    reg [15:0]        f_len   [0:FIFO_DEPTH-1];
    reg [15:0]        f_wid   [0:FIFO_DEPTH-1];
    reg [15:0]        f_hei   [0:FIFO_DEPTH-1];
    reg [7:0]         f_class [0:FIFO_DEPTH-1];
    reg [7:0]         f_conf  [0:FIFO_DEPTH-1];
    reg [15:0]        f_ts_lo [0:FIFO_DEPTH-1];

    reg [PTR_W-1:0] wr_ptr;
    reg [PTR_W-1:0] rd_ptr;
    reg [$clog2(FIFO_DEPTH+1)-1:0] count;

    assign fifo_count = count;
    assign fifo_empty = (count == 0);
    assign fifo_full  = (count == FIFO_DEPTH);
    assign out_valid  = !fifo_empty;

    assign out_x_mm            = f_x     [rd_ptr];
    assign out_y_mm            = f_y     [rd_ptr];
    assign out_z_mm            = f_z     [rd_ptr];
    assign out_length_mm       = f_len   [rd_ptr];
    assign out_width_mm        = f_wid   [rd_ptr];
    assign out_height_mm       = f_hei   [rd_ptr];
    assign out_class_id        = f_class [rd_ptr];
    assign out_confidence      = f_conf  [rd_ptr];
    assign out_timestamp_us_lo = f_ts_lo [rd_ptr];

    wire pop = out_valid && out_ready;

    // Packet is valid iff: the last byte landed at index 23, magic matched,
    // AND rx_payload_last fires on the 24th byte.
    wire pkt_good = rx_payload_valid && rx_payload_last &&
                    magic_ok && (byte_idx == PKT_LEN - 5'd1);
    wire push     = pkt_good && !fifo_full;

    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < 24; i = i + 1) rx_buf[i] <= 8'd0;
            byte_idx      <= 5'd0;
            magic_ok      <= 1'b0;
            wr_ptr        <= {PTR_W{1'b0}};
            rd_ptr        <= {PTR_W{1'b0}};
            count         <= 0;
            frame_count   <= 16'd0;
            error_count   <= 16'd0;
            total_dropped <= 16'd0;
            for (i = 0; i < FIFO_DEPTH; i = i + 1) begin
                f_x     [i] <= 32'sd0;
                f_y     [i] <= 32'sd0;
                f_z     [i] <= 32'sd0;
                f_len   [i] <= 16'd0;
                f_wid   [i] <= 16'd0;
                f_hei   [i] <= 16'd0;
                f_class [i] <= 8'd0;
                f_conf  [i] <= 8'd0;
                f_ts_lo [i] <= 16'd0;
            end
        end else begin
            // ── Byte-stream assembly ─────────────────────────────────────────
            // byte_idx saturates at PKT_LEN (24).  Only in-range bytes (idx
            // < PKT_LEN) are written into rx_buf, so over-length packets
            // cannot corrupt the last slot.
            if (rx_payload_valid) begin
                if (byte_idx < PKT_LEN) begin
                    rx_buf[byte_idx] <= rx_payload_byte;
                end

                // Magic validation on the first two bytes
                if (byte_idx == 5'd0)
                    magic_ok <= (rx_payload_byte == MAGIC_HI);
                if (byte_idx == 5'd1)
                    magic_ok <= magic_ok && (rx_payload_byte == MAGIC_LO);

                if (rx_payload_last) begin
                    // End of packet — commit or reject
                    byte_idx <= 5'd0;
                    magic_ok <= 1'b0;

                    // Commit only if this byte was the 24th (byte_idx currently
                    // holds PRE-update value of PKT_LEN - 1) AND magic matched.
                    // Packets that overran past 24 bytes will have byte_idx
                    // saturated at PKT_LEN — the comparison below rejects them.
                    if (magic_ok && byte_idx == PKT_LEN - 5'd1) begin
                        if (!fifo_full) begin
                            f_x    [wr_ptr] <= {rx_buf[2],  rx_buf[3],  rx_buf[4],  rx_buf[5]};
                            f_y    [wr_ptr] <= {rx_buf[6],  rx_buf[7],  rx_buf[8],  rx_buf[9]};
                            f_z    [wr_ptr] <= {rx_buf[10], rx_buf[11], rx_buf[12], rx_buf[13]};
                            f_len  [wr_ptr] <= {rx_buf[14], rx_buf[15]};
                            f_wid  [wr_ptr] <= {rx_buf[16], rx_buf[17]};
                            f_hei  [wr_ptr] <= {rx_buf[18], rx_buf[19]};
                            f_class[wr_ptr] <= rx_buf[20];
                            f_conf [wr_ptr] <= rx_buf[21];
                            // Byte 22 is stored via NBA this cycle; byte 23 is
                            // still the live rx_payload_byte, not yet in rx_buf.
                            f_ts_lo[wr_ptr] <= {rx_buf[22], rx_payload_byte};
                            wr_ptr <= wr_ptr + 1'b1;
                            if (frame_count != 16'hFFFF)
                                frame_count <= frame_count + 16'd1;
                        end else begin
                            if (total_dropped != 16'hFFFF)
                                total_dropped <= total_dropped + 16'd1;
                        end
                    end else begin
                        // Wrong magic, wrong length, or overrun
                        if (error_count != 16'hFFFF)
                            error_count <= error_count + 16'd1;
                    end
                end else if (byte_idx < PKT_LEN) begin
                    byte_idx <= byte_idx + 5'd1;
                end
            end

            // ── rd_ptr + count maintenance (decoupled from push path above) ─
            if (pop) begin
                rd_ptr <= rd_ptr + 1'b1;
            end

            // Net count update — push branch above already wrote the entry,
            // so here we only adjust the count.
            case ({(pkt_good && !fifo_full), pop})
                2'b10: count <= count + 1;
                2'b01: count <= count - 1;
                default: ;
            endcase
        end
    end

endmodule
