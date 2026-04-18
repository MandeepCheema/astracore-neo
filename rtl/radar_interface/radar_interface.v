`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — Radar Interface  (radar_interface.v)
// =============================================================================
// Layer 1 sensor interface.  Deserialises SPI-framed radar_object_t records
// from an external automotive radar sensor into an internal FIFO consumed
// by object_tracker via an AXI4-Stream-style valid/ready port.
//
// ── radar_object_t (13 bytes big-endian) ─────────────────────────────────────
//   Bytes 0-1  : range_cm        (s16)
//   Bytes 2-3  : velocity_cms    (s16 — negative = approaching)
//   Bytes 4-5  : azimuth_mdeg    (s16 — 0 = straight ahead)
//   Bytes 6-7  : rcs_dbsm        (u16 — radar cross section)
//   Byte  8    : confidence      (u8)
//   Bytes 9-12 : timestamp_us    (u32 big-endian)
//
//   Frame boundary is delivered via spi_frame_end (1+ clocks after the last
//   byte), matching the imu_interface contract.  A frame with the wrong
//   byte count on frame_end is rejected and error_count increments.
//
// ── FIFO semantics ───────────────────────────────────────────────────────────
//   FIFO_DEPTH parameterisable (default 16, production 64).
//   out_valid / out_ready handshake on the downstream side.
//   On a successful frame parse the object is enqueued (or dropped if full).
//
// ── Interface ─────────────────────────────────────────────────────────────────
//   spi_byte_valid, spi_byte[7:0], spi_frame_end
//   out_valid, out_ready,
//   out_range_cm, out_velocity_cms, out_azimuth_mdeg,
//   out_rcs_dbsm, out_confidence, out_timestamp_us
//   fifo_count, fifo_full, fifo_empty
//   frame_count, error_count, total_dropped
// =============================================================================

module radar_interface #(
    parameter integer FIFO_DEPTH = 16
)(
    input  wire        clk,
    input  wire        rst_n,

    // ── SPI byte stream input ─────────────────────────────────────────────────
    input  wire        spi_byte_valid,
    input  wire [7:0]  spi_byte,
    input  wire        spi_frame_end,

    // ── AXI-Stream style output (1 object per transaction) ──────────────────
    output wire        out_valid,
    input  wire        out_ready,
    output wire signed [15:0] out_range_cm,
    output wire signed [15:0] out_velocity_cms,
    output wire signed [15:0] out_azimuth_mdeg,
    output wire        [15:0] out_rcs_dbsm,
    output wire        [7:0]  out_confidence,
    output wire        [31:0] out_timestamp_us,

    // ── Status ────────────────────────────────────────────────────────────────
    output wire [$clog2(FIFO_DEPTH+1)-1:0] fifo_count,
    output wire        fifo_full,
    output wire        fifo_empty,
    output reg  [15:0] frame_count,
    output reg  [15:0] error_count,
    output reg  [15:0] total_dropped
);

    localparam integer FRAME_LEN = 13;
    localparam PTR_W = $clog2(FIFO_DEPTH);

    // =========================================================================
    // 1. Frame-assembly scratch buffer
    // =========================================================================
    reg [7:0] rx_buf [0:12];
    reg [3:0] byte_count;

    // =========================================================================
    // 2. Parallel-array FIFO (same pattern as cam_detection_receiver)
    // =========================================================================
    reg signed [15:0] f_range   [0:FIFO_DEPTH-1];
    reg signed [15:0] f_vel     [0:FIFO_DEPTH-1];
    reg signed [15:0] f_az      [0:FIFO_DEPTH-1];
    reg [15:0]        f_rcs     [0:FIFO_DEPTH-1];
    reg [7:0]         f_conf    [0:FIFO_DEPTH-1];
    reg [31:0]        f_ts      [0:FIFO_DEPTH-1];

    reg [PTR_W-1:0] wr_ptr;
    reg [PTR_W-1:0] rd_ptr;
    reg [$clog2(FIFO_DEPTH+1)-1:0] count;

    assign fifo_count = count;
    assign fifo_empty = (count == 0);
    assign fifo_full  = (count == FIFO_DEPTH);
    assign out_valid  = !fifo_empty;

    assign out_range_cm     = f_range[rd_ptr];
    assign out_velocity_cms = f_vel  [rd_ptr];
    assign out_azimuth_mdeg = f_az   [rd_ptr];
    assign out_rcs_dbsm     = f_rcs  [rd_ptr];
    assign out_confidence   = f_conf [rd_ptr];
    assign out_timestamp_us = f_ts   [rd_ptr];

    wire pop = out_valid && out_ready;

    // Push decision: valid frame boundary and room in FIFO
    wire frame_good = spi_frame_end && (byte_count == FRAME_LEN[3:0]);
    wire push = frame_good && !fifo_full;

    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < FRAME_LEN; i = i + 1) rx_buf[i] <= 8'd0;
            byte_count     <= 4'd0;
            wr_ptr         <= {PTR_W{1'b0}};
            rd_ptr         <= {PTR_W{1'b0}};
            count          <= 0;
            frame_count    <= 16'd0;
            error_count    <= 16'd0;
            total_dropped  <= 16'd0;
            for (i = 0; i < FIFO_DEPTH; i = i + 1) begin
                f_range[i] <= 16'sd0;
                f_vel  [i] <= 16'sd0;
                f_az   [i] <= 16'sd0;
                f_rcs  [i] <= 16'd0;
                f_conf [i] <= 8'd0;
                f_ts   [i] <= 32'd0;
            end
        end else begin
            // ── Accumulate SPI bytes ─────────────────────────────────────────
            if (spi_byte_valid) begin
                if (byte_count < FRAME_LEN[3:0]) begin
                    rx_buf[byte_count] <= spi_byte;
                    byte_count         <= byte_count + 4'd1;
                end
            end

            // ── Frame-end handling: push valid frames, count errors/drops ───
            if (spi_frame_end) begin
                byte_count <= 4'd0;
                if (frame_good) begin
                    if (!fifo_full) begin
                        f_range[wr_ptr] <= {rx_buf[0],  rx_buf[1]};
                        f_vel  [wr_ptr] <= {rx_buf[2],  rx_buf[3]};
                        f_az   [wr_ptr] <= {rx_buf[4],  rx_buf[5]};
                        f_rcs  [wr_ptr] <= {rx_buf[6],  rx_buf[7]};
                        f_conf [wr_ptr] <= rx_buf[8];
                        f_ts   [wr_ptr] <= {rx_buf[9], rx_buf[10],
                                            rx_buf[11], rx_buf[12]};
                        wr_ptr <= wr_ptr + 1'b1;
                        if (frame_count != 16'hFFFF)
                            frame_count <= frame_count + 16'd1;
                    end else begin
                        if (total_dropped != 16'hFFFF)
                            total_dropped <= total_dropped + 16'd1;
                    end
                end else begin
                    if (error_count != 16'hFFFF)
                        error_count <= error_count + 16'd1;
                end
            end

            // ── rd_ptr advance on pop ────────────────────────────────────────
            if (pop)
                rd_ptr <= rd_ptr + 1'b1;

            // ── Count update: net change from (push, pop) ───────────────────
            case ({push, pop})
                2'b10: count <= count + 1;
                2'b01: count <= count - 1;
                default: ;
            endcase
        end
    end

endmodule
