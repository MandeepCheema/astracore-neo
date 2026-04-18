`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — Camera Detection Receiver  (cam_detection_receiver.v)
// =============================================================================
// Layer 1 sensor interface.  Buffers camera_detection_t records written by
// an external CNN (over AXI4-Lite, bridged to this module's parallel write
// port) into an internal FIFO that downstream fusion stages drain via a
// standard valid/ready handshake.
//
// ── camera_detection_t fields (all parallel regs, no bit-packing) ────────────
//   class_id[15:0], confidence[15:0],
//   bbox_x[15:0], bbox_y[15:0], bbox_w[15:0], bbox_h[15:0],
//   timestamp_us[31:0], camera_id[7:0]
//
// ── FIFO semantics ───────────────────────────────────────────────────────────
//   wr_valid + wr_ready handshake on push (wr_ready = !fifo_full).
//   rd_valid + rd_ready handshake on pop  (rd_valid = !fifo_empty).
//   A write while fifo_full is silently dropped and total_dropped++.
//
//   Simultaneous push + pop is supported: the head reads the current front
//   entry (pre-push); new entry lands at tail.
//
// ── Parameters ────────────────────────────────────────────────────────────────
//   FIFO_DEPTH — power of 2 (default 16; production: 256)
// =============================================================================

module cam_detection_receiver #(
    parameter integer FIFO_DEPTH = 16
)(
    input  wire        clk,
    input  wire        rst_n,

    // ── Write port (from AXI-Lite bridge / CNN host) ─────────────────────────
    input  wire        wr_valid,
    input  wire [15:0] wr_class_id,
    input  wire [15:0] wr_confidence,
    input  wire [15:0] wr_bbox_x,
    input  wire [15:0] wr_bbox_y,
    input  wire [15:0] wr_bbox_w,
    input  wire [15:0] wr_bbox_h,
    input  wire [31:0] wr_timestamp_us,
    input  wire [7:0]  wr_camera_id,
    output wire        wr_ready,

    // ── Read port (AXI-Stream style, to downstream fusion) ───────────────────
    output wire        rd_valid,
    input  wire        rd_ready,
    output wire [15:0] rd_class_id,
    output wire [15:0] rd_confidence,
    output wire [15:0] rd_bbox_x,
    output wire [15:0] rd_bbox_y,
    output wire [15:0] rd_bbox_w,
    output wire [15:0] rd_bbox_h,
    output wire [31:0] rd_timestamp_us,
    output wire [7:0]  rd_camera_id,

    // ── Status ────────────────────────────────────────────────────────────────
    output wire [$clog2(FIFO_DEPTH+1)-1:0] fifo_count,
    output wire        fifo_full,
    output wire        fifo_empty,
    output reg  [15:0] total_received,
    output reg  [15:0] total_dropped
);

    localparam PTR_W = $clog2(FIFO_DEPTH);

    reg [15:0] f_class_id    [0:FIFO_DEPTH-1];
    reg [15:0] f_confidence  [0:FIFO_DEPTH-1];
    reg [15:0] f_bbox_x      [0:FIFO_DEPTH-1];
    reg [15:0] f_bbox_y      [0:FIFO_DEPTH-1];
    reg [15:0] f_bbox_w      [0:FIFO_DEPTH-1];
    reg [15:0] f_bbox_h      [0:FIFO_DEPTH-1];
    reg [31:0] f_timestamp   [0:FIFO_DEPTH-1];
    reg [7:0]  f_camera_id   [0:FIFO_DEPTH-1];

    reg [PTR_W-1:0] wr_ptr;
    reg [PTR_W-1:0] rd_ptr;
    reg [$clog2(FIFO_DEPTH+1)-1:0] count;

    assign fifo_count = count;
    assign fifo_empty = (count == 0);
    assign fifo_full  = (count == FIFO_DEPTH);
    assign wr_ready   = !fifo_full;
    assign rd_valid   = !fifo_empty;

    assign rd_class_id    = f_class_id   [rd_ptr];
    assign rd_confidence  = f_confidence [rd_ptr];
    assign rd_bbox_x      = f_bbox_x     [rd_ptr];
    assign rd_bbox_y      = f_bbox_y     [rd_ptr];
    assign rd_bbox_w      = f_bbox_w     [rd_ptr];
    assign rd_bbox_h      = f_bbox_h     [rd_ptr];
    assign rd_timestamp_us= f_timestamp  [rd_ptr];
    assign rd_camera_id   = f_camera_id  [rd_ptr];

    wire push = wr_valid && !fifo_full;
    wire pop  = rd_valid && rd_ready;

    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            wr_ptr         <= {PTR_W{1'b0}};
            rd_ptr         <= {PTR_W{1'b0}};
            count          <= 0;
            total_received <= 16'd0;
            total_dropped  <= 16'd0;
            for (i = 0; i < FIFO_DEPTH; i = i + 1) begin
                f_class_id   [i] <= 16'd0;
                f_confidence [i] <= 16'd0;
                f_bbox_x     [i] <= 16'd0;
                f_bbox_y     [i] <= 16'd0;
                f_bbox_w     [i] <= 16'd0;
                f_bbox_h     [i] <= 16'd0;
                f_timestamp  [i] <= 32'd0;
                f_camera_id  [i] <= 8'd0;
            end
        end else begin
            // Dropped-write accounting
            if (wr_valid && fifo_full && total_dropped != 16'hFFFF)
                total_dropped <= total_dropped + 16'd1;

            // Push
            if (push) begin
                f_class_id   [wr_ptr] <= wr_class_id;
                f_confidence [wr_ptr] <= wr_confidence;
                f_bbox_x     [wr_ptr] <= wr_bbox_x;
                f_bbox_y     [wr_ptr] <= wr_bbox_y;
                f_bbox_w     [wr_ptr] <= wr_bbox_w;
                f_bbox_h     [wr_ptr] <= wr_bbox_h;
                f_timestamp  [wr_ptr] <= wr_timestamp_us;
                f_camera_id  [wr_ptr] <= wr_camera_id;
                wr_ptr <= wr_ptr + 1'b1;
                if (total_received != 16'hFFFF)
                    total_received <= total_received + 16'd1;
            end

            // Pop
            if (pop) begin
                rd_ptr <= rd_ptr + 1'b1;
            end

            // Count update — handle simultaneous push/pop correctly
            case ({push, pop})
                2'b10: count <= count + 1;
                2'b01: count <= count - 1;
                default: ;   // both or neither: net zero
            endcase
        end
    end

endmodule
