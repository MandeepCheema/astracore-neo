`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — NPU DMA Engine  (npu_dma.v)
// =============================================================================
// Tiled memory-to-memory transfer with configurable source stride and
// zero-padding on all four edges.  Sits between external memory (DDR /
// LPDDR5 via the memory controller) and the on-chip SRAM scratchpad.
//
// ── Logical transfer ────────────────────────────────────────────────────────
//   Output region size: (tile_h + pad_top + pad_bot) × (tile_w + pad_left + pad_right)
//   Destination layout: row-major, linear from cfg_dst_addr.
//
//   For output position (i, j), i in [0, output_h), j in [0, output_w):
//     if pad_top ≤ i < pad_top + tile_h  AND  pad_left ≤ j < pad_left + tile_w
//       → fetch src[cfg_src_addr + (i - pad_top) × src_stride + (j - pad_left)]
//     else
//       → write 0  (zero-padding, no source read)
//
//   This is the standard behaviour a conv-style compute layer expects:
//   a feature-map tile surrounded by a border of zeros so a 3×3 conv can
//   hit the edge pixels without out-of-bounds access.
//
// ── Pipeline ────────────────────────────────────────────────────────────────
//   2 stages.  Steady-state throughput = 1 output position per cycle.
//
//   Stage A (this cycle)
//     • holds (i_pos, j_pos) — the position whose source to fetch THIS cycle
//     • drives mem_re + mem_raddr if the position is real
//     • advances (i_pos, j_pos) on the clock edge
//
//   Stage B (registered from A)
//     • 1 cycle later, writes the destination word
//     • dst_wdata = mem_rdata (real) or 0 (pad)
//     • dst_waddr = cfg_dst_addr + linear counter
//
//   Total transfer: (output_h × output_w) + 1 fill + 1 done-pulse cycles.
//
// ── Handshake ───────────────────────────────────────────────────────────────
//   start  — caller pulses high for one cycle with config inputs stable.
//            Ignored while busy=1.
//   busy   — high from the cycle after start until just before done.
//   done   — 1-cycle pulse after the final destination write has committed.
//
// ── Parameter constraints (caller's responsibility) ────────────────────────
//   • tile_h + pad_top + pad_bot ≥ 1 and tile_w + pad_left + pad_right ≥ 1.
//     Zero-size outputs are undefined behaviour (the FSM's last-position
//     detection uses output_w - 1 which underflows at output_w = 0).
//   • (tile_h × src_stride) must fit in LEN_W bits.  The internal multiply
//     `src_row * src_stride_r` is LEN_W × LEN_W → LEN_W (Verilog truncates
//     to max operand width).  For LEN_W = 16 this caps at 65,536.
//   • output_h × output_w must fit in DST_ADDR_W bits.  No overflow check.
//
// ── V1 gaps (deferred to later revisions) ───────────────────────────────────
//   • Narrow destination port (DATA_W bits/cycle).  Loading the wide AI/AO
//     SRAM banks needs a packer in V2.
//   • Single channel; no concurrent transfers.
//   • No abort, no error reporting, no interrupts.
//   • No DDR burst issuing — one memory read per real position.
//   • No backpressure on mem read — external memory is assumed always-ready
//     with 1-cycle latency.
// =============================================================================

module npu_dma #(
    parameter integer DATA_W     = 8,
    parameter integer SRC_ADDR_W = 32,
    parameter integer DST_ADDR_W = 16,
    parameter integer LEN_W      = 16
)(
    input  wire                        clk,
    input  wire                        rst_n,

    // Configuration — sampled on start when idle.
    input  wire [SRC_ADDR_W-1:0]       cfg_src_addr,
    input  wire [DST_ADDR_W-1:0]       cfg_dst_addr,
    input  wire [LEN_W-1:0]            cfg_tile_h,
    input  wire [LEN_W-1:0]            cfg_tile_w,
    input  wire [LEN_W-1:0]            cfg_src_stride,
    input  wire [3:0]                  cfg_pad_top,
    input  wire [3:0]                  cfg_pad_bot,
    input  wire [3:0]                  cfg_pad_left,
    input  wire [3:0]                  cfg_pad_right,

    // Control
    input  wire                        start,
    output reg                         busy,
    output reg                         done,

    // Source memory read port (1-cycle latency)
    output wire                        mem_re,
    output wire [SRC_ADDR_W-1:0]       mem_raddr,
    input  wire [DATA_W-1:0]           mem_rdata,

    // Destination write port (narrow, 1 word per cycle)
    output wire                        dst_we,
    output wire [DST_ADDR_W-1:0]       dst_waddr,
    output wire [DATA_W-1:0]           dst_wdata
);

    // =========================================================================
    // Latched configuration (stable for the duration of one transfer)
    // =========================================================================
    reg [SRC_ADDR_W-1:0] src_addr_r;
    reg [DST_ADDR_W-1:0] dst_addr_r;
    reg [LEN_W-1:0]      tile_h_r;
    reg [LEN_W-1:0]      tile_w_r;
    reg [LEN_W-1:0]      src_stride_r;
    reg [3:0]            pad_top_r;
    reg [3:0]            pad_bot_r;
    reg [3:0]            pad_left_r;
    reg [3:0]            pad_right_r;

    wire [LEN_W-1:0] output_h = tile_h_r + {{(LEN_W-4){1'b0}}, pad_top_r}
                                          + {{(LEN_W-4){1'b0}}, pad_bot_r};
    wire [LEN_W-1:0] output_w = tile_w_r + {{(LEN_W-4){1'b0}}, pad_left_r}
                                          + {{(LEN_W-4){1'b0}}, pad_right_r};

    // =========================================================================
    // Stage A — position registers and source-address computation
    // =========================================================================
    reg                    busy_a;
    reg  [LEN_W-1:0]       i_pos;
    reg  [LEN_W-1:0]       j_pos;
    reg  [DST_ADDR_W-1:0]  dst_offset_a;

    wire i_in_tile =
        (i_pos >= {{(LEN_W-4){1'b0}}, pad_top_r}) &&
        (i_pos < (tile_h_r + {{(LEN_W-4){1'b0}}, pad_top_r}));
    wire j_in_tile =
        (j_pos >= {{(LEN_W-4){1'b0}}, pad_left_r}) &&
        (j_pos < (tile_w_r + {{(LEN_W-4){1'b0}}, pad_left_r}));
    wire is_real_a = busy_a && i_in_tile && j_in_tile;

    wire j_is_last   = (j_pos == output_w - 1);
    wire i_is_last   = (i_pos == output_h - 1);
    wire is_last_pos = j_is_last && i_is_last;

    // Source address (only valid when is_real_a)
    wire [LEN_W-1:0] src_row = i_pos - {{(LEN_W-4){1'b0}}, pad_top_r};
    wire [LEN_W-1:0] src_col = j_pos - {{(LEN_W-4){1'b0}}, pad_left_r};

    assign mem_re    = is_real_a;
    assign mem_raddr = src_addr_r +
                       {{(SRC_ADDR_W-LEN_W){1'b0}}, (src_row * src_stride_r)} +
                       {{(SRC_ADDR_W-LEN_W){1'b0}}, src_col};

    // =========================================================================
    // Stage B — write registers (registered from stage A)
    // =========================================================================
    reg                    busy_b;
    reg                    is_real_b;
    reg  [DST_ADDR_W-1:0]  dst_offset_b;

    assign dst_we    = busy_b;
    assign dst_waddr = dst_addr_r + dst_offset_b;
    assign dst_wdata = is_real_b ? mem_rdata : {DATA_W{1'b0}};

    // =========================================================================
    // Sequential — FSM, position advance, pipeline register, done pulse
    // =========================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            busy         <= 1'b0;
            done         <= 1'b0;
            busy_a       <= 1'b0;
            busy_b       <= 1'b0;
            i_pos        <= {LEN_W{1'b0}};
            j_pos        <= {LEN_W{1'b0}};
            dst_offset_a <= {DST_ADDR_W{1'b0}};
            dst_offset_b <= {DST_ADDR_W{1'b0}};
            is_real_b    <= 1'b0;
            src_addr_r   <= {SRC_ADDR_W{1'b0}};
            dst_addr_r   <= {DST_ADDR_W{1'b0}};
            tile_h_r     <= {LEN_W{1'b0}};
            tile_w_r     <= {LEN_W{1'b0}};
            src_stride_r <= {LEN_W{1'b0}};
            pad_top_r    <= 4'd0;
            pad_bot_r    <= 4'd0;
            pad_left_r   <= 4'd0;
            pad_right_r  <= 4'd0;
        end else begin
            // default: done is a 1-cycle pulse
            done <= 1'b0;

            // ── Start of a new transfer ──────────────────────────────────────
            if (start && !busy) begin
                src_addr_r   <= cfg_src_addr;
                dst_addr_r   <= cfg_dst_addr;
                tile_h_r     <= cfg_tile_h;
                tile_w_r     <= cfg_tile_w;
                src_stride_r <= cfg_src_stride;
                pad_top_r    <= cfg_pad_top;
                pad_bot_r    <= cfg_pad_bot;
                pad_left_r   <= cfg_pad_left;
                pad_right_r  <= cfg_pad_right;
                busy         <= 1'b1;
                busy_a       <= 1'b1;
                i_pos        <= {LEN_W{1'b0}};
                j_pos        <= {LEN_W{1'b0}};
                dst_offset_a <= {DST_ADDR_W{1'b0}};
            end

            // ── Stage A: advance through positions ───────────────────────────
            if (busy_a) begin
                if (is_last_pos) begin
                    // Final position is being issued this cycle; next cycle
                    // stage A goes idle while stage B drains.
                    busy_a <= 1'b0;
                end else if (j_is_last) begin
                    j_pos        <= {LEN_W{1'b0}};
                    i_pos        <= i_pos + 1'b1;
                    dst_offset_a <= dst_offset_a + 1'b1;
                end else begin
                    j_pos        <= j_pos + 1'b1;
                    dst_offset_a <= dst_offset_a + 1'b1;
                end
            end

            // ── Pipeline: Stage A → Stage B registers (every cycle) ──────────
            busy_b       <= busy_a;
            is_real_b    <= is_real_a;
            dst_offset_b <= dst_offset_a;

            // ── Transfer completion ──────────────────────────────────────────
            // When stage A finished last cycle (busy_b now =1 but busy_a =0),
            // the CURRENT cycle is the final write.  Drop busy and pulse done
            // for NEXT cycle (so caller observing done=1 knows all writes are
            // committed to destination).
            if (busy_b && !busy_a) begin
                busy <= 1'b0;
                done <= 1'b1;
            end
        end
    end

endmodule
