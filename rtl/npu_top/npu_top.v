`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — NPU Top (npu_top.v)
// =============================================================================
// Structural integration of the NPU subsystem (V1):
//
//   ┌───────────────────┐    ┌──────────────────────┐    ┌─────────────┐
//   │  npu_sram_ctrl    │◄──►│  npu_systolic_array  │    │  npu_tile_  │
//   │  WA/WB/AI/AO/SC   │    │  N_ROWS × N_COLS MVM │    │  ctrl (FSM) │
//   └────────▲──────────┘    └──────────▲───────────┘    └──────┬──────┘
//            │                          │                       │
//            │       ┌──────────────────┴───────────────────────┘
//            │       │
//            │       ▼  (c_vec[0 +: ACC_W])
//            │   ┌──────────────────┐
//            │   │ npu_activation   │  ← column-0 debug AFU (V1)
//            │   └──────────────────┘
//            │
//         external SRAM ports for loading weights/activations and reading
//         outputs.  Caller drives these while npu_top is not busy.
//
// ── V1 data path ────────────────────────────────────────────────────────────
//   1. External loads WA (weights) and AI (activation vectors) via ext_* ports.
//   2. Caller pulses `start` with cfg_k, cfg_ai_base, cfg_ao_base,
//      cfg_acc_init_mode/data, and cfg_afu_mode.
//   3. tile_ctrl FSM:
//        a. PRELOAD: walks SRAM WA, latches into array's internal W grid.
//        b. EXEC_PREP: clears accumulators (init_mode=0) or loads them
//           from cfg_acc_init_data (init_mode=1, k-tile chaining).
//        c. EXECUTE: streams cfg_k activation vectors through the array.
//        d. DRAIN: last DRAIN cycle pulses array_afu_in_valid so the
//           N_COLS writeback AFUs register their activated output.
//        e. STORE: writes activated c_vec into SRAM AO at cfg_ao_base.
//        f. DONE: pulses `done`, returns to IDLE.
//   4. External reads AO via ext_ao_* to retrieve the result.
//
// ── AFU wiring ──────────────────────────────────────────────────────────────
//   Two paths:
//     • Writeback path (gap #3): N_COLS parallel AFUs on the AO path,
//       triggered once per tile by tile_ctrl's array_afu_in_valid pulse.
//       cfg_afu_mode is latched on start (cfg_afu_mode_r) so it's stable
//       across the tile regardless of the port's live value. mode=PASS
//       is a functional no-op and gives raw-c_vec writeback semantics.
//     • Debug tap (unchanged): a single AFU instance consumes c_vec[0]
//       and pulses on array c_valid every EXECUTE cycle, driven by the
//       LIVE cfg_afu_mode port. Useful for observing partial-sum
//       activation under real stimulus.
//
// ── Access arbitration ──────────────────────────────────────────────────────
//   While `busy = 0`:
//     • ext_w_* writes to SRAM WA (w_bank_sel forced to 1 internally so
//       SRAM routes the write to physical bank A).
//     • ext_ai_* writes to AI.
//     • tile_ctrl's outputs are ignored (gated by busy).
//
//   While `busy = 1`:
//     • tile_ctrl drives SRAM control (w_bank_sel=0 so array reads bank A).
//     • ext_w_we and ext_ai_we MUST be held low.  The module does NOT
//       arbitrate — the caller is responsible.  (V2 will add an internal
//       arbiter if multi-master access becomes necessary.)
//
//   ext_ao_re is always live — tile_ctrl only WRITES AO, never reads.
//
// ── V1 gaps ─────────────────────────────────────────────────────────────────
//   • No DMA integration.  Caller loads weights and activations directly.
//   • Single-tile-per-start (tile_ctrl V1 limitation).
//   • No access arbitration: caller must respect busy.
//   • No ECC, no BIST, no debug mux (those are Phase 2 hardening items).
//   • LUT-based AFU modes (SiLU/GELU/Sigmoid/Tanh) still reserved —
//     hardware placeholders only; v2 adds the LUT backends.
// =============================================================================

module npu_top #(
    parameter integer DATA_W         = 8,
    parameter integer ACC_W          = 32,
    parameter integer N_ROWS         = 4,
    parameter integer N_COLS         = 4,
    parameter integer WEIGHT_DEPTH   = 16,
    parameter integer ACT_IN_DEPTH   = 16,
    parameter integer ACT_OUT_DEPTH  = 16,
    parameter integer SCRATCH_DEPTH  = 16,
    parameter integer K_W            = 16,
    parameter integer DRAIN_CYCLES   = 2,

    // Derived address widths (SRAM controller exposes these)
    parameter integer W_ADDR_W      = (WEIGHT_DEPTH  <= 1) ? 1 : $clog2(WEIGHT_DEPTH),
    parameter integer W_ROW_DEPTH   = WEIGHT_DEPTH / N_COLS,
    parameter integer W_ROW_ADDR_W  = (W_ROW_DEPTH   <= 1) ? 1 : $clog2(W_ROW_DEPTH),
    parameter integer W_ROW_DATA_W  = N_COLS * DATA_W,
    parameter integer AI_ADDR_W     = (ACT_IN_DEPTH  <= 1) ? 1 : $clog2(ACT_IN_DEPTH),
    parameter integer AO_ADDR_W     = (ACT_OUT_DEPTH <= 1) ? 1 : $clog2(ACT_OUT_DEPTH),
    parameter integer AI_DATA_W     = N_ROWS * DATA_W,
    parameter integer AO_DATA_W     = N_COLS * ACC_W
)(
    input  wire                        clk,
    input  wire                        rst_n,

    // ── Control ─────────────────────────────────────────────────────────────
    input  wire                        start,
    input  wire [K_W-1:0]              cfg_k,
    input  wire [AI_ADDR_W-1:0]        cfg_ai_base,
    input  wire [AO_ADDR_W-1:0]        cfg_ao_base,
    input  wire [2:0]                  cfg_afu_mode,
    // Gap #2 Phase 2: per-tile accumulator init mode.  0 = clear at
    // EXEC_PREP; 1 = load accumulator from cfg_acc_init_data (for
    // software-managed k-tile chaining, partial sums of prior k-tile fed back).
    input  wire                        cfg_acc_init_mode,
    input  wire [AO_DATA_W-1:0]        cfg_acc_init_data,
    // Precision mode (latched on start to keep stable across the tile):
    //   00 INT8 (baseline)   01 INT4 (2x MACs/cycle)
    //   10 INT2 (4x MACs/cycle)   11 FP16 (placeholder, falls back to INT8)
    input  wire [1:0]                  cfg_precision_mode,
    output wire                        busy,
    output wire                        done,

    // ── External weight write (valid only when busy=0) ──────────────────────
    input  wire                        ext_w_we,
    input  wire [W_ADDR_W-1:0]         ext_w_waddr,
    input  wire [DATA_W-1:0]           ext_w_wdata,

    // ── External activation-input write (valid only when busy=0) ────────────
    input  wire                        ext_ai_we,
    input  wire [AI_ADDR_W-1:0]        ext_ai_waddr,
    input  wire [AI_DATA_W-1:0]        ext_ai_wdata,

    // ── External activation-output read (always live) ──────────────────────
    input  wire                        ext_ao_re,
    input  wire [AO_ADDR_W-1:0]        ext_ao_raddr,
    output wire [AO_DATA_W-1:0]        ext_ao_rdata,

    // ── 2:4-style sparsity skip mask (per-cycle, per-row) ───────────────────
    // Bit k = 1 means the product W[k,n]*a_vec[k] is zeroed on this cycle.
    // Pipelined internally 1 cycle so the caller drives it in sync with
    // the activation address (matches ai_rdata timing). V1 minimum plumbing;
    // 2:4 index decoder / metadata-backed decompression is future work.
    input  wire [N_ROWS-1:0]           ext_sparse_skip_vec,

    // ── DMA path: DDR → AI bank (optional; tied-off when dma_start never pulses)
    // WP-9: npu_dma fetches a tile from external memory and a narrow→wide
    // packer accumulates N_ROWS bytes into one AI-bank wide write. Valid
    // only when busy=0 (i.e. no tile execution in progress).
    input  wire                        dma_start,
    input  wire [31:0]                 dma_cfg_src_addr,
    input  wire [AI_ADDR_W-1:0]        dma_cfg_ai_base,
    input  wire [15:0]                 dma_cfg_tile_h,  // # of AI rows (each row = N_ROWS bytes)
    input  wire [15:0]                 dma_cfg_src_stride,
    output wire                        dma_busy,
    output wire                        dma_done,
    // External memory read port (1-cycle latency, always-ready)
    output wire                        mem_re,
    output wire [31:0]                 mem_raddr,
    input  wire [DATA_W-1:0]           mem_rdata,

    // ── AFU debug output (column-0 of array, element-wise activated) ────────
    output wire                        afu_out_valid,
    output wire signed [ACC_W-1:0]     afu_out_data,
    output wire                        afu_out_saturated
);

    // =========================================================================
    // Tile controller
    // =========================================================================
    wire                    tc_w_bank_sel;
    wire                    tc_w_re;
    wire [W_ROW_ADDR_W-1:0] tc_w_raddr;
    wire                    tc_array_load_valid;
    wire [W_ROW_ADDR_W-1:0] tc_array_load_cell_addr;
    wire                    tc_array_clear_acc;
    wire                    tc_array_acc_load_valid;
    wire [AO_DATA_W-1:0]    tc_array_acc_load_data;
    wire                    tc_ai_re;
    wire [AI_ADDR_W-1:0]    tc_ai_raddr;
    wire                    tc_array_exec_valid;
    wire                    tc_array_afu_in_valid;
    wire                    tc_ao_we;
    wire [AO_ADDR_W-1:0]    tc_ao_waddr;

    npu_tile_ctrl #(
        .N_ROWS       (N_ROWS),
        .N_COLS       (N_COLS),
        .ACC_W        (ACC_W),
        .W_ADDR_W     (W_ROW_ADDR_W),
        .AI_ADDR_W    (AI_ADDR_W),
        .AO_ADDR_W    (AO_ADDR_W),
        .K_W          (K_W),
        .DRAIN_CYCLES (DRAIN_CYCLES)
    ) u_tile_ctrl (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (start),
        .cfg_k       (cfg_k),
        .cfg_ai_base (cfg_ai_base),
        .cfg_ao_base (cfg_ao_base),
        .cfg_acc_init_mode (cfg_acc_init_mode),
        .cfg_acc_init_data (cfg_acc_init_data),
        .busy        (busy),
        .done        (done),
        .w_bank_sel            (tc_w_bank_sel),
        .w_re                  (tc_w_re),
        .w_raddr               (tc_w_raddr),
        .array_load_valid      (tc_array_load_valid),
        .array_load_cell_addr  (tc_array_load_cell_addr),
        .array_clear_acc       (tc_array_clear_acc),
        .array_acc_load_valid  (tc_array_acc_load_valid),
        .array_acc_load_data   (tc_array_acc_load_data),
        .ai_re                 (tc_ai_re),
        .ai_raddr              (tc_ai_raddr),
        .array_exec_valid      (tc_array_exec_valid),
        .array_afu_in_valid    (tc_array_afu_in_valid),
        .ao_we                 (tc_ao_we),
        .ao_waddr              (tc_ao_waddr)
    );

    // =========================================================================
    // Latch cfg_afu_mode on start so it stays stable for the whole tile
    // (caller can freely reprogram the port once busy rises).
    // =========================================================================
    reg [2:0] cfg_afu_mode_r;
    reg [1:0] cfg_precision_mode_r;
    always @(posedge clk) begin
        if (!rst_n) begin
            cfg_afu_mode_r       <= 3'b0;
            cfg_precision_mode_r <= 2'b0;
        end else if (start && !busy) begin
            cfg_afu_mode_r       <= cfg_afu_mode;
            cfg_precision_mode_r <= cfg_precision_mode;
        end
    end

    // =========================================================================
    // Exec-valid pipeline register — aligns array's a_valid with SRAM ai_rdata.
    // tile_ctrl drives array_exec_valid live; array needs it 1 cycle later.
    // =========================================================================
    reg exec_valid_r;
    reg [N_ROWS-1:0] sparse_skip_vec_r;
    always @(posedge clk) begin
        if (!rst_n) begin
            exec_valid_r       <= 1'b0;
            sparse_skip_vec_r  <= {N_ROWS{1'b0}};
        end else begin
            exec_valid_r       <= tc_array_exec_valid;
            sparse_skip_vec_r  <= ext_sparse_skip_vec;
        end
    end

    // =========================================================================
    // SRAM port arbitration — tile_ctrl during busy, external when idle
    // =========================================================================
    // Weight bank routing: ext writes need sel=1 (→ WA).  tile_ctrl reads
    // with sel=0 (also reads WA).  Read addr is ROW-indexed (narrow);
    // write addr is LINEAR weight index (wide).
    wire sram_w_bank_sel = busy ? tc_w_bank_sel : 1'b1;
    wire sram_w_re       = busy ? tc_w_re       : 1'b0;
    wire [W_ROW_ADDR_W-1:0] sram_w_raddr = busy ? tc_w_raddr : {W_ROW_ADDR_W{1'b0}};
    wire sram_w_we       = busy ? 1'b0           : ext_w_we;
    wire [W_ADDR_W-1:0] sram_w_waddr = busy ? {W_ADDR_W{1'b0}} : ext_w_waddr;
    wire [DATA_W-1:0]   sram_w_wdata = busy ? {DATA_W{1'b0}}   : ext_w_wdata;

    // =========================================================================
    // WP-9: DMA → AI bank via narrow-to-wide packer
    // =========================================================================
    // npu_dma outputs DATA_W bits/cycle. AI bank is N_ROWS × DATA_W bits wide.
    // The packer shift-registers N_ROWS consecutive DMA writes into one
    // AI-bank-wide word and commits the AI write on every Nth DMA write.
    // Caller drives dma_start with cfg_dma_* stable; dma_busy stays high
    // until transfer completes, then dma_done pulses for 1 cycle.
    wire                   dma_dst_we;
    wire [15:0]            dma_dst_waddr;        // linear DMA dest addr
    wire [DATA_W-1:0]      dma_dst_wdata;
    npu_dma #(
        .DATA_W     (DATA_W),
        .SRC_ADDR_W (32),
        .DST_ADDR_W (16),
        .LEN_W      (16)
    ) u_dma (
        .clk(clk), .rst_n(rst_n),
        .cfg_src_addr   (dma_cfg_src_addr),
        .cfg_dst_addr   (16'd0),                  // packer uses its own counter
        .cfg_tile_h     (dma_cfg_tile_h),
        .cfg_tile_w     ({{(16-$clog2(N_ROWS+1)){1'b0}}, N_ROWS[$clog2(N_ROWS+1)-1:0]}),
        .cfg_src_stride (dma_cfg_src_stride),
        .cfg_pad_top    (4'd0),
        .cfg_pad_bot    (4'd0),
        .cfg_pad_left   (4'd0),
        .cfg_pad_right  (4'd0),
        .start          (dma_start && !busy),
        .busy           (dma_busy),
        .done           (dma_done),
        .mem_re         (mem_re),
        .mem_raddr      (mem_raddr),
        .mem_rdata      (mem_rdata),
        .dst_we         (dma_dst_we),
        .dst_waddr      (dma_dst_waddr),
        .dst_wdata      (dma_dst_wdata)
    );

    // Narrow→wide packer: shift N_ROWS bytes into pack_buf, commit AI write
    // on byte N_ROWS-1 of each group. `pack_byte_idx` tracks within-row position.
    localparam PACK_IDX_W = (N_ROWS <= 1) ? 1 : $clog2(N_ROWS);
    reg [PACK_IDX_W-1:0] pack_byte_idx;
    reg [AI_DATA_W-1:0]  pack_buf;
    reg                  pack_ai_we;
    reg [AI_ADDR_W-1:0]  pack_ai_waddr;
    wire [AI_ADDR_W-1:0] pack_ai_base_plus_row =
        dma_cfg_ai_base + dma_dst_waddr[AI_ADDR_W-1+$clog2(N_ROWS) : $clog2(N_ROWS)];

    always @(posedge clk) begin
        if (!rst_n) begin
            pack_byte_idx <= {PACK_IDX_W{1'b0}};
            pack_buf      <= {AI_DATA_W{1'b0}};
            pack_ai_we    <= 1'b0;
            pack_ai_waddr <= {AI_ADDR_W{1'b0}};
        end else begin
            pack_ai_we <= 1'b0;
            if (dma_dst_we) begin
                // Shift this byte into the low end of the buffer
                pack_buf[pack_byte_idx*DATA_W +: DATA_W] <= dma_dst_wdata;
                if (pack_byte_idx == (N_ROWS[PACK_IDX_W-1:0] - 1'b1)) begin
                    // Full row accumulated — commit AI write NEXT cycle so
                    // the final byte has landed in pack_buf via NBA.
                    pack_byte_idx <= {PACK_IDX_W{1'b0}};
                    pack_ai_we    <= 1'b1;
                    pack_ai_waddr <= pack_ai_base_plus_row;
                end else begin
                    pack_byte_idx <= pack_byte_idx + 1'b1;
                end
            end
        end
    end

    // AI port: tile_ctrl reads during EXECUTE; writes from either DMA (while
    // DMA busy and tile not running) or external (when idle).
    wire sram_ai_re = busy ? tc_ai_re    : 1'b0;
    wire [AI_ADDR_W-1:0] sram_ai_raddr = busy ? tc_ai_raddr : {AI_ADDR_W{1'b0}};
    wire sram_ai_we =
        busy       ? 1'b0                 :
        pack_ai_we ? 1'b1                 :
                     ext_ai_we;
    wire [AI_ADDR_W-1:0] sram_ai_waddr =
        busy       ? {AI_ADDR_W{1'b0}}    :
        pack_ai_we ? pack_ai_waddr        :
                     ext_ai_waddr;
    wire [AI_DATA_W-1:0] sram_ai_wdata =
        busy       ? {AI_DATA_W{1'b0}}    :
        pack_ai_we ? pack_buf             :
                     ext_ai_wdata;

    // AO write from tile_ctrl, AO read always external.
    wire sram_ao_we    = tc_ao_we;
    wire [AO_ADDR_W-1:0] sram_ao_waddr = tc_ao_waddr;
    wire sram_ao_re    = ext_ao_re;
    wire [AO_ADDR_W-1:0] sram_ao_raddr = ext_ao_raddr;

    // =========================================================================
    // SRAM instance
    // =========================================================================
    wire [W_ROW_DATA_W-1:0] sram_w_rdata;   // wide: one full row of weights
    wire [AI_DATA_W-1:0]    sram_ai_rdata;
    wire [AO_DATA_W-1:0]    sram_ao_rdata;

    npu_sram_ctrl #(
        .DATA_W        (DATA_W),
        .ACC_W         (ACC_W),
        .N_ROWS        (N_ROWS),
        .N_COLS        (N_COLS),
        .WEIGHT_DEPTH  (WEIGHT_DEPTH),
        .ACT_IN_DEPTH  (ACT_IN_DEPTH),
        .ACT_OUT_DEPTH (ACT_OUT_DEPTH),
        .SCRATCH_DEPTH (SCRATCH_DEPTH)
    ) u_sram (
        .clk(clk), .rst_n(rst_n),
        .w_bank_sel (sram_w_bank_sel),
        .w_re   (sram_w_re),
        .w_raddr(sram_w_raddr),
        .w_rdata(sram_w_rdata),
        .w_we   (sram_w_we),
        .w_waddr(sram_w_waddr),
        .w_wdata(sram_w_wdata),

        .ai_re   (sram_ai_re),
        .ai_raddr(sram_ai_raddr),
        .ai_rdata(sram_ai_rdata),
        .ai_we   (sram_ai_we),
        .ai_waddr(sram_ai_waddr),
        .ai_wdata(sram_ai_wdata),

        .ao_re   (sram_ao_re),
        .ao_raddr(sram_ao_raddr),
        .ao_rdata(sram_ao_rdata),
        .ao_we   (sram_ao_we),
        .ao_waddr(sram_ao_waddr),
        .ao_wdata(writeback_afu_out),  // gap #3: activated c_vec via N_COLS AFUs

        // Scratch bank unused in V1.  sc_rdata explicitly left floating
        // (documented by the wire below rather than implicitly unconnected,
        // which Verilator flags as PINMISSING).
        .sc_re(1'b0), .sc_raddr({16{1'b0}}), .sc_rdata(/* unused */),
        .sc_we(1'b0), .sc_waddr({16{1'b0}}), .sc_wdata(8'h00)
    );

    assign ext_ao_rdata = sram_ao_rdata;

    // =========================================================================
    // Systolic array
    // =========================================================================
    wire                             array_c_valid;
    wire [AO_DATA_W-1:0]             array_c_vec;

    npu_systolic_array #(
        .N_ROWS(N_ROWS),
        .N_COLS(N_COLS),
        .DATA_W(DATA_W),
        .ACC_W (ACC_W)
    ) u_array (
        .clk(clk), .rst_n(rst_n),
        .w_load         (tc_array_load_valid),
        .w_addr         (tc_array_load_cell_addr),
        .w_data         (sram_w_rdata),
        .clear_acc      (tc_array_clear_acc),
        .acc_load_valid (tc_array_acc_load_valid),
        .acc_load_data  (tc_array_acc_load_data),
        .a_valid        (exec_valid_r),
        .a_vec          (sram_ai_rdata),
        .sparse_skip_vec(sparse_skip_vec_r),
        .precision_mode (cfg_precision_mode_r),
        .c_valid        (array_c_valid),
        .c_vec          (array_c_vec)
    );

    // =========================================================================
    // Gap #3: writeback AFUs — N_COLS parallel npu_activation instances on
    // the AO writeback path. tile_ctrl pulses `array_afu_in_valid` once
    // per tile (last DRAIN cycle), when the array accumulator is stable.
    // The AFUs register their activated output; by S_STORE (1 cycle later)
    // `writeback_afu_out` holds the final activated c_vec and the SRAM AO
    // captures it. PASS mode (cfg_afu_mode_r = 0) is a no-op, giving
    // backward-compatible raw-c_vec writeback semantics.
    // =========================================================================
    wire [AO_DATA_W-1:0] writeback_afu_out;
    wire [N_COLS-1:0]    writeback_afu_sat_unused;
    wire [N_COLS-1:0]    writeback_afu_valid_unused;

    genvar wb;
    generate
        for (wb = 0; wb < N_COLS; wb = wb + 1) begin : g_writeback_afu
            npu_activation #(
                .ACC_W(ACC_W),
                .OUT_W(ACC_W)
            ) u_wb_afu (
                .clk          (clk),
                .rst_n        (rst_n),
                .mode         (cfg_afu_mode_r),
                .in_valid     (tc_array_afu_in_valid),
                .in_data      (array_c_vec[wb*ACC_W +: ACC_W]),
                .out_valid    (writeback_afu_valid_unused[wb]),
                .out_data     (writeback_afu_out[wb*ACC_W +: ACC_W]),
                .out_saturated(writeback_afu_sat_unused[wb])
            );
        end
    endgenerate

    // =========================================================================
    // Column-0 debug AFU (live tap on array output during EXECUTE).
    // Kept as-is for observability; uses the LIVE cfg_afu_mode port so the
    // external test-bench can retune mid-simulation without waiting for a
    // tile boundary.
    // =========================================================================
    npu_activation #(
        .ACC_W(ACC_W),
        .OUT_W(ACC_W)
    ) u_afu (
        .clk(clk), .rst_n(rst_n),
        .mode(cfg_afu_mode),
        .in_valid(array_c_valid),
        .in_data(array_c_vec[0 +: ACC_W]),
        .out_valid(afu_out_valid),
        .out_data(afu_out_data),
        .out_saturated(afu_out_saturated)
    );

endmodule
