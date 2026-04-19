`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — NPU SRAM Controller  (npu_sram_ctrl.v)
// =============================================================================
// On-chip working-memory subsystem for the NPU.  Holds weight tiles,
// activation tiles, output partial sums, and scratch.  Sits between the
// DMA engine (external DDR → on-chip) and the systolic array.
//
// ── Bank map (V1) ────────────────────────────────────────────────────────────
//
//   Name  Width                 Depth              Purpose
//   ----  --------------------  -----------------  ---------------------------
//   WA    DATA_W                WEIGHT_DEPTH       Weight tile A (double-buf)
//   WB    DATA_W                WEIGHT_DEPTH       Weight tile B (double-buf)
//   AI    N_ROWS * DATA_W       ACT_IN_DEPTH       Activation input vectors
//   AO    N_COLS * ACC_W        ACT_OUT_DEPTH      Output partial-sum vectors
//   SC    DATA_W                SCRATCH_DEPTH      Scratch / misc
//
// ── Double-buffer semantics (weights only in V1) ─────────────────────────────
//   w_bank_sel = 0:  array reads WA, DMA writes WB
//   w_bank_sel = 1:  array reads WB, DMA writes WA
//
//   The external orchestrator (tile controller / CPU) toggles w_bank_sel
//   between tiles to swap roles, giving zero-stall compute↔load overlap.
//
//   Only ONE pair of weight ports is exposed to the outside: a read port
//   (array side) and a write port (DMA side).  Which physical bank each
//   port reaches is determined entirely by w_bank_sel inside this module.
//
// ── Port semantics ───────────────────────────────────────────────────────────
//   All ports follow the same protocol as npu_sram_bank.v:
//     read : (re, raddr) on cycle N → rdata on cycle N+1
//     write: (we, waddr, wdata) on cycle N latches at end of cycle N
//     rdata holds when re=0
//     same-address same-cycle R+W: write wins
//
//   AI and AO expose wide ports matching the array's vector widths, so
//   one bank access delivers a full activation or output vector per cycle.
//
// ── What this module DOES NOT do (deferred to later revs) ────────────────────
//   - No ECC / parity on the data (add in hardening phase for ASIL-D).
//   - No byte-enable writes.
//   - No arbitration between multiple requestors on the same port — the
//     caller guarantees DMA and array don't simultaneously write the same
//     bank.  Arbitration belongs in a wrapper above this module.
//   - No double-buffer on AI / AO / SC.  Trivial extension, left for V2
//     once we know real usage patterns from the tile controller.
// =============================================================================

module npu_sram_ctrl #(
    // Defaults match the small-instance regression harness (4x4 grid,
    // depth-16 banks).  Production integrators (npu_top, npu_tile_harness)
    // always pass explicit parameters, so these defaults exist purely so
    // the standalone cocotb tests can elaborate without a runner-side
    // parameter override.  Earlier this module shipped with 16x16/depth-256
    // defaults, which caused the tests to read X from never-written
    // sub-banks (only cols 0..N_cols-1 were being prefilled while the
    // RTL instantiated N_COLS=16 sub-banks).
    parameter integer DATA_W         = 8,
    parameter integer ACC_W          = 32,
    parameter integer N_ROWS         = 4,
    parameter integer N_COLS         = 4,
    // Auto-derives from array size. Matches npu_top's derivation.
    // Overriding this inconsistently used to silently break at non-
    // default array sizes (GAP-3 finding, fixed 2026-04-19).
    parameter integer WEIGHT_DEPTH   = N_ROWS * N_COLS,
    parameter integer ACT_IN_DEPTH   = 16,
    parameter integer ACT_OUT_DEPTH  = 16,
    parameter integer SCRATCH_DEPTH  = 16,

    // Derived (not user-settable — used for port widths)
    // W_ADDR_W: linear weight-index width (for per-weight WRITES).
    //   Accepts 0..WEIGHT_DEPTH-1; internally split into (row, col).
    // W_ROW_ADDR_W: row-index width (for ROW-wide READS).
    parameter integer W_ADDR_W       = (WEIGHT_DEPTH  <= 1) ? 1 : $clog2(WEIGHT_DEPTH),
    parameter integer W_ROW_DEPTH    = WEIGHT_DEPTH / N_COLS,
    parameter integer W_ROW_ADDR_W   = (W_ROW_DEPTH   <= 1) ? 1 : $clog2(W_ROW_DEPTH),
    parameter integer W_COL_IDX_W    = (N_COLS        <= 1) ? 1 : $clog2(N_COLS),
    parameter integer W_ROW_DATA_W   = N_COLS * DATA_W,
    parameter integer AI_ADDR_W      = (ACT_IN_DEPTH  <= 1) ? 1 : $clog2(ACT_IN_DEPTH),
    parameter integer AO_ADDR_W      = (ACT_OUT_DEPTH <= 1) ? 1 : $clog2(ACT_OUT_DEPTH),
    parameter integer SC_ADDR_W      = (SCRATCH_DEPTH <= 1) ? 1 : $clog2(SCRATCH_DEPTH),
    parameter integer AI_DATA_W      = N_ROWS * DATA_W,
    parameter integer AO_DATA_W      = N_COLS * ACC_W
)(
    input  wire                        clk,
    input  wire                        rst_n,

    // ── Weight port (muxed WA/WB via w_bank_sel) ─────────────────────────────
    //   w_bank_sel=0 → read from WA, write to WB
    //   w_bank_sel=1 → read from WB, write to WA
    //
    // READ is ROW-wide: one w_re pulse returns a full row (N_COLS weights).
    // WRITE is per-weight narrow: waddr is the LINEAR weight index and
    // decomposes internally into (row, col) across N_COLS parallel sub-banks.
    input  wire                        w_bank_sel,

    // Read (array side) — wide, row-addressed
    input  wire                        w_re,
    input  wire [W_ROW_ADDR_W-1:0]     w_raddr,
    output wire [W_ROW_DATA_W-1:0]     w_rdata,

    // Write (DMA side) — narrow, linear-addressed; waddr = row*N_COLS + col
    input  wire                        w_we,
    input  wire [W_ADDR_W-1:0]         w_waddr,
    input  wire [DATA_W-1:0]           w_wdata,

    // ── Activation Input bank ───────────────────────────────────────────────
    input  wire                        ai_re,
    input  wire [AI_ADDR_W-1:0]        ai_raddr,
    output wire [AI_DATA_W-1:0]        ai_rdata,
    input  wire                        ai_we,
    input  wire [AI_ADDR_W-1:0]        ai_waddr,
    input  wire [AI_DATA_W-1:0]        ai_wdata,

    // ── Activation Output bank ──────────────────────────────────────────────
    input  wire                        ao_re,
    input  wire [AO_ADDR_W-1:0]        ao_raddr,
    output wire [AO_DATA_W-1:0]        ao_rdata,
    input  wire                        ao_we,
    input  wire [AO_ADDR_W-1:0]        ao_waddr,
    input  wire [AO_DATA_W-1:0]        ao_wdata,

    // ── Scratch bank ────────────────────────────────────────────────────────
    input  wire                        sc_re,
    input  wire [SC_ADDR_W-1:0]        sc_raddr,
    output wire [DATA_W-1:0]           sc_rdata,
    input  wire                        sc_we,
    input  wire [SC_ADDR_W-1:0]        sc_waddr,
    input  wire [DATA_W-1:0]           sc_wdata
);

    // =========================================================================
    // Weight double-buffer routing
    //
    //   WA / WB are each implemented as N_COLS parallel narrow sub-banks,
    //   one per weight column.  Sub-bank depth = WEIGHT_DEPTH / N_COLS.
    //   A ROW-wide read addresses all sub-banks with the same row_addr and
    //   concatenates their rdata.  A narrow WRITE decomposes the linear
    //   waddr into (row, col) and routes to only the selected sub-bank.
    // =========================================================================
    // External write decomposition: low bits = col index, high bits = row.
    wire [W_COL_IDX_W-1:0]  ext_col_idx = w_waddr[W_COL_IDX_W-1:0];
    wire [W_ROW_ADDR_W-1:0] ext_row_addr = w_waddr[W_ADDR_W-1:W_COL_IDX_W];

    // Registered w_bank_sel for aligning read-data mux with SRAM latency.
    reg w_bank_sel_r;
    always @(posedge clk) begin
        if (!rst_n) w_bank_sel_r <= 1'b0;
        else        w_bank_sel_r <= w_bank_sel;
    end

    // Per-sub-bank wires (declared at top so the generate block can drive them)
    wire [DATA_W-1:0] wa_sub_rdata [0:N_COLS-1];
    wire [DATA_W-1:0] wb_sub_rdata [0:N_COLS-1];

    // Assemble wide row rdata from per-sub-bank outputs
    wire [W_ROW_DATA_W-1:0] wa_rdata_wide;
    wire [W_ROW_DATA_W-1:0] wb_rdata_wide;
    genvar gci;
    generate
        for (gci = 0; gci < N_COLS; gci = gci + 1) begin : WCOLS
            assign wa_rdata_wide[gci*DATA_W +: DATA_W] = wa_sub_rdata[gci];
            assign wb_rdata_wide[gci*DATA_W +: DATA_W] = wb_sub_rdata[gci];
        end
    endgenerate

    assign w_rdata = (w_bank_sel_r == 1'b0) ? wa_rdata_wide : wb_rdata_wide;

    // Instantiate N_COLS parallel sub-banks per weight bank (WA + WB)
    genvar gcj;
    generate
        for (gcj = 0; gcj < N_COLS; gcj = gcj + 1) begin : WA_SUB
            // WA sub-bank for column gcj.
            // Write only when (w_we && sel==1 && ext_col_idx == gcj).
            // Read always on same row addr when sel==0 && w_re.
            wire wa_sub_we = w_we && (w_bank_sel == 1'b1)
                             && (ext_col_idx == gcj[W_COL_IDX_W-1:0]);
            wire wa_sub_re = w_re && (w_bank_sel == 1'b0);
            npu_sram_bank #(
                .DATA_W(DATA_W),
                .DEPTH(W_ROW_DEPTH)
            ) u_wa_col (
                .clk(clk), .rst_n(rst_n),
                .we(wa_sub_we),
                .waddr(ext_row_addr),
                .wdata(w_wdata),
                .re(wa_sub_re),
                .raddr(w_raddr),
                .rdata(wa_sub_rdata[gcj])
            );
        end
        for (gcj = 0; gcj < N_COLS; gcj = gcj + 1) begin : WB_SUB
            wire wb_sub_we = w_we && (w_bank_sel == 1'b0)
                             && (ext_col_idx == gcj[W_COL_IDX_W-1:0]);
            wire wb_sub_re = w_re && (w_bank_sel == 1'b1);
            npu_sram_bank #(
                .DATA_W(DATA_W),
                .DEPTH(W_ROW_DEPTH)
            ) u_wb_col (
                .clk(clk), .rst_n(rst_n),
                .we(wb_sub_we),
                .waddr(ext_row_addr),
                .wdata(w_wdata),
                .re(wb_sub_re),
                .raddr(w_raddr),
                .rdata(wb_sub_rdata[gcj])
            );
        end
    endgenerate

    npu_sram_bank #(
        .DATA_W(AI_DATA_W),
        .DEPTH(ACT_IN_DEPTH)
    ) u_ai (
        .clk(clk), .rst_n(rst_n),
        .we(ai_we), .waddr(ai_waddr), .wdata(ai_wdata),
        .re(ai_re), .raddr(ai_raddr), .rdata(ai_rdata)
    );

    npu_sram_bank #(
        .DATA_W(AO_DATA_W),
        .DEPTH(ACT_OUT_DEPTH)
    ) u_ao (
        .clk(clk), .rst_n(rst_n),
        .we(ao_we), .waddr(ao_waddr), .wdata(ao_wdata),
        .re(ao_re), .raddr(ao_raddr), .rdata(ao_rdata)
    );

    npu_sram_bank #(
        .DATA_W(DATA_W),
        .DEPTH(SCRATCH_DEPTH)
    ) u_sc (
        .clk(clk), .rst_n(rst_n),
        .we(sc_we), .waddr(sc_waddr), .wdata(sc_wdata),
        .re(sc_re), .raddr(sc_raddr), .rdata(sc_rdata)
    );

endmodule
