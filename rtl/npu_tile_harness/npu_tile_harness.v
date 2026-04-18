`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — NPU Tile Integration Harness  (npu_tile_harness.v)
// =============================================================================
// Minimal wrapper instantiating `npu_sram_ctrl` + `npu_systolic_array` to
// prove end-to-end tile execution bit-exact against the Python golden model.
//
// Data path:
//
//   DMA-side ports (driven by testbench)
//     ├─ w_we/w_waddr/w_wdata   → SRAM weight bank (WA or WB via sel)
//     └─ ai_we/ai_waddr/ai_wdata → SRAM activation-in bank
//
//   Array-driver ports (driven by testbench to sequence a tile)
//     ├─ array_load_valid + array_load_addr   — serial weight preload
//     └─ array_exec_valid  + array_exec_addr  — per-cycle activation stream
//
//   Output ports (readable by testbench)
//     ├─ array_c_valid / array_c_vec          — raw array output
//     └─ ao_*                                  — SRAM AO bank (writeback)
//
// The harness does NOT include a tile controller; it exposes the primitive
// sequences so the testbench can drive them one cycle at a time and verify
// the full datapath matches a Python reference tick-for-tick.
// =============================================================================

module npu_tile_harness #(
    parameter integer DATA_W        = 8,
    parameter integer ACC_W         = 32,
    parameter integer N_ROWS        = 4,
    parameter integer N_COLS        = 4,
    parameter integer WEIGHT_DEPTH  = 16,
    parameter integer ACT_IN_DEPTH  = 16,
    parameter integer ACT_OUT_DEPTH = 16,

    parameter integer W_ADDR_W  = (WEIGHT_DEPTH  <= 1) ? 1 : $clog2(WEIGHT_DEPTH),
    parameter integer W_ROW_DEPTH  = WEIGHT_DEPTH / N_COLS,
    parameter integer W_ROW_ADDR_W = (W_ROW_DEPTH <= 1) ? 1 : $clog2(W_ROW_DEPTH),
    parameter integer W_ROW_DATA_W = N_COLS * DATA_W,
    parameter integer AI_ADDR_W = (ACT_IN_DEPTH  <= 1) ? 1 : $clog2(ACT_IN_DEPTH),
    parameter integer AO_ADDR_W = (ACT_OUT_DEPTH <= 1) ? 1 : $clog2(ACT_OUT_DEPTH),
    parameter integer AI_DATA_W = N_ROWS * DATA_W,
    parameter integer AO_DATA_W = N_COLS * ACC_W,
    // Array weight addr is row-indexed after the wide-SRAM change
    parameter integer WA_ADDR_W = (N_ROWS <= 1) ? 1 : $clog2(N_ROWS)
)(
    input  wire                        clk,
    input  wire                        rst_n,

    // ── DMA-side SRAM writes ─────────────────────────────────────────────────
    input  wire                        w_bank_sel,
    input  wire                        w_we,
    input  wire [W_ADDR_W-1:0]         w_waddr,
    input  wire [DATA_W-1:0]           w_wdata,
    input  wire                        ai_we,
    input  wire [AI_ADDR_W-1:0]        ai_waddr,
    input  wire [AI_DATA_W-1:0]        ai_wdata,

    // ── Tile sequencing (testbench drives) ───────────────────────────────────
    // Weight preload phase: walk ROW addresses 0..N_ROWS-1 and copy from
    // SRAM weight bank (wide read) into the systolic array's internal W grid
    // (wide load, one full row per cycle).
    input  wire                        array_load_valid,
    input  wire [W_ROW_ADDR_W-1:0]     array_load_sram_addr,
    input  wire [WA_ADDR_W-1:0]        array_load_cell_addr,
    input  wire                        array_clear_acc,

    // Execute phase: pull one activation vector per cycle from SRAM AI
    // and feed it to the array.
    input  wire                        array_exec_valid,
    input  wire [AI_ADDR_W-1:0]        array_exec_ai_addr,

    // Writeback phase: when array_c_valid fires, capture array_c_vec
    // into SRAM AO at the next address.  Writeback addresses are driven
    // by the testbench explicitly to keep the harness strictly primitive.
    input  wire                        ao_wb_enable,
    input  wire [AO_ADDR_W-1:0]        ao_wb_addr,

    // Testbench-facing read ports for inspection
    input  wire                        ao_re,
    input  wire [AO_ADDR_W-1:0]        ao_raddr,
    output wire [AO_DATA_W-1:0]        ao_rdata,

    // Exposure of the array output (combinational, tap of the array port)
    output wire                        array_c_valid,
    output wire [N_COLS*ACC_W-1:0]     array_c_vec
);

    // =========================================================================
    // SRAM controller
    // =========================================================================
    wire [W_ROW_DATA_W-1:0] sram_w_rdata;
    wire [AI_DATA_W-1:0]    sram_ai_rdata;

    npu_sram_ctrl #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .N_ROWS(N_ROWS),
        .N_COLS(N_COLS),
        .WEIGHT_DEPTH(WEIGHT_DEPTH),
        .ACT_IN_DEPTH(ACT_IN_DEPTH),
        .ACT_OUT_DEPTH(ACT_OUT_DEPTH),
        .SCRATCH_DEPTH(16)
    ) u_sram (
        .clk(clk), .rst_n(rst_n),

        .w_bank_sel(w_bank_sel),
        .w_re   (array_load_valid),
        .w_raddr(array_load_sram_addr),
        .w_rdata(sram_w_rdata),
        .w_we   (w_we),
        .w_waddr(w_waddr),
        .w_wdata(w_wdata),

        .ai_re   (array_exec_valid),
        .ai_raddr(array_exec_ai_addr),
        .ai_rdata(sram_ai_rdata),
        .ai_we   (ai_we),
        .ai_waddr(ai_waddr),
        .ai_wdata(ai_wdata),

        .ao_re   (ao_re),
        .ao_raddr(ao_raddr),
        .ao_rdata(ao_rdata),
        .ao_we   (ao_wb_enable),
        .ao_waddr(ao_wb_addr),
        .ao_wdata(array_c_vec),

        .sc_re(1'b0), .sc_raddr({16{1'b0}}), .sc_rdata(/* unused */),
        .sc_we(1'b0), .sc_waddr({16{1'b0}}), .sc_wdata(8'h00)
    );

    // =========================================================================
    // Delay-align the 1-cycle SRAM read latency with the array inputs.
    //
    //   At cycle N the testbench asserts array_load_valid + load_sram_addr.
    //   At cycle N+1 sram_w_rdata is valid.  The array expects w_load + w_addr
    //   + w_data on the SAME cycle, so we register array_load_cell_addr by one
    //   cycle to align with the SRAM rdata.
    //
    //   The same trick delays array_exec_valid by one cycle for the activation
    //   stream.  The SRAM ai_rdata at cycle N+1 corresponds to the activation
    //   the array should consume on cycle N+1 via a_valid.
    // =========================================================================
    reg                        w_load_r;
    reg  [WA_ADDR_W-1:0]       w_cell_addr_r;
    reg                        exec_valid_r;
    reg                        ao_wb_enable_r;

    always @(posedge clk) begin
        if (!rst_n) begin
            w_load_r       <= 1'b0;
            w_cell_addr_r  <= {WA_ADDR_W{1'b0}};
            exec_valid_r   <= 1'b0;
            ao_wb_enable_r <= 1'b0;
        end else begin
            w_load_r       <= array_load_valid;
            w_cell_addr_r  <= array_load_cell_addr;
            exec_valid_r   <= array_exec_valid;
            ao_wb_enable_r <= ao_wb_enable;  // unused externally; placeholder
        end
    end

    // =========================================================================
    // Systolic array
    // =========================================================================
    npu_systolic_array #(
        .N_ROWS(N_ROWS),
        .N_COLS(N_COLS),
        .DATA_W(DATA_W),
        .ACC_W(ACC_W)
    ) u_array (
        .clk(clk), .rst_n(rst_n),
        .w_load         (w_load_r),
        .w_addr         (w_cell_addr_r),
        .w_data         (sram_w_rdata),
        .clear_acc      (array_clear_acc),
        .acc_load_valid (1'b0),                  // gap #2 phase 2 wires this
        .acc_load_data  ({(N_COLS*ACC_W){1'b0}}),
        .a_valid        (exec_valid_r),
        .a_vec          (sram_ai_rdata),
        .sparse_skip_vec({N_ROWS{1'b0}}),        // not exercised in harness
        .precision_mode (2'b00),                 // INT8 default
        .c_valid        (array_c_valid),
        .c_vec          (array_c_vec)
    );

endmodule
