`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — NPU Systolic Array  (npu_systolic_array.v)
// =============================================================================
// Parametric N_ROWS x N_COLS weight-stationary matrix-vector multiply (MVM)
// engine.  One cycle consumes an N_ROWS-long activation vector and produces
// an N_COLS-long partial-sum vector that is accumulated into per-column
// registers.  The accumulator is cleared by clear_acc; weights are loaded
// serially via the (w_load, w_addr, w_data) interface at bring-up time.
//
// ── Dataflow (V1 — parallel weight-stationary) ────────────────────────────────
//
//   weights: W[k, n] held in a flop array, loaded once per tile via w_load.
//   input  : a_vec[k] broadcast across all columns at cycle t.
//   output : c_vec[n] = sum_k a_vec[k] * W[k, n] accumulated over cycles.
//
//   One cycle of latency from (a_valid, a_vec) → (c_valid, c_vec): the
//   products and reduction tree are combinational, but the accumulator
//   update and output vector are registered on the clock edge.
//
//   c_valid pulses when a new partial sum has just been accumulated.
//   Drain the final output by either sampling c_vec after the last tile
//   cycle or by issuing clear_acc to reset for the next tile.
//
// ── Comparison to "true" pipelined systolic ──────────────────────────────────
//
//   A classic TPU-style systolic array flows activations rightward and
//   partial sums downward through a grid of small PEs, avoiding long
//   reduction wires at the cost of K + N - 2 cycles of fill latency.
//   V1 is a combinational reduction tree — functionally identical, simpler
//   to verify, but has O(log N) wire delay that will limit Fmax at large
//   sizes.  V2 will keep this external interface and replace the internal
//   tree with the pipelined dataflow.  Any bit-exact test that passes on
//   V1 must continue to pass on V2 (with added latency tolerance).
//
// ── Weight load protocol ─────────────────────────────────────────────────────
//
//   w_load pulses for one cycle with w_addr[*] in row-major order:
//     w_addr = k * N_COLS + n  selects W[k, n]
//   w_data is latched into the selected cell on that cycle.  Loading a
//   full N_ROWS * N_COLS tile takes that many cycles.  A "broadcast" or
//   DMA-stream load path will be added later; the serial interface here
//   is sufficient for bring-up and test.
//
// ── Precision ─────────────────────────────────────────────────────────────────
//   V1 implements INT8 × INT8 with INT32 accumulator.  Precision mode is
//   not plumbed yet; the (weight_reg × a_in) multiply will be replaced by
//   a precision-aware datapath when INT4/INT2/FP16 modes land.
//
// ── Parameters ───────────────────────────────────────────────────────────────
//   N_ROWS — reduction dimension (length of activation vector)
//   N_COLS — number of output channels produced per cycle
//   DATA_W — width of one weight or activation element (default 8 for INT8)
//   ACC_W  — width of one accumulator (default 32; ≥ 2*DATA_W + log2(depth))
// =============================================================================

module npu_systolic_array #(
    parameter integer N_ROWS = 4,
    parameter integer N_COLS = 4,
    parameter integer DATA_W = 8,
    parameter integer ACC_W  = 32,

    // Derived — not user-settable (used for port widths)
    parameter integer W_ROW_ADDR_W = (N_ROWS <= 1) ? 1 : $clog2(N_ROWS),
    parameter integer W_ROW_DATA_W = N_COLS * DATA_W
)(
    input  wire                                clk,
    input  wire                                rst_n,

    // ── Wide weight load (one full row per cycle) ───────────────────────────
    // w_load pulses for one cycle with w_addr selecting the row to latch
    // (range 0..N_ROWS-1).  w_data is the packed row: column n occupies
    // bits [n*DATA_W +: DATA_W].  PRELOAD walks N_ROWS cycles total.
    input  wire                                w_load,
    input  wire [W_ROW_ADDR_W-1:0]             w_addr,
    input  wire [W_ROW_DATA_W-1:0]             w_data,

    // ── Accumulator control ──────────────────────────────────────────────────
    // clear_acc     : zero all accumulators this cycle (wins over acc_load).
    // acc_load_valid: load accumulators from acc_load_data this cycle.
    //                 Used for the gap-#2 "scratch-based k-tile accumulation"
    //                 pattern: on a new k-tile boundary, tile_ctrl loads the
    //                 previous k-tile's partial sum from SRAM scratch so this
    //                 k-tile can resume accumulation without restarting.
    // Priority: clear_acc > acc_load_valid > (a_valid ? accumulate : hold).
    input  wire                                clear_acc,
    input  wire                                acc_load_valid,
    input  wire [N_COLS*ACC_W-1:0]             acc_load_data,

    // ── Activation input (packed vector, per cycle) ──────────────────────────
    input  wire                                a_valid,
    input  wire [N_ROWS*DATA_W-1:0]            a_vec,

    // ── 2:4-style sparsity skip mask (per-cycle, per-row) ───────────────────
    // bit k = 1 means the product W[k,n] * a_vec[k] is forced to zero for
    // every column on this cycle (the k-th activation slot corresponds to a
    // pruned weight). This is the skip-gate portion of 2:4 sparsity; the
    // 2:4 index decoder that generates this mask from stored weight
    // metadata is a separate future deliverable.
    input  wire [N_ROWS-1:0]                   sparse_skip_vec,

    // ── Precision mode (shared by every product in the array this cycle) ────
    //   2'b00 INT8  (1 MAC per lane — baseline)
    //   2'b01 INT4  (2 MACs per lane — each DATA_W slot packs 2 INT4 ops)
    //   2'b10 INT2  (4 MACs per lane — each DATA_W slot packs 4 INT2 ops)
    //   2'b11 FP16  (placeholder — falls back to INT8 path for now)
    // Must be stable for the duration of a tile; npu_top latches it on start.
    input  wire [1:0]                          precision_mode,

    // ── Output (packed vector, registered) ───────────────────────────────────
    output reg                                 c_valid,
    output reg  [N_COLS*ACC_W-1:0]             c_vec
);

    // =========================================================================
    // 1. Weight storage — 2-D array indexed by [row][col].
    //    Synthesised as a flop array; larger tiles will move this to SRAM.
    //    Wide load: one w_load pulse latches an entire ROW (N_COLS weights).
    // =========================================================================
    reg signed [DATA_W-1:0] W [0:N_ROWS-1][0:N_COLS-1];

    integer wr, wc;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (wr = 0; wr < N_ROWS; wr = wr + 1)
                for (wc = 0; wc < N_COLS; wc = wc + 1)
                    W[wr][wc] <= {DATA_W{1'b0}};
        end else if (w_load) begin
            for (wc = 0; wc < N_COLS; wc = wc + 1)
                W[w_addr][wc] <= w_data[wc*DATA_W +: DATA_W];
        end
    end

    // =========================================================================
    // 2. Per-column combinational reduction tree
    //    For each output column n, compute sum over k of W[k, n] * a_vec[k].
    //    A genvar loop builds N_COLS parallel trees.  Each product fits in
    //    2*DATA_W bits and is sign-extended to ACC_W before the reduction.
    // =========================================================================
    // Unpack a_vec into a signed-per-row view for readability.
    wire signed [DATA_W-1:0] a_row [0:N_ROWS-1];
    genvar gk;
    generate
        for (gk = 0; gk < N_ROWS; gk = gk + 1) begin : A_UNPACK
            assign a_row[gk] = a_vec[gk*DATA_W +: DATA_W];
        end
    endgenerate

    // Per-column dot product (combinational).
    wire signed [ACC_W-1:0] dot [0:N_COLS-1];

    genvar gn, gk2;
    generate
        for (gn = 0; gn < N_COLS; gn = gn + 1) begin : COL
            // Products for this column: for each row k, select the INT8 /
            // INT4-packed-sum / INT2-packed-sum version based on
            // precision_mode, then sign-extend to ACC_W. Shared structure
            // across all modes so the reduction tree stays one code path.
            wire signed [ACC_W-1:0] prod [0:N_ROWS-1];
            for (gk2 = 0; gk2 < N_ROWS; gk2 = gk2 + 1) begin : PROD
                // INT8 product (always computed; primary output at mode=00)
                wire signed [2*DATA_W-1:0] p_int8 =
                    W[gk2][gn] * a_row[gk2];

                // INT4 packed: 2 INT4×INT4 products, summed (9-bit result)
                wire signed [3:0] w_hi_i4 = W[gk2][gn][7:4];
                wire signed [3:0] w_lo_i4 = W[gk2][gn][3:0];
                wire signed [3:0] a_hi_i4 = a_row[gk2][7:4];
                wire signed [3:0] a_lo_i4 = a_row[gk2][3:0];
                wire signed [7:0] p_int4_hi = w_hi_i4 * a_hi_i4;
                wire signed [7:0] p_int4_lo = w_lo_i4 * a_lo_i4;
                wire signed [8:0] p_int4    = p_int4_hi + p_int4_lo;

                // INT2 packed: 4 INT2×INT2 products, summed (6-bit result)
                wire signed [1:0] w_i2_3 = W[gk2][gn][7:6];
                wire signed [1:0] w_i2_2 = W[gk2][gn][5:4];
                wire signed [1:0] w_i2_1 = W[gk2][gn][3:2];
                wire signed [1:0] w_i2_0 = W[gk2][gn][1:0];
                wire signed [1:0] a_i2_3 = a_row[gk2][7:6];
                wire signed [1:0] a_i2_2 = a_row[gk2][5:4];
                wire signed [1:0] a_i2_1 = a_row[gk2][3:2];
                wire signed [1:0] a_i2_0 = a_row[gk2][1:0];
                wire signed [3:0] p_i2_3 = w_i2_3 * a_i2_3;
                wire signed [3:0] p_i2_2 = w_i2_2 * a_i2_2;
                wire signed [3:0] p_i2_1 = w_i2_1 * a_i2_1;
                wire signed [3:0] p_i2_0 = w_i2_0 * a_i2_0;
                wire signed [5:0] p_int2 = p_i2_3 + p_i2_2 + p_i2_1 + p_i2_0;

                // Mode select + sign-extend to ACC_W; zero out when the
                // corresponding row is marked pruned via sparse_skip_vec[k].
                wire signed [ACC_W-1:0] prod_sel =
                    (precision_mode == 2'b01) ?
                        {{(ACC_W - 9){p_int4[8]}}, p_int4} :
                    (precision_mode == 2'b10) ?
                        {{(ACC_W - 6){p_int2[5]}}, p_int2} :
                        {{(ACC_W - 2*DATA_W){p_int8[2*DATA_W-1]}}, p_int8};
                assign prod[gk2] = sparse_skip_vec[gk2] ? {ACC_W{1'b0}} : prod_sel;
            end

            // Linear reduction (synth tools infer balanced tree; fine for V1).
            // Declaring as a function avoids declaring a per-column loop var.
            reg signed [ACC_W-1:0] sum_r;
            integer ki;
            always @* begin
                sum_r = {ACC_W{1'b0}};
                for (ki = 0; ki < N_ROWS; ki = ki + 1)
                    sum_r = sum_r + prod[ki];
            end
            assign dot[gn] = sum_r;
        end
    endgenerate

    // =========================================================================
    // 3. Per-column accumulator + output registration
    // =========================================================================
    reg signed [ACC_W-1:0] acc [0:N_COLS-1];

    integer ai;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (ai = 0; ai < N_COLS; ai = ai + 1)
                acc[ai] <= {ACC_W{1'b0}};
            c_valid <= 1'b0;
            c_vec   <= {(N_COLS*ACC_W){1'b0}};
        end else begin
            // Priority: clear_acc > acc_load_valid > (a_valid ? accumulate : hold)
            if (clear_acc) begin
                for (ai = 0; ai < N_COLS; ai = ai + 1)
                    acc[ai] <= {ACC_W{1'b0}};
            end else if (acc_load_valid) begin
                // Load partial sums from SRAM scratch (unpack wide input).
                for (ai = 0; ai < N_COLS; ai = ai + 1)
                    acc[ai] <= acc_load_data[ai*ACC_W +: ACC_W];
            end else if (a_valid) begin
                for (ai = 0; ai < N_COLS; ai = ai + 1)
                    acc[ai] <= acc[ai] + dot[ai];
            end

            // Output vector: next cycle's view of the accumulator.
            // c_valid pulses on cycles where the accumulator was updated
            // by an accumulate operation (not clear/load-only).
            c_valid <= a_valid && !clear_acc && !acc_load_valid;
            for (ai = 0; ai < N_COLS; ai = ai + 1) begin
                if (clear_acc)
                    c_vec[ai*ACC_W +: ACC_W] <= {ACC_W{1'b0}};
                else if (acc_load_valid)
                    c_vec[ai*ACC_W +: ACC_W] <= acc_load_data[ai*ACC_W +: ACC_W];
                else if (a_valid)
                    c_vec[ai*ACC_W +: ACC_W] <= acc[ai] + dot[ai];
                // else: hold
            end
        end
    end

endmodule
