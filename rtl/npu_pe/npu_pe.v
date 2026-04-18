`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — NPU Processing Element (npu_pe.v)
// =============================================================================
// Single processing element of the NPU systolic array.  Weight-stationary
// dataflow: the weight is loaded once per tile and held while activations
// stream through.  Each cycle the PE forms one multiply-accumulate.
//
// ── Dataflow ─────────────────────────────────────────────────────────────────
//   Activations flow left-to-right across the array.  Each PE consumes the
//   incoming activation, multiplies by its held weight, adds the product to
//   its local accumulator, and forwards the activation (delayed by one clock)
//   to the neighbour PE to its right.
//
//   psum_out is the running accumulator.  The array drains psum_out on a row
//   boundary; clear_acc resets the accumulator between tiles.
//
// ── Precision modes ──────────────────────────────────────────────────────────
//   precision_mode[1:0]:
//     2'b00 — INT8 × INT8 → INT32 accumulator (1 MAC/cycle)      [IMPLEMENTED]
//     2'b01 — INT4 × INT4 packed (2 MACs/cycle)                  [IMPLEMENTED]
//     2'b10 — INT2 × INT2 packed (4 MACs/cycle)                  [IMPLEMENTED]
//     2'b11 — FP16 × FP16 → FP32 accumulator                     [PLACEHOLDER]
//
//   Packed-precision modes reuse the same DATA_W=8 `a_in` / `weight_reg`
//   ports, decomposing each 8-bit word into 2 INT4 or 4 INT2 operands:
//     INT4: weight = {w_hi, w_lo} with w_hi/w_lo signed 4-bit.
//           activation = {a_hi, a_lo}. Two products summed per cycle.
//     INT2: weight = {w3, w2, w1, w0}, activation = {a3, a2, a1, a0}.
//           All four products summed per cycle.
//
//   The multi-precision multiplier uses a single shared adder tree; only the
//   partial-product width changes with precision. Verified against
//   tools/npu_ref/pe_ref.py per-mode.
//
// ── 2:4 structured sparsity ──────────────────────────────────────────────────
//   sparse_en tells the PE that the upstream weight tile follows 2:4 sparse
//   packing.  For each incoming activation, sparse_skip=1 means "this slot
//   corresponds to a pruned weight — do not accumulate, just pass activation."
//   The sparse_skip flag rides alongside the activation through the array so
//   every PE sees the same skip pattern for the same activation element.
//
// ── Timing ───────────────────────────────────────────────────────────────────
//   All outputs are registered.  One-cycle latency from a_in to a_out.
//   One-cycle latency from (a_in, weight_reg) to psum_out update.
//   Safe for back-to-back weight reloads (load_w and clear_acc are independent).
//
// ── Parameters ───────────────────────────────────────────────────────────────
//   DATA_W — width of activation and weight operands (default 8 for INT8)
//   ACC_W  — width of the accumulator (default 32; must be ≥ 2*DATA_W + log2(K))
//            where K is the maximum accumulation depth per tile.  32 gives
//            plenty of headroom for any realistic CNN/transformer layer.
// =============================================================================

module npu_pe #(
    parameter integer DATA_W = 8,
    parameter integer ACC_W  = 32
)(
    input  wire                        clk,
    input  wire                        rst_n,

    // ── Control ──────────────────────────────────────────────────────────────
    input  wire [1:0]                  precision_mode, // 00=INT8, 01=INT4, 10=INT2, 11=FP16
    input  wire                        sparse_en,      // 1 = 2:4 structured sparse tile
    input  wire                        load_w,         // 1-cycle pulse: latch weight_in
    input  wire                        clear_acc,      // 1-cycle pulse: zero the accumulator

    // ── Weight input (loaded once per tile) ──────────────────────────────────
    input  wire signed [DATA_W-1:0]    weight_in,

    // ── Activation input (streams every cycle) ───────────────────────────────
    input  wire                        a_valid,        // 1 when a_in carries real data
    input  wire signed [DATA_W-1:0]    a_in,
    input  wire                        sparse_skip,    // 1 = skip this slot (2:4 zero)

    // ── Activation pass-through to neighbour PE (1-cycle delayed copies) ─────
    output reg                         a_valid_out,
    output reg  signed [DATA_W-1:0]    a_out,
    output reg                         sparse_skip_out,

    // ── Running partial-sum output (combinational view of accumulator) ───────
    output wire signed [ACC_W-1:0]     psum_out
);

    // =========================================================================
    // Held weight register (stationary across the tile)
    // =========================================================================
    reg signed [DATA_W-1:0] weight_reg;

    // =========================================================================
    // Accumulator
    // =========================================================================
    reg signed [ACC_W-1:0]  acc;

    // =========================================================================
    // Multi-precision multiply (combinational)
    //
    //   INT8 mode (00): one INT8 × INT8 → INT16 product.
    //   INT4 mode (01): two INT4 × INT4 → INT8 products, summed to INT9.
    //   INT2 mode (10): four INT2 × INT2 → INT4 products, summed to INT6.
    //
    //   All mode outputs sign-extended to ACC_W before accumulator add.
    // =========================================================================

    // INT8 product (always computed; used when precision_mode == 2'b00)
    wire signed [2*DATA_W-1:0] int8_product = weight_reg * a_in;

    // INT4 pair — unpack and multiply
    wire signed [3:0] w_hi_i4 = weight_reg[7:4];
    wire signed [3:0] w_lo_i4 = weight_reg[3:0];
    wire signed [3:0] a_hi_i4 = a_in[7:4];
    wire signed [3:0] a_lo_i4 = a_in[3:0];
    wire signed [7:0] i4_prod_hi = w_hi_i4 * a_hi_i4;
    wire signed [7:0] i4_prod_lo = w_lo_i4 * a_lo_i4;
    wire signed [8:0] int4_sum   = i4_prod_hi + i4_prod_lo;

    // INT2 quads — unpack and multiply
    wire signed [1:0] w_i2_3 = weight_reg[7:6];
    wire signed [1:0] w_i2_2 = weight_reg[5:4];
    wire signed [1:0] w_i2_1 = weight_reg[3:2];
    wire signed [1:0] w_i2_0 = weight_reg[1:0];
    wire signed [1:0] a_i2_3 = a_in[7:6];
    wire signed [1:0] a_i2_2 = a_in[5:4];
    wire signed [1:0] a_i2_1 = a_in[3:2];
    wire signed [1:0] a_i2_0 = a_in[1:0];
    wire signed [3:0] i2_p3 = w_i2_3 * a_i2_3;
    wire signed [3:0] i2_p2 = w_i2_2 * a_i2_2;
    wire signed [3:0] i2_p1 = w_i2_1 * a_i2_1;
    wire signed [3:0] i2_p0 = w_i2_0 * a_i2_0;
    wire signed [5:0] int2_sum = i2_p3 + i2_p2 + i2_p1 + i2_p0;

    // Select + sign-extend to ACC_W based on precision_mode
    wire signed [ACC_W-1:0] product_ext =
        (precision_mode == 2'b01) ?
            {{(ACC_W - 9){int4_sum[8]}}, int4_sum} :
        (precision_mode == 2'b10) ?
            {{(ACC_W - 6){int2_sum[5]}}, int2_sum} :
        // default (INT8 and FP16-placeholder) uses INT8 path
            {{(ACC_W - 2*DATA_W){int8_product[2*DATA_W-1]}}, int8_product};

    // Accumulate only when activation is valid and the slot is not pruned.
    wire do_acc = a_valid && !sparse_skip;

    // =========================================================================
    // State update
    // =========================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            weight_reg      <= {DATA_W{1'b0}};
            acc             <= {ACC_W{1'b0}};
            a_out           <= {DATA_W{1'b0}};
            a_valid_out     <= 1'b0;
            sparse_skip_out <= 1'b0;
        end else begin
            // Weight load (single-cycle pulse; holds otherwise)
            if (load_w)
                weight_reg <= weight_in;

            // Accumulator: clear wins over accumulate in the same cycle.
            if (clear_acc)
                acc <= {ACC_W{1'b0}};
            else if (do_acc)
                acc <= acc + product_ext;

            // Activation pass-through (1-cycle delay to neighbour PE).
            a_out           <= a_in;
            a_valid_out     <= a_valid;
            sparse_skip_out <= sparse_skip;
        end
    end

    assign psum_out = acc;

    // =========================================================================
    // Simulation-only: flag FP16 mode (still a placeholder; V2 deliverable).
    // =========================================================================
    // synthesis translate_off
    always @(posedge clk) begin
        if (rst_n && a_valid && precision_mode == 2'b11) begin
            $display("[%0t] npu_pe: precision_mode=FP16 is placeholder (v2); using INT8",
                     $time);
        end
    end
    // synthesis translate_on

endmodule
