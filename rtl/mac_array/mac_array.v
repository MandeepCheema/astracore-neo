`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — MAC Array RTL
// =============================================================================
// Implements a signed INT8 multiply-accumulate unit with 32-bit accumulator.
//
// Each valid cycle computes:
//   result = result + (signed(a) * signed(b))    when clear == 0
//   result = signed(a) * signed(b)               when clear == 1
//
// This represents one leaf MAC unit of the 24,576-MAC array. The accumulator
// models the ACCUMULATE pipeline stage; clear resets it at the start of a
// new matrix tile.
//
// Precision:
//   INT8 (8-bit signed × 8-bit signed → 16-bit product, 32-bit accumulate)
//
// Interface:
//   clk     — system clock (rising edge active)
//   rst_n   — active-low synchronous reset
//   valid   — pulse high when (a, b) inputs are ready
//   clear   — clear accumulator before adding (start of new dot-product)
//   a       — signed 8-bit input (activation / weight)
//   b       — signed 8-bit input (weight / activation)
//   result  — 32-bit signed accumulated result
//   ready   — registered valid: 1 cycle after valid, result is stable
//
// Saturation:
//   The 32-bit accumulator does NOT saturate (overflow wraps) — matching
//   the Python np.int32 behaviour. No saturation flag is implemented.
// =============================================================================

module mac_array (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid,
    input  wire        clear,
    input  wire signed [7:0]  a,
    input  wire signed [7:0]  b,

    output reg  signed [31:0] result,
    output reg                ready
);

    // -------------------------------------------------------------------------
    // Signed 8×8 multiply → 16-bit product (combinational)
    // -------------------------------------------------------------------------
    wire signed [15:0] product = a * b;   // sign-extends automatically in Verilog

    // -------------------------------------------------------------------------
    // Accumulate / clear
    // -------------------------------------------------------------------------
    wire signed [31:0] next_result =
        clear ? {{16{product[15]}}, product} :          // sign-extend to 32 bits
                result + {{16{product[15]}}, product};  // accumulate

    // -------------------------------------------------------------------------
    // Sequential logic
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            result <= 32'sd0;
            ready  <= 1'b0;
        end else begin
            ready  <= valid;    // ready pulses one cycle after valid
            if (valid) begin
                result <= next_result;
            end
        end
    end

endmodule
