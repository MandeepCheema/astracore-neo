`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — NPU FP Processing Element  (npu_fp_pe.v)  [F1-A1]
// =============================================================================
// Weight-stationary FP processing element built around `npu_fp_mac`.  Drops
// into a systolic row in parallel with `npu_pe.v` (the integer PE); a
// PE-level mux on `precision_mode[2]` selects which accumulator result
// leaves the array.
//
//   precision_mode[2:0]:
//     3'b100 — FP8 E4M3
//     3'b101 — FP8 E5M2
//     3'b110 — FP16
//   MSB = 1 routes through FP path; MSB = 0 uses `npu_pe` (integer).
//   Outer systolic wrapper handles the mux — this module only fires when
//   precision_mode[2] == 1.
//
// ── Dataflow ───────────────────────────────────────────────────────────────
//   Same shape as `npu_pe`:
//     - weight latches on `load_w` (held until next load_w).
//     - activations stream in; a_out is the 1-cycle-delayed copy to the
//       next PE east of this one.
//     - psum_out is a combinational view of the FP accumulator.
//
//   FP MAC is en'd when `a_valid && !sparse_skip`. sparse_skip travels the
//   same 1-cycle delay to the east neighbour as in `npu_pe`.
//
// ── Bit-width notes ────────────────────────────────────────────────────────
//   weight_in + a_in are 16 bits wide. For FP8 modes only the low 8 bits
//   are meaningful (caller zero-pads the upper byte); for FP16 all 16 are
//   used. This matches `npu_fp_mac`'s port contract.
//
//   psum_out is 64 bits: $realtobits of the module's FP64 accumulator.
//   The synthesis-ready F1-A1.1 replacement will publish a native 32-bit
//   IEEE-754 FP32 accumulator instead. The 64-bit sim-time interface is
//   documented as the sim gate contract.
// =============================================================================

module npu_fp_pe (
    input  wire          clk,
    input  wire          rst_n,

    input  wire [2:0]    precision_mode,
    input  wire          sparse_en,     // kept for interface symmetry with npu_pe
    input  wire          load_w,
    input  wire          clear_acc,

    input  wire [15:0]   weight_in,

    input  wire          a_valid,
    input  wire [15:0]   a_in,
    input  wire          sparse_skip,

    output reg           a_valid_out,
    output reg  [15:0]   a_out,
    output reg           sparse_skip_out,

    output wire [63:0]   psum_out
);

    // ── Latched weight ───────────────────────────────────────────────────────
    reg [15:0] weight_reg;

    always @(posedge clk) begin
        if (!rst_n) begin
            weight_reg      <= 16'd0;
            a_out           <= 16'd0;
            a_valid_out     <= 1'b0;
            sparse_skip_out <= 1'b0;
        end else begin
            if (load_w) weight_reg <= weight_in;
            a_out           <= a_in;
            a_valid_out     <= a_valid;
            sparse_skip_out <= sparse_skip;
        end
    end

    // ── FP MAC instance ──────────────────────────────────────────────────────
    wire en_mac = a_valid && !sparse_skip && precision_mode[2];

    npu_fp_mac u_mac (
        .clk            (clk),
        .rst_n          (rst_n),
        .en             (en_mac),
        .clear_acc      (clear_acc),
        .precision_mode (precision_mode),
        .a              (a_in),
        .b              (weight_reg),
        .acc_out        (psum_out)
    );

    // `sparse_en` is not consumed on purpose — interface symmetry with
    // `npu_pe` only. 2:4 sparsity in FP modes is deferred to F1-A3.
    wire _unused_sparse_en = sparse_en;

endmodule
