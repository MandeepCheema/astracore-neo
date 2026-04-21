`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — ECC SECDED RTL
// =============================================================================
// Implements SECDED (Single Error Correct, Double Error Detect) using
// Hamming(72,64): 64 data bits + 8 parity bits.
//
// Parity computation:
//   h[i] = XOR of all data bits j where ((j+1) >> i) & 1 == 1, for i in 0..6
//   p7   = XOR of all data bits and all h[6:0] bits (overall even parity)
//   parity[7:0] = {p7, h[6:0]}
//
// Encode (mode == 0):
//   parity_out = computed parity for data_in
//   data_out   = data_in (pass-through)
//   single_err = double_err = corrected = 0
//
// Decode (mode == 1):
//   Takes data_in[63:0] + parity_in[7:0] (received 72-bit codeword)
//   Recomputes syndrome; classifies and corrects error.
//   single_err=1 : single-bit error corrected (data_out has corrected value)
//   double_err=1  : double-bit error detected (uncorrectable — data_out invalid)
//   corrected=1   : same as single_err (alias)
//   err_pos        : 1-indexed error bit position (0 = parity bit was flipped)
//
// Interface:
//   clk        — system clock (rising edge active)
//   rst_n      — active-low synchronous reset
//   valid      — pulse high when inputs are ready
//   mode       — 0=encode, 1=decode
//   data_in    — 64-bit data input
//   parity_in  — 8-bit parity (decode mode only)
//   data_out   — 64-bit output (corrected data in decode mode)
//   parity_out — 8-bit computed parity (encode mode)
//   single_err — single-bit error detected and corrected
//   double_err — double-bit error detected (uncorrectable)
//   corrected  — synonym for single_err
//   err_pos    — 7-bit error position (1-indexed; 0 = parity bit)
// =============================================================================

module ecc_secded (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid,
    input  wire        mode,           // 0=encode, 1=decode
    input  wire [63:0] data_in,
    input  wire [7:0]  parity_in,     // decode mode: received parity

    output reg  [63:0] data_out,
    output reg  [7:0]  parity_out,
    output reg         single_err,
    output reg         double_err,
    output reg         corrected,
    output reg  [6:0]  err_pos
);

    // -------------------------------------------------------------------------
    // Combinational Hamming parity computation
    // -------------------------------------------------------------------------
    // h[i] = XOR of data[j] where ((j+1) >> i) & 1 is set, for j in 0..63.
    //
    // Refactored 2026-04-21 from a nested-for-loop in always_comb to
    // explicit XOR-reduce over precomputed bit-masks per parity bit.
    // The original nested-for produced ~448 conditional XORs that
    // g++ 11 -Os spent 10+ minutes optimising on the generated
    // Vtop__ALL.cpp — a real pathological case.  XOR-reduce of
    // (data & mask) compiles to a single x86 popcnt-parity instruction
    // via __builtin_parityll, so g++ closes -Os in ms.
    //
    // Bit-exact equivalence vs the original nested loop verified by
    // tools/_verify_ecc_masks.py (100/100 random 64-bit data words
    // round-trip identical); see the masks' provenance in that script.
    //
    // Mask coverage per parity bit (1-indexed, j=0..63 → codeword pos j+1):
    //   H_MASK_0 (bit 0 of pos): every odd position    → 0x5555_5555_5555_5555
    //   H_MASK_1 (bit 1 of pos): pos mod 4 in {2,3}    → 0x6666_6666_6666_6666
    //   H_MASK_2 (bit 2 of pos): pos mod 8 in {4..7}   → 0x7878_7878_7878_7878
    //   H_MASK_3 (bit 3 of pos): pos mod 16 in {8..15} → 0x7F80_7F80_7F80_7F80
    //   H_MASK_4 (bit 4 of pos): pos mod 32 in {16..31}→ 0x7FFF_8000_7FFF_8000
    //   H_MASK_5 (bit 5 of pos): pos mod 64 in {32..63}→ 0x7FFF_FFFF_8000_0000
    //   H_MASK_6 (bit 6 of pos): pos == 64             → 0x8000_0000_0000_0000
    localparam [63:0] H_MASK_0 = 64'h5555_5555_5555_5555;
    localparam [63:0] H_MASK_1 = 64'h6666_6666_6666_6666;
    localparam [63:0] H_MASK_2 = 64'h7878_7878_7878_7878;
    localparam [63:0] H_MASK_3 = 64'h7F80_7F80_7F80_7F80;
    localparam [63:0] H_MASK_4 = 64'h7FFF_8000_7FFF_8000;
    localparam [63:0] H_MASK_5 = 64'h7FFF_FFFF_8000_0000;
    localparam [63:0] H_MASK_6 = 64'h8000_0000_0000_0000;

    wire [6:0] h_comb;
    assign h_comb[0] = ^(data_in & H_MASK_0);
    assign h_comb[1] = ^(data_in & H_MASK_1);
    assign h_comb[2] = ^(data_in & H_MASK_2);
    assign h_comb[3] = ^(data_in & H_MASK_3);
    assign h_comb[4] = ^(data_in & H_MASK_4);
    assign h_comb[5] = ^(data_in & H_MASK_5);
    assign h_comb[6] = ^(data_in & H_MASK_6);

    // Overall parity of data bits (used in encode and decode)
    wire data_parity = ^data_in;    // XOR reduction of all 64 data bits

    // -------------------------------------------------------------------------
    // Encode path
    // -------------------------------------------------------------------------
    wire [6:0] h_parity_bits = total_h_parity;
    wire overall_parity_enc;

    // Total 1-bits in data + h[6:0] → must be even; p7 set to fix.
    // overall_parity_enc = XOR of data bits ^ XOR of h bits;
    // if 1, total is odd, set p7 to make even.
    wire [6:0] total_h_parity = h_comb;
    wire h_xor               = ^h_comb;
    assign overall_parity_enc = data_parity ^ h_xor;
    wire [7:0] parity_computed = {overall_parity_enc, h_comb};

    // -------------------------------------------------------------------------
    // Decode path
    // -------------------------------------------------------------------------
    // Syndrome: compare received Hamming bits with recomputed Hamming bits
    wire [6:0] h_syndrome = parity_in[6:0] ^ h_comb;

    // Overall parity check: XOR of all received bits (data + h[6:0] + p7)
    // Should be 0 (even) for clean codeword
    wire overall_recv_parity = data_parity ^ (^parity_in[6:0]) ^ parity_in[7];

    // Error classification
    wire no_error     = (h_syndrome == 7'h0) && (overall_recv_parity == 1'b0);
    wire single_error = (overall_recv_parity == 1'b1);                // any single bit flip
    wire double_error = (overall_recv_parity == 1'b0) && (h_syndrome != 7'h0);

    // Correction: h_syndrome is 1-indexed position of erroneous data bit
    // Valid data bit range: 1..64.  0 or >64 means parity bit was flipped.
    wire [6:0] error_position = h_syndrome;   // 1-indexed, 0 if parity bit err
    wire data_bit_err = single_error && (error_position >= 7'd1) && (error_position <= 7'd64);

    // Corrected data word. error_position is 1-indexed; flip bit
    // (error_position - 1) when data_bit_err is asserted.
    //
    // Refactored 2026-04-21 from a 64-iteration for-loop with
    // per-iteration equality compare to a single shift-and-XOR.
    // The simulator emits a 64-bit shift expression that g++ -Os
    // compiles trivially (bit-exact equivalent — verified against
    // the prior loop semantics by inspection: only the bit at
    // position (error_position - 1) is XOR-flipped, exactly matching
    // the loop's per-iteration condition).
    wire [63:0] correction_mask = data_bit_err
                                  ? (64'd1 << (error_position - 7'd1))
                                  : 64'd0;
    wire [63:0] data_corrected  = data_in ^ correction_mask;

    // -------------------------------------------------------------------------
    // Sequential output register
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            data_out   <= 64'h0;
            parity_out <= 8'h0;
            single_err <= 1'b0;
            double_err <= 1'b0;
            corrected  <= 1'b0;
            err_pos    <= 7'h0;
        end else if (valid) begin
            if (!mode) begin
                // Encode
                parity_out <= parity_computed;
                data_out   <= data_in;
                single_err <= 1'b0;
                double_err <= 1'b0;
                corrected  <= 1'b0;
                err_pos    <= 7'h0;
            end else begin
                // Decode
                parity_out <= parity_computed;   // expose recomputed syndrome
                data_out   <= data_corrected;
                single_err <= single_error;
                double_err <= double_error;
                corrected  <= single_error;
                err_pos    <= error_position;
            end
        end
    end

    // =========================================================================
    // SECDED safety invariants — Phase-F formal verification targets
    // =========================================================================
`ifndef SYNTHESIS
`ifndef __ICARUS__
    // Invariant 1: single_err and double_err are mutually exclusive.
    property p_err_mutex;
        @(posedge clk) disable iff (!rst_n)
        !(single_err && double_err);
    endproperty
    a_err_mutex: assert property (p_err_mutex)
        else $error("ECC: single_err and double_err both set (mutex violation)");

    // Invariant 2: corrected=1 iff single_err=1 (only single-bit errors are
    // corrected by SECDED; double-bit errors are detected, not corrected).
    property p_corrected_iff_single;
        @(posedge clk) disable iff (!rst_n)
        corrected == single_err;
    endproperty
    a_corrected_iff_single: assert property (p_corrected_iff_single)
        else $error("ECC: corrected != single_err");

    // Invariant 3: encode mode (mode=0) never asserts error outputs.
    property p_encode_no_errors;
        @(posedge clk) disable iff (!rst_n)
        valid && (mode == 1'b0)
            |=> (!single_err && !double_err && !corrected);
    endproperty
    a_encode_no_errors: assert property (p_encode_no_errors)
        else $error("ECC: encode mode produced error flag");
`endif
`endif

endmodule
