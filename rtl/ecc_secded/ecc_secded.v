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
    // h[i] = XOR of data[j] where ((j+1) >> i) & 1 is set, for j in 0..63
    integer i_h, j_h;
    reg [6:0] h_comb;
    always @(*) begin
        h_comb = 7'h0;
        for (i_h = 0; i_h < 7; i_h = i_h + 1) begin
            for (j_h = 0; j_h < 64; j_h = j_h + 1) begin
                if (((j_h + 1) >> i_h) & 1)
                    h_comb[i_h] = h_comb[i_h] ^ data_in[j_h];
            end
        end
    end

    // Overall parity of data bits (used in encode and decode)
    wire data_parity = ^data_in;    // XOR reduction of all 64 data bits

    // -------------------------------------------------------------------------
    // Encode path
    // -------------------------------------------------------------------------
    wire [6:0] h_parity_bits = total_h_parity;
    wire overall_parity_enc;

    // Total 1-bits in data + h[6:0] → must be even; p7 set to fix
    // overall_parity_enc = XOR of data bits ^ XOR of h bits
    // If result is 1, total is odd, set p7=1 to make even
    reg [6:0] total_h_parity;
    always @(*) begin
        total_h_parity = h_comb;
    end

    wire h_xor = ^h_comb;                          // XOR of 7 Hamming bits
    assign overall_parity_enc = data_parity ^ h_xor; // = 0 if even, 1 if odd
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

    // Corrected data word
    reg [63:0] data_corrected;
    integer cp;
    always @(*) begin
        data_corrected = data_in;
        if (data_bit_err) begin
            for (cp = 0; cp < 64; cp = cp + 1) begin
                if (error_position == (cp + 1))
                    data_corrected[cp] = data_in[cp] ^ 1'b1;
            end
        end
    end

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

endmodule
