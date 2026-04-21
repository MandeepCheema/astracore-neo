`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — NPU SRAM Bank with SECDED ECC  (npu_sram_bank_ecc.v)
// =============================================================================
// Combinational-ECC wrapper around npu_sram_bank that preserves the
// 1-cycle read latency of the underlying primitive while adding
// SECDED (Single Error Correct, Double Error Detect) coverage on
// every read.
//
// ── Why this exists (F4-A-1) ────────────────────────────────────────────────
// As of 2026-04-20 npu_top.v's "Current gaps" comment block reads:
//   "No ECC, no BIST, no debug mux (those are Phase 2 hardening items)."
// The FMEDA on npu_top (docs/safety/fmeda/npu_top_fmeda.md) flagged
// the resulting SRAM data exposure (npu_top.sram_data.seu) as ~0.076
// FIT of dangerous-undetected — the single largest closure
// opportunity in Phase A of the remediation plan
// (docs/safety/findings_remediation_plan_v0_1.md, F4-A-1).
//
// ── Integration status ──────────────────────────────────────────────────────
// This wrapper is delivered as a drop-in for new npu_top instantiations.
// The actual npu_top.v swap (replacing each npu_sram_bank with this
// wrapper, threading the ECC fault flags up to fault_detected[]) is
// follow-up WP F4-A-1.1 — scheduled for the next WSL session because
// it changes module interfaces in npu_top and requires the cocotb
// regression suite to re-validate.
//
// ── Architecture ────────────────────────────────────────────────────────────
//   Storage layout:
//     - One underlying npu_sram_bank instance with DATA_W = DATA_W + PARITY_W
//     - Each row stores {parity[7:0], data_padded[63:0]}
//     - Data widths > 64 bits are not yet supported; assert at elaboration
//   Encode (write path, combinational):
//     - Pad wdata to 64 bits with zeros
//     - Compute SECDED parity combinationally (re-use ecc_secded encode logic)
//     - Concatenate {parity, data_padded} and write to bank
//   Decode (read path, combinational):
//     - Unpack stored row into {parity, data_padded}
//     - Recompute syndrome combinationally
//     - Output corrected data + single_err / double_err flags
//   Latency:
//     - Read: 1 cycle (matches plain npu_sram_bank — no extra register)
//     - Write: 1 cycle (combinational encode, then 1-cycle bank write)
//
// ── Counter outputs (matched to SEooC §2.3 boundary signals) ────────────────
//   ecc_corrected_count[15:0]   — saturating count of single-bit corrections
//   ecc_uncorrected_count[7:0]  — saturating count of double-bit detections
//   The counters wrap to *saturate* (not roll over) so the licensee
//   supervisor sees a monotone trend per AoU-8 in SEooC §6.2.
//
// ── DC contribution ─────────────────────────────────────────────────────────
//   target_dc_pct: 99.5  (per safety_mechanisms.yaml ecc_secded entry)
//   target_dc_lf_pct: 99.0
//   These targets become measured numbers after the ecc_secded_bf_10k
//   fault-injection campaign (F4-C-2, planned W4 per gap analysis).
// =============================================================================

module npu_sram_bank_ecc #(
    parameter integer DATA_W = 32,
    parameter integer DEPTH  = 256,
    parameter integer ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // Write port
    input  wire                 we,
    input  wire [ADDR_W-1:0]    waddr,
    input  wire [DATA_W-1:0]    wdata,

    // Read port
    input  wire                 re,
    input  wire [ADDR_W-1:0]    raddr,
    output wire [DATA_W-1:0]    rdata,

    // ── ECC fault outputs ──────────────────────────────────────────────────
    output wire                 single_err,        // 1-cycle pulse on read
    output wire                 double_err,        // 1-cycle pulse on read
    output reg  [15:0]          ecc_corrected_count,   // saturating
    output reg  [7:0]           ecc_uncorrected_count  // saturating
);

    // -------------------------------------------------------------------------
    // Width sanity check — we wrap ecc_secded which is fixed at 64-bit data.
    // -------------------------------------------------------------------------
    initial begin
        if (DATA_W > 64) begin
            $display("[%0t] npu_sram_bank_ecc: DATA_W=%0d > 64 not supported. "
                     "Slice into multiple 64-bit ECC words at the next layer.",
                     $time, DATA_W);
            $stop;
        end
    end

    localparam integer PARITY_W = 8;
    localparam integer STORE_W  = 64 + PARITY_W;   // 72-bit codeword

    // -------------------------------------------------------------------------
    // Underlying SRAM stores 72-bit codewords
    // -------------------------------------------------------------------------
    wire [STORE_W-1:0] bank_rdata;
    wire [STORE_W-1:0] bank_wdata;

    // Pad write data to 64 bits before encoding
    wire [63:0] wdata_pad = { {(64-DATA_W){1'b0}}, wdata };

    // Combinational SECDED encode: same Hamming(72,64) bit pattern as
    // rtl/ecc_secded/ecc_secded.v but inlined so the round-trip stays
    // single-cycle (the registered ecc_secded module would add a cycle).
    function automatic [PARITY_W-1:0] secded_encode(input [63:0] data);
        integer i, j;
        reg [6:0] h;
        reg overall;
        begin
            h = 7'b0;
            for (i = 0; i < 7; i = i + 1) begin
                for (j = 0; j < 64; j = j + 1) begin
                    if (((j + 1) >> i) & 1)
                        h[i] = h[i] ^ data[j];
                end
            end
            overall = ^data ^ ^h;
            secded_encode = {overall, h};
        end
    endfunction

    // Combinational SECDED decode: returns {corrected_data, single, double}
    // single → corrected_data is valid; double → uncorrectable.
    function automatic [65:0] secded_decode(input [63:0] data, input [7:0] parity);
        integer i, j;
        reg [6:0] h_recv;
        reg overall_recv;
        reg [6:0] syndrome;
        reg overall_xor;
        reg [63:0] corrected;
        reg single, double;
        integer pos;
        begin
            h_recv = parity[6:0];
            overall_recv = parity[7];
            // Recompute Hamming on received data
            syndrome = 7'b0;
            for (i = 0; i < 7; i = i + 1) begin
                for (j = 0; j < 64; j = j + 1) begin
                    if (((j + 1) >> i) & 1)
                        syndrome[i] = syndrome[i] ^ data[j];
                end
                syndrome[i] = syndrome[i] ^ h_recv[i];
            end
            overall_xor = ^data ^ ^h_recv ^ overall_recv;
            // Classification:
            //   syndrome=0 + overall_xor=0 → no error
            //   syndrome=0 + overall_xor=1 → parity-bit single-bit flip → corrected
            //   syndrome!=0 + overall_xor=1 → data single-bit flip → correct it
            //   syndrome!=0 + overall_xor=0 → double-bit error → uncorrectable
            corrected = data;
            single = 1'b0;
            double = 1'b0;
            if (syndrome == 0 && overall_xor == 1'b1) begin
                single = 1'b1;  // parity bit was flipped; data is intact
            end else if (syndrome != 0 && overall_xor == 1'b1) begin
                single = 1'b1;
                pos = syndrome;  // 1-indexed bit position
                if (pos >= 1 && pos <= 64)
                    corrected[pos-1] = ~data[pos-1];
            end else if (syndrome != 0 && overall_xor == 1'b0) begin
                double = 1'b1;
            end
            secded_decode = {corrected, single, double};
        end
    endfunction

    // Encode on write
    wire [PARITY_W-1:0] wparity = secded_encode(wdata_pad);
    assign bank_wdata = {wparity, wdata_pad};

    // Decode on read
    wire [63:0] r_data_pad = bank_rdata[63:0];
    wire [PARITY_W-1:0] r_parity = bank_rdata[STORE_W-1 -: PARITY_W];
    wire [65:0] decoded = secded_decode(r_data_pad, r_parity);
    wire [63:0] corrected_pad = decoded[65:2];
    assign single_err = decoded[1];
    assign double_err = decoded[0];
    assign rdata = corrected_pad[DATA_W-1:0];

    // Underlying bank
    npu_sram_bank #(
        .DATA_W (STORE_W),
        .DEPTH  (DEPTH)
    ) u_bank (
        .clk    (clk),
        .rst_n  (rst_n),
        .we     (we),
        .waddr  (waddr),
        .wdata  (bank_wdata),
        .re     (re),
        .raddr  (raddr),
        .rdata  (bank_rdata)
    );

    // -------------------------------------------------------------------------
    // Saturating counters for licensee supervisor (SEooC §2.3 + AoU-8)
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            ecc_corrected_count   <= 16'h0000;
            ecc_uncorrected_count <= 8'h00;
        end else begin
            if (re && single_err && (ecc_corrected_count != 16'hFFFF))
                ecc_corrected_count <= ecc_corrected_count + 16'd1;
            if (re && double_err && (ecc_uncorrected_count != 8'hFF))
                ecc_uncorrected_count <= ecc_uncorrected_count + 8'd1;
        end
    end

endmodule
