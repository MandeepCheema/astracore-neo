`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — NPU SRAM Bank Primitive  (npu_sram_bank.v)
// =============================================================================
// Parameterised synchronous 1-read / 1-write memory.  Models what a real
// 1R1W foundry SRAM compiler produces: one address in, one address out,
// one cycle of read latency, write-wins on same-address same-cycle access.
//
// Intentionally simple.  Any ECC, byte-enable, latch-based redundancy, or
// BIST wrapper lives in a higher-level module on top of this primitive.
//
// ── Access semantics (important for tests) ───────────────────────────────────
//   Read  : rdata is registered.  Assert `re` with `raddr` on cycle N → rdata
//           reflects mem[raddr] on cycle N+1.  When `re` is low, rdata holds
//           its previous value (latched output, matches FPGA BRAM + foundry
//           SRAM).
//   Write : `we` with `waddr`/`wdata` on cycle N → mem[waddr] updated at the
//           end of cycle N (post-NBA).
//   Same-cycle same-address R + W : WRITE WINS.  If `re` and `we` are both
//           asserted on the same cycle with the same address, the read that
//           completes on cycle N+1 returns the NEW data written on cycle N.
//           Real SRAM behavior is implementation-defined; tests treat this
//           as the contract of this primitive.
//
// ── Reset behavior ───────────────────────────────────────────────────────────
//   `rdata` is cleared to 0 on rst_n=0.  Memory CONTENTS are NOT cleared —
//   this matches real SRAM, which comes up with undefined contents at power
//   on.  Callers MUST initialise any location before reading it.
// =============================================================================

module npu_sram_bank #(
    parameter integer DATA_W = 8,
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
    output reg  [DATA_W-1:0]    rdata
);

    // Storage — inferred as RAM by most synth tools.  Foundry SRAM compilers
    // replace this with a macro instance during back-end flow.
    reg [DATA_W-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (!rst_n) begin
            rdata <= {DATA_W{1'b0}};
        end else begin
            if (we)
                mem[waddr] <= wdata;
            if (re) begin
                // Write-wins on same-cycle same-address access.
                if (we && (waddr == raddr))
                    rdata <= wdata;
                else
                    rdata <= mem[raddr];
            end
            // When re is low, rdata holds its previous value.
        end
    end

endmodule
