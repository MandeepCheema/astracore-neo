// =============================================================================
// AstraCore Neo — Fault-injection testbench wrapper for ecc_secded.
//
// Drives a steady encode-then-decode round-trip on a fixed data word.
// During the decode cycle, cocotb runner forces bit flips on the
// testbench-level data_in_reg / parity_in_reg (the regs that drive
// the dut's input ports). Internal output registers (single_err /
// double_err / corrected) are sampled as the oracle.
//
// Test pattern per "tick":
//   cycle N+0: assert valid + mode=0 + data_in_reg=0xDEAD..BEEF →
//              dut latches encoded parity into parity_out (1 cycle later)
//   cycle N+2: load parity_in_reg from the latched parity_out
//              (round-trip back through decode)
//   cycle N+3: assert valid + mode=1 → dut decodes; single_err /
//              double_err sample into the oracle
//
// Cocotb runner injects flips during the cycle BEFORE decode (cycle
// N+2 here), so the decode sees a perturbed (data_in_reg, parity_in_reg)
// pair.
// =============================================================================
`timescale 1ns/1ps

module tb_ecc_secded_fi;
    reg          clk;
    reg          rst_n;
    reg          valid;
    reg          mode;       // 0 = encode, 1 = decode
    reg  [63:0]  data_in_reg;
    reg  [7:0]   parity_in_reg;

    wire [63:0]  data_out;
    wire [7:0]   parity_out;
    wire         single_err;
    wire         double_err;
    wire         corrected;
    wire [6:0]   err_pos;

    ecc_secded u_dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .valid      (valid),
        .mode       (mode),
        .data_in    (data_in_reg),
        .parity_in  (parity_in_reg),
        .data_out   (data_out),
        .parity_out (parity_out),
        .single_err (single_err),
        .double_err (double_err),
        .corrected  (corrected),
        .err_pos    (err_pos)
    );

    // clk / rst_n / valid / mode are driven by the cocotb runner
    // (Python side) via Timer/RisingEdge — no absolute delays here so
    // the simulator runs without needing the timing flag. Runner
    // pre-loads parity_in_reg from a known-good encode per injection.
    initial begin
        data_in_reg    = 64'hDEADBEEFCAFEBABE;
        parity_in_reg  = 8'h00;
        valid          = 1'b0;
        mode           = 1'b0;
    end

    // After reset, hold valid=1 + mode=1 (decode mode) every cycle so
    // the cocotb runner can inject on data_in_reg / parity_in_reg and
    // observe single_err / double_err on the next clock edge.
    always @(posedge clk) begin
        if (rst_n) begin
            valid <= 1'b1;
            mode  <= 1'b1;
        end else begin
            valid <= 1'b0;
            mode  <= 1'b0;
        end
    end

    initial begin
        $dumpfile("ecc_secded_fi.vcd");
        $dumpvars(0, tb_ecc_secded_fi);
    end
endmodule
