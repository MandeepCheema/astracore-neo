// =============================================================================
// AstraCore Neo — Fault-injection testbench wrapper for tmr_voter.
//
// Instantiates rtl/tmr_voter/tmr_voter.v as `u_dut` so the cocotb
// runner can resolve hierarchical paths like `tmr_voter_fi_tb.lane_a_reg`.
//
// Drives a steady, consistent vote pattern so any single-lane
// perturbation is exclusively the result of the cocotb-applied
// injection — no stimulus-induced disagreement to confound DC.
//
// Note on injection targets: tmr_voter is mostly combinational and
// has no internal `lane_a_reg`. The injectable storage is the
// *testbench* reg that drives the input port. Cocotb runner therefore
// forces `tmr_voter_fi_tb.lane_a_reg` (this file's reg), not
// `u_dut.lane_a` (the dut's input port — Verilator force does not
// reliably reach port-constant drivers; see
// memory/feedback_verilator_force.md).
// =============================================================================
`timescale 1ns/1ps

module tb_tmr_voter_fi;
    reg                clk;
    reg                rst_n;
    reg                valid;
    reg  [31:0]        lane_a_reg;
    reg  [31:0]        lane_b_reg;
    reg  [31:0]        lane_c_reg;
    wire [31:0]        voted;
    wire               agreement;
    wire               fault_a;
    wire               fault_b;
    wire               fault_c;
    wire               triple_fault;
    wire [1:0]         vote_count;

    // tmr_voter is fixed at 32-bit; no parameter override.
    tmr_voter u_dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .valid        (valid),
        .lane_a       (lane_a_reg),
        .lane_b       (lane_b_reg),
        .lane_c       (lane_c_reg),
        .voted        (voted),
        .agreement    (agreement),
        .fault_a      (fault_a),
        .fault_b      (fault_b),
        .fault_c      (fault_c),
        .triple_fault (triple_fault),
        .vote_count   (vote_count)
    );

    // Steady consistent input — 0xA5A5A5A5 on all three lanes.
    // clk + rst_n + valid are driven by the cocotb runner (Python side)
    // via Timer/RisingEdge; this SV initial block just sets the stable
    // input pattern the runner reads. No absolute delays here so the
    // simulator runs without needing the timing flag.
    initial begin
        lane_a_reg  = 32'hA5A5A5A5;
        lane_b_reg  = 32'hA5A5A5A5;
        lane_c_reg  = 32'hA5A5A5A5;
        valid       = 1'b0;
    end

    // Hold valid = 1 once reset clears, so the voter runs every cycle.
    always @(posedge clk) begin
        if (rst_n) valid <= 1'b1;
        else       valid <= 1'b0;
    end

    // Optional: dump waves for post-mortem if needed.
    initial begin
        $dumpfile("tmr_voter_fi.vcd");
        $dumpvars(0, tb_tmr_voter_fi);
    end
endmodule
