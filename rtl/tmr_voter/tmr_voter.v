`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — TMR Voter RTL
// =============================================================================
// Implements Triple Modular Redundancy majority voting for ASIL-D safety paths.
//
// Three independent 32-bit compute lanes (A, B, C) produce the same result.
// This voter selects the correct output via majority vote and flags the
// disagreeing lane (if any).
//
// Vote rules:
//   A == B == C : voted = A, agreement = 1, fault_* = 0, vote_count = 3
//   A == B != C : voted = A, fault_c = 1, vote_count = 2
//   A == C != B : voted = A, fault_b = 1, vote_count = 2
//   B == C != A : voted = B, fault_a = 1, vote_count = 2
//   A != B != C : triple_fault = 1, agreement = 0, voted = A (undefined)
//
// Interface:
//   clk         — system clock (rising edge active)
//   rst_n       — active-low synchronous reset
//   valid       — pulse high when lane values are ready
//   lane_a/b/c  — 32-bit input from each redundant lane
//   voted       — 32-bit majority-voted output
//   agreement   — high if at least 2 lanes agreed
//   fault_a/b/c — high if that lane produced a minority value
//   triple_fault— high if all three lanes disagree (no majority)
//   vote_count  — 2 or 3 (number of lanes that agreed on voted value)
// =============================================================================

module tmr_voter (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid,
    input  wire [31:0] lane_a,
    input  wire [31:0] lane_b,
    input  wire [31:0] lane_c,

    output reg  [31:0] voted,
    output reg         agreement,
    output reg         fault_a,
    output reg         fault_b,
    output reg         fault_c,
    output reg         triple_fault,
    output reg  [1:0]  vote_count
);

    // -------------------------------------------------------------------------
    // Combinational vote logic
    // -------------------------------------------------------------------------
    wire ab_eq = (lane_a == lane_b);
    wire ac_eq = (lane_a == lane_c);
    wire bc_eq = (lane_b == lane_c);

    // Majority selection
    wire [31:0] voted_next =
        (ab_eq || ac_eq) ? lane_a :   // A wins (also covers all-agree)
        bc_eq            ? lane_b :   // B==C, A is faulty
                           lane_a;    // triple fault — default A (undefined)

    wire agreement_next   = ab_eq || ac_eq || bc_eq;
    wire fault_a_next     = !ab_eq && !ac_eq && bc_eq;
    wire fault_b_next     = !ab_eq && ac_eq  && !bc_eq;
    wire fault_c_next     = ab_eq  && !ac_eq && !bc_eq;
    wire triple_fault_next= !ab_eq && !ac_eq && !bc_eq;
    wire [1:0] vote_count_next =
        (ab_eq && ac_eq) ? 2'd3 :
        (ab_eq || ac_eq || bc_eq) ? 2'd2 :
        2'd0;

    // -------------------------------------------------------------------------
    // Sequential registers
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            voted       <= 32'h0;
            agreement   <= 1'b0;
            fault_a     <= 1'b0;
            fault_b     <= 1'b0;
            fault_c     <= 1'b0;
            triple_fault<= 1'b0;
            vote_count  <= 2'd0;
        end else if (valid) begin
            voted       <= voted_next;
            agreement   <= agreement_next;
            fault_a     <= fault_a_next;
            fault_b     <= fault_b_next;
            fault_c     <= fault_c_next;
            triple_fault<= triple_fault_next;
            vote_count  <= vote_count_next;
        end
    end

    // =========================================================================
    // ASIL-D safety invariants (SystemVerilog Assertions)
    // =========================================================================
    // These assertions encode the safety properties the module must
    // preserve.  Verified by simulation (cocotb) and targeted by the
    // Phase-F formal flow (SymbiYosys / JasperGold).
    //
    // Iverilog 12 does not support SVA property/assert property syntax.
    // Assertions are guarded by `ifndef SYNTHESIS` (simulation-only) and
    // `ifndef __ICARUS__` (skipped in iverilog).  Compile in Verilator
    // v5.030+ and Cadence Xcelium / Synopsys VCS, which is what Phase-F
    // formal verification targets.
`ifndef SYNTHESIS
`ifndef __ICARUS__
    // Invariant 1: agreement implies at least one pair of lanes matched.
    property p_agreement_consistent;
        @(posedge clk) disable iff (!rst_n)
        agreement |-> (ab_eq || ac_eq || bc_eq);
    endproperty
    a_agreement_consistent: assert property (p_agreement_consistent)
        else $error("TMR: agreement=1 but no lane-pair match");

    // Invariant 2: triple_fault is mutually exclusive with agreement.
    property p_no_agreement_in_triple_fault;
        @(posedge clk) disable iff (!rst_n)
        triple_fault |-> !agreement;
    endproperty
    a_no_agreement_in_triple_fault:
        assert property (p_no_agreement_in_triple_fault)
        else $error("TMR: triple_fault=1 and agreement=1 simultaneously");

    // Invariant 3: vote_count is one of {0, 2, 3} — 1 is impossible.
    property p_vote_count_valid;
        @(posedge clk) disable iff (!rst_n)
        (vote_count != 2'd1);
    endproperty
    a_vote_count_valid: assert property (p_vote_count_valid)
        else $error("TMR: vote_count=1 is impossible by design");

    // Invariant 4: at most one single-lane fault can be set at a time.
    property p_single_fault_lane;
        @(posedge clk) disable iff (!rst_n)
        ($countones({fault_a, fault_b, fault_c}) <= 1);
    endproperty
    a_single_fault_lane: assert property (p_single_fault_lane)
        else $error("TMR: multiple single-lane faults set (impossible)");
`endif
`endif

endmodule
