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

endmodule
