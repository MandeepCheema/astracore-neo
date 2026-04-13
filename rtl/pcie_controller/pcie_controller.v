`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — PCIe Controller RTL
// =============================================================================
// Implements a PCIe link state machine and TLP header assembler.
//
// Link training FSM:
//   3'd0 DETECT   — initial state, receiver detection
//   3'd1 POLLING  — bit-lock and symbol lock
//   3'd2 CONFIG   — lane/link width negotiation
//   3'd3 L0       — active (data transfer enabled)
//   3'd4 L1       — low-power ASPM
//   3'd5 L2       — powered-down
//
// TLP header assembly (3 DWORDs = 96 bits):
//   DWORD 0: {fmt[1:0], type[4:0], T9, TC[2:0], T8, attr, LN, TH, TD, EP,
//              attr[1:0], AT[1:0], length[9:0]}
//   DWORD 1: requester_id[15:0], tag[7:0], last_be[3:0], first_be[3:0]
//   DWORD 2: address[31:0]
//
// TLP types (tlp_type input):
//   2'd0 MEM_READ  (fmt=2'b00, type=5'h0)
//   2'd1 MEM_WRITE (fmt=2'b10, type=5'h0)
//   2'd2 COMPLETION_DATA (fmt=2'b10, type=5'hA)
//
// Interface:
//   clk         — system clock
//   rst_n       — active-low synchronous reset
//   link_up     — assert to advance link state (simulates link training)
//   link_down   — assert to force link to DETECT state
//   tlp_start   — pulse: begin TLP header assembly (only valid in L0 state)
//   tlp_type    — 2-bit TLP type selector
//   req_id      — 16-bit requester ID (bus:dev:fn)
//   tag         — 8-bit transaction tag
//   addr        — 32-bit target address (must be DWORD-aligned)
//   length_dw   — 10-bit length in DWORDs
//   link_state  — 3-bit current link state
//   busy        — TLP assembly in progress
//   tlp_done    — pulsed one cycle when TLP header is complete
//   tlp_hdr     — 96-bit assembled TLP header
// =============================================================================

module pcie_controller (
    input  wire        clk,
    input  wire        rst_n,
    // Link control
    input  wire        link_up,
    input  wire        link_down,
    // TLP request
    input  wire        tlp_start,
    input  wire [1:0]  tlp_type,
    input  wire [15:0] req_id,
    input  wire [7:0]  tag,
    input  wire [31:0] addr,
    input  wire [9:0]  length_dw,

    output reg  [2:0]  link_state,
    output reg         busy,
    output reg         tlp_done,
    output reg  [95:0] tlp_hdr
);

    // Link states
    localparam LS_DETECT  = 3'd0;
    localparam LS_POLLING = 3'd1;
    localparam LS_CONFIG  = 3'd2;
    localparam LS_L0      = 3'd3;
    localparam LS_L1      = 3'd4;
    localparam LS_L2      = 3'd5;

    // TLP types
    localparam TLP_MEM_READ  = 2'd0;
    localparam TLP_MEM_WRITE = 2'd1;
    localparam TLP_CPL_DATA  = 2'd2;

    // TLP assembly states
    localparam TLP_IDLE   = 2'd0;
    localparam TLP_DW0    = 2'd1;
    localparam TLP_DW1    = 2'd2;
    localparam TLP_DW2    = 2'd3;

    reg [1:0] tlp_state;

    // -------------------------------------------------------------------------
    // Link state machine
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n || link_down) begin
            link_state <= LS_DETECT;
        end else if (link_up) begin
            case (link_state)
                LS_DETECT : link_state <= LS_POLLING;
                LS_POLLING: link_state <= LS_CONFIG;
                LS_CONFIG : link_state <= LS_L0;
                default   : /* L0/L1/L2 stay unless link_down */ ;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // TLP header assembly FSM
    // -------------------------------------------------------------------------
    // Format bits: MRd=2'b00, MWr=2'b10, CplD=2'b10
    wire [1:0] fmt = (tlp_type == TLP_MEM_READ) ? 2'b00 : 2'b10;
    // Type code: Mem=5'h00, CplD=5'h0A
    wire [4:0] tcode = (tlp_type == TLP_CPL_DATA) ? 5'h0A : 5'h00;

    // DWORD 0: fmt[1:0] | type[4:0] | reserved | TC[2:0] | reserved x5 |
    //           TD | EP | attr[1:0] | AT[1:0] | length[9:0]
    wire [31:0] dw0 = {fmt, tcode, 1'b0, 3'b000, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0,
                        1'b0, 1'b0, 2'b00, 2'b00, length_dw};

    // DWORD 1: requester_id | tag | last_BE | first_BE
    wire [31:0] dw1 = {req_id, tag, 4'hF, 4'hF};

    // DWORD 2: 32-bit address (DWORD-aligned, lower 2 bits = 0)
    wire [31:0] dw2 = {addr[31:2], 2'b00};

    always @(posedge clk) begin
        if (!rst_n) begin
            busy      <= 1'b0;
            tlp_done  <= 1'b0;
            tlp_state <= TLP_IDLE;
            tlp_hdr   <= 96'h0;
        end else begin
            tlp_done <= 1'b0;   // default de-assert

            case (tlp_state)
                TLP_IDLE: begin
                    if (tlp_start && (link_state == LS_L0)) begin
                        busy      <= 1'b1;
                        tlp_state <= TLP_DW0;
                    end
                end

                TLP_DW0: begin
                    tlp_hdr[31:0]  <= dw0;
                    tlp_state <= TLP_DW1;
                end

                TLP_DW1: begin
                    tlp_hdr[63:32] <= dw1;
                    tlp_state <= TLP_DW2;
                end

                TLP_DW2: begin
                    tlp_hdr[95:64] <= dw2;
                    busy      <= 1'b0;
                    tlp_done  <= 1'b1;
                    tlp_state <= TLP_IDLE;
                end

                default: tlp_state <= TLP_IDLE;
            endcase
        end
    end

endmodule
