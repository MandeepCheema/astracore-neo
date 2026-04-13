`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — ThermalZone RTL
// =============================================================================
// Implements the on-chip thermal zone state machine.
//
// temp_in: 8-bit unsigned value representing degrees Celsius (0–255 °C).
//
// State encoding:
//   3'd0  NOMINAL   — below warning threshold
//   3'd1  WARNING   — above warning, below throttle
//   3'd2  THROTTLED — above throttle, clock reduced
//   3'd3  CRITICAL  — above critical, prepare for shutdown
//   3'd4  SHUTDOWN  — above shutdown threshold
//
// Interface:
//   clk          — system clock (rising edge active)
//   rst_n        — active-low synchronous reset
//   valid        — pulse high for one cycle when new temp reading is ready
//   temp_in      — 8-bit unsigned temperature (°C)
//   state        — 3-bit current thermal state
//   throttle_en  — asserted when state == THROTTLED
//   shutdown_req — asserted when state == SHUTDOWN
// =============================================================================

module thermal_zone #(
    parameter WARN_THRESH     = 8'd75,    // 75°C warning threshold
    parameter THROTTLE_THRESH = 8'd85,    // 85°C throttle threshold
    parameter CRITICAL_THRESH = 8'd95,    // 95°C critical threshold
    parameter SHUTDOWN_THRESH = 8'd105    // 105°C shutdown threshold
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       valid,
    input  wire [7:0] temp_in,

    output reg  [2:0] state,
    output wire       throttle_en,
    output wire       shutdown_req
);

    localparam S_NOMINAL   = 3'd0;
    localparam S_WARNING   = 3'd1;
    localparam S_THROTTLED = 3'd2;
    localparam S_CRITICAL  = 3'd3;
    localparam S_SHUTDOWN  = 3'd4;

    assign throttle_en  = (state == S_THROTTLED);
    assign shutdown_req = (state == S_SHUTDOWN);

    // -------------------------------------------------------------------------
    // Next-state combinational decode
    // -------------------------------------------------------------------------
    wire [2:0] next_state =
        (temp_in >= SHUTDOWN_THRESH) ? S_SHUTDOWN  :
        (temp_in >= CRITICAL_THRESH) ? S_CRITICAL  :
        (temp_in >= THROTTLE_THRESH) ? S_THROTTLED :
        (temp_in >= WARN_THRESH)     ? S_WARNING   :
                                       S_NOMINAL;

    // -------------------------------------------------------------------------
    // Sequential update
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_NOMINAL;
        end else if (valid) begin
            state <= next_state;
        end
    end

endmodule
