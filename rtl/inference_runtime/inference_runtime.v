`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — Inference Runtime RTL
// =============================================================================
// Implements the inference session state machine.
//
// Session lifecycle:
//   UNLOADED (3'd0) → LOADED (3'd1) → RUNNING (3'd2) → DONE (3'd3)
//                                                     ↑
//                  ERROR (3'd4) ←── any state on abort
//
// Transitions:
//   UNLOADED  + load_start + model_valid → LOADED
//   UNLOADED  + load_start + !model_valid→ ERROR
//   LOADED    + run_start               → RUNNING
//   RUNNING   + run_done_in             → DONE
//   DONE      + load_start              → LOADED  (reload for next session)
//   DONE      + run_start               → RUNNING (re-run same model)
//   any state + abort                   → ERROR
//   ERROR     + load_start              → UNLOADED (recovery)
//
// Interface:
//   clk          — system clock (rising edge active)
//   rst_n        — active-low synchronous reset
//   load_start   — request to load a model
//   model_valid  — model data is valid when load_start is asserted
//   run_start    — request to start inference
//   abort        — abort current session
//   run_done_in  — inference engine signals completion
//   state        — 3-bit session state
//   busy         — asserted when state is LOADED or RUNNING
//   session_done — pulsed one cycle when RUNNING → DONE
//   error        — asserted when state is ERROR
// =============================================================================

module inference_runtime (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       load_start,
    input  wire       model_valid,
    input  wire       run_start,
    input  wire       abort,
    input  wire       run_done_in,

    output reg  [2:0] state,
    output wire       busy,
    output reg        session_done,
    output wire       error
);

    localparam ST_UNLOADED = 3'd0;
    localparam ST_LOADED   = 3'd1;
    localparam ST_RUNNING  = 3'd2;
    localparam ST_DONE     = 3'd3;
    localparam ST_ERROR    = 3'd4;

    assign busy  = (state == ST_LOADED) || (state == ST_RUNNING);
    assign error = (state == ST_ERROR);

    // -------------------------------------------------------------------------
    // Next-state logic
    // -------------------------------------------------------------------------
    reg [2:0] next_state;
    always @(*) begin
        next_state = state;

        if (abort) begin
            next_state = ST_ERROR;
        end else begin
            case (state)
                ST_UNLOADED: begin
                    if (load_start)
                        next_state = model_valid ? ST_LOADED : ST_ERROR;
                end

                ST_LOADED: begin
                    if (run_start)
                        next_state = ST_RUNNING;
                end

                ST_RUNNING: begin
                    if (run_done_in)
                        next_state = ST_DONE;
                end

                ST_DONE: begin
                    if (load_start)
                        next_state = model_valid ? ST_LOADED : ST_ERROR;
                    else if (run_start)
                        next_state = ST_RUNNING;
                end

                ST_ERROR: begin
                    if (load_start)
                        next_state = ST_UNLOADED;   // recovery: go back to unloaded
                end

                default: next_state = ST_UNLOADED;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Sequential state register
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state        <= ST_UNLOADED;
            session_done <= 1'b0;
        end else begin
            state        <= next_state;
            // session_done pulses for one cycle on RUNNING → DONE
            session_done <= (state == ST_RUNNING) && (next_state == ST_DONE);
        end
    end

endmodule
