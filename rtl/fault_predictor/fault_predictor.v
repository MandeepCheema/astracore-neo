`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — Fault Predictor RTL
// =============================================================================
// Implements threshold-based fault risk classification for chip health metrics.
//
// Tracks a rolling window of WINDOW_SIZE samples. Classifies risk based on:
//   1. Threshold breach: value vs WARN_THRESH / CRITICAL_THRESH
//   2. Spike detection: value > rolling_mean + SPIKE_OFFSET (fixed threshold)
//
// Risk levels (3-bit encoding):
//   3'd0  NONE     — metric healthy
//   3'd1  LOW      — above warning (0–30% of warn–critical range)
//   3'd2  MEDIUM   — above warning (30–70%) OR spike detected
//   3'd3  HIGH     — above warning (70–100%)
//   3'd4  CRITICAL — value >= critical threshold
//
// Interface:
//   clk           — system clock (rising edge active)
//   rst_n         — active-low synchronous reset
//   valid         — pulse high when new value is ready
//   value         — 16-bit unsigned metric value
//   risk          — 3-bit risk level
//   alarm         — asserted when risk >= HIGH (3 or 4)
//   rolling_mean  — 16-bit running mean over window (saturates at 0xFFFF)
// =============================================================================

module fault_predictor #(
    parameter WARN_THRESH     = 16'd50,
    parameter CRITICAL_THRESH = 16'd100,
    parameter WINDOW_SIZE     = 16,         // rolling window depth (power of 2)
    parameter SPIKE_OFFSET    = 16'd30      // value > mean + SPIKE_OFFSET → spike
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid,
    input  wire [15:0] value,

    output reg  [2:0]  risk,
    output wire        alarm,
    output reg  [15:0] rolling_mean
);

    assign alarm = (risk >= 3'd3);

    // -------------------------------------------------------------------------
    // Rolling window shift register
    // -------------------------------------------------------------------------
    reg [15:0] window [0:WINDOW_SIZE-1];
    reg [15:0] window_sum;  // running sum of window entries
    reg  [4:0] fill_count;  // number of samples received (caps at WINDOW_SIZE)
    integer wi;

    // -------------------------------------------------------------------------
    // Mean computation
    // -------------------------------------------------------------------------
    // rolling_mean = window_sum / WINDOW_SIZE (right-shift for power-of-2 window)
    localparam SHIFT = $clog2(WINDOW_SIZE);
    wire [15:0] mean_wire = window_sum >> SHIFT;

    // -------------------------------------------------------------------------
    // Risk classification (combinational)
    // -------------------------------------------------------------------------
    wire [15:0] range       = CRITICAL_THRESH - WARN_THRESH;
    wire [15:0] above_warn  = (value >= WARN_THRESH) ? (value - WARN_THRESH) : 16'd0;

    // Threshold breach risk
    wire [2:0] thresh_risk =
        (value >= CRITICAL_THRESH) ? 3'd4 :
        (value >= WARN_THRESH) ?
            (above_warn >= (range * 7 / 10)) ? 3'd3 :
            (above_warn >= (range * 3 / 10)) ? 3'd2 :
                                               3'd1 :
        3'd0;

    // Spike detection: value > rolling_mean + SPIKE_OFFSET
    // Only enabled once we have >= 4 samples (matching Python model behaviour)
    wire spike_det = (fill_count >= 5'd4) && (value > (mean_wire + SPIKE_OFFSET));

    wire [2:0] next_risk =
        (thresh_risk >= 3'd3)   ? thresh_risk :          // threshold wins
        spike_det               ? 3'd2        :           // spike → MEDIUM
                                  thresh_risk;

    // -------------------------------------------------------------------------
    // Sequential logic
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            window_sum   <= 16'd0;
            rolling_mean <= 16'd0;
            fill_count   <= 5'd0;
            risk         <= 3'd0;
            for (wi = 0; wi < WINDOW_SIZE; wi = wi + 1)
                window[wi] <= 16'd0;
        end else if (valid) begin
            // Shift window and update sum
            window_sum <= window_sum - window[WINDOW_SIZE-1] + value;
            for (wi = WINDOW_SIZE-1; wi > 0; wi = wi - 1)
                window[wi] <= window[wi-1];
            window[0] <= value;

            // Update fill count (saturate at WINDOW_SIZE)
            if (fill_count < WINDOW_SIZE)
                fill_count <= fill_count + 5'd1;

            // Update mean and risk
            rolling_mean <= mean_wire;
            risk         <= next_risk;
        end
    end

endmodule
