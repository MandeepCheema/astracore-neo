`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — Sensor Sync  (sensor_sync.v)
// =============================================================================
// Layer 2 fusion module. Aligns detections from 4 sensor channels by timestamp,
// gates them into a fusion window, and watchdogs silent sensors.
//
// ── Design ───────────────────────────────────────────────────────────────────
//   Each of 4 sensor channels provides a 1-cycle valid pulse with a μs
//   timestamp (GPS/PTP synchronized from gnss_interface / ptp_clock_sync).
//
//   Fusion window lifecycle:
//     IDLE → first sensor_valid fires → OPEN (anchor = that sensor's timestamp)
//     OPEN → accumulate in-window sensors → sensors_ready bits accumulate
//     CLOSE on: (a) sensors_ready == 4'hF (all 4 aligned)
//               (b) WINDOW_CYCLES internal timeout expires
//     window_release pulses 1 cycle on close.
//
//   In-window check: |s_time_us − window_center| ≤ WINDOW_US
//
//   Stale watchdog: per-sensor cycle counter; resets on sensor_valid.
//   If counter reaches STALE_CYCLES → sensor_stale bit asserts.
//   Stale = sensor silent for too long (hw fault / disconnected sensor).
//
// ── Parameters (simulation-friendly defaults, adjust for silicon) ────────────
//   WINDOW_US        — half-width tolerance in μs (default 50 → ±50 μs window)
//   WINDOW_CYCLES    — window open timeout in clock cycles (default 100 for sim;
//                      production: 5000 = 100μs @ 50MHz)
//   STALE_CYCLES     — legacy alias; applied to every sensor that does not
//                      override via STALE_CYCLES_S{0..3}.  Default 200 (sim).
//   STALE_CYCLES_S0  — per-sensor stale threshold for channel 0 (camera).
//   STALE_CYCLES_S1  — per-sensor stale threshold for channel 1 (radar).
//   STALE_CYCLES_S2  — per-sensor stale threshold for channel 2 (LiDAR).
//   STALE_CYCLES_S3  — per-sensor stale threshold for channel 3 (ultrasonic).
//     Production sensors run at very different rates (cam ~60 Hz, radar ~20 Hz,
//     lidar ~10 Hz, ultra ~40 Hz).  A single shared threshold forces either
//     premature lidar-stale false-positives or delayed camera-outage detection;
//     four per-channel thresholds let each rate be set independently.
//
// ── Interface ─────────────────────────────────────────────────────────────────
//   sensor_valid[3:0]      — 1-cycle detection pulse per sensor
//   s0_time_us..s3_time_us — detection timestamps in μs
//   window_open            — 1 while fusion window is active
//   window_center          — anchor timestamp (μs) of current window
//   sensors_ready[3:0]     — bitmask of in-window sensors so far
//   window_release         — 1-cycle pulse when window closes
//   sensor_stale[3:0]      — bitmask: sensor silent past its per-channel
//                            STALE_CYCLES_S{i} threshold
// =============================================================================

module sensor_sync #(
    parameter [31:0] WINDOW_US       = 32'd50,   // half-width in μs (50 → ±50 μs window)
    parameter [23:0] WINDOW_CYCLES   = 24'd100,  // window timeout clock cycles
    parameter [23:0] STALE_CYCLES    = 24'd200,  // legacy shared default
    parameter [23:0] STALE_CYCLES_S0 = STALE_CYCLES,
    parameter [23:0] STALE_CYCLES_S1 = STALE_CYCLES,
    parameter [23:0] STALE_CYCLES_S2 = STALE_CYCLES,
    parameter [23:0] STALE_CYCLES_S3 = STALE_CYCLES
)(
    input  wire        clk,
    input  wire        rst_n,

    // ── Per-sensor inputs (4 channels) ───────────────────────────────────────
    input  wire [3:0]  sensor_valid,   // [i]=1 for 1 cycle when sensor i has detection
    input  wire [31:0] s0_time_us,     // sensor 0 detection timestamp (μs)
    input  wire [31:0] s1_time_us,     // sensor 1 detection timestamp (μs)
    input  wire [31:0] s2_time_us,     // sensor 2 detection timestamp (μs)
    input  wire [31:0] s3_time_us,     // sensor 3 detection timestamp (μs)

    // ── Fusion window outputs ─────────────────────────────────────────────────
    output reg         window_open,    // 1 while fusion window is active
    output reg  [31:0] window_center,  // anchor timestamp of current window
    output reg  [3:0]  sensors_ready,  // bitmask of aligned sensors
    output reg         window_release, // 1-cycle pulse: window closed
    output reg  [3:0]  sensor_stale    // bitmask: sensor exceeded STALE_CYCLES
);

    // =========================================================================
    // 1. Per-sensor stale watchdog counters (generate block, 4 instances)
    //    Each channel compares against its own STALE_CYCLES_S{i} threshold so
    //    sensors at different nominal rates can be watchdogged independently.
    // =========================================================================
    reg  [23:0] stale_cnt    [0:3];
    wire [23:0] stale_thresh [0:3];
    assign stale_thresh[0] = STALE_CYCLES_S0;
    assign stale_thresh[1] = STALE_CYCLES_S1;
    assign stale_thresh[2] = STALE_CYCLES_S2;
    assign stale_thresh[3] = STALE_CYCLES_S3;

    genvar gi;
    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : STALE_WDT
            always @(posedge clk) begin
                if (!rst_n) begin
                    stale_cnt[gi]    <= 24'd0;
                    sensor_stale[gi] <= 1'b0;
                end else if (sensor_valid[gi]) begin
                    stale_cnt[gi]    <= 24'd0;
                    sensor_stale[gi] <= 1'b0;
                end else begin
                    if (stale_cnt[gi] < stale_thresh[gi])
                        stale_cnt[gi] <= stale_cnt[gi] + 24'd1;
                    else
                        sensor_stale[gi] <= 1'b1;
                end
            end
        end
    endgenerate

    // =========================================================================
    // 2. Fusion window FSM
    // =========================================================================
    reg [23:0] window_timer;   // cycles since window opened

    // Signed timestamp deltas relative to current window_center
    wire signed [31:0] delta0 = $signed(s0_time_us) - $signed(window_center);
    wire signed [31:0] delta1 = $signed(s1_time_us) - $signed(window_center);
    wire signed [31:0] delta2 = $signed(s2_time_us) - $signed(window_center);
    wire signed [31:0] delta3 = $signed(s3_time_us) - $signed(window_center);

    // In-window: |delta| ≤ WINDOW_US
    wire [3:0] in_window;
    assign in_window[0] = (delta0 >= -$signed({1'b0, WINDOW_US})) && (delta0 <= $signed({1'b0, WINDOW_US}));
    assign in_window[1] = (delta1 >= -$signed({1'b0, WINDOW_US})) && (delta1 <= $signed({1'b0, WINDOW_US}));
    assign in_window[2] = (delta2 >= -$signed({1'b0, WINDOW_US})) && (delta2 <= $signed({1'b0, WINDOW_US}));
    assign in_window[3] = (delta3 >= -$signed({1'b0, WINDOW_US})) && (delta3 <= $signed({1'b0, WINDOW_US}));

    // Close conditions (combinatorial, using pre-clock values)
    wire window_timeout = window_open && (window_timer >= WINDOW_CYCLES);
    wire all_ready      = window_open && (sensors_ready == 4'hF);
    wire do_close       = window_timeout || all_ready;

    always @(posedge clk) begin
        if (!rst_n) begin
            window_open    <= 1'b0;
            window_center  <= 32'd0;
            sensors_ready  <= 4'h0;
            window_release <= 1'b0;
            window_timer   <= 24'd0;
        end else begin
            window_release <= 1'b0;   // default de-assert

            if (!window_open) begin
                // ── IDLE: open window on first arriving detection ─────────────
                if (sensor_valid != 4'h0) begin
                    window_open  <= 1'b1;
                    window_timer <= 24'd0;
                    // Anchor = first active sensor (priority 0 > 1 > 2 > 3)
                    if      (sensor_valid[0]) window_center <= s0_time_us;
                    else if (sensor_valid[1]) window_center <= s1_time_us;
                    else if (sensor_valid[2]) window_center <= s2_time_us;
                    else                      window_center <= s3_time_us;
                    // All simultaneously-firing sensors are automatically in-window
                    sensors_ready <= sensor_valid;
                end
            end else begin
                // ── OPEN: accumulate in-window sensors ───────────────────────
                window_timer <= window_timer + 24'd1;

                if (sensor_valid[0] && in_window[0]) sensors_ready[0] <= 1'b1;
                if (sensor_valid[1] && in_window[1]) sensors_ready[1] <= 1'b1;
                if (sensor_valid[2] && in_window[2]) sensors_ready[2] <= 1'b1;
                if (sensor_valid[3] && in_window[3]) sensors_ready[3] <= 1'b1;

                // ── CLOSE on all-ready or timeout ────────────────────────────
                if (do_close) begin
                    window_open    <= 1'b0;
                    sensors_ready  <= 4'h0;
                    window_release <= 1'b1;
                    window_timer   <= 24'd0;
                end
            end
        end
    end

    // =========================================================================
    // Sensor-sync safety invariants — Phase-F formal verification targets
    // =========================================================================
`ifndef SYNTHESIS
`ifndef __ICARUS__
    // Invariant: window_release is a 1-cycle pulse; never held for 2+ cycles.
    property p_window_release_pulse;
        @(posedge clk) disable iff (!rst_n)
        window_release |=> !window_release;
    endproperty
    a_window_release_pulse: assert property (p_window_release_pulse)
        else $error("sensor_sync: window_release held for >1 cycle");
`endif
`endif

endmodule
