`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — GNSS Interface  (gnss_interface.v)
// =============================================================================
// Layer 1 sensor interface.  Provides the absolute μs time base used by
// sensor_sync and ptp_clock_sync, plus cached GPS fix state (lat/lon).
//
// NMEA ASCII parsing of $GPRMC / $GPGGA is handled by an external software/
// hardware parser; this module consumes its post-parse output as a simple
// "set time" pulse carrying an absolute μs value, and independently detects
// PPS rising edges for precision time alignment.
//
// ── Design ───────────────────────────────────────────────────────────────────
//   μs counter:
//     us_sub counts system clock cycles 0..CYCLES_PER_US-1, rolling over to
//     increment time_us[63:0] by 1.  At 50 MHz with CYCLES_PER_US=50 this
//     yields exactly 1 μs resolution.
//
//   Time set:
//     When time_set_valid pulses, time_us snaps to time_set_us and the sub-μs
//     counter resets.  Used on cold-start from NMEA time and after jam-sync
//     corrections.
//
//   PPS edge detect:
//     pps_in is assumed already synchronised to clk.  A 1-cycle pulse on
//     pps_valid is emitted on its rising edge, pps_time_us latches the
//     current time_us value at the edge, and pps_count increments.
//
//   GPS fix cache:
//     fix_valid, lat_mdeg, lon_mdeg are latched on fix_set_valid for
//     downstream consumers.
//
// ── Parameters ────────────────────────────────────────────────────────────────
//   CYCLES_PER_US — clk cycles per μs tick (default 50 for 50 MHz clk)
//
// ── Interface ─────────────────────────────────────────────────────────────────
//   pps_in                           — synchronised PPS input from GPS
//   time_set_valid / time_set_us     — NMEA time load pulse (64-bit μs)
//   fix_set_valid / fix_valid_in /
//     lat_mdeg_in / lon_mdeg_in       — NMEA fix + position load pulse
//   time_us[63:0]                    — free-running absolute μs time
//   pps_valid                         — 1-cycle pulse at each PPS rising edge
//   pps_time_us[63:0]                 — time_us latched at last PPS edge
//   pps_count[15:0]                   — saturating PPS edge count
//   gps_fix_valid, lat_mdeg, lon_mdeg — latched fix output
// =============================================================================

module gnss_interface #(
    parameter [7:0] CYCLES_PER_US = 8'd50
)(
    input  wire        clk,
    input  wire        rst_n,

    // ── PPS edge input (synchronised externally) ─────────────────────────────
    input  wire        pps_in,

    // ── NMEA time load (from software/HW NMEA parser) ────────────────────────
    input  wire        time_set_valid,
    input  wire [63:0] time_set_us,

    // ── NMEA fix + position load ──────────────────────────────────────────────
    input  wire        fix_set_valid,
    input  wire        fix_valid_in,
    input  wire signed [31:0] lat_mdeg_in,
    input  wire signed [31:0] lon_mdeg_in,

    // ── Time output ───────────────────────────────────────────────────────────
    output reg  [63:0] time_us,
    output reg         pps_valid,
    output reg  [63:0] pps_time_us,
    output reg  [15:0] pps_count,

    // ── GPS fix output ────────────────────────────────────────────────────────
    output reg         gps_fix_valid,
    output reg signed [31:0] lat_mdeg,
    output reg signed [31:0] lon_mdeg
);

    // =========================================================================
    // 1. Sub-μs cycle counter → time_us increment
    // =========================================================================
    reg [7:0] us_sub;

    // =========================================================================
    // 2. PPS rising-edge detector
    // =========================================================================
    reg pps_prev;
    wire pps_rise = pps_in && !pps_prev;

    always @(posedge clk) begin
        if (!rst_n) begin
            time_us       <= 64'd0;
            us_sub        <= 8'd0;
            pps_prev      <= 1'b0;
            pps_valid     <= 1'b0;
            pps_time_us   <= 64'd0;
            pps_count     <= 16'd0;
            gps_fix_valid <= 1'b0;
            lat_mdeg      <= 32'sd0;
            lon_mdeg      <= 32'sd0;
        end else begin
            pps_prev  <= pps_in;
            pps_valid <= 1'b0;   // default de-assert

            // ── Free-running μs counter ──────────────────────────────────────
            if (time_set_valid) begin
                // Jam-sync load from NMEA time
                time_us <= time_set_us;
                us_sub  <= 8'd0;
            end else begin
                if (us_sub == CYCLES_PER_US - 8'd1) begin
                    us_sub  <= 8'd0;
                    time_us <= time_us + 64'd1;
                end else begin
                    us_sub <= us_sub + 8'd1;
                end
            end

            // ── PPS rising-edge handling ─────────────────────────────────────
            if (pps_rise) begin
                pps_valid   <= 1'b1;
                // Latch the time_us value that will be visible at the
                // NEXT clock (NBA from this edge takes one clock to appear),
                // so we capture the current count before the increment path.
                pps_time_us <= time_us;
                if (pps_count != 16'hFFFF)
                    pps_count <= pps_count + 16'd1;
            end

            // ── Fix/position load ────────────────────────────────────────────
            if (fix_set_valid) begin
                gps_fix_valid <= fix_valid_in;
                lat_mdeg      <= lat_mdeg_in;
                lon_mdeg      <= lon_mdeg_in;
            end
        end
    end

endmodule
