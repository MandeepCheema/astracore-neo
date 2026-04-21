// =============================================================================
// AstraCore Neo — Fault-injection testbench wrapper for safe_state_controller.
//
// Drives a steady NORMAL-state baseline (no faults). Generates a
// 1 ms tick (every 100 cycles at 100 MHz baseline) so the DUT's
// escalation timer is exercisable.
//
// cocotb runner injects SEUs on:
//   - u_dut.safe_state (the 2-bit FSM)
//   - u_dut.timer (escalation counter)
//   - critical_faults_reg (drives the input port; simulates fault
//     arrival to validate normal escalation)
//
// Oracle: safe_state_seu_detected — currently a placeholder always-0
// signal, since today's safe_state_controller has no internal SEU
// detection. Post-F4-A-7 (TMR or Hamming on safe_state), this oracle
// will be wired to the new fault flag. The campaign runs against
// today's RTL to establish the BASELINE coverage (expected: 0 % on
// safe_state SEUs); after F4-A-7 lands, the same campaign
// re-runs and coverage should swing to ~100 %.
// =============================================================================
`timescale 1ns/1ps

module tb_safe_state_controller_fi;
    reg          clk;
    reg          rst_n;
    reg  [7:0]   critical_faults_reg;
    reg  [7:0]   warning_faults_reg;
    reg          tick_1ms;
    reg          operator_reset;

    wire [1:0]   safe_state;
    wire         alert_driver;
    wire         limit_speed;
    wire [7:0]   max_speed_kmh;
    wire         mrc_pull_over;
    wire [15:0]  latched_faults;

    safe_state_controller u_dut (
        .clk             (clk),
        .rst_n           (rst_n),
        .critical_faults (critical_faults_reg),
        .warning_faults  (warning_faults_reg),
        .tick_1ms        (tick_1ms),
        .operator_reset  (operator_reset),
        .safe_state      (safe_state),
        .alert_driver    (alert_driver),
        .limit_speed     (limit_speed),
        .max_speed_kmh   (max_speed_kmh),
        .mrc_pull_over   (mrc_pull_over),
        .latched_faults  (latched_faults)
    );

    // Pre-F4-A-7 placeholder oracle: always 0 (no internal SEU
    // detection today). After F4-A-7 lands, this becomes the
    // disagreement flag from the new TMR / Hamming check on safe_state.
    wire safe_state_seu_detected = 1'b0;

    integer tick_counter;

    // clk + rst_n are driven by the cocotb runner (Python side).
    // This initial block sets the steady-state "no faults" stimulus.
    initial begin
        critical_faults_reg = 8'h00;
        warning_faults_reg  = 8'h00;
        tick_1ms            = 0;
        operator_reset      = 0;
        tick_counter        = 0;
    end

    // 1 ms tick = every 100 cycles at 100 MHz (10 ns clock).
    // For sim speed we compress to every 10 cycles.
    always @(posedge clk) begin
        if (rst_n) begin
            tick_counter <= tick_counter + 1;
            tick_1ms     <= ((tick_counter % 10) == 9);
        end
    end

    initial begin
        $dumpfile("safe_state_controller_fi.vcd");
        $dumpvars(0, tb_safe_state_controller_fi);
    end
endmodule
