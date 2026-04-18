# =============================================================================
# AstraCore Neo — Fusion Top Timing Constraints
# =============================================================================
# Out-of-context timing constraints for astracore_fusion_top.
# Target: Arty A7-35T (xc7a35ticsg324-1L).
#
# Clock:
#   clk runs at 100 MHz (10 ns period).  Matches the Arty A7 board oscillator
#   that astracore_top already uses, so a future combined system top can
#   share this constraint.
#
# I/O delays are set generously (2 ns input, 2 ns output) so the tool reports
# realistic slack without stressing the sensor byte-stream pads — these
# interfaces (SPI, UART, MIPI byte stream post-D-PHY) are all slow compared
# to the internal clock and the external deserialiser/synchroniser handles
# any tight edge requirements upstream.
# =============================================================================

# --- Primary clock ---
create_clock -name clk -period 10.000 -waveform {0.000 5.000} [get_ports clk]

# --- Generous asynchronous reset ---
set_false_path -from [get_ports rst_n]

# --- I/O timing budget (OOC) ---
# Inputs: 2 ns external delay (fast synchronous host or registered source)
set_input_delay  -clock clk -max 2.000 [all_inputs]
set_input_delay  -clock clk -min 0.500 [all_inputs]
# Remove input delay from clk and rst_n (handled separately above)
set_input_delay  -clock clk 0.000 [get_ports clk]
set_false_path   -from       [get_ports rst_n]

# Outputs: 2 ns external hold requirement
set_output_delay -clock clk -max 2.000 [all_outputs]
set_output_delay -clock clk -min 0.500 [all_outputs]
