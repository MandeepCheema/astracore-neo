# =============================================================================
# AstraCore Neo — Synopsys Design Constraints
# Target: sky130A  |  Clock: 50 MHz (20 ns period)
# OpenSTA units: time=ns, capacitance=pF, resistance=kΩ
# =============================================================================

# Primary clock
create_clock -name clk -period 20.000 [get_ports clk]

# Clock uncertainty (setup/hold margin for sky130 process variation)
set_clock_uncertainty -setup 0.5 [get_clocks clk]
set_clock_uncertainty -hold  0.2 [get_clocks clk]

# Input / output delays relative to clk (30% of period)
# Apply to all inputs then override non-data ports below
set_input_delay  -clock clk -max 6.0 [all_inputs]
set_input_delay  -clock clk -min 0.0 [all_inputs]
set_output_delay -clock clk -max 6.0 [all_outputs]
set_output_delay -clock clk -min 0.0 [all_outputs]

# Reset is asynchronous — cut all timing paths through it
set_false_path -from [get_ports rst_n]

# Drive strength for top-level inputs (assume 4× drive from pad ring)
set_driving_cell -lib_cell sky130_fd_sc_hd__buf_4 [all_inputs]

# Output load: 0.05 pF — realistic for sky130 block-level signoff
# (sky130 gate input cap ~17 fF; 0.05 pF = ~3 gate fanout)
set_load 0.05 [all_outputs]

# Max transition / max capacitance for sky130_fd_sc_hd
set_max_transition 1.5 [current_design]
set_max_capacitance 0.5 [current_design]
