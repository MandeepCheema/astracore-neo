## =============================================================================
## AstraCore Neo — Arty A7-35T Pin Constraints
## Target: Digilent Arty A7-35T (xc7a35ticsg324-1L)
## =============================================================================
## Board documentation: https://reference.digilentinc.com/reference/programmable-logic/arty-a7/
## =============================================================================

## ─── Clock (E3 — 100 MHz on-board oscillator) ────────────────────────────────
set_property -dict { PACKAGE_PIN E3  IOSTANDARD LVCMOS33 } [get_ports clk]
create_clock -name sys_clk -period 10.000 -waveform {0.000 5.000} [get_ports clk]

## ─── Reset (CPU_RESET push-button, active-high on board; invert to rst_n) ───
## BTN0 = C2 (leftmost push-button)
set_property -dict { PACKAGE_PIN C2  IOSTANDARD LVCMOS33 } [get_ports rst_n]

## ─── LEDs ────────────────────────────────────────────────────────────────────
## LD0–LD3 (green LEDs)
set_property -dict { PACKAGE_PIN H5  IOSTANDARD LVCMOS33 } [get_ports {led[0]}]
set_property -dict { PACKAGE_PIN J5  IOSTANDARD LVCMOS33 } [get_ports {led[1]}]
set_property -dict { PACKAGE_PIN T9  IOSTANDARD LVCMOS33 } [get_ports {led[2]}]
set_property -dict { PACKAGE_PIN T10 IOSTANDARD LVCMOS33 } [get_ports {led[3]}]

## ─── AXI4-Lite — mapped to PMOD JA for JTAG/debug access ───────────────────
## In a real integration the AXI-Lite bus is driven by a MicroBlaze or JTAG-AXI
## IP core inside Vivado BD. These pin constraints are for standalone GPIO debug
## only.  Remove this section when integrating into a block design.
##
## PMOD JA (J1 header, top row = pins 1-4, bottom row = pins 7-10)
##   JA1  = G13   JA2  = B11   JA3  = A11   JA4  = D12
##   JA7  = D13   JA8  = B18   JA9  = A18   JA10 = K16
##
## AXI4-Lite write address channel (3 pins: AWVALID, AWREADY, AWADDR[7:0])
## — not mapped to physical pins in this constraint file; driven from ILA/VIO
## inside Vivado IP Integrator.

## ─── Timing exceptions ───────────────────────────────────────────────────────
## All paths inside the design are synchronous to sys_clk.
## The AXI-Lite register accesses are not time-critical; relax to 15 ns for
## vivado placement flexibility on the A7-35T.
set_false_path -from [get_ports rst_n]

## ─── Configuration mode ──────────────────────────────────────────────────────
set_property CONFIG_VOLTAGE     3.3 [current_design]
set_property CFGBVS             VCCO [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
set_property BITSTREAM.GENERAL.COMPRESS    TRUE [current_design]
