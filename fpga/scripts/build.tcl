## =============================================================================
## AstraCore Neo — Vivado Non-Project Build Script
## Target : Arty A7-35T (xc7a35ticsg324-1L)
## Flow   : synthesis → opt → place → route → reports → bitstream
##
## Usage (from repo root):
##   vivado -mode batch -source fpga/scripts/build.tcl
##
## Outputs:
##   fpga/reports/utilization.rpt
##   fpga/reports/timing.rpt
##   fpga/reports/power.rpt
##   fpga/output/astracore_neo.bit
## =============================================================================

set PART      "xc7a35ticsg324-1L"
set TOP       "astracore_top"
set REPO_ROOT [file normalize [file join [file dirname [info script]] "../.."]]

puts "=== AstraCore Neo FPGA Build ==="
puts "Part     : $PART"
puts "Top      : $TOP"
puts "Repo root: $REPO_ROOT"

## ─── Collect all RTL sources ─────────────────────────────────────────────────
set rtl_files [glob -nocomplain $REPO_ROOT/rtl/*/*.v]
if {[llength $rtl_files] == 0} {
    error "No Verilog files found under $REPO_ROOT/rtl/"
}
puts "\nReading [llength $rtl_files] RTL files:"
foreach f $rtl_files { puts "  $f" }

read_verilog $rtl_files

## ─── Constraints ─────────────────────────────────────────────────────────────
read_xdc $REPO_ROOT/fpga/constraints/arty_a7_35t.xdc

## ─── Synthesis ───────────────────────────────────────────────────────────────
puts "\n[clock format [clock seconds] -format {%T}] — Starting synthesis..."
synth_design \
    -top  $TOP \
    -part $PART \
    -flatten_hierarchy rebuilt \
    -directive PerformanceOptimized

## Post-synthesis utilization snapshot
report_utilization -file $REPO_ROOT/fpga/reports/post_synth_utilization.rpt
report_timing_summary -max_paths 10 \
    -file $REPO_ROOT/fpga/reports/post_synth_timing.rpt

## ─── Optimisation ────────────────────────────────────────────────────────────
puts "\n[clock format [clock seconds] -format {%T}] — Optimising design..."
opt_design

## ─── Placement ───────────────────────────────────────────────────────────────
puts "\n[clock format [clock seconds] -format {%T}] — Placing design..."
place_design -directive Default
phys_opt_design

## ─── Routing ─────────────────────────────────────────────────────────────────
puts "\n[clock format [clock seconds] -format {%T}] — Routing design..."
route_design -directive Default

## ─── Final Reports ───────────────────────────────────────────────────────────
puts "\n[clock format [clock seconds] -format {%T}] — Writing reports..."

report_utilization \
    -file $REPO_ROOT/fpga/reports/utilization.rpt

report_timing_summary \
    -max_paths 20 \
    -report_unconstrained \
    -file $REPO_ROOT/fpga/reports/timing.rpt

report_power \
    -file $REPO_ROOT/fpga/reports/power.rpt

report_drc \
    -file $REPO_ROOT/fpga/reports/drc.rpt

report_clock_utilization \
    -file $REPO_ROOT/fpga/reports/clk_utilization.rpt

## ─── Bitstream ───────────────────────────────────────────────────────────────
## The AXI4-Lite ports have no physical pin constraints because they are
## intended for connection to a soft CPU (Microblaze) or Zynq PS, not to
## physical I/O. Downgrade the unconstrained-I/O DRC checks to warnings so
## the bitstream can be generated for functional validation.
set_property SEVERITY {Warning} [get_drc_checks NSTD-1]
set_property SEVERITY {Warning} [get_drc_checks UCIO-1]

puts "\n[clock format [clock seconds] -format {%T}] — Writing bitstream..."
file mkdir $REPO_ROOT/fpga/output
write_bitstream \
    -force \
    $REPO_ROOT/fpga/output/astracore_neo.bit

write_debug_probes \
    -force \
    $REPO_ROOT/fpga/output/astracore_neo.ltx

puts "\n=== Build complete: fpga/output/astracore_neo.bit ==="

## ─── Timing check ────────────────────────────────────────────────────────────
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
puts "Worst Negative Slack (setup): ${wns} ns"
puts "Worst Hold  Slack    (hold) : ${whs} ns"

if {$wns < 0} {
    puts "WARNING: Timing NOT met — WNS = ${wns} ns"
} else {
    puts "PASS: Timing met — WNS = ${wns} ns"
}
