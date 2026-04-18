# =============================================================================
# AstraCore Neo — Vivado Out-of-Context Synthesis (fusion pipeline)
# =============================================================================
# Usage (from repo root, in Vivado Tcl shell or batch):
#   vivado -mode batch -source scripts/synth/vivado_synth.tcl
#
# Target part: Arty A7-35T (xc7a35ticsg324-1L) — matches the existing
# astracore_top.v FPGA bring-up design.  For a different part, edit -part.
#
# Output (in ./build/vivado_fusion/):
#   • synth_design checkpoint    (fusion_top_synth.dcp)
#   • Utilization report         (utilization.rpt)
#   • Timing summary report      (timing_summary.rpt)
#   • DRC report                 (drc.rpt)
# =============================================================================

set PART   xc7a35ticsg324-1L
set TOP    astracore_fusion_top
set OUTDIR build/vivado_fusion

file mkdir $OUTDIR

# Read every RTL file in the fusion source list
set src_files {
    rtl/astracore_fusion_top/astracore_fusion_top.v
    rtl/det_arbiter/det_arbiter.v
    rtl/mipi_csi2_rx/mipi_csi2_rx.v
    rtl/imu_interface/imu_interface.v
    rtl/gnss_interface/gnss_interface.v
    rtl/ptp_clock_sync/ptp_clock_sync.v
    rtl/canfd_controller/canfd_controller.v
    rtl/can_odometry_decoder/can_odometry_decoder.v
    rtl/radar_interface/radar_interface.v
    rtl/ultrasonic_interface/ultrasonic_interface.v
    rtl/ethernet_controller/ethernet_controller.v
    rtl/lidar_interface/lidar_interface.v
    rtl/cam_detection_receiver/cam_detection_receiver.v
    rtl/sensor_sync/sensor_sync.v
    rtl/coord_transform/coord_transform.v
    rtl/ego_motion_estimator/ego_motion_estimator.v
    rtl/object_tracker/object_tracker.v
    rtl/lane_fusion/lane_fusion.v
    rtl/plausibility_checker/plausibility_checker.v
    rtl/ttc_calculator/ttc_calculator.v
    rtl/aeb_controller/aeb_controller.v
    rtl/ldw_lka_controller/ldw_lka_controller.v
    rtl/safe_state_controller/safe_state_controller.v
}
foreach f $src_files {
    read_verilog $f
}

# Read the XDC timing constraints (100 MHz clock, I/O budgets)
read_xdc constraints/astracore_fusion_top.xdc

# Out-of-context synthesis
synth_design -top $TOP -part $PART -mode out_of_context

# Write checkpoint + reports
write_checkpoint -force $OUTDIR/fusion_top_synth.dcp
report_utilization -file $OUTDIR/utilization.rpt
report_timing_summary -file $OUTDIR/timing_summary.rpt
report_drc -file $OUTDIR/drc.rpt

puts "Vivado OOC synthesis complete.  Reports in $OUTDIR/"
