#!/bin/bash
# =============================================================================
# AstraCore Neo — full regression runner
# =============================================================================
# Covers all 42 test-bearing modules. Some modules need extra RTL sources
# at elaboration time (NPU sub-hierarchy, fusion tops). Those are listed in
# the per-module source-set tables below.
#
# Modules with no direct regression:
#   - npu_sram_bank (tested via npu_sram_ctrl)
#   - astracore_top (legacy wrapper — top-level AXI test is test_astracore_top.py
#                    added as of this audit pass)
# =============================================================================

set -e
export PATH="$PATH:/c/iverilog/bin"

VENV_PYTHON="$(pwd)/.venv/Scripts/python.exe"

mkdir -p logs

# -----------------------------------------------------------------------------
# Single-file leaf modules: one .v source = the DUT.
# -----------------------------------------------------------------------------
LEAF_MODULES=(
    # Legacy base chip modules (11 − 0 = 11; gaze_tracker is in the list now)
    gaze_tracker thermal_zone canfd_controller ecc_secded tmr_voter
    fault_predictor head_pose_tracker pcie_controller ethernet_controller
    mac_array inference_runtime

    # Sensor fusion Layer 1 — sensor interfaces (10)
    dms_fusion mipi_csi2_rx imu_interface gnss_interface ptp_clock_sync
    can_odometry_decoder radar_interface ultrasonic_interface
    cam_detection_receiver lidar_interface

    # Fusion Layer 2 — processing (6)
    sensor_sync coord_transform ego_motion_estimator object_tracker
    lane_fusion plausibility_checker

    # Fusion Layer 3 — decision (4)
    ttc_calculator aeb_controller ldw_lka_controller safe_state_controller

    # Miscellaneous
    det_arbiter

    # NPU leaf modules (the ones that can elaborate standalone)
    npu_pe npu_systolic_array npu_activation npu_dma npu_tile_ctrl
)

# -----------------------------------------------------------------------------
# Modules with multi-file dependencies. Each entry: "<dut>|<space-separated-src>"
# -----------------------------------------------------------------------------
MULTI_SOURCE=(
    "npu_sram_ctrl|rtl/npu_sram_bank/npu_sram_bank.v rtl/npu_sram_ctrl/npu_sram_ctrl.v"
    "npu_tile_harness|rtl/npu_pe/npu_pe.v rtl/npu_systolic_array/npu_systolic_array.v rtl/npu_sram_bank/npu_sram_bank.v rtl/npu_sram_ctrl/npu_sram_ctrl.v rtl/npu_tile_harness/npu_tile_harness.v"
    "npu_top|rtl/npu_pe/npu_pe.v rtl/npu_systolic_array/npu_systolic_array.v rtl/npu_sram_bank/npu_sram_bank.v rtl/npu_sram_ctrl/npu_sram_ctrl.v rtl/npu_activation/npu_activation.v rtl/npu_tile_ctrl/npu_tile_ctrl.v rtl/npu_top/npu_top.v"
    "astracore_top|rtl/gaze_tracker/gaze_tracker.v rtl/thermal_zone/thermal_zone.v rtl/canfd_controller/canfd_controller.v rtl/ecc_secded/ecc_secded.v rtl/tmr_voter/tmr_voter.v rtl/fault_predictor/fault_predictor.v rtl/head_pose_tracker/head_pose_tracker.v rtl/pcie_controller/pcie_controller.v rtl/ethernet_controller/ethernet_controller.v rtl/mac_array/mac_array.v rtl/inference_runtime/inference_runtime.v rtl/astracore_top/astracore_top.v"
)

# -----------------------------------------------------------------------------
# Integration tops (need everything except the other top + NPU-specific files)
# -----------------------------------------------------------------------------
# Fusion top: all non-NPU, non-astracore_top, non-astracore_system_top modules
FUSION_TOP_EXCLUDES="astracore_top astracore_system_top npu_pe npu_systolic_array npu_sram_bank npu_sram_ctrl npu_dma npu_activation npu_tile_ctrl npu_tile_harness npu_top"

# -----------------------------------------------------------------------------
run_leaf() {
    local mod="$1"
    mkdir -p "sim/$mod/sim_build"
    "$VENV_PYTHON" - "$mod" <<'PYEOF'
import sys, pathlib
mod = sys.argv[1]
sys.path.insert(0, str(pathlib.Path('src').resolve()))
from cocotb_tools.runner import get_runner
runner = get_runner('icarus')
runner.build(sources=[pathlib.Path(f'rtl/{mod}/{mod}.v').resolve()],
             hdl_toplevel=mod, build_dir=f'sim/{mod}/sim_build', always=True)
runner.test(hdl_toplevel=mod, test_module=f'test_{mod}',
            test_dir=str(pathlib.Path(f'sim/{mod}').resolve()),
            build_dir=f'sim/{mod}/sim_build')
PYEOF
}

run_multi() {
    local mod="$1"
    local srcs="$2"
    mkdir -p "sim/$mod/sim_build"
    "$VENV_PYTHON" - "$mod" "$srcs" <<'PYEOF'
import sys, pathlib
mod = sys.argv[1]
src_paths = sys.argv[2].split()
sys.path.insert(0, str(pathlib.Path('src').resolve()))
from cocotb_tools.runner import get_runner
sources = [pathlib.Path(s).resolve() for s in src_paths]
runner = get_runner('icarus')
runner.build(sources=sources, hdl_toplevel=mod,
             build_dir=f'sim/{mod}/sim_build', always=True)
runner.test(hdl_toplevel=mod, test_module=f'test_{mod}',
            test_dir=str(pathlib.Path(f'sim/{mod}').resolve()),
            build_dir=f'sim/{mod}/sim_build')
PYEOF
}

run_fusion_top() {
    mkdir -p "sim/astracore_fusion_top/sim_build"
    "$VENV_PYTHON" - "$FUSION_TOP_EXCLUDES" <<'PYEOF'
import sys, pathlib
excludes = set(sys.argv[1].split())
sys.path.insert(0, str(pathlib.Path('src').resolve()))
from cocotb_tools.runner import get_runner
rtl = pathlib.Path('rtl').resolve()
sources = []
for sub in sorted(rtl.iterdir()):
    if sub.is_dir() and sub.name not in excludes:
        v = sub / f'{sub.name}.v'
        if v.exists():
            sources.append(v)
runner = get_runner('icarus')
runner.build(sources=sources, hdl_toplevel='astracore_fusion_top',
             build_dir='sim/astracore_fusion_top/sim_build', always=True)
runner.test(hdl_toplevel='astracore_fusion_top',
            test_module='test_astracore_fusion_top',
            test_dir=str(pathlib.Path('sim/astracore_fusion_top').resolve()),
            build_dir='sim/astracore_fusion_top/sim_build')
PYEOF
}

run_system_top() {
    mkdir -p "sim/astracore_system_top/sim_build"
    "$VENV_PYTHON" - <<'PYEOF'
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('src').resolve()))
from cocotb_tools.runner import get_runner
rtl = pathlib.Path('rtl').resolve()
# Everything except NPU modules (system_top = base + fusion only today)
excludes = {'npu_pe','npu_systolic_array','npu_sram_bank','npu_sram_ctrl',
            'npu_dma','npu_activation','npu_tile_ctrl','npu_tile_harness',
            'npu_top'}
sources = []
for sub in sorted(rtl.iterdir()):
    if sub.is_dir() and sub.name not in excludes:
        v = sub / f'{sub.name}.v'
        if v.exists():
            sources.append(v)
runner = get_runner('icarus')
runner.build(sources=sources, hdl_toplevel='astracore_system_top',
             build_dir='sim/astracore_system_top/sim_build', always=True)
runner.test(hdl_toplevel='astracore_system_top',
            test_module='test_astracore_system_top',
            test_dir=str(pathlib.Path('sim/astracore_system_top').resolve()),
            build_dir='sim/astracore_system_top/sim_build')
PYEOF
}

# -----------------------------------------------------------------------------
# Run everything
# -----------------------------------------------------------------------------
PASS=0
FAIL=0
SUMMARY=""

record() {
    local mod="$1"
    local log="logs/rtl_${mod}.log"
    local result
    result=$(grep -E "TESTS=[0-9]+ PASS=[0-9]+" "$log" 2>/dev/null | tail -1 || true)
    if [[ -z "$result" ]]; then
        echo "  [!!!] $mod — no summary line found (build error?)"
        FAIL=$((FAIL+1))
        SUMMARY="${SUMMARY}BUILD_FAIL  $mod\n"
    else
        local fails
        fails=$(echo "$result" | grep -oE "FAIL=[0-9]+" | head -1 | cut -d= -f2)
        if [[ "$fails" == "0" ]]; then
            PASS=$((PASS+1))
            SUMMARY="${SUMMARY}PASS        $mod  $result\n"
        else
            FAIL=$((FAIL+1))
            SUMMARY="${SUMMARY}FAIL        $mod  $result\n"
        fi
    fi
}

echo "=============================================================="
echo "  LEAF MODULES (${#LEAF_MODULES[@]} modules)"
echo "=============================================================="
for mod in "${LEAF_MODULES[@]}"; do
    echo "--- $mod ---"
    run_leaf "$mod" 2>&1 | tee "logs/rtl_${mod}.log" | grep -E "TESTS=" || true
    record "$mod"
done

echo ""
echo "=============================================================="
echo "  MULTI-SOURCE MODULES (${#MULTI_SOURCE[@]} entries)"
echo "=============================================================="
for entry in "${MULTI_SOURCE[@]}"; do
    mod="${entry%%|*}"
    srcs="${entry#*|}"
    echo "--- $mod ---"
    run_multi "$mod" "$srcs" 2>&1 | tee "logs/rtl_${mod}.log" | grep -E "TESTS=" || true
    record "$mod"
done

echo ""
echo "=============================================================="
echo "  INTEGRATION TOPS"
echo "=============================================================="
echo "--- astracore_fusion_top ---"
run_fusion_top 2>&1 | tee "logs/rtl_astracore_fusion_top.log" | grep -E "TESTS=" || true
record "astracore_fusion_top"

echo "--- astracore_system_top ---"
run_system_top 2>&1 | tee "logs/rtl_astracore_system_top.log" | grep -E "TESTS=" || true
record "astracore_system_top"

echo ""
echo "=============================================================="
echo "  SUMMARY"
echo "=============================================================="
printf "$SUMMARY"
echo ""
echo "TOTAL PASS: $PASS   FAIL: $FAIL"
[[ "$FAIL" -eq 0 ]]
