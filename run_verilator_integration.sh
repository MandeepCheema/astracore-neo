#!/usr/bin/env bash
# =============================================================================
# AstraCore Neo — Verilator Integration Test Runner
# Runs full-chip integration tests using Verilator (faster than icarus)
# Usage: wsl -d Ubuntu-22.04 -- bash /mnt/c/Users/mande/astracore-neo/run_verilator_integration.sh
# =============================================================================
set -euo pipefail

PROJECT="/mnt/c/Users/mande/astracore-neo"
cd "$PROJECT"

# Collect all RTL source files
SOURCES=$(find rtl -name "*.v" | sort | sed "s|^|$PROJECT/|")

echo "══════════════════════════════════════════════════════════════"
echo "  AstraCore Neo — Verilator Integration Tests"
echo "══════════════════════════════════════════════════════════════"
echo ""

# -----------------------------------------------------------------
# Test 1: Verilator lint check (catches issues before simulation)
# -----------------------------------------------------------------
echo "[LINT] Running Verilator lint on astracore_system_top..."
if verilator --lint-only --Wall \
    -Wno-UNUSEDSIGNAL -Wno-UNDRIVEN -Wno-PINCONNECTEMPTY -Wno-WIDTHEXPAND \
    -Wno-WIDTHTRUNC -Wno-CASEINCOMPLETE -Wno-UNSIGNED \
    --top-module astracore_system_top \
    $SOURCES 2>&1 | tee logs/verilator_lint.log; then
    echo "[LINT PASS]"
else
    echo "[LINT FAIL] — see logs/verilator_lint.log"
    echo "Continuing to simulation anyway (lint warnings may be non-fatal)..."
fi
echo ""

# -----------------------------------------------------------------
# Test 2: Verilator lint on fusion_top alone
# -----------------------------------------------------------------
echo "[LINT] Running Verilator lint on astracore_fusion_top..."
FUSION_SOURCES=$(find rtl -name "*.v" ! -path "*/astracore_top/*" ! -path "*/astracore_system_top/*" | sort | sed "s|^|$PROJECT/|")
if verilator --lint-only --Wall \
    -Wno-UNUSEDSIGNAL -Wno-UNDRIVEN -Wno-PINCONNECTEMPTY -Wno-WIDTHEXPAND \
    -Wno-WIDTHTRUNC -Wno-CASEINCOMPLETE -Wno-UNSIGNED \
    --top-module astracore_fusion_top \
    $FUSION_SOURCES 2>&1 | tee logs/verilator_lint_fusion.log; then
    echo "[LINT PASS]"
else
    echo "[LINT FAIL]"
fi
echo ""

# -----------------------------------------------------------------
# Test 3: cocotb integration tests via Verilator
# -----------------------------------------------------------------
run_cocotb_verilator() {
    local TOP="$1"
    local TEST_DIR="$2"
    local TEST_MODULE="$3"
    local SRCS="$4"

    echo "[SIM] Running cocotb tests for $TOP via Verilator..."

    mkdir -p "$TEST_DIR/sim_build_verilator"

    python3 -c "
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('src').resolve()))
from cocotb_tools.runner import get_runner

sources = '''$SRCS'''.strip().split('\n')
runner = get_runner('verilator')
runner.build(
    sources=[pathlib.Path(s) for s in sources],
    hdl_toplevel='$TOP',
    build_dir='$TEST_DIR/sim_build_verilator',
    always=True,
    build_args=['--trace', '-Wno-fatal', '-Wno-WIDTHEXPAND', '-Wno-WIDTHTRUNC',
                '-Wno-CASEINCOMPLETE', '-Wno-UNSIGNED', '-Wno-UNUSEDSIGNAL',
                '-Wno-UNDRIVEN', '-Wno-PINCONNECTEMPTY'],
)
runner.test(
    hdl_toplevel='$TOP',
    test_module='$TEST_MODULE',
    test_dir=str(pathlib.Path('$TEST_DIR').resolve()),
    build_dir='$TEST_DIR/sim_build_verilator',
)
" 2>&1 | tee "logs/verilator_${TOP}.log"

    if grep -q "FAIL=0" "logs/verilator_${TOP}.log" 2>/dev/null; then
        echo "[SIM PASS] $TOP"
        return 0
    else
        echo "[SIM FAIL] $TOP — see logs/verilator_${TOP}.log"
        return 1
    fi
}

mkdir -p logs

# Run fusion_top integration tests
FUSION_SRC_LIST=$(find rtl -name "*.v" ! -path "*/astracore_top/*" ! -path "*/astracore_system_top/*" | sort | sed "s|^|$PROJECT/|")

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Fusion Top — 5 integration smoke tests"
echo "══════════════════════════════════════════════════════════════"
run_cocotb_verilator \
    "astracore_fusion_top" \
    "sim/astracore_fusion_top" \
    "test_astracore_fusion_top" \
    "$FUSION_SRC_LIST" || true

# Run system_top integration tests
ALL_SRC_LIST=$(find rtl -name "*.v" | sort | sed "s|^|$PROJECT/|")

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  System Top — 2 integration smoke tests"
echo "══════════════════════════════════════════════════════════════"
run_cocotb_verilator \
    "astracore_system_top" \
    "sim/astracore_system_top" \
    "test_astracore_system_top" \
    "$ALL_SRC_LIST" || true

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Verilator integration tests complete"
echo "  Check logs/verilator_*.log for details"
echo "══════════════════════════════════════════════════════════════"
