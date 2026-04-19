#!/usr/bin/env bash
# =============================================================================
# Run cocotb tests for the F1-A4 multi-pass extension of npu_tile_ctrl.
# =============================================================================
set -euo pipefail

PROJECT="/mnt/c/Users/mande/astracore-neo"
cd "$PROJECT"

TEST_CASE="${TEST_CASE:-}"
SOURCES="$PROJECT/rtl/npu_tile_ctrl/npu_tile_ctrl.v"
BUILD_DIR="$PROJECT/sim/npu_tile_ctrl_mp/sim_build_verilator"

mkdir -p logs "$BUILD_DIR"

echo "══════════════════════════════════════════════════════════════"
echo "  npu_tile_ctrl (multi-pass) cocotb via Verilator (F1-A4)"
echo "══════════════════════════════════════════════════════════════"

python3 - "$PROJECT" "$SOURCES" "$BUILD_DIR" <<'PYEOF' 2>&1 | tee logs/verilator_npu_tile_ctrl_mp.log
import os, pathlib, sys
project, sources_blob, build_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sources = [pathlib.Path(s) for s in sources_blob.strip().split('\n') if s.strip()]

sys.path.insert(0, project)
from cocotb_tools.runner import get_runner

test_dir  = pathlib.Path(project) / "sim" / "npu_tile_ctrl_mp"
runner    = get_runner('verilator')
runner.build(
    sources=sources,
    hdl_toplevel='npu_tile_ctrl',
    build_dir=build_dir,
    always=True,
    build_args=[
        '-Wno-fatal',
        '-Wno-WIDTHEXPAND', '-Wno-WIDTHTRUNC',
        '-Wno-CASEINCOMPLETE', '-Wno-UNSIGNED',
        '-Wno-UNUSEDSIGNAL', '-Wno-UNDRIVEN', '-Wno-PINCONNECTEMPTY',
        '-Wno-SYMRSVDWORD',
    ],
)

test_case = os.environ.get('TEST_CASE', '').strip()
kwargs = dict(
    hdl_toplevel='npu_tile_ctrl',
    test_module='test_npu_tile_ctrl_mp',
    test_dir=str(test_dir),
    build_dir=build_dir,
)
if test_case:
    kwargs['testcase'] = test_case
runner.test(**kwargs)
PYEOF

if grep -q "FAIL=0" logs/verilator_npu_tile_ctrl_mp.log 2>/dev/null; then
    echo "[PASS] npu_tile_ctrl (multi-pass) cocotb tests"
    exit 0
else
    echo "[FAIL] npu_tile_ctrl (multi-pass) cocotb tests"
    exit 1
fi
