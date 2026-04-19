#!/usr/bin/env bash
# =============================================================================
# Run cocotb tests for npu_fp_pe under Verilator via WSL.  (F1-A1)
# =============================================================================
set -euo pipefail

PROJECT="/mnt/c/Users/mande/astracore-neo"
cd "$PROJECT"

TEST_CASE="${TEST_CASE:-}"
SOURCES=$(printf '%s\n' \
    "$PROJECT/rtl/npu_fp/npu_fp_mac.v" \
    "$PROJECT/rtl/npu_fp/npu_fp_pe.v")
BUILD_DIR="$PROJECT/sim/npu_fp_pe/sim_build_verilator"

mkdir -p logs "$BUILD_DIR"

echo "══════════════════════════════════════════════════════════════"
echo "  npu_fp_pe cocotb via Verilator (F1-A1)"
echo "══════════════════════════════════════════════════════════════"

python3 - "$PROJECT" "$SOURCES" "$BUILD_DIR" <<'PYEOF' 2>&1 | tee logs/verilator_npu_fp_pe.log
import os, pathlib, sys
project, sources_blob, build_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sources = [pathlib.Path(s) for s in sources_blob.strip().split('\n') if s.strip()]

sys.path.insert(0, project)
from cocotb_tools.runner import get_runner

test_dir  = pathlib.Path(project) / "sim" / "npu_fp_pe"
runner    = get_runner('verilator')
runner.build(
    sources=sources,
    hdl_toplevel='npu_fp_pe',
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
    hdl_toplevel='npu_fp_pe',
    test_module='test_npu_fp_pe',
    test_dir=str(test_dir),
    build_dir=build_dir,
)
if test_case:
    kwargs['testcase'] = test_case
runner.test(**kwargs)
PYEOF

if grep -q "FAIL=0" logs/verilator_npu_fp_pe.log 2>/dev/null; then
    echo "[PASS] npu_fp_pe cocotb tests"
    exit 0
else
    echo "[FAIL] npu_fp_pe cocotb tests"
    exit 1
fi
