#!/usr/bin/env bash
# =============================================================================
# Run cocotb tests for npu_layernorm under Verilator via WSL.  (F1-A4)
#
# Usage (from Windows Git Bash):
#   wsl -d Ubuntu-22.04 -- bash -c 'bash /mnt/c/Users/mande/astracore-neo/tools/run_verilator_npu_layernorm.sh'
# =============================================================================
set -euo pipefail

PROJECT="/mnt/c/Users/mande/astracore-neo"
cd "$PROJECT"

TEST_CASE="${TEST_CASE:-}"

SOURCES="$PROJECT/rtl/npu_layernorm/npu_layernorm.v"

BUILD_DIR="$PROJECT/sim/npu_layernorm/sim_build_verilator"
mkdir -p logs "$BUILD_DIR"

python3 - "$BUILD_DIR" <<'PYEOF'
import sys, pathlib
build_dir = pathlib.Path(sys.argv[1])
sys.path.insert(0, "/mnt/c/Users/mande/astracore-neo")
from tools.npu_ref.layernorm_luts import make_rsqrt_lut
arr = make_rsqrt_lut()
(build_dir / "rsqrt_lut.mem").write_text(
    "".join(f"{int(v):04x}\n" for v in arr))
print(f"[LUT] wrote {build_dir}/rsqrt_lut.mem")
PYEOF

echo "══════════════════════════════════════════════════════════════"
echo "  npu_layernorm cocotb via Verilator (F1-A4)"
echo "══════════════════════════════════════════════════════════════"

python3 - "$PROJECT" "$SOURCES" "$BUILD_DIR" <<'PYEOF' 2>&1 | tee logs/verilator_npu_layernorm.log
import os, pathlib, sys
project, sources_blob, build_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sources = [pathlib.Path(s) for s in sources_blob.strip().split('\n') if s.strip()]

sys.path.insert(0, project)
sys.path.insert(0, str(pathlib.Path(project) / "src"))

from cocotb_tools.runner import get_runner

test_dir  = pathlib.Path(project) / "sim" / "npu_layernorm"
runner    = get_runner('verilator')

rsqrt_lut_abs = f'{build_dir}/rsqrt_lut.mem'

runner.build(
    sources=sources,
    hdl_toplevel='npu_layernorm',
    build_dir=build_dir,
    always=True,
    parameters={
        'RSQRT_LUT_FILE': f'"{rsqrt_lut_abs}"',
    },
    build_args=[
        '-Wno-fatal',
        '-Wno-WIDTHEXPAND', '-Wno-WIDTHTRUNC',
        '-Wno-CASEINCOMPLETE', '-Wno-UNSIGNED',
        '-Wno-UNUSEDSIGNAL', '-Wno-UNDRIVEN', '-Wno-PINCONNECTEMPTY',
        '-Wno-SYMRSVDWORD',
    ],
)

test_case = os.environ.get('TEST_CASE', '').strip()
test_kwargs = dict(
    hdl_toplevel='npu_layernorm',
    test_module='test_npu_layernorm',
    test_dir=str(test_dir),
    build_dir=build_dir,
)
if test_case:
    test_kwargs['testcase'] = test_case

runner.test(**test_kwargs)
PYEOF

if grep -q "FAIL=0" logs/verilator_npu_layernorm.log 2>/dev/null; then
    echo ""
    echo "[PASS] npu_layernorm cocotb tests"
    exit 0
else
    echo ""
    echo "[FAIL] npu_layernorm cocotb tests — see logs/verilator_npu_layernorm.log"
    exit 1
fi
