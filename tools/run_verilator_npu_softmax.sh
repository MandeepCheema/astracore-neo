#!/usr/bin/env bash
# =============================================================================
# Run cocotb tests for npu_softmax under Verilator via WSL.  (F1-A4)
#
# Usage (from Windows Git Bash):
#   wsl -d Ubuntu-22.04 -- bash /mnt/c/Users/mande/astracore-neo/tools/run_verilator_npu_softmax.sh
#
# Optional env:
#   TEST_CASE=test_uniform_input   # run a single test
# =============================================================================
set -euo pipefail

PROJECT="/mnt/c/Users/mande/astracore-neo"
cd "$PROJECT"

TEST_CASE="${TEST_CASE:-}"

SOURCES="$PROJECT/rtl/npu_softmax/npu_softmax.v"

BUILD_DIR="$PROJECT/sim/npu_softmax/sim_build_verilator"
mkdir -p logs "$BUILD_DIR"

# Generate LUT .mem files into the build directory.  Verilator's $readmemh
# resolves paths relative to the CWD of the simulator binary, which the
# cocotb runner launches from `build_dir`.
python3 - "$BUILD_DIR" <<'PYEOF'
import sys, pathlib, numpy as np
build_dir = pathlib.Path(sys.argv[1])

sys.path.insert(0, "/mnt/c/Users/mande/astracore-neo")
from tools.npu_ref.softmax_luts import make_exp_lut, make_recip_lut

def dump(path, arr):
    lines = [f"{int(v):08x}\n" for v in arr]
    path.write_text("".join(lines))

dump(build_dir / "exp_lut.mem",   make_exp_lut())
dump(build_dir / "recip_lut.mem", make_recip_lut())
print(f"[LUT] wrote {build_dir}/exp_lut.mem and recip_lut.mem")
PYEOF

echo "══════════════════════════════════════════════════════════════"
echo "  npu_softmax cocotb via Verilator (F1-A4)"
echo "══════════════════════════════════════════════════════════════"

python3 - "$PROJECT" "$SOURCES" "$BUILD_DIR" <<'PYEOF' 2>&1 | tee logs/verilator_npu_softmax.log
import os, pathlib, sys
project, sources_blob, build_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sources = [pathlib.Path(s) for s in sources_blob.strip().split('\n') if s.strip()]

sys.path.insert(0, project)
sys.path.insert(0, str(pathlib.Path(project) / "src"))

from cocotb_tools.runner import get_runner

test_dir  = pathlib.Path(project) / "sim" / "npu_softmax"
runner    = get_runner('verilator')

exp_lut_abs   = f'{build_dir}/exp_lut.mem'
recip_lut_abs = f'{build_dir}/recip_lut.mem'

runner.build(
    sources=sources,
    hdl_toplevel='npu_softmax',
    build_dir=build_dir,
    always=True,
    parameters={
        'EXP_LUT_FILE':   f'"{exp_lut_abs}"',
        'RECIP_LUT_FILE': f'"{recip_lut_abs}"',
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
    hdl_toplevel='npu_softmax',
    test_module='test_npu_softmax',
    test_dir=str(test_dir),
    build_dir=build_dir,
)
if test_case:
    test_kwargs['testcase'] = test_case
    print(f"[RUNNER] running single testcase: {test_case}")

runner.test(**test_kwargs)
PYEOF

if grep -q "FAIL=0" logs/verilator_npu_softmax.log 2>/dev/null; then
    echo ""
    echo "[PASS] npu_softmax cocotb tests"
    exit 0
else
    echo ""
    echo "[FAIL] npu_softmax cocotb tests — see logs/verilator_npu_softmax.log"
    exit 1
fi
