#!/usr/bin/env bash
# =============================================================================
# GAP-3 — build + run npu_top cocotb at N_ROWS=N_COLS=8.
#
# Parameter override is passed via cocotb's runner `parameters` dict,
# which Verilator surfaces as -G<name>=<value>. Uses a separate
# sim_build dir so the 4×4 baseline stays cached.
#
# Usage (from Windows Git Bash):
#   wsl -d Ubuntu-22.04 -- bash /mnt/c/Users/mande/astracore-neo/tools/run_verilator_npu_top_8x8.sh
# =============================================================================
set -euo pipefail

PROJECT="/mnt/c/Users/mande/astracore-neo"
cd "$PROJECT"

TEST_CASE="${TEST_CASE:-}"

SOURCES=$(find rtl/npu_top rtl/npu_pe rtl/npu_sram_bank rtl/npu_sram_ctrl \
               rtl/npu_systolic_array rtl/npu_tile_ctrl rtl/npu_activation \
               rtl/npu_dma rtl/mac_array rtl/ecc_secded \
               rtl/npu_softmax rtl/npu_layernorm rtl/npu_fp \
               -name "*.v" 2>/dev/null | sort | sed "s|^|$PROJECT/|")

mkdir -p logs sim/npu_top/sim_build_verilator_8x8

BUILD_DIR="$PROJECT/sim/npu_top/sim_build_verilator_8x8"
python3 - "$BUILD_DIR" <<'PYEOF'
import sys, pathlib
build_dir = pathlib.Path(sys.argv[1])
sys.path.insert(0, "/mnt/c/Users/mande/astracore-neo")
from tools.npu_ref.softmax_luts  import make_exp_lut, make_recip_lut
from tools.npu_ref.layernorm_luts import make_rsqrt_lut
(build_dir / "exp_lut.mem"  ).write_text("".join(f"{int(v):08x}\n" for v in make_exp_lut()))
(build_dir / "recip_lut.mem").write_text("".join(f"{int(v):08x}\n" for v in make_recip_lut()))
(build_dir / "rsqrt_lut.mem").write_text("".join(f"{int(v):04x}\n" for v in make_rsqrt_lut()))
print(f"[LUT] wrote {build_dir}/{{exp,recip,rsqrt}}_lut.mem")
PYEOF

echo "=============================================================="
echo "  npu_top cocotb via Verilator — GAP-3 8×8 array"
echo "=============================================================="

python3 - "$PROJECT" "$SOURCES" <<'PYEOF' 2>&1 | tee logs/verilator_npu_top_8x8.log
import os, pathlib, sys
project, sources_blob = sys.argv[1], sys.argv[2]
sources = [pathlib.Path(s) for s in sources_blob.strip().split('\n') if s.strip()]

sys.path.insert(0, project)
sys.path.insert(0, str(pathlib.Path(project) / "src"))

from cocotb_tools.runner import get_runner

test_dir   = pathlib.Path(project) / "sim" / "npu_top"
build_dir  = test_dir / "sim_build_verilator_8x8"
runner = get_runner('verilator')

runner.build(
    sources=sources,
    hdl_toplevel='npu_top',
    build_dir=str(build_dir),
    always=True,
    parameters={
        'N_ROWS': 8,
        'N_COLS': 8,
        # WEIGHT_DEPTH now auto-derives from N_ROWS * N_COLS in
        # npu_top.v (2026-04-19 fix after GAP-3). No longer needs to
        # be passed here. Kept as a comment for future 64×64 reviewers.
        'MP_EXP_LUT_FILE':   f'"{build_dir}/exp_lut.mem"',
        'MP_RECIP_LUT_FILE': f'"{build_dir}/recip_lut.mem"',
        'MP_RSQRT_LUT_FILE': f'"{build_dir}/rsqrt_lut.mem"',
    },
    build_args=[
        '-Wno-fatal',
        '-Wno-WIDTHEXPAND', '-Wno-WIDTHTRUNC',
        '-Wno-CASEINCOMPLETE', '-Wno-UNSIGNED',
        '-Wno-UNUSEDSIGNAL', '-Wno-UNDRIVEN', '-Wno-PINCONNECTEMPTY',
        '-Wno-SYMRSVDWORD',
        f'-I{project}/rtl/npu_activation',
        '-CFLAGS',
        '-UINT8_MAX -UINT8_MIN -UINT16_MAX -UINT16_MIN -UINT32_MAX -UINT32_MIN',
    ],
)

test_case = os.environ.get('TEST_CASE', '').strip()
test_kwargs = dict(
    hdl_toplevel='npu_top',
    test_module='test_npu_compiled_8x8',
    test_dir=str(test_dir),
    build_dir=str(build_dir),
)
if test_case:
    test_kwargs['testcase'] = test_case

runner.test(**test_kwargs)
PYEOF

if grep -q "FAIL=0" logs/verilator_npu_top_8x8.log 2>/dev/null; then
    echo "[PASS] npu_top 8x8 cocotb tests"
    exit 0
else
    echo "[FAIL] see logs/verilator_npu_top_8x8.log"
    exit 1
fi
