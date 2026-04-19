#!/usr/bin/env bash
# =============================================================================
# End-to-end softmax via npu_top multi-pass path (F1-A4 integration test).
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

mkdir -p logs sim/npu_top/sim_build_verilator_mp

BUILD_DIR="$PROJECT/sim/npu_top/sim_build_verilator_mp"

# Generate LUT .mem files
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

echo "══════════════════════════════════════════════════════════════"
echo "  npu_top softmax e2e cocotb via Verilator (F1-A4 integration)"
echo "══════════════════════════════════════════════════════════════"

python3 - "$PROJECT" "$SOURCES" "$BUILD_DIR" <<'PYEOF' 2>&1 | tee logs/verilator_npu_top_softmax.log
import os, pathlib, sys
project, sources_blob, build_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sources = [pathlib.Path(s) for s in sources_blob.strip().split('\n') if s.strip()]

sys.path.insert(0, project)
sys.path.insert(0, str(pathlib.Path(project) / "src"))

from cocotb_tools.runner import get_runner

test_dir  = pathlib.Path(project) / "sim" / "npu_top"
runner    = get_runner('verilator')

runner.build(
    sources=sources,
    hdl_toplevel='npu_top',
    build_dir=build_dir,
    always=True,
    parameters={
        'MP_EXP_LUT_FILE':   f'"{build_dir}/exp_lut.mem"',
        'MP_RECIP_LUT_FILE': f'"{build_dir}/recip_lut.mem"',
        'MP_RSQRT_LUT_FILE': f'"{build_dir}/rsqrt_lut.mem"',
        # Softmax VEC_LEN=64 requires ≥64 AI entries; LN VEC_LEN=256 would
        # need 256. For this test, 128 covers softmax comfortably.
        'ACT_IN_DEPTH':  '128',
        'ACT_OUT_DEPTH': '128',
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
kwargs = dict(
    hdl_toplevel='npu_top',
    test_module='test_npu_top_softmax',
    test_dir=str(test_dir),
    build_dir=build_dir,
)
if test_case:
    kwargs['testcase'] = test_case

runner.test(**kwargs)
PYEOF

if grep -q "FAIL=0" logs/verilator_npu_top_softmax.log 2>/dev/null; then
    echo "[PASS] npu_top softmax e2e"
    exit 0
else
    echo "[FAIL] npu_top softmax e2e"
    exit 1
fi
