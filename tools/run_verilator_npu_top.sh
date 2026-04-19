#!/usr/bin/env bash
# =============================================================================
# Run cocotb tests for npu_top (including the F1-C3 K-chain acceptance)
# under Verilator via WSL.
#
# Usage (from Windows Git Bash):
#   wsl -d Ubuntu-22.04 -- bash /mnt/c/Users/mande/astracore-neo/tools/run_verilator_npu_top.sh
#
# Optional env:
#   TEST_CASE=test_compiled_matmul_k_chain_16   # run a single test
# =============================================================================
set -euo pipefail

PROJECT="/mnt/c/Users/mande/astracore-neo"
cd "$PROJECT"

TEST_CASE="${TEST_CASE:-}"

# RTL sources — npu_top + all npu_* subcomponents + ecc_secded (if
# any submodule references it; harmless to include).
SOURCES=$(find rtl/npu_top rtl/npu_pe rtl/npu_sram_bank rtl/npu_sram_ctrl \
               rtl/npu_systolic_array rtl/npu_tile_ctrl rtl/npu_activation \
               rtl/npu_dma rtl/mac_array rtl/ecc_secded \
               rtl/npu_softmax rtl/npu_layernorm rtl/npu_fp \
               -name "*.v" 2>/dev/null | sort | sed "s|^|$PROJECT/|")

mkdir -p logs sim/npu_top/sim_build_verilator_chain

# F1-A4 multi-pass AFU LUTs — the npu_softmax / npu_layernorm instances
# inside npu_top load these via $readmemh. Generate fresh each build.
BUILD_DIR="$PROJECT/sim/npu_top/sim_build_verilator_chain"
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
echo "  npu_top cocotb via Verilator (F1-C3 chain acceptance)"
echo "══════════════════════════════════════════════════════════════"
echo "sources:"
echo "$SOURCES" | sed 's/^/  /'
echo ""

# cocotb runner — same pattern as run_verilator_integration.sh.
python3 - "$PROJECT" "$SOURCES" <<'PYEOF' 2>&1 | tee logs/verilator_npu_top_compiled.log
import os, pathlib, sys
project, sources_blob = sys.argv[1], sys.argv[2]
sources = [pathlib.Path(s) for s in sources_blob.strip().split('\n') if s.strip()]

sys.path.insert(0, project)
sys.path.insert(0, str(pathlib.Path(project) / "src"))

from cocotb_tools.runner import get_runner

test_dir   = pathlib.Path(project) / "sim" / "npu_top"
build_dir  = test_dir / "sim_build_verilator_chain"
runner = get_runner('verilator')

runner.build(
    sources=sources,
    hdl_toplevel='npu_top',
    build_dir=str(build_dir),
    always=True,
    parameters={
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
        # afu_luts.vh is `included by npu_activation.v and lives beside
        # it — need the include-path pointed at its directory.
        f'-I{project}/rtl/npu_activation',
        # npu_activation.v has localparams INT8_MAX / INT8_MIN that
        # collide with stdint.h macros when cocotb's --public-flat-rw
        # exposes them to the generated symbol table. Undefine the
        # macros for the generated C++ compile so the param names
        # stay as identifiers. Verilator's -CFLAGS takes one string;
        # GCC then splits it itself.
        '-CFLAGS',
        '-UINT8_MAX -UINT8_MIN -UINT16_MAX -UINT16_MIN -UINT32_MAX -UINT32_MIN',
    ],
)

test_case = os.environ.get('TEST_CASE', '').strip()
test_kwargs = dict(
    hdl_toplevel='npu_top',
    test_module='test_npu_compiled',
    test_dir=str(test_dir),
    build_dir=str(build_dir),
)
if test_case:
    test_kwargs['testcase'] = test_case
    print(f"[RUNNER] running single testcase: {test_case}")

runner.test(**test_kwargs)
PYEOF

if grep -q "FAIL=0" logs/verilator_npu_top_compiled.log 2>/dev/null; then
    echo ""
    echo "[PASS] npu_top cocotb tests"
    exit 0
else
    echo ""
    echo "[FAIL] npu_top cocotb tests — see logs/verilator_npu_top_compiled.log"
    exit 1
fi
