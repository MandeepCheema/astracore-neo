#!/bin/bash
# Run cocotb integration tests for astracore_fusion_top via Verilator 4.038.
cd /mnt/c/Users/mande/astracore-neo || exit 1
mkdir -p logs sim/astracore_fusion_top/sim_build_verilator

PROJECT=/mnt/c/Users/mande/astracore-neo
SRC_LIST=$(find rtl -name '*.v' ! -path '*/astracore_top/*' ! -path '*/astracore_system_top/*' | sort | sed "s|^|$PROJECT/|" | tr '\n' ' ')

python3 <<PYEOF 2>&1 | tee logs/verilator_fusion_top.log
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('/mnt/c/Users/mande/astracore-neo/src').resolve()))
from cocotb_tools.runner import get_runner

sources_str = """$SRC_LIST""".strip()
sources = [pathlib.Path(s) for s in sources_str.split() if s]
print(f"Source count: {len(sources)}")

runner = get_runner('verilator')
runner.build(
    sources=sources,
    hdl_toplevel='astracore_fusion_top',
    build_dir='/mnt/c/Users/mande/astracore-neo/sim/astracore_fusion_top/sim_build_verilator',
    always=True,
    build_args=[
        '-Wno-fatal',
        '-Wno-UNUSEDSIGNAL', '-Wno-UNDRIVEN', '-Wno-PINCONNECTEMPTY',
        '-Wno-WIDTHEXPAND', '-Wno-WIDTHTRUNC',
        '-Wno-CASEINCOMPLETE', '-Wno-UNSIGNED',
        '-Wno-BLKANDNBLK', '-Wno-CMPCONST', '-Wno-SYMRSVDWORD',
        '--trace',
    ],
)
runner.test(
    hdl_toplevel='astracore_fusion_top',
    test_module='test_astracore_fusion_top',
    test_dir='/mnt/c/Users/mande/astracore-neo/sim/astracore_fusion_top',
    build_dir='/mnt/c/Users/mande/astracore-neo/sim/astracore_fusion_top/sim_build_verilator',
)
PYEOF

RC=${PIPESTATUS[0]}
echo "---"
echo "Python exit: $RC"
exit $RC
