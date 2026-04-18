#!/bin/bash
cd /mnt/c/Users/mande/astracore-neo || exit 1
mkdir -p logs

SRCS=$(find rtl -name '*.v' | sort)
echo "Source count: $(echo "$SRCS" | wc -l)"

verilator --lint-only \
    +incdir+rtl/npu_activation \
    -Wno-fatal \
    -Wno-UNUSED \
    -Wno-UNDRIVEN \
    -Wno-PINCONNECTEMPTY \
    -Wno-WIDTH \
    -Wno-CASEINCOMPLETE \
    -Wno-UNSIGNED \
    -Wno-BLKANDNBLK \
    -Wno-CMPCONST \
    -Wno-SYMRSVDWORD \
    --top-module astracore_system_top \
    $SRCS 2>&1 | tee logs/verilator_lint_system.log

RC=${PIPESTATUS[0]}
echo "---"
echo "Verilator exit: $RC"
echo "Error count:   $(grep -c '^%Error' logs/verilator_lint_system.log)"
echo "Warning count: $(grep -c '^%Warning' logs/verilator_lint_system.log)"
exit $RC
