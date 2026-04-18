#!/bin/bash
# Verilator lint of astracore_fusion_top (Verilator 4.038 compatible flags).
cd /mnt/c/Users/mande/astracore-neo || exit 1
mkdir -p logs

SRCS=$(find rtl -name '*.v' ! -path '*/astracore_top/*' ! -path '*/astracore_system_top/*' | sort)
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
    --top-module astracore_fusion_top \
    $SRCS 2>&1 | tee logs/verilator_lint_fusion.log

RC=${PIPESTATUS[0]}
echo "---"
echo "Verilator exit: $RC"
echo "Error count:   $(grep -c '^%Error' logs/verilator_lint_fusion.log)"
echo "Warning count: $(grep -c '^%Warning' logs/verilator_lint_fusion.log)"
exit $RC
