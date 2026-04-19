#!/usr/bin/env bash
set -uo pipefail

export PDK_ROOT=/home/mandeep/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af
export PDK=sky130A

ASIC_DIR="/mnt/c/Users/mande/astracore-neo/asic"
cd "$ASIC_DIR"
mkdir -p reports

PASS=0
FAIL=0
FAILED=""
TOTAL=0

for cfg in config_*.yaml; do
    mod="$(basename "$cfg" .yaml)"
    mod="${mod#config_}"

    if [ "$mod" = "astracore_fusion_top" ]; then
        echo "[SKIP] astracore_fusion_top (integration-level, run separately)"
        continue
    fi

    TOTAL=$((TOTAL + 1))
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "[START] $mod ($TOTAL/32)"
    echo "══════════════════════════════════════════════════════════════"

    if openlane --design-dir . --run-tag "$mod" "$cfg" > "reports/${mod}_run.log" 2>&1; then
        drc=$(grep -c "DRC.*Passed" "reports/${mod}_run.log" 2>/dev/null || echo 0)
        lvs=$(grep -c "LVS.*Passed" "reports/${mod}_run.log" 2>/dev/null || echo 0)
        echo "[PASS]  $mod  (DRC=$drc LVS=$lvs)"
        PASS=$((PASS + 1))
    else
        echo "[FAIL]  $mod  (see reports/${mod}_run.log)"
        FAIL=$((FAIL + 1))
        FAILED="$FAILED $mod"
    fi
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "BATCH COMPLETE: $PASS passed, $FAIL failed out of $TOTAL"
[ -n "$FAILED" ] && echo "FAILED:$FAILED"
echo "══════════════════════════════════════════════════════════════"
