#!/usr/bin/env bash
# =============================================================================
# AstraCore Neo — OpenLane 2 run script (Nix install)
# Usage:
#   ./asic/scripts/run_openlane.sh                        # full chip (astracore_top)
#   ./asic/scripts/run_openlane.sh --module gaze_tracker  # single module
#   ./asic/scripts/run_openlane.sh --batch                # all module configs
#   ./asic/scripts/run_openlane.sh --batch --jobs 4       # parallel batch (4 at a time)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASIC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$ASIC_DIR/.." && pwd)"

# ── Source Nix profile so openlane is on PATH ─────────────────────────────────
if [[ -f "$HOME/.nix-profile/etc/profile.d/nix.sh" ]]; then
    source "$HOME/.nix-profile/etc/profile.d/nix.sh"
fi

command -v openlane >/dev/null 2>&1 || { echo "ERROR: openlane not found. Is Nix installed?"; exit 1; }

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[run_openlane] $*"; }
die() { echo "[run_openlane] ERROR: $*" >&2; exit 1; }

# ── Parse args ────────────────────────────────────────────────────────────────
MODULE=""
BATCH=0
JOBS=1
ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    arg="${ARGS[$i]}"
    case "$arg" in
        --module)     ((i++)); MODULE="${ARGS[$i]}" ;;
        --module=*)   MODULE="${arg#--module=}" ;;
        --batch)      BATCH=1 ;;
        --jobs)       ((i++)); JOBS="${ARGS[$i]}" ;;
        --jobs=*)     JOBS="${arg#--jobs=}" ;;
        --help)
            echo "Usage: $0 [--module <name>] [--batch [--jobs N]]"
            echo "  (default)                    Run full astracore_top design"
            echo "  --module gaze_tracker        Run single module"
            echo "  --batch                      Run all config_*.yaml modules"
            echo "  --batch --jobs 4             Run batch with N parallel jobs"
            exit 0 ;;
        *) die "Unknown argument: $arg" ;;
    esac
    ((i++))
done

# ── Batch mode ────────────────────────────────────────────────────────────────
if [[ "$BATCH" -eq 1 ]]; then
    CONFIGS=("$ASIC_DIR"/config_*.yaml)
    [[ ${#CONFIGS[@]} -gt 0 ]] || die "No config_*.yaml files found in $ASIC_DIR"

    mkdir -p "$ASIC_DIR/reports"
    log "Batch run: ${#CONFIGS[@]} module configs (jobs=$JOBS)"
    log "OpenLane: $(openlane --version 2>&1 | head -1)"

    PASS=0
    FAIL=0
    FAILED_MODULES=""

    run_one() {
        local cfg="$1"
        local mod
        mod=$(basename "$cfg" .yaml)
        mod="${mod#config_}"
        log "START  $mod"
        if openlane --design-dir "$ASIC_DIR" --run-tag "$mod" "$cfg" > "$ASIC_DIR/reports/${mod}_run.log" 2>&1; then
            log "PASS   $mod"
            return 0
        else
            log "FAIL   $mod (see reports/${mod}_run.log)"
            return 1
        fi
    }
    export -f run_one log
    export ASIC_DIR

    if [[ "$JOBS" -gt 1 ]] && command -v parallel >/dev/null 2>&1; then
        parallel --jobs "$JOBS" --halt soon,fail=50% run_one ::: "${CONFIGS[@]}" || true
    else
        for cfg in "${CONFIGS[@]}"; do
            if run_one "$cfg"; then
                ((PASS++))
            else
                ((FAIL++))
                mod=$(basename "$cfg" .yaml); mod="${mod#config_}"
                FAILED_MODULES="$FAILED_MODULES $mod"
            fi
        done
    fi

    log "═══════════════════════════════════════════"
    log "Batch complete: $PASS passed, $FAIL failed out of ${#CONFIGS[@]}"
    [[ -n "$FAILED_MODULES" ]] && log "Failed:$FAILED_MODULES"
    log "Reports in: $ASIC_DIR/reports/"
    exit $FAIL
fi

# ── Select config ─────────────────────────────────────────────────────────────
if [[ -n "$MODULE" ]]; then
    CONFIG="$ASIC_DIR/config_${MODULE}.yaml"
    TAG="${MODULE}"
    [[ -f "$CONFIG" ]] || die "No config found for module '$MODULE' at $CONFIG"
    log "Module run: $MODULE"
else
    CONFIG="$ASIC_DIR/config.json"
    TAG="astracore_top"
    log "Full chip run: astracore_top (~2-4 hours)"
fi

OUTPUT_DIR="$ASIC_DIR/output"
mkdir -p "$OUTPUT_DIR" "$ASIC_DIR/reports"

log "OpenLane: $(openlane --version 2>&1 | head -1)"
log "Config:   $CONFIG"
log "Output:   $OUTPUT_DIR"

# ── Run ───────────────────────────────────────────────────────────────────────
openlane \
    --design-dir "$ASIC_DIR" \
    --run-tag "$TAG" \
    "$CONFIG"

# ── Copy key reports ──────────────────────────────────────────────────────────
log "Run complete. Copying reports..."
LATEST=$(find "$OUTPUT_DIR" -maxdepth 2 -name "*.log" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)
RUN_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "*${TAG}*" 2>/dev/null | sort | tail -1)
if [[ -n "$RUN_DIR" ]]; then
    find "$RUN_DIR" -name "metrics.csv"      -exec cp -f {} "$ASIC_DIR/reports/" \; 2>/dev/null || true
    find "$RUN_DIR" -name "*utilization*"    -exec cp -f {} "$ASIC_DIR/reports/" \; 2>/dev/null || true
    find "$RUN_DIR" -name "*timing*"         -exec cp -f {} "$ASIC_DIR/reports/" \; 2>/dev/null || true
    log "Reports copied to $ASIC_DIR/reports/"
fi

log "Done. GDSII at: $OUTPUT_DIR"
