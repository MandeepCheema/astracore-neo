#!/usr/bin/env bash
# One-shot runner for every Track 2 cocotb validation suite.
# Runs after Verilator 5.036+ is installed at $HOME/.local/bin.
#
# Outputs one-line summary per suite: "<suite>: PASS=N FAIL=M"
# then writes full transcripts to /tmp/astracore_wsl_<suite>.log
#
# Usage:
#   bash tools/run_all_wsl_safety_tests.sh

set -u

# Ensure locally-installed Verilator + system cocotb 2.0.1 are used
case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) PATH="$HOME/.local/bin:$PATH" ;;
esac
export PATH

# Revert user-local cocotb 1.9.2 override if present — Verilator 5.036+
# supports cocotb 2.0.1 which is what the test files expect.
pip3 uninstall -y --quiet "cocotb" 2>/dev/null || true

verilator --version
python3 -c "import cocotb; print(f'cocotb {cocotb.__version__}')"

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p /tmp/astracore_wsl_logs

declare -a SUITES=(
    "sim/dms_fusion:rtl/tmr_voter/tmr_voter.v"
    "sim/tmr_voter"
    "sim/ecc_secded"
    "sim/npu_top"
)

# Note: fault-injection campaigns use a separate Makefile at
# sim/fault_injection/Makefile; run those via `make CAMPAIGN=<name>` after
# the unit suites pass.

for entry in "${SUITES[@]}"; do
    IFS=':' read -r sim_subdir extra_srcs <<< "$entry"
    name=$(basename "$sim_subdir")
    logfile=/tmp/astracore_wsl_logs/${name}.log
    echo "=== $sim_subdir ==="
    if bash "$repo/tools/run_wsl_cocotb.sh" "$sim_subdir" $extra_srcs > "$logfile" 2>&1; then
        summary=$(grep -E "TESTS=" "$logfile" | tail -1 | sed 's/.*TESTS=/TESTS=/' | cut -c1-80)
        echo "    OK $summary"
    else
        echo "    FAIL (see $logfile)"
        tail -10 "$logfile" | sed 's/^/    | /'
    fi
done

echo ""
echo "=== Fault-injection campaigns ==="
for campaign in tmr_voter_seu_1k ecc_secded_bf_10k dms_fusion_inj_5k safe_state_controller_inj_1k; do
    echo "--- $campaign ---"
    logfile=/tmp/astracore_wsl_logs/fi_${campaign}.log
    if (cd "$repo/sim/fault_injection" && make SIM=verilator CAMPAIGN=$campaign > "$logfile" 2>&1); then
        echo "    OK campaign ran; out/${campaign}.jsonl"
    else
        echo "    FAIL (see $logfile)"
        tail -10 "$logfile" | sed 's/^/    | /'
    fi
done
