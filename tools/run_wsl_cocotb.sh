#!/usr/bin/env bash
# Helper to invoke a cocotb sim Makefile from WSL with the live
# system cocotb (not the .venv path that existing per-module Makefiles
# hard-code). Works around the Makefile.sim path mismatch.
#
# Usage:
#   bash tools/run_wsl_cocotb.sh <sim_subdir> [extra VERILOG_SOURCES]
#
# Examples:
#   bash tools/run_wsl_cocotb.sh sim/dms_fusion rtl/tmr_voter/tmr_voter.v
#   bash tools/run_wsl_cocotb.sh sim/tmr_voter
#   bash tools/run_wsl_cocotb.sh sim/ecc_secded

set -euo pipefail

# Ensure user-local pip bin (cocotb 1.9.2 compatible with Verilator 5.030)
# precedes the system cocotb 2.0.1 which needs Verilator 5.036+.
case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) PATH="$HOME/.local/bin:$PATH" ;;
esac
export PATH

sim_subdir="${1:?usage: run_wsl_cocotb.sh <sim_subdir> [extra_srcs...]}"
shift || true

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# cocotb 1.9.x uses cocotb-config --makefiles; cocotb 2.0+ uses cocotb_tools
if command -v cocotb-config >/dev/null 2>&1; then
    cocotb_mk=$(cocotb-config --makefiles)
else
    cocotb_mk=$(python3 -c "import cocotb_tools, os; print(os.path.dirname(cocotb_tools.__file__) + '/makefiles')")
fi

cd "$repo_root/$sim_subdir"
rm -rf sim_build results.xml

# Collect extra Verilog sources (absolute paths)
extra_srcs=""
for src in "$@"; do
    extra_srcs+=" $repo_root/$src"
done

# Discover the top Verilog source from the Makefile's VERILOG_SOURCES line.
# If we have extras, append them to the list.
default_srcs=$(awk -F'=' '/^VERILOG_SOURCES[[:space:]]*=/ {
    gsub(/\$\(abspath[[:space:]]*/, "");
    gsub(/\)/, "");
    gsub(/\.\.\/\.\.\//, "'"$repo_root"'/");
    print $2
}' Makefile)

all_srcs="${default_srcs}${extra_srcs}"

exec make SIM=verilator \
    VERILOG_SOURCES="$all_srcs" \
    COCOTB_MAKEFILES="$cocotb_mk" \
    EXTRA_ARGS="-Wno-fatal -Wno-WIDTH -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-CASEINCOMPLETE -Wno-UNSIGNED -Wno-UNUSEDSIGNAL -Wno-UNDRIVEN -Wno-PINCONNECTEMPTY -Wno-SYMRSVDWORD --no-timing"
