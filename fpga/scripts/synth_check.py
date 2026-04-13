"""
AstraCore Neo — Pre-synthesis lint check (iverilog -tnull)
Verifies that all Verilog files including the top-level integration
pass elaboration without errors before running Vivado.

Usage:
    python fpga/scripts/synth_check.py
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RTL_DIR = REPO / "rtl"
IVERILOG = r"C:\iverilog\bin\iverilog.exe"

def main():
    # Collect all .v files in RTL order: submodules first, top last
    submodule_dirs = [
        "gaze_tracker", "thermal_zone", "canfd_controller", "ecc_secded",
        "tmr_voter", "fault_predictor", "head_pose_tracker", "pcie_controller",
        "ethernet_controller", "mac_array", "inference_runtime",
    ]
    top_dir = ["astracore_top"]

    vfiles = []
    for d in submodule_dirs + top_dir:
        p = RTL_DIR / d
        found = list(p.glob("*.v"))
        if not found:
            print(f"ERROR: no .v files in {p}")
            sys.exit(1)
        vfiles.extend(found)

    cmd = [IVERILOG, "-tnull", "-Wall", "-g2012"] + [str(f) for f in vfiles]
    print(f"Running: {' '.join(cmd[:4])} ... ({len(vfiles)} files)")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode == 0:
        print(f"\nPASS — {len(vfiles)} files elaborate cleanly.")
        sys.exit(0)
    else:
        print(f"\nFAIL — iverilog returned exit code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
