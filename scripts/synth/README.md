# Fusion Pipeline Lint + Synthesis Scripts

Tool-ready scripts for pushing `astracore_fusion_top` through two common
signoff flows.  Neither tool is installed on the development machine as of
this commit, so the scripts are provided for future use.

## Yosys lint sweep

Detects inferred latches, combinational loops, multi-driver nets, and
unsupported-for-synth constructs that Icarus Verilog does not flag.

```bash
# Install Yosys on Windows:
#   Option A) MSYS2:      pacman -S mingw-w64-x86_64-yosys
#   Option B) Prebuilt:   https://github.com/YosysHQ/oss-cad-suite-build/releases
#
# From repo root:
yosys -s scripts/synth/yosys_lint.ys
```

Expected clean run: the `check -assert` stage prints a summary and exits 0.
If violations appear they are printed with source file + line number.

## Vivado FPGA synthesis

Out-of-context synthesis for the Arty A7-35T part; produces utilization +
timing reports without needing top-level pad constraints.

```bash
# Install Vivado (Xilinx ML Standard edition is free for this part):
#   https://www.xilinx.com/support/download.html
#
# From repo root:
vivado -mode batch -source scripts/synth/vivado_synth.tcl
```

Reports land in `build/vivado_fusion/`:

- `fusion_top_synth.dcp` — post-synth design checkpoint
- `utilization.rpt`      — LUT / FF / BRAM / DSP usage
- `timing_summary.rpt`   — worst negative slack per clock domain
- `drc.rpt`              — design rule check violations

## Source list

Single source-of-truth file list:

```
scripts/synth/fusion_sources.f
```

Both of the above scripts and any future tool integration should consume
this list rather than redeclare it, so there is one place to add a new
RTL file.
