# cocotb + Verilator runner for fault-injection campaigns.
#
# Usage:
#   cd sim/fault_injection
#   make CAMPAIGN=tmr_voter_seu_1k
#
# Requires WSL Ubuntu 22.04 with:
#   * Verilator 5.030 (memory/wsl_verilator_setup.md)
#   * cocotb 2.0.1
#   * pyyaml
#
# Reads sim/fault_injection/campaigns/$(CAMPAIGN).yaml, runs all
# injections, writes out/$(CAMPAIGN).jsonl. Use the planner CLI
# from any host to render the markdown report:
#
#   python -m tools.safety.fault_injection \
#       --campaign sim/fault_injection/campaigns/$(CAMPAIGN).yaml \
#       --results sim/fault_injection/out/$(CAMPAIGN).jsonl \
#       --output docs/safety/fault_injection/$(CAMPAIGN).md

# ---------------------------------------------------------------------
# Campaign + DUT selection
# ---------------------------------------------------------------------
CAMPAIGN ?= tmr_voter_seu_1k

# Per-campaign DUT mapping. Add a new line per campaign; keep tmr_voter
# the default (smallest DUT, fastest iteration).
ifeq ($(CAMPAIGN),tmr_voter_seu_1k)
TOPLEVEL := tb_tmr_voter_fi
SOURCES  := tb_tmr_voter_fi.sv \
            ../../rtl/tmr_voter/tmr_voter.v
endif

ifeq ($(CAMPAIGN),ecc_secded_bf_10k)
TOPLEVEL := tb_ecc_secded_fi
SOURCES  := tb_ecc_secded_fi.sv \
            ../../rtl/ecc_secded/ecc_secded.v
endif

ifeq ($(CAMPAIGN),dms_fusion_inj_5k)
TOPLEVEL := tb_dms_fusion_fi
SOURCES  := tb_dms_fusion_fi.sv \
            ../../rtl/dms_fusion/dms_fusion.v \
            ../../rtl/tmr_voter/tmr_voter.v
endif

ifeq ($(CAMPAIGN),safe_state_controller_inj_1k)
TOPLEVEL := tb_safe_state_controller_fi
SOURCES  := tb_safe_state_controller_fi.sv \
            ../../rtl/safe_state_controller/safe_state_controller.v
endif

# ---------------------------------------------------------------------
# Cocotb / Verilator boilerplate
# ---------------------------------------------------------------------
SIM            ?= verilator
TOPLEVEL_LANG  ?= verilog
COCOTB_HDL_TIMEUNIT      = 1ns
COCOTB_HDL_TIMEPRECISION = 1ps

VERILOG_SOURCES := $(SOURCES)
TOPLEVEL        := $(TOPLEVEL)
MODULE          := runner

EXTRA_ARGS += -Wno-fatal -Wno-WIDTH -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
              -Wno-CASEINCOMPLETE -Wno-UNSIGNED -Wno-UNUSEDSIGNAL \
              -Wno-UNDRIVEN -Wno-PINCONNECTEMPTY -Wno-SYMRSVDWORD

export ASTRACORE_FI_CAMPAIGN := $(CURDIR)/campaigns/$(CAMPAIGN).yaml
export ASTRACORE_FI_OUTDIR   := $(CURDIR)/out

# Pick up cocotb's standard rules
include $(shell cocotb-config --makefiles)/Makefile.sim
