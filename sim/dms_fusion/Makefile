# =============================================================================
# AstraCore Neo — DMS Fusion cocotb simulation Makefile
# =============================================================================
# Usage:
#   make              — run all tests with Icarus Verilog
#   make clean        — remove simulation artefacts
#   make WAVES=1      — generate VCD waveform
# =============================================================================

SIM            ?= icarus
TOPLEVEL_LANG  ?= verilog

VERILOG_SOURCES  = $(abspath ../../rtl/dms_fusion/dms_fusion.v)

TOPLEVEL         = dms_fusion
MODULE           = test_dms_fusion

ifeq ($(WAVES),1)
COMPILE_ARGS    += -DVCD_DUMP
PLUSARGS        += +VCD_FILE=dms_fusion.vcd
endif

VENV_BIN         := $(abspath ../../.venv/bin)
export PATH      := $(VENV_BIN):$(PATH)
export PYTHONPATH := $(abspath ../../src):$(PYTHONPATH)

COCOTB_MAKEFILES = $(abspath ../../.venv/lib/python3.13/site-packages/cocotb_tools/makefiles)

include $(COCOTB_MAKEFILES)/Makefile.sim
