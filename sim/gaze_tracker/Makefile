# =============================================================================
# AstraCore Neo — GazeTracker cocotb simulation Makefile
# =============================================================================
# Usage:
#   make              — run all tests with Icarus Verilog
#   make clean        — remove simulation artefacts
#   make WAVES=1      — generate VCD waveform (sim_build/gaze_tracker.vcd)
# =============================================================================

# Simulator backend
SIM            ?= icarus
TOPLEVEL_LANG  ?= verilog

# Verilog source(s) — path relative to this Makefile
VERILOG_SOURCES  = $(abspath ../../rtl/gaze_tracker/gaze_tracker.v)

# Top-level module name (must match the module in the .v file)
TOPLEVEL         = gaze_tracker

# cocotb test module (the Python file, without .py)
MODULE           = test_gaze_tracker

# Optional: generate VCD waveform when WAVES=1
ifeq ($(WAVES),1)
COMPILE_ARGS    += -DVCD_DUMP
PLUSARGS        += +VCD_FILE=gaze_tracker.vcd
endif

# Prepend the venv bin so cocotb-config resolves to the venv install
VENV_BIN         := $(abspath ../../.venv/bin)
export PATH      := $(VENV_BIN):$(PATH)
export PYTHONPATH := $(abspath ../../src):$(PYTHONPATH)

# cocotb makefiles location (cocotb 2.x ships them via cocotb_tools)
COCOTB_MAKEFILES = $(abspath ../../.venv/lib/python3.13/site-packages/cocotb_tools/makefiles)

include $(COCOTB_MAKEFILES)/Makefile.sim
