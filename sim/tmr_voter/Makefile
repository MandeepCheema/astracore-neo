# AstraCore Neo — tmr_voter cocotb simulation Makefile
SIM            ?= icarus
TOPLEVEL_LANG  ?= verilog
VERILOG_SOURCES = $(abspath ../../rtl/tmr_voter/tmr_voter.v)
TOPLEVEL        = tmr_voter
MODULE          = test_tmr_voter

VENV_BIN        := $(abspath ../../.venv/Scripts)
export PATH     := $(VENV_BIN):$(PATH)
export PYTHONPATH := $(abspath ../../src):$(PYTHONPATH)

COCOTB_MAKEFILES = $(abspath ../../.venv/Lib/site-packages/cocotb_tools/makefiles)
include $(COCOTB_MAKEFILES)/Makefile.sim
