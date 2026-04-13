# AstraCore Neo — mac_array cocotb simulation Makefile
SIM            ?= icarus
TOPLEVEL_LANG  ?= verilog
VERILOG_SOURCES = $(abspath ../../rtl/mac_array/mac_array.v)
TOPLEVEL        = mac_array
MODULE          = test_mac_array

VENV_BIN        := $(abspath ../../.venv/Scripts)
export PATH     := $(VENV_BIN):$(PATH)
export PYTHONPATH := $(abspath ../../src):$(PYTHONPATH)

COCOTB_MAKEFILES = $(abspath ../../.venv/Lib/site-packages/cocotb_tools/makefiles)
include $(COCOTB_MAKEFILES)/Makefile.sim
