# AstraCore Neo — thermal_zone cocotb simulation Makefile
SIM            ?= icarus
TOPLEVEL_LANG  ?= verilog
VERILOG_SOURCES = $(abspath ../../rtl/thermal_zone/thermal_zone.v)
TOPLEVEL        = thermal_zone
MODULE          = test_thermal_zone

VENV_BIN        := $(abspath ../../.venv/Scripts)
export PATH     := $(VENV_BIN):$(PATH)
export PYTHONPATH := $(abspath ../../src):$(PYTHONPATH)

COCOTB_MAKEFILES = $(abspath ../../.venv/Lib/site-packages/cocotb_tools/makefiles)
include $(COCOTB_MAKEFILES)/Makefile.sim
