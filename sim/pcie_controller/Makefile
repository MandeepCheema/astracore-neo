# AstraCore Neo — pcie_controller cocotb simulation Makefile
SIM            ?= icarus
TOPLEVEL_LANG  ?= verilog
VERILOG_SOURCES = $(abspath ../../rtl/pcie_controller/pcie_controller.v)
TOPLEVEL        = pcie_controller
MODULE          = test_pcie_controller

VENV_BIN        := $(abspath ../../.venv/Scripts)
export PATH     := $(VENV_BIN):$(PATH)
export PYTHONPATH := $(abspath ../../src):$(PYTHONPATH)

COCOTB_MAKEFILES = $(abspath ../../.venv/Lib/site-packages/cocotb_tools/makefiles)
include $(COCOTB_MAKEFILES)/Makefile.sim
