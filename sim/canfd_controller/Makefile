# AstraCore Neo — canfd_controller cocotb simulation Makefile
SIM            ?= icarus
TOPLEVEL_LANG  ?= verilog
VERILOG_SOURCES = $(abspath ../../rtl/canfd_controller/canfd_controller.v)
TOPLEVEL        = canfd_controller
MODULE          = test_canfd_controller

VENV_BIN        := $(abspath ../../.venv/Scripts)
export PATH     := $(VENV_BIN):$(PATH)
export PYTHONPATH := $(abspath ../../src):$(PYTHONPATH)

COCOTB_MAKEFILES = $(abspath ../../.venv/Lib/site-packages/cocotb_tools/makefiles)
include $(COCOTB_MAKEFILES)/Makefile.sim
