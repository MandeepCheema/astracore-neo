"""WP-3 end-to-end compiler → NPU integration test.

Compiles a simple Matmul tile using tools/npu_ref/compiler.py, executes
the resulting instruction stream via the npu_top external interface, and
verifies the hardware output bit-exactly matches the compiler's software
emulation. This is the first full compile-then-run proof of the compiler
contract for single-tile matmul.
"""

import os
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
from tools.npu_ref.compiler import (  # noqa: E402
    Matmul, compile_matmul, emulate_matmul,
    LoadWeight, LoadActivation, RunTile, ReadAO,
)


DATA_W    = 8
ACC_W     = 32
N_ROWS    = 4
N_COLS    = 4
AI_DATA_W = N_ROWS * DATA_W
AO_DATA_W = N_COLS * ACC_W


def _mask(v, w): return v & ((1 << w) - 1)


def _to_signed(val, w):
    val &= (1 << w) - 1
    return val - (1 << w) if val & (1 << (w - 1)) else val


def _unpack_cvec(val):
    return [_to_signed((val >> (i * ACC_W)) & ((1 << ACC_W) - 1), ACC_W)
            for i in range(N_COLS)]


async def _reset(dut):
    dut.rst_n.value = 0
    for s in ("start","cfg_k","cfg_ai_base","cfg_ao_base","cfg_afu_mode",
              "cfg_acc_init_mode","cfg_acc_init_data","cfg_precision_mode",
              "ext_w_we","ext_w_waddr","ext_w_wdata",
              "ext_ai_we","ext_ai_waddr","ext_ai_wdata",
              "ext_ao_re","ext_ao_raddr","ext_sparse_skip_vec",
              "dma_start","dma_cfg_src_addr","dma_cfg_ai_base",
              "dma_cfg_tile_h","dma_cfg_src_stride","mem_rdata"):
        getattr(dut, s).value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def _execute_program(dut, program):
    """Drive the compiled instruction stream through the npu_top ports.

    Returns a dict mapping ReadAO address → unpacked column vector so
    the test can compare against the compiler emulation.
    """
    results = {}

    for instr in program:
        if isinstance(instr, LoadWeight):
            dut.ext_w_we.value    = 1
            dut.ext_w_waddr.value = instr.addr
            dut.ext_w_wdata.value = _mask(instr.data, DATA_W)
            await RisingEdge(dut.clk); await Timer(1, unit="ns")
            dut.ext_w_we.value = 0

        elif isinstance(instr, LoadActivation):
            dut.ext_ai_we.value    = 1
            dut.ext_ai_waddr.value = instr.addr
            dut.ext_ai_wdata.value = instr.packed & ((1 << AI_DATA_W) - 1)
            await RisingEdge(dut.clk); await Timer(1, unit="ns")
            dut.ext_ai_we.value = 0

        elif isinstance(instr, RunTile):
            dut.start.value             = 1
            dut.cfg_k.value             = instr.k
            dut.cfg_ai_base.value       = instr.ai_base
            dut.cfg_ao_base.value       = instr.ao_base
            dut.cfg_afu_mode.value      = instr.afu_mode
            dut.cfg_acc_init_mode.value = instr.acc_init_mode
            dut.cfg_acc_init_data.value = instr.acc_init_data & ((1 << AO_DATA_W) - 1)
            await RisingEdge(dut.clk); await Timer(1, unit="ns")
            dut.start.value = 0
            for _ in range(500):
                await RisingEdge(dut.clk); await Timer(1, unit="ns")
                if int(dut.done.value):
                    break
            else:
                raise AssertionError("RunTile did not complete in 500 cycles")
            await RisingEdge(dut.clk); await Timer(1, unit="ns")

        elif isinstance(instr, ReadAO):
            dut.ext_ao_re.value    = 1
            dut.ext_ao_raddr.value = instr.addr
            await RisingEdge(dut.clk); await Timer(1, unit="ns")
            dut.ext_ao_re.value = 0
            await RisingEdge(dut.clk); await Timer(1, unit="ns")
            results[instr.addr] = _unpack_cvec(int(dut.ext_ao_rdata.value))

        else:
            raise TypeError(f"Unknown instruction type: {type(instr).__name__}")

    return results


@cocotb.test()
async def test_compiled_identity_matmul(dut):
    """Compile a 4×4 identity matmul and run it end-to-end on npu_top."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)

    W = [[1 if i == j else 0 for j in range(4)] for i in range(4)]
    layer = Matmul(N_rows=4, N_cols=4, cfg_k=1, weights=W, afu_mode=0)
    acts = [[7, -3, 11, -5]]

    program = compile_matmul(layer, ai_base=0, ao_base=0,
                              activation_vectors=acts)
    dut._log.info(f"compiled {len(program)} instructions")

    results = await _execute_program(dut, program)
    expected = emulate_matmul(layer, acts)

    assert 0 in results, f"no AO read captured"
    rtl = results[0]
    assert rtl == expected, (
        f"compiled identity matmul mismatch: rtl={rtl} expected={expected}")
    dut._log.info(f"compiled identity matmul PASS: {rtl}")


@cocotb.test()
async def test_compiled_nontrivial_matmul_cfg_k3(dut):
    """Compile a cfg_k=3 accumulated matmul and run end-to-end."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)

    W = [[1, 0, 2, 0],
         [0, 1, 0, 2],
         [1, 1, 1, 1],
         [-1, -1, -1, -1]]
    layer = Matmul(N_rows=4, N_cols=4, cfg_k=3, weights=W, afu_mode=0)
    acts = [[1, 2, 3, 4], [-1, 0, 1, 2], [2, -1, 3, 1]]

    program = compile_matmul(layer, ai_base=0, ao_base=0,
                              activation_vectors=acts)
    results = await _execute_program(dut, program)
    expected = emulate_matmul(layer, acts)

    rtl = results[0]
    assert rtl == expected, (
        f"compiled cfg_k=3 matmul mismatch: rtl={rtl} expected={expected}")
    dut._log.info(f"compiled cfg_k=3 matmul PASS: {rtl}")


@cocotb.test()
async def test_compiled_matmul_with_relu(dut):
    """Compile a matmul with afu_mode=RELU and verify activation at writeback."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)

    W = [[1, -2, 3, -4],
         [2, -3, 4, -5],
         [3, -4, 5, -6],
         [4, -5, 6, -7]]
    layer = Matmul(N_rows=4, N_cols=4, cfg_k=1, weights=W, afu_mode=1)  # RELU
    acts = [[1, 1, 1, 1]]

    program = compile_matmul(layer, ai_base=0, ao_base=0,
                              activation_vectors=acts)
    results = await _execute_program(dut, program)
    expected = emulate_matmul(layer, acts)

    rtl = results[0]
    assert rtl == expected, (
        f"compiled RELU matmul mismatch: rtl={rtl} expected={expected}")
    # Sanity: raw would have negatives, RELU zeros them
    assert any(v == 0 for v in rtl), f"RELU should produce at least one 0: {rtl}"
    dut._log.info(f"compiled matmul+RELU PASS: {rtl}")
