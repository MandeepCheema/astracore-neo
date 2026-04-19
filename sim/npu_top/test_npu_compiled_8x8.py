"""GAP-3 — npu_top cocotb tests at N_ROWS=N_COLS=8.

Mirrors test_npu_compiled.py but at 8×8 array dimensions. Proves the
compile→RTL contract scales past 4×4 before F1-F1 stands up a 64×64
synthesis on VU9P. Shares the _execute_program harness helpers by
inline-duplicating the small parts — avoids a cross-file import
whose constants would need to differ per build.
"""

import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
from tools.npu_ref.compiler import (  # noqa: E402
    ACC_INIT_FROM_PREV_AO, Matmul,
    compile_matmul, compile_matmul_chained, emulate_matmul,
    LoadWeight, LoadActivation, RunTile, ReadAO,
)
from tools.npu_ref.conv_compiler import (  # noqa: E402
    compile_conv2d, reassemble_conv_output, reference_conv2d_int8,
)


DATA_W    = 8
ACC_W     = 32
N_ROWS    = 8
N_COLS    = 8
AI_DATA_W = N_ROWS * DATA_W
AO_DATA_W = N_COLS * ACC_W


def _mask(v, w): return v & ((1 << w) - 1)


def _to_signed(val, w):
    val &= (1 << w) - 1
    return val - (1 << w) if val & (1 << (w - 1)) else val


def _unpack_cvec(val):
    return [_to_signed((val >> (i * ACC_W)) & ((1 << ACC_W) - 1), ACC_W)
            for i in range(N_COLS)]


def _pack_cvec_signed(vec):
    out = 0
    for i, v in enumerate(vec):
        out |= (v & ((1 << ACC_W) - 1)) << (i * ACC_W)
    return out


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


async def _execute_program(dut, program, return_read_log=False):
    results = {}
    read_log = []
    last_ao_read = None

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
            seed = instr.acc_init_data
            if seed == ACC_INIT_FROM_PREV_AO:
                if last_ao_read is None:
                    raise AssertionError(
                        "Chained RunTile without prior ReadAO"
                    )
                seed = _pack_cvec_signed(last_ao_read)

            dut.start.value             = 1
            dut.cfg_k.value             = instr.k
            dut.cfg_ai_base.value       = instr.ai_base
            dut.cfg_ao_base.value       = instr.ao_base
            dut.cfg_afu_mode.value      = instr.afu_mode
            dut.cfg_acc_init_mode.value = instr.acc_init_mode
            dut.cfg_acc_init_data.value = seed & ((1 << AO_DATA_W) - 1)
            await RisingEdge(dut.clk); await Timer(1, unit="ns")
            dut.start.value = 0
            for _ in range(500):
                await RisingEdge(dut.clk); await Timer(1, unit="ns")
                if int(dut.done.value):
                    break
            else:
                raise AssertionError("RunTile did not complete")
            await RisingEdge(dut.clk); await Timer(1, unit="ns")

        elif isinstance(instr, ReadAO):
            dut.ext_ao_re.value    = 1
            dut.ext_ao_raddr.value = instr.addr
            await RisingEdge(dut.clk); await Timer(1, unit="ns")
            dut.ext_ao_re.value = 0
            await RisingEdge(dut.clk); await Timer(1, unit="ns")
            vec = _unpack_cvec(int(dut.ext_ao_rdata.value))
            results[instr.addr] = vec
            read_log.append((instr.addr, list(vec)))
            last_ao_read = vec

        else:
            raise TypeError(f"Unknown instruction: {type(instr).__name__}")

    if return_read_log:
        return results, read_log
    return results


@cocotb.test()
async def test_8x8_identity_matmul(dut):
    """Smoke: 8×8 identity matmul, cfg_k=1. Proves the new array
    dimensions plumbed through weight/activation/AO paths."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    W = [[1 if i == j else 0 for j in range(8)] for i in range(8)]
    layer = Matmul(N_rows=8, N_cols=8, cfg_k=1, weights=W, afu_mode=0)
    acts = [[i * 3 - 10 for i in range(8)]]
    program = compile_matmul(layer, ai_base=0, ao_base=0,
                              activation_vectors=acts)
    results = await _execute_program(dut, program)
    expected = emulate_matmul(layer, acts)
    assert results[0] == expected, (
        f"8×8 identity matmul rtl={results[0]} expected={expected}")
    dut._log.info(f"8×8 identity matmul PASS: {results[0]}")


@cocotb.test()
async def test_8x8_conv2d_bit_exact(dut):
    """GAP-3 core: real conv shape compiled at 8×8 and run on 8×8-
    parameterised RTL. Bit-exact vs reference."""
    import numpy as np
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)

    rng = np.random.default_rng(42)
    x = rng.integers(-30, 30, size=(1, 4, 3, 3), dtype=np.int8)
    w = rng.integers(-10, 10, size=(16, 4, 3, 3), dtype=np.int8)

    res = compile_conv2d(w, x, n_rows=N_ROWS, n_cols=N_COLS,
                          stride=(1, 1), pad=(1, 1, 1, 1))
    dut._log.info(
        f"8x8 conv2d: {len(res.program)} instrs, "
        f"K_total={res.K_total}, M={res.M}, "
        f"n_chunks={res.n_chunks}, k_chunks={res.k_chunks}"
    )
    _, read_log = await _execute_program(dut, res.program,
                                           return_read_log=True)
    out = reassemble_conv_output(read_log, res)
    ref = reference_conv2d_int8(x, w, stride=(1, 1), pad=(1, 1, 1, 1))
    if not np.array_equal(out, ref):
        dut._log.error(f"rtl shape {out.shape}, ref shape {ref.shape}")
        dut._log.error(f"rtl[0,0] = {out[0,0].tolist()}")
        dut._log.error(f"ref[0,0] = {ref[0,0].tolist()}")
        assert False, "8×8 conv RTL diverges from reference"
    dut._log.info(f"8x8 conv bit-exact PASS — output {out.shape}")
