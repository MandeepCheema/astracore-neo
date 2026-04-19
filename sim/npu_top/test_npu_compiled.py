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
    ACC_INIT_FROM_PREV_AO, Matmul,
    compile_matmul, compile_matmul_chained, emulate_matmul,
    LoadWeight, LoadActivation, RunTile, ReadAO,
)
from tools.npu_ref.conv_compiler import (  # noqa: E402
    compile_conv2d, reference_conv2d_int8, reassemble_conv_output,
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


def _pack_cvec_signed(vec):
    """Pack N_COLS signed int32 accumulator values into a single wide
    integer (little-endian column order) to feed cfg_acc_init_data.
    Mirrors the RTL's bit layout."""
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
    """Drive the compiled instruction stream through the npu_top ports.

    Default return: a dict mapping ReadAO address → unpacked column
    vector. When return_read_log=True, returns
    `(results_dict, read_log)` where read_log is a chronological list
    of `(addr, vec)` tuples — one per ReadAO executed. F1-C4 conv
    compilation needs the log because M-loops reuse AO slots.

    Handles the F1-C3 chained-tile sentinel: a RunTile with
    `acc_init_data == ACC_INIT_FROM_PREV_AO` is seeded from the most
    recent ReadAO result (packed with _pack_cvec_signed). Fails loudly
    if no prior ReadAO is available — a programmer error, not silent.
    """
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
                        "Chained RunTile (acc_init_mode=1, "
                        "ACC_INIT_FROM_PREV_AO) issued with no prior "
                        "ReadAO to source the seed from"
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
                raise AssertionError("RunTile did not complete in 500 cycles")
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
            raise TypeError(f"Unknown instruction type: {type(instr).__name__}")

    if return_read_log:
        return results, read_log
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
async def test_compiled_matmul_k_chain_16(dut):
    """F1-C3 acceptance — 4×4 array runs K=16 matmul split into 2
    tiles of K=8 each via cfg_acc_init_mode chaining. Output must be
    bit-exact vs the single-tile K=16 emulation.
    """
    import random
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)

    rng = random.Random(0)
    W = [[rng.randint(-10, 10) for _ in range(4)] for _ in range(4)]
    acts = [[rng.randint(-10, 10) for _ in range(4)] for _ in range(16)]
    layer = Matmul(N_rows=4, N_cols=4, cfg_k=16, weights=W, afu_mode=0)

    program = compile_matmul_chained(
        layer,
        ai_base=0,
        ao_base_per_tile=[0, 1],
        activation_vectors=acts,
        k_per_tile=8,
    )
    dut._log.info(f"compiled chained K=16 program: {len(program)} instructions")

    results = await _execute_program(dut, program)
    expected = emulate_matmul(layer, acts)
    rtl = results[1]  # final tile writes to ao_base_per_tile[-1]
    assert rtl == expected, (
        f"K=16 chained matmul mismatch: rtl={rtl} expected={expected}"
    )
    dut._log.info(f"K=16 chained matmul PASS: {rtl}")


@cocotb.test()
async def test_compiled_conv2d_3x3_bit_exact(dut):
    """F1-C4 acceptance — 3×3 conv on a 4×4 input compiled via im2col
    (K-chunking + N-splitting + acc_init chaining), executed on RTL,
    compared bit-exactly against the reference_conv2d_int8 oracle.

    Small shape to fit the 4×4 array's AI/AO SRAM depth (16): H=W=4,
    C_in=1, C_out=2, pad=1, stride=1. K_total=9 → 3 K-chunks on a
    4-row array; M=16 output positions; C_out=2 fits in N_cols=4
    (1 N-chunk).
    """
    import numpy as np
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)

    rng = np.random.default_rng(42)
    x = rng.integers(-30, 30, size=(1, 1, 4, 4), dtype=np.int8)
    w = rng.integers(-10, 10, size=(2, 1, 3, 3), dtype=np.int8)

    res = compile_conv2d(w, x, n_rows=N_ROWS, n_cols=N_COLS,
                          stride=(1, 1), pad=(1, 1, 1, 1))
    dut._log.info(
        f"compiled conv2d: {len(res.program)} instrs, "
        f"K_total={res.K_total}, M={res.M}, "
        f"n_chunks={res.n_chunks}, k_chunks={res.k_chunks}"
    )

    _, read_log = await _execute_program(dut, res.program,
                                          return_read_log=True)
    out = reassemble_conv_output(read_log, res)
    ref = reference_conv2d_int8(x, w, stride=(1, 1), pad=(1, 1, 1, 1))

    if not np.array_equal(out, ref):
        dut._log.error(f"rtl out = {out[0, 0].tolist()}")
        dut._log.error(f"ref out = {ref[0, 0].tolist()}")
        assert False, "conv2d RTL output diverges from reference_conv2d_int8"
    dut._log.info(f"conv2d bit-exact PASS — output shape {out.shape}")


@cocotb.test()
async def test_compiled_conv2d_multi_round_m(dut):
    """GAP-2 — RTL bit-exact on a conv that exercises MULTI-ROUND M
    tiling in addition to K-chunking. Previous F1-C4 cocotb test
    (test_compiled_conv2d_3x3_bit_exact) had M=16 which fits one
    round. This test has M=25 (5×5 output), forcing M_per_round=16 +
    trailing round of 9 — the code path F1-C5 BIC-1 added but only
    Python-simulator-tested.

    Shape: 4 input channels × 3×3 kernel, pad=1 stride=1, 5×5 input.
      M=25 · K_total=36 (9 chunks × 4 rows) · C_out=4 (1 N-chunk).
    """
    import numpy as np
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)

    rng = np.random.default_rng(11)
    x = rng.integers(-30, 30, size=(1, 4, 5, 5), dtype=np.int8)
    w = rng.integers(-10, 10, size=(4, 4, 3, 3), dtype=np.int8)

    res = compile_conv2d(w, x, n_rows=N_ROWS, n_cols=N_COLS,
                          stride=(1, 1), pad=(1, 1, 1, 1))
    dut._log.info(
        f"multi-round M conv: {len(res.program)} instrs, "
        f"K_total={res.K_total}, M={res.M}, "
        f"n_chunks={res.n_chunks}, k_chunks={res.k_chunks}"
    )
    assert res.M == 25, f"expected M=25 to force multi-round M, got {res.M}"

    _, read_log = await _execute_program(dut, res.program,
                                          return_read_log=True)
    out = reassemble_conv_output(read_log, res)
    ref = reference_conv2d_int8(x, w, stride=(1, 1), pad=(1, 1, 1, 1))

    if not np.array_equal(out, ref):
        dut._log.error(f"rtl out[0,0] = {out[0, 0].tolist()}")
        dut._log.error(f"ref out[0,0] = {ref[0, 0].tolist()}")
        assert False, "multi-round M conv diverges from reference"
    dut._log.info(f"multi-round M conv PASS — output {out.shape}")


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
