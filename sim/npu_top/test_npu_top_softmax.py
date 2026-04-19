"""End-to-end softmax via npu_top multi-pass path (F1-A4 integration).

Pre-populates AI SRAM with a 64-element INT32 input vector, pulses start
with cfg_mp_mode=8 (MODE_SOFTMAX) + cfg_mp_vec_len=64, waits for done,
reads AO via ext_ao_re, and compares the Q0.8 outputs bit-for-bit
against `softmax_rtl_mirror` (the authoritative Python mirror of the
softmax RTL).

This exercises the full wiring:
  ext_ai_we → AI SRAM → npu_softmax (driven by tile_ctrl's mp_in_valid)
  → npu_softmax.out_data → tile_ctrl's mp_ao_we+mp_ao_waddr → AO SRAM
  → ext_ao_rdata.
"""

import sys
from pathlib import Path

import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
from tools.npu_ref.softmax_luts import (   # noqa: E402
    EXP_SCALE, softmax_rtl_mirror,
)


VEC_LEN = 64
CLK_NS  = 10

DATA_W    = 8
ACC_W     = 32
N_ROWS    = 4
N_COLS    = 4
AI_DATA_W = N_ROWS * DATA_W     # 32 bits
AO_DATA_W = N_COLS * ACC_W      # 128 bits


def _mask(v, w): return v & ((1 << w) - 1)


async def _reset(dut):
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.cfg_k.value = 0
    dut.cfg_ai_base.value = 0
    dut.cfg_ao_base.value = 0
    dut.cfg_afu_mode.value = 0
    dut.cfg_acc_init_mode.value = 0
    dut.cfg_acc_init_data.value = 0
    dut.cfg_precision_mode.value = 0
    dut.cfg_mp_mode.value = 0
    dut.cfg_mp_vec_len.value = 0
    dut.ext_w_we.value = 0
    dut.ext_ai_we.value = 0
    dut.ext_ao_re.value = 0
    dut.ext_sparse_skip_vec.value = 0
    dut.dma_start.value = 0
    dut.mem_rdata.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def _write_ai(dut, addr, value):
    """Write one 32-bit value into AI SRAM at `addr`."""
    dut.ext_ai_we.value = 1
    dut.ext_ai_waddr.value = addr
    dut.ext_ai_wdata.value = _mask(value, AI_DATA_W)
    await RisingEdge(dut.clk)
    dut.ext_ai_we.value = 0
    await Timer(1, unit="ns")


async def _read_ao(dut, addr) -> int:
    """Synchronous read: drive ext_ao_re + addr; SRAM returns rdata
    1 cycle later. Returns the low byte of the word (Q0.8 softmax out)."""
    dut.ext_ao_re.value = 1
    dut.ext_ao_raddr.value = addr
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    val = int(dut.ext_ao_rdata.value)
    return val & 0xFF


@cocotb.test()
async def test_softmax_end_to_end_uniform(dut):
    """Uniform input → uniform output ≈ 256/64 = 4 per slot."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await _reset(dut)

    x = np.zeros(VEC_LEN, dtype=np.int32)
    # Pre-load AI SRAM
    for i, v in enumerate(x):
        await _write_ai(dut, i, int(v))

    # Configure + start softmax
    dut.cfg_mp_mode.value = 8
    dut.cfg_mp_vec_len.value = VEC_LEN
    dut.cfg_ai_base.value = 0
    dut.cfg_ao_base.value = 0
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await Timer(1, unit="ns")

    # Wait for done (2*VEC_LEN + gap + drain ≈ 140 cycles)
    done_seen = False
    for _ in range(300):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.done.value) == 1:
            done_seen = True
            break
    assert done_seen, "done never pulsed"

    # Read back AO
    y = np.zeros(VEC_LEN, dtype=np.uint8)
    for i in range(VEC_LEN):
        y[i] = await _read_ao(dut, i)
    dut.ext_ao_re.value = 0

    expected = softmax_rtl_mirror(x)
    dut._log.info(f"uniform e2e: y[0..4]={y[:4].tolist()}  sum={int(y.sum())}  expected={expected[:4].tolist()}")
    assert np.array_equal(y, expected), (
        f"uniform e2e mismatch:\n  RTL:    {y.tolist()}\n  mirror: {expected.tolist()}")


@cocotb.test()
async def test_softmax_end_to_end_random(dut):
    """Random input → matches bit-exact mirror."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await _reset(dut)

    rng = np.random.default_rng(0xA4F00D)
    x_fp = rng.standard_normal(VEC_LEN).astype(np.float32) * 2.0
    x_int = np.round(x_fp * EXP_SCALE).astype(np.int32)

    for i, v in enumerate(x_int):
        await _write_ai(dut, i, int(v))

    dut.cfg_mp_mode.value = 8
    dut.cfg_mp_vec_len.value = VEC_LEN
    dut.cfg_ai_base.value = 0
    dut.cfg_ao_base.value = 0
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await Timer(1, unit="ns")

    for _ in range(300):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "done never pulsed"

    y = np.zeros(VEC_LEN, dtype=np.uint8)
    for i in range(VEC_LEN):
        y[i] = await _read_ao(dut, i)
    dut.ext_ao_re.value = 0

    expected = softmax_rtl_mirror(x_int)
    diffs = int(np.sum(y != expected))
    dut._log.info(f"random e2e: {diffs} mismatches; sum={int(y.sum())}")
    assert diffs == 0, (
        f"random e2e has {diffs} mismatches:\n  RTL[:8]={y[:8].tolist()}\n  mir[:8]={expected[:8].tolist()}")
