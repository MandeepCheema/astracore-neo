"""Cycle-count validation: measure actual RTL cycles from start→done for a
matmul tile and compare against the perf_model analytical prediction."""

import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
from tools.npu_ref.perf_model import NpuConfig, one_tile_cycles  # noqa

DATA_W    = 8
N_ROWS    = 4
N_COLS    = 4


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


async def _preload_weights(dut, W):
    for i, w in enumerate(W):
        dut.ext_w_we.value = 1
        dut.ext_w_waddr.value = i
        dut.ext_w_wdata.value = w & 0xFF
        await RisingEdge(dut.clk); await Timer(1, unit="ns")
    dut.ext_w_we.value = 0


async def _load_act(dut, addr, vec):
    packed = 0
    for i, v in enumerate(vec):
        packed |= (v & 0xFF) << (i * 8)
    dut.ext_ai_we.value = 1
    dut.ext_ai_waddr.value = addr
    dut.ext_ai_wdata.value = packed
    await RisingEdge(dut.clk); await Timer(1, unit="ns")
    dut.ext_ai_we.value = 0


@cocotb.test()
async def test_measure_tile_cycles_k1(dut):
    """Measure RTL tile cycles from start→done at cfg_k=1, compare to perf_model."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    W = [1 if (a // N_COLS) == (a % N_COLS) else 0
         for a in range(N_ROWS * N_COLS)]
    await _preload_weights(dut, W)
    await _load_act(dut, 0, [7, -3, 11, -5])

    # Trigger tile, count cycles until done
    dut.start.value   = 1
    dut.cfg_k.value   = 1
    dut.cfg_ai_base.value = 0
    dut.cfg_ao_base.value = 0
    await RisingEdge(dut.clk); await Timer(1, unit="ns")
    dut.start.value = 0
    count = 1  # first cycle after start
    while not int(dut.done.value):
        await RisingEdge(dut.clk); await Timer(1, unit="ns")
        count += 1
        assert count < 100, "tile took too long"

    cfg = NpuConfig(
        name="m", n_rows=N_ROWS, n_cols=N_COLS, clock_hz=1e8,
        sram_bytes=256, weights_per_cycle=N_COLS,
    )
    predicted = one_tile_cycles(cfg, k=1)
    delta = count - predicted
    dut._log.info(f"RTL cfg_k=1 tile cycles (start→done) = {count}, "
                  f"perf_model prediction = {predicted}, delta = {delta}")
    # Within 1 cycle is the "validated" bar
    assert abs(delta) <= 1, f"RTL vs perf_model delta > 1 cycle: {delta}"


@cocotb.test()
async def test_measure_tile_cycles_k3(dut):
    """cfg_k=3 accumulated — 3 execute cycles added."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    W = [1 if (a // N_COLS) == (a % N_COLS) else 0
         for a in range(N_ROWS * N_COLS)]
    await _preload_weights(dut, W)
    for i in range(3):
        await _load_act(dut, i, [1, 2, 3, 4])
    dut.start.value = 1; dut.cfg_k.value = 3
    dut.cfg_ai_base.value = 0; dut.cfg_ao_base.value = 0
    await RisingEdge(dut.clk); await Timer(1, unit="ns")
    dut.start.value = 0
    count = 1
    while not int(dut.done.value):
        await RisingEdge(dut.clk); await Timer(1, unit="ns")
        count += 1
        assert count < 100
    cfg = NpuConfig(name="m", n_rows=N_ROWS, n_cols=N_COLS, clock_hz=1e8,
                    sram_bytes=256, weights_per_cycle=N_COLS)
    predicted = one_tile_cycles(cfg, k=3)
    delta = count - predicted
    dut._log.info(f"RTL cfg_k=3 tile cycles = {count}, perf_model = {predicted}, delta = {delta}")
    assert abs(delta) <= 1


@cocotb.test()
async def test_measure_tile_cycles_k8(dut):
    """cfg_k=8 — longer run, verify cycle count scales linearly."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    W = [1 if (a // N_COLS) == (a % N_COLS) else 0
         for a in range(N_ROWS * N_COLS)]
    await _preload_weights(dut, W)
    for i in range(8):
        await _load_act(dut, i, [1, 2, 3, 4])
    dut.start.value = 1; dut.cfg_k.value = 8
    dut.cfg_ai_base.value = 0; dut.cfg_ao_base.value = 0
    await RisingEdge(dut.clk); await Timer(1, unit="ns")
    dut.start.value = 0
    count = 1
    while not int(dut.done.value):
        await RisingEdge(dut.clk); await Timer(1, unit="ns")
        count += 1
        assert count < 100
    cfg = NpuConfig(name="m", n_rows=N_ROWS, n_cols=N_COLS, clock_hz=1e8,
                    sram_bytes=256, weights_per_cycle=N_COLS)
    predicted = one_tile_cycles(cfg, k=8)
    delta = count - predicted
    dut._log.info(f"RTL cfg_k=8 tile cycles = {count}, perf_model = {predicted}, delta = {delta}")
    assert abs(delta) <= 1
