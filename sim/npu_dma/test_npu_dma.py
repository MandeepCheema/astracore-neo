"""cocotb bit-exact regression for rtl/npu_dma/npu_dma.v.

Cases:
  - 1D linear transfer (tile_h=1, no pad)
  - 2D tile with src_stride != tile_w
  - 2D tile with symmetric padding
  - 2D tile with asymmetric padding
  - Back-to-back transfers (busy must drop, then second start must latch
    fresh config)

Mechanism: the testbench runs a memory responder coroutine that drives
mem_rdata one cycle after each mem_re/mem_raddr request (matching the
1-cycle SRAM read latency the DMA is designed for).  A capture
coroutine records every destination write to a Python dict, then we
compare the dict to what Python's `Dma` golden reference produces for
the same config and source data.
"""

import os
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
from tools.npu_ref.dma_ref import Dma  # noqa: E402


DATA_W     = 8
SRC_ADDR_W = 32
DST_ADDR_W = 16
LEN_W      = 16


def _mask(v, w): return v & ((1 << w) - 1)


async def _reset(dut):
    dut.rst_n.value = 0
    for s in ("cfg_src_addr", "cfg_dst_addr", "cfg_tile_h", "cfg_tile_w",
              "cfg_src_stride", "cfg_pad_top", "cfg_pad_bot",
              "cfg_pad_left", "cfg_pad_right", "start", "mem_rdata"):
        getattr(dut, s).value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def _mem_responder(dut, src_mem):
    """Model a synchronous memory with 1-cycle read latency.

    Mid-cycle (FallingEdge) sample what the DMA is asking for RIGHT NOW.
    Immediately after the next RisingEdge, drive mem_rdata so it is stable
    for the rest of that cycle, well before the next FallingEdge.
    """
    while True:
        await FallingEdge(dut.clk)
        if int(dut.mem_re.value):
            captured = int(dut.mem_raddr.value)
        else:
            captured = None
        # Drive mem_rdata on the next rising edge with no extra delay so
        # it has settled by mid-cycle when dst_capture samples.
        await RisingEdge(dut.clk)
        if captured is not None:
            dut.mem_rdata.value = src_mem.get(captured, 0) & ((1 << DATA_W) - 1)
        else:
            dut.mem_rdata.value = 0


async def _dst_capture(dut, captured):
    """Record every destination write into the captured dict.

    Sample at FallingEdge (mid-cycle) so mem_rdata has been driven by the
    responder after the preceding RisingEdge and the combinational
    dst_wdata = is_real_b ? mem_rdata : 0 has stabilised.
    """
    while True:
        await FallingEdge(dut.clk)
        if int(dut.dst_we.value):
            captured[int(dut.dst_waddr.value)] = int(dut.dst_wdata.value)


async def _run_transfer(dut, cfg, src_mem):
    """Start a transfer, run until done=1, and return the captured writes."""
    captured = {}
    responder = cocotb.start_soon(_mem_responder(dut, src_mem))
    capture = cocotb.start_soon(_dst_capture(dut, captured))

    # Drive config and start
    dut.cfg_src_addr.value   = cfg["cfg_src_addr"]
    dut.cfg_dst_addr.value   = cfg["cfg_dst_addr"]
    dut.cfg_tile_h.value     = cfg["cfg_tile_h"]
    dut.cfg_tile_w.value     = cfg["cfg_tile_w"]
    dut.cfg_src_stride.value = cfg["cfg_src_stride"]
    dut.cfg_pad_top.value    = cfg.get("cfg_pad_top", 0)
    dut.cfg_pad_bot.value    = cfg.get("cfg_pad_bot", 0)
    dut.cfg_pad_left.value   = cfg.get("cfg_pad_left", 0)
    dut.cfg_pad_right.value  = cfg.get("cfg_pad_right", 0)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.start.value = 0

    # Wait for done (with bound)
    for _ in range(20_000):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.done.value):
            break
    else:
        responder.kill(); capture.kill()
        raise cocotb.result.TestFailure("DMA did not complete within bound")
    # Give one more cycle for last capture
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    responder.kill()
    capture.kill()
    return captured


def _expected(cfg, src_mem):
    """Reference: run the same transfer on the Python model and capture writes."""
    captured = {}
    dma = Dma(data_w=DATA_W, src_addr_w=SRC_ADDR_W,
              dst_addr_w=DST_ADDR_W, len_w=LEN_W)
    dma.reset()
    dma.tick(start=1, **cfg)
    pending = None
    for _ in range(20_000):
        if pending is not None:
            dma.set_mem_rdata(src_mem.get(pending, 0))
        else:
            dma.set_mem_rdata(0)
        pending = dma.mem_raddr if dma.mem_re else None
        if dma.dst_we:
            captured[dma.dst_waddr] = dma.dst_wdata
        if dma.done:
            return captured
        dma.tick()
    raise RuntimeError("Python DMA did not complete")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_linear_1d(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    src = {100 + k: 0x10 + k for k in range(5)}
    cfg = dict(cfg_src_addr=100, cfg_dst_addr=0,
               cfg_tile_h=1, cfg_tile_w=5, cfg_src_stride=5)
    captured = await _run_transfer(dut, cfg, src)
    expected = _expected(cfg, src)
    assert captured == expected, f"rtl={captured} ref={expected}"


@cocotb.test()
async def test_2d_tile_with_stride(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    src = {200 + r * 10 + c: 0x20 + r * 4 + c
           for r in range(3) for c in range(4)}
    cfg = dict(cfg_src_addr=200, cfg_dst_addr=0,
               cfg_tile_h=3, cfg_tile_w=4, cfg_src_stride=10)
    captured = await _run_transfer(dut, cfg, src)
    expected = _expected(cfg, src)
    assert captured == expected, f"rtl={captured} ref={expected}"


@cocotb.test()
async def test_2d_with_symmetric_padding(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    src = {50: 0xA, 51: 0xB, 60: 0xC, 61: 0xD}
    cfg = dict(cfg_src_addr=50, cfg_dst_addr=0,
               cfg_tile_h=2, cfg_tile_w=2, cfg_src_stride=10,
               cfg_pad_top=1, cfg_pad_bot=1,
               cfg_pad_left=1, cfg_pad_right=1)
    captured = await _run_transfer(dut, cfg, src)
    expected = _expected(cfg, src)
    assert captured == expected, f"rtl={captured} ref={expected}"


@cocotb.test()
async def test_2d_with_asymmetric_padding(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    src = {0: 0x1, 1: 0x2, 2: 0x3}
    cfg = dict(cfg_src_addr=0, cfg_dst_addr=0,
               cfg_tile_h=1, cfg_tile_w=3, cfg_src_stride=3,
               cfg_pad_top=0, cfg_pad_bot=1,
               cfg_pad_left=2, cfg_pad_right=0)
    captured = await _run_transfer(dut, cfg, src)
    expected = _expected(cfg, src)
    assert captured == expected, f"rtl={captured} ref={expected}"


@cocotb.test()
async def test_back_to_back_transfers(dut):
    """Two transfers in succession: busy must drop after the first so
    a subsequent start pulse latches fresh config."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    # First transfer
    src1 = {100 + k: 0x10 + k for k in range(4)}
    cfg1 = dict(cfg_src_addr=100, cfg_dst_addr=0,
                cfg_tile_h=1, cfg_tile_w=4, cfg_src_stride=4)
    c1 = await _run_transfer(dut, cfg1, src1)
    assert c1 == _expected(cfg1, src1)
    # Second transfer to a DIFFERENT destination address
    src2 = {200 + k: 0xA0 + k for k in range(3)}
    cfg2 = dict(cfg_src_addr=200, cfg_dst_addr=8,
                cfg_tile_h=1, cfg_tile_w=3, cfg_src_stride=3)
    c2 = await _run_transfer(dut, cfg2, src2)
    assert c2 == _expected(cfg2, src2)


@cocotb.test()
async def test_3x3_with_2_pad(dut):
    """Bigger tile: 3x3 src data with pad=2 all around (for 5x5 conv).
    Exercises deeper-loop address arithmetic."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    src = {0x300 + r * 20 + c: 0x40 + r * 3 + c
           for r in range(3) for c in range(3)}
    cfg = dict(cfg_src_addr=0x300, cfg_dst_addr=0,
               cfg_tile_h=3, cfg_tile_w=3, cfg_src_stride=20,
               cfg_pad_top=2, cfg_pad_bot=2,
               cfg_pad_left=2, cfg_pad_right=2)
    captured = await _run_transfer(dut, cfg, src)
    expected = _expected(cfg, src)
    assert captured == expected, f"rtl={captured} ref={expected}"
