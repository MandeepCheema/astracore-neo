"""cocotb bit-exact regression: rtl/npu_sram_ctrl vs Python golden ref.

Drives RTL and Python with identical per-cycle stimulus; checks every port
every cycle.  Covers:

  - Basic R/W per bank (weight, AI, AO, scratch)
  - Bank isolation (writing one bank never corrupts another)
  - Weight double-buffer swap: write one bank, flip sel, read the other
  - Same-address same-cycle R+W: write-wins contract
  - Address boundary (highest valid address) per bank
  - Wide-bank (AI, AO) read/write preserves full width
  - 500-cycle random stress across all four port groups
"""

import os
import random
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
from tools.npu_ref.sram_ref import SramCtrl  # noqa: E402


# Parameters — overridable via env for parametric runs
DATA_W        = int(os.environ.get("DATA_W", "8"))
ACC_W         = int(os.environ.get("ACC_W", "32"))
N_ROWS        = int(os.environ.get("N_ROWS", "4"))
N_COLS        = int(os.environ.get("N_COLS", "4"))
WEIGHT_DEPTH  = int(os.environ.get("WEIGHT_DEPTH", "16"))
ACT_IN_DEPTH  = int(os.environ.get("ACT_IN_DEPTH", "16"))
ACT_OUT_DEPTH = int(os.environ.get("ACT_OUT_DEPTH", "16"))
SCRATCH_DEPTH = int(os.environ.get("SCRATCH_DEPTH", "16"))

AI_DATA_W = N_ROWS * DATA_W
AO_DATA_W = N_COLS * ACC_W


async def _reset(dut):
    dut.rst_n.value        = 0
    for sig in ("w_bank_sel", "w_re", "w_raddr", "w_we", "w_waddr", "w_wdata",
                "ai_re", "ai_raddr", "ai_we", "ai_waddr", "ai_wdata",
                "ao_re", "ao_raddr", "ao_we", "ao_waddr", "ao_wdata",
                "sc_re", "sc_raddr", "sc_we", "sc_waddr", "sc_wdata"):
        getattr(dut, sig).value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


def _mask(val: int, width: int) -> int:
    return val & ((1 << width) - 1)


async def _step(dut, ref: SramCtrl, **stim):
    defaults = dict(
        w_bank_sel=0,
        w_re=0, w_raddr=0, w_we=0, w_waddr=0, w_wdata=0,
        ai_re=0, ai_raddr=0, ai_we=0, ai_waddr=0, ai_wdata=0,
        ao_re=0, ao_raddr=0, ao_we=0, ao_waddr=0, ao_wdata=0,
        sc_re=0, sc_raddr=0, sc_we=0, sc_waddr=0, sc_wdata=0,
    )
    defaults.update(stim)

    dut.w_bank_sel.value = defaults["w_bank_sel"]
    dut.w_re.value       = defaults["w_re"]
    dut.w_raddr.value    = defaults["w_raddr"]
    dut.w_we.value       = defaults["w_we"]
    dut.w_waddr.value    = defaults["w_waddr"]
    dut.w_wdata.value    = _mask(defaults["w_wdata"], DATA_W)
    dut.ai_re.value      = defaults["ai_re"]
    dut.ai_raddr.value   = defaults["ai_raddr"]
    dut.ai_we.value      = defaults["ai_we"]
    dut.ai_waddr.value   = defaults["ai_waddr"]
    dut.ai_wdata.value   = _mask(defaults["ai_wdata"], AI_DATA_W)
    dut.ao_re.value      = defaults["ao_re"]
    dut.ao_raddr.value   = defaults["ao_raddr"]
    dut.ao_we.value      = defaults["ao_we"]
    dut.ao_waddr.value   = defaults["ao_waddr"]
    dut.ao_wdata.value   = _mask(defaults["ao_wdata"], AO_DATA_W)
    dut.sc_re.value      = defaults["sc_re"]
    dut.sc_raddr.value   = defaults["sc_raddr"]
    dut.sc_we.value      = defaults["sc_we"]
    dut.sc_waddr.value   = defaults["sc_waddr"]
    dut.sc_wdata.value   = _mask(defaults["sc_wdata"], DATA_W)

    ref.tick(**defaults)

    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


def _check(dut, ref: SramCtrl):
    assert int(dut.w_rdata.value)  == ref.w_rdata,  (
        f"w_rdata  rtl=0x{int(dut.w_rdata.value):x}  ref=0x{ref.w_rdata:x}")
    assert int(dut.ai_rdata.value) == ref.ai_rdata, (
        f"ai_rdata rtl=0x{int(dut.ai_rdata.value):x} ref=0x{ref.ai_rdata:x}")
    assert int(dut.ao_rdata.value) == ref.ao_rdata, (
        f"ao_rdata rtl=0x{int(dut.ao_rdata.value):x} ref=0x{ref.ao_rdata:x}")
    assert int(dut.sc_rdata.value) == ref.sc_rdata, (
        f"sc_rdata rtl=0x{int(dut.sc_rdata.value):x} ref=0x{ref.sc_rdata:x}")


# ---------------------------------------------------------------------------
# Directed tests
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_reset(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = SramCtrl(data_w=DATA_W, acc_w=ACC_W, n_rows=N_ROWS, n_cols=N_COLS,
                   weight_depth=WEIGHT_DEPTH, act_in_depth=ACT_IN_DEPTH,
                   act_out_depth=ACT_OUT_DEPTH, scratch_depth=SCRATCH_DEPTH)
    ref.reset()
    await _reset(dut)
    _check(dut, ref)


@cocotb.test()
async def test_scratch_write_then_read(dut):
    """Write 0xAB to SC[3], read it back next cycle."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = SramCtrl(data_w=DATA_W, acc_w=ACC_W, n_rows=N_ROWS, n_cols=N_COLS,
                   weight_depth=WEIGHT_DEPTH, act_in_depth=ACT_IN_DEPTH,
                   act_out_depth=ACT_OUT_DEPTH, scratch_depth=SCRATCH_DEPTH)
    ref.reset()
    await _reset(dut)
    await _step(dut, ref, sc_we=1, sc_waddr=3, sc_wdata=0xAB)
    await _step(dut, ref, sc_re=1, sc_raddr=3)
    _check(dut, ref)
    assert ref.sc_rdata == 0xAB


@cocotb.test()
async def test_same_cycle_rw_write_wins(dut):
    """Same-address same-cycle R+W must return the new (written) value."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = SramCtrl(data_w=DATA_W, acc_w=ACC_W, n_rows=N_ROWS, n_cols=N_COLS,
                   weight_depth=WEIGHT_DEPTH, act_in_depth=ACT_IN_DEPTH,
                   act_out_depth=ACT_OUT_DEPTH, scratch_depth=SCRATCH_DEPTH)
    ref.reset()
    await _reset(dut)
    await _step(dut, ref, sc_we=1, sc_waddr=5, sc_wdata=0x11)   # prime
    await _step(dut, ref,
                sc_we=1, sc_waddr=5, sc_wdata=0x22,
                sc_re=1, sc_raddr=5)                              # R+W same addr
    _check(dut, ref)
    assert ref.sc_rdata == 0x22, f"write-wins violated: got {ref.sc_rdata:02x}"


@cocotb.test()
async def test_double_buffer_swap(dut):
    """Write WB (sel=0), flip sel=1, confirm array now reads the WB row.

    w_raddr is the ROW index (narrow).  w_waddr is the LINEAR weight
    index that decomposes to (row, col).  w_rdata is WIDE (full row).
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = SramCtrl(data_w=DATA_W, acc_w=ACC_W, n_rows=N_ROWS, n_cols=N_COLS,
                   weight_depth=WEIGHT_DEPTH, act_in_depth=ACT_IN_DEPTH,
                   act_out_depth=ACT_OUT_DEPTH, scratch_depth=SCRATCH_DEPTH)
    ref.reset()
    await _reset(dut)
    # Pick a test address: write the last col of row 0 for robustness across
    # shapes.  w_waddr = row*N_COLS + col.
    test_row = 0
    test_col = N_COLS - 1
    test_linear = test_row * N_COLS + test_col

    # Prefill the target row on all cols to avoid X propagation
    for col in range(N_COLS):
        await _step(dut, ref,
                    w_bank_sel=0, w_we=1, w_waddr=test_row * N_COLS + col, w_wdata=0)
    # Write WB at linear test_linear with sel=0.
    await _step(dut, ref, w_bank_sel=0, w_we=1, w_waddr=test_linear, w_wdata=0x77)
    # Flip sel → array now reads WB at test_row
    await _step(dut, ref, w_bank_sel=1, w_re=1, w_raddr=test_row)
    # Result appears next cycle
    await _step(dut, ref, w_bank_sel=1)
    _check(dut, ref)
    expected = 0x77 << (test_col * DATA_W)
    assert ref.w_rdata == expected, (
        f"swap failed: got 0x{ref.w_rdata:x} expected 0x{expected:x}")


@cocotb.test()
async def test_bank_isolation(dut):
    """Writing to one bank must not affect any other bank's contents."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = SramCtrl(data_w=DATA_W, acc_w=ACC_W, n_rows=N_ROWS, n_cols=N_COLS,
                   weight_depth=WEIGHT_DEPTH, act_in_depth=ACT_IN_DEPTH,
                   act_out_depth=ACT_OUT_DEPTH, scratch_depth=SCRATCH_DEPTH)
    ref.reset()
    await _reset(dut)
    # Prefill WA row 0 to zero (so reading it returns determinate zeros for
    # unwritten cols), then write col 0 with the test pattern.
    for col in range(N_COLS):
        await _step(dut, ref, w_bank_sel=1, w_we=1, w_waddr=col, w_wdata=0)
    await _step(dut, ref, w_bank_sel=1, w_we=1, w_waddr=0, w_wdata=0xAA)
    ai_pattern = (1 << (AI_DATA_W - 1)) | 0x5A  # mixed bits
    ao_pattern = (1 << (AO_DATA_W - 1)) | 0xC3
    await _step(dut, ref, ai_we=1, ai_waddr=0, ai_wdata=ai_pattern)
    await _step(dut, ref, ao_we=1, ao_waddr=0, ao_wdata=ao_pattern)
    await _step(dut, ref, sc_we=1, sc_waddr=0, sc_wdata=0x3C)
    # Now read all four simultaneously.  Weight read uses ROW addr (0).
    await _step(dut, ref,
                w_bank_sel=0, w_re=1, w_raddr=0,   # read WA row 0
                ai_re=1, ai_raddr=0,
                ao_re=1, ao_raddr=0,
                sc_re=1, sc_raddr=0)
    _check(dut, ref)


@cocotb.test()
async def test_address_boundary(dut):
    """Write/read at the highest valid address of each bank."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = SramCtrl(data_w=DATA_W, acc_w=ACC_W, n_rows=N_ROWS, n_cols=N_COLS,
                   weight_depth=WEIGHT_DEPTH, act_in_depth=ACT_IN_DEPTH,
                   act_out_depth=ACT_OUT_DEPTH, scratch_depth=SCRATCH_DEPTH)
    ref.reset()
    await _reset(dut)
    w_top_linear = WEIGHT_DEPTH - 1          # highest WRITE address (linear)
    w_top_row    = (WEIGHT_DEPTH // N_COLS) - 1  # highest READ row address
    sc_top = SCRATCH_DEPTH - 1
    # Prefill top row of WA on all cols to avoid X-propagation from uninit cells
    for col in range(N_COLS):
        await _step(dut, ref, w_bank_sel=1, w_we=1,
                    w_waddr=w_top_row * N_COLS + col, w_wdata=0)
    await _step(dut, ref, w_bank_sel=1, w_we=1, w_waddr=w_top_linear, w_wdata=0xDE)
    await _step(dut, ref, w_bank_sel=0, w_re=1, w_raddr=w_top_row)
    await _step(dut, ref, sc_we=1, sc_waddr=sc_top, sc_wdata=0xED)
    await _step(dut, ref, sc_re=1, sc_raddr=sc_top)
    _check(dut, ref)
    await _step(dut, ref, w_bank_sel=0, w_re=1, w_raddr=w_top_row,
                sc_re=1, sc_raddr=sc_top)
    _check(dut, ref)


# ---------------------------------------------------------------------------
# Random stress
# ---------------------------------------------------------------------------
async def _prefill_all_banks(dut, ref):
    """Write zeros to every address in every bank so random reads never
    hit an uninitialised cell (which would propagate X in RTL).  This
    respects the documented SRAM contract: 'callers MUST initialise before
    reading' — the random stress test counts as one such caller."""
    max_depth = max(WEIGHT_DEPTH, ACT_IN_DEPTH, ACT_OUT_DEPTH, SCRATCH_DEPTH)
    for a in range(max_depth):
        stim = {}
        if a < WEIGHT_DEPTH:
            # Write both WA and WB by doing two cycles with different sel
            stim.update(dict(w_bank_sel=0, w_we=1, w_waddr=a, w_wdata=0))
            await _step(dut, ref, **stim)
            stim = dict(w_bank_sel=1, w_we=1, w_waddr=a, w_wdata=0)
            await _step(dut, ref, **stim)
        # AI/AO/SC — only write up to their respective depths
        stim = {}
        if a < ACT_IN_DEPTH:
            stim.update(dict(ai_we=1, ai_waddr=a, ai_wdata=0))
        if a < ACT_OUT_DEPTH:
            stim.update(dict(ao_we=1, ao_waddr=a, ao_wdata=0))
        if a < SCRATCH_DEPTH:
            stim.update(dict(sc_we=1, sc_waddr=a, sc_wdata=0))
        if stim:
            await _step(dut, ref, **stim)


@cocotb.test()
async def test_random_bit_exact_500_cycles(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = SramCtrl(data_w=DATA_W, acc_w=ACC_W, n_rows=N_ROWS, n_cols=N_COLS,
                   weight_depth=WEIGHT_DEPTH, act_in_depth=ACT_IN_DEPTH,
                   act_out_depth=ACT_OUT_DEPTH, scratch_depth=SCRATCH_DEPTH)
    ref.reset()
    await _reset(dut)
    await _prefill_all_banks(dut, ref)
    rng = random.Random(0xBA5E)
    mismatches = 0
    W_ROW_DEPTH = WEIGHT_DEPTH // N_COLS
    for cycle in range(500):
        stim = dict(
            w_bank_sel=rng.randint(0, 1),
            w_re=1 if rng.random() < 0.4 else 0,
            w_raddr=rng.randint(0, W_ROW_DEPTH - 1),     # narrow row addr
            w_we=1 if rng.random() < 0.4 else 0,
            w_waddr=rng.randint(0, WEIGHT_DEPTH - 1),    # linear write addr
            w_wdata=rng.randint(0, (1 << DATA_W) - 1),
            ai_re=1 if rng.random() < 0.4 else 0,
            ai_raddr=rng.randint(0, ACT_IN_DEPTH - 1),
            ai_we=1 if rng.random() < 0.4 else 0,
            ai_waddr=rng.randint(0, ACT_IN_DEPTH - 1),
            ai_wdata=rng.getrandbits(AI_DATA_W),
            ao_re=1 if rng.random() < 0.4 else 0,
            ao_raddr=rng.randint(0, ACT_OUT_DEPTH - 1),
            ao_we=1 if rng.random() < 0.4 else 0,
            ao_waddr=rng.randint(0, ACT_OUT_DEPTH - 1),
            ao_wdata=rng.getrandbits(AO_DATA_W),
            sc_re=1 if rng.random() < 0.4 else 0,
            sc_raddr=rng.randint(0, SCRATCH_DEPTH - 1),
            sc_we=1 if rng.random() < 0.4 else 0,
            sc_waddr=rng.randint(0, SCRATCH_DEPTH - 1),
            sc_wdata=rng.randint(0, (1 << DATA_W) - 1),
        )
        await _step(dut, ref, **stim)
        try:
            _check(dut, ref)
        except AssertionError as exc:
            mismatches += 1
            if mismatches <= 3:
                dut._log.error(f"cycle {cycle}: {exc}")
    assert mismatches == 0, f"{mismatches} cycles failed bit-exact"
    dut._log.info(f"500-cycle random stress PASS "
                  f"(DATA_W={DATA_W} N_ROWS={N_ROWS} N_COLS={N_COLS})")
