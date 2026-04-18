"""cocotb bit-exact regression for rtl/npu_tile_ctrl/npu_tile_ctrl.v.

Standalone FSM test: no SRAM/array/AFU instantiated.  Drives start and config
on the tile_ctrl, then compares every control output cycle-by-cycle against
the Python golden reference.  This proves the sequencer produces the right
control signals in the right order; integration with the datapath follows
once AFU and DMA are in place.
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
from tools.npu_ref.tile_ctrl_ref import (  # noqa: E402
    TileCtrl,
    S_IDLE, S_PRELOAD, S_EXEC_PREP, S_EXECUTE, S_DRAIN, S_STORE, S_DONE,
)


N_ROWS    = 4
N_COLS    = 4
AI_ADDR_W = 8
AO_ADDR_W = 8
K_W       = 16
DRAIN_CYCLES = 2
# After the wide-weight change, W_ADDR_W is the ROW-index width.
W_ADDR_W  = max(1, (N_ROWS - 1).bit_length())


async def _reset(dut):
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.cfg_k.value = 0
    dut.cfg_ai_base.value = 0
    dut.cfg_ao_base.value = 0
    dut.cfg_acc_init_mode.value = 0
    dut.cfg_acc_init_data.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def _step(dut, ref: TileCtrl, *, start=0, cfg_k=0,
                cfg_ai_base=0, cfg_ao_base=0,
                cfg_acc_init_mode=0, cfg_acc_init_data=0):
    dut.start.value = start
    dut.cfg_k.value = cfg_k
    dut.cfg_ai_base.value = cfg_ai_base
    dut.cfg_ao_base.value = cfg_ao_base
    dut.cfg_acc_init_mode.value = cfg_acc_init_mode
    dut.cfg_acc_init_data.value = cfg_acc_init_data & ((1 << (N_COLS * 32)) - 1)
    ref.tick(start=start, cfg_k=cfg_k,
             cfg_ai_base=cfg_ai_base, cfg_ao_base=cfg_ao_base,
             cfg_acc_init_mode=cfg_acc_init_mode,
             cfg_acc_init_data=cfg_acc_init_data)
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


def _check(dut, ref: TileCtrl):
    checks = [
        ("busy",                   int(dut.busy.value),                   ref.busy),
        ("done",                   int(dut.done.value),                   ref.done),
        ("w_bank_sel",             int(dut.w_bank_sel.value),             ref.w_bank_sel),
        ("w_re",                   int(dut.w_re.value),                   ref.w_re),
        ("w_raddr",                int(dut.w_raddr.value),                ref.w_raddr),
        ("array_load_valid",       int(dut.array_load_valid.value),       ref.array_load_valid),
        ("array_load_cell_addr",   int(dut.array_load_cell_addr.value),   ref.array_load_cell_addr),
        ("array_clear_acc",        int(dut.array_clear_acc.value),        ref.array_clear_acc),
        ("array_acc_load_valid",   int(dut.array_acc_load_valid.value),   ref.array_acc_load_valid),
        ("array_acc_load_data",    int(dut.array_acc_load_data.value),    ref.array_acc_load_data),
        ("ai_re",                  int(dut.ai_re.value),                  ref.ai_re),
        ("ai_raddr",               int(dut.ai_raddr.value),               ref.ai_raddr),
        ("array_exec_valid",       int(dut.array_exec_valid.value),       ref.array_exec_valid),
        ("array_afu_in_valid",     int(dut.array_afu_in_valid.value),     ref.array_afu_in_valid),
        ("ao_we",                  int(dut.ao_we.value),                  ref.ao_we),
        ("ao_waddr",               int(dut.ao_waddr.value),               ref.ao_waddr),
    ]
    for name, rtl_v, ref_v in checks:
        assert rtl_v == ref_v, f"{name} mismatch: rtl={rtl_v} ref={ref_v}"


# ---------------------------------------------------------------------------
@cocotb.test()
async def test_reset_and_idle(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = TileCtrl(n_rows=N_ROWS, n_cols=N_COLS,
                   ai_addr_w=AI_ADDR_W, ao_addr_w=AO_ADDR_W,
                   k_w=K_W, drain_cycles=DRAIN_CYCLES)
    ref.reset()
    await _reset(dut)
    _check(dut, ref)
    # A few cycles of idle should keep everything at 0
    for _ in range(5):
        await _step(dut, ref)
        _check(dut, ref)


@cocotb.test()
async def test_one_tile_k3(dut):
    """Full tile with K=3 activation vectors.  Validates the full control
    sequence cycle-by-cycle."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = TileCtrl(n_rows=N_ROWS, n_cols=N_COLS,
                   ai_addr_w=AI_ADDR_W, ao_addr_w=AO_ADDR_W,
                   k_w=K_W, drain_cycles=DRAIN_CYCLES)
    ref.reset()
    await _reset(dut)

    # Fire start with config
    await _step(dut, ref, start=1, cfg_k=3, cfg_ai_base=5, cfg_ao_base=9)
    _check(dut, ref)

    # Let it run to completion — far more cycles than strictly needed
    for _ in range(80):
        await _step(dut, ref)
        _check(dut, ref)
        if int(dut.done.value):
            break
    else:
        raise cocotb.result.TestFailure("tile_ctrl did not assert done")

    # After done, one more cycle back to idle
    await _step(dut, ref)
    _check(dut, ref)
    assert int(dut.busy.value) == 0


@cocotb.test()
async def test_back_to_back_tiles(dut):
    """Run two tiles in succession; busy must drop between them so the
    second start latches fresh config."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = TileCtrl(n_rows=N_ROWS, n_cols=N_COLS,
                   ai_addr_w=AI_ADDR_W, ao_addr_w=AO_ADDR_W,
                   k_w=K_W, drain_cycles=DRAIN_CYCLES)
    ref.reset()
    await _reset(dut)

    for tile in range(2):
        base_ai = 10 + tile * 20
        base_ao = 30 + tile * 10
        k       = 2 + tile
        await _step(dut, ref, start=1, cfg_k=k,
                    cfg_ai_base=base_ai, cfg_ao_base=base_ao)
        _check(dut, ref)
        for _ in range(80):
            await _step(dut, ref)
            _check(dut, ref)
            if int(dut.done.value):
                break
        else:
            raise cocotb.result.TestFailure(f"tile {tile} did not complete")


@cocotb.test()
async def test_k_equals_one(dut):
    """Degenerate K=1 case: single activation vector, minimal EXECUTE phase."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = TileCtrl(n_rows=N_ROWS, n_cols=N_COLS,
                   ai_addr_w=AI_ADDR_W, ao_addr_w=AO_ADDR_W,
                   k_w=K_W, drain_cycles=DRAIN_CYCLES)
    ref.reset()
    await _reset(dut)

    await _step(dut, ref, start=1, cfg_k=1, cfg_ai_base=7, cfg_ao_base=3)
    _check(dut, ref)
    for _ in range(80):
        await _step(dut, ref)
        _check(dut, ref)
        if int(dut.done.value):
            break
    else:
        raise cocotb.result.TestFailure("K=1 tile did not complete")


@cocotb.test()
async def test_start_ignored_while_busy(dut):
    """Re-asserting start mid-tile must NOT restart or corrupt the config."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = TileCtrl(n_rows=N_ROWS, n_cols=N_COLS,
                   ai_addr_w=AI_ADDR_W, ao_addr_w=AO_ADDR_W,
                   k_w=K_W, drain_cycles=DRAIN_CYCLES)
    ref.reset()
    await _reset(dut)
    # First tile
    await _step(dut, ref, start=1, cfg_k=2, cfg_ai_base=11, cfg_ao_base=17)
    _check(dut, ref)
    # While busy, keep re-asserting start with different config — must be ignored.
    # Run until done; whole sequence is ~10+K ≈ 12 cycles with wide preload.
    saw_done = False
    for _ in range(40):
        await _step(dut, ref, start=1, cfg_k=99,
                    cfg_ai_base=0xFF, cfg_ao_base=0xFF)
        _check(dut, ref)
        if int(dut.done.value):
            saw_done = True
            break
    assert saw_done, "tile did not complete"


@cocotb.test()
async def test_acc_init_mode_load(dut):
    """Start with cfg_acc_init_mode=1 — EXEC_PREP pulses array_acc_load_valid
    (NOT array_clear_acc) and propagates cfg_acc_init_data."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = TileCtrl(n_rows=N_ROWS, n_cols=N_COLS,
                   ai_addr_w=AI_ADDR_W, ao_addr_w=AO_ADDR_W,
                   k_w=K_W, drain_cycles=DRAIN_CYCLES)
    ref.reset()
    await _reset(dut)

    # Build a wide init pattern (N_COLS × 32 bits).  Cols 0..N-1 each get
    # a distinct value so we can verify the whole packed word propagates.
    init_vals = [0x100 + n for n in range(N_COLS)]
    packed = 0
    for n, v in enumerate(init_vals):
        packed |= (v & 0xFFFFFFFF) << (n * 32)

    await _step(dut, ref, start=1, cfg_k=1, cfg_ai_base=0, cfg_ao_base=0,
                cfg_acc_init_mode=1, cfg_acc_init_data=packed)
    _check(dut, ref)
    # Run the whole tile and confirm:
    #  - array_clear_acc never pulses (mode=1 takes the load path)
    #  - array_acc_load_valid pulses exactly once in EXEC_PREP
    #  - array_acc_load_data on that cycle equals the init pattern
    clear_pulses = 0
    load_pulses = 0
    seen_load_data = None
    for _ in range(40):
        await _step(dut, ref)
        _check(dut, ref)
        if int(dut.array_clear_acc.value):
            clear_pulses += 1
        if int(dut.array_acc_load_valid.value):
            load_pulses += 1
            seen_load_data = int(dut.array_acc_load_data.value)
        if int(dut.done.value):
            break
    assert clear_pulses == 0, f"clear_acc should not pulse: {clear_pulses}"
    assert load_pulses == 1, f"acc_load_valid should pulse once: {load_pulses}"
    assert seen_load_data == packed, (
        f"acc_load_data mismatch: 0x{seen_load_data:x} vs 0x{packed:x}")


@cocotb.test()
async def test_afu_in_valid_pulse_timing(dut):
    """Gap #3: array_afu_in_valid pulses exactly once per tile, on the
    cycle the FSM transitions DRAIN → STORE.  npu_top's writeback AFUs
    use this pulse to latch the final activated c_vec."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    ref = TileCtrl(n_rows=N_ROWS, n_cols=N_COLS,
                   ai_addr_w=AI_ADDR_W, ao_addr_w=AO_ADDR_W,
                   k_w=K_W, drain_cycles=DRAIN_CYCLES)
    ref.reset()
    await _reset(dut)

    await _step(dut, ref, start=1, cfg_k=3, cfg_ai_base=0, cfg_ao_base=0)
    _check(dut, ref)

    pulses = 0
    pulse_followed_by_store = False
    for _ in range(60):
        await _step(dut, ref)
        _check(dut, ref)
        if int(dut.array_afu_in_valid.value):
            pulses += 1
            # After this clock edge, ao_we should pulse on the NEXT cycle.
            await _step(dut, ref)
            _check(dut, ref)
            if int(dut.ao_we.value):
                pulse_followed_by_store = True
        if int(dut.done.value):
            break
    assert pulses == 1, f"afu_in_valid should pulse exactly once: {pulses}"
    assert pulse_followed_by_store, "ao_we must follow afu_in_valid by 1 cycle"
