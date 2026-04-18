"""End-to-end smoke: SRAM controller + systolic array compute one matmul tile.

Proves the datapath works:
  DMA writes weights → array reads via SRAM → array loads internal grid
  DMA writes activations → array reads → array multiplies → outputs captured

For a first integration pass we stick to a 4x4 tile with identity weights,
then verify the output vector equals the input activation vector (identity
matmul).  A second test uses a known non-trivial weight matrix so bit-exact
correctness is actually checked, not just structure.
"""

import os
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))


DATA_W = 8
ACC_W = 32
N_ROWS = 4
N_COLS = 4
DEPTH = 16


def _pack(vec, width):
    out = 0
    for i, v in enumerate(vec):
        out |= (v & ((1 << width) - 1)) << (i * width)
    return out


def _unpack_cvec(val):
    out = []
    for i in range(N_COLS):
        raw = (val >> (i * ACC_W)) & ((1 << ACC_W) - 1)
        if raw & (1 << (ACC_W - 1)):
            raw -= 1 << ACC_W
        out.append(raw)
    return out


def _mask(v, w):
    return v & ((1 << w) - 1)


async def _clear(dut):
    for s in (
        "w_bank_sel", "w_we", "w_waddr", "w_wdata",
        "ai_we", "ai_waddr", "ai_wdata",
        "array_load_valid", "array_load_sram_addr", "array_load_cell_addr",
        "array_clear_acc", "array_exec_valid", "array_exec_ai_addr",
        "ao_wb_enable", "ao_wb_addr", "ao_re", "ao_raddr",
    ):
        getattr(dut, s).value = 0


async def _reset(dut):
    dut.rst_n.value = 0
    await _clear(dut)
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def _write_weights_to_wb(dut, W_flat):
    """DMA-side: write W_flat[0..N-1] into SRAM WB with sel=0.

    With w_bank_sel=0, w_we=1 routes the write into physical bank B.
    """
    dut.w_bank_sel.value = 0
    for addr, wd in enumerate(W_flat):
        dut.w_we.value    = 1
        dut.w_waddr.value = addr
        dut.w_wdata.value = wd & 0xFF
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
    dut.w_we.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def _write_ai_vector(dut, addr, vec):
    dut.ai_we.value    = 1
    dut.ai_waddr.value = addr
    dut.ai_wdata.value = _pack(vec, DATA_W)
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.ai_we.value = 0


async def _load_array_from_wb(dut):
    """Flip sel=1 and walk 0..N_ROWS-1 through wide weight-ROW reads.

    Cycle N: array_load_valid=1, sram_addr=row, cell_addr=row → SRAM reads
        WB full row (N_COLS weights packed).
    Cycle N+1: sram_w_rdata = WB[row, :], w_load_r=1 → array latches the
        full row into W[row][:].
    """
    dut.w_bank_sel.value = 1           # array reads WB
    for row in range(N_ROWS):
        dut.array_load_valid.value     = 1
        dut.array_load_sram_addr.value = row
        dut.array_load_cell_addr.value = row
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
    # One more cycle to flush the last latched row into the array
    dut.array_load_valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def _exec_activation(dut, ai_addr):
    """Drive one cycle of execute from AI[ai_addr]; array output appears 2 cycles later."""
    dut.array_exec_valid.value    = 1
    dut.array_exec_ai_addr.value  = ai_addr
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.array_exec_valid.value = 0
    # Give the pipeline 2 cycles to produce c_valid
    for _ in range(3):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")


def _expected_matmul(W, a):
    """Python ground-truth: c[n] = sum_k W_row_major[k*N_COLS+n] * a[k].

    Caller may pass negative Python ints directly; we mask to 8 bits and
    re-interpret as signed INT8 to match what the RTL sees after the
    `_pack` step feeds the signal through the bus.
    """
    def _as_int8(v: int) -> int:
        v &= 0xFF
        return v - 0x100 if v & 0x80 else v

    out = [0] * N_COLS
    for n in range(N_COLS):
        s = 0
        for k in range(N_ROWS):
            s += _as_int8(W[k * N_COLS + n]) * _as_int8(a[k])
        out[n] = s
    return out


@cocotb.test()
async def test_identity_matmul(dut):
    """Identity 4x4 weights × activation vector = activation vector (zero-extended)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)

    # Identity weights (row-major): W[k][n] = 1 if k==n else 0
    W_flat = [1 if (addr // N_COLS) == (addr % N_COLS) else 0
              for addr in range(N_ROWS * N_COLS)]
    a = [7, -3, 11, -5]

    await _write_weights_to_wb(dut, W_flat)
    await _write_ai_vector(dut, 0, a)
    await _load_array_from_wb(dut)
    await _exec_activation(dut, 0)

    c = _unpack_cvec(int(dut.array_c_vec.value))
    expected = _expected_matmul(W_flat, a)
    assert c == expected, f"c={c}  expected={expected}"
    dut._log.info(f"identity matmul PASS: c={c}")


@cocotb.test()
async def test_nontrivial_matmul(dut):
    """Pre-chosen non-identity weights and activation — full bit-exact check."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)

    # Weights in a clearly asymmetric pattern so any routing bug shows
    W_flat = [1, 2, 3, 4,
              5, 6, 7, 8,
              -1, -2, -3, -4,
              9, 10, 11, 12]
    a = [2, -1, 3, 1]

    await _write_weights_to_wb(dut, W_flat)
    await _write_ai_vector(dut, 0, a)
    await _load_array_from_wb(dut)
    await _exec_activation(dut, 0)

    c = _unpack_cvec(int(dut.array_c_vec.value))
    expected = _expected_matmul(W_flat, a)
    assert c == expected, f"c={c}  expected={expected}"
    dut._log.info(f"nontrivial matmul PASS: c={c}")


@cocotb.test()
async def test_two_activation_accumulation(dut):
    """Two activation vectors in succession accumulate into c_vec (no clear between)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)

    W_flat = [1, 2, 3, 4,
              0, 1, 0, 1,
              2, 0, 2, 0,
              1, 1, 1, 1]
    a0 = [1, 2, 3, 4]
    a1 = [-1, 0, 1, 2]

    await _write_weights_to_wb(dut, W_flat)
    await _write_ai_vector(dut, 0, a0)
    await _write_ai_vector(dut, 1, a1)
    await _load_array_from_wb(dut)

    # First execute
    await _exec_activation(dut, 0)
    c0 = _unpack_cvec(int(dut.array_c_vec.value))
    # Second execute accumulates on top of the first (no clear)
    await _exec_activation(dut, 1)
    c1 = _unpack_cvec(int(dut.array_c_vec.value))

    e0 = _expected_matmul(W_flat, a0)
    e1 = [e0[n] + v for n, v in enumerate(_expected_matmul(W_flat, a1))]
    assert c0 == e0, f"after a0: {c0} vs {e0}"
    assert c1 == e1, f"after a0+a1: {c1} vs {e1}"
    dut._log.info(f"accumulation PASS: a0={c0}, a0+a1={c1}")
