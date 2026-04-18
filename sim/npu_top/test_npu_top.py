"""End-to-end integration test for rtl/npu_top/npu_top.v.

Loads a weight tile + activation vectors via external write ports,
triggers `start`, waits for `done`, and reads back AO.  Verifies the
accumulated matmul result matches a Python ground truth exactly.

Covers:
  • Identity matmul — outputs equal sum of activation vectors, per column.
  • Non-trivial matmul — specific known weights + activations, exact.
  • K=1 (single activation) — simplest execute path.
  • Two consecutive tiles — busy/done handshake + config re-latching.
"""

import os
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))


DATA_W        = 8
ACC_W         = 32
N_ROWS        = 4
N_COLS        = 4
WEIGHT_DEPTH  = 16
ACT_IN_DEPTH  = 16
ACT_OUT_DEPTH = 16
AI_DATA_W     = N_ROWS * DATA_W
AO_DATA_W     = N_COLS * ACC_W


def _mask(v, w): return v & ((1 << w) - 1)


def _to_signed(val, width):
    val &= (1 << width) - 1
    if val & (1 << (width - 1)):
        return val - (1 << width)
    return val


def _pack_vec(vec, elem_w):
    out = 0
    for i, v in enumerate(vec):
        out |= (v & ((1 << elem_w) - 1)) << (i * elem_w)
    return out


def _unpack_cvec(val):
    return [_to_signed((val >> (i * ACC_W)) & ((1 << ACC_W) - 1), ACC_W)
            for i in range(N_COLS)]


def _as_int8(v: int) -> int:
    v &= 0xFF
    return v - 0x100 if v & 0x80 else v


def _expected_matmul_accumulated(W_flat, act_vecs):
    """Run K accumulated matmuls: c[n] = sum over all k in K, over r in N_ROWS,
    of W[r*N_COLS+n] * act_vecs[k][r]."""
    result = [0] * N_COLS
    for act in act_vecs:
        for n in range(N_COLS):
            for r in range(N_ROWS):
                result[n] += _as_int8(W_flat[r * N_COLS + n]) * _as_int8(act[r])
    return result


async def _reset(dut):
    dut.rst_n.value          = 0
    dut.start.value          = 0
    dut.cfg_k.value          = 0
    dut.cfg_ai_base.value    = 0
    dut.cfg_ao_base.value    = 0
    dut.cfg_afu_mode.value   = 0
    dut.cfg_acc_init_mode.value = 0
    dut.cfg_acc_init_data.value = 0
    dut.cfg_precision_mode.value = 0
    dut.ext_w_we.value       = 0
    dut.ext_w_waddr.value    = 0
    dut.ext_w_wdata.value    = 0
    dut.ext_ai_we.value      = 0
    dut.ext_ai_waddr.value   = 0
    dut.ext_ai_wdata.value   = 0
    dut.ext_ao_re.value      = 0
    dut.ext_ao_raddr.value   = 0
    dut.ext_sparse_skip_vec.value = 0
    # DMA inputs
    dut.dma_start.value          = 0
    dut.dma_cfg_src_addr.value   = 0
    dut.dma_cfg_ai_base.value    = 0
    dut.dma_cfg_tile_h.value     = 0
    dut.dma_cfg_src_stride.value = 0
    dut.mem_rdata.value          = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def _load_weights(dut, W_flat):
    for addr, w in enumerate(W_flat):
        dut.ext_w_we.value    = 1
        dut.ext_w_waddr.value = addr
        dut.ext_w_wdata.value = _mask(w, DATA_W)
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
    dut.ext_w_we.value = 0


async def _load_activation(dut, addr, vec):
    dut.ext_ai_we.value    = 1
    dut.ext_ai_waddr.value = addr
    dut.ext_ai_wdata.value = _pack_vec(vec, DATA_W)
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.ext_ai_we.value = 0


async def _run_tile(dut, k, ai_base, ao_base, afu_mode=0,
                    acc_init_mode=0, acc_init_data=0, precision_mode=0):
    """Trigger one tile and wait for done.  cfg_afu_mode and
    cfg_precision_mode are latched by npu_top on start."""
    dut.start.value        = 1
    dut.cfg_k.value        = k
    dut.cfg_ai_base.value  = ai_base
    dut.cfg_ao_base.value  = ao_base
    dut.cfg_afu_mode.value = afu_mode
    dut.cfg_precision_mode.value = precision_mode
    dut.cfg_acc_init_mode.value = acc_init_mode
    dut.cfg_acc_init_data.value = acc_init_data & ((1 << AO_DATA_W) - 1)
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.start.value = 0
    dut.cfg_acc_init_mode.value = 0
    dut.cfg_acc_init_data.value = 0
    for _ in range(500):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.done.value):
            break
    else:
        raise cocotb.result.TestFailure("npu_top did not complete tile")
    # Give one more cycle for writes to settle
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def _read_ao(dut, addr):
    dut.ext_ao_re.value    = 1
    dut.ext_ao_raddr.value = addr
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.ext_ao_re.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    return _unpack_cvec(int(dut.ext_ao_rdata.value))


# ---------------------------------------------------------------------------
@cocotb.test()
async def test_identity_matmul(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    # Identity W (4x4)
    W = [1 if (a // N_COLS) == (a % N_COLS) else 0
         for a in range(N_ROWS * N_COLS)]
    await _load_weights(dut, W)
    # One activation vector
    act = [7, -3, 11, -5]
    await _load_activation(dut, 0, act)
    # Run tile
    await _run_tile(dut, k=1, ai_base=0, ao_base=0)
    # Read AO
    result = await _read_ao(dut, 0)
    expected = _expected_matmul_accumulated(W, [act])
    assert result == expected, f"rtl={result} expected={expected}"
    dut._log.info(f"identity PASS: {result}")


@cocotb.test()
async def test_nontrivial_matmul(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    W = [1, 2, 3, 4,
         5, 6, 7, 8,
         -1, -2, -3, -4,
         9, 10, 11, 12]
    act = [2, -1, 3, 1]
    await _load_weights(dut, W)
    await _load_activation(dut, 0, act)
    await _run_tile(dut, k=1, ai_base=0, ao_base=0)
    result = await _read_ao(dut, 0)
    expected = _expected_matmul_accumulated(W, [act])
    assert result == expected, f"rtl={result} expected={expected}"
    dut._log.info(f"nontrivial PASS: {result}")


@cocotb.test()
async def test_k_equals_3_accumulation(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    W = [1, 0, 2, 0,
         0, 1, 0, 2,
         1, 1, 1, 1,
         -1, -1, -1, -1]
    acts = [
        [1, 2, 3, 4],
        [-1, 0, 1, 2],
        [2, -1, 3, 1],
    ]
    await _load_weights(dut, W)
    for i, a in enumerate(acts):
        await _load_activation(dut, i, a)
    await _run_tile(dut, k=3, ai_base=0, ao_base=0)
    result = await _read_ao(dut, 0)
    expected = _expected_matmul_accumulated(W, acts)
    assert result == expected, f"rtl={result} expected={expected}"
    dut._log.info(f"K=3 accum PASS: {result}")


def _pack_cvec_signed(vec):
    """Pack N_COLS signed acc values into AO_DATA_W-bit word."""
    out = 0
    for i, v in enumerate(vec):
        out |= (v & ((1 << ACC_W) - 1)) << (i * ACC_W)
    return out


@cocotb.test()
async def test_k_tile_chaining(dut):
    """Gap #2 Phase 2: software-managed k-tile chaining.

    Splits a K=4 matmul into two K=2 sub-tiles. The first runs with
    cfg_acc_init_mode=0 (clear at EXEC_PREP) and writes its partial sum
    to AO. The second runs with cfg_acc_init_mode=1, feeding that partial
    sum back via cfg_acc_init_data, and accumulates the remaining 2
    activations on top. The final result must match a single K=4 tile
    bit-for-bit — this proves partial sums can be carried across tiles
    for K larger than the scratch/SRAM can hold in one pass."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)

    W = [1, 2, 3, 4,
         5, 6, 7, 8,
         -1, -2, -3, -4,
         9, 10, 11, 12]
    acts = [
        [1, 2, 3, 4],
        [-1, 0, 1, 2],
        [2, -1, 3, 1],
        [-3, 4, -2, 1],
    ]
    await _load_weights(dut, W)
    for i, a in enumerate(acts):
        await _load_activation(dut, i, a)

    # Reference: single K=4 tile
    expected_full = _expected_matmul_accumulated(W, acts)

    # First sub-tile: K=2 over acts[0..1], clear-mode, writes to AO[0]
    await _run_tile(dut, k=2, ai_base=0, ao_base=0, acc_init_mode=0)
    partial = await _read_ao(dut, 0)
    exp_partial = _expected_matmul_accumulated(W, acts[:2])
    assert partial == exp_partial, f"partial mismatch rtl={partial} exp={exp_partial}"

    # Second sub-tile: K=2 over acts[2..3], load-mode with partial as init,
    # writes final to AO[1]
    init_data = _pack_cvec_signed(partial)
    await _run_tile(dut, k=2, ai_base=2, ao_base=1,
                    acc_init_mode=1, acc_init_data=init_data)
    final = await _read_ao(dut, 1)
    assert final == expected_full, (
        f"chained K=4 mismatch rtl={final} expected={expected_full} "
        f"(partial was {partial})")
    dut._log.info(f"k-tile chaining PASS: {final}")


AFU_MODE_PASS           = 0
AFU_MODE_RELU           = 1
AFU_MODE_LEAKY_RELU     = 2
AFU_MODE_CLIP_INT8      = 3
AFU_MODE_RELU_CLIP_INT8 = 4


def _apply_afu(vec, mode):
    def _one(v):
        if mode == AFU_MODE_PASS:
            return v
        if mode == AFU_MODE_RELU:
            return 0 if v < 0 else v
        if mode == AFU_MODE_LEAKY_RELU:
            return (v >> 3) if v < 0 else v  # arith shift == >>> on neg
        if mode == AFU_MODE_CLIP_INT8:
            return max(-128, min(127, v))
        if mode == AFU_MODE_RELU_CLIP_INT8:
            return 0 if v < 0 else min(127, v)
        raise ValueError(f"unknown mode {mode}")
    return [_one(v) for v in vec]


@cocotb.test()
async def test_relu_writeback(dut):
    """Gap #3: RELU on AO writeback zeroes negative columns."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    # Weights chosen so c[0], c[2] positive and c[1], c[3] negative.
    W = [ 1, -2,  3, -4,
          2, -3,  4, -5,
          3, -4,  5, -6,
          4, -5,  6, -7]
    act = [1, 1, 1, 1]
    await _load_weights(dut, W)
    await _load_activation(dut, 0, act)
    await _run_tile(dut, k=1, ai_base=0, ao_base=0,
                    afu_mode=AFU_MODE_RELU)
    result = await _read_ao(dut, 0)
    raw = _expected_matmul_accumulated(W, [act])
    expected = _apply_afu(raw, AFU_MODE_RELU)
    assert result == expected, (
        f"relu rtl={result} expected={expected} raw={raw}")
    # Sanity: raw has both negative and positive entries → test is meaningful
    assert any(v < 0 for v in raw) and any(v > 0 for v in raw), raw
    dut._log.info(f"relu writeback PASS raw={raw} activated={result}")


@cocotb.test()
async def test_clip_int8_writeback(dut):
    """Gap #3: CLIP_INT8 saturates accumulator values outside [-128, 127]."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    # Large positive and negative accumulator values.
    W = [ 127,  127, -128, -128,
          127,  127, -128, -128,
          127,  127, -128, -128,
          127,  127, -128, -128]
    act = [1, 1, 1, 1]
    await _load_weights(dut, W)
    await _load_activation(dut, 0, act)
    await _run_tile(dut, k=1, ai_base=0, ao_base=0,
                    afu_mode=AFU_MODE_CLIP_INT8)
    result = await _read_ao(dut, 0)
    raw = _expected_matmul_accumulated(W, [act])
    expected = _apply_afu(raw, AFU_MODE_CLIP_INT8)
    assert result == expected, (
        f"clip rtl={result} expected={expected} raw={raw}")
    # Sanity: raw must exceed INT8 range on at least one column
    assert any(v > 127 or v < -128 for v in raw), raw
    dut._log.info(f"clip_int8 writeback PASS raw={raw} clipped={result}")


@cocotb.test()
async def test_relu_clip_writeback(dut):
    """Gap #3: RELU_CLIP_INT8 combined — negatives → 0, positives clipped."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    # Mix of huge-positive and huge-negative columns.
    W = [ 127,  127, -128, -128,
          127,  127, -128, -128,
          127,  127, -128, -128,
          127,  127, -128, -128]
    act = [1, 1, 1, 1]
    await _load_weights(dut, W)
    await _load_activation(dut, 0, act)
    await _run_tile(dut, k=1, ai_base=0, ao_base=0,
                    afu_mode=AFU_MODE_RELU_CLIP_INT8)
    result = await _read_ao(dut, 0)
    raw = _expected_matmul_accumulated(W, [act])
    expected = _apply_afu(raw, AFU_MODE_RELU_CLIP_INT8)
    assert result == expected, (
        f"relu_clip rtl={result} expected={expected} raw={raw}")
    dut._log.info(f"relu_clip writeback PASS raw={raw} out={result}")


@cocotb.test()
async def test_afu_mode_latched_across_tile(dut):
    """Gap #3: cfg_afu_mode is sampled on start; changing it mid-tile must
    NOT affect the currently-running tile's activation."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    W = [ 1, -2,  3, -4,
          2, -3,  4, -5,
          3, -4,  5, -6,
          4, -5,  6, -7]
    act = [1, 1, 1, 1]
    await _load_weights(dut, W)
    await _load_activation(dut, 0, act)

    # Start with RELU, then corrupt the input port during busy.
    dut.start.value        = 1
    dut.cfg_k.value        = 1
    dut.cfg_ai_base.value  = 0
    dut.cfg_ao_base.value  = 0
    dut.cfg_afu_mode.value = AFU_MODE_RELU
    dut.cfg_acc_init_mode.value = 0
    dut.cfg_acc_init_data.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.start.value = 0
    # Corrupt mode mid-tile — must be ignored because it's latched.
    dut.cfg_afu_mode.value = AFU_MODE_CLIP_INT8
    for _ in range(500):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.done.value):
            break
    else:
        raise cocotb.result.TestFailure("tile did not complete")
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    result = await _read_ao(dut, 0)
    raw = _expected_matmul_accumulated(W, [act])
    expected_relu = _apply_afu(raw, AFU_MODE_RELU)
    assert result == expected_relu, (
        f"latched mode failed: rtl={result} relu_expected={expected_relu} "
        f"raw={raw}")
    dut._log.info(f"afu mode latched PASS result={result}")


@cocotb.test()
async def test_two_successive_tiles(dut):
    """Two separate tile runs with different weights; confirms end-to-end
    reset-less operation between tiles."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)

    # Tile 1
    W1 = [1 if (a // N_COLS) == (a % N_COLS) else 0
          for a in range(N_ROWS * N_COLS)]
    act1 = [5, 10, -2, 7]
    await _load_weights(dut, W1)
    await _load_activation(dut, 0, act1)
    await _run_tile(dut, k=1, ai_base=0, ao_base=0)
    result1 = await _read_ao(dut, 0)
    exp1 = _expected_matmul_accumulated(W1, [act1])
    assert result1 == exp1, f"tile1 mismatch rtl={result1} exp={exp1}"
    dut._log.info(f"tile1 PASS: {result1}")

    # Tile 2 — different weights, different activations, different AO addr
    W2 = [2, 0, 0, 0,
          0, 3, 0, 0,
          0, 0, 4, 0,
          0, 0, 0, 5]
    act2 = [1, 1, 1, 1]
    await _load_weights(dut, W2)
    await _load_activation(dut, 1, act2)
    await _run_tile(dut, k=1, ai_base=1, ao_base=3)
    result2 = await _read_ao(dut, 3)
    exp2 = _expected_matmul_accumulated(W2, [act2])
    assert result2 == exp2, f"tile2 mismatch rtl={result2} exp={exp2}"
    dut._log.info(f"tile2 PASS: {result2}")



@cocotb.test()
async def test_dma_loads_ai_bank(dut):
    """WP-9: DMA fetches bytes from a simulated DDR and the narrow-to-wide
    packer writes them into the AI bank as N_ROWS-byte rows. Then run a
    regular tile using AI data that came *entirely from the DMA path* —
    verifies the DDR → DMA → packer → AI → systolic loop end-to-end."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)

    # Load weights via the external port (not DMA in this test)
    W = [1, 2, 3, 4,
         5, 6, 7, 8,
         -1, -2, -3, -4,
         9, 10, 11, 12]
    await _load_weights(dut, W)

    # Set up a simulated DDR as a Python dict. The DMA presents mem_raddr
    # every cycle when mem_re=1; the testbench supplies mem_rdata one
    # cycle later.  We pre-populate so the DMA reads K activation vectors
    # worth of bytes starting at src address 0x1000.
    # Activation vectors we want in AI bank:
    act = [3, -7, 5, 2]   # will live at AI[0] after DMA run
    ddr = {}
    src_base = 0x1000
    # DMA tile layout: tile_h = 1 row, tile_w = N_ROWS bytes, stride unused.
    # DMA will emit N_ROWS sequential reads from src_base..src_base+N_ROWS-1.
    for r in range(N_ROWS):
        ddr[src_base + r] = act[r] & 0xFF

    # Memory responder — copies sim/npu_dma/test_npu_dma.py pattern exactly:
    # FallingEdge of cycle N samples mem_re/mem_raddr; RisingEdge of cycle
    # N+1 drives mem_rdata so DMA sees data on cycle N+1 (1-cycle latency).
    from cocotb.triggers import FallingEdge
    async def mem_responder():
        while True:
            await FallingEdge(dut.clk)
            captured = int(dut.mem_raddr.value) if int(dut.mem_re.value) else None
            await RisingEdge(dut.clk)
            if captured is not None:
                dut.mem_rdata.value = ddr.get(captured, 0) & 0xFF
            else:
                dut.mem_rdata.value = 0
    mem_task = cocotb.start_soon(mem_responder())

    # Kick the DMA: fetch 1 row of N_ROWS bytes starting at src_base, land
    # packed into AI[0].
    dut.dma_cfg_src_addr.value   = src_base
    dut.dma_cfg_ai_base.value    = 0
    dut.dma_cfg_tile_h.value     = 1
    dut.dma_cfg_src_stride.value = N_ROWS
    dut.dma_start.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.dma_start.value = 0
    # Wait for DMA done + one more cycle so the packer's NBA commits
    for _ in range(200):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.dma_done.value):
            break
    else:
        raise AssertionError("DMA did not complete")
    for _ in range(4):   # let packer's pack_ai_we commit + settle
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")

    mem_task.kill()

    # Now run a tile — AI bank should have the activation we DMA'd in
    await _run_tile(dut, k=1, ai_base=0, ao_base=0)
    result = await _read_ao(dut, 0)
    expected = _expected_matmul_accumulated(W, [act])
    assert result == expected, (
        f"DMA→AI→tile path mismatch: rtl={result} expected={expected}")
    dut._log.info(f"DMA load + tile exec PASS: {result}")


@cocotb.test()
async def test_precision_int4_end_to_end(dut):
    """WP-1 plumbing: drive npu_top with cfg_precision_mode=INT4 and
    verify the INT4 arithmetic is reachable through the whole chip top.

    Each weight / activation byte packs 2 INT4 values.  For a 4x4 array,
    that gives effectively 8 MACs of reduction per (row, col) pair per
    EXECUTE cycle (N_ROWS=4 rows * 2 INT4 per byte).  Compare the AO
    output against software emulation.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)

    # Weights: each byte = {w_hi: 4-bit signed, w_lo: 4-bit signed}.
    # Pick patterns with known INT4 decompositions.
    def pack_i4(hi, lo):
        return ((hi & 0xF) << 4) | (lo & 0xF)
    # Row-major [N_ROWS x N_COLS] of (hi, lo) pairs
    W_i4 = [
        [(1, 2), (-1, 3), (2, -1), (-2, 1)],
        [(3, 1), (1, -2), (-3, 0), (0, 2)],
        [(-1, -1), (2, 1), (1, 1), (-2, -2)],
        [(0, 3), (-2, 1), (1, -3), (3, 0)],
    ]
    W_flat = [pack_i4(h, l) for row in W_i4 for (h, l) in row]
    await _load_weights(dut, W_flat)

    # Activation: each byte = 2 INT4. Pick a known pattern.
    act_i4 = [(1, 2), (-1, 1), (2, -1), (-2, 2)]   # N_ROWS pairs
    act_bytes = [pack_i4(h, l) for (h, l) in act_i4]
    await _load_activation(dut, 0, act_bytes)

    await _run_tile(dut, k=1, ai_base=0, ao_base=0, precision_mode=0b01)

    result = await _read_ao(dut, 0)

    # Software ref: INT4 packed MAC per (k, n): w_hi*a_hi + w_lo*a_lo
    expected = [0] * N_COLS
    for n in range(N_COLS):
        for k in range(N_ROWS):
            wh, wl = W_i4[k][n]
            ah, al = act_i4[k]
            expected[n] += wh * ah + wl * al
    assert result == expected, (
        f"INT4 end-to-end: rtl={result} expected={expected}")
    dut._log.info(f"INT4 end-to-end PASS: {result}")


@cocotb.test()
async def test_precision_int2_end_to_end(dut):
    """WP-1 plumbing: INT2 mode through npu_top, 4 INT2x INT2 per byte."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)

    def pack_i2(q3, q2, q1, q0):
        return ((q3 & 0x3) << 6) | ((q2 & 0x3) << 4) | ((q1 & 0x3) << 2) | (q0 & 0x3)

    # INT2 values: 00=0, 01=1, 10=-2, 11=-1
    W_i2 = [
        [(1, -1, 1, 0), (0, 1, -1, -2), (1, 1, -1, 0), (-2, 0, 1, 1)],
        [(0, 1, -1, -2), (1, 0, 1, 0), (-1, -1, 0, 1), (0, 1, 1, -1)],
        [(1, 0, 1, -1), (-2, 0, 0, 1), (1, 1, 1, 1), (0, -1, 1, 0)],
        [(-1, 1, 0, 1), (0, 0, 0, 0), (1, -1, 1, -2), (1, 1, 0, 1)],
    ]
    W_flat = [pack_i2(*W_i2[r][c]) for r in range(N_ROWS) for c in range(N_COLS)]
    await _load_weights(dut, W_flat)

    act_i2 = [(1, 1, -1, 0), (0, -1, 1, 1), (1, 0, 1, -2), (-1, 1, 0, 1)]
    act_bytes = [pack_i2(*q) for q in act_i2]
    await _load_activation(dut, 0, act_bytes)

    await _run_tile(dut, k=1, ai_base=0, ao_base=0, precision_mode=0b10)

    result = await _read_ao(dut, 0)

    # Ref: each (k, n) slot has 4 INT2×INT2 products summed
    expected = [0] * N_COLS
    for n in range(N_COLS):
        for k in range(N_ROWS):
            for idx in range(4):
                expected[n] += W_i2[k][n][idx] * act_i2[k][idx]
    assert result == expected, (
        f"INT2 end-to-end: rtl={result} expected={expected}")
    dut._log.info(f"INT2 end-to-end PASS: {result}")


@cocotb.test()
async def test_precision_mode_latched_across_tile(dut):
    """cfg_precision_mode is latched on start; changing it mid-tile must
    NOT affect the currently-running tile's arithmetic."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)

    def pack_i4(hi, lo):
        return ((hi & 0xF) << 4) | (lo & 0xF)
    W_i4 = [[(1, 1), (2, 2), (3, 3), (4, 4)]] * 4   # simple
    W_flat = [pack_i4(h, l) for row in W_i4 for (h, l) in row]
    await _load_weights(dut, W_flat)
    act_i4 = [(1, 1)] * N_ROWS
    act_bytes = [pack_i4(h, l) for (h, l) in act_i4]
    await _load_activation(dut, 0, act_bytes)

    # Launch tile in INT4, then corrupt cfg_precision_mode mid-run
    dut.start.value = 1
    dut.cfg_k.value = 1
    dut.cfg_ai_base.value = 0
    dut.cfg_ao_base.value = 0
    dut.cfg_precision_mode.value = 0b01   # INT4
    dut.cfg_afu_mode.value = 0
    dut.cfg_acc_init_mode.value = 0
    dut.cfg_acc_init_data.value = 0
    await RisingEdge(dut.clk); await Timer(1, unit="ns")
    dut.start.value = 0
    dut.cfg_precision_mode.value = 0b10   # try to corrupt to INT2
    for _ in range(200):
        await RisingEdge(dut.clk); await Timer(1, unit="ns")
        if int(dut.done.value):
            break
    else:
        raise AssertionError("tile did not complete")
    await RisingEdge(dut.clk); await Timer(1, unit="ns")

    result = await _read_ao(dut, 0)
    # Expected: INT4 math, not INT2
    expected = [0] * N_COLS
    for n in range(N_COLS):
        for k in range(N_ROWS):
            wh, wl = W_i4[k][n]
            ah, al = act_i4[k]
            expected[n] += wh * ah + wl * al
    assert result == expected, (
        f"latched precision failed: rtl={result} expected={expected}")
    dut._log.info(f"precision_mode latch PASS: {result}")


@cocotb.test()
async def test_sparse_skip_zeros_products(dut):
    """WP-1 2:4-skip plumbing: driving ext_sparse_skip_vec=1 for a row
    should zero that row's contribution to every column's accumulator.
    Same workload as test_identity_matmul but with row-3 skipped → c[3]
    must be 0 instead of act[3]."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    W = [1 if (a // N_COLS) == (a % N_COLS) else 0
         for a in range(N_ROWS * N_COLS)]
    await _load_weights(dut, W)
    act = [7, -3, 11, -5]
    await _load_activation(dut, 0, act)

    # Drive sparse_skip_vec = 0b1000 during the EXECUTE cycle (row 3 pruned).
    # npu_top pipelines it 1 cycle before feeding the array, so set it
    # and keep it asserted for the duration of the tile.
    dut.ext_sparse_skip_vec.value = 0b1000
    await _run_tile(dut, k=1, ai_base=0, ao_base=0)
    dut.ext_sparse_skip_vec.value = 0

    result = await _read_ao(dut, 0)
    # Identity W → out[n] = act[n]; with row 3 skipped, out[3] = 0.
    expected = [act[0], act[1], act[2], 0]
    assert result == expected, (
        f"sparse_skip: rtl={result} expected={expected}")
    dut._log.info(f"sparse_skip_vec zeros row-3 contribution PASS: {result}")
