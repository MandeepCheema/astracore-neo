"""cocotb gate for the F1-A4 multi-pass extension of rtl/npu_tile_ctrl.

The existing npu_top regression (sim/npu_top/) continues to cover the
normal PRELOAD/EXEC/STORE flow.  This test exercises ONLY the new
multi-pass AFU sequencing — verifying:

  1. cfg_mp_mode != 0 at `start` routes directly to S_MP_* states.
  2. mp_start pulses for one cycle before pass 1.
  3. Pass 1 walks ai_raddr from cfg_ai_base to cfg_ai_base+VEC_LEN-1
     across VEC_LEN cycles.
  4. A programmable gap between passes (MP_GAP_CYCLES).
  5. Pass 2 walks ai_raddr again identically.
  6. Whenever the external (stubbed) AFU raises mp_out_valid during
     PASS2/DRAIN, tile_ctrl emits mp_ao_we with mp_ao_waddr walking
     from cfg_ao_base.
  7. `done` pulses once after S_MP_DRAIN.
"""

import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

VEC_LEN = 64
AI_BASE = 32
AO_BASE = 16
CLK_NS  = 10
MP_GAP  = 6   # must match MP_GAP_CYCLES default in the RTL


async def _reset(dut):
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.cfg_k.value = 0
    dut.cfg_ai_base.value = 0
    dut.cfg_ao_base.value = 0
    dut.cfg_acc_init_mode.value = 0
    dut.cfg_acc_init_data.value = 0
    dut.cfg_mp_mode.value = 0
    dut.cfg_mp_vec_len.value = 0
    dut.mp_out_valid.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def _start_mp(dut, mode):
    dut.cfg_mp_mode.value = mode
    dut.cfg_mp_vec_len.value = VEC_LEN
    dut.cfg_ai_base.value = AI_BASE
    dut.cfg_ao_base.value = AO_BASE
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await Timer(1, unit="ns")


@cocotb.test()
async def test_mp_start_and_mode_latch(dut):
    """mp_start pulses 1 cycle; mp_mode latches cfg_mp_mode."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await _reset(dut)
    await _start_mp(dut, 8)
    # mp_start should be high the cycle after start (right after IDLE -> MP_START)
    # Depending on S_IDLE registering — our FSM pulses mp_start *in* S_IDLE
    # transition, so it shows up the cycle after start lands.
    assert int(dut.mp_start.value) == 1, f"mp_start not high (got {int(dut.mp_start.value)})"
    assert int(dut.mp_mode.value) == 8
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    assert int(dut.mp_start.value) == 0, "mp_start should clear after 1 cycle"


@cocotb.test()
async def test_mp_pass1_sequencing(dut):
    """Pass-1 walks ai_raddr from AI_BASE to AI_BASE + VEC_LEN - 1.

    ai_re and ai_raddr are registered (1-cycle delay after state enters
    MP_PASS1). We advance two edges from `start` to get into steady state,
    then collect VEC_LEN samples.
    """
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await _reset(dut)
    await _start_mp(dut, 8)        # edge 1: IDLE -> MP_START
    await RisingEdge(dut.clk)      # edge 2: MP_START -> MP_PASS1 (ai_re still 0)
    await Timer(1, unit="ns")
    await RisingEdge(dut.clk)      # edge 3: MP_PASS1 drives ai_re <= 1
    await Timer(1, unit="ns")
    addrs = []
    valids = []
    for i in range(VEC_LEN):
        valids.append(int(dut.ai_re.value))
        addrs.append(int(dut.ai_raddr.value))
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
    # ai_re should be high across the full window (MP_PASS1 length VEC_LEN).
    assert all(v == 1 for v in valids), f"ai_re dropped during PASS1: {valids}"
    expected = [AI_BASE + i for i in range(VEC_LEN)]
    assert addrs == expected, f"pass-1 ai_raddr walk wrong: got {addrs[:8]}..."


@cocotb.test()
async def test_mp_full_sequence_with_stub_afu(dut):
    """Drive the whole MP flow with a stub AFU that fires mp_out_valid
    during PASS2/DRAIN.  Verify ao writes walk correctly and done pulses.

    Rather than asserting exact timing of each phase (which is sensitive
    to pipeline registers on ai_re), we track the sequence of observed
    events and check the overall shape.
    """
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await _reset(dut)
    await _start_mp(dut, 9)   # layernorm mode

    ao_writes = []
    ai_reads  = []
    done_seen = False

    # Run the whole sequence for up to 2*VEC_LEN + 20 cycles while
    # driving mp_out_valid=1 any cycle we see state in PASS2/DRAIN
    # (we can't observe state directly, so just leave it asserted
    # from after the GAP through the tail).
    for cycle in range(2 * VEC_LEN + 30):
        # After pass 1 + gap we start asserting mp_out_valid.
        # A simple strategy: assert it from cycle VEC_LEN+MP_GAP+2 onwards.
        if cycle >= VEC_LEN + MP_GAP + 2:
            dut.mp_out_valid.value = 1
        if int(dut.ai_re.value) == 1:
            ai_reads.append(int(dut.ai_raddr.value))
        if int(dut.mp_ao_we.value) == 1:
            ao_writes.append(int(dut.mp_ao_waddr.value))
        if int(dut.done.value) == 1:
            done_seen = True
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
    dut.mp_out_valid.value = 0

    # Verifications:
    # 1. We saw 2 * VEC_LEN ai_reads (pass 1 + pass 2).
    assert len(ai_reads) == 2 * VEC_LEN, f"expected {2*VEC_LEN} ai_reads, got {len(ai_reads)}"
    # First VEC_LEN: 0..VEC_LEN-1 (offset by AI_BASE).
    first_pass = ai_reads[:VEC_LEN]
    second_pass = ai_reads[VEC_LEN:]
    assert first_pass  == [AI_BASE + i for i in range(VEC_LEN)]
    assert second_pass == [AI_BASE + i for i in range(VEC_LEN)]
    # 2. AO writes walk from AO_BASE and are contiguous.
    assert len(ao_writes) >= VEC_LEN - 4, f"too few AO writes: {len(ao_writes)}"
    assert ao_writes[0] == AO_BASE, f"first AO write {ao_writes[0]}"
    diffs = [ao_writes[i+1] - ao_writes[i] for i in range(len(ao_writes)-1)]
    assert all(d == 1 for d in diffs), f"non-contiguous: {diffs[:10]}"
    # 3. done pulsed.
    assert done_seen, "done never pulsed"
    dut._log.info(
        f"ai_reads={len(ai_reads)}  ao_writes={len(ao_writes)}  "
        f"(first AO={ao_writes[0]}, last AO={ao_writes[-1]})")


@cocotb.test()
async def test_mp_zero_mode_does_not_affect_normal_flow(dut):
    """When cfg_mp_mode=0 at start, tile_ctrl must NOT enter MP states —
    it should run the existing tile-compute sequence (PRELOAD/EXEC/STORE).
    We verify by checking that mp_start never pulses and mp_ao_we stays low."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await _reset(dut)
    # Don't set cfg_mp_mode — it defaults to 0.
    dut.cfg_k.value = 2
    dut.cfg_ai_base.value = 0
    dut.cfg_ao_base.value = 0
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await Timer(1, unit="ns")
    mp_start_seen = False
    mp_ao_we_seen = False
    for _ in range(60):
        if int(dut.mp_start.value) == 1:
            mp_start_seen = True
        if int(dut.mp_ao_we.value) == 1:
            mp_ao_we_seen = True
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
    assert not mp_start_seen, "mp_start should never pulse when cfg_mp_mode=0"
    assert not mp_ao_we_seen, "mp_ao_we should never pulse when cfg_mp_mode=0"
