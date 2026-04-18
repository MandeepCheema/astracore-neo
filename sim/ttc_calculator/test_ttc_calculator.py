"""
AstraCore Neo — TTC Calculator cocotb testbench

Verifies the multiply-compare TTC decision logic:
  TTC_ms = range_mm * 1000 / (-closure_mms)
  warning/prepare/brake fire when TTC < threshold AND object approaching.

Defaults: WARN=3000ms, PREP=1500ms, BRAKE=700ms.

Timing: 2-stage pipeline.
  Stage 1 registers the 4 multiplier outputs (DSP48E1 MREG slot).
  Stage 2 compares against each threshold and registers the flags.
obj_valid on EDGE A → stage 1 visible on EDGE B → ttc_valid on EDGE C.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


WARN_MS  = 3000
PREP_MS  = 1500
BRAKE_MS = 700


async def reset_dut(dut):
    dut.rst_n.value            = 0
    dut.obj_valid.value        = 0
    dut.obj_track_id.value     = 0
    dut.obj_range_mm.value     = 0
    dut.obj_closure_mms.value  = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def compute_ttc(dut, range_mm, closure_mms, track_id=1):
    """Fire a 1-cycle obj_valid; read result after 2-stage pipeline settles."""
    dut.obj_track_id.value    = track_id
    dut.obj_range_mm.value    = range_mm    & 0xFFFFFFFF
    dut.obj_closure_mms.value = closure_mms & 0xFFFFFFFF
    dut.obj_valid.value       = 1
    await RisingEdge(dut.clk)   # EDGE A: sampled
    dut.obj_valid.value       = 0
    await RisingEdge(dut.clk)   # EDGE B: stage 1 visible
    await RisingEdge(dut.clk)   # EDGE C: stage 2 visible — ttc_* readable


def expected_ttc_ms(range_mm, closure_mms):
    """Python reference model; returns None if not approaching."""
    if closure_mms >= 0:
        return None
    return (range_mm * 1000) // (-closure_mms)


# ===========================================================================
# Tests
# ===========================================================================

@cocotb.test()
async def test_initial_state(dut):
    """After reset: no valid, all flags cleared."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    assert int(dut.ttc_valid.value)       == 0
    assert int(dut.ttc_warning.value)     == 0
    assert int(dut.ttc_prepare.value)     == 0
    assert int(dut.ttc_brake.value)       == 0
    assert int(dut.ttc_approaching.value) == 0
    dut._log.info("initial_state passed")


@cocotb.test()
async def test_object_moving_away_no_flags(dut):
    """closure > 0 (object moving away) → no flags, no threat."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # range 5m, closure +20 m/s (object opening up)
    await compute_ttc(dut, range_mm=5000, closure_mms=20000)

    assert int(dut.ttc_valid.value)       == 1
    assert int(dut.ttc_approaching.value) == 0
    assert int(dut.ttc_warning.value)     == 0
    assert int(dut.ttc_prepare.value)     == 0
    assert int(dut.ttc_brake.value)       == 0
    dut._log.info("object_moving_away passed")


@cocotb.test()
async def test_stationary_object_no_flags(dut):
    """closure == 0 → not approaching → no flags."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await compute_ttc(dut, range_mm=5000, closure_mms=0)
    assert int(dut.ttc_approaching.value) == 0
    assert int(dut.ttc_warning.value)     == 0
    dut._log.info("stationary_object passed")


@cocotb.test()
async def test_far_approaching_object_only_warning(dut):
    """TTC ~5s: above all thresholds → no flags."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # range 100m, closing at 20 m/s → TTC = 5000 ms
    await compute_ttc(dut, range_mm=100_000, closure_mms=-20_000)
    ttc = expected_ttc_ms(100_000, -20_000)
    assert ttc == 5000
    assert int(dut.ttc_approaching.value) == 1
    assert int(dut.ttc_warning.value) == 0, "5s TTC > 3s warn threshold"
    dut._log.info(f"far_approaching passed: ttc={ttc}ms, no flags")


@cocotb.test()
async def test_warning_threshold_fires(dut):
    """TTC ~2.5s: below WARN (3s), above PREP (1.5s)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # range 50m, closing at 20 m/s → TTC = 2500 ms
    await compute_ttc(dut, range_mm=50_000, closure_mms=-20_000)
    ttc = expected_ttc_ms(50_000, -20_000)
    assert ttc == 2500
    assert int(dut.ttc_warning.value) == 1, "2.5s TTC should fire warning"
    assert int(dut.ttc_prepare.value) == 0, "2.5s TTC should not fire prepare"
    assert int(dut.ttc_brake.value)   == 0
    dut._log.info(f"warning_threshold_fires passed: ttc={ttc}ms")


@cocotb.test()
async def test_prepare_threshold_fires(dut):
    """TTC ~1s: below PREP (1.5s) and WARN, above BRAKE (0.7s)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # range 20m, closing at 20 m/s → TTC = 1000 ms
    await compute_ttc(dut, range_mm=20_000, closure_mms=-20_000)
    assert int(dut.ttc_warning.value) == 1
    assert int(dut.ttc_prepare.value) == 1
    assert int(dut.ttc_brake.value)   == 0
    dut._log.info("prepare_threshold_fires passed")


@cocotb.test()
async def test_brake_threshold_fires(dut):
    """TTC ~0.5s: below all three thresholds → all flags fire."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # range 10m, closing at 20 m/s → TTC = 500 ms
    await compute_ttc(dut, range_mm=10_000, closure_mms=-20_000)
    assert int(dut.ttc_warning.value) == 1
    assert int(dut.ttc_prepare.value) == 1
    assert int(dut.ttc_brake.value)   == 1
    dut._log.info("brake_threshold_fires passed")


@cocotb.test()
async def test_threshold_hierarchy(dut):
    """brake ⊂ prepare ⊂ warning — brake never fires alone."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Sweep several ranges at fixed closure rate
    for range_mm in [100_000, 50_000, 20_000, 10_000, 5_000, 1_000]:
        await compute_ttc(dut, range_mm=range_mm, closure_mms=-20_000)
        w = int(dut.ttc_warning.value)
        p = int(dut.ttc_prepare.value)
        b = int(dut.ttc_brake.value)
        # Hierarchy: b⇒p, p⇒w
        if b:
            assert p and w, f"brake set but hierarchy broken @ range={range_mm}"
        if p:
            assert w, f"prepare set but warning not @ range={range_mm}"
    dut._log.info("threshold_hierarchy passed")


@cocotb.test()
async def test_slow_closure_still_warns_at_close_range(dut):
    """Slow closure but very close → still triggers thresholds."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # range 1m, closing at 1 m/s → TTC = 1000 ms → warn + prepare
    await compute_ttc(dut, range_mm=1000, closure_mms=-1000)
    assert int(dut.ttc_warning.value) == 1
    assert int(dut.ttc_prepare.value) == 1
    assert int(dut.ttc_brake.value)   == 0
    dut._log.info("slow_closure_close_range passed")


@cocotb.test()
async def test_track_id_echoed(dut):
    """obj_track_id is passed through to ttc_track_id."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for tid in [1, 42, 0xABCD, 0xFFFF]:
        await compute_ttc(dut, range_mm=5000, closure_mms=-10000, track_id=tid)
        assert int(dut.ttc_track_id.value) == tid, \
            f"track_id: got {int(dut.ttc_track_id.value)}, expected {tid}"
    dut._log.info("track_id_echoed passed")


@cocotb.test()
async def test_ttc_valid_deasserts(dut):
    """ttc_valid pulses for exactly 1 cycle, 2 clocks after obj_valid."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    dut.obj_track_id.value    = 1
    dut.obj_range_mm.value    = 5000
    dut.obj_closure_mms.value = (-10000) & 0xFFFFFFFF
    dut.obj_valid.value       = 1
    await RisingEdge(dut.clk)   # EDGE A: sampled, s1_valid scheduled
    dut.obj_valid.value       = 0
    assert int(dut.ttc_valid.value) == 0, "ttc_valid should be 0 at EDGE A"

    await RisingEdge(dut.clk)   # EDGE B: s1 visible, ttc_valid scheduled
    assert int(dut.ttc_valid.value) == 0, "ttc_valid should still be 0 at EDGE B (stage 1 only)"

    await RisingEdge(dut.clk)   # EDGE C: ttc_valid now visible
    assert int(dut.ttc_valid.value) == 1, "ttc_valid should pulse at EDGE C"

    await RisingEdge(dut.clk)   # EDGE D: default de-assert
    assert int(dut.ttc_valid.value) == 0, "ttc_valid should de-assert"
    dut._log.info("ttc_valid_deasserts passed")
