"""
AstraCore Neo — TMR Voter cocotb testbench.

Python TMRVoter is the GOLDEN REFERENCE.
The Verilog DUT implements majority voting for 32-bit values.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from safety.tmr import TMRVoter
from safety.exceptions import TMRError


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.valid.value = 0
    dut.lane_a.value = 0
    dut.lane_b.value = 0
    dut.lane_c.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def vote(dut, ref: TMRVoter, a: int, b: int, c: int):
    """Drive one vote into DUT and Python reference. Returns (dut_outputs, ref_result)."""
    dut.lane_a.value = a & 0xFFFFFFFF
    dut.lane_b.value = b & 0xFFFFFFFF
    dut.lane_c.value = c & 0xFFFFFFFF
    dut.valid.value  = 1
    await RisingEdge(dut.clk)
    dut.valid.value  = 0
    await RisingEdge(dut.clk)

    ref_result = None
    ref_triple = False
    try:
        ref_result = ref.vote(a, b, c)
    except TMRError:
        ref_triple = True

    return ref_result, ref_triple


@cocotb.test()
async def test_all_agree(dut):
    """All three lanes equal → voted = any lane, agreement = 1, no faults."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = TMRVoter()
    result, _ = await vote(dut, ref, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF)

    assert int(dut.voted.value)       == 0xDEADBEEF, f"voted={hex(dut.voted.value)}"
    assert int(dut.agreement.value)   == 1
    assert int(dut.fault_a.value)     == 0
    assert int(dut.fault_b.value)     == 0
    assert int(dut.fault_c.value)     == 0
    assert int(dut.triple_fault.value)== 0
    assert int(dut.vote_count.value)  == 3
    dut._log.info("all_agree test passed")


@cocotb.test()
async def test_lane_c_fault(dut):
    """A == B != C → voted = A, fault_c = 1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = TMRVoter()
    result, _ = await vote(dut, ref, 42, 42, 43)

    assert result.voted_value  == 42
    assert result.faulty_lane  == "C"
    assert int(dut.voted.value)    == 42, f"voted={dut.voted.value}"
    assert int(dut.fault_c.value)  == 1
    assert int(dut.fault_a.value)  == 0
    assert int(dut.fault_b.value)  == 0
    assert int(dut.agreement.value)== 1
    assert int(dut.vote_count.value)== 2
    dut._log.info("lane_c_fault test passed")


@cocotb.test()
async def test_lane_b_fault(dut):
    """A == C != B → voted = A, fault_b = 1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = TMRVoter()
    result, _ = await vote(dut, ref, 100, 200, 100)

    assert result.voted_value == 100
    assert result.faulty_lane == "B"
    assert int(dut.voted.value)    == 100
    assert int(dut.fault_b.value)  == 1
    assert int(dut.fault_a.value)  == 0
    assert int(dut.fault_c.value)  == 0
    dut._log.info("lane_b_fault test passed")


@cocotb.test()
async def test_lane_a_fault(dut):
    """B == C != A → voted = B, fault_a = 1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = TMRVoter()
    result, _ = await vote(dut, ref, 999, 500, 500)

    assert result.voted_value == 500
    assert result.faulty_lane == "A"
    assert int(dut.voted.value)    == 500
    assert int(dut.fault_a.value)  == 1
    assert int(dut.fault_b.value)  == 0
    assert int(dut.fault_c.value)  == 0
    dut._log.info("lane_a_fault test passed")


@cocotb.test()
async def test_triple_fault(dut):
    """All three lanes differ → triple_fault = 1, agreement = 0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = TMRVoter()
    _, ref_triple = await vote(dut, ref, 1, 2, 3)

    assert ref_triple is True
    assert int(dut.triple_fault.value) == 1
    assert int(dut.agreement.value)    == 0
    dut._log.info("triple_fault test passed")


@cocotb.test()
async def test_zero_value_vote(dut):
    """All lanes zero → voted = 0, all agree."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = TMRVoter()
    result, _ = await vote(dut, ref, 0, 0, 0)

    assert int(dut.voted.value)    == 0
    assert int(dut.agreement.value)== 1
    assert int(dut.vote_count.value)== 3
    dut._log.info("zero_value_vote test passed")


@cocotb.test()
async def test_max_value_vote(dut):
    """All lanes = 0xFFFFFFFF → voted = 0xFFFFFFFF."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = TMRVoter()
    result, _ = await vote(dut, ref, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)

    assert int(dut.voted.value) == 0xFFFFFFFF
    dut._log.info("max_value_vote test passed")


@cocotb.test()
async def test_sequence_vote_reference_match(dut):
    """Drive a mixed sequence of agree/disagree votes and verify DUT matches ref."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = TMRVoter()
    cases = [
        (100, 100, 100),     # all agree
        (200, 200, 201),     # C fault
        (300, 301, 300),     # B fault
        (401, 400, 400),     # A fault
        (10,  10,  10),      # all agree
        (0,   0,   0),       # all zero
    ]

    for a, b, c in cases:
        result, triple = await vote(dut, ref, a, b, c)
        if not triple:
            assert int(dut.voted.value)    == result.voted_value, (
                f"({a},{b},{c}): DUT voted={dut.voted.value} REF={result.voted_value}"
            )
            assert int(dut.agreement.value)== int(result.agreement)
        else:
            assert int(dut.triple_fault.value) == 1

    dut._log.info("sequence test passed")


@cocotb.test()
async def test_reset_clears_outputs(dut):
    """After reset, all outputs should be 0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = TMRVoter()
    await vote(dut, ref, 0xDEAD, 0xBEEF, 0xCAFE)  # triple fault case

    # Reset
    dut.rst_n.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    assert int(dut.voted.value)       == 0
    assert int(dut.agreement.value)   == 0
    assert int(dut.triple_fault.value)== 0
    dut._log.info("reset_clears_outputs test passed")
