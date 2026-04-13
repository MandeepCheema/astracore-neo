"""
AstraCore Neo — PCIe Controller cocotb testbench.

Validates the PCIe link state machine and TLP header assembly.

TLP type encoding:
  2'd0  MEM_READ   (fmt=2'b00, type=5'h00)
  2'd1  MEM_WRITE  (fmt=2'b10, type=5'h00)
  2'd2  CPL_DATA   (fmt=2'b10, type=5'h0A)

Link state encoding:
  3'd0 DETECT, 3'd1 POLLING, 3'd2 CONFIG, 3'd3 L0, 3'd4 L1, 3'd5 L2
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from connectivity.pcie import PCIeLinkState, TLPType


async def reset_dut(dut):
    dut.rst_n.value   = 0
    dut.link_up.value = 0
    dut.link_down.value = 0
    dut.tlp_start.value = 0
    dut.tlp_type.value  = 0
    dut.req_id.value    = 0
    dut.tag.value       = 0
    dut.addr.value      = 0
    dut.length_dw.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def train_link_to_l0(dut):
    """Drive link_up three times to advance DETECT→POLLING→CONFIG→L0."""
    for _ in range(3):
        dut.link_up.value = 1
        await RisingEdge(dut.clk)
        dut.link_up.value = 0
        await RisingEdge(dut.clk)


async def send_tlp(dut, tlp_type: int, req_id: int, tag: int, addr: int, length_dw: int):
    """Initiate TLP assembly and wait for done. Returns assembled header."""
    dut.tlp_type.value  = tlp_type
    dut.req_id.value    = req_id & 0xFFFF
    dut.tag.value       = tag & 0xFF
    dut.addr.value      = addr & 0xFFFFFFFF
    dut.length_dw.value = length_dw & 0x3FF
    dut.tlp_start.value = 1
    await RisingEdge(dut.clk)
    dut.tlp_start.value = 0

    # Wait for done (max 10 cycles)
    for _ in range(10):
        await RisingEdge(dut.clk)
        if dut.tlp_done.value:
            return int(dut.tlp_hdr.value)

    raise TimeoutError("TLP done not received within 10 cycles")


@cocotb.test()
async def test_initial_link_state(dut):
    """After reset, link_state should be DETECT (3'd0)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    assert int(dut.link_state.value) == 0, (
        f"Expected DETECT(0), got {dut.link_state.value}"
    )
    dut._log.info("initial_link_state test passed")


@cocotb.test()
async def test_link_training_sequence(dut):
    """Link advances DETECT→POLLING→CONFIG→L0 with three link_up pulses."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    expected_states = [1, 2, 3]   # POLLING, CONFIG, L0
    for expected in expected_states:
        dut.link_up.value = 1
        await RisingEdge(dut.clk)
        dut.link_up.value = 0
        await RisingEdge(dut.clk)
        assert int(dut.link_state.value) == expected, (
            f"Expected state={expected}, got {dut.link_state.value}"
        )

    dut._log.info("link_training_sequence test passed")


@cocotb.test()
async def test_link_down_resets_to_detect(dut):
    """link_down from any state returns to DETECT (0)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await train_link_to_l0(dut)
    assert int(dut.link_state.value) == 3

    dut.link_down.value = 1
    await RisingEdge(dut.clk)
    dut.link_down.value = 0
    await RisingEdge(dut.clk)

    assert int(dut.link_state.value) == 0, (
        f"Expected DETECT(0) after link_down, got {dut.link_state.value}"
    )
    dut._log.info("link_down test passed")


@cocotb.test()
async def test_tlp_mem_write_header(dut):
    """MEM_WRITE TLP header must have correct fmt=2'b10, type=5'h00."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await train_link_to_l0(dut)

    req_id    = 0x0100   # bus=0, dev=2, fn=0
    tag       = 0xAB
    addr      = 0xDEAD0000
    length_dw = 4

    hdr = await send_tlp(dut, tlp_type=1, req_id=req_id, tag=tag,
                         addr=addr, length_dw=length_dw)

    # DW0 [31:24]: fmt[1:0]=2'b10, type[4:0]=5'h00 → bits 31:24 = 8'b10_00000_0 = 0x40
    dw0 = hdr & 0xFFFFFFFF
    fmt_type = (dw0 >> 24) & 0xFC   # bits 31:26 = fmt + type
    assert (dw0 >> 30) == 0b10, f"MEM_WRITE fmt should be 2'b10, got {(dw0>>30):02b}"
    assert ((dw0 >> 25) & 0x1F) == 0, f"MEM_WRITE type should be 5'h00"

    # DW0 length field = bits [9:0]
    assert (dw0 & 0x3FF) == length_dw, f"DW0 length mismatch"

    # DW1 = requester_id + tag
    dw1 = (hdr >> 32) & 0xFFFFFFFF
    assert (dw1 >> 16) == req_id, f"req_id mismatch in DW1"
    assert ((dw1 >> 8) & 0xFF) == tag, f"tag mismatch in DW1"

    # DW2 = addr (DWORD-aligned, lower 2 bits = 0)
    dw2 = (hdr >> 64) & 0xFFFFFFFF
    assert dw2 == (addr & 0xFFFFFFFC), f"DW2 addr mismatch: 0x{dw2:08X}"

    dut._log.info(f"tlp_mem_write_header test passed: hdr=0x{hdr:024X}")


@cocotb.test()
async def test_tlp_mem_read_header(dut):
    """MEM_READ TLP must have fmt=2'b00."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await train_link_to_l0(dut)

    hdr = await send_tlp(dut, tlp_type=0, req_id=0x0200, tag=0x55,
                         addr=0xABCD0004, length_dw=1)

    dw0 = hdr & 0xFFFFFFFF
    assert (dw0 >> 30) == 0b00, f"MEM_READ fmt should be 2'b00, got {(dw0>>30):02b}"
    dut._log.info("tlp_mem_read_header test passed")


@cocotb.test()
async def test_tlp_not_sent_when_not_l0(dut):
    """TLP start when not in L0 should leave busy=0 and no tlp_done."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Do NOT train to L0 — stay in DETECT
    dut.tlp_start.value = 1
    await RisingEdge(dut.clk)
    dut.tlp_start.value = 0

    for _ in range(5):
        await RisingEdge(dut.clk)
        assert int(dut.tlp_done.value) == 0, "tlp_done should not assert in DETECT state"

    dut._log.info("tlp_not_sent_when_not_l0 test passed")


@cocotb.test()
async def test_tlp_busy_during_assembly(dut):
    """busy should be asserted during TLP header assembly."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await train_link_to_l0(dut)

    dut.tlp_type.value  = 1
    dut.req_id.value    = 0x0100
    dut.tag.value       = 0x01
    dut.addr.value      = 0x1000
    dut.length_dw.value = 2
    dut.tlp_start.value = 1
    await RisingEdge(dut.clk)
    dut.tlp_start.value = 0

    # busy must be asserted during assembly (first cycle after start)
    await RisingEdge(dut.clk)
    assert int(dut.busy.value) == 1, "busy should be asserted during assembly"

    # Wait for completion
    for _ in range(10):
        await RisingEdge(dut.clk)
        if int(dut.tlp_done.value):
            assert int(dut.busy.value) == 0, "busy should deassert on done"
            break

    dut._log.info("tlp_busy_during_assembly test passed")


@cocotb.test()
async def test_completion_data_tlp_type(dut):
    """COMPLETION_DATA TLP must have fmt=2'b10, type=5'h0A."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await train_link_to_l0(dut)

    hdr = await send_tlp(dut, tlp_type=2, req_id=0x0300, tag=0x77,
                         addr=0x0, length_dw=1)

    dw0 = hdr & 0xFFFFFFFF
    assert (dw0 >> 30) == 0b10, f"CplD fmt should be 2'b10"
    assert ((dw0 >> 25) & 0x1F) == 0x0A, (
        f"CplD type should be 5'h0A, got 0x{((dw0>>25)&0x1F):02X}"
    )
    dut._log.info("completion_data_tlp test passed")
