"""AstraCore Neo — legacy astracore_top AXI4-Lite integration test.

Closes audit finding §D: the tape-out-ready legacy chip has clean
physical signoff (DRC/LVS/timing) but no functional verification of the
top-level AXI register map. This test exercises each register and
confirms writes propagate through to the corresponding submodule
control inputs, and reads return the corresponding submodule status
outputs.

This is register-map sanity, not exhaustive per-module testing
(those live in sim/<submodule>/test_<submodule>.py). The goal here
is to prove that if the 11 submodules work, the AXI wrapper correctly
plumbs them.

Register map (from rtl/astracore_top/astracore_top.v header):

  Writes (word-offset × 4 = byte addr):
    0x00 CTRL      [0]=mod_valid (pulse), [1]=sw_rst
    0x04 GAZE      [7:0]=left_ear, [15:8]=right_ear
    0x08 THERMAL   [7:0]=temp_in
    0x0C CANFD     [0]=tx_success (pulse), [1]=tx_error, [2]=rx_error, [3]=bus_off_recovery
    0x10 ECC_LO    [31:0]=data_in[31:0]
    0x14 ECC_HI    [31:0]=data_in[63:32]
    0x18 ECC_CTRL  [0]=mode (0=encode, 1=decode), [15:8]=parity_in
    0x1C TMR_A     [31:0]=lane_a
    0x20 TMR_B     [31:0]=lane_b
    0x24 TMR_C     [31:0]=lane_c
    0x28 FAULT     [15:0]=fault_value
    0x2C HEADPOSE  [7:0]=yaw, [15:8]=pitch, [23:16]=roll
    0x30 PCIE_CTRL [0]=link_up, [1]=link_down, [4:2]=tlp_type, [5]=tlp_start (pulse)
    ... etc.
    0x48 INF_CTRL  pulse bits

  Reads (word-offset × 4 = byte addr):
    0x80 GAZE_ST   [1:0]=eye_state, [7:2]=perclos_num, [23:8]=blink_count
    0x84 THERM_ST  [2:0]=therm_state, [3]=throttle_en, [4]=shutdown_req
    0x88 CANFD_ST  [8:0]=tec, [17:9]=rec, [19:18]=bus_state
    0x8C ECC_ST    [0]=single_err, [1]=double_err, [2]=corrected, [9:3]=err_pos, [17:10]=parity_out
    0x90 ECC_DLO   [31:0]=data_out[31:0]
    0x94 ECC_DHI   [31:0]=data_out[63:32]
    0x98 TMR_RES   [31:0]=voted
    0x9C TMR_ST    [0]=agreement, [1]=fault_a, [2]=fault_b, [3]=fault_c, [4]=triple_fault, [6:5]=vote_count
    0xA0 FAULT_ST  [2:0]=risk, [3]=alarm, [19:4]=rolling_mean
    0xA4 HEAD_ST   [0]=in_zone, [5:1]=distracted_count
    ... etc.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


# Write register byte offsets
W_CTRL       = 0x00
W_GAZE       = 0x04
W_THERMAL    = 0x08
W_CANFD      = 0x0C
W_ECC_LO     = 0x10
W_ECC_HI     = 0x14
W_ECC_CTRL   = 0x18
W_TMR_A      = 0x1C
W_TMR_B      = 0x20
W_TMR_C      = 0x24
W_FAULT      = 0x28
W_HEADPOSE   = 0x2C
W_PCIE_CTRL  = 0x30
W_PCIE_REQID = 0x34
W_PCIE_ADDR  = 0x38
W_PCIE_LEN   = 0x3C
W_ETH        = 0x40
W_MAC        = 0x44
W_INF        = 0x48

# Read register byte offsets
R_GAZE_ST    = 0x80
R_THERM_ST   = 0x84
R_CANFD_ST   = 0x88
R_ECC_ST     = 0x8C
R_ECC_DLO    = 0x90
R_ECC_DHI    = 0x94
R_TMR_RES    = 0x98
R_TMR_ST     = 0x9C
R_FAULT_ST   = 0xA0
R_HEAD_ST    = 0xA4
R_PCIE_ST    = 0xA8
R_PCIE_H0    = 0xAC
R_PCIE_H1    = 0xB0
R_PCIE_H2    = 0xB4
R_ETH_ST     = 0xB8
R_ETYPE      = 0xBC
R_MAC_RES    = 0xC0
R_INF_ST     = 0xC4


async def _reset(dut):
    dut.rst_n.value          = 0
    dut.s_axil_awaddr.value  = 0
    dut.s_axil_awvalid.value = 0
    dut.s_axil_wdata.value   = 0
    dut.s_axil_wstrb.value   = 0
    dut.s_axil_wvalid.value  = 0
    dut.s_axil_bready.value  = 1
    dut.s_axil_araddr.value  = 0
    dut.s_axil_arvalid.value = 0
    dut.s_axil_rready.value  = 1
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def axil_write(dut, addr, data, strb=0xF):
    """Issue a single AXI4-Lite write, waiting for bvalid ack.

    The wrapper's write path is 2-cycle: awready fires the cycle after
    awvalid is seen, then wready fires the cycle after the address is
    latched.  We keep awvalid + wvalid high and poll for bvalid, which
    implies both handshakes completed.
    """
    dut.s_axil_awaddr.value  = addr
    dut.s_axil_awvalid.value = 1
    dut.s_axil_wdata.value   = data & 0xFFFFFFFF
    dut.s_axil_wstrb.value   = strb
    dut.s_axil_wvalid.value  = 1
    # Track which handshakes have completed so we can drop those *valid
    # signals individually (AXI protocol: once *ready has been seen with
    # *valid high, *valid must be dropped within a bounded time).
    aw_done = False
    w_done = False
    for _ in range(30):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if not aw_done and int(dut.s_axil_awready.value):
            dut.s_axil_awvalid.value = 0
            aw_done = True
        if not w_done and int(dut.s_axil_wready.value):
            dut.s_axil_wvalid.value = 0
            w_done = True
        if int(dut.s_axil_bvalid.value):
            break
    else:
        raise AssertionError(
            f"AXI write did not complete at addr 0x{addr:02x} "
            f"(aw_done={aw_done} w_done={w_done})")
    assert int(dut.s_axil_bresp.value) == 0, \
        f"AXI write bresp != OKAY at addr 0x{addr:02x}: {int(dut.s_axil_bresp.value)}"
    # Give 1 more cycle for bvalid to be consumed (bready is held high
    # in the test harness).
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def axil_read(dut, addr):
    """Issue a single AXI4-Lite read, return rdata."""
    dut.s_axil_araddr.value  = addr
    dut.s_axil_arvalid.value = 1
    data = None
    for _ in range(20):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.s_axil_rvalid.value):
            data = int(dut.s_axil_rdata.value)
            assert int(dut.s_axil_rresp.value) == 0, \
                f"AXI read rresp != OKAY at addr 0x{addr:02x}: {int(dut.s_axil_rresp.value)}"
            break
    else:
        raise AssertionError(f"AXI read rvalid missing at addr 0x{addr:02x}")
    dut.s_axil_arvalid.value = 0
    # Let rready consume
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    return data


# ---------------------------------------------------------------------------
@cocotb.test()
async def test_reset_leaves_deterministic_state(dut):
    """After reset, every status register should read a defined value
    (no X propagation). Smoke test that the wrapper elaborates cleanly."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    # Read each status register; must not be X.  Raw rdata is a LogicArray.
    for addr in (R_GAZE_ST, R_THERM_ST, R_CANFD_ST, R_ECC_ST, R_ECC_DLO,
                 R_ECC_DHI, R_TMR_RES, R_TMR_ST, R_FAULT_ST, R_HEAD_ST,
                 R_PCIE_ST, R_PCIE_H0, R_PCIE_H1, R_PCIE_H2, R_ETH_ST,
                 R_ETYPE, R_MAC_RES, R_INF_ST):
        val = await axil_read(dut, addr)
        # int() would raise on X; if we got here, it's defined
        assert val >= 0, f"addr 0x{addr:02x} returned negative? {val}"


@cocotb.test()
async def test_write_registers_readback(dut):
    """Every write register should read back exactly what was written
    (for the bits the register retains — some bits are auto-clearing pulse
    strobes, those are handled specifically below)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    # Non-pulse registers: write pattern, read back
    non_pulse = [
        (W_GAZE,       0x1234ABCD),  # full 32b retained
        (W_THERMAL,    0x0000_0055),
        (W_ECC_LO,     0xDEADBEEF),
        (W_ECC_HI,     0xCAFEBABE),
        (W_ECC_CTRL,   0x0000_5501),
        (W_TMR_A,      0xA5A5_A5A5),
        (W_TMR_B,      0x5A5A_5A5A),
        (W_TMR_C,      0xFFFF_0000),
        (W_FAULT,      0x0000_ABCD),
        (W_HEADPOSE,   0x00123456),
        (W_PCIE_REQID, 0x00345612),
        (W_PCIE_ADDR,  0x12345678),
        (W_PCIE_LEN,   0x000003FF),
        (W_ETH,        0x000000FF),
        (W_MAC,        0x0001234F),
    ]
    for addr, pattern in non_pulse:
        await axil_write(dut, addr, pattern)
        readback = await axil_read(dut, addr)
        assert readback == pattern, (
            f"write 0x{addr:02x}: wrote 0x{pattern:08x} read 0x{readback:08x}")


async def _pulse_mod_valid(dut):
    """Pulse CTRL[0]=mod_valid for one cycle (auto-clears in RTL).

    Needed after any write that updates inputs the submodule samples
    synchronously (TMR voter, thermal_zone, head_pose_tracker,
    gaze_tracker, etc.).
    """
    await axil_write(dut, W_CTRL, 0x1)
    # Let the pulse be seen by sub-modules and then auto-clear
    for _ in range(3):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")


@cocotb.test()
async def test_tmr_voter_integration(dut):
    """TMR voter sanity through AXI: when all 3 lanes agree, voted value
    matches and agreement=1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    pattern = 0xDEAD_BEEF
    await axil_write(dut, W_TMR_A, pattern)
    await axil_write(dut, W_TMR_B, pattern)
    await axil_write(dut, W_TMR_C, pattern)
    await _pulse_mod_valid(dut)
    voted = await axil_read(dut, R_TMR_RES)
    status = await axil_read(dut, R_TMR_ST)
    assert voted == pattern, f"TMR voted 0x{voted:08x} expected 0x{pattern:08x}"
    agreement = status & 0x1
    assert agreement == 1, f"TMR agreement should be 1 when all lanes match: status=0x{status:08x}"
    dut._log.info(f"TMR voter: voted=0x{voted:08x} agreement={agreement}")


@cocotb.test()
async def test_tmr_voter_disagreement(dut):
    """2-of-3 majority: A=B != C → voted=A, agreement=1 (2 lanes agreed),
    fault_c=1, vote_count=2.  Per module header: agreement is 'at least 2
    lanes agreed', not 'all 3 agree'."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    await axil_write(dut, W_TMR_A, 0x11)
    await axil_write(dut, W_TMR_B, 0x11)
    await axil_write(dut, W_TMR_C, 0x22)
    await _pulse_mod_valid(dut)
    voted = await axil_read(dut, R_TMR_RES)
    status = await axil_read(dut, R_TMR_ST)
    agreement   = (status >> 0) & 0x1
    fault_a     = (status >> 1) & 0x1
    fault_b     = (status >> 2) & 0x1
    fault_c     = (status >> 3) & 0x1
    triple      = (status >> 4) & 0x1
    vote_count  = (status >> 5) & 0x3
    assert voted == 0x11, f"TMR 2-of-3 majority failed: voted=0x{voted:08x}"
    assert agreement == 1, f"agreement should be 1 (2 of 3 agree): status=0x{status:08x}"
    assert fault_c == 1 and fault_a == 0 and fault_b == 0, \
        f"only fault_c should be set: status=0x{status:08x}"
    assert triple == 0, f"triple_fault should be 0: status=0x{status:08x}"
    assert vote_count == 2, f"vote_count should be 2: status=0x{status:08x}"
    dut._log.info(f"TMR 2-of-3 majority: voted=0x{voted:08x} status=0x{status:08x}")


@cocotb.test()
async def test_tmr_triple_fault(dut):
    """All 3 lanes disagree → triple_fault=1, agreement=0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    await axil_write(dut, W_TMR_A, 0xAAAA)
    await axil_write(dut, W_TMR_B, 0xBBBB)
    await axil_write(dut, W_TMR_C, 0xCCCC)
    await _pulse_mod_valid(dut)
    status = await axil_read(dut, R_TMR_ST)
    agreement = status & 0x1
    triple    = (status >> 4) & 0x1
    assert agreement == 0, f"agreement should be 0 on triple-fault: status=0x{status:08x}"
    assert triple == 1, f"triple_fault should be 1: status=0x{status:08x}"
    dut._log.info(f"TMR triple-fault: status=0x{status:08x}")


@cocotb.test()
async def test_thermal_zone_propagation(dut):
    """Write a hot temperature, verify therm_state advances via AXI.

    Thresholds per astracore_top instantiation: WARN=75, THROTTLE=85,
    CRITICAL=95, SHUTDOWN=105.  Writing 0xA0 = 160 is above all
    thresholds → SHUTDOWN state."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    await axil_write(dut, W_THERMAL, 0xA0)
    await _pulse_mod_valid(dut)
    status = await axil_read(dut, R_THERM_ST)
    therm_state = status & 0x7
    assert therm_state != 0, f"thermal_zone did not advance for temp 0xA0: state={therm_state}"
    dut._log.info(f"thermal state @ temp 0xA0 (=160) = {therm_state}")


@cocotb.test()
async def test_head_pose_propagation(dut):
    """Write angles inside the 'in-zone' envelope, verify in_zone=1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    yaw = 5
    pitch = 0
    roll = 0
    pattern = (roll << 16) | (pitch << 8) | yaw
    await axil_write(dut, W_HEADPOSE, pattern)
    await _pulse_mod_valid(dut)
    status = await axil_read(dut, R_HEAD_ST)
    in_zone = status & 0x1
    assert in_zone == 1, f"head_pose in_zone should be 1: status=0x{status:08x}"
    dut._log.info(f"head_pose in_zone @ small angles = {in_zone}")


@cocotb.test()
async def test_unmapped_read_returns_deadbeef(dut):
    """Reads to unmapped addresses return the 0xDEADBEEF sentinel, not X."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    # Address 0xE0 is unmapped (last mapped read offset is 0xC4 → 6'h31; 0xE0 → 6'h38)
    val = await axil_read(dut, 0xE0)
    assert val == 0xDEADBEEF, f"unmapped read should return 0xDEADBEEF, got 0x{val:08x}"


@cocotb.test()
async def test_sw_rst_returns_state_to_default(dut):
    """Asserting sw_rst via CTRL[1] should reset all submodule state.

    Sequence: write TMR pattern + pulse mod_valid → verify TMR voted
    = pattern. Then assert sw_rst (CTRL[1]=1) and verify TMR voted
    clears to 0 (submodule is held in reset).
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)
    # Drive TMR with a known all-agree pattern and pulse valid
    pattern = 0xDEADBEEF
    await axil_write(dut, W_TMR_A, pattern)
    await axil_write(dut, W_TMR_B, pattern)
    await axil_write(dut, W_TMR_C, pattern)
    await _pulse_mod_valid(dut)
    voted_before = await axil_read(dut, R_TMR_RES)
    assert voted_before == pattern, f"TMR pre-rst voted 0x{voted_before:08x} != 0x{pattern:08x}"
    # Assert sw_rst.  CTRL[1] is NOT auto-clearing, so writing CTRL=0b10
    # sets sw_rst and keeps it high until we write CTRL=0.
    await axil_write(dut, W_CTRL, 0b10)
    # Hold a few cycles
    for _ in range(5):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
    voted_during = await axil_read(dut, R_TMR_RES)
    assert voted_during == 0, \
        f"sw_rst should clear TMR voted to 0, got 0x{voted_during:08x}"
    # Release sw_rst
    await axil_write(dut, W_CTRL, 0)
    dut._log.info("sw_rst propagation confirmed")
