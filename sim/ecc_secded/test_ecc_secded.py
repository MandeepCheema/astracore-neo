"""
AstraCore Neo — ECC SECDED cocotb testbench.

Python ECCEngine is the GOLDEN REFERENCE.
The Verilog DUT implements SECDED Hamming(72,64): 64 data bits + 8 parity bits.

Encoding convention:
  Python encode(data) returns a 72-bit integer:
    bits  0-63: data
    bits 64-71: 8-bit parity (h[6:0] + overall parity bit)
  Verilog encode mode: parity_out[7:0] = the 8 parity bits

Decode convention:
  Python decode(codeword) receives the 72-bit integer.
  Verilog decode mode: data_in[63:0] + parity_in[7:0].
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from safety.ecc import ECCEngine
from safety.exceptions import ECCError

ENCODE = 0
DECODE = 1


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.valid.value = 0
    dut.mode.value = 0
    dut.data_in.value = 0
    dut.parity_in.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def encode_word(dut, ref: ECCEngine, data: int):
    """Drive encode operation; return (dut_parity, ref_codeword)."""
    dut.mode.value    = ENCODE
    dut.data_in.value = data & 0xFFFFFFFFFFFFFFFF
    dut.parity_in.value = 0
    dut.valid.value   = 1
    await RisingEdge(dut.clk)
    dut.valid.value   = 0
    await RisingEdge(dut.clk)

    ref_codeword = ref.encode(data)
    ref_parity   = (ref_codeword >> 64) & 0xFF
    return int(dut.parity_out.value), ref_parity


async def decode_word(dut, ref: ECCEngine, data: int, parity: int, bank=0, addr=0):
    """Drive decode operation; return DUT outputs and ref result."""
    dut.mode.value    = DECODE
    dut.data_in.value = data & 0xFFFFFFFFFFFFFFFF
    dut.parity_in.value = parity & 0xFF
    dut.valid.value   = 1
    await RisingEdge(dut.clk)
    dut.valid.value   = 0
    await RisingEdge(dut.clk)

    codeword = data | (parity << 64)
    ref_result = None
    ref_double = False
    try:
        ref_result = ref.decode(codeword, bank=bank, address=addr)
    except ECCError:
        ref_double = True

    return ref_result, ref_double


@cocotb.test()
async def test_encode_clean_word(dut):
    """Encoding a known word must produce matching parity."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = ECCEngine()
    data = 0xDEADBEEFCAFEBABE

    dut_parity, ref_parity = await encode_word(dut, ref, data)
    assert dut_parity == ref_parity, (
        f"Parity mismatch: DUT=0x{dut_parity:02X} REF=0x{ref_parity:02X}"
    )
    dut._log.info(f"encode_clean_word passed: parity=0x{dut_parity:02X}")


@cocotb.test()
async def test_encode_zero(dut):
    """Encoding all-zeros data."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = ECCEngine()
    dut_parity, ref_parity = await encode_word(dut, ref, 0)
    assert dut_parity == ref_parity, (
        f"Zero encode mismatch: DUT=0x{dut_parity:02X} REF=0x{ref_parity:02X}"
    )
    dut._log.info("encode_zero passed")


@cocotb.test()
async def test_encode_all_ones(dut):
    """Encoding all-ones data."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = ECCEngine()
    data = 0xFFFFFFFFFFFFFFFF
    dut_parity, ref_parity = await encode_word(dut, ref, data)
    assert dut_parity == ref_parity, (
        f"All-ones encode mismatch: DUT=0x{dut_parity:02X} REF=0x{ref_parity:02X}"
    )
    dut._log.info("encode_all_ones passed")


@cocotb.test()
async def test_decode_no_error(dut):
    """Decoding a clean codeword → no error flags."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = ECCEngine()
    data = 0x0123456789ABCDEF
    _, ref_parity = await encode_word(dut, ref, data)

    ref_result, ref_double = await decode_word(dut, ref, data, ref_parity)

    assert not dut.single_err.value, "No single error expected"
    assert not dut.double_err.value, "No double error expected"
    assert int(dut.data_out.value) == (data & 0xFFFFFFFFFFFFFFFF), (
        f"data_out mismatch: DUT=0x{dut.data_out.value:016X} expected=0x{data:016X}"
    )
    dut._log.info("decode_no_error passed")


@cocotb.test()
async def test_decode_single_bit_error(dut):
    """Flip one data bit → DUT must detect and correct it."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = ECCEngine()
    data = 0xDEADBEEFCAFEBABE

    _, ref_parity = await encode_word(dut, ref, data)

    # Flip bit 5 of data
    corrupted_data = data ^ (1 << 5)
    ref_result, ref_double = await decode_word(dut, ref, corrupted_data, ref_parity)

    assert ref_result is not None and ref_result.corrected
    assert int(dut.single_err.value) == 1, "single_err should be set"
    assert int(dut.double_err.value) == 0, "double_err should not be set"
    assert int(dut.data_out.value) == (data & 0xFFFFFFFFFFFFFFFF), (
        f"Corrected data mismatch: DUT=0x{int(dut.data_out.value):016X} "
        f"expected=0x{data:016X}"
    )
    dut._log.info(f"single_bit_error test passed: err_pos={dut.err_pos.value}")


@cocotb.test()
async def test_decode_multiple_single_bit_positions(dut):
    """Test single-bit error correction at various bit positions."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = ECCEngine()
    data = 0xAAAA5555CCCC3333

    _, ref_parity = await encode_word(dut, ref, data)

    # Test bit flips at positions 0, 7, 31, 63
    for bit_pos in [0, 7, 15, 31, 47, 63]:
        corrupted = data ^ (1 << bit_pos)
        await decode_word(dut, ref, corrupted, ref_parity)

        assert int(dut.single_err.value) == 1, (
            f"bit_pos={bit_pos}: single_err should be set"
        )
        assert int(dut.data_out.value) == (data & 0xFFFFFFFFFFFFFFFF), (
            f"bit_pos={bit_pos}: corrected data mismatch"
        )

    dut._log.info("multi-position single-bit test passed")


@cocotb.test()
async def test_decode_double_bit_error(dut):
    """Flip two data bits → DUT must detect double-bit error."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = ECCEngine()
    data = 0xDEADBEEFCAFEBABE

    _, ref_parity = await encode_word(dut, ref, data)

    # Flip bits 3 and 7 simultaneously
    corrupted = data ^ (1 << 3) ^ (1 << 7)
    ref_result, ref_double = await decode_word(dut, ref, corrupted, ref_parity)

    assert ref_double is True, "Python ref should detect double-bit error"
    assert int(dut.double_err.value) == 1, "DUT double_err should be set"
    assert int(dut.single_err.value) == 0, "DUT single_err should not be set"
    dut._log.info("double_bit_error test passed")


@cocotb.test()
async def test_encode_decode_roundtrip(dut):
    """Encode then decode a word with no errors → data must be preserved."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = ECCEngine()
    words = [0x0, 0xFFFFFFFFFFFFFFFF, 0xDEADBEEFCAFEBABE, 0x123456789ABCDEF0]

    for data in words:
        dut_parity, _ = await encode_word(dut, ref, data)
        await decode_word(dut, ref, data, dut_parity)

        assert int(dut.single_err.value) == 0
        assert int(dut.double_err.value) == 0
        assert int(dut.data_out.value) == (data & 0xFFFFFFFFFFFFFFFF), (
            f"Roundtrip failed for 0x{data:016X}"
        )

    dut._log.info("encode_decode_roundtrip passed")


@cocotb.test()
async def test_reset_clears_outputs(dut):
    """After reset, all error outputs should be cleared."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ref = ECCEngine()
    data = 0xDEADBEEFCAFEBABE
    _, ref_parity = await encode_word(dut, ref, data)
    corrupted = data ^ (1 << 5)
    await decode_word(dut, ref, corrupted, ref_parity)

    # Now reset
    dut.rst_n.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    assert int(dut.single_err.value) == 0
    assert int(dut.double_err.value) == 0
    assert int(dut.data_out.value)   == 0
    dut._log.info("reset_clears_outputs passed")
