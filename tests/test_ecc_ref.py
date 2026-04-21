"""Tests for tools/safety/ecc_ref.py — Python mirror of npu_sram_bank_ecc.v.

The RTL mirror is bit-for-bit identical to the inline Verilog
``secded_encode`` and ``secded_decode`` functions in
``rtl/npu_sram_bank_ecc/npu_sram_bank_ecc.v``. These tests verify the
SECDED algorithm contract end-to-end on Windows; the cocotb gate (WSL)
will additionally verify wire-level RTL behaviour.

Properties tested:
1. Round-trip: every encoded word decodes to itself with no errors.
2. Single-bit flip in any of the 72 positions is detected and (for data
   bits) corrected.
3. Double-bit flip is detected with high reliability (some syndromes
   alias a single-bit error — that is the SECDED limit, not a bug).
"""

from __future__ import annotations

import random

import pytest

from tools.safety.ecc_ref import decode, encode


# Use a fixed seed so the property test is deterministic + reproducible.
_RNG = random.Random(0xCAFEBABE)


@pytest.mark.parametrize(
    "data",
    [
        0x0000000000000000,
        0xFFFFFFFFFFFFFFFF,
        0xDEADBEEFCAFEBABE,
        0x0123456789ABCDEF,
        0xFEDCBA9876543210,
        0x5555555555555555,
        0xAAAAAAAAAAAAAAAA,
    ],
)
def test_round_trip_no_error(data):
    parity = encode(data)
    r = decode(data, parity)
    assert r.corrected_data == data
    assert r.single_err is False
    assert r.double_err is False


def test_round_trip_random_words():
    for _ in range(200):
        data = _RNG.randrange(0, 1 << 64)
        parity = encode(data)
        r = decode(data, parity)
        assert r.corrected_data == data
        assert not r.single_err
        assert not r.double_err


@pytest.mark.parametrize("data", [0xDEADBEEFCAFEBABE, 0x0123456789ABCDEF])
@pytest.mark.parametrize("flip_bit", list(range(64)))
def test_single_data_bit_flip_corrected(data, flip_bit):
    parity = encode(data)
    corrupted_data = data ^ (1 << flip_bit)
    r = decode(corrupted_data, parity)
    assert r.single_err is True, f"bit {flip_bit}: single_err not raised"
    assert r.double_err is False, f"bit {flip_bit}: spurious double_err"
    assert r.corrected_data == data, (
        f"bit {flip_bit}: not corrected (got {r.corrected_data:#018x}, "
        f"expected {data:#018x})"
    )


@pytest.mark.parametrize("data", [0xDEADBEEFCAFEBABE])
@pytest.mark.parametrize("flip_bit", list(range(8)))
def test_single_parity_bit_flip_detected(data, flip_bit):
    """Parity-bit flips are *detected*. See ecc_ref.py module docstring
    for the layout limitation: in the systematic SECDED layout we
    inherit from rtl/ecc_secded/ecc_secded.v, the decoder may
    "correct" one data bit when only a parity bit was flipped (for
    flip_bit ∈ {0..6}, since syndrome = 2**flip_bit aliases data bit
    {0, 1, 3, 7, 15, 31, 63}). Detection still works; correction is
    spurious. Phase D WP F4-D-6 closes this with the standard
    interleaved layout."""
    parity = encode(data)
    corrupted_parity = parity ^ (1 << flip_bit)
    r = decode(data, corrupted_parity)
    assert r.single_err is True, f"flip_bit {flip_bit}: single_err not raised"
    assert r.double_err is False
    if flip_bit == 7:
        # p7 (overall) bit flip: syndrome = 0, no data correction attempted
        assert r.corrected_data == data
    else:
        # h[flip_bit] flip → syndrome = 1 << flip_bit ∈ {1,2,4,8,16,32,64}
        # Decoder will spuriously flip data[(1<<flip_bit) - 1]. Document
        # the actual behavior so any future change is intentional.
        expected_corrupted = data ^ (1 << ((1 << flip_bit) - 1))
        assert r.corrected_data == expected_corrupted, (
            f"flip_bit {flip_bit}: SECDED layout aliasing changed; "
            f"expected spurious-flip on data[{(1<<flip_bit)-1}]"
        )


def test_double_data_bit_flip_detected_majority():
    """Double-bit data flips must be detected with high reliability.

    SECDED guarantees double-bit *detection*, not 100 % classification —
    a small fraction of double-flip syndromes alias single-bit
    syndromes. We assert ≥ 99 % detection rate over a sweep.
    """
    data = 0xDEADBEEFCAFEBABE
    parity = encode(data)
    detected = 0
    total = 0
    for i in range(64):
        for j in range(i + 1, 64):
            corrupted = data ^ (1 << i) ^ (1 << j)
            r = decode(corrupted, parity)
            total += 1
            if r.double_err:
                detected += 1
    # Hamming(72,64) detects 100 % of double-bit errors that are
    # entirely in the data field — there is no aliasing in this case.
    # Aliasing only occurs across data+parity boundaries.
    assert detected == total, (
        f"only {detected}/{total} double-data-bit flips detected"
    )


def test_zero_word_round_trip_with_no_parity_bits_set():
    """The all-zero data word must encode to all-zero parity — sanity."""
    assert encode(0) == 0
    r = decode(0, 0)
    assert r.corrected_data == 0
    assert not r.single_err
    assert not r.double_err


def test_encode_rejects_oversize_data():
    with pytest.raises(ValueError):
        encode(1 << 64)


def test_decode_rejects_oversize_parity():
    with pytest.raises(ValueError):
        decode(0, 1 << 8)
