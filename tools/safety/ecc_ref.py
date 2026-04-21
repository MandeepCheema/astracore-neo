"""Bit-exact Python mirror of rtl/npu_sram_bank_ecc/npu_sram_bank_ecc.v.

Mirrors the inline ``secded_encode`` and ``secded_decode`` Verilog
functions so the algorithm can be unit-tested on Windows without WSL
+ cocotb.

This is the same Hamming(72,64) SECDED layout as the existing
``rtl/ecc_secded/ecc_secded.v`` primitive (and ``src/safety/ecc.py``):

* h[i] = XOR of data[j] where ((j+1) >> i) & 1, for j in 0..63, i in 0..6
* overall = XOR(data) ^ XOR(h)
* parity = {overall (msb), h[6:0]} — same packing as the RTL

When the cocotb gate (planned WP F4-A-1.2 follow-up) runs, it will
import this module to drive the bit-exact comparison.

Known limitation (carried forward from rtl/ecc_secded/ecc_secded.v)
-----------------------------------------------------------------
This is a *systematic* SECDED layout (parity stored separately from
data), not the standard *interleaved* Hamming(72,64) layout where
parity bits live at codeword positions 1, 2, 4, 8, 16, 32, 64.

In the systematic layout, a single-bit flip in a Hamming parity bit
``h[i]`` produces syndrome ``2**i``, which is the same syndrome that a
single-bit flip in data bit ``data[2**i - 1]`` produces. The decoder
cannot disambiguate the two cases from the syndrome alone, and will
"correct" one of the affected data bits ``{data[0], data[1], data[3],
data[7], data[15], data[31], data[63]}`` when in fact only the parity
bit was flipped. The flip is *detected* (single_err asserts) but the
correction may corrupt one previously-clean data bit.

For ASIL-B coverage of *data* SEU, this still works (data-bit flips
are reliably detected and corrected). For ASIL-D — where the false
correction itself becomes a hazard — switch to the standard interleaved
layout. Tracked as remediation WP **F4-D-6** in
``docs/safety/findings_remediation_plan_v0_1.md`` (Phase D).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeResult:
    corrected_data: int  # 64-bit
    single_err: bool
    double_err: bool


def encode(data: int) -> int:
    """Return the 8-bit parity word for a 64-bit data input."""
    if not 0 <= data < (1 << 64):
        raise ValueError("data must fit in 64 bits")
    h = 0
    for i in range(7):
        bit = 0
        for j in range(64):
            if ((j + 1) >> i) & 1:
                bit ^= (data >> j) & 1
        h |= (bit & 1) << i
    overall = _xor_reduce(data) ^ _xor_reduce(h)
    return ((overall & 1) << 7) | (h & 0x7F)


def decode(data: int, parity: int) -> DecodeResult:
    """Decode a 72-bit codeword; classify and (if possible) correct."""
    if not 0 <= data < (1 << 64):
        raise ValueError("data must fit in 64 bits")
    if not 0 <= parity < (1 << 8):
        raise ValueError("parity must fit in 8 bits")
    h_recv = parity & 0x7F
    overall_recv = (parity >> 7) & 1

    syndrome = 0
    for i in range(7):
        bit = 0
        for j in range(64):
            if ((j + 1) >> i) & 1:
                bit ^= (data >> j) & 1
        bit ^= (h_recv >> i) & 1
        syndrome |= (bit & 1) << i

    overall_xor = _xor_reduce(data) ^ _xor_reduce(h_recv) ^ overall_recv

    corrected = data
    single = False
    double = False
    if syndrome == 0 and overall_xor == 1:
        # parity bit was flipped; data is intact
        single = True
    elif syndrome != 0 and overall_xor == 1:
        single = True
        pos = syndrome  # 1-indexed bit position
        if 1 <= pos <= 64:
            corrected ^= 1 << (pos - 1)
    elif syndrome != 0 and overall_xor == 0:
        double = True
    # else: no error
    return DecodeResult(corrected_data=corrected, single_err=single, double_err=double)


def _xor_reduce(x: int) -> int:
    out = 0
    while x:
        out ^= x & 1
        x >>= 1
    return out
