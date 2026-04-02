"""
AstraCore Neo — ECC (Error Correcting Code) engine simulation.

Models SECDED (Single Error Correct, Double Error Detect) ECC:
  - Hamming(72,64): 64 data bits + 8 parity bits per word
  - Single-bit errors are corrected transparently
  - Double-bit errors are detected and raise ECCError
  - Tracks error counters per memory bank
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

from .exceptions import ECCError


class BitFlipType(Enum):
    NONE        = auto()   # no error
    SINGLE_BIT  = auto()   # correctable
    DOUBLE_BIT  = auto()   # detectable, uncorrectable
    MULTI_BIT   = auto()   # catastrophic


@dataclass
class ECCConfig:
    """ECC engine configuration."""
    data_bits: int = 64           # data word width
    parity_bits: int = 8          # Hamming parity bits (SECDED for 64-bit)
    banks: int = 4                # number of memory banks monitored
    scrub_interval_words: int = 1024  # background scrub period


@dataclass
class CorrectionResult:
    """Result of an ECC check on a memory word."""
    corrected: bool               # True if a single-bit error was corrected
    error_type: BitFlipType
    bank: int
    address: int
    bit_position: Optional[int]   # which bit was flipped (single-bit only)
    original_word: int
    corrected_word: int


class ECCEngine:
    """
    Simulates SECDED ECC for on-chip SRAM.

    Usage::

        ecc = ECCEngine()
        word = 0xDEADBEEFCAFEBABE
        encoded = ecc.encode(word)
        # inject a single-bit error
        corrupted = encoded ^ (1 << 5)
        result = ecc.decode(corrupted, bank=0, address=0x100)
        assert result.corrected
    """

    def __init__(self, config: Optional[ECCConfig] = None) -> None:
        self._cfg = config or ECCConfig()
        # Per-bank error counters
        self._single_bit_errors: list[int] = [0] * self._cfg.banks
        self._double_bit_errors: list[int] = [0] * self._cfg.banks
        self._corrections: list[int] = [0] * self._cfg.banks
        self._scrub_count: int = 0

    # ------------------------------------------------------------------
    # Hamming parity computation (simplified for 64-bit words)
    # ------------------------------------------------------------------

    def _compute_hamming(self, data: int) -> int:
        """
        Compute 7 Hamming syndrome bits for a 64-bit data word.

        Parity bit i covers data bits j where bit i is set in (j+1),
        mapping syndrome bits 0-6 directly to error positions 1-64.
        """
        h = 0
        for i in range(7):
            p = 0
            for j in range(64):
                if ((j + 1) >> i) & 1:
                    p ^= (data >> j) & 1
            h |= (p & 1) << i
        return h

    def _compute_parity(self, data: int) -> int:
        """
        Compute full 8-bit SECDED parity (7 Hamming bits + 1 overall bit).

        P_overall (bit 7) is set so the total number of 1-bits across all
        64 data bits and all 7 Hamming parity bits is even.  This allows
        distinguishing single-bit from double-bit errors in decode.
        """
        h = self._compute_hamming(data)
        # Overall parity: XOR of all data bits and 7 Hamming bits
        total = bin(data).count('1') + bin(h).count('1')
        p_overall = total & 1
        return h | (p_overall << 7)

    def encode(self, data: int) -> int:
        """
        Encode a 64-bit data word with 8 parity bits.

        Returns a 72-bit integer (parity in bits 64-71).
        """
        data = data & 0xFFFFFFFFFFFFFFFF
        parity = self._compute_parity(data)
        return data | (parity << 64)

    def decode(
        self,
        codeword: int,
        bank: int = 0,
        address: int = 0,
    ) -> CorrectionResult:
        """
        Decode a 72-bit ECC codeword.

        - No error: returns result with error_type=NONE
        - Single-bit error: corrects and returns corrected word
        - Double-bit error: raises ECCError

        Args:
            codeword: 72-bit integer (data bits 0-63, parity bits 64-71)
            bank: memory bank index (for error tracking)
            address: word address (for logging)
        """
        if bank < 0 or bank >= self._cfg.banks:
            raise ECCError(f"Bank {bank} out of range [0, {self._cfg.banks - 1}]")

        data = codeword & 0xFFFFFFFFFFFFFFFF
        recv_h = (codeword >> 64) & 0x7F          # received 7 Hamming bits
        recv_overall = (codeword >> 71) & 1        # received overall parity bit

        # Recompute Hamming syndrome from received data
        exp_h = self._compute_hamming(data)
        h_syndrome = recv_h ^ exp_h                # non-zero → bit error

        # Recompute overall parity of all received bits (data + 7 Hamming bits)
        total_recv = (bin(data).count('1')
                      + bin(recv_h).count('1')
                      + recv_overall)
        # After encoding, total 1-bits is even; parity=1 means odd → change
        overall_syndrome = total_recv & 1

        if h_syndrome == 0 and overall_syndrome == 0:
            return CorrectionResult(
                corrected=False,
                error_type=BitFlipType.NONE,
                bank=bank, address=address,
                bit_position=None,
                original_word=data,
                corrected_word=data,
            )

        if overall_syndrome == 1:
            # Single-bit error anywhere in the codeword
            self._single_bit_errors[bank] += 1
            self._corrections[bank] += 1

            # h_syndrome 1-64 → data bit (h_syndrome - 1); 0 → parity bit
            error_pos = h_syndrome  # 1-indexed data position
            corrected = data
            if 1 <= error_pos <= 64:
                corrected = data ^ (1 << (error_pos - 1))
            return CorrectionResult(
                corrected=True,
                error_type=BitFlipType.SINGLE_BIT,
                bank=bank, address=address,
                bit_position=error_pos,
                original_word=data,
                corrected_word=corrected,
            )
        else:
            # overall_syndrome == 0 but h_syndrome != 0 → double-bit error
            self._double_bit_errors[bank] += 1
            raise ECCError(
                f"Uncorrectable double-bit ECC error at bank={bank} addr=0x{address:08X}, "
                f"h_syndrome=0x{h_syndrome:02X}"
            )

    # ------------------------------------------------------------------
    # Scrubbing
    # ------------------------------------------------------------------

    def scrub_bank(self, bank: int, words: list[int]) -> tuple[list[int], int]:
        """
        Background scrub: decode each encoded word, re-encode corrected version.

        Returns (scrubbed_words, corrections_made).
        """
        if bank < 0 or bank >= self._cfg.banks:
            raise ECCError(f"Bank {bank} out of range")

        scrubbed = []
        corrections = 0
        for i, word in enumerate(words):
            try:
                result = self.decode(word, bank=bank, address=i)
                if result.corrected:
                    corrections += 1
                    scrubbed.append(self.encode(result.corrected_word))
                else:
                    scrubbed.append(word)
            except ECCError:
                scrubbed.append(word)  # uncorrectable, pass through

        self._scrub_count += 1
        return scrubbed, corrections

    # ------------------------------------------------------------------
    # Counters & diagnostics
    # ------------------------------------------------------------------

    def single_bit_error_count(self, bank: int) -> int:
        return self._single_bit_errors[bank]

    def double_bit_error_count(self, bank: int) -> int:
        return self._double_bit_errors[bank]

    def correction_count(self, bank: int) -> int:
        return self._corrections[bank]

    def total_errors(self) -> int:
        return sum(self._single_bit_errors) + sum(self._double_bit_errors)

    def reset_counters(self) -> None:
        self._single_bit_errors = [0] * self._cfg.banks
        self._double_bit_errors = [0] * self._cfg.banks
        self._corrections = [0] * self._cfg.banks

    @property
    def scrub_count(self) -> int:
        return self._scrub_count

    @property
    def config(self) -> ECCConfig:
        return self._cfg

    def __repr__(self) -> str:
        return (
            f"ECCEngine(SECDED {self._cfg.data_bits}+{self._cfg.parity_bits}bit, "
            f"{self._cfg.banks} banks, "
            f"errors={self.total_errors()})"
        )
