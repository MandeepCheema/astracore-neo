"""
AstraCore Neo Memory — SRAM Controller.

Simulates the chip's 128MB on-chip SRAM:
  - 16 banks × 8MB each
  - ECC: single-bit correction, double-bit detection (SECDED)
  - Dual-port: simultaneous read + write to different banks
  - Bank enable/disable via MEM_CTRL register bits
  - Per-bank ECC error injection for ASIL-D fault testing

Address layout:
  [27:23]  bank select  (bits 27–23, 5 bits → 0–15)
  [22:0]   byte offset within bank (8MB = 2^23 bytes)
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from typing import Dict, List, Optional, Tuple
from .exceptions import BankError, EccError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_BANKS        = 16
BANK_SIZE_BYTES  = 8 * 1024 * 1024   # 8 MB
TOTAL_SIZE_BYTES = NUM_BANKS * BANK_SIZE_BYTES  # 128 MB
BANK_ADDR_BITS   = 23                 # log2(8MB)
BANK_ADDR_MASK   = (1 << BANK_ADDR_BITS) - 1


def _bank_and_offset(addr: int) -> Tuple[int, int]:
    """Split a flat address into (bank_index, byte_offset_within_bank)."""
    bank = (addr >> BANK_ADDR_BITS) & 0xF
    offset = addr & BANK_ADDR_MASK
    return bank, offset


# ---------------------------------------------------------------------------
# ECC helpers  (SECDED simulation — not full Hamming, just fault injection)
# ---------------------------------------------------------------------------

def _ecc_encode(data: bytes) -> bytes:
    """Simulate ECC encoding (identity — real chip handles in HW)."""
    return data  # In simulation we track errors via injection, not real codes


class SRAMBank:
    """
    Single 8MB SRAM bank with ECC fault injection support.

    In simulation, storage is a bytearray.  ECC is modelled as a set of
    injected single-bit and double-bit error addresses rather than actual
    Hamming codes — this lets tests exercise the correction/detection paths
    without implementing a full SECDED encoder.
    """

    def __init__(self, bank_id: int) -> None:
        self.bank_id = bank_id
        self._data = bytearray(BANK_SIZE_BYTES)
        self._enabled: bool = True
        # ECC fault maps: addr → bit_position(s)
        self._single_bit_errors: Dict[int, int] = {}   # correctable
        self._double_bit_errors: Dict[int, Tuple[int, int]] = {}  # uncorrectable
        self.ecc_corrections: int = 0   # corrected error count
        self.ecc_detections: int = 0    # detected-but-uncorrectable count

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def read_bytes(self, offset: int, length: int) -> bytes:
        self._check_enabled()
        self._check_bounds(offset, length)
        raw = bytes(self._data[offset: offset + length])
        return self._apply_ecc_read(offset, raw)

    def write_bytes(self, offset: int, data: bytes) -> None:
        self._check_enabled()
        self._check_bounds(offset, len(data))
        self._data[offset: offset + len(data)] = data

    def read_word(self, offset: int) -> int:
        """Read 4-byte little-endian word."""
        b = self.read_bytes(offset, 4)
        return int.from_bytes(b, "little")

    def write_word(self, offset: int, value: int) -> None:
        """Write 4-byte little-endian word."""
        self.write_bytes(offset, value.to_bytes(4, "little"))

    def zero_fill(self) -> None:
        self._data[:] = b"\x00" * BANK_SIZE_BYTES
        self._single_bit_errors.clear()
        self._double_bit_errors.clear()

    # ------------------------------------------------------------------
    # ECC fault injection (for ASIL-D testing)
    # ------------------------------------------------------------------

    def inject_single_bit_error(self, offset: int, bit: int = 0) -> None:
        """Mark *offset* as having a correctable single-bit ECC error."""
        self._single_bit_errors[offset] = bit

    def inject_double_bit_error(self, offset: int, bits: Tuple[int, int] = (0, 1)) -> None:
        """Mark *offset* as having an uncorrectable double-bit ECC error."""
        self._double_bit_errors[offset] = bits

    def clear_ecc_faults(self) -> None:
        self._single_bit_errors.clear()
        self._double_bit_errors.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_ecc_read(self, offset: int, raw: bytes) -> bytes:
        result = bytearray(raw)
        for i in range(len(raw)):
            addr = offset + i
            if addr in self._double_bit_errors:
                self.ecc_detections += 1
                raise EccError(addr, self.bank_id)
            if addr in self._single_bit_errors:
                # Correct: flip the injected bit back
                bit = self._single_bit_errors.pop(addr)
                result[i] ^= (1 << (bit % 8))
                self.ecc_corrections += 1
        return bytes(result)

    def _check_enabled(self) -> None:
        if not self._enabled:
            raise BankError(f"SRAM bank {self.bank_id} is disabled")

    def _check_bounds(self, offset: int, length: int) -> None:
        if offset < 0 or length < 0 or offset + length > BANK_SIZE_BYTES:
            raise BankError(
                f"Access out of bounds: bank={self.bank_id} "
                f"offset=0x{offset:06X} len={length} "
                f"(bank size=0x{BANK_SIZE_BYTES:06X})"
            )


# ---------------------------------------------------------------------------
# SRAM Controller — owns all 16 banks and the HAL register interface
# ---------------------------------------------------------------------------

class SRAMController:
    """
    128 MB SRAM controller: 16 banks of 8 MB each.

    Flat address space 0x0000_0000–0x07FF_FFFF maps to:
        bank = addr >> 23
        offset = addr & 0x7FFFFF

    Usage::

        ctrl = SRAMController(dev)          # pass AstraCoreDevice for register sync
        ctrl.write(0x00_0000, b"hello")
        data = ctrl.read(0x00_0000, 5)
        ctrl.disable_bank(3)
        ctrl.enable_bank(3)
    """

    def __init__(self, dev=None) -> None:
        """
        *dev* is an optional AstraCoreDevice.  If provided, bank enable/disable
        updates are mirrored to MEM_CTRL register (0x0038).
        """
        self._dev = dev
        self._banks: List[SRAMBank] = [SRAMBank(i) for i in range(NUM_BANKS)]

    # ------------------------------------------------------------------
    # Bank control
    # ------------------------------------------------------------------

    def enable_bank(self, bank_id: int) -> None:
        self._validate_bank(bank_id)
        self._banks[bank_id].enable()
        self._sync_mem_ctrl()

    def disable_bank(self, bank_id: int) -> None:
        self._validate_bank(bank_id)
        self._banks[bank_id].disable()
        self._sync_mem_ctrl()

    def bank_enabled(self, bank_id: int) -> bool:
        self._validate_bank(bank_id)
        return self._banks[bank_id].enabled

    def enabled_bank_mask(self) -> int:
        """Return 16-bit mask of enabled banks."""
        mask = 0
        for i, b in enumerate(self._banks):
            if b.enabled:
                mask |= (1 << i)
        return mask

    # ------------------------------------------------------------------
    # Flat address-space read / write
    # ------------------------------------------------------------------

    def read(self, addr: int, length: int) -> bytes:
        """Read *length* bytes from flat address *addr*."""
        self._validate_addr(addr, length)
        bank_id, offset = _bank_and_offset(addr)
        return self._banks[bank_id].read_bytes(offset, length)

    def write(self, addr: int, data: bytes) -> None:
        """Write *data* to flat address *addr*."""
        self._validate_addr(addr, len(data))
        bank_id, offset = _bank_and_offset(addr)
        self._banks[bank_id].write_bytes(offset, data)

    def read_word(self, addr: int) -> int:
        bank_id, offset = _bank_and_offset(addr)
        return self._banks[bank_id].read_word(offset)

    def write_word(self, addr: int, value: int) -> None:
        bank_id, offset = _bank_and_offset(addr)
        self._banks[bank_id].write_word(offset, value)

    # ------------------------------------------------------------------
    # Dual-port simulation: simultaneous read+write to different banks
    # ------------------------------------------------------------------

    def dual_port_transfer(
        self,
        read_addr: int, read_len: int,
        write_addr: int, write_data: bytes,
    ) -> bytes:
        """
        Simulate dual-port access: read and write occur in the same cycle
        provided they target different banks.  Same-bank access serialises.
        """
        read_bank, _ = _bank_and_offset(read_addr)
        write_bank, _ = _bank_and_offset(write_addr)
        if read_bank == write_bank:
            # Same bank — serialise (write first, then read)
            self.write(write_addr, write_data)
            return self.read(read_addr, read_len)
        else:
            # Different banks — true parallel (order doesn't matter)
            result = self.read(read_addr, read_len)
            self.write(write_addr, write_data)
            return result

    # ------------------------------------------------------------------
    # ECC interface
    # ------------------------------------------------------------------

    def bank(self, bank_id: int) -> SRAMBank:
        self._validate_bank(bank_id)
        return self._banks[bank_id]

    def total_ecc_corrections(self) -> int:
        return sum(b.ecc_corrections for b in self._banks)

    def total_ecc_detections(self) -> int:
        return sum(b.ecc_detections for b in self._banks)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        for b in self._banks:
            b.zero_fill()
            b.enable()
        self._sync_mem_ctrl()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_bank(self, bank_id: int) -> None:
        if not (0 <= bank_id < NUM_BANKS):
            raise BankError(f"Invalid bank_id {bank_id}: must be 0–{NUM_BANKS - 1}")

    def _validate_addr(self, addr: int, length: int) -> None:
        if addr < 0 or length < 0 or addr + length > TOTAL_SIZE_BYTES:
            raise BankError(
                f"Address 0x{addr:08X}+{length} out of SRAM range "
                f"(max 0x{TOTAL_SIZE_BYTES:08X})"
            )

    def _sync_mem_ctrl(self) -> None:
        """Mirror bank enable mask to HAL MEM_CTRL register if device attached."""
        if self._dev is not None:
            self._dev.regs.write(0x0038, self.enabled_bank_mask())
