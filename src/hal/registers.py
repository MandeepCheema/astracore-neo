"""
AstraCore Neo HAL — Register map and access layer.

Models the chip's memory-mapped register space as an address→value dict.
Supports full 32-bit reads/writes and named bitfield access.
"""

from __future__ import annotations
from typing import Dict, Tuple
from .exceptions import RegisterError

# ---------------------------------------------------------------------------
# Register address map (word-addressed, 32-bit registers)
# ---------------------------------------------------------------------------
REGISTER_MAP: Dict[int, str] = {
    0x0000: "CHIP_ID",           # R/O  — always 0xA2_NE0_1
    0x0004: "CHIP_REV",          # R/O  — silicon revision
    0x0008: "CTRL",              # R/W  — global control
    0x000C: "STATUS",            # R/O  — global status
    0x0010: "CLK_CTRL",          # R/W  — clock / DVFS control
    0x0014: "CLK_STATUS",        # R/O  — PLL lock, current freq
    0x0018: "PWR_CTRL",          # R/W  — power domain enables
    0x001C: "PWR_STATUS",        # R/O  — power domain status
    0x0020: "RESET_CTRL",        # W/O  — write 1 to reset domains
    0x0024: "INT_ENABLE",        # R/W  — interrupt enable mask
    0x0028: "INT_STATUS",        # R/O  — pending interrupt bits
    0x002C: "INT_CLEAR",         # W/O  — write 1 to clear interrupt
    0x0030: "MAC_CTRL",          # R/W  — MAC array config
    0x0034: "MAC_STATUS",        # R/O  — MAC busy/idle, utilisation
    0x0038: "MEM_CTRL",          # R/W  — SRAM bank enables
    0x003C: "MEM_STATUS",        # R/O  — ECC errors, bank status
    0x0040: "THERMAL_STATUS",    # R/O  — current temp reading
    0x0044: "THERMAL_THRESH",    # R/W  — thermal throttle threshold
    0x0048: "SECURITY_STATUS",   # R/O  — secure boot, TEE status
    0x004C: "OTA_CTRL",          # R/W  — OTA control
}

# Default power-on reset values
_RESET_VALUES: Dict[int, int] = {
    0x0000: 0xA2_4E_E0_01,   # CHIP_ID
    0x0004: 0x0000_0013,     # CHIP_REV  (revision 1.3)
    0x0008: 0x0000_0000,     # CTRL
    0x000C: 0x0000_0001,     # STATUS  bit0=ready
    0x0010: 0x0000_0320,     # CLK_CTRL  default 3.2 GHz encoded as 800 (×4 MHz)
    0x0014: 0x0000_0320,     # CLK_STATUS
    0x0018: 0x0000_00FF,     # PWR_CTRL  all domains on
    0x001C: 0x0000_00FF,     # PWR_STATUS
    0x0020: 0x0000_0000,     # RESET_CTRL
    0x0024: 0x0000_0000,     # INT_ENABLE
    0x0028: 0x0000_0000,     # INT_STATUS
    0x002C: 0x0000_0000,     # INT_CLEAR
    0x0030: 0x0000_0001,     # MAC_CTRL  enabled
    0x0034: 0x0000_0000,     # MAC_STATUS idle
    0x0038: 0x0000_FFFF,     # MEM_CTRL  all 16 banks enabled
    0x003C: 0x0000_0000,     # MEM_STATUS no errors
    0x0040: 0x0000_0019,     # THERMAL_STATUS  25°C
    0x0044: 0x0000_007D,     # THERMAL_THRESH  125°C
    0x0048: 0x0000_0000,     # SECURITY_STATUS
    0x004C: 0x0000_0000,     # OTA_CTRL
}

# Read-only registers — writes are silently ignored with an error raise
_READ_ONLY = {0x0000, 0x0004, 0x000C, 0x0014, 0x001C, 0x0028, 0x0034, 0x003C, 0x0040, 0x0048}
# Write-only registers
_WRITE_ONLY = {0x0020, 0x002C}


class RegisterFile:
    """In-process simulation of the AstraCore Neo register space."""

    def __init__(self) -> None:
        self._regs: Dict[int, int] = dict(_RESET_VALUES)

    # ------------------------------------------------------------------
    # Core access
    # ------------------------------------------------------------------

    def read(self, addr: int) -> int:
        """Read 32-bit register at *addr*."""
        if addr not in REGISTER_MAP:
            raise RegisterError(f"Invalid register address: 0x{addr:04X}")
        if addr in _WRITE_ONLY:
            raise RegisterError(f"Register 0x{addr:04X} ({REGISTER_MAP[addr]}) is write-only")
        return self._regs.get(addr, 0)

    def write(self, addr: int, value: int) -> None:
        """Write 32-bit *value* to register at *addr*."""
        if addr not in REGISTER_MAP:
            raise RegisterError(f"Invalid register address: 0x{addr:04X}")
        if addr in _READ_ONLY:
            raise RegisterError(f"Register 0x{addr:04X} ({REGISTER_MAP[addr]}) is read-only")
        if not (0 <= value <= 0xFFFF_FFFF):
            raise RegisterError(f"Value 0x{value:X} out of 32-bit range")
        self._regs[addr] = value

    # ------------------------------------------------------------------
    # Bitfield helpers
    # ------------------------------------------------------------------

    def read_field(self, addr: int, msb: int, lsb: int) -> int:
        """Extract bitfield [msb:lsb] from register at *addr*."""
        if msb < lsb or msb > 31 or lsb < 0:
            raise RegisterError(f"Invalid bitfield [{msb}:{lsb}]")
        raw = self.read(addr)
        mask = (1 << (msb - lsb + 1)) - 1
        return (raw >> lsb) & mask

    def write_field(self, addr: int, msb: int, lsb: int, value: int) -> None:
        """Write *value* into bitfield [msb:lsb] of register at *addr*."""
        if msb < lsb or msb > 31 or lsb < 0:
            raise RegisterError(f"Invalid bitfield [{msb}:{lsb}]")
        width = msb - lsb + 1
        max_val = (1 << width) - 1
        if not (0 <= value <= max_val):
            raise RegisterError(f"Value {value} too wide for {width}-bit field [{msb}:{lsb}]")
        current = self.read(addr)
        mask = max_val << lsb
        self.write(addr, (current & ~mask) | (value << lsb))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def _hw_write(self, addr: int, value: int) -> None:
        """
        Internal hardware-side write — bypasses read-only protection.

        Use only from device simulation code (device.py) to model the chip
        updating its own status registers (STATUS, CLK_STATUS, etc.).
        External software must use write() which enforces read-only guards.
        """
        if addr not in REGISTER_MAP:
            raise RegisterError(f"Invalid register address: 0x{addr:04X}")
        if not (0 <= value <= 0xFFFF_FFFF):
            raise RegisterError(f"Value 0x{value:X} out of 32-bit range")
        self._regs[addr] = value

    def reset(self) -> None:
        """Restore all registers to power-on reset values."""
        self._regs = dict(_RESET_VALUES)

    def dump(self) -> Dict[str, int]:
        """Return all registers as {name: value} for debugging."""
        return {REGISTER_MAP[addr]: val for addr, val in sorted(self._regs.items())}

    def named_read(self, name: str) -> int:
        """Read register by symbolic name (e.g. 'CTRL')."""
        for addr, n in REGISTER_MAP.items():
            if n == name:
                return self.read(addr)
        raise RegisterError(f"Unknown register name: {name!r}")

    def named_write(self, name: str, value: int) -> None:
        """Write register by symbolic name."""
        for addr, n in REGISTER_MAP.items():
            if n == name:
                self.write(addr, value)
                return
        raise RegisterError(f"Unknown register name: {name!r}")
