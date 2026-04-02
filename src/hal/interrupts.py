"""
AstraCore Neo HAL — Interrupt controller simulation.

Models a 32-bit interrupt controller with enable masking, pending status,
and synchronous handler dispatch.

Interrupt bit assignments:
  Bit 0  — MAC_DONE       compute array completed
  Bit 1  — DMA_DONE       DMA transfer complete
  Bit 2  — MEM_ECC_ERR    uncorrectable ECC error
  Bit 3  — THERMAL_WARN   temperature above threshold
  Bit 4  — THERMAL_CRIT   critical thermal shutdown
  Bit 5  — SAFETY_TMR     safety watchdog expired
  Bit 6  — V2X_RX         V2X frame received
  Bit 7  — OTA_READY      OTA package validated
  Bit 8  — DMS_ALERT      driver monitoring alert
  Bit 9  — SECURE_BOOT_OK secure boot completed
  Bits 10–31 — reserved
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional
from .exceptions import InterruptError

# Symbolic IRQ numbers
IRQ_MAC_DONE       = 0
IRQ_DMA_DONE       = 1
IRQ_MEM_ECC_ERR    = 2
IRQ_THERMAL_WARN   = 3
IRQ_THERMAL_CRIT   = 4
IRQ_SAFETY_TMR     = 5
IRQ_V2X_RX         = 6
IRQ_OTA_READY      = 7
IRQ_DMS_ALERT      = 8
IRQ_SECURE_BOOT_OK = 9

IRQ_NAMES: Dict[int, str] = {
    IRQ_MAC_DONE:       "MAC_DONE",
    IRQ_DMA_DONE:       "DMA_DONE",
    IRQ_MEM_ECC_ERR:    "MEM_ECC_ERR",
    IRQ_THERMAL_WARN:   "THERMAL_WARN",
    IRQ_THERMAL_CRIT:   "THERMAL_CRIT",
    IRQ_SAFETY_TMR:     "SAFETY_TMR",
    IRQ_V2X_RX:         "V2X_RX",
    IRQ_OTA_READY:      "OTA_READY",
    IRQ_DMS_ALERT:      "DMS_ALERT",
    IRQ_SECURE_BOOT_OK: "SECURE_BOOT_OK",
}

Handler = Callable[[int], None]


class InterruptController:
    """Simulated interrupt controller for AstraCore Neo."""

    def __init__(self) -> None:
        self._enable_mask: int = 0          # INT_ENABLE register
        self._pending: int = 0              # INT_STATUS register
        self._masked: int = 0               # software-masked IRQs
        self._handlers: Dict[int, List[Handler]] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def enable(self, irq: int) -> None:
        """Enable interrupt *irq* in the controller."""
        self._validate(irq)
        self._enable_mask |= (1 << irq)

    def disable(self, irq: int) -> None:
        """Disable (mask) interrupt *irq*."""
        self._validate(irq)
        self._enable_mask &= ~(1 << irq)

    def register_handler(self, irq: int, handler: Handler) -> None:
        """Attach *handler* to interrupt *irq*. Multiple handlers allowed."""
        self._validate(irq)
        self._handlers.setdefault(irq, []).append(handler)

    def unregister_handlers(self, irq: int) -> None:
        """Remove all handlers for *irq*."""
        self._validate(irq)
        self._handlers.pop(irq, None)

    # ------------------------------------------------------------------
    # Fire / clear
    # ------------------------------------------------------------------

    def fire(self, irq: int) -> None:
        """Assert interrupt *irq*. Dispatches handlers if enabled."""
        self._validate(irq)
        self._pending |= (1 << irq)
        if self._enable_mask & (1 << irq):
            for handler in self._handlers.get(irq, []):
                handler(irq)

    def clear(self, irq: int) -> None:
        """Clear pending status for *irq*."""
        self._validate(irq)
        self._pending &= ~(1 << irq)

    def clear_all(self) -> None:
        """Clear all pending interrupts."""
        self._pending = 0

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def is_pending(self, irq: int) -> bool:
        """Return True if *irq* is pending (asserted but not cleared)."""
        self._validate(irq)
        return bool(self._pending & (1 << irq))

    def is_enabled(self, irq: int) -> bool:
        self._validate(irq)
        return bool(self._enable_mask & (1 << irq))

    @property
    def pending_mask(self) -> int:
        """Raw 32-bit pending register."""
        return self._pending

    @property
    def enable_mask(self) -> int:
        """Raw 32-bit enable register."""
        return self._enable_mask

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._enable_mask = 0
        self._pending = 0
        self._masked = 0
        self._handlers.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(irq: int) -> None:
        if not (0 <= irq <= 31):
            raise InterruptError(f"IRQ {irq} out of range [0–31]")
        if irq > 9 and irq not in IRQ_NAMES:
            raise InterruptError(f"IRQ {irq} is reserved")
