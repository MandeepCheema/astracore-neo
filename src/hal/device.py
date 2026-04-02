"""
AstraCore Neo HAL — Top-level device abstraction.

AstraCoreDevice is the single entry point for all HAL operations.
It owns the register file and interrupt controller, and manages the
chip's power/clock state machine.

Power states:
    OFF → RESET → IDLE → ACTIVE → LOW_POWER
                ↑__________________________|  (any state → RESET via hard reset)
"""

from __future__ import annotations
from enum import Enum, auto
from typing import Optional
import time

from .registers import RegisterFile, REGISTER_MAP
from .interrupts import InterruptController
from .exceptions import DeviceError, ClockError

# ---------------------------------------------------------------------------
# Clock constants (chip spec: 2.5–3.2 GHz)
# ---------------------------------------------------------------------------
CLK_MIN_GHZ = 2.5
CLK_MAX_GHZ = 3.2
CLK_DEFAULT_GHZ = 3.2

# Low-power mode: 500 MHz for always-on DMS
CLK_LOW_POWER_GHZ = 0.5


class PowerState(Enum):
    OFF       = auto()
    RESET     = auto()
    IDLE      = auto()
    ACTIVE    = auto()
    LOW_POWER = auto()


class AstraCoreDevice:
    """
    Simulated AstraCore Neo inference accelerator device.

    Usage::

        dev = AstraCoreDevice()
        dev.power_on()
        dev.set_clock_ghz(3.2)
        # ... interact via dev.regs and dev.irq ...
        dev.power_off()
    """

    CHIP_ID  = 0xA2_4E_E0_01
    CHIP_REV = 0x0000_0013   # 1.3

    def __init__(self, name: str = "astracore-neo-0") -> None:
        self.name = name
        self._state = PowerState.OFF
        self._clock_ghz: float = 0.0
        self._power_on_time: Optional[float] = None

        self.regs = RegisterFile()
        self.irq  = InterruptController()

    # ------------------------------------------------------------------
    # Power control
    # ------------------------------------------------------------------

    def power_on(self) -> None:
        """Transition OFF → RESET → IDLE."""
        if self._state != PowerState.OFF:
            raise DeviceError(f"power_on() called in state {self._state.name}")
        self._state = PowerState.RESET
        self.regs.reset()
        self.irq.reset()
        self._clock_ghz = CLK_DEFAULT_GHZ
        self._power_on_time = time.monotonic()
        self._state = PowerState.IDLE

    def power_off(self) -> None:
        """Transition any state → OFF."""
        self._state = PowerState.OFF
        self._clock_ghz = 0.0
        self._power_on_time = None

    def reset(self) -> None:
        """Hard reset: any state → RESET → IDLE (preserves power)."""
        if self._state == PowerState.OFF:
            raise DeviceError("Cannot reset a powered-off device")
        self._state = PowerState.RESET
        self.regs.reset()
        self.irq.reset()
        self._clock_ghz = CLK_DEFAULT_GHZ
        self._state = PowerState.IDLE

    # ------------------------------------------------------------------
    # Active / low-power mode
    # ------------------------------------------------------------------

    def start(self) -> None:
        """IDLE → ACTIVE."""
        self._require_state(PowerState.IDLE)
        self._state = PowerState.ACTIVE
        # HW updates STATUS register — bit1 = active
        self.regs._hw_write(0x000C, 0x0000_0003)

    def stop(self) -> None:
        """ACTIVE → IDLE."""
        self._require_state(PowerState.ACTIVE)
        self._state = PowerState.IDLE
        self.regs._hw_write(0x000C, 0x0000_0001)

    def enter_low_power(self) -> None:
        """IDLE/ACTIVE → LOW_POWER (256 MACs @ 500 MHz for always-on DMS)."""
        if self._state not in (PowerState.IDLE, PowerState.ACTIVE):
            raise DeviceError(f"Cannot enter low-power from {self._state.name}")
        self._state = PowerState.LOW_POWER
        self._clock_ghz = CLK_LOW_POWER_GHZ
        self.regs.write(0x0010, int(CLK_LOW_POWER_GHZ * 1000))  # MHz encoding
        self.regs._hw_write(0x000C, 0x0000_0005)   # bit2 = low-power flag

    def exit_low_power(self) -> None:
        """LOW_POWER → IDLE."""
        self._require_state(PowerState.LOW_POWER)
        self._clock_ghz = CLK_DEFAULT_GHZ
        self.regs.write(0x0010, int(CLK_DEFAULT_GHZ * 1000))
        self.regs._hw_write(0x000C, 0x0000_0001)
        self._state = PowerState.IDLE

    # ------------------------------------------------------------------
    # Clock / DVFS
    # ------------------------------------------------------------------

    def set_clock_ghz(self, freq_ghz: float) -> None:
        """
        Set compute clock via DVFS.

        Valid range: 2.5–3.2 GHz (normal) or 0.5 GHz (low-power mode).
        """
        self._require_powered()
        lp = (self._state == PowerState.LOW_POWER)
        if lp:
            if freq_ghz != CLK_LOW_POWER_GHZ:
                raise ClockError(
                    f"In LOW_POWER mode clock is fixed at {CLK_LOW_POWER_GHZ} GHz"
                )
        else:
            if not (CLK_MIN_GHZ <= freq_ghz <= CLK_MAX_GHZ):
                raise ClockError(
                    f"Clock {freq_ghz} GHz out of range [{CLK_MIN_GHZ}–{CLK_MAX_GHZ}] GHz"
                )
        self._clock_ghz = freq_ghz
        # Encode as MHz integer in CLK_CTRL (R/W) and CLK_STATUS (HW-written)
        self.regs.write(0x0010, int(freq_ghz * 1000))
        self.regs._hw_write(0x0014, int(freq_ghz * 1000))  # CLK_STATUS mirrors

    @property
    def clock_ghz(self) -> float:
        return self._clock_ghz

    # ------------------------------------------------------------------
    # Identity / status
    # ------------------------------------------------------------------

    @property
    def state(self) -> PowerState:
        return self._state

    @property
    def chip_id(self) -> int:
        return self.regs.read(0x0000)

    @property
    def chip_rev(self) -> int:
        return self.regs.read(0x0004)

    @property
    def uptime_seconds(self) -> float:
        if self._power_on_time is None:
            return 0.0
        return time.monotonic() - self._power_on_time

    def __repr__(self) -> str:
        return (
            f"AstraCoreDevice(name={self.name!r}, "
            f"state={self._state.name}, "
            f"clock={self._clock_ghz:.2f}GHz)"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_state(self, expected: PowerState) -> None:
        if self._state != expected:
            raise DeviceError(
                f"Operation requires state {expected.name}, "
                f"current state is {self._state.name}"
            )

    def _require_powered(self) -> None:
        if self._state == PowerState.OFF:
            raise DeviceError("Device is powered off")
