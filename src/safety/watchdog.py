"""
AstraCore Neo — Watchdog Timer simulation.

Models the hardware watchdog used in ASIL-D systems:
  - Configurable timeout window (min/max service window)
  - Windowed watchdog: too-early service also triggers fault
  - Service token validation (prevents random kicks from runaway code)
  - Escalating response: interrupt → NMI → chip reset
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from .exceptions import WatchdogError


class WatchdogResponse(Enum):
    INTERRUPT = auto()    # first timeout — trigger recoverable interrupt
    NMI       = auto()    # second timeout — non-maskable interrupt
    RESET     = auto()    # third timeout — full chip reset


@dataclass
class WatchdogConfig:
    """Watchdog timer configuration."""
    timeout_ms: float = 100.0         # must be serviced within this window
    window_open_ms: float = 50.0      # earliest allowed service time (windowed WDT)
    token: int = 0xA5A5               # expected service token
    max_timeouts_before_reset: int = 3


class WatchdogTimer:
    """
    Simulated windowed hardware watchdog timer.

    The watchdog must be serviced (kicked) within a defined time window:
      - Too early (before window_open_ms): raises WatchdogError (window violation)
      - Within window: accepted, timer resets
      - Too late (after timeout_ms): raises WatchdogError (timeout)

    Usage::

        wdt = WatchdogTimer()
        wdt.start()
        # ... do work ...
        wdt.service(token=0xA5A5)
        wdt.stop()
    """

    def __init__(self, config: Optional[WatchdogConfig] = None) -> None:
        self._cfg = config or WatchdogConfig()
        self._running: bool = False
        self._start_time: float = 0.0
        self._last_service_time: float = 0.0
        self._timeout_count: int = 0
        self._service_count: int = 0
        self._early_kick_count: int = 0
        # For simulation: allow manual time injection
        self._simulated_elapsed: Optional[float] = None

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the watchdog timer."""
        if self._running:
            raise WatchdogError("Watchdog already running")
        self._running = True
        self._start_time = time.monotonic()
        self._last_service_time = self._start_time
        self._simulated_elapsed = None

    def stop(self) -> None:
        """Stop (disable) the watchdog timer. Only valid during maintenance."""
        if not self._running:
            raise WatchdogError("Watchdog not running")
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Simulation helpers (for deterministic testing)
    # ------------------------------------------------------------------

    def _inject_elapsed(self, elapsed_ms: float) -> None:
        """
        Inject a simulated elapsed time (ms since last service).
        Used in tests to simulate time passage without real sleeping.
        """
        self._simulated_elapsed = elapsed_ms

    def _elapsed_ms(self) -> float:
        """Return ms since last service (real or simulated)."""
        if self._simulated_elapsed is not None:
            return self._simulated_elapsed
        return (time.monotonic() - self._last_service_time) * 1000.0

    # ------------------------------------------------------------------
    # Service (kick)
    # ------------------------------------------------------------------

    def service(self, token: int) -> None:
        """
        Service (kick) the watchdog with a security token.

        Args:
            token: must match configured token value

        Raises:
            WatchdogError: wrong token, too early, or timeout already expired
        """
        if not self._running:
            raise WatchdogError("Cannot service: watchdog not running")

        if token != self._cfg.token:
            raise WatchdogError(
                f"Wrong service token: expected 0x{self._cfg.token:04X}, "
                f"got 0x{token:04X}"
            )

        elapsed = self._elapsed_ms()

        # Check: has timeout already passed?
        if elapsed > self._cfg.timeout_ms:
            self._timeout_count += 1
            raise WatchdogError(
                f"Watchdog timeout: {elapsed:.1f}ms elapsed "
                f"(limit={self._cfg.timeout_ms}ms)"
            )

        # Windowed WDT: too early is also a fault
        if elapsed < self._cfg.window_open_ms:
            self._early_kick_count += 1
            raise WatchdogError(
                f"Watchdog window violation: serviced too early at "
                f"{elapsed:.1f}ms (window opens at {self._cfg.window_open_ms}ms)"
            )

        # Valid service
        self._service_count += 1
        self._last_service_time = time.monotonic()
        self._simulated_elapsed = None

    def check_timeout(self) -> bool:
        """
        Check if the watchdog has timed out without raising.
        Returns True if timeout has occurred.
        """
        if not self._running:
            return False
        return self._elapsed_ms() > self._cfg.timeout_ms

    # ------------------------------------------------------------------
    # Response escalation
    # ------------------------------------------------------------------

    def escalation_level(self) -> WatchdogResponse:
        """Return current escalation level based on timeout count."""
        if self._timeout_count == 0:
            return WatchdogResponse.INTERRUPT
        elif self._timeout_count == 1:
            return WatchdogResponse.INTERRUPT
        elif self._timeout_count == 2:
            return WatchdogResponse.NMI
        else:
            return WatchdogResponse.RESET

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    @property
    def timeout_count(self) -> int:
        return self._timeout_count

    @property
    def service_count(self) -> int:
        return self._service_count

    @property
    def early_kick_count(self) -> int:
        return self._early_kick_count

    @property
    def config(self) -> WatchdogConfig:
        return self._cfg

    def __repr__(self) -> str:
        return (
            f"WatchdogTimer(timeout={self._cfg.timeout_ms}ms, "
            f"window={self._cfg.window_open_ms}ms, "
            f"running={self._running}, "
            f"services={self._service_count})"
        )
