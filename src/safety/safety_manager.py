"""
AstraCore Neo — Safety Manager.

Centralised ASIL-D safety coordinator:
  - Aggregates events from ECC, TMR, watchdog, clock monitor
  - Severity classification (INFO → WARNING → CRITICAL → FATAL)
  - Safety state machine: NORMAL → DEGRADED → SAFE_STATE → SHUTDOWN
  - Event log with timestamps
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .exceptions import SafetyError


class SafetySeverity(Enum):
    INFO     = 0
    WARNING  = 1
    CRITICAL = 2
    FATAL    = 3


class SafetyState(Enum):
    NORMAL     = auto()   # all systems nominal
    DEGRADED   = auto()   # minor fault, continue with monitoring
    SAFE_STATE = auto()   # significant fault, restrict operation
    SHUTDOWN   = auto()   # fatal fault, chip must be reset


@dataclass
class SafetyEvent:
    """A logged safety event."""
    severity: SafetySeverity
    source: str               # e.g. "ECC", "TMR", "WATCHDOG", "CLOCK"
    message: str
    timestamp_us: float
    state_after: SafetyState


class SafetyManager:
    """
    Centralised safety manager for ASIL-D compliance.

    Collects safety events from subsystems, updates chip safety state,
    and provides a queryable event log.

    Usage::

        sm = SafetyManager()
        sm.start()
        sm.report_event("ECC", SafetySeverity.WARNING, "Single-bit error corrected bank=0")
        sm.report_event("WATCHDOG", SafetySeverity.FATAL, "Timeout: no service in 100ms")
        assert sm.state == SafetyState.SHUTDOWN
    """

    # Severity → state transition table
    _SEVERITY_TO_STATE: dict[SafetySeverity, SafetyState] = {
        SafetySeverity.INFO:     SafetyState.NORMAL,
        SafetySeverity.WARNING:  SafetyState.DEGRADED,
        SafetySeverity.CRITICAL: SafetyState.SAFE_STATE,
        SafetySeverity.FATAL:    SafetyState.SHUTDOWN,
    }

    def __init__(self) -> None:
        self._state: SafetyState = SafetyState.NORMAL
        self._running: bool = False
        self._event_log: list[SafetyEvent] = []
        self._event_counts: dict[SafetySeverity, int] = {
            s: 0 for s in SafetySeverity
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            raise SafetyError("Safety manager already running")
        self._running = True
        self._state = SafetyState.NORMAL

    def shutdown(self) -> None:
        if not self._running:
            raise SafetyError("Safety manager not running")
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Event reporting
    # ------------------------------------------------------------------

    def report_event(
        self,
        source: str,
        severity: SafetySeverity,
        message: str,
    ) -> SafetyEvent:
        """
        Report a safety event from a subsystem.

        The safety state is escalated (never downgraded) based on severity.
        Returns the logged SafetyEvent.
        """
        if not self._running:
            raise SafetyError("Safety manager not running — call start() first")

        # State only escalates, never goes back down
        new_state = self._SEVERITY_TO_STATE[severity]
        if new_state.value > self._state.value:
            self._state = new_state

        self._event_counts[severity] += 1

        event = SafetyEvent(
            severity=severity,
            source=source,
            message=message,
            timestamp_us=time.monotonic() * 1e6,
            state_after=self._state,
        )
        self._event_log.append(event)
        return event

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    @property
    def state(self) -> SafetyState:
        return self._state

    def is_safe_to_operate(self) -> bool:
        """Return True if the chip can continue normal operation."""
        return self._state in (SafetyState.NORMAL, SafetyState.DEGRADED)

    def reset_state(self) -> None:
        """Reset safety state to NORMAL (only valid after SHUTDOWN is resolved)."""
        self._state = SafetyState.NORMAL

    # ------------------------------------------------------------------
    # Event log & counters
    # ------------------------------------------------------------------

    def event_log(self) -> list[SafetyEvent]:
        return list(self._event_log)

    def event_count(self, severity: SafetySeverity) -> int:
        return self._event_counts[severity]

    def total_events(self) -> int:
        return len(self._event_log)

    def clear_log(self) -> None:
        self._event_log.clear()
        self._event_counts = {s: 0 for s in SafetySeverity}

    def events_by_source(self, source: str) -> list[SafetyEvent]:
        return [e for e in self._event_log if e.source == source]

    def highest_severity(self) -> Optional[SafetySeverity]:
        """Return the highest severity seen, or None if no events."""
        if not self._event_log:
            return None
        return max((e.severity for e in self._event_log), key=lambda s: s.value)

    def __repr__(self) -> str:
        return (
            f"SafetyManager(state={self._state.name}, "
            f"events={self.total_events()}, "
            f"running={self._running})"
        )
