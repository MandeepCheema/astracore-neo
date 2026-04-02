"""
AstraCore Neo — Structured Telemetry Logger.

Models the chip's on-die event logging subsystem:
  - Severity levels: DEBUG → INFO → WARNING → ERROR → CRITICAL
  - Ring buffer with configurable capacity (auto-evicts oldest on overflow)
  - Source tagging (module name, subsystem ID)
  - Structured log entries with timestamp, level, source, message, metadata
  - Snapshot export (list of entries)
  - Per-level counters for quick health checks
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

from .exceptions import LoggerError


class LogLevel(IntEnum):
    DEBUG    = 0
    INFO     = 1
    WARNING  = 2
    ERROR    = 3
    CRITICAL = 4


@dataclass
class LogEntry:
    """A single structured log entry."""
    level: LogLevel
    source: str
    message: str
    timestamp_us: float
    sequence: int
    metadata: dict[str, Any] = field(default_factory=dict)


class TelemetryLogger:
    """
    Structured ring-buffer telemetry logger.

    Usage::

        log = TelemetryLogger(capacity=1000)
        log.info("hal", "Chip boot complete")
        log.error("ecc", "Double-bit error", bank=2, address=0x1000)
        entries = log.snapshot()
    """

    def __init__(
        self,
        capacity: int = 4096,
        min_level: LogLevel = LogLevel.DEBUG,
    ) -> None:
        if capacity <= 0:
            raise LoggerError(f"Capacity must be > 0, got {capacity}")
        self._capacity = capacity
        self._min_level = min_level
        self._buffer: deque[LogEntry] = deque(maxlen=capacity)
        self._sequence = 0
        self._counters: dict[LogLevel, int] = {lvl: 0 for lvl in LogLevel}
        self._dropped: int = 0

    # ------------------------------------------------------------------
    # Core write
    # ------------------------------------------------------------------

    def log(
        self,
        level: LogLevel,
        source: str,
        message: str,
        **metadata: Any,
    ) -> Optional[LogEntry]:
        """
        Write a log entry.

        Returns the LogEntry, or None if filtered by min_level.
        """
        if level < self._min_level:
            return None

        # Track drops when buffer is full (deque auto-evicts, we just count)
        if len(self._buffer) == self._capacity:
            self._dropped += 1

        self._sequence += 1
        entry = LogEntry(
            level=level,
            source=source,
            message=message,
            timestamp_us=time.monotonic() * 1e6,
            sequence=self._sequence,
            metadata=metadata,
        )
        self._buffer.append(entry)
        self._counters[level] += 1
        return entry

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def debug(self, source: str, message: str, **kw: Any) -> Optional[LogEntry]:
        return self.log(LogLevel.DEBUG, source, message, **kw)

    def info(self, source: str, message: str, **kw: Any) -> Optional[LogEntry]:
        return self.log(LogLevel.INFO, source, message, **kw)

    def warning(self, source: str, message: str, **kw: Any) -> Optional[LogEntry]:
        return self.log(LogLevel.WARNING, source, message, **kw)

    def error(self, source: str, message: str, **kw: Any) -> Optional[LogEntry]:
        return self.log(LogLevel.ERROR, source, message, **kw)

    def critical(self, source: str, message: str, **kw: Any) -> Optional[LogEntry]:
        return self.log(LogLevel.CRITICAL, source, message, **kw)

    # ------------------------------------------------------------------
    # Query & export
    # ------------------------------------------------------------------

    def snapshot(self) -> list[LogEntry]:
        """Return a copy of all entries currently in the buffer."""
        return list(self._buffer)

    def filter_by_level(self, min_level: LogLevel) -> list[LogEntry]:
        """Return entries at or above the given level."""
        return [e for e in self._buffer if e.level >= min_level]

    def filter_by_source(self, source: str) -> list[LogEntry]:
        """Return entries from a specific source."""
        return [e for e in self._buffer if e.source == source]

    def latest(self, n: int) -> list[LogEntry]:
        """Return the n most recent entries."""
        entries = list(self._buffer)
        return entries[-n:] if n <= len(entries) else entries

    def clear(self) -> None:
        """Clear the log buffer (counters preserved)."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Counters & status
    # ------------------------------------------------------------------

    def count(self, level: LogLevel) -> int:
        return self._counters[level]

    def total_logged(self) -> int:
        return self._sequence

    def current_size(self) -> int:
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def min_level(self) -> LogLevel:
        return self._min_level

    @min_level.setter
    def min_level(self, level: LogLevel) -> None:
        self._min_level = level

    def has_errors(self) -> bool:
        return self._counters[LogLevel.ERROR] + self._counters[LogLevel.CRITICAL] > 0

    def __repr__(self) -> str:
        return (
            f"TelemetryLogger(capacity={self._capacity}, "
            f"size={self.current_size()}, "
            f"errors={self.count(LogLevel.ERROR)}, "
            f"critical={self.count(LogLevel.CRITICAL)})"
        )
