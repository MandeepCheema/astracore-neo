"""
AstraCore Neo — Thermal Monitor simulation.

Models the on-chip thermal management unit:
  - Multiple thermal zones (CPU core, NPU, DRAM PHY, I/O ring)
  - Per-zone thresholds: nominal → warning → throttle → critical → shutdown
  - Throttling response: reduce clock frequency when temp exceeds throttle threshold
  - Thermal shutdown: raise ThermalShutdownError when critical threshold exceeded
  - Temperature trend tracking (rolling window for slope estimation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
from typing import Optional

from .exceptions import ThermalError, ThermalShutdownError


class ThermalState(Enum):
    NOMINAL   = auto()   # below warning threshold
    WARNING   = auto()   # above warning, below throttle
    THROTTLED = auto()   # above throttle, clock reduced
    CRITICAL  = auto()   # above critical, prepare for shutdown
    SHUTDOWN  = auto()   # above shutdown threshold


@dataclass
class ThermalZoneConfig:
    """Per-zone thermal thresholds (all in °C)."""
    warning_c: float   = 75.0
    throttle_c: float  = 85.0
    critical_c: float  = 95.0
    shutdown_c: float  = 105.0
    throttle_pct: float = 50.0    # clock reduction % when throttled


@dataclass
class ThermalReading:
    """A single temperature sample for a zone."""
    zone: str
    temp_c: float
    state: ThermalState
    throttle_active: bool
    timestamp_us: float


class ThermalZone:
    """
    Monitors one thermal zone.

    Usage::

        zone = ThermalZone("npu", ThermalZoneConfig(shutdown_c=110.0))
        reading = zone.update(92.0)
        assert reading.state == ThermalState.CRITICAL
    """

    def __init__(self, name: str, config: Optional[ThermalZoneConfig] = None) -> None:
        self._name = name
        self._cfg = config or ThermalZoneConfig()
        self._state = ThermalState.NOMINAL
        self._current_temp: float = 25.0
        self._history: deque[float] = deque(maxlen=16)
        self._peak_temp: float = 25.0
        self._sample_count: int = 0
        self._throttle_count: int = 0

    def update(self, temp_c: float) -> ThermalReading:
        """
        Submit a new temperature reading.

        Updates zone state and raises ThermalShutdownError if shutdown threshold exceeded.
        """
        import time
        self._current_temp = temp_c
        self._history.append(temp_c)
        self._sample_count += 1
        if temp_c > self._peak_temp:
            self._peak_temp = temp_c

        cfg = self._cfg
        if temp_c >= cfg.shutdown_c:
            self._state = ThermalState.SHUTDOWN
            raise ThermalShutdownError(
                f"Zone '{self._name}' shutdown threshold exceeded: "
                f"{temp_c:.1f}°C >= {cfg.shutdown_c:.1f}°C"
            )
        elif temp_c >= cfg.critical_c:
            self._state = ThermalState.CRITICAL
        elif temp_c >= cfg.throttle_c:
            self._state = ThermalState.THROTTLED
            self._throttle_count += 1
        elif temp_c >= cfg.warning_c:
            self._state = ThermalState.WARNING
        else:
            self._state = ThermalState.NOMINAL

        return ThermalReading(
            zone=self._name,
            temp_c=temp_c,
            state=self._state,
            throttle_active=self._state == ThermalState.THROTTLED,
            timestamp_us=time.monotonic() * 1e6,
        )

    def effective_clock_pct(self) -> float:
        """Return effective clock percentage (100% nominal, reduced when throttled)."""
        if self._state == ThermalState.THROTTLED:
            return 100.0 - self._cfg.throttle_pct
        elif self._state in (ThermalState.CRITICAL, ThermalState.SHUTDOWN):
            return 0.0
        return 100.0

    def temperature_slope(self) -> float:
        """
        Estimate temperature rise rate (°C per sample) over the history window.
        Returns 0.0 if fewer than 2 samples.
        """
        h = list(self._history)
        if len(h) < 2:
            return 0.0
        # Simple linear regression slope
        n = len(h)
        xs = list(range(n))
        x_mean = (n - 1) / 2.0
        y_mean = sum(h) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, h))
        den = sum((x - x_mean) ** 2 for x in xs)
        return num / den if den != 0 else 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> ThermalState:
        return self._state

    @property
    def current_temp(self) -> float:
        return self._current_temp

    @property
    def peak_temp(self) -> float:
        return self._peak_temp

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def throttle_count(self) -> int:
        return self._throttle_count

    @property
    def config(self) -> ThermalZoneConfig:
        return self._cfg

    def __repr__(self) -> str:
        return (
            f"ThermalZone('{self._name}', "
            f"{self._current_temp:.1f}°C, "
            f"state={self._state.name})"
        )


class ThermalMonitor:
    """
    Multi-zone thermal monitor.

    Manages a set of ThermalZone instances and provides an aggregated view.

    Usage::

        monitor = ThermalMonitor()
        monitor.add_zone("cpu", ThermalZoneConfig(shutdown_c=110.0))
        monitor.add_zone("npu", ThermalZoneConfig(shutdown_c=105.0))
        monitor.update_zone("cpu", 80.0)
        assert monitor.any_throttled()
    """

    def __init__(self) -> None:
        self._zones: dict[str, ThermalZone] = {}

    def add_zone(self, name: str, config: Optional[ThermalZoneConfig] = None) -> ThermalZone:
        """Register a new thermal zone."""
        if name in self._zones:
            raise ThermalError(f"Zone '{name}' already registered")
        zone = ThermalZone(name, config)
        self._zones[name] = zone
        return zone

    def remove_zone(self, name: str) -> None:
        if name not in self._zones:
            raise ThermalError(f"Zone '{name}' not found")
        del self._zones[name]

    def get_zone(self, name: str) -> ThermalZone:
        if name not in self._zones:
            raise ThermalError(f"Zone '{name}' not found")
        return self._zones[name]

    def update_zone(self, name: str, temp_c: float) -> ThermalReading:
        """Update temperature for a named zone. Propagates ThermalShutdownError."""
        return self.get_zone(name).update(temp_c)

    def zone_names(self) -> list[str]:
        return list(self._zones.keys())

    def any_throttled(self) -> bool:
        return any(z.state == ThermalState.THROTTLED for z in self._zones.values())

    def any_critical(self) -> bool:
        return any(z.state in (ThermalState.CRITICAL, ThermalState.SHUTDOWN)
                   for z in self._zones.values())

    def hottest_zone(self) -> Optional[ThermalZone]:
        """Return the zone with the highest current temperature."""
        if not self._zones:
            return None
        return max(self._zones.values(), key=lambda z: z.current_temp)

    def summary(self) -> dict[str, dict]:
        """Return a dict of zone_name → {temp, state, clock_pct}."""
        return {
            name: {
                "temp_c": z.current_temp,
                "state": z.state.name,
                "clock_pct": z.effective_clock_pct(),
            }
            for name, z in self._zones.items()
        }

    def __repr__(self) -> str:
        return f"ThermalMonitor(zones={len(self._zones)}, throttled={self.any_throttled()})"
