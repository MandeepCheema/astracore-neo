"""
AstraCore Neo — Clock Monitor simulation.

Models the clock monitoring unit that detects clock faults:
  - Frequency drift detection (above/below tolerance band)
  - Clock loss detection (no edges within timeout)
  - Glitch detection (pulse width too narrow)
  - Multi-clock domain management
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .exceptions import SafetyBaseError


class ClockFaultType(Enum):
    NONE          = auto()   # no fault
    FREQ_HIGH     = auto()   # frequency above upper tolerance
    FREQ_LOW      = auto()   # frequency below lower tolerance
    CLOCK_LOSS    = auto()   # no clock edges detected
    GLITCH        = auto()   # pulse width violation
    JITTER_EXCESS = auto()   # jitter exceeds limit


@dataclass
class ClockFault:
    """A detected clock fault."""
    fault_type: ClockFaultType
    domain: str
    measured_mhz: float
    expected_mhz: float
    deviation_pct: float


@dataclass
class ClockMonitorConfig:
    """Clock monitor configuration per domain."""
    expected_mhz: float
    tolerance_pct: float = 5.0          # ±5% frequency tolerance
    loss_timeout_us: float = 10.0       # declare loss after 10µs of no edges
    min_pulse_width_ns: float = 1.0     # glitch filter
    max_jitter_ps: float = 500.0        # max allowed jitter


class ClockMonitor:
    """
    Simulated clock monitoring unit.

    Monitors multiple clock domains for frequency, loss, and glitch faults.

    Usage::

        monitor = ClockMonitor()
        monitor.add_domain("core", ClockMonitorConfig(expected_mhz=1000.0))
        monitor.add_domain("pll", ClockMonitorConfig(expected_mhz=400.0))
        fault = monitor.check_frequency("core", measured_mhz=950.0)
        assert fault.fault_type == ClockFaultType.FREQ_LOW
    """

    def __init__(self) -> None:
        self._domains: dict[str, ClockMonitorConfig] = {}
        self._fault_log: list[ClockFault] = []
        self._domain_status: dict[str, ClockFaultType] = {}

    # ------------------------------------------------------------------
    # Domain management
    # ------------------------------------------------------------------

    def add_domain(self, name: str, config: ClockMonitorConfig) -> None:
        """Register a clock domain for monitoring."""
        self._domains[name] = config
        self._domain_status[name] = ClockFaultType.NONE

    def remove_domain(self, name: str) -> None:
        if name not in self._domains:
            raise KeyError(f"Clock domain '{name}' not registered")
        del self._domains[name]
        del self._domain_status[name]

    def domain_names(self) -> list[str]:
        return list(self._domains.keys())

    # ------------------------------------------------------------------
    # Frequency check
    # ------------------------------------------------------------------

    def check_frequency(self, domain: str, measured_mhz: float) -> ClockFault:
        """
        Check if measured frequency is within tolerance.

        Returns a ClockFault (fault_type=NONE if in bounds).
        """
        if domain not in self._domains:
            raise KeyError(f"Clock domain '{domain}' not registered")

        cfg = self._domains[domain]
        expected = cfg.expected_mhz
        deviation_pct = (measured_mhz - expected) / expected * 100.0
        upper = expected * (1.0 + cfg.tolerance_pct / 100.0)
        lower = expected * (1.0 - cfg.tolerance_pct / 100.0)

        if measured_mhz > upper:
            fault_type = ClockFaultType.FREQ_HIGH
        elif measured_mhz < lower:
            fault_type = ClockFaultType.FREQ_LOW
        else:
            fault_type = ClockFaultType.NONE

        fault = ClockFault(
            fault_type=fault_type,
            domain=domain,
            measured_mhz=measured_mhz,
            expected_mhz=expected,
            deviation_pct=deviation_pct,
        )

        self._domain_status[domain] = fault_type
        if fault_type != ClockFaultType.NONE:
            self._fault_log.append(fault)

        return fault

    # ------------------------------------------------------------------
    # Clock loss detection
    # ------------------------------------------------------------------

    def check_clock_loss(self, domain: str, time_since_last_edge_us: float) -> ClockFault:
        """
        Check if clock has been absent for too long.

        Args:
            time_since_last_edge_us: microseconds since last detected clock edge
        """
        if domain not in self._domains:
            raise KeyError(f"Clock domain '{domain}' not registered")

        cfg = self._domains[domain]
        if time_since_last_edge_us > cfg.loss_timeout_us:
            fault = ClockFault(
                fault_type=ClockFaultType.CLOCK_LOSS,
                domain=domain,
                measured_mhz=0.0,
                expected_mhz=cfg.expected_mhz,
                deviation_pct=-100.0,
            )
            self._domain_status[domain] = ClockFaultType.CLOCK_LOSS
            self._fault_log.append(fault)
            return fault

        return ClockFault(
            fault_type=ClockFaultType.NONE,
            domain=domain,
            measured_mhz=cfg.expected_mhz,
            expected_mhz=cfg.expected_mhz,
            deviation_pct=0.0,
        )

    # ------------------------------------------------------------------
    # Glitch detection
    # ------------------------------------------------------------------

    def check_glitch(self, domain: str, pulse_width_ns: float) -> ClockFault:
        """
        Check if a pulse width is too narrow (glitch).

        Args:
            pulse_width_ns: measured pulse width in nanoseconds
        """
        if domain not in self._domains:
            raise KeyError(f"Clock domain '{domain}' not registered")

        cfg = self._domains[domain]
        if pulse_width_ns < cfg.min_pulse_width_ns:
            fault = ClockFault(
                fault_type=ClockFaultType.GLITCH,
                domain=domain,
                measured_mhz=1e3 / max(pulse_width_ns, 1e-9),
                expected_mhz=cfg.expected_mhz,
                deviation_pct=0.0,
            )
            self._domain_status[domain] = ClockFaultType.GLITCH
            self._fault_log.append(fault)
            return fault

        return ClockFault(
            fault_type=ClockFaultType.NONE,
            domain=domain,
            measured_mhz=cfg.expected_mhz,
            expected_mhz=cfg.expected_mhz,
            deviation_pct=0.0,
        )

    # ------------------------------------------------------------------
    # Status & diagnostics
    # ------------------------------------------------------------------

    def domain_status(self, domain: str) -> ClockFaultType:
        if domain not in self._domain_status:
            raise KeyError(f"Clock domain '{domain}' not registered")
        return self._domain_status[domain]

    def any_fault(self) -> bool:
        """Return True if any domain has an active fault."""
        return any(v != ClockFaultType.NONE for v in self._domain_status.values())

    def fault_log(self) -> list[ClockFault]:
        return list(self._fault_log)

    def clear_fault_log(self) -> None:
        self._fault_log.clear()
        for d in self._domain_status:
            self._domain_status[d] = ClockFaultType.NONE

    def __repr__(self) -> str:
        faults = sum(1 for v in self._domain_status.values() if v != ClockFaultType.NONE)
        return f"ClockMonitor(domains={len(self._domains)}, active_faults={faults})"
