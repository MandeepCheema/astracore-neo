"""
AstraCore Neo Telemetry Subsystem.

Public API::

    from telemetry import TelemetryLogger, LogLevel, LogEntry
    from telemetry import ThermalMonitor, ThermalZone, ThermalZoneConfig, ThermalState
    from telemetry import FaultPredictor, MetricTracker, MetricConfig, FaultRisk, FaultPrediction
    from telemetry import TelemetryBaseError, LoggerError, ThermalError, ThermalShutdownError
"""

from .logger import (
    TelemetryLogger, LogLevel, LogEntry,
)
from .thermal import (
    ThermalMonitor, ThermalZone, ThermalZoneConfig, ThermalState, ThermalReading,
)
from .fault_predictor import (
    FaultPredictor, MetricTracker, MetricConfig, FaultRisk, FaultPrediction,
)
from .exceptions import (
    TelemetryBaseError, LoggerError, ThermalError, ThermalShutdownError, FaultPredictorError,
)

__all__ = [
    # Logger
    "TelemetryLogger", "LogLevel", "LogEntry",
    # Thermal
    "ThermalMonitor", "ThermalZone", "ThermalZoneConfig", "ThermalState", "ThermalReading",
    # Fault predictor
    "FaultPredictor", "MetricTracker", "MetricConfig", "FaultRisk", "FaultPrediction",
    # Exceptions
    "TelemetryBaseError", "LoggerError", "ThermalError", "ThermalShutdownError",
    "FaultPredictorError",
]
