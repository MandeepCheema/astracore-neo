"""Telemetry module exceptions."""


class TelemetryBaseError(Exception):
    """Base exception for all telemetry subsystem errors."""


class LoggerError(TelemetryBaseError):
    """Logger configuration or write error."""


class ThermalError(TelemetryBaseError):
    """Thermal zone fault — chip too hot."""


class ThermalShutdownError(ThermalError):
    """Critical thermal threshold exceeded — chip must shut down."""


class FaultPredictorError(TelemetryBaseError):
    """Fault predictor error."""
