"""
AstraCore Neo HAL — Exception hierarchy.
"""


class HalError(Exception):
    """Base exception for all HAL errors."""


class DeviceError(HalError):
    """Raised on invalid device state transitions."""


class RegisterError(HalError):
    """Raised on invalid register access."""


class InterruptError(HalError):
    """Raised on invalid interrupt operations."""


class ClockError(HalError):
    """Raised when requested clock frequency is out of range."""
