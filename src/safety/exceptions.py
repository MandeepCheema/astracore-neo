"""Safety module base exceptions."""


class SafetyBaseError(Exception):
    """Base exception for all safety subsystem errors."""


class ECCError(SafetyBaseError):
    """ECC uncorrectable error (double-bit or higher)."""


class TMRError(SafetyBaseError):
    """Triple Modular Redundancy voting failure (all three lanes disagree)."""


class WatchdogError(SafetyBaseError):
    """Watchdog timer expired without being serviced."""


class SafetyError(SafetyBaseError):
    """General safety manager error."""
