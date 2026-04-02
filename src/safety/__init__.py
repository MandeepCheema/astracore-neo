"""
AstraCore Neo Safety Subsystem.

Public API::

    from safety import ECCEngine, ECCConfig, ECCError, BitFlipType
    from safety import TMRVoter, TMRResult
    from safety import WatchdogTimer, WatchdogError
    from safety import ClockMonitor, ClockMonitorConfig, ClockFaultType
    from safety import SafetyManager, SafetyEvent, SafetySeverity
    from safety import SafetyError
"""

from .ecc import (
    ECCEngine, ECCConfig, ECCError, BitFlipType, CorrectionResult,
)
from .tmr import (
    TMRVoter, TMRResult, TMRError,
)
from .watchdog import (
    WatchdogTimer, WatchdogConfig, WatchdogError,
)
from .clock_monitor import (
    ClockMonitor, ClockMonitorConfig, ClockFaultType, ClockFault,
)
from .safety_manager import (
    SafetyManager, SafetyEvent, SafetySeverity, SafetyError,
)
from .exceptions import SafetyBaseError

__all__ = [
    # ECC
    "ECCEngine", "ECCConfig", "ECCError", "BitFlipType", "CorrectionResult",
    # TMR
    "TMRVoter", "TMRResult", "TMRError",
    # Watchdog
    "WatchdogTimer", "WatchdogConfig", "WatchdogError",
    # Clock monitor
    "ClockMonitor", "ClockMonitorConfig", "ClockFaultType", "ClockFault",
    # Safety manager
    "SafetyManager", "SafetyEvent", "SafetySeverity", "SafetyError",
    # Base
    "SafetyBaseError",
]
