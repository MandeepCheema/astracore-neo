"""
AstraCore Neo HAL — Hardware Abstraction Layer.

Public API::

    from hal import AstraCoreDevice, PowerState
    from hal import RegisterFile, InterruptController
    from hal import (
        HalError, DeviceError, RegisterError, InterruptError, ClockError
    )
    from hal.interrupts import IRQ_MAC_DONE, IRQ_DMA_DONE, ...
"""

from .device import AstraCoreDevice, PowerState
from .registers import RegisterFile, REGISTER_MAP
from .interrupts import InterruptController, IRQ_NAMES
from .exceptions import HalError, DeviceError, RegisterError, InterruptError, ClockError

__all__ = [
    "AstraCoreDevice",
    "PowerState",
    "RegisterFile",
    "REGISTER_MAP",
    "InterruptController",
    "IRQ_NAMES",
    "HalError",
    "DeviceError",
    "RegisterError",
    "InterruptError",
    "ClockError",
]
