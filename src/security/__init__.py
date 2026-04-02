"""
AstraCore Neo Security Subsystem.

Public API::

    from security import SecureBootEngine, BootImage, BootStage, BootState, FuseState
    from security import TEE, WorldState, KeyUsage, SecureKey
    from security import OTAManager, UpdatePackage, UpdateState, SlotID
    from security import SecureBootError, SignatureError, TEEError, OTAError, RollbackError
"""

from .secure_boot import (
    SecureBootEngine, BootImage, BootStage, BootState, FuseState,
)
from .tee import (
    TEE, WorldState, KeyUsage, SecureKey, SecureMemoryRegion,
)
from .ota import (
    OTAManager, UpdatePackage, UpdateState, SlotID, FirmwareSlot,
)
from .exceptions import (
    SecurityBaseError, SecureBootError, SignatureError,
    TEEError, OTAError, RollbackError,
)

__all__ = [
    # Secure boot
    "SecureBootEngine", "BootImage", "BootStage", "BootState", "FuseState",
    # TEE
    "TEE", "WorldState", "KeyUsage", "SecureKey", "SecureMemoryRegion",
    # OTA
    "OTAManager", "UpdatePackage", "UpdateState", "SlotID", "FirmwareSlot",
    # Exceptions
    "SecurityBaseError", "SecureBootError", "SignatureError",
    "TEEError", "OTAError", "RollbackError",
]
