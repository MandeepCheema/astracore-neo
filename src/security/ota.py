"""
AstraCore Neo — OTA (Over-the-Air) Update Manager simulation.

Models secure firmware update pipeline:
  - Update package validation (hash + signature)
  - Anti-rollback version enforcement
  - A/B slot (redundant) update scheme
  - Update state machine: IDLE → DOWNLOADING → VALIDATING → APPLYING → COMPLETE
  - Rollback on failed update
"""

from __future__ import annotations

import hashlib
import hmac
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .exceptions import OTAError, RollbackError, SignatureError


class UpdateState(Enum):
    IDLE        = auto()
    DOWNLOADING = auto()
    VALIDATING  = auto()
    APPLYING    = auto()
    COMPLETE    = auto()
    FAILED      = auto()
    ROLLED_BACK = auto()


class SlotID(Enum):
    SLOT_A = "A"
    SLOT_B = "B"


@dataclass
class FirmwareSlot:
    """An A/B firmware slot."""
    slot_id: SlotID
    version: int = 0
    payload_hash: bytes = field(default_factory=lambda: bytes(32))
    valid: bool = False
    active: bool = False


@dataclass
class UpdatePackage:
    """An OTA update package."""
    version: int
    payload: bytes
    signature: bytes          # HMAC-SHA256 of (hash + version)
    description: str = ""
    _payload_hash: Optional[bytes] = field(default=None)

    def __post_init__(self) -> None:
        if self._payload_hash is None:
            self._payload_hash = hashlib.sha256(self.payload).digest()

    @property
    def payload_hash(self) -> bytes:
        return self._payload_hash  # type: ignore[return-value]


class OTAManager:
    """
    Simulates the OTA firmware update manager.

    Uses an A/B slot scheme: the inactive slot receives the update,
    is validated, then activated.  On failure, the active slot is unchanged.

    Usage::

        ota = OTAManager(signing_key=b"secret")
        pkg = ota.create_package(version=2, payload=b"new_firmware")
        ota.begin_update(pkg)
        ota.apply()
        assert ota.active_slot().version == 2
    """

    def __init__(
        self,
        signing_key: bytes = b"astracore-neo-ota-key",
        min_version: int = 0,
    ) -> None:
        self._signing_key = signing_key
        self._min_version = min_version
        self._state = UpdateState.IDLE
        self._slot_a = FirmwareSlot(SlotID.SLOT_A, version=1, valid=True, active=True)
        self._slot_b = FirmwareSlot(SlotID.SLOT_B, version=0, valid=False, active=False)
        self._pending_package: Optional[UpdatePackage] = None
        self._pending_slot: Optional[FirmwareSlot] = None
        self._update_count: int = 0
        self._rollback_count: int = 0

    # ------------------------------------------------------------------
    # Package creation (signing)
    # ------------------------------------------------------------------

    def create_package(
        self,
        version: int,
        payload: bytes,
        description: str = "",
    ) -> UpdatePackage:
        """Create and sign an OTA update package."""
        h = hashlib.sha256(payload).digest()
        sig_msg = h + struct.pack(">I", version)
        signature = hmac.new(self._signing_key, sig_msg, hashlib.sha256).digest()
        return UpdatePackage(
            version=version,
            payload=payload,
            signature=signature,
            description=description,
            _payload_hash=h,
        )

    # ------------------------------------------------------------------
    # Update pipeline
    # ------------------------------------------------------------------

    def begin_update(self, package: UpdatePackage) -> None:
        """
        Start an update: download and stage the package.
        """
        if self._state not in (UpdateState.IDLE, UpdateState.FAILED, UpdateState.ROLLED_BACK, UpdateState.COMPLETE):
            raise OTAError(f"Cannot begin update in state {self._state.name}")
        self._state = UpdateState.DOWNLOADING
        self._pending_package = package
        self._state = UpdateState.VALIDATING

    def validate(self) -> None:
        """
        Validate the staged package: hash, signature, version.
        Raises OTAError, SignatureError, or RollbackError on failure.
        """
        if self._state != UpdateState.VALIDATING:
            raise OTAError(f"Not in VALIDATING state (current: {self._state.name})")
        pkg = self._pending_package
        if pkg is None:
            raise OTAError("No pending package")

        # Hash check
        computed_hash = hashlib.sha256(pkg.payload).digest()
        if not hmac.compare_digest(computed_hash, pkg.payload_hash):
            self._state = UpdateState.FAILED
            raise OTAError("Package hash mismatch")

        # Signature check
        sig_msg = pkg.payload_hash + struct.pack(">I", pkg.version)
        expected_sig = hmac.new(self._signing_key, sig_msg, hashlib.sha256).digest()
        if not hmac.compare_digest(pkg.signature, expected_sig):
            self._state = UpdateState.FAILED
            raise SignatureError("OTA package signature verification failed")

        # Anti-rollback
        if pkg.version < self._min_version:
            self._state = UpdateState.FAILED
            raise RollbackError(
                f"Package version {pkg.version} < min_version {self._min_version}"
            )

        active = self.active_slot()
        if pkg.version <= active.version:
            self._state = UpdateState.FAILED
            raise RollbackError(
                f"Package version {pkg.version} is not newer than "
                f"active slot version {active.version}"
            )

        # Stage to inactive slot
        inactive = self._inactive_slot()
        inactive.version = pkg.version
        inactive.payload_hash = pkg.payload_hash
        inactive.valid = True
        self._pending_slot = inactive
        self._state = UpdateState.APPLYING

    def apply(self) -> None:
        """
        Activate the validated update (swap slots).
        """
        if self._state != UpdateState.APPLYING:
            raise OTAError(f"Not in APPLYING state (current: {self._state.name})")
        if self._pending_slot is None or not self._pending_slot.valid:
            self._state = UpdateState.FAILED
            raise OTAError("Pending slot is not valid")

        # Deactivate current active, activate pending
        for slot in (self._slot_a, self._slot_b):
            slot.active = False
        self._pending_slot.active = True

        self._update_count += 1
        self._state = UpdateState.COMPLETE
        self._pending_package = None
        self._pending_slot = None

    def rollback(self) -> None:
        """
        Rollback: mark pending slot invalid, keep current active slot.
        """
        if self._pending_slot is not None:
            self._pending_slot.valid = False
            self._pending_slot.version = 0
        self._pending_package = None
        self._pending_slot = None
        self._rollback_count += 1
        self._state = UpdateState.ROLLED_BACK

    def set_min_version(self, version: int) -> None:
        """Advance anti-rollback minimum version (monotonic)."""
        if version < self._min_version:
            raise RollbackError(
                f"Cannot lower min_version from {self._min_version} to {version}"
            )
        self._min_version = version

    # ------------------------------------------------------------------
    # Slot management
    # ------------------------------------------------------------------

    def active_slot(self) -> FirmwareSlot:
        for slot in (self._slot_a, self._slot_b):
            if slot.active:
                return slot
        raise OTAError("No active slot found")

    def _inactive_slot(self) -> FirmwareSlot:
        for slot in (self._slot_a, self._slot_b):
            if not slot.active:
                return slot
        raise OTAError("No inactive slot found")

    def get_slot(self, slot_id: SlotID) -> FirmwareSlot:
        return self._slot_a if slot_id == SlotID.SLOT_A else self._slot_b

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def state(self) -> UpdateState:
        return self._state

    @property
    def min_version(self) -> int:
        return self._min_version

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def rollback_count(self) -> int:
        return self._rollback_count

    def __repr__(self) -> str:
        active = self.active_slot()
        return (
            f"OTAManager(state={self._state.name}, "
            f"active_slot={active.slot_id.value} v{active.version}, "
            f"updates={self._update_count})"
        )
