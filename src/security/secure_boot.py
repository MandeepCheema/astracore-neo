"""
AstraCore Neo — Secure Boot simulation.

Models the hardware root-of-trust boot chain:
  - ROM → BL1 → BL2 → OS image, each stage verifies the next
  - SHA-256 hash chain (simulated)
  - RSA-2048 signature verification (simulated with HMAC proxy)
  - Anti-rollback version counter
  - Boot lock fuses (one-way, cannot be cleared)
"""

from __future__ import annotations

import hashlib
import hmac
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .exceptions import SecureBootError, SignatureError, RollbackError


class BootStage(Enum):
    ROM    = auto()   # immutable ROM code
    BL1    = auto()   # first bootloader (in protected SRAM)
    BL2    = auto()   # second bootloader
    OS     = auto()   # OS / firmware image
    APP    = auto()   # application


class BootState(Enum):
    UNINIT     = auto()
    ROM_OK     = auto()
    BL1_OK     = auto()
    BL2_OK     = auto()
    OS_OK      = auto()
    BOOT_COMPLETE = auto()
    FAILED     = auto()


@dataclass
class BootImage:
    """A firmware image with metadata for secure boot verification."""
    stage: BootStage
    version: int                   # monotonic version counter
    payload: bytes                 # simulated firmware content
    signature: bytes               # RSA signature (simulated as HMAC-SHA256)
    hash_value: bytes              # SHA-256 of payload


@dataclass
class FuseState:
    """One-way programmable fuse state (cannot be reset once blown)."""
    secure_boot_enabled: bool = False
    debug_disabled: bool = False
    jtag_locked: bool = False
    min_version: int = 0           # anti-rollback minimum allowed version


class SecureBootEngine:
    """
    Simulates hardware secure boot chain.

    Each stage is verified in order: ROM → BL1 → BL2 → OS.
    Verification checks:
      1. SHA-256 hash of payload matches header
      2. Signature (HMAC-SHA256 proxy for RSA) valid against root key
      3. Version >= anti-rollback minimum

    Usage::

        engine = SecureBootEngine(root_key=b"secret")
        engine.blow_fuse_secure_boot()
        image = engine.create_image(BootStage.BL1, version=1, payload=b"bl1_code")
        engine.verify_stage(image)
        engine.advance()
    """

    def __init__(self, root_key: bytes = b"astracore-neo-root-key-v1") -> None:
        self._root_key = root_key
        self._fuses = FuseState()
        self._state = BootState.UNINIT
        self._verified_stages: list[BootStage] = []
        self._measurement_log: list[bytes] = []   # TPM-style PCR log

    # ------------------------------------------------------------------
    # Fuse management
    # ------------------------------------------------------------------

    def blow_fuse_secure_boot(self) -> None:
        """Enable secure boot (irreversible)."""
        self._fuses.secure_boot_enabled = True

    def blow_fuse_disable_debug(self) -> None:
        """Disable debug interfaces (irreversible)."""
        self._fuses.debug_disabled = True

    def blow_fuse_lock_jtag(self) -> None:
        """Lock JTAG port (irreversible)."""
        self._fuses.jtag_locked = True

    def set_min_version(self, version: int) -> None:
        """
        Advance anti-rollback minimum version (can only increase).
        """
        if version < self._fuses.min_version:
            raise RollbackError(
                f"Cannot lower min_version from {self._fuses.min_version} to {version}"
            )
        self._fuses.min_version = version

    @property
    def fuses(self) -> FuseState:
        return self._fuses

    # ------------------------------------------------------------------
    # Image creation (signing)
    # ------------------------------------------------------------------

    def create_image(
        self,
        stage: BootStage,
        version: int,
        payload: bytes,
    ) -> BootImage:
        """
        Create and sign a boot image.

        Args:
            stage: which boot stage this image belongs to
            version: monotonic version number
            payload: firmware bytes

        Returns:
            BootImage with hash and signature populated.
        """
        hash_val = hashlib.sha256(payload).digest()
        # Simulate RSA signing with HMAC-SHA256 over (hash + version)
        sig_msg = hash_val + struct.pack(">I", version)
        signature = hmac.new(self._root_key, sig_msg, hashlib.sha256).digest()
        return BootImage(
            stage=stage,
            version=version,
            payload=payload,
            signature=signature,
            hash_value=hash_val,
        )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _verify_hash(self, image: BootImage) -> None:
        computed = hashlib.sha256(image.payload).digest()
        if not hmac.compare_digest(computed, image.hash_value):
            raise SecureBootError(
                f"Hash mismatch for stage {image.stage.name}: "
                f"expected {image.hash_value.hex()}, got {computed.hex()}"
            )

    def _verify_signature(self, image: BootImage) -> None:
        sig_msg = image.hash_value + struct.pack(">I", image.version)
        expected_sig = hmac.new(self._root_key, sig_msg, hashlib.sha256).digest()
        if not hmac.compare_digest(image.signature, expected_sig):
            raise SignatureError(
                f"Signature verification failed for stage {image.stage.name}"
            )

    def _verify_version(self, image: BootImage) -> None:
        if image.version < self._fuses.min_version:
            raise RollbackError(
                f"Anti-rollback: image version {image.version} < "
                f"min_version {self._fuses.min_version}"
            )

    def verify_stage(self, image: BootImage) -> None:
        """
        Verify a boot image.

        Checks hash integrity, signature validity, and anti-rollback.
        Records measurement in the PCR log.

        Raises SecureBootError, SignatureError, or RollbackError on failure.
        """
        if self._fuses.secure_boot_enabled:
            self._verify_hash(image)
            self._verify_signature(image)
            self._verify_version(image)
        else:
            # Secure boot not enabled — only verify hash
            self._verify_hash(image)

        self._verified_stages.append(image.stage)
        self._measurement_log.append(image.hash_value)

    def advance(self) -> BootState:
        """
        Advance boot state machine based on verified stages.
        """
        stages = set(self._verified_stages)
        if BootStage.OS in stages:
            self._state = BootState.BOOT_COMPLETE
        elif BootStage.BL2 in stages:
            self._state = BootState.BL2_OK
        elif BootStage.BL1 in stages:
            self._state = BootState.BL1_OK
        elif BootStage.ROM in stages:
            self._state = BootState.ROM_OK
        return self._state

    def mark_failed(self) -> None:
        """Mark boot as failed (e.g. after verification error)."""
        self._state = BootState.FAILED

    # ------------------------------------------------------------------
    # Status & diagnostics
    # ------------------------------------------------------------------

    @property
    def state(self) -> BootState:
        return self._state

    @property
    def verified_stages(self) -> list[BootStage]:
        return list(self._verified_stages)

    def measurement_log(self) -> list[bytes]:
        """Return PCR-style measurement log (SHA-256 of each verified image)."""
        return list(self._measurement_log)

    def combined_measurement(self) -> bytes:
        """Return SHA-256 of all stage measurements chained (like a TPM PCR extend)."""
        combined = b"".join(self._measurement_log)
        return hashlib.sha256(combined).digest()

    def __repr__(self) -> str:
        return (
            f"SecureBootEngine(state={self._state.name}, "
            f"secure_boot={self._fuses.secure_boot_enabled}, "
            f"stages_verified={len(self._verified_stages)})"
        )
