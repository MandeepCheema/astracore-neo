"""
AstraCore Neo — Trusted Execution Environment (TEE) simulation.

Models ARM TrustZone-style secure/non-secure world separation:
  - Secure world: trusted apps, key storage, crypto operations
  - Normal world: OS, applications, untrusted code
  - Secure memory partitioning (regions locked to secure world)
  - Secure key vault with access control
  - SMC (Secure Monitor Call) interface between worlds
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .exceptions import TEEError


class WorldState(Enum):
    NORMAL  = auto()   # non-secure world (OS/apps)
    SECURE  = auto()   # secure world (TEE)


class KeyUsage(Enum):
    ENCRYPT   = auto()
    DECRYPT   = auto()
    SIGN      = auto()
    VERIFY    = auto()
    DERIVE    = auto()


@dataclass
class SecureKey:
    """A key stored in the secure key vault."""
    key_id: str
    key_material: bytes        # simulated key bytes
    usage: list[KeyUsage]
    exportable: bool = False   # if False, key never leaves TEE


@dataclass
class SecureMemoryRegion:
    """A locked memory region accessible only from secure world."""
    base_addr: int
    size_bytes: int
    label: str

    def contains(self, addr: int) -> bool:
        return self.base_addr <= addr < self.base_addr + self.size_bytes


class TEE:
    """
    Simulated Trusted Execution Environment.

    Provides secure/non-secure world separation, key vault,
    and a Secure Monitor Call interface.

    Usage::

        tee = TEE()
        tee.initialize()
        tee.switch_to_secure()
        key = tee.generate_key("device_key", [KeyUsage.SIGN])
        signature = tee.sign(key_id="device_key", data=b"hello")
        tee.switch_to_normal()
    """

    def __init__(self) -> None:
        self._initialized = False
        self._world = WorldState.NORMAL
        self._key_vault: dict[str, SecureKey] = {}
        self._secure_regions: list[SecureMemoryRegion] = []
        self._smc_call_count: int = 0
        self._access_violations: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize TEE (must be called from secure world at boot)."""
        self._initialized = True
        self._world = WorldState.NORMAL

    def is_initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------
    # World switching (via SMC)
    # ------------------------------------------------------------------

    def switch_to_secure(self) -> None:
        """Enter secure world via Secure Monitor Call."""
        if not self._initialized:
            raise TEEError("TEE not initialized")
        self._smc_call_count += 1
        self._world = WorldState.SECURE

    def switch_to_normal(self) -> None:
        """Return to normal world via Secure Monitor Call."""
        if not self._initialized:
            raise TEEError("TEE not initialized")
        self._smc_call_count += 1
        self._world = WorldState.NORMAL

    @property
    def current_world(self) -> WorldState:
        return self._world

    def _require_secure(self, operation: str) -> None:
        """Raise TEEError if not in secure world."""
        if self._world != WorldState.SECURE:
            self._access_violations += 1
            raise TEEError(
                f"Operation '{operation}' requires secure world, "
                f"currently in {self._world.name}"
            )

    # ------------------------------------------------------------------
    # Secure memory regions
    # ------------------------------------------------------------------

    def add_secure_region(self, base_addr: int, size_bytes: int, label: str) -> None:
        """Register a memory region as secure-world-only."""
        self._require_secure("add_secure_region")
        self._secure_regions.append(
            SecureMemoryRegion(base_addr=base_addr, size_bytes=size_bytes, label=label)
        )

    def check_address(self, addr: int) -> bool:
        """
        Check if an address is in a secure region.
        Returns True if secure, False if normal.
        """
        return any(r.contains(addr) for r in self._secure_regions)

    def access_secure_memory(self, addr: int) -> bytes:
        """
        Simulate a read from a secure memory region.
        Raises TEEError if accessed from normal world.
        """
        if self.check_address(addr) and self._world != WorldState.SECURE:
            self._access_violations += 1
            raise TEEError(
                f"Secure memory access violation at 0x{addr:08X} from NORMAL world"
            )
        return bytes(4)   # simulated 4-byte read

    # ------------------------------------------------------------------
    # Key vault
    # ------------------------------------------------------------------

    def generate_key(
        self,
        key_id: str,
        usage: list[KeyUsage],
        key_size_bytes: int = 32,
        exportable: bool = False,
    ) -> SecureKey:
        """
        Generate and store a key in the secure vault.
        Must be called from secure world.
        """
        self._require_secure("generate_key")
        if key_id in self._key_vault:
            raise TEEError(f"Key '{key_id}' already exists in vault")
        key_material = os.urandom(key_size_bytes)
        key = SecureKey(
            key_id=key_id,
            key_material=key_material,
            usage=usage,
            exportable=exportable,
        )
        self._key_vault[key_id] = key
        return key

    def delete_key(self, key_id: str) -> None:
        """Delete a key from the vault (secure world only)."""
        self._require_secure("delete_key")
        if key_id not in self._key_vault:
            raise TEEError(f"Key '{key_id}' not found in vault")
        del self._key_vault[key_id]

    def key_exists(self, key_id: str) -> bool:
        return key_id in self._key_vault

    def export_key(self, key_id: str) -> bytes:
        """
        Export key material (secure world only, key must be exportable).
        """
        self._require_secure("export_key")
        if key_id not in self._key_vault:
            raise TEEError(f"Key '{key_id}' not found")
        key = self._key_vault[key_id]
        if not key.exportable:
            raise TEEError(f"Key '{key_id}' is non-exportable")
        return key.key_material

    # ------------------------------------------------------------------
    # Crypto operations
    # ------------------------------------------------------------------

    def sign(self, key_id: str, data: bytes) -> bytes:
        """
        Sign data using a key in the vault (secure world only).
        Returns simulated HMAC-SHA256 signature.
        """
        self._require_secure("sign")
        if key_id not in self._key_vault:
            raise TEEError(f"Key '{key_id}' not found")
        key = self._key_vault[key_id]
        if KeyUsage.SIGN not in key.usage:
            raise TEEError(f"Key '{key_id}' not permitted for SIGN")
        import hmac as hmac_mod
        return hmac_mod.new(key.key_material, data, hashlib.sha256).digest()

    def verify(self, key_id: str, data: bytes, signature: bytes) -> bool:
        """
        Verify a signature (secure world only).
        """
        self._require_secure("verify")
        if key_id not in self._key_vault:
            raise TEEError(f"Key '{key_id}' not found")
        key = self._key_vault[key_id]
        if KeyUsage.VERIFY not in key.usage and KeyUsage.SIGN not in key.usage:
            raise TEEError(f"Key '{key_id}' not permitted for VERIFY")
        import hmac as hmac_mod
        expected = hmac_mod.new(key.key_material, data, hashlib.sha256).digest()
        return hmac_mod.compare_digest(signature, expected)

    def derive_key(self, parent_key_id: str, context: bytes, new_key_id: str) -> SecureKey:
        """
        Derive a child key from a parent key using HKDF-like derivation.
        """
        self._require_secure("derive_key")
        if parent_key_id not in self._key_vault:
            raise TEEError(f"Parent key '{parent_key_id}' not found")
        parent = self._key_vault[parent_key_id]
        if KeyUsage.DERIVE not in parent.usage:
            raise TEEError(f"Key '{parent_key_id}' not permitted for DERIVE")
        # Simulated HKDF extract+expand
        derived = hashlib.sha256(parent.key_material + context).digest()
        child = SecureKey(
            key_id=new_key_id,
            key_material=derived,
            usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
            exportable=False,
        )
        self._key_vault[new_key_id] = child
        return child

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def smc_call_count(self) -> int:
        return self._smc_call_count

    @property
    def access_violations(self) -> int:
        return self._access_violations

    def key_count(self) -> int:
        return len(self._key_vault)

    def secure_region_count(self) -> int:
        return len(self._secure_regions)

    def __repr__(self) -> str:
        return (
            f"TEE(world={self._world.name}, "
            f"keys={self.key_count()}, "
            f"smc_calls={self._smc_call_count})"
        )
