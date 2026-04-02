"""
AstraCore Neo — Module 7: Security testbench.

Coverage:
  - SecureBoot: image creation, hash verification, signature, anti-rollback, fuses, PCR log
  - TEE: world switching, secure regions, key vault, crypto ops, access violations
  - OTA: full update pipeline, signature, rollback protection, A/B slots
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from security import (
    SecureBootEngine, BootImage, BootStage, BootState, FuseState,
    TEE, WorldState, KeyUsage, SecureKey, SecureMemoryRegion,
    OTAManager, UpdatePackage, UpdateState, SlotID, FirmwareSlot,
    SecureBootError, SignatureError, TEEError, OTAError, RollbackError,
)
from security.ota import UpdateState


# ===========================================================================
# Secure Boot tests
# ===========================================================================

class TestSecureBootFuses:
    def test_blow_secure_boot_fuse(self):
        engine = SecureBootEngine()
        assert not engine.fuses.secure_boot_enabled
        engine.blow_fuse_secure_boot()
        assert engine.fuses.secure_boot_enabled

    def test_blow_debug_fuse(self):
        engine = SecureBootEngine()
        engine.blow_fuse_disable_debug()
        assert engine.fuses.debug_disabled

    def test_blow_jtag_fuse(self):
        engine = SecureBootEngine()
        engine.blow_fuse_lock_jtag()
        assert engine.fuses.jtag_locked

    def test_set_min_version_increases(self):
        engine = SecureBootEngine()
        engine.set_min_version(5)
        assert engine.fuses.min_version == 5

    def test_set_min_version_cannot_decrease(self):
        engine = SecureBootEngine()
        engine.set_min_version(5)
        with pytest.raises(RollbackError):
            engine.set_min_version(3)

    def test_set_min_version_same_is_ok(self):
        engine = SecureBootEngine()
        engine.set_min_version(5)
        engine.set_min_version(5)  # same value — no error


class TestSecureBootImageCreation:
    def setup_method(self):
        self.engine = SecureBootEngine(root_key=b"test-key")

    def test_create_image_has_hash(self):
        img = self.engine.create_image(BootStage.BL1, version=1, payload=b"bl1code")
        assert len(img.hash_value) == 32  # SHA-256

    def test_create_image_has_signature(self):
        img = self.engine.create_image(BootStage.BL1, version=1, payload=b"bl1code")
        assert len(img.signature) == 32

    def test_create_image_stage_set(self):
        img = self.engine.create_image(BootStage.OS, version=3, payload=b"os")
        assert img.stage == BootStage.OS
        assert img.version == 3


class TestSecureBootVerification:
    def setup_method(self):
        self.engine = SecureBootEngine(root_key=b"test-key")

    def test_verify_valid_image(self):
        self.engine.blow_fuse_secure_boot()
        img = self.engine.create_image(BootStage.BL1, version=1, payload=b"bl1")
        self.engine.verify_stage(img)
        assert BootStage.BL1 in self.engine.verified_stages

    def test_verify_tampered_hash_raises(self):
        self.engine.blow_fuse_secure_boot()
        img = self.engine.create_image(BootStage.BL1, version=1, payload=b"bl1")
        # Tamper with hash
        tampered = BootImage(
            stage=img.stage, version=img.version, payload=img.payload,
            signature=img.signature,
            hash_value=bytes(32),  # zeroed hash
        )
        with pytest.raises(SecureBootError):
            self.engine.verify_stage(tampered)

    def test_verify_tampered_payload_raises(self):
        self.engine.blow_fuse_secure_boot()
        img = self.engine.create_image(BootStage.BL1, version=1, payload=b"bl1")
        tampered = BootImage(
            stage=img.stage, version=img.version,
            payload=b"tampered",          # different payload
            signature=img.signature,
            hash_value=img.hash_value,    # hash no longer matches payload
        )
        with pytest.raises(SecureBootError):
            self.engine.verify_stage(tampered)

    def test_verify_bad_signature_raises(self):
        self.engine.blow_fuse_secure_boot()
        img = self.engine.create_image(BootStage.BL1, version=1, payload=b"bl1")
        tampered = BootImage(
            stage=img.stage, version=img.version, payload=img.payload,
            signature=bytes(32),          # zeroed signature
            hash_value=img.hash_value,
        )
        with pytest.raises(SignatureError):
            self.engine.verify_stage(tampered)

    def test_verify_rollback_raises(self):
        self.engine.blow_fuse_secure_boot()
        self.engine.set_min_version(5)
        img = self.engine.create_image(BootStage.BL1, version=3, payload=b"old")
        with pytest.raises(RollbackError):
            self.engine.verify_stage(img)

    def test_verify_without_secure_boot_skips_signature(self):
        # Secure boot not enabled: only hash is checked, signature bypassed
        engine = SecureBootEngine(root_key=b"test-key")
        img = engine.create_image(BootStage.BL1, version=1, payload=b"bl1")
        # Corrupt signature — should still pass (secure boot disabled)
        img_bad_sig = BootImage(
            stage=img.stage, version=img.version, payload=img.payload,
            signature=bytes(32),
            hash_value=img.hash_value,
        )
        engine.verify_stage(img_bad_sig)  # should not raise
        assert BootStage.BL1 in engine.verified_stages

    def test_measurement_log_grows(self):
        self.engine.blow_fuse_secure_boot()
        for stage in (BootStage.BL1, BootStage.BL2, BootStage.OS):
            img = self.engine.create_image(stage, version=1, payload=b"code")
            self.engine.verify_stage(img)
        assert len(self.engine.measurement_log()) == 3

    def test_combined_measurement_is_bytes32(self):
        img = self.engine.create_image(BootStage.BL1, version=1, payload=b"x")
        self.engine.verify_stage(img)
        assert len(self.engine.combined_measurement()) == 32


class TestSecureBootStateMachine:
    def setup_method(self):
        self.engine = SecureBootEngine(root_key=b"key")
        self.engine.blow_fuse_secure_boot()

    def test_initial_state_uninit(self):
        engine = SecureBootEngine()
        assert engine.state == BootState.UNINIT

    def test_advance_after_bl1(self):
        img = self.engine.create_image(BootStage.BL1, version=1, payload=b"b")
        self.engine.verify_stage(img)
        self.engine.advance()
        assert self.engine.state == BootState.BL1_OK

    def test_advance_after_os(self):
        for stage in (BootStage.BL1, BootStage.BL2, BootStage.OS):
            img = self.engine.create_image(stage, version=1, payload=b"x")
            self.engine.verify_stage(img)
        self.engine.advance()
        assert self.engine.state == BootState.BOOT_COMPLETE

    def test_mark_failed(self):
        self.engine.mark_failed()
        assert self.engine.state == BootState.FAILED

    def test_repr(self):
        r = repr(self.engine)
        assert "SecureBootEngine" in r


# ===========================================================================
# TEE tests
# ===========================================================================

class TestTEELifecycle:
    def test_initialize(self):
        tee = TEE()
        assert not tee.is_initialized()
        tee.initialize()
        assert tee.is_initialized()

    def test_initial_world_normal(self):
        tee = TEE()
        tee.initialize()
        assert tee.current_world == WorldState.NORMAL

    def test_switch_to_secure(self):
        tee = TEE()
        tee.initialize()
        tee.switch_to_secure()
        assert tee.current_world == WorldState.SECURE

    def test_switch_back_to_normal(self):
        tee = TEE()
        tee.initialize()
        tee.switch_to_secure()
        tee.switch_to_normal()
        assert tee.current_world == WorldState.NORMAL

    def test_smc_call_counter(self):
        tee = TEE()
        tee.initialize()
        tee.switch_to_secure()
        tee.switch_to_normal()
        tee.switch_to_secure()
        assert tee.smc_call_count == 3

    def test_switch_without_init_raises(self):
        tee = TEE()
        with pytest.raises(TEEError):
            tee.switch_to_secure()

    def test_repr(self):
        tee = TEE()
        tee.initialize()
        r = repr(tee)
        assert "TEE" in r


class TestTEESecureMemory:
    def setup_method(self):
        self.tee = TEE()
        self.tee.initialize()
        self.tee.switch_to_secure()

    def test_add_secure_region(self):
        self.tee.add_secure_region(0x10000000, 0x1000, "secure_sram")
        assert self.tee.secure_region_count() == 1

    def test_address_in_secure_region(self):
        self.tee.add_secure_region(0x10000000, 0x1000, "sram")
        assert self.tee.check_address(0x10000100)

    def test_address_outside_secure_region(self):
        self.tee.add_secure_region(0x10000000, 0x1000, "sram")
        assert not self.tee.check_address(0x20000000)

    def test_add_region_from_normal_world_raises(self):
        self.tee.switch_to_normal()
        with pytest.raises(TEEError):
            self.tee.add_secure_region(0x10000000, 0x1000, "sram")

    def test_access_secure_memory_from_secure_ok(self):
        self.tee.add_secure_region(0x10000000, 0x1000, "sram")
        data = self.tee.access_secure_memory(0x10000000)
        assert isinstance(data, bytes)

    def test_access_secure_memory_from_normal_raises(self):
        self.tee.add_secure_region(0x10000000, 0x1000, "sram")
        self.tee.switch_to_normal()
        with pytest.raises(TEEError):
            self.tee.access_secure_memory(0x10000000)

    def test_access_violation_counter(self):
        self.tee.add_secure_region(0x10000000, 0x1000, "sram")
        self.tee.switch_to_normal()
        with pytest.raises(TEEError):
            self.tee.access_secure_memory(0x10000000)
        assert self.tee.access_violations == 1


class TestTEEKeyVault:
    def setup_method(self):
        self.tee = TEE()
        self.tee.initialize()
        self.tee.switch_to_secure()

    def test_generate_key(self):
        key = self.tee.generate_key("k1", [KeyUsage.SIGN])
        assert self.tee.key_exists("k1")
        assert len(key.key_material) == 32

    def test_generate_key_from_normal_raises(self):
        self.tee.switch_to_normal()
        with pytest.raises(TEEError):
            self.tee.generate_key("k1", [KeyUsage.SIGN])

    def test_duplicate_key_raises(self):
        self.tee.generate_key("k1", [KeyUsage.SIGN])
        with pytest.raises(TEEError):
            self.tee.generate_key("k1", [KeyUsage.SIGN])

    def test_delete_key(self):
        self.tee.generate_key("k1", [KeyUsage.SIGN])
        self.tee.delete_key("k1")
        assert not self.tee.key_exists("k1")

    def test_delete_nonexistent_raises(self):
        with pytest.raises(TEEError):
            self.tee.delete_key("ghost")

    def test_export_non_exportable_raises(self):
        self.tee.generate_key("k1", [KeyUsage.SIGN], exportable=False)
        with pytest.raises(TEEError):
            self.tee.export_key("k1")

    def test_export_exportable_key(self):
        self.tee.generate_key("k1", [KeyUsage.SIGN], exportable=True)
        material = self.tee.export_key("k1")
        assert len(material) == 32

    def test_key_count(self):
        self.tee.generate_key("k1", [KeyUsage.SIGN])
        self.tee.generate_key("k2", [KeyUsage.ENCRYPT])
        assert self.tee.key_count() == 2


class TestTEECrypto:
    def setup_method(self):
        self.tee = TEE()
        self.tee.initialize()
        self.tee.switch_to_secure()
        self.tee.generate_key("sign_key", [KeyUsage.SIGN, KeyUsage.VERIFY])

    def test_sign_returns_32_bytes(self):
        sig = self.tee.sign("sign_key", b"hello world")
        assert len(sig) == 32

    def test_verify_valid_signature(self):
        data = b"test data"
        sig = self.tee.sign("sign_key", data)
        assert self.tee.verify("sign_key", data, sig)

    def test_verify_invalid_signature(self):
        data = b"test data"
        sig = self.tee.sign("sign_key", data)
        bad_sig = bytes(b ^ 0xFF for b in sig)
        assert not self.tee.verify("sign_key", data, bad_sig)

    def test_sign_from_normal_world_raises(self):
        self.tee.switch_to_normal()
        with pytest.raises(TEEError):
            self.tee.sign("sign_key", b"data")

    def test_sign_wrong_usage_raises(self):
        self.tee.generate_key("enc_only", [KeyUsage.ENCRYPT])
        with pytest.raises(TEEError):
            self.tee.sign("enc_only", b"data")

    def test_derive_key(self):
        self.tee.generate_key("root", [KeyUsage.DERIVE])
        child = self.tee.derive_key("root", b"context", "child_key")
        assert self.tee.key_exists("child_key")
        assert len(child.key_material) == 32

    def test_derive_from_non_derive_key_raises(self):
        with pytest.raises(TEEError):
            self.tee.derive_key("sign_key", b"ctx", "child")

    def test_sign_nonexistent_key_raises(self):
        with pytest.raises(TEEError):
            self.tee.sign("ghost", b"data")


# ===========================================================================
# OTA tests
# ===========================================================================

class TestOTAPackage:
    def setup_method(self):
        self.ota = OTAManager(signing_key=b"test-ota-key")

    def test_create_package(self):
        pkg = self.ota.create_package(version=2, payload=b"firmware_v2")
        assert pkg.version == 2
        assert len(pkg.signature) == 32
        assert len(pkg.payload_hash) == 32

    def test_package_hash_matches_payload(self):
        import hashlib
        pkg = self.ota.create_package(version=2, payload=b"fw")
        assert pkg.payload_hash == hashlib.sha256(b"fw").digest()


class TestOTAUpdatePipeline:
    def setup_method(self):
        self.ota = OTAManager(signing_key=b"test-ota-key", min_version=1)

    def _do_full_update(self, version: int, payload: bytes = b"fw") -> None:
        pkg = self.ota.create_package(version=version, payload=payload)
        self.ota.begin_update(pkg)
        self.ota.validate()
        self.ota.apply()

    def test_initial_state_idle(self):
        assert self.ota.state == UpdateState.IDLE

    def test_full_update_succeeds(self):
        self._do_full_update(version=2)
        assert self.ota.state == UpdateState.COMPLETE

    def test_active_slot_version_updated(self):
        self._do_full_update(version=2)
        assert self.ota.active_slot().version == 2

    def test_update_count_increments(self):
        self._do_full_update(version=2)
        self._do_full_update(version=3)
        assert self.ota.update_count == 2

    def test_sequential_updates_increase_version(self):
        self._do_full_update(version=2)
        self._do_full_update(version=3)
        assert self.ota.active_slot().version == 3

    def test_slots_alternate(self):
        # Slot A starts active
        first_active = self.ota.active_slot().slot_id
        self._do_full_update(version=2)
        second_active = self.ota.active_slot().slot_id
        assert first_active != second_active


class TestOTAValidation:
    def setup_method(self):
        self.ota = OTAManager(signing_key=b"test-ota-key", min_version=1)

    def test_tampered_payload_fails(self):
        pkg = self.ota.create_package(version=2, payload=b"firmware")
        # Tamper payload after signing — hash auto-updates so signature mismatch fires
        tampered = UpdatePackage(
            version=pkg.version,
            payload=b"tampered",
            signature=pkg.signature,
        )
        self.ota.begin_update(tampered)
        from security import SecurityBaseError
        with pytest.raises(SecurityBaseError):
            self.ota.validate()
        assert self.ota.state == UpdateState.FAILED

    def test_bad_signature_fails(self):
        pkg = self.ota.create_package(version=2, payload=b"fw")
        bad = UpdatePackage(
            version=pkg.version,
            payload=pkg.payload,
            signature=bytes(32),  # zeroed
        )
        self.ota.begin_update(bad)
        with pytest.raises(SignatureError):
            self.ota.validate()

    def test_same_version_fails_rollback(self):
        # Active slot is at version 1
        pkg = self.ota.create_package(version=1, payload=b"fw")
        self.ota.begin_update(pkg)
        with pytest.raises(RollbackError):
            self.ota.validate()

    def test_older_version_fails_rollback(self):
        pkg = self.ota.create_package(version=0, payload=b"fw")
        self.ota.begin_update(pkg)
        with pytest.raises(RollbackError):
            self.ota.validate()

    def test_below_min_version_fails(self):
        self.ota.set_min_version(5)
        pkg = self.ota.create_package(version=3, payload=b"fw")
        self.ota.begin_update(pkg)
        with pytest.raises(RollbackError):
            self.ota.validate()

    def test_validate_wrong_state_raises(self):
        with pytest.raises(OTAError):
            self.ota.validate()

    def test_apply_wrong_state_raises(self):
        with pytest.raises(OTAError):
            self.ota.apply()


class TestOTARollback:
    def setup_method(self):
        self.ota = OTAManager(signing_key=b"test-ota-key")

    def test_rollback_after_failed(self):
        pkg = self.ota.create_package(version=2, payload=b"fw")
        bad = UpdatePackage(version=pkg.version, payload=pkg.payload, signature=bytes(32))
        self.ota.begin_update(bad)
        with pytest.raises(SignatureError):
            self.ota.validate()
        self.ota.rollback()
        assert self.ota.state == UpdateState.ROLLED_BACK

    def test_rollback_preserves_active_slot(self):
        original_version = self.ota.active_slot().version
        pkg = self.ota.create_package(version=5, payload=b"fw")
        bad = UpdatePackage(version=pkg.version, payload=pkg.payload, signature=bytes(32))
        self.ota.begin_update(bad)
        with pytest.raises(SignatureError):
            self.ota.validate()
        self.ota.rollback()
        assert self.ota.active_slot().version == original_version

    def test_rollback_count_increments(self):
        pkg = self.ota.create_package(version=2, payload=b"fw")
        bad = UpdatePackage(version=pkg.version, payload=pkg.payload, signature=bytes(32))
        self.ota.begin_update(bad)
        with pytest.raises(SignatureError):
            self.ota.validate()
        self.ota.rollback()
        assert self.ota.rollback_count == 1

    def test_set_min_version_cannot_decrease(self):
        self.ota.set_min_version(10)
        with pytest.raises(RollbackError):
            self.ota.set_min_version(5)


class TestOTASlots:
    def setup_method(self):
        self.ota = OTAManager(signing_key=b"test-ota-key")

    def test_get_slot_a(self):
        slot = self.ota.get_slot(SlotID.SLOT_A)
        assert slot.slot_id == SlotID.SLOT_A

    def test_get_slot_b(self):
        slot = self.ota.get_slot(SlotID.SLOT_B)
        assert slot.slot_id == SlotID.SLOT_B

    def test_repr(self):
        r = repr(self.ota)
        assert "OTAManager" in r

    def test_begin_update_wrong_state_raises(self):
        pkg = self.ota.create_package(version=2, payload=b"fw")
        self.ota.begin_update(pkg)
        # Already in VALIDATING, can't begin again
        pkg2 = self.ota.create_package(version=3, payload=b"fw")
        with pytest.raises(OTAError):
            self.ota.begin_update(pkg2)
