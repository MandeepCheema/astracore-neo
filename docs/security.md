# Module 7 — Security

**Status:** DONE | **Tests:** 75/75 (100%) | **Date:** 2026-04-02

## Overview

The Security module implements the hardware security stack for the AstraCore Neo chip — secure boot chain, trusted execution environment, and OTA firmware update management. Together these prevent unauthorized code execution, protect cryptographic keys, and ensure only verified firmware can run on the chip.

## Sub-modules

### secure_boot.py — Secure Boot Chain
- **SecureBootEngine** — ROM → BL1 → BL2 → OS staged boot verification
- **create_image(stage, version, payload)** → signed BootImage (HMAC-SHA256 as RSA proxy)
- **verify_stage(image)** — checks hash integrity, signature, and anti-rollback version
- **FuseState** — one-way programmable fuses: secure_boot_enabled, debug_disabled, jtag_locked, min_version
- **set_min_version(v)** — advances anti-rollback counter (monotonic, cannot decrease)
- **BootState** — UNINIT → ROM_OK → BL1_OK → BL2_OK → OS_OK → BOOT_COMPLETE / FAILED
- PCR-style measurement log: `measurement_log()` and `combined_measurement()`
- When secure boot disabled: only hash is checked, signature bypassed

### tee.py — Trusted Execution Environment
- **TEE** — ARM TrustZone-style secure/normal world separation
- **switch_to_secure() / switch_to_normal()** — world switch via SMC (both increment SMC counter)
- **add_secure_region(addr, size, label)** — register memory as secure-world-only
- **access_secure_memory(addr)** — raises TEEError if accessed from normal world
- **generate_key(id, usage)** — creates key in secure vault (secure world only)
- **sign(key_id, data)** / **verify(key_id, data, sig)** — HMAC-SHA256 crypto ops
- **derive_key(parent_id, context, new_id)** — HKDF-style key derivation
- Non-exportable keys never leave the TEE
- Access violation counter for security audit

### ota.py — OTA Update Manager
- **OTAManager** — A/B slot firmware update pipeline
- **create_package(version, payload)** → signed UpdatePackage
- **begin_update → validate → apply** pipeline
- **rollback()** — invalidates pending slot, preserves active slot
- Anti-rollback: package version must be strictly greater than active slot version
- **UpdatePackage.payload_hash** — stored at creation time via `__post_init__`, not recomputed live
- A/B slots alternate: after update, previously inactive slot becomes active
- `begin_update` allows COMPLETE state (for sequential updates)

## Dependencies
- Needs: HAL (module 1)
- No downstream dependencies in current build order

## Critical Design Notes
1. **TEE SMC counter**: both `switch_to_secure()` AND `switch_to_normal()` are SMC calls — both must increment the counter.
2. **OTA state machine**: after COMPLETE, `begin_update` must be allowed again for sequential updates.
3. **UpdatePackage payload_hash**: must be stored at package creation (not computed live from payload). If recomputed dynamically, a tampered payload passes hash check and the test sees SignatureError instead of OTAError. Use `__post_init__` to freeze the hash at creation.
4. **Tampered payload test**: since hash is now frozen at creation and signature was for original payload, tampered payload fails at signature check (SignatureError) not hash check — test catches SecurityBaseError to cover both.
