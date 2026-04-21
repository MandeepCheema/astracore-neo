# AstraCore Neo IP — Safety Manual

> **SUPERSEDED 2026-04-20** by `docs/safety/safety_manual_v0_5.md` — that revision fills out the licensee-critical sections (§3.4 safe-state behaviour, §4 clock/reset, §5 init sequencing, §6 watchdog handling, §7 fault signalling incl. full `fault_detected[15:0]` bitfield, §8 secure boot reference flow, §10 AoU index, §11 licensee verification activities). This v0.1 skeleton is retained as revision history.

**Document ID:** ASTR-SAFETY-MANUAL-V0.1
**Date:** 2026-04-20
**Status:** v0.1 — outline / skeleton. Sections marked TBD complete during weeks W2–W12.
**Standard:** ISO 26262-10 §9 + ISO 26262-11 §4.7 (IP supplier safety manual)
**Companion documents:**
- `docs/safety/seooc_declaration_v0_1.md` — SEooC declaration (the contractual element)
- `docs/safety/iso26262_gap_analysis_v0_1.md` — process gap analysis
- `docs/best_in_class_design.md` — strategic context (§7 founder direction)

> **Purpose.** The Safety Manual is the document a licensee uses to integrate AstraCore Neo IP into their item-level safety case. It tells the licensee **what to do, what not to do, what assumptions to respect, and what diagnostics to wire** in order to inherit the AstraCore safety case.
>
> The SEooC declaration is the *contract*; the Safety Manual is the *user guide*. Both are required deliverables to a licensee.

---

## 1. Introduction

### 1.1 Scope
TBD — copy from SEooC §2 once confirmed.

### 1.2 Intended audience
- Licensee functional-safety manager
- Licensee SoC integration engineers
- Licensee verification engineers
- Vehicle OEM safety reviewer (read-only)
- TÜV / external assessor (audit reference)

### 1.3 How to read this manual
Each section pairs **what AstraCore provides** with **what the licensee must do**. Mandatory steps are marked **[REQUIRED]**; recommended steps are **[RECOMMENDED]**.

---

## 2. IP element overview

### 2.1 Block diagram
TBD — link to `docs/architecture.md` once a safety-annotated version exists. Plan: produce safety-annotated block diagram by W4 with TMR/ECC/safe-state nets highlighted.

### 2.2 Lifecycle
TBD — describe development → integration → operation → decommissioning.

### 2.3 Versioning and configuration management
- Repo HEAD on date of release is the baseline.
- Safety-relevant changes follow change-impact analysis per ISO 26262-8 §8.
- Each licensee receives a tagged baseline; field updates require licensee re-verification.

---

## 3. Configuration parameters

### 3.1 MAC array sizing
- `N_ROWS × N_COLS` valid range: 4×4 (development) ↔ 48×512 (full spec).
- Performance / power / area scale documented in `docs/best_in_class_design.md` §7.4 once Track 1 measurements complete.
- **[REQUIRED]** Set `WEIGHT_DEPTH = N_ROWS × N_COLS` per the latent issue documented in `memory/pre_awsf1_gaps_complete.md`. Failure to do so produces silent data corruption at compile time.

### 3.2 Sensor I/O selection
TBD — document which sensor interfaces can be disabled and how.

### 3.3 Safety mechanism enables
- TMR voter: always-on for safety-critical paths; cannot be disabled (compiled-in).
- SECDED ECC: always-on for SRAM banks.
- Plausibility checker: per-sensor enable.
- Watchdog: per-sensor timeout configurable per ASR-HW-08.

### 3.4 Safe-state behaviour
TBD — document `safe_state_active` assertion conditions and licensee handshake.

---

## 4. Clock and reset

TBD — clock source requirements (jitter, frequency stability, PLL bypass), reset protocol (asynchronous assert, synchronous deassert, minimum width, post-reset stabilization).

---

## 5. Reset and initialization sequencing

TBD — power-on reset → BIST (if implemented) → boot signature verification → mission mode handshake.

---

## 6. Watchdog handling

TBD — `watchdog_kick` period, late-kick window, escalation behavior.

---

## 7. Fault signalling

### 7.1 `safe_state_active` semantics
TBD — defined safe state, latency from internal fault to assertion, deassert behaviour.

### 7.2 `fault_detected[15:0]` bitfield
TBD — table mapping each bit to fault category + recommended licensee response.

### 7.3 Counter signals
- `tmr_disagree_count[7:0]` — saturating count of TMR disagreements; clearable on cold reset.
- `ecc_corrected_count[15:0]` — saturating count of ECC single-bit corrections.
- `ecc_uncorrected_count[7:0]` — saturating count of ECC double-bit detections; **[REQUIRED]** licensee escalates per item-level ASIL.

---

## 8. Boot integrity (secure boot)

TBD — RSA-2048 + SHA-256 signature verification of weights and configuration. Public key provisioning per AoU-9. Track 3 deliverable (W4-W7).

---

## 9. Diagnostic services

### 9.1 Built-in diagnostics
TBD — list of always-on diagnostics.

### 9.2 On-demand diagnostics
TBD — LBIST trigger interface (if implemented), MBIST coverage.

### 9.3 Field diagnostic data export
TBD — interface to read out fault counters + history for field monitoring per ISO 26262-7 §6.

---

## 10. Assumptions of use (AoU)

This is a denormalized index of the AoUs declared in the SEooC §6. The licensee must respect every entry in this table.

| AoU | Summary | Verification by licensee |
|---|---|---|
| AoU-1 to AoU-4 | Operational environment | Licensee silicon + package qualification reports |
| AoU-5 to AoU-8 | Functional integration | Licensee SoC integration test |
| AoU-9 to AoU-12 | Software / data integration | Licensee toolchain qualification |
| AoU-13 to AoU-15 | Process | Signed DIA + audit trail |
| AoU-16 to AoU-17 | Configuration | Licensee parameter freeze + characterization |

Full text: see `docs/safety/seooc_declaration_v0_1.md` §6.

---

## 11. Verification activities required by licensee

TBD — minimum integration test set (functional, safety-mechanism, fault-injection).

---

## 12. Reporting field anomalies

TBD — AstraCore field-monitoring contact, severity classification, response SLA. Required by AoU-14.

---

## 13. Document control

| Field | Value |
|---|---|
| Document ID | ASTR-SAFETY-MANUAL-V0.1 |
| Revision | 0.1 (skeleton) |
| Revision date | 2026-04-20 |
| Author | TBD (Track 2 lead at W1 hire) |
| Reviewer | TBD |
| Approver | TBD |
| Distribution | Internal only at v0.1; first licensee draft at v1.0 |
| Retention | 15 years post-product-discontinuation |
| Supersedes | None |

### 13.1 Completion plan

| Wk | Section completed |
|---|---|
| 2 | §1, §2.3, §3.1 (already partial) |
| 4 | §2.1 safety-annotated block diagram, §3.4 safe-state, §4 clock/reset |
| 6 | §5 init sequence, §6 watchdog, §7 fault signalling |
| 7 | §8 secure boot (with Track 3 OpenTitan integration) |
| 9 | §9 diagnostic services |
| 11 | §11 licensee verification, §12 field anomalies |
| 12 | v1.0 release for TÜV interim review |
