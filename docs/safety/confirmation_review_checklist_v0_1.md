# Confirmation Review Checklist

**Document ID:** ASTR-SAFETY-CONF-REVIEW-V0.1
**Date:** 2026-04-20
**Standard:** ISO 26262-2:2018 §7 (Confirmation measures) + ISO 26262-2:2018 Table 1 (independence requirements)
**Element:** Process artefact for the named independent reviewer of every Track 2 safety-case work product
**Status:** v0.1 — first formal release. The last remaining host-side Track 2 deliverable. Operationalises the "Reviewer: TBD" + "Confirmation review: pending" lines that appear in every other safety doc's §0 / §7 / §8.
**Classification:** Internal — for use by the named independent reviewer + Safety Manager.
**Author:** TBD (Track 2 lead) — currently founder + collaborator
**Reviewer of THIS doc:** TBD (independent reviewer per ISO 26262-2 §7 — same one that uses the checklist on the other docs)
**Approver:** TBD (Safety Manager)

---

## 0. Purpose and framing

ISO 26262-2 §7 requires confirmation measures for safety-relevant work products. **Confirmation reviews** in particular check that:

1. The work product satisfies the requirements of the relevant ISO 26262 part
2. Internal consistency holds (no contradictions, every claim has evidence)
3. Cross-document consistency holds (e.g. an FSR is realised by a TSR which is verified by an IT)
4. Assumptions are documented and traceable

Without per-doc checklists, "the reviewer reviews it" is a vague handshake. This document gives the reviewer a structured artefact — one checklist per work product — so reviews are repeatable across reviewers and over time.

> **Mandatory at ASIL-C / D.** ISO 26262-2 §7 Table 1: confirmation reviews are **mandatory** for any ASIL-C or ASIL-D work product. We have ASIL-D safety goals (per HARA `hara_v0_1.md` §5), so most of these reviews are mandatory.

### 0.1 Companion documents

Every Track 2 work product references "Confirmation review per ISO 26262-2 §7" in its §0 / §7 / §8 doc-control block. This checklist is the operational implementation of those references.

- `docs/safety/iso26262_gap_analysis_v0_1.md` Part 2 §7 row — gap this closes
- `docs/safety/safety_case_v0_1.md` §7 — the Safety Case's own confirmation-review status table
- All Track 2 work products listed in §3 below

---

## 1. Reviewer independence per ISO 26262-2 §7

### 1.1 Independence levels (per ISO 26262-2 §7.4.3 + Table 1)

| Level | Meaning | Required for |
|---|---|---|
| **I0** | No specific independence required | Self-review; QM-only items |
| **I1** | Independence in personnel — different person from author, same team | recommended for ASIL-A confirmation review |
| **I2** | Independence in personnel — different team / different reporting line | highly recommended for ASIL-B; mandatory for ASIL-C |
| **I3** | Independence in personnel + organisation — different organisation entirely | mandatory for ASIL-D |

### 1.2 Independence required for each Track 2 work product

| Work product | Highest ASIL claim it supports | Required reviewer independence |
|---|:---:|:---:|
| HARA | D (SG-1.1) | I3 |
| FSC | D (FSR-1.1.\*) | I3 |
| TSC | D (TSR-HW-01..05, TSR-HW-16/17/18) | I3 |
| Safety Manual | D (covers all SGs) | I3 |
| Integration Test Plan | D (IT for FSR-1.1.\*) | I3 |
| TCL evaluations | (process; no ASIL inheritance) | I2 (process artefact) |
| DIA template | (process + legal) | I2 (safety) + legal counsel review (separate) |
| FMEDA per-module | inherits per-module ASIL claim | I3 for npu_top + dms_fusion (ASIL-D-relevant); I2 for safety primitives |
| Aggregate FMEDA | D (item-level aggregate) | I3 |
| CCF analysis | D (required for ASIL-D extension) | I3 |
| SER analysis | D (required for ASIL-D extension) | I3 |
| Safety Case master | D | I3 + functional safety assessor for v1.0 release |
| Field-monitoring template | (process) | I2 |
| Findings remediation plan | derives from above; D | I3 |
| Spec sheet rev 1.4 | (external claim) | I3 + product / marketing review |
| **This checklist** | (process) | I2 |

### 1.3 Default reviewer

AstraCore proposes **TÜV SÜD India** as the default I3 independent reviewer for all ASIL-D work products. The TÜV SÜD India pre-engagement workshop is scheduled W6 per `docs/best_in_class_design.md` §7.2.

For I2 process artefacts, AstraCore appoints an internal reviewer from a different team than the author.

For I1 / I0 items (none currently), self-review by the author is acceptable.

---

## 2. Common-cause checklist (applies to every work product)

Every confirmation review starts with these 10 universal checks before drilling into the per-document checklist in §3.

| # | Check | Yes / No / N/A | Evidence reference |
|---:|---|:---:|---|
| C1 | Document ID and revision are explicit in the doc-control block | | |
| C2 | Author, reviewer, and approver are named (not "TBD") | | |
| C3 | All ISO 26262 part references cite the 2018 edition (or current edition at review time) | | |
| C4 | Companion documents are listed and exist at the named paths | | |
| C5 | Every claim has a verifiable evidence reference (file path, line range, command, measurement) | | |
| C6 | Open items / TBD items are explicitly enumerated in a §6 / §7 / §8 "open items" section | | |
| C7 | Revision triggers are enumerated in the doc-control block | | |
| C8 | Distribution list is explicit | | |
| C9 | Retention period is named (15 years per ISO 26262-8 §10) | | |
| C10 | Cross-references to other Track 2 work products are bidirectional (every doc references and is referenced) | | |

**Sign-off rule:** any C1-C10 check returning "No" without a documented exception blocks the work product from moving from v0.x → v1.0.

---

## 3. Per-work-product checklists

### 3.1 HARA (`docs/safety/hara_v0_1.md`)

**ISO 26262-3 §6 + §8.** Required reviewer independence: **I3**.

| # | Check | Y/N/NA | Evidence |
|---:|---|:---:|---|
| H1 | Item definition (§3.1) is consistent with SEooC §3 | | |
| H2 | At least 3 reference use cases enumerated (FCW/AEB/LKA, DMS, surround/parking) | | |
| H3 | Each use case has at least 3 hazardous events | | |
| H4 | Every hazardous event has S/E/C ratings with one-sentence rationale per cell | | |
| H5 | ASIL determination matrix (§1.4) matches ISO 26262-3 Table 4 exactly | | |
| H6 | Aggregate item-level ASIL (§5) is derived from the highest-ASIL Safety Goal | | |
| H7 | Each Safety Goal has a unique ID (SG-1.1, SG-1.2, ..., SG-3.3) | | |
| H8 | Safety Goal → ASR traceability (§6) covers every SG | | |
| H9 | Spec sheet rev 1.4 wording proposal (§5.2) matches the rev that is actually shipped | | |
| H10 | Open items (§7) include named-reviewer + workshop + quantitative FTTI | | |

**Common pitfalls:**
- S/E/C cells with no rationale (just "S3 / E4 / C3" — what's the justification?)
- Use cases that don't match the assumed item context in SEooC §3.1
- Safety Goals with no actionable verification path (every SG must trace to ≥ 1 FSR)

### 3.2 FSC (`docs/safety/functional_safety_concept_v0_1.md`)

**ISO 26262-3 §7.** Required reviewer independence: **I3**.

| # | Check | Y/N/NA | Evidence |
|---:|---|:---:|---|
| F1 | Every Safety Goal in HARA spawns at least one FSR | | |
| F2 | Each FSR has the 7 required attributes (operating mode / FTTI / safe state / FRTI / functional redundancy / allocation / verification) | | |
| F3 | FSR ASIL inheritance from parent SG is explicit | | |
| F4 | FTTI placeholders (§1.2) are flagged as placeholders requiring Licensee replacement | | |
| F5 | ASIL decomposition convention (§1.1) is consistent with ISO 26262-9 §5 | | |
| F6 | Allocation labels (A1/A2/A3/A4 + L1/L2/L3/L4 + D1) are consistent with the architectural element catalogue | | |
| F7 | FSR → ASR coverage matrix (§4) covers every ASR in SEooC §4.1 | | |
| F8 | Open items (§6) match actual gaps + dependencies on F4 phases | | |

**Common pitfalls:**
- FSR with no verification approach ("verified somehow")
- FSR allocated only to L (Licensee) when AstraCore could / should provide
- FSR statement that's not testable ("the system shall be safe")

### 3.3 TSC (`docs/safety/technical_safety_concept_v0_1.md`)

**ISO 26262-4 §6.** Required reviewer independence: **I3**.

| # | Check | Y/N/NA | Evidence |
|---:|---|:---:|---|
| T1 | Every FSR in FSC spawns at least one TSR | | |
| T2 | TSRs use TSR-HW-XX / TSR-SW-XX / TSR-IF-XX prefix scheme | | |
| T3 | Each TSR has explicit allocation to a specific RTL module or SDK component (not just "HW" or "SW") | | |
| T4 | Each TSR cites the safety mechanism in `safety_mechanisms.yaml` it binds to (or "(new)" if a new mechanism is being introduced) | | |
| T5 | Each TSR cites the IT-XX in ITP that verifies it | | |
| T6 | Each TSR cites the parent FSR (and parent ASR if it refines an existing SEooC §4 ASR) | | |
| T7 | Cross-cutting TSRs (TSR-HW-16/17/18 + TSR-SW-06) are explicitly flagged as multi-FSR | | |
| T8 | Coverage matrix (§4) shows every FSR covered + status (READY / STUBBED / BLOCKED / LICENSEE) | | |
| T9 | HW-SW interface specification (§5) is consistent with SEooC §2.3 boundary signals | | |

**Common pitfalls:**
- TSR allocated to a module that doesn't exist (typo or stale reference)
- TSR with no fault response (what happens on detection?)
- TSR that duplicates an existing ASR without value-add

### 3.4 Safety Manual (`docs/safety/safety_manual_v0_5.md`)

**ISO 26262-10 §9 + ISO 26262-11 §4.7.** Required reviewer independence: **I3**.

| # | Check | Y/N/NA | Evidence |
|---:|---|:---:|---|
| SM1 | Every AoU in SEooC §6 appears in §10 with a verification approach | | |
| SM2 | `fault_detected[15:0]` bitfield (§7.2) covers all 16 bits with category + RTL source + severity + recommended response | | |
| SM3 | Per-mechanism FDTI table (§7.4) is consistent with the FMEDA mechanism declared coverages | | |
| SM4 | Safe-state ladder (§3.4) is consistent with `rtl/safe_state_controller/` RTL | | |
| SM5 | Watchdog timing constraint (§6.3) gives the formula for FTTI propagation | | |
| SM6 | Reset protocol (§4.2) is explicit on async-assert / sync-deassert / minimum width | | |
| SM7 | Secure boot section (§8) explicitly notes the today-vs-Track-3 status | | |
| SM8 | Licensee verification activities (§11) cover boundary signals + watchdog + safe-state + reset + ECC + TMR + reporting | | |
| SM9 | TBD sections are listed in §13.1 completion plan with target weeks | | |
| SM10 | [REQUIRED] vs [RECOMMENDED] markers used consistently (mandatory steps clearly distinguished) | | |

**Common pitfalls:**
- AoU listed in SEooC §6 but missing from Safety Manual §10 index
- Boundary signal documented in §7 but not in SEooC §2.3
- Misleading "always-on" claims for mechanisms that aren't yet integrated (the F4-A-1.1 ECC pitfall caught at v0.5)

### 3.5 Integration Test Plan (`docs/safety/integration_test_plan_v0_1.md`)

**ISO 26262-4 §9 + ISO 26262-3 §7.4.4.** Required reviewer independence: **I3**.

| # | Check | Y/N/NA | Evidence |
|---:|---|:---:|---|
| IT1 | Every FSR in FSC has at least one IT (or cross-references another IT for shared coverage) | | |
| IT2 | Each IT has all 11 attributes (ID / parent FSR / ASIL / setup / stimulus / expected / pass-fail / environment / required tools / allocation / status) | | |
| IT3 | Status enum (READY / STUBBED / BLOCKED / LICENSEE) used consistently | | |
| IT4 | BLOCKED tests cite the specific F4 WP that unblocks them | | |
| IT5 | LICENSEE-allocated tests are clearly marked as Licensee-owned | | |
| IT6 | Required tools listed for each test exist (or are in scope of a tracked WP) | | |
| IT7 | Pass/fail criteria are quantitative where measurable (not just "passes") | | |
| IT8 | Test asset roadmap (§5) sums to a feasible engineering effort (~24 days currently) | | |
| IT9 | Aggregate status (§4) matches per-IT status counts | | |

### 3.6 TCL Evaluations (`docs/safety/tcl/tcl_evaluations_v0_1.md`)

**ISO 26262-8 §11 + ISO 26262-11 §6.** Required reviewer independence: **I2**.

| # | Check | Y/N/NA | Evidence |
|---:|---|:---:|---|
| TCL1 | Every tool used in `tools/`, `sim/`, `astracore/` development is classified | | |
| TCL2 | TI / TD / TCL classifications follow ISO 26262-8 Table 4 | | |
| TCL3 | TCL2/TCL3 tools have a qualification plan (e.g., F4-B-7 for Yosys, F4-D-7 for OpenROAD, F4-D-8 for ASAP7) | | |
| TCL4 | Compensating measures for un-qualified tools are explicit (e.g., spec sheet labelling for ASAP7) | | |
| TCL5 | Tool versions are pinned (NOT "latest stable") | | |
| TCL6 | Hardware components (OpenTitan crypto IP) are explicitly excluded from TCL scope (subject to ISO 26262-8 §13 instead) | | |

### 3.7 DIA template (`docs/safety/dia_template_v0_1.md`)

**ISO 26262-8 §5 + ISO 26262-10 §6.** Required reviewer independence: **I2 (safety)** + **legal counsel** (separate review).

| # | Check | Y/N/NA | Evidence |
|---:|---|:---:|---|
| DIA1 | All `[PLACEHOLDER]` fields are documented (i.e., the placeholder is intentional, not a missed field) | | |
| DIA2 | §3 lifecycle phase allocation table covers every ISO 26262 phase | | |
| DIA3 | §4 + §5 work products are bidirectional (S → L and L → S) | | |
| DIA4 | §8 change-management classes (A/B/C/D) have explicit SLA per class | | |
| DIA5 | §9 anomaly handling cross-references the field-monitoring template | | |
| DIA6 | §11 long-tail safety obligations match ISO 26262-7 §6 (15-year retention) | | |
| DIA7 | §12 risk acknowledgement matches the current Safety Case §6 risk register | | |
| DIA8 | §13 amendment triggers cover all safety-relevant change scenarios | | |

### 3.8 FMEDA reports (`docs/safety/fmeda/*.md`)

**ISO 26262-5 §7.4.5 + §8.4.** Required reviewer independence: **I3 for ASIL-D-relevant modules**, **I2 for others**.

| # | Check | Y/N/NA | Evidence |
|---:|---|:---:|---|
| FM1 | Every failure mode in `failure_modes.yaml` for the module is included | | |
| FM2 | Per-mode classification (safe / dangerous / no-effect) has explicit rationale | | |
| FM3 | Mechanism IDs cited exist in `safety_mechanisms.yaml` | | |
| FM4 | SPFM / LFM / PMHF formulas match ISO 26262-5 §8.4 | | |
| FM5 | ASIL target metrics (Table 4 of Annex C) used correctly | | |
| FM6 | Findings + closure options (§4) reference real F4 WPs (not invented) | | |
| FM7 | Failure-rate baselines documented as placeholders + sourced (IEC 62380 / SN29500 / Robinson 2017 / etc.) | | |
| FM8 | For aggregate FMEDA: per-module breakdown table + improvement trajectory cited | | |

### 3.9 CCF analysis (`docs/safety/ccf_analysis_v0_1.md`)

**ISO 26262-9 §7.** Required reviewer independence: **I3**.

| # | Check | Y/N/NA | Evidence |
|---:|---|:---:|---|
| CCF1 | All 8 CCF initiator categories (§1.1) are evaluated for each redundant element | | |
| CCF2 | β-factor estimates within industry consensus ranges (§1.3) | | |
| CCF3 | Mitigations cite F4 WPs that close them | | |
| CCF4 | Voter/ECC common-mode failure analysis included (not just lane CCF) | | |
| CCF5 | Licensee-recommended patterns (lockstep, dual-channel decomposition) consistent with FSC §1.1 | | |
| CCF6 | CCF detection mechanisms (§4) cross-reference fault_detected[] bits | | |
| CCF7 | Verification approach (§5) cites Phase C campaigns + formal proofs | | |

### 3.10 SER analysis (`docs/safety/ser_analysis_v0_1.md`)

**ISO 26262-11 §7 + JEDEC JESD89A.** Required reviewer independence: **I3**.

| # | Check | Y/N/NA | Evidence |
|---:|---|:---:|---|
| SER1 | Per-element baselines cite primary literature (Robinson 2017, Calivá 2019) | | |
| SER2 | Operating-environment assumption (§1.2) matches HARA §3.1 OS-1.A | | |
| SER3 | Per-module FF + SRAM census (§2) consistent with `failure_modes.yaml` populations | | |
| SER4 | Aggregate PMHF cross-validates against the FMEDA aggregate roll-up (§3.1 vs FMEDA aggregate) | | |
| SER5 | Mitigation effectiveness factors (700× ECC, 10× scrub, 10× DICE) cite literature | | |
| SER6 | PMHF derivation against ASIL-B (≤100) and ASIL-D (≤10) shown explicitly | | |
| SER7 | Improvement trajectory (§3.3) consistent with remediation plan F4-A/B/C/D phases | | |

### 3.11 Safety Case master (`docs/safety/safety_case_v0_1.md`)

**ISO 26262-8 §6.** Required reviewer independence: **I3 + functional safety assessor for v1.0 release**.

| # | Check | Y/N/NA | Evidence |
|---:|---|:---:|---|
| SC1 | Headline claim (§1.1) does not over-claim certified status | | |
| SC2 | "What this claim does NOT say" (§1.2) explicitly disclaims certification + item-level + Licensee-HARA-substitute | | |
| SC3 | Verifiable-evidence table (§1.3) reproduces on a fresh checkout | | |
| SC4 | Per-SG argument (§3) covers every Safety Goal in HARA | | |
| SC5 | Each sub-goal cites TSR + mechanism + FMEDA + IT (4-leaf evidence) | | |
| SC6 | Risk register (§6) categorised Critical / High / Medium / Low with owner + week | | |
| SC7 | Confirmation review status table (§7.1) matches the actual state of named reviewers | | |
| SC8 | ISO 26262 conformance summary (§8) per Part with status icon | | |
| SC9 | v1.0 release criteria (§10) are objective + measurable | | |
| SC10 | Aggregate ASIL achievability (§1.4 / §4) consistent with FMEDA aggregate + remediation plan trajectory | | |

### 3.12 Findings Remediation Plan (`docs/safety/findings_remediation_plan_v0_1.md`)

**Internal planning artefact.** Required reviewer independence: **I3 (since it drives ASIL-D closure)**.

| # | Check | Y/N/NA | Evidence |
|---:|---|:---:|---|
| RP1 | Every finding F1-F7 + H1-H9 from buyer DD has a remediation WP or explicit deferral | | |
| RP2 | F4-A WPs sum to ~14-15 days; F4-B WPs sum to ~31-36 days; numbers in §3.x match | | |
| RP3 | New WPs added during execution (F4-A-7, F4-B-7, F4-C-5, F4-D-6/7/8, F4-IT-1..9) are listed in the right phase table | | |
| RP4 | Each WP cites the gap it closes | | |
| RP5 | Expected SPFM trajectory (§4) consistent with FMEDA aggregate trajectory | | |
| RP6 | Open questions for founder (§7) listed |

### 3.13 Field-monitoring template (`docs/safety/field_monitoring_template_v0_1.md`)

**ISO 26262-7 §6 + ISO 26262-8 §6.4.7.** Required reviewer independence: **I2**.

| # | Check | Y/N/NA | Evidence |
|---:|---|:---:|---|
| FM-T1 | Schema (§1.1) covers all DIA §9.1 reporting requirements | | |
| FM-T2 | Severity SLA (§1.2) matches DIA §9.2 | | |
| FM-T3 | Access roles (§5.2) consistent with DIA §10.3 right-to-inspect | | |
| FM-T4 | Cross-Licensee anonymisation (§5.3) defined | | |
| FM-T5 | Safety-case re-review (§8) and safety-advisory (§9) processes have explicit timelines | | |
| FM-T6 | Example records use real RTL paths and real failure-mode IDs | | |

### 3.14 Spec sheet rev 1.4 (`docs/spec_sheet_rev_1_4.md`)

**External claim surface.** Required reviewer independence: **I3 + product / marketing review**.

| # | Check | Y/N/NA | Evidence |
|---:|---|:---:|---|
| SS1 | IP-datasheet framing consistent — every silicon-only item is in "licensee-supplied" bucket | | |
| SS2 | ASIL claims cite HARA, not just "ASIL-D" | | |
| SS3 | §10 Validation snapshot numbers reproduce on `pytest --no-header` + `bash tools/run_verilator_*.sh` | | |
| SS4 | §12 "what is NOT in this datasheet" matches DIA §12 risk acknowledgement | | |
| SS5 | Roadmap WPs (§11) cite the right F4 IDs | | |
| SS6 | Numbers consistent with FMEDA + SER + CCF (e.g., npu_top SPFM 2.08 % matches FMEDA report) | | |
| SS7 | Reworded ⚠ claims from rev 1.3 do not silently slip back in | | |

---

## 4. Review process

### 4.1 Trigger

A confirmation review is triggered when a work product moves from v0.x → v1.0 (per the work product's own §13.x revision triggers). Within-version revisions (v0.1 → v0.2 → ... → v0.9) do not require full confirmation reviews; they may have abbreviated reviews per §4.4 below.

### 4.2 Review session

1. Reviewer receives the work product + this checklist + companion documents
2. Reviewer works through the §2 common-cause checks then the relevant §3 per-doc checklist
3. Each Y/N/NA + evidence reference recorded in a per-review record (template at Appendix A)
4. Findings classified as: **MUST FIX** (blocks v1.0), **SHOULD FIX** (logged for next revision), **OBSERVATION** (non-binding)
5. Reviewer drafts written opinion within 10 business days of receiving the package

### 4.3 Sign-off

Work product moves to v1.0 when:
- All MUST FIX items resolved
- Independent reviewer signs the per-review record
- Safety Manager countersigns

### 4.4 Within-version refresh

For minor revisions (v0.x → v0.x+1) that touch < 20% of the document, an abbreviated review may be conducted: §2 common-cause checks (mandatory) + targeted §3 checks for the changed sections only. Full per-doc checklist reapplies at v1.0.

### 4.5 Escalation

If reviewer and author disagree on a MUST FIX classification:
- Try to resolve at the safety-manager level within 5 business days
- If unresolved, founder makes the final call (Safety Manager has veto on any item that compromises the safety case)
- If founder + Safety Manager disagree, halt the work product and engage a second independent reviewer for arbitration

---

## 5. Evidence retention

Per ISO 26262-8 §10, all confirmation review records retained for **15 years post-product-discontinuation**. Records include:

- The completed checklist (Y/N/NA + evidence references)
- Reviewer's written opinion (with MUST FIX / SHOULD FIX / OBSERVATION classification)
- Author's response to findings
- Safety Manager countersignature
- Any escalation correspondence

Per-review records are stored at `docs/safety/reviews/<doc-id>-<rev>-review.md` (directory created at first executed review).

---

## 6. Open items for v0.2

1. **Named independent reviewer** for each work product per §1.2 (currently TBD — TÜV SÜD India proposed default for I3)
2. **TÜV SÜD India pre-engagement workshop** scheduled W6 — workshop output expected to refine §3 checklists
3. **Per-review record template** (Appendix A) to be expanded with TÜV-recommended fields
4. **Within-version refresh threshold** — currently "20% of document"; v0.2 may refine based on actual experience
5. **Tool support** — checklist could be a YAML / JSON form for tooling; currently markdown for human consumption

---

## 7. Document control

| Field | Value |
|---|---|
| Document ID | ASTR-SAFETY-CONF-REVIEW-V0.1 |
| Revision | 0.1 |
| Revision date | 2026-04-20 |
| Author | TBD (Track 2 lead) — currently founder + collaborator |
| Reviewer of THIS doc | TBD (the same reviewer who uses the checklist on other docs — circular but per ISO 26262-2 §7 process artefacts get an I2 review by default) |
| Approver | TBD (Safety Manager) |
| Distribution | Internal + named independent reviewers + TÜV SÜD India |
| Retention | 15 years post-product-discontinuation per ISO 26262-8 §10 |
| Supersedes | None — this is v0.1; closes ISO 26262 gap analysis Part 2 §7 |

### 7.1 Revision triggers

This checklist is re-issued (with revision bump) on any of:

1. New Track 2 work product added → spawn a §3 per-doc checklist
2. ISO 26262 part edition update
3. Confirmation-review feedback that adds a previously-missed check
4. New common-pitfall identified during a review → §2 / §3 update
5. Reviewer independence requirements change per ISO 26262-2 §7 amendment

---

## Appendix A — Per-review record template

```
# Confirmation review record — [Work Product Name]

Document under review: [doc ID + rev]
Reviewer: [name + organisation + independence level (I1/I2/I3)]
Review date: [YYYY-MM-DD]
Review session duration: [hours]
Companion documents reviewed: [list]

## §2 Common-cause checklist results

| # | Check | Result | Evidence |
|---|---|---|---|
| C1 | ... | YES / NO / N/A | [file:line] |
| ... |

## §3.X Per-doc checklist results

| # | Check | Result | Evidence |
|---|---|---|---|
| H1 / F1 / T1 / ... | ... | YES / NO / N/A | [file:line] |
| ... |

## Findings

| # | Classification | Description | Recommended action |
|---|---|---|---|
| 1 | MUST FIX | [...] | [...] |
| 2 | SHOULD FIX | [...] | [...] |
| 3 | OBSERVATION | [...] | [...] |

## Reviewer's written opinion

[free-text overall assessment]

## Sign-off

Reviewer:                        Date: [YYYY-MM-DD]   Signature: ____________________

Safety Manager (countersign):    Date: [YYYY-MM-DD]   Signature: ____________________
```
