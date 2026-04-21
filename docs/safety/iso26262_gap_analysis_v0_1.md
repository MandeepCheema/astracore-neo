# ISO 26262 Gap Analysis — AstraCore Neo IP

**Document ID:** ASTR-SAFETY-GAP-V0.1
**Date:** 2026-04-19
**Classification:** Internal — pre-engagement draft for TÜV SÜD India workshop
**Scope:** AstraCore Neo NPU IP + sensor-fusion IP block, as licensed (not the licensee's full SoC).
**Target item definition:** *AI inference accelerator IP block providing perception, sensor-fusion, and decision-support for automotive E/E architectures, integrated by the SoC licensee.*
**Initial ASIL target:** ASIL-B (Q4 2026). ASIL-D extension scoped separately.
**Companion documents:**
- `docs/safety/seooc_declaration_v0_1.md` — formal SEooC declaration (the contractual element that this gap analysis informs)
- `docs/safety/safety_manual_v0_1.md` — Safety Manual outline (licensee user-guide companion to the SEooC)
- `docs/safety/fmeda/`, `docs/safety/fault_injection/`, `docs/safety/tcl/` — directories where the quantitative artefacts referenced in §1 land
- `docs/best_in_class_design.md` §7 — strategic context (Path A IP licensing + ≤10% spec variance + safety-cert lead direction)

> **How to read this doc.** Each ISO 26262 part lists (a) what we have today (verifiable in repo), (b) the gap, (c) the work package + owner + week to close it. We separate **process** (organizational) from **technical** (RTL/tooling) artefacts because they have different bottlenecks: process work is calendar-bound, technical work is engineering-bound.

---

## 0. Boundary statement (read this first)

AstraCore Neo is licensed as **IP**, not as a complete SoC. The safety case has two layers:

| Layer | Owned by | Cert artefact |
|---|---|---|
| **IP block safety case** | AstraCore | This document + Safety Manual + FMEDA + fault-injection campaign + safety mechanism descriptions |
| **Item-level safety case** (vehicle / SoC) | Licensee | Built on top of our IP-block safety case + their PHY/memory/package + their item definition |

Per ISO 26262-10 §6.1, an IP supplier delivers a **Development Interface Agreement (DIA)** plus a **Safety Manual** to the licensee. The licensee is responsible for the item-level claim. **Our job is to make the IP-block safety case rigorous enough that an ASIL-B (or ASIL-D) item-level claim is achievable on top of it.**

---

## 1. Part-by-part gap matrix

### Part 2 — Functional Safety Management

| Requirement | Current state | Gap | WP / Wk |
|---|---|---|---|
| §5: Safety culture, training, lifecycle definition | None documented | Need org-level safety policy + training records | T2-W1 |
| §6: Functional Safety Management for development | None | DIA template + Safety Manager role assignment | T2-W2 |
| §7: Confirmation measures (review, audit, assessment) | ✅ Confirmation review checklist v0.1 (2026-04-20) at `docs/safety/confirmation_review_checklist_v0_1.md` — per-doc checklists for 14 Track 2 work products + reviewer-independence matrix per ASIL + review process + escalation. Named reviewer + TÜV SÜD pre-engagement remain as W6 organisational items | TÜV SÜD India workshop output may refine checklists | T2-W6 ✅ |
| Safety culture artefact | None | One-page policy signed by founder | T2-W1 |

**Gap severity:** HIGH — but cheap to close. Most of Part 2 is paperwork that takes a week of focused founder + functional-safety-engineer time.

### Part 3 — Concept Phase

| Requirement | Current state | Gap | WP / Wk |
|---|---|---|---|
| §5: Item definition | Implicit (in `docs/architecture.md`) | Formal item definition document for IP-block boundary | T2-W1 |
| §6: HARA (Hazard Analysis & Risk Assessment) | ✅ v0.1 (2026-04-20, `docs/safety/hara_v0_1.md`) on three reference use cases: FCW/AEB/LKA, DMS, surround/parking | v0.2 needs: named reviewer + workshop with first NDA licensee + quantitative FTTI numbers | T2-W3 ✅ |
| §7: Functional safety concept | ✅ partial — eleven safety goals derived in HARA §2.4/§3.4/§4.4 | Full functional-safety-concept document linking SGs to architectural mitigations | T2-W4 |
| §8: ASIL determination | ✅ derived per HARA §1.4 + §5: UC-1 ASIL-D, UC-2 ASIL-C, UC-3 ASIL-C; aggregate item-level ASIL-D driven by SG-1.1. Spec sheet rev 1.4 wording proposed in HARA §5.2 | Confirmation review per ISO 26262-2 §7 mandatory for ASIL-C/D | T2-W4 ✅ |

**Gap severity:** HIGH. The "ASIL-D" claim on the spec sheet has no formal HARA backing. This is the single biggest credibility gap for any safety-aware OEM evaluator. **Must close by W4.**

### Part 4 — Product Development at the System Level

| Requirement | Current state | Gap | WP / Wk |
|---|---|---|---|
| §6: Technical safety concept | ✅ FSC v0.1 + **TSC v0.1 (2026-04-20)** at `docs/safety/technical_safety_concept_v0_1.md` — 38 TSRs derived from 25 FSRs, allocated to specific RTL modules / SDK components / HW-SW interfaces | TSC v0.2 closes BLOCKED TSRs as F4 phases land; quantify thresholds against measured numbers | T2-W12-W14 ✅ (early) |
| §7: System architectural design | `docs/astracore_v2_npu_architecture.md` exists; **FSC §5 indexes safety-relevant subset and the SEooC §2.3 boundary signal contract** | Full safety-annotated architecture diagram (TSC scope) | T2-W12-W14 |
| §8: Hardware-software interface | Partial in `docs/hal.md`; **SEooC §2.3 enumerates 10 safety-relevant boundary signals** (safe_state_active, fault_detected[15:0], external_safe_state_request, lockstep_compare_in, tmr_disagree_count, ECC counters, watchdog_kick, dft_isolation_enable, clock_monitor_alert) | Quantitative timing (latency, polarity) per signal in Safety Manual §4-§7 (W4-W6) | T2-W4-W6 |
| §9: System integration & testing | `tests/` covers functional + 208 safety-tooling tests; **FSC §3 verification column names integration tests required per FSR** | Per-FSR integration test plan companion document | T2-W8 |

### Part 5 — Product Development at the Hardware Level (THE BIG ONE)

| Requirement | Current state | Gap | WP / Wk |
|---|---|---|---|
| §7.4: Hardware safety requirements specification | None | Per-module HW safety requirements doc | T2-W3-7 |
| §7.4.5: Quantitative analysis (FMEDA) | None | FMEDA tool + per-module FMEDA. **Cover ≥90% of die area for ASIL-B.** | T2-W2-10 |
| §8.4.1: Diagnostic Coverage (DC) per failure mode | None | Per-mechanism DC calculation | T2-W4-10 |
| §8.4.2: Single-Point Fault Metric (SPFM) | None | Aggregate SPFM ≥ 90% (ASIL-B), ≥ 99% (ASIL-D) | T2-W10-12 |
| §8.4.3: Latent-Fault Metric (LFM) | None | Aggregate LFM ≥ 60% (ASIL-B), ≥ 90% (ASIL-D) | T2-W10-12 |
| §8.4.4: Probabilistic Metric for random HW Failures (PMHF) | None | Estimate PMHF ≤ 100 FIT (ASIL-B), ≤ 10 FIT (ASIL-D) | T2-W12 |
| §10: Verification of safety analyses | None | Formal proofs (Symbiyosys/Yosys-SBY) on TMR + ECC | T2-W7-9 |
| §11: Hardware integration & testing | None | Fault-injection campaign (cocotb harness) | T2-W3-9 |
| Annex D: Common Cause Failure (CCF) | None | CCF analysis for redundant elements (TMR lanes, dual-rail interfaces) | T2-W14-15 |

**Gap severity:** CRITICAL but tractable. This is the bulk of Track 2 work. The good news: AstraCore *has* the safety mechanisms (TMR, ECC, plausibility, safe-state, fault predictor) — what's missing is the **quantitative analysis** of how well they cover failure modes. That's tooling work, not RTL work.

#### What AstraCore has today that feeds Part 5

| Mechanism | RTL location | Coverage claim (to be quantified) |
|---|---|---|
| TMR voter (3-of-3 majority) | `rtl/tmr_voter/tmr_voter.v` | High DC for stuck-at + transient on critical paths; used in `dms_fusion`, `safe_state_controller` |
| ECC SECDED (72,64) | `rtl/ecc_secded/ecc_secded.v` | Single-bit correct, double-bit detect on SRAM |
| Safe-state controller | `rtl/safe_state_controller/safe_state_controller.v` | Drives chip to defined safe state on aggregated fault |
| Plausibility checker | `rtl/plausibility_checker/plausibility_checker.v` | Range/rate/cross-sensor consistency on sensor inputs |
| Fault predictor | `rtl/fault_predictor/fault_predictor.v` | Pattern-based prediction of incipient faults (currently rule-based; ML extension is F1-A9) |
| Watchdog (in `dms_fusion`) | `rtl/dms_fusion/dms_fusion.v` (parameter `WATCHDOG_CYCLES = 200ms @ 50MHz`) | Sensor-stuck detection + SENSOR_FAIL state |
| Clock monitor | implicit in `rtl/thermal_zone/` + Python `src/safety/clock_monitor.py` | Clock loss / glitch / freq-out-of-bounds — **needs RTL counterpart for ASIL-B** |
| TRNG | None today | OpenTitan PTRNG/CSRNG integration in Track 3 |

### Part 6 — Product Development at the Software Level

| Requirement | Current state | Gap | WP / Wk |
|---|---|---|---|
| §7: Specification of software safety requirements | Implicit | Formal SW safety requirement spec for `tools/`+`astracore/` | T2-W4 |
| §8.4.5: MISRA-C compliance (for production runtime) | None | Apply MISRA-C / MISRA-Python style rules in CI; ratchet up | T2-W3 onwards |
| §8.4.5: Coding guidelines | None | `docs/coding_guidelines.md` covering Python + SystemVerilog (RTL) | T2-W2 |
| §9: Unit design & implementation | 1140 tests collected | Map tests to safety requirements (traceability matrix) | T2-W6 |
| §10: Unit verification | pytest passes | Add coverage report (coverage.py); target ≥ 90% on safety-critical modules | T2-W4 |
| §11: SW integration & testing | Some integration tests | Document integration test plan tied to safety requirements | T2-W8 |

**Gap severity:** MEDIUM. Python-side compiler/runtime is "qualified for use" rather than "developed per ISO 26262" — both are valid paths under §11.4.9 ("software developed in compliance with another standard"). For Phase A (Python compiler offline) this is acceptable; the **C++ runtime that runs on the target SoC must be developed to ISO 26262 from the start.** Defer C++ runtime safety coding rules until Track 1 starts that work.

### Part 7 — Production and Operation

| Requirement | Current state | Gap | WP / Wk |
|---|---|---|---|
| §5: Production planning | N/A pre-tape-out | Licensee responsibility |  |
| §6: Operation, service, decommissioning | N/A pre-tape-out | Licensee responsibility |  |

**Gap severity:** LOW (out of scope for IP block).

### Part 8 — Supporting Processes

| Requirement | Current state | Gap | WP / Wk |
|---|---|---|---|
| §5: Interface within distributed developments → **DIA** | ✅ template v0.1 (2026-04-20) at `docs/safety/dia_template_v0_1.md`. Licensee-specific instantiation drives a fresh execution. | Per-engagement instantiation when first NDA licensee signs | T2-W6 ✅ |
| §6: Safety case management | ✅ Safety Case master doc v0.1 (2026-04-20) at `docs/safety/safety_case_v0_1.md`. Per-SG argument structure, aggregate ASIL achievability table, cross-cutting evidence, 22-row risk register | v1.0 release per §10 criteria (after W12 Phase A+B+C close) | T2-W11 ✅ (early) |
| §7: Configuration management | Git is in use | Document branching/release/baseline policy | T2-W3 |
| §8: Change management | Ad hoc | Document change-impact analysis procedure | T2-W3 |
| §9: Verification | Per-section above | — |  |
| §10: Documentation management | Markdown in repo | Document control policy (review, approval, retention) | T2-W4 |
| §11: Confidence in use of software tools (TCL) | ✅ v0.1 (2026-04-20) at `docs/safety/tcl/tcl_evaluations_v0_1.md`; 9 tools classified (Verilator/cocotb/Yosys/SBY/OpenROAD/ASAP7/pytest/numpy/onnx); 3 require formal qualification (Yosys=TCL2, OpenROAD=TCL2, ASAP7=TCL3) — qualification WPs F4-B-7, F4-D-7, F4-D-8 | Pin tool versions before TÜV workshop; complete qualification WPs | T2-W7 ✅ |
| §12: Qualification of software components | None | Qualify ONNX / ONNX Runtime / numpy / pytorch as off-the-shelf SW components per §12 | T2-W9 |
| §13: Qualification of hardware components | None | Qualify OpenTitan crypto blocks per §13 | T2-W9 |
| §14: Proven in use argument | N/A new IP | — |  |

**Gap severity:** HIGH for §11 (tool qualification). Verilator, Yosys, etc. all need a TCL-rated argument or compensating verification. This is a checklist exercise, not an engineering one — but it's mandatory.

### Part 9 — ASIL- and Safety-oriented Analyses

| Requirement | Current state | Gap | WP / Wk |
|---|---|---|---|
| §5: Decomposition of safety requirements | None | Apply where TMR / ECC justify ASIL decomposition | T2-W7 |
| §6: Criticality analysis | None | Per-module criticality rating | T2-W5 |
| §7: Analysis of dependent failures | ✅ CCF analysis v0.1 (2026-04-20) at `docs/safety/ccf_analysis_v0_1.md` — 8 CCF initiator categories, per-redundant-element β analysis (TMR voter, ECC SECDED, lockstep, dual-channel decomposition), Phase D mitigations for ASIL-D | Quantitative β measurement via Phase C multi-fault injection extension; cascading failure analysis (FTA) v0.2 | T2-W14-15 ✅ (early) |
| §8: Safety analysis (FMEA, FTA) | None | FMEA per top-level safety mechanism; FTA for top hazards | T2-W4-8 |

### Part 11 — Guidelines on application to semiconductors (CRITICAL FOR US)

This part is **specifically for IP suppliers and semiconductor vendors**. It is the most important section for AstraCore.

| Requirement | Current state | Gap | WP / Wk |
|---|---|---|---|
| §4.6: IP supplier responsibilities | Implicit | Formal Safety Element out of Context (SEooC) declaration | T2-W2 |
| §4.7: Assumptions on the use of IP | None | Assumptions of Use document (what the licensee must do) | T2-W6 |
| §4.8: Safety analysis of digital components | None | Covered by Part 5 FMEDA |  |
| §4.9: Safety analysis of analog components | N/A (no analog in our IP) | — |  |
| §5: Hardware integration verification | None | IP integration test suite + safety test patterns | T2-W8 |
| §6: Software tools used to design HW | None | TCL evaluation (overlaps Part 8 §11) | T2-W7 |
| §7: Soft errors (alpha, neutron, transient) | ✅ SER analysis v0.1 (2026-04-20) at `docs/safety/ser_analysis_v0_1.md`. Per-element 7nm baselines (5e-4 FIT/FF; 7e-3 FIT/bit SRAM); per-module FF+SRAM census; aggregate raw SER (~73.7 FIT today 4×4) → Phase A residual (~1.34 FIT) → full-spec Phase D residual (~6.1 FIT) ✅ within ASIL-D 10 FIT cap | Empirical irradiation measurement deferred to Licensee post-tape-out per JEDEC JESD89A | T2-W10 ✅ |

**SEooC declaration (Part 11 §4.6) is the single most important Track 2 W2 deliverable.** It defines us as a "Safety Element out of Context" — i.e., IP designed for ASIL-X without knowing the specific item — and makes our Safety Manual the contractual interface to the licensee.

---

## 2. Prioritized close-out (the order to actually do these)

Sorted by leverage on the W12 ASIL-B target and W20 ASIL-D safety-case-document target.

| # | Gap | Why first | Wk |
|---:|---|---|---:|
| 1 | Item definition + IP-block boundary statement (Part 3 §5; Part 11 §4.6) | Everything downstream traces here | 1 |
| 2 | SEooC declaration (Part 11 §4.6) | Defines our deliverable contract with licensees | 2 |
| 3 | HARA + ASIL derivation on 3 reference use cases (Part 3 §6, §8) | The "ASIL-D" claim has no current backing — fix this first | ✅ 2026-04-20 — `docs/safety/hara_v0_1.md` v0.1 |
| 4 | FMEDA tool (Python) + first FMEDA on `dms_fusion` (Part 5 §7.4.5) | Quantitative analysis is the bulk of Part 5; tool first, modules in series | 1-3 |
| 5 | Safety Manual v0.1 (Part 11 §4.7) | First doc TÜV will ask for at the W6 workshop | 1-2 |
| 6 | Cocotb fault-injection harness + first 1k-fault campaign on `tmr_voter` (Part 5 §11) | Defensible DC numbers > spec sheet promises | 3-4 |
| 7 | Coding guidelines + MISRA gates in CI (Part 6 §8.4.5; Part 11 §6) | Cheap to start, gets harder if deferred | 2-3 |
| 8 | Tool Confidence Level evaluation (Part 8 §11) | Required artefact; checklist work | 7 |
| 9 | DC/LFM/SPFM aggregate report v0.1 (Part 5 §8.4) | The "real" ASIL-B numbers | 10 |
| 10 | TÜV SÜD pre-assessment workshop | External validation of the above | 6 |
| 11 | Safety case document v0.1 (Part 8 §6) | Pulls everything together | 11 |
| 12 | ASIL-B safety case v1.0 + TÜV interim review | W12 milestone | 12 |
| 13 | CCF analysis (Part 9 §7) | Required for ASIL-D | 14-15 |
| 14 | Soft Error Rate analysis (Part 11 §7) | Required for ASIL-D semiconductor | 10 |
| 15 | ASIL-D safety case document v1.0 | W20 milestone (out of scope this 16-wk plan; placeholder) | 20 |

---

## 3. What this means for the spec sheet

The spec sheet rev 1.3 currently says:
> **ISO 26262 ASIL-D, certified**

This is **not defensible today.** Mechanisms exist (TMR, ECC, etc.) but:

1. No HARA → no derivation of ASIL-D
2. No FMEDA → no DC / LFM / SPFM numbers
3. No fault-injection campaign → no coverage evidence
4. No safety case document
5. No external assessment

**Recommended spec sheet rev 1.4 wording:**

> **ISO 26262 — Designed for ASIL-D as Safety Element out of Context (SEooC).**
> Safety mechanisms include TMR voting, SECDED ECC, watchdog, plausibility checking, and safe-state controller. ASIL-B safety case targeted Q4 2026 with TÜV SÜD India pre-engagement underway. ASIL-D safety case extension and licensee-side certification are in scope as documented in the IP Safety Manual and Development Interface Agreement.

This is **honest** (no false certification claim) and **strong** (designed-for + active TÜV engagement + named ASIL-B timeline). It survives DD better than the current text.

---

## 4. Reference open-source tools to use

| Need | Tool | License | Why this one |
|---|---|---|---|
| FMEDA spreadsheet/tool | Custom Python (build) | Apache-2.0 (ours) | Off-the-shelf FMEDA tools (Plato, medini, ANSYS) are commercial and overkill for IP-block scope. A Python tool that reads RTL module list + failure-mode YAML + mechanism YAML and outputs DC/LFM/SPFM is ~500 LoC. |
| Fault injection | cocotb + custom harness | BSD-3 (cocotb) | Already in our stack. Stuck-at + bit-flip + transient injection via `force`/`release`. |
| Formal verification | Yosys + Symbiyosys (SBY) | ISC (Yosys), MIT (SBY) | Open-source formal flow. Sufficient for TMR + ECC properties. Commercial (Cadence Jasper, Synopsys VC Formal) at $200k+/seat is overkill at this stage. |
| Crypto IP | OpenTitan | Apache-2.0 | Production-grade, used by Google. Avoids per-IP-license cost. |
| Coverage | Verilator coverage + Python coverage.py | LGPL / Apache-2.0 | Already in stack. |
| Soft Error Rate | NASA SAVR + Cypress / Synopsys SER calculators | Mixed | No fully-OSS SER tool; use vendor calculators with documented assumptions. |

---

## 5. Open questions for TÜV SÜD pre-assessment workshop (W6 input)

1. Is our SEooC declaration scope (NPU IP + sensor-fusion IP block) acceptable as a single safety element, or must they be split?
2. For ASIL-B initial cert, will TÜV accept open-source tool flow (Verilator, Yosys, OpenROAD) with documented TCL, or do they require commercial EDA evidence?
3. For the licensee's PHY/memory/package, what assumptions of use must we put in the Safety Manual to avoid scoping-in licensee silicon into our cert?
4. Is OpenTitan crypto IP acceptable as "qualified hardware component" per Part 8 §13, given its production use in Google Titan chips?
5. What's the realistic calendar for ASIL-B cert mark assuming W12 safety-case-v1.0 → W14 TÜV audit → W26 cert mark?

---

## 6. Author's note (Track 2 lead)

The state of the art today: AstraCore *has the mechanisms* but *does not have the analysis*. This document, the Safety Manual, the FMEDA, and the fault-injection campaign close the analysis gap. None of this requires new RTL — it requires roughly 12 weeks of disciplined functional-safety engineering work plus calendar time for TÜV engagement.

The credibility unlock is real: an OEM evaluator who reads the rev 1.3 spec sheet (which says "ASIL-D, certified") and then asks for the safety case will today find nothing. The same evaluator at W12 will find: SEooC declaration, HARA, FMEDA with quantified DC/LFM/SPFM, Safety Manual v1.0, fault-injection campaign coverage report, formal proofs of TMR+ECC, and a TÜV-validated path. **That is the difference between losing the deal and starting a paid IP evaluation.**

— Document version v0.1. Next revision after W2 SEooC declaration is signed off.
