# Aggregate FMEDA Roll-up — Executive Summary

**Document ID:** ASTR-FMEDA-AGG-SUMMARY-V0.1
**Date:** 2026-04-20
**Standard:** ISO 26262-5:2018 §8.4 (HW architectural metrics) — aggregate computation
**Element:** AstraCore Neo NPU + Sensor-Fusion IP block (SEooC) — all 6 catalogued modules
**Status:** v0.1 — first formal release. Closes risk R12 in `docs/safety/safety_case_v0_1.md` §6.2 + W10 deliverable in `docs/safety/findings_remediation_plan_v0_1.md` §3.3.
**Companion:** `docs/safety/fmeda/aggregate_fmeda_v0_1.md` — auto-generated full per-mode detail across all 60+ failure modes from all modules.

---

## 0. Purpose

The aggregate FMEDA combines per-module ISO 26262-5 §7.4.5 analyses into the single ASIL claim that licensees and assessors care about: **does the IP block as a whole meet the SPFM / LFM / PMHF thresholds for the target ASIL?**

This document produces the executive summary; the per-mode detail is in the auto-generated companion at `aggregate_fmeda_v0_1.md`. Both are regenerated on every change to `failure_modes.yaml` or `safety_mechanisms.yaml`.

### 0.1 Methodology

**v0.1 uses straight summation** across the 6 catalogued modules (per `tools/safety/fmeda.py:compute_aggregate_fmeda`):

```
λ_total_agg = Σ λ_total_per_module
λ_dangerous_agg = Σ λ_dangerous_per_module
λ_DD_agg = Σ λ_DD_per_module
λ_DU_agg = Σ λ_DU_per_module
λ_LF_agg = Σ λ_LF_per_module
SPFM_agg = 1 - λ_DU_agg / λ_dangerous_agg
LFM_agg  = 1 - λ_LF_agg / (λ_dangerous_agg - λ_DU_agg)
PMHF_agg = λ_DU_agg + 0.5 × λ_DPF_agg
```

**v0.2 adds cross-module re-crediting** — some module-level λ_DU contributions could be re-classified as aggregate-level λ_DD when `safe_state_controller` catches them at item scope. The current FMEDA already credits this via the `aggregate_safe_state_fault_out` mechanism (target DC 40 %); v0.2 will accept measured per-row credit values from Phase C fault-injection campaigns.

---

## 1. Per-module breakdown (regenerate from `tools/safety/fmeda.py`)

| Module | λ_total (FIT) | λ_DU (FIT) | SPFM | LFM | PMHF (FIT) | ASIL-B pass? |
|---|---:|---:|---:|---:|---:|:---:|
| `dms_fusion` | 0.0517 | 0.0074 | 85.52 % | 92.99 % | 0.008 | ⚠ SPFM short by 4.48 pp |
| `ecc_secded` | 0.0341 | 0.0336 | **1.49 %** | 0 % | 0.034 | ❌ |
| `npu_top` (4×4 default) | 0.5414 | 0.5302 | **2.08 %** | 0 % | 0.530 | ❌ |
| `plausibility_checker` | 0.0211 | 0.0055 | 27.20 % | 0 % | 0.005 | ❌ |
| `safe_state_controller` | 0.0127 | 0.0083 | 34.80 % | 0 % | 0.008 | ❌ |
| `tmr_voter` | 0.0197 | 0.0135 | 31.35 % | 0 % | 0.013 | ❌ |
| **AGGREGATE** | **0.6807** | **0.5984** | **10.23 %** | **0.00 %** | **0.5984** | ❌ |

**Single dominant contributor:** `npu_top` accounts for 80 % of λ_total and 89 % of λ_DU. Closing the npu_top SPFM gap (Phase A + B per `findings_remediation_plan_v0_1.md`) is therefore the single highest-leverage path to lift the aggregate.

---

## 2. ASIL achievability vs targets

Per ISO 26262-5 Annex C:

| Tier | SPFM target | LFM target | PMHF target | Aggregate today | Pass? |
|---|---:|---:|---:|---|:---:|
| ASIL-B | ≥ 90 % | ≥ 60 % | ≤ 100 FIT | SPFM 10.23 % / LFM 0 % / PMHF 0.60 FIT | ❌ SPFM + LFM short; **PMHF passes** |
| ASIL-C | ≥ 97 % | ≥ 80 % | ≤ 100 FIT | same | ❌ |
| ASIL-D | ≥ 99 % | ≥ 90 % | ≤ 10 FIT | same | ❌ |

**Key observation:** **PMHF is already within both ASIL-B (≤ 100 FIT) and ASIL-D (≤ 10 FIT) caps** today (0.60 FIT << 10 FIT). The blocker is **SPFM** — uncovered single-point faults dominate. SPFM closure does not require new RTL beyond F4-A Phase A; it requires the existing safety-mechanism RTL (ECC, parity, TMR) to be **wired in** at npu_top scope (F4-A-1.1, F4-A-2/3/4/6, F4-A-7).

---

## 3. Improvement trajectory (per remediation plan)

| Milestone | Aggregate SPFM | Aggregate LFM | Aggregate PMHF | ASIL pass |
|---|---:|---:|---:|:---:|
| Today | 10.23 % | 0 % | 0.60 FIT | ❌ all |
| Phase A complete (W4) — F4-A-1.1 ECC + F4-A-2/3/4/6 parity + F4-A-7 safe_state TMR | ~75 % | ~50 % | ~0.13 FIT | ❌ ASIL-B SPFM still short |
| Phase B complete (W8) — F4-B-1 PE acc parity + F4-B-2 dup tile_ctrl FSM + LBIST + MBIST + F4-B-7 Yosys qual | ~92 % | ~75 % | ~0.04 FIT | ✅ ASIL-B |
| Phase C complete (W10) — fault-injection measures real DC, lifts conservative targets | ~94 % | ~80 % | ~0.04 FIT | ✅ ASIL-B + ASIL-C |
| Phase D complete (W18) — F4-D-1 TMR formal + F4-D-2 ECC formal + F4-D-3 ECC on PE acc + F4-D-6 interleaved Hamming + DICE on critical | ~99 % | ~92 % | ~0.008 FIT | ✅ ASIL-B + ASIL-C + ASIL-D |

The trajectory is consistent with `safety_case_v0_1.md` §1.4 / §4 aggregate ASIL achievability (ASIL-B v1.0 at W12; ASIL-D extension at W18). It also aligns with `ser_analysis_v0_1.md` §3 PMHF derivation (~6.1 FIT residual at full-spec post-Phase-D, with the same SRAM-ECC dominant lever).

---

## 4. Sensitivity analysis

Top 5 contributors to aggregate λ_DU (driving the SPFM gap):

| # | Failure mode | Module | λ_DU (FIT) | % of aggregate | Mitigation WP |
|---:|---|---|---:|---:|---|
| 1 | `npu_top.pe_acc.seu` | npu_top | 0.213 | 35.5 % | F4-B-1 (parity per accumulator) |
| 2 | `npu_top.sram_data.seu` | npu_top | 0.076 | 12.7 % | F4-A-1.1 (ECC integration) |
| 3 | `npu_top.pe_dataflow.seu` | npu_top | 0.068 | 11.3 % | F4-A-6 (parity on dataflow) |
| 4 | `npu_top.pe_weight_reg.seu` | npu_top | 0.053 | 8.9 % | F4-A-2 (parity on weight) |
| 5 | `ecc_secded.data_out.seu` | ecc_secded | 0.027 | 4.4 % | F4-D-2 (formal proof) + parity on data_out |

**Closing #1-#4 (~68 % of aggregate λ_DU) lifts SPFM from 10.23 % to ~71 % — within range of ASIL-B target after Phase A + B.**

---

## 5. Cross-module re-crediting opportunity (deferred to v0.2)

Per `tools/safety/safety_mechanisms.yaml`, the `aggregate_safe_state_fault_out` mechanism currently has `target_dc_pct: 40.0` reflecting conservative escalation coverage. Several module-level rows that today carry `mechanism_id: aggregate_safe_state_fault_out` (e.g. `npu_top.systolic_drain.seu`, `npu_top.tile_ctrl_fsm.seu`, several plausibility / safe_state failure modes) are credited at this 40 %.

**v0.2 of this aggregate roll-up** will quantify the actual aggregate-level escalation rate from Phase C `safe_state_controller_inj_1k` campaign data and replace the conservative 40 % with the measured value. Expected lift: **+5 to +10 percentage points** of aggregate SPFM (assuming measured aggregate-coverage rate is ~70-80 %, which is typical for well-designed safe-state aggregators).

This is purely a **measurement + YAML update**, not new RTL. Schedule: post-W7 once the Phase C campaigns have run.

---

## 6. Reproduce

```bash
# Generate the per-mode detail (auto-rendered)
python -m tools.safety.fmeda --aggregate --asil ASIL-B \
    --output docs/safety/fmeda/aggregate_fmeda_v0_1.md

# Get JSON for programmatic consumption (e.g. CI gate)
python -m tools.safety.fmeda --aggregate --asil ASIL-B --json

# Regression check vs committed baseline
python -m tools.safety.regress_check --baseline docs/safety/fmeda/baseline.json
```

The aggregate JSON is included in `baseline.json` future revisions for regression-gate purposes.

---

## 7. Open items for v0.2

1. **Named Track 2 lead, independent reviewer, Safety Manager / Approver** — currently TBD per Safety Case §0
2. **Cross-module re-crediting** per §5 above — requires Phase C campaign data
3. **Add aggregate row to `baseline.json`** so the regression gate catches aggregate drift, not just per-module drift (currently regress_check operates per-module only)
4. **Re-run after Phase A milestone closes** to validate the §3 trajectory's first row of forecast numbers
5. **Add lane_fusion + sensor I/O modules** to failure_modes.yaml so the aggregate covers the full IP scope (currently only 6 modules; lane_fusion FMEDA scheduled W6 per `fmeda/README.md`)

---

## 8. Document control

| Field | Value |
|---|---|
| Document ID | ASTR-FMEDA-AGG-SUMMARY-V0.1 |
| Revision | 0.1 |
| Revision date | 2026-04-20 |
| Author | TBD (Track 2 lead) |
| Reviewer | TBD |
| Approver | TBD |
| Distribution | Internal + TÜV SÜD India + first NDA evaluation licensee |
| Retention | 15 years post-product-discontinuation per ISO 26262-8 §10 |
| Supersedes | None — this is v0.1 |

### 8.1 Revision triggers

This summary is re-issued on any of:

1. New module added to `failure_modes.yaml` → §1 table grows
2. F4 phase milestone closes → re-run, refresh §1 + §3 trajectory measured numbers
3. Cross-module re-crediting v0.2 lands per §5 → new column in §1 table
4. Mechanism `target_dc_pct` revised based on Phase C measurement → re-render
5. Confirmation review feedback that changes any number
