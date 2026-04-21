# Safety Findings — Remediation Plan v0.1

**Document ID:** ASTR-SAFETY-REMEDIATION-V0.1
**Date:** 2026-04-20
**Status:** Proposed. Each WP opens as a tracked task in the Track 1 (compute IP) or Track 2 (safety-cert) backlog.
**Inputs:** `docs/safety/fmeda/dms_fusion_fmeda.md`, `docs/safety/fmeda/npu_top_fmeda.md`
**Target:** module-level ASIL-B pass (SPFM ≥ 90 %, LFM ≥ 60 %, PMHF ≤ 100 FIT) across all safety-critical modules by W12. ASIL-D extension scoped separately (W14-W20).

---

## 0. TL;DR

Two modules have FMEDA reports today. Both fail their ASIL targets at module scope. The fixes fall into five categories:

1. **Low-cost RTL hardening** (parity + ECC integration): ~3 weeks, closes most of the gap.
2. **Mechanism characterization** (fault-injection to measure real DC): ~2 weeks, improves declared coverage from conservative targets to measured values.
3. **Aggregate safe-state escalation** (map module-level uncovered to cross-module DD): cross-cuts Tracks 1+2.
4. **Formal verification** of TMR + ECC: required for ASIL-D, targeted W7-W9.
5. **Higher-cost RTL hardening** (LBIST, MBIST, duplicated FSMs): ~5 weeks, closes ASIL-D residual.

Expected SPFM trajectory after fixes:

| Module | Today | After Phase A (W4) | After Phase B (W8) | After Phase C (W10) |
|---|---:|---:|---:|---:|
| `dms_fusion` | 84.7 % | 95 %* | 97 %* | 98.5 %* |
| `npu_top`    |  2.1 % | 75 %* | 92 %* | 95 %* |

*Estimates — replaced with measured FMEDA re-runs after each phase. Phase C = post fault-injection campaign.

---

## 1. Findings summary (what we're fixing)

### 1.1 dms_fusion (SPFM 84.70 %, LFM 91.95 %, PMHF 0.008 FIT)

| # | Finding | λ_DU (FIT) | Root cause |
|---:|---|---:|---|
| D1 | `tmr_valid_r` SEU uncovered at module scope | 0.000425 | No mechanism on the 1-bit valid flag driving the voter |
| D2 | `cont_closed` SEU partial coverage (65 % DC) | 0.001 | Counter-reset mechanism is conservative target |
| D3 | `cont_distracted` SEU partial coverage (65 % DC) | 0.001 | Same |
| D4 | `score_filt_x4` SEU partial coverage (70 % DC) | 0.0011 | IIR self-correction is conservative target |
| D5 | `raw_score` stuck-at partial coverage (60 % DC) | 0.00056 | Priority-encoder safe-default is conservative target |

### 1.2 npu_top (SPFM 2.08 %, LFM 0 %, PMHF 0.53 FIT)

| # | Finding | λ_DU (FIT) | Root cause |
|---:|---|---:|---|
| N1 | PE accumulators — no mechanism (SEU + stuck) | **0.256 SEU + 0.044 stuck = 0.300** | No parity / ECC on 16 × 32-bit accumulator bank |
| N2 | PE weight registers — no mechanism | 0.053 + 0.011 = 0.064 | No parity on 16 × 8-bit weight FFs |
| N3 | SRAM data — ECC RTL exists but not wired | 0.076 | `npu_top.v` "Current gaps" comment documents this |
| N4 | PE dataflow pass-through — no mechanism | 0.068 | No parity on inter-PE activation bus |
| N5 | tile_ctrl config registers — no mechanism | 0.023 | No parity on captured cfg_k / cfg_ai_base / etc. |
| N6 | Multiplier tree stuck-at — no mechanism | 0.022 | No LBIST |
| N7 | DMA FSM / address — partial aggregate coverage | 0.010 (DU) | Aggregate only catches stall-to-watchdog path |
| N8 | SRAM address decoder stuck-at — no mechanism | 0.005 | No MBIST |
| N9 | Precision mode broadcast stuck-at | 0.0007 | No parity or boot self-check |
| N10 | busy/done handshake SEU | 0.0005 | Partial aggregate coverage only |
| N11 | tile_ctrl FSM SEU | 0.0004 | Partial aggregate coverage only |
| N12 | systolic_drain c_valid SEU | 0.002 | Partial aggregate coverage only |

**Verdict:** npu_top today has almost no runtime safety coverage. This matches the repo's own honest comment (`npu_top.v:68` "No ECC, no BIST"). It is exactly the evidence an OEM auditor would produce from an independent FMEDA, so fixing it *before* a licensee does their audit is the right move.

---

## 2. Fix categories

| Cat | What | When it applies | Typical cost |
|---|---|---|---|
| **RTL-H (hardening)** | Add new RTL: parity, ECC, LBIST, MBIST, TMR, duplicate FSM | Uncovered dangerous failure modes | 2–8 days per fix |
| **RTL-I (integration)** | Wire an *existing* mechanism RTL block into a module that needs it | ECC SECDED + npu_sram_bank (not wired today) | 2–5 days |
| **MECH-C (characterize)** | Run fault-injection to measure real DC; raise declared target if measured > target | Mechanism with conservative declared DC | 1 campaign per mechanism; ~1 day per campaign to write + 1 day WSL runtime |
| **AGG-E (aggregate escalation)** | Route a module-level SPF to `safe_state_controller` so the fault becomes dangerous-detected at aggregate scope | Low-rate single FFs where module-level mechanism is disproportionate | Hours of RTL + documentation |
| **FORMAL (formal verification)** | Prove safety-mechanism properties via Yosys + SBY | TMR voter, ECC SECDED, safe-state FSM | 3–5 days per property set |
| **SCOPE (spec re-scope)** | Carry an item-level safety-goal claim differently (e.g. ASIL decomposition, reduced scope, licensee-shared responsibility) | When silicon-scope fixes are infeasible or uneconomic | Paperwork only |

---

## 3. Prioritized backlog (by SPFM impact × cost)

Priorities below are by **FIT-per-engineer-week** — the fastest SPFM lift per engineering effort. We execute Phase A first to close the biggest cheap wins, then Phase B, then characterization (Phase C) to lift DC on mechanisms already in place.

### Phase A — Low-cost, high-impact RTL (Wk 2–4, Track 1)

| WP | Fix | Covers findings | Est effort | λ_DU closed (FIT) |
|---|---|---|---:|---:|
| **F4-A-1** | Wire `ecc_secded` into `npu_sram_bank` (instantiate + codeword routing). Add SRAM scrub loop. | N3, N8 (partial) | 5 days | 0.076 |
| **F4-A-2** | Parity bit per `npu_pe.weight_reg` (16 PEs × 9 bits = 144 FFs). Fault flag aggregated to `fault_detected` output. | N2 | 3 days | 0.064 |
| **F4-A-3** | Parity per tile_ctrl latched cfg word. | N5 | 2 days | 0.023 |
| **F4-A-4** | Parity on precision_mode broadcast. | N9 | 1 day | 0.0007 |
| **F4-A-5** | TMR or 3-state parity on `dms_fusion.tmr_valid_r` (1-bit fix). | D1 | 0.5 day | 0.000425 |
| **F4-A-6** | Parity per `npu_pe.a_out/a_valid_out/sparse_skip_out` pass-through (16 PEs × 10 bits). | N4 | 2 days | 0.068 |
| **F4-A-7** | TMR or Hamming-encode the 2-bit `safe_state` FSM register in `safe_state_controller.v`. Cheapest, biggest single-module SPFM lift on safe_state_controller (the most safety-critical FFs in the chip). **MUST FIX before ASIL-B safety case v1.0.** Discovered 2026-04-20 from `safe_state_controller` FMEDA (SPFM 34.80 %). | safe_state_controller.state.seu | 0.5 day | 0.0009 (small λ but uncovered single-point fault on the chip's most critical state) |
| **Phase A total** | | | **~14.5 days** | **~0.233 FIT** |

**Expected post-Phase-A:** `npu_top` SPFM ~75 %, `dms_fusion` SPFM ~95 % (the tmr_valid fix alone closes most of the dms_fusion gap).

### Phase B — Higher-cost RTL (Wk 5–8, Track 1)

| WP | Fix | Covers findings | Est effort | λ_DU closed (FIT) |
|---|---|---|---:|---:|
| **F4-B-1** | Parity per `npu_pe.acc` (single-bit parity over 32 bits, 16 PEs). Alt: ECC — stronger but costlier. | N1 | 8 days | 0.300 |
| **F4-B-2** | Duplicated-and-compared tile_ctrl FSM (two instances, comparator on state vector). | N11 | 4 days | 0.0004 (but big DC lift on aggregate) |
| **F4-B-3** | LBIST scaffold for PE multiplier tree (scan-chain based, boot-time). | N6 | 10 days | 0.022 (latent → DD at boot) |
| **F4-B-4** | MBIST on SRAM address decoder. | N8 | 5 days | 0.004 |
| **F4-B-5** | TMR on `busy`/`done` handshake (2 bits × 3 lanes). | N10 | 1 day | 0.0005 |
| **F4-B-6** | Harden systolic_drain `c_valid` (parity + re-issue on mismatch). | N12 | 3 days | 0.0015 |
| **F4-B-7** | **Yosys TCL2 qualification.** Set up gate-level cocotb simulation against the same suite that validates RTL; cross-check Yosys-synthesised netlist behaviour matches RTL bit-exactly for every safety-critical module. Validates the synthesis tool per ISO 26262-8 §11.4.7 Method 3. Discovered 2026-04-20 from TCL evaluation `docs/safety/tcl/tcl_evaluations_v0_1.md` §3.3. | Yosys TCL2 closure | 5 days | n/a (process WP, not λ_DU closure) |
| **Phase B total** | | | **~36 days** | **~0.328 FIT** |

**Expected post-Phase-B:** `npu_top` SPFM ~92 %, crossing ASIL-B SPFM target (90 %).

### Phase C — Mechanism characterization (Wk 7–10, Track 2)

Fault-injection campaigns measure real DC vs the conservative placeholder targets. Typical result: measured > target, so we can raise the declared DC in `safety_mechanisms.yaml` and the FMEDA aggregate improves automatically.

| WP | Campaign | Mechanism(s) characterized | Covers findings | Est effort |
|---|---|---|---|---:|
| **F4-C-1** | `tmr_voter_seu_1k` (sample already shipped) | `tmr_voter` | D1 (partial), N2+N4 post-Phase-A | 3 days |
| **F4-C-2** | `ecc_secded_bf_10k` | `ecc_secded` | N3 (validates 99.5 % target) | 2 days |
| **F4-C-3** | `dms_fusion_inj_5k` | `iir_self_correcting`, `counter_resets_periodically`, `blink_window_resamples` | D2, D3, D4 | 5 days |
| **F4-C-4** | `safe_state_inj_1k` | `aggregate_safe_state_fault_out` | N7, N10, N11, N12 | 3 days |
| **F4-C-5** | **Aggregate FMEDA cross-module re-crediting v0.2.** Replace conservative `aggregate_safe_state_fault_out` `target_dc_pct: 40.0` with measured value from F4-C-4 campaign. Expected aggregate SPFM lift +5..+10 pp. Pure YAML update + re-render of `docs/safety/fmeda/aggregate_summary_v0_1.md` per §5. | N7, N10, N11, N12 (re-crediting) | 1 day | aggregate SPFM lift |
| **F4-TEST-1** | **Fix pre-existing `test_reset_state` cold-start window.** Discovered 2026-04-20 during WSL validation (`docs/safety/wsl_validation_session_2026_04_20.md` §1.4). Test asserts `dms_confidence == 100` immediately after `rst_n` deasserts, but `rtl/tmr_voter/tmr_voter.v` resets `voted` to `32'h0` (not matching the input lanes' reset values). Three options: (a) parametrise the voter reset value to match the lanes, (b) change the test to wait one `valid=1` cycle, (c) document cold-start behaviour as design decision. Recommend option (b). | Not a remediation gap; test-quality cleanup | 0.5 day | n/a |
| **Phase C total** | | | | **~14.5 days** |

**Expected post-Phase-C:** both modules cross ASIL-B. Mechanism DC targets now *measured*, not declared — defensible at TÜV audit.

### Phase D — ASIL-D extension (Wk 13–18, Track 2)

| WP | Fix | Covers findings | Est effort |
|---|---|---|---:|
| **F4-D-1** | Formal proofs of `tmr_voter` (single-lane-fault tolerance, no-triple-disagree-voted-output) | Formal evidence for ASIL-D | 5 days |
| **F4-D-2** | Formal proofs of `ecc_secded` (single-bit-correct, double-bit-detect) | Formal evidence for ASIL-D | 4 days |
| **F4-D-3** | Upgrade `npu_pe.acc` parity → ECC for ASIL-D margin. | N1 | 6 days |
| **F4-D-4** | CCF analysis per ISO 26262-9 §7 | All redundant elements | 8 days |
| **F4-D-5** | Soft Error Rate (SER) analysis per ISO 26262-11 §7 | All SRAM + large FF banks | 5 days |
| **F4-D-6** | Switch SECDED layout from systematic (parity stored separately) to standard interleaved Hamming(72,64) where parity bits occupy codeword positions 1, 2, 4, 8, 16, 32, 64. Closes the parity-bit-flip aliasing limitation documented in `tools/safety/ecc_ref.py` and inherited from `rtl/ecc_secded/ecc_secded.v`. Today's layout *detects* parity-bit single flips but may spuriously "correct" one of `{data[0,1,3,7,15,31,63]}`. For ASIL-B this is acceptable (detection lifts to safe state); for ASIL-D the spurious correction itself is a hazard. | Closes systematic-layout aliasing on N3 + future ECC integrations | 6 days |
| **F4-D-7** | **OpenROAD TCL2 qualification.** Validate OpenROAD physical-synthesis flow against a published reference design (e.g., a small RISC-V core with known timing); document delta to commercial STA tools per ISO 26262-8 §11.4.7. Discovered 2026-04-20 from TCL evaluation §3.5. | OpenROAD TCL2 closure | 4 days |
| **F4-D-8** | **ASAP7 PDK TCL3 qualification.** Validate ASAP7 area / timing / power projections against published academic benchmarks (ASAP7 RISC-V reference designs); document the delta vs commercial PDKs at the same node. Compensating measure already in place: spec sheet rev 1.4 labels every ASAP7-derived number as "Tape-out target" / "projection". Discovered 2026-04-20 from TCL evaluation §3.6. | ASAP7 TCL3 closure (final residual after labelling) | 3 days |

---

## 4. Expected FMEDA trajectory

### dms_fusion

| Milestone | SPFM | LFM | PMHF | Notes |
|---|---:|---:|---:|---|
| Today (v0.1) | 84.70 % | 91.95 % | 0.008 FIT | Baseline |
| After F4-A-5 (W2) | ~90 % | ~93 % | 0.008 FIT | tmr_valid fix closes the single uncovered SPF |
| After Phase B (W8) | ~95 % | ~94 % | 0.008 FIT | mostly stable; dms_fusion was already close |
| After Phase C (W10) | ~97–98 % | ~96 % | 0.008 FIT | measured DC replaces conservative targets |
| After Phase D (W18) | ~99.5 % | ~99 % | 0.008 FIT | ECC on critical FFs + formal proofs |

### npu_top

| Milestone | SPFM | LFM | PMHF | Notes |
|---|---:|---:|---:|---|
| Today (v0.1) | 2.08 % | 0 % | 0.53 FIT | Baseline — catastrophic |
| After F4-A-1..6 (W4) | ~75 % | ~50 % | ~0.13 FIT | Biggest jumps: ECC-in-SRAM + PE parity |
| After Phase B (W8) | ~92 % | ~75 % | ~0.04 FIT | Crosses ASIL-B SPFM + LFM |
| After Phase C (W10) | ~94 % | ~80 % | ~0.04 FIT | measured DC + aggregate escalation |
| After Phase D (W18) | ~99 % | ~92 % | ~0.008 FIT | LBIST + ECC on acc + formal |

---

## 5. Mapping to the existing 16-week plan

These remediation WPs slot into `docs/best_in_class_design.md` §7.2 as follows:

| Plan week | Track 2 (safety) | Track 1 (compute) — **NEW from this plan** |
|---:|---|---|
| 2 | SEooC + Safety Manual outline ✅ | **F4-A-5** dms_fusion tmr_valid fix (half-day) |
| 3 | First FMEDA on npu_top ✅; Cocotb fault-injection harness scaffold ✅ | **F4-A-1** wire ECC into npu_sram_bank |
| 4 | 1000-fault campaign on tmr_voter + ecc_secded | **F4-A-2, F4-A-3, F4-A-4, F4-A-6** parity WPs |
| 5 | TÜV SÜD pre-engagement | **F4-B-2** duplicated tile_ctrl FSM |
| 6 | TÜV pre-assessment workshop | **F4-B-1** PE accumulator parity |
| 7 | Formal setup (Yosys + SBY) | **F4-B-4** SRAM MBIST |
| 8 | Formal proofs of tmr_voter | **F4-B-3** LBIST scaffold (can slip to W10) |
| 9 | 10K-fault dms_fusion campaign (F4-C-3) | **F4-B-5, F4-B-6** busy/done TMR + drain parity |
| 10 | DC/LFM/SPFM aggregate report v0.1 — **uses measured numbers from F4-C-1..4** | — |
| 11 | Safety case document outline | — |
| 12 | **ASIL-B safety case v1.0** — target achievable after Phase A+B+C | — |
| 13+ | ASIL-D extension (F4-D-1..5) | — |

**Net impact on plan calendar:** the RTL hardening WPs were not explicitly in the prior §7.2 plan. Adding them consumes ~45 Track 1 engineer-days across W2-W9. This is in addition to the existing Track 1 scope (F1, F2, F3, F7). If resourcing is 2 RTL engineers, Track 1 work extends ~3 weeks — still within W12 for ASIL-B. If resourcing is 1 engineer, prioritise Phase A + F4-B-1 (PE accumulator parity) + Phase C characterization; defer LBIST/MBIST to Phase D.

---

## 6. Recommended execution order (single-engineer case)

If only one Track 1 engineer is available, execute in this sequence:

1. **W2**: F4-A-5 (tmr_valid — half-day). Closes dms_fusion SPFM gap immediately.
2. **W2-W3**: F4-A-1 (wire ECC into npu_sram_bank). Biggest single npu_top λ_DU closure.
3. **W3-W4**: F4-A-2, F4-A-3, F4-A-4, F4-A-6 (parity batch).
4. **W5-W6**: F4-B-1 (PE accumulator parity — biggest remaining λ_DU).
5. **W7**: F4-B-2 (duplicated tile_ctrl FSM).
6. **W8-W9**: Rerun FMEDAs. Write fault-injection campaigns for Phase C.
7. **W9-W10**: F4-C campaigns run in WSL. Aggregate + update mechanism DC.
8. **W11**: Rerun FMEDAs with measured DC. Confirm ASIL-B pass.
9. **W12**: Safety case v1.0.

LBIST / MBIST / formal / CCF / SER slip to Phase D (post-ASIL-B).

---

## 7. Open questions for the founder

1. **Resourcing**: how many Track 1 RTL engineers (1 or 2)? Drives calendar.
2. **Parity vs ECC on PE accumulator** (F4-B-1): parity is 1/16 the overhead but 1/2 the detection; ECC is stronger for ASIL-D. Default to parity for ASIL-B, upgrade in Phase D.
3. **LBIST scope** (F4-B-3): boot-time only, or also periodic during mission mode? Periodic LBIST requires idle windows the compiler must schedule.
4. **ASIL-D for npu_top**: is this a real target, or is ASIL-B on compute + ASIL-D only on safety-wrapper modules (dms_fusion, safe_state_controller) the realistic claim? Mobileye and Qualcomm both use the latter pattern.
5. **Licensee-visible changes**: these RTL additions alter AstraCore's `fault_detected[15:0]` bitfield layout and add interrupt paths. Do we freeze the licensee-visible interface at W4 (before parity batch lands) or at W12 (after all Phase A+B)? Frozen-at-W12 is safer for the safety case but slows any W2-W11 evaluation kit.

---

## 8. Regression + sign-off gate

Proposed CI gate (runs on every PR touching `rtl/`, `tools/safety/`, or `docs/safety/`):

```bash
# 1. Unit tests
python -m pytest tests/test_fmeda.py tests/test_fault_injection_planner.py

# 2. Re-run all per-module FMEDAs
for mod in dms_fusion npu_top tmr_voter ecc_secded safe_state_controller plausibility_checker; do
    python -m tools.safety.fmeda --module $mod --asil ASIL-B --json > /tmp/$mod.json
done

# 3. Assert no regression vs baseline
python tools/safety/regress_check.py --baseline docs/safety/fmeda/baseline.json
```

`tools/safety/regress_check.py` (to be written at W4): fails if any module's SPFM or LFM drops > 1 percentage point vs the committed baseline. Prevents silent safety erosion.

---

## 9. Document control

| Field | Value |
|---|---|
| Document ID | ASTR-SAFETY-REMEDIATION-V0.1 |
| Revision | 0.1 |
| Revision date | 2026-04-20 |
| Next revision | After W4 FMEDA re-run (expected W5) |
| Supersedes | None |
