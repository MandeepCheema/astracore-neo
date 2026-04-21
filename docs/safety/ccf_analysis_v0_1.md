# Common Cause Failure (CCF) Analysis

**Document ID:** ASTR-SAFETY-CCF-V0.1
**Date:** 2026-04-20
**Standard:** ISO 26262-9:2018 §7 (Analysis of dependent failures) + ISO 26262-5:2018 Annex D (Common cause failure)
**Element:** AstraCore Neo NPU + Sensor-Fusion IP block (SEooC)
**Status:** v0.1 — first formal release. Required for ASIL-D extension. Closes risk R15 in `docs/safety/safety_case_v0_1.md` §6.3 and F4-D-4 in `docs/safety/findings_remediation_plan_v0_1.md`.
**Classification:** Internal — pre-engagement draft for TÜV SÜD India workshop.
**Author:** TBD (Track 2 lead) — currently founder + collaborator
**Reviewer:** TBD (independent reviewer per ISO 26262-2 §7)
**Approver:** TBD (Safety Manager)

---

## 0. Purpose and framing

A redundant safety architecture (TMR voter, dual-channel ECC, lockstep) is only as strong as its **independence**. A Common Cause Failure (CCF) is a single root cause that defeats multiple supposedly-independent channels simultaneously, collapsing the redundancy back to a single point of failure.

> **Why this analysis matters at ASIL-D.** ISO 26262-5 Annex D and ISO 26262-9 §7 require that for any safety mechanism relying on redundancy to claim coverage above what a single channel could provide, the CCF analysis must:
>
> 1. Identify plausible CCF initiators (the things that could simultaneously affect multiple channels)
> 2. Quantify the residual CCF-induced failure rate (β-factor times the channel failure rate)
> 3. Document mitigation measures (separation, diversity, monitoring) that reduce β
> 4. Show that the resulting CCF-residual rate is consistent with the FMEDA's assumed coverage
>
> Without CCF analysis, a TMR voter cannot claim 99 % DC at ASIL-D — only the single-channel coverage (~67 % at best, since two-of-three is just majority of three independent FFs).

### 0.1 Companion documents

- `docs/safety/safety_case_v0_1.md` §6.3 R15 — risk this analysis closes
- `docs/safety/findings_remediation_plan_v0_1.md` Phase D F4-D-4 — work package this analysis is the deliverable for
- `docs/safety/seooc_declaration_v0_1.md` §5 — declared mechanism coverages this analysis verifies independence for
- `docs/safety/fmeda/baseline.json` — FMEDA numbers that this analysis qualifies
- `tools/safety/safety_mechanisms.yaml` — mechanisms whose β-factors land back here
- `docs/safety/iso26262_gap_analysis_v0_1.md` Part 9 §7 row — gap this closes

---

## 1. Methodology

### 1.1 CCF initiator categories (per ISO 26262-9 §7.4.2)

CCF initiators that can defeat redundancy in a digital ASIC:

| Category | Description | Typical examples |
|---|---|---|
| **Environmental** | External condition affecting all channels equally | Junction temperature spike, EMI burst, supply-voltage glitch |
| **Manufacturing** | Defect inserted at fabrication that affects all replicated instances | Lithography defect spanning multiple cells, etch variation, identical via failure |
| **Design (systematic)** | Specification or RTL bug present in every replicated channel | Same RTL module instantiated three times = same bug three times |
| **Software (systematic)** | Same SW bug affecting every replicated execution | Compiler bug; off-by-one in kernel scheduler |
| **Common services** | Shared infrastructure feeding all channels | Single clock tree; single reset distribution; single power rail |
| **Test / debug intrusion** | DFT or scan logic that bypasses the safety mechanism if mis-configured | DFT_isolation_enable accidentally asserted in mission mode |
| **Aging / wear-out** | Common degradation mechanism affecting all channels with the same exposure | Electromigration on identical wires; bias-temperature instability on identical FFs |
| **Radiation / soft errors** | Single particle strike that affects multiple cells (multi-cell upset, MCU) | Heavy-ion strike causing 2-3 adjacent FFs to flip |

### 1.2 Beta-factor (β) approach

Per IEC 61508 Part 6 Annex D (referenced by ISO 26262-9 §7.4.4) and the **MIL-HDBK-217** β-factor model:

```
λ_CCF = β × λ_channel
```

where:
- `λ_channel` = failure rate of a single redundant channel (FIT)
- `β` = common-cause factor (0.0 = perfectly independent; 1.0 = fully common-cause; typical values 0.01–0.1 for well-designed redundancy)
- `λ_CCF` = residual failure rate of the redundancy due to common cause

For the FMEDA aggregate to claim the redundancy's full DC, the licensee must satisfy:

```
λ_CCF + λ_uncovered_independent ≤ FMEDA-declared λ_DU per mechanism
```

### 1.3 β-factor target ranges (industry consensus)

| Redundancy type | Typical β | Notes |
|---|---:|---|
| Identical replicated channels (same RTL × 3 in same die, same clock tree, same supply) | 0.05 – 0.10 | High β — most CCF initiators apply |
| Replicated channels with physical separation (different die regions, separated supply) | 0.02 – 0.05 | |
| Diverse channels (different RTL implementations, different timing, etc.) | 0.005 – 0.02 | Hard to achieve at IP scale |
| Lockstep across two independent dies (separate package) | 0.001 – 0.005 | Best achievable; requires 2 IP instances + comparator |

**For AstraCore IP today**, all redundant elements (TMR voter lanes, ECC parity bits, etc.) are within the same die using the same clock tree → β ~0.05 is the realistic baseline; mitigation moves it toward the lower end.

---

## 2. CCF analysis per redundant element

### 2.1 TMR voter (rtl/tmr_voter/)

**Redundant element:** 3 input lanes (lane_a, lane_b, lane_c), each 32-bit.

**Use sites:** `dms_fusion.u_tmr_dal` (TMR vote on driver_attention_level + dms_confidence). Other instantiation patterns when added by F4 follow-up.

#### 2.1.1 CCF initiator analysis

| Initiator | Applies? | Mitigation in current RTL | Residual β contribution |
|---|:---:|---|---:|
| Junction temperature spike | YES | None at IP scope; thermal_zone monitor at SoC scope | 0.02 |
| EMI burst | YES | None at IP scope; EMI shielding at package scope (Licensee) | 0.005 |
| Supply-voltage glitch | YES | None at IP scope; PMIC + decoupling at SoC scope (Licensee) | 0.005 |
| Lithography / manufacturing defect | YES | None — 3 lanes are identical RTL instances; same place-and-route region likely | 0.01 |
| Same RTL bug × 3 | YES | tmr_voter is a single small module; bug in the *voter* defeats all 3 | covered in §2.1.4 below |
| Common clock | YES | Single clk distribution to all 3 lanes; clock failure = all 3 fail | 0.005 |
| Common reset | YES | Single rst_n distribution; reset glitch = all 3 fail | 0.002 |
| DFT intrusion | YES | dft_isolation_enable per AoU-7 (mission mode = low) | mitigated by AoU |
| Multi-cell upset (radiation) | YES | If 3 lane FFs are physically adjacent in layout, single-particle strike could flip 2 or 3 simultaneously | 0.005 |
| **Subtotal β (no extra mitigation)** | | | **~0.054** |

#### 2.1.2 Mitigations

| Mitigation | β reduction | Status |
|---|---:|---|
| Physical separation: synthesis constraints place lane_a, lane_b, lane_c in separated die regions (≥ 50 µm apart) | ~0.01 → 0.005 | **STUBBED** — synthesis floorplan constraints not yet documented |
| Diverse encoding: lane_a, lane_b, lane_c use different bit ordering or transformation, voter de-transforms before vote | ~0.01 → 0.002 | **DEFERRED** to v0.2 — implementation cost vs benefit assessment |
| Cross-lane clock-domain monitor: detect simultaneous lane disagreement and assert separate fault flag | n/a (improves detection of CCF, not avoidance) | **F4-D-4 follow-up** |
| Voter formal verification (F4-D-1): proves voter correctness regardless of lane state | mitigates the "same RTL bug" CCF | F4-D-1 |
| LBIST (F4-B-3): detects stuck-at faults in lane FFs at boot, before the CCF can manifest | reduces wear-out CCF contribution | F4-B-3 |

#### 2.1.3 Residual β estimate

| State | β estimate | TMR effective DC at ASIL-D |
|---|---:|---:|
| Today (no extra mitigation) | 0.054 | ~94.6 % (vs 99 % declared) — **insufficient for ASIL-D** |
| Post F4-B-3 + floorplan separation | 0.020 | ~98.0 % — close, still short of 99 % |
| Post F4-D-1 formal proofs + diverse encoding | 0.005 | ~99.5 % ✅ ASIL-D achievable |

#### 2.1.4 Voter common-mode failure

The voter itself is the CCF single-point — a bug in `rtl/tmr_voter/tmr_voter.v` is a single design defect affecting every TMR-voted output. Mitigations:

1. **Formal proof** (F4-D-1) — prove the voter satisfies SVA invariants 1-4 already in the RTL
2. **Independent test corpus** — `tmr_voter_seu_1k` campaign (already shipped) plus boundary-case tests
3. **Diverse second voter** (paranoid pattern) — instantiate a dual-rail comparator on the voted output; β between two voters drops to ~0.001 because the second voter can be a different RTL implementation. **DEFERRED** to v0.2 — cost-benefit tradeoff for ASIL-D margin.

### 2.2 ECC SECDED (rtl/ecc_secded/)

**Redundant element:** 8 parity bits providing single-bit error correct + double-bit error detect on a 64-bit data word.

**Use sites:** `rtl/npu_sram_bank_ecc/` (combinational wrapper; npu_top integration pending F4-A-1.1).

#### 2.2.1 CCF initiator analysis

| Initiator | Applies? | Mitigation | β contribution |
|---|:---:|---|---:|
| Multi-bit SRAM upset (single-particle hit affects 2 adjacent bits) | YES | Standard mitigation: physical separation of bit-lines (`scrambling`); typically achieves ~0.01 with proper interleaving | 0.01 (assumes Licensee uses scrambled bit-line layout per foundry SRAM compiler) |
| Same SECDED RTL bug | YES | Bug in `ecc_secded.v` defeats all SRAM banks using the wrapper | covered in §2.2.4 |
| Common power rail to all SRAM banks | YES | Power glitch could corrupt all banks simultaneously | 0.005 |
| Common clock | YES | Same clk drives all bank reads | 0.002 |
| Manufacturing defect on parity-bit storage | LOW | Parity bits are part of the SRAM array; same defect probability per bit | 0.005 |
| Aging on identical bit cells | YES | All 72 bits per word age identically; bias-temperature instability could correlate failures | 0.002 |
| **Subtotal β** | | | **~0.024** |

#### 2.2.2 Mitigations

| Mitigation | β reduction | Status |
|---|---:|---|
| Bit-line scrambling at foundry SRAM compiler | mitigates multi-bit upset | **LICENSEE responsibility per AoU**; document in Safety Manual |
| Background scrubbing (read-and-rewrite at slow rate) | reduces aging-correlated failures | **STUBBED** — scrub controller not yet in npu_sram_bank_ecc wrapper |
| Formal proof of SECDED (F4-D-2) | mitigates "same RTL bug" CCF | F4-D-2 |
| Interleaved Hamming layout (F4-D-6) | closes the parity-bit aliasing limitation | F4-D-6 |
| Per-bank ECC counter telemetry (already in wrapper) | enables wear-out trend detection by Licensee supervisor | ✅ DONE in npu_sram_bank_ecc |

#### 2.2.3 Residual β estimate

| State | β estimate | ECC effective DC at ASIL-D |
|---|---:|---:|
| Today (npu_top still uses bare npu_sram_bank — no ECC) | n/a | n/a — ECC not active |
| Post F4-A-1.1 + Licensee bit-line scrambling | 0.024 | ~97.6 % (vs 99.5 % declared) |
| Post F4-D-2 formal + scrubber + F4-D-6 interleaved | 0.008 | ~99.2 % ✅ ASIL-D achievable |

#### 2.2.4 SECDED common-mode failure

A bug in `ecc_secded.v` propagates to every wrapper instance. Mitigations:

1. **Formal proof (F4-D-2)** — prove the SVA invariants `a_err_mutex` + `a_corrected_iff_single` + `a_encode_no_errors`
2. **Bit-exact mirror** in `tools/safety/ecc_ref.py` cross-validates the encode/decode logic at CI time (TI2/TD1/TCL1 per `tcl_evaluations_v0_1.md` §3.8)
3. **Standardised ECC algorithm** — Hamming(72,64) is one of the most-tested algorithms in computer engineering; the chance of an undetected bug in an algorithm this scrutinised is low

### 2.3 Lockstep (dual AstraCore IP instance, planned)

**Redundant element:** Two complete AstraCore IP block instances running the same workload, with `lockstep_compare_in` boundary signal allowing the Licensee to compare outputs cycle-by-cycle.

**Status:** AstraCore IP exposes the boundary signal per SEooC §2.3, but lockstep instantiation is **Licensee-implemented**. The CCF analysis below is for guidance to the Licensee.

#### 2.3.1 CCF initiator analysis (Licensee scope)

| Initiator | Mitigation by Licensee | β contribution |
|---|---|---:|
| Same RTL bug × 2 instances | Lockstep with diverse implementation (e.g., two different RTL revisions) — usually impractical at IP scope | 0.05 if identical; 0.005 if diverse |
| Common clock to both instances | **Diverse clocks** (two PLLs from same crystal but different multiplier ratios) | 0.005 |
| Common power rail | **Separate power islands** with independent regulators | 0.005 |
| Same package thermal | **Physical separation** at package level (separate dies on same substrate) | 0.005 |
| Manufacturing defect (per-die yield) | Different dies = different defect probabilities | 0.001 |
| **Subtotal β (with diversity + separation mitigations)** | | **~0.021** |
| **Subtotal β (no mitigations — naive lockstep)** | | **~0.066** |

#### 2.3.2 Recommended Licensee architecture for ASIL-D via lockstep

> **AstraCore recommended pattern for ASIL-D item-level claim via lockstep:**
>
> 1. Two AstraCore IP instances on the same SoC, **separated by ≥ 200 µm** to limit shared-environment CCF
> 2. **Diverse PLLs** feeding the two clocks (same crystal, different multipliers within FSC FTTI budget)
> 3. **Independent power-island** for each instance
> 4. `lockstep_compare_in` cross-wired such that each instance's output feeds the other's compare input
> 5. Comparator in Licensee SoC asserts `safe_state_active` on any cycle disagreement
> 6. Comparator **must use a diverse implementation** (FPGA hard-IP comparator, not the same RTL as either AstraCore instance) to avoid the comparator becoming the new CCF single-point

This pattern achieves β ~0.005 → effective DC ~99.5 % at the lockstep level, sufficient to close the ASIL-D requirement on SG-1.1 when combined with the underlying AstraCore IP's per-channel SPFM (post-Phase-D ~99 %).

### 2.4 Dual-channel ASIL decomposition (FSC §1.1 pattern)

**Redundant element:** Two independent perception channels per ASIL-D = ASIL-B(D) + ASIL-B(D) decomposition (FSC §1.1).

**Pattern A (default):** Channel 1 = AstraCore camera+radar+lidar fusion; Channel 2 = Licensee independent radar-only AEB fallback.

#### 2.4.1 CCF initiator analysis

| Initiator | Channel 1 | Channel 2 | Common to both? | β contribution |
|---|---|---|:---:|---:|
| Same camera input | YES | NO (radar-only) | partial — camera failure affects only Ch1 | 0.005 |
| Same radar input | YES | YES | YES — radar failure affects both | 0.030 |
| Same vehicle controller decision logic | YES | YES | depends on Licensee | 0.020 |
| Different perception algorithms | NO (Ch1 = NN; Ch2 = classical signal processing) | NO | NO — diverse | 0 |
| Different ASIL teams | (depends on Licensee org) | | | 0.005 |
| Common Licensee SoC integration | YES | YES | YES — SoC bug affects both | 0.005 |
| **Subtotal β** | | | | **~0.065** |

#### 2.4.2 Mitigation: diverse sensor fusion

For the dual-channel pattern to genuinely achieve ASIL-D, the Licensee must:

1. **Choose diverse sensors** — NOT both camera-only or both radar-only; a camera + radar pair is genuinely diverse (different physics)
2. **Use diverse perception algorithms** — Channel 1 NN-based, Channel 2 classical signal processing (or rule-based)
3. **Wire the channel comparator independently from either channel** (per §2.3.2)
4. **Limit common Licensee SoC integration** — separate decision logic for each channel feeds into a final voter, not a shared decision logic

With these mitigations, β drops to ~0.015, providing genuine ASIL-D coverage.

---

## 3. Aggregate β-factor summary

| Redundant element | Today β | Achievable β (with all Phase D mitigations) | Effective DC at ASIL-D? |
|---|---:|---:|:---:|
| TMR voter (intra-die) | 0.054 | 0.005 | ✅ post Phase D |
| ECC SECDED (intra-die SRAM) | 0.024 | 0.008 | ✅ post Phase D |
| Lockstep (dual instance) | 0.066 | 0.005 | ✅ Licensee-implemented per §2.3.2 |
| Dual-channel ASIL decomposition | 0.065 | 0.015 | ✅ Licensee-implemented per §2.4.2 |

Phase D Mitigations needed to achieve the right column:
- Physical separation (synthesis floorplan constraints) — F4-D-4 follow-up
- Bit-line scrambling at foundry SRAM compiler — Licensee scope (Safety Manual §3.1 documents requirement)
- Background SRAM scrubbing — F4-A-1.1 expansion (post-W4)
- Formal proofs of TMR + ECC — F4-D-1 + F4-D-2
- Interleaved SECDED layout — F4-D-6
- Diverse channel architecture — Licensee per §2.3.2 + §2.4.2 patterns

---

## 4. CCF detection mechanisms

In addition to avoidance (β reduction), ISO 26262-9 §7.4.5 recommends CCF *detection* — separate monitoring that can flag when a CCF has occurred:

| Detection mechanism | What it catches | Allocation |
|---|---|---|
| Cross-lane clock-domain monitor | Common clock failure that simultaneously affects all 3 TMR lanes | F4-D-4 follow-up — new RTL |
| ECC scrub counter trend | Aging-correlated bit failures (rising single-bit correction rate is a leading indicator of imminent CCF) | ✅ Already in `npu_sram_bank_ecc` (`ecc_corrected_count`) |
| Lockstep comparator + watchdog | Both instances simultaneously asserting same-but-wrong output | Licensee scope (recommended in §2.3.2) |
| Independent thermal monitor | Junction-temperature CCF on TMR voter | `rtl/thermal_zone/` (rule-based today; F1-A9 ML upgrade) |
| Power-supply monitor | Supply-voltage glitch CCF | Licensee PMIC scope |
| Independent supervisor watchdog (per Safety Manual §3.4 "RECOMMENDED") | `safe_state_active` stuck high or stuck low | Licensee SoC supervisor MCU |

Each detection mechanism asserts a distinct fault flag in `fault_detected[]` (Safety Manual §7.2) so the Licensee can distinguish single-channel faults from CCF-induced faults — informational for telemetry + field anomaly database (DIA §9).

---

## 5. CCF verification approach

ISO 26262-9 §7.4.6 requires verification that the CCF analysis is complete and the assumed β-factors are achieved. Verification activities for AstraCore IP:

| Activity | When | Output |
|---|---|---|
| Floorplan review at every ASIC place-and-route iteration | At each Yosys + OpenROAD run on a target node | Synthesis report shows lane-pair separation distances |
| Bit-line scrambling audit | Once per Licensee silicon target (pre-tape-out) | Foundry SRAM compiler config doc |
| Formal proofs of TMR + ECC (F4-D-1 + F4-D-2) | Phase D | SBY proof artefacts |
| Interleaved SECDED layout (F4-D-6) | Phase D | Updated `rtl/ecc_secded/` + Python mirror |
| Multi-fault injection campaign | Phase C extension | Cocotb harness extended to dual-injection patterns; expected detection rate vs single-injection baseline measures effective CCF coverage |
| Cross-lane clock monitor (F4-D-4) | Phase D | New `rtl/clock_monitor/` validates per §4 detection mechanism |

Each verification activity produces an artefact stored in `docs/safety/` and referenced from the Safety Case master `safety_case_v0_1.md` §3 evidence chain.

---

## 6. Open items for v0.2

1. **Named Track 2 lead, independent reviewer, Safety Manager / Approver** — currently TBD per §0
2. **Quantitative β-factor measurement** — currently estimated from MIL-HDBK-217 + IEC 61508 industry consensus; replace with measurements from multi-fault injection campaigns at Phase C+
3. **Floorplan separation target** — §2.1.2 says "≥ 50 µm" as a guideline; Licensee silicon program must confirm at target node
4. **Multi-fault injection campaign** — extension to fault-injection harness to inject correlated faults in multiple targets simultaneously; F4-C extension WP
5. **Background SRAM scrubber RTL** — currently STUBBED in §2.2.2; needs RTL authoring as F4-A-1.1 expansion
6. **Synthesis floorplan constraints** — currently STUBBED in §2.1.2; needs documentation as part of F4-B-7 (Yosys qualification)
7. **Safety Case §6.3 R15 closure** — once this analysis is approved at v1.0, R15 status flips from "open" to "mitigated"

---

## 7. Document control

| Field | Value |
|---|---|
| Document ID | ASTR-SAFETY-CCF-V0.1 |
| Revision | 0.1 |
| Revision date | 2026-04-20 |
| Author | TBD (Track 2 lead) — currently founder + collaborator |
| Reviewer | TBD |
| Approver | TBD |
| Distribution | Internal + TÜV SÜD India + first NDA evaluation licensee |
| Retention | 15 years post-product-discontinuation per ISO 26262-8 §10 |
| Supersedes | None — this is v0.1; closes ISO 26262 gap analysis Part 9 §7 |

### 7.1 Revision triggers

This CCF analysis is re-issued (with revision bump) on any of:

1. New redundant element added to the IP (e.g., dual NPU lanes for ASIL-D decomposition) → spawn new §2 entry
2. β-factor measurement from a fault-injection campaign supersedes the estimated value → §3 update
3. F4 phase milestone closes that adds a CCF mitigation → §2 mitigation tables update
4. New CCF initiator identified during field monitoring or licensee integration review
5. Change to bit-line scrambling assumption per Licensee silicon target
6. Safety Case §6.3 R15 status changes
7. Confirmation review feedback that changes any β estimate or mitigation
