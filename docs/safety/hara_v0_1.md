# Hazard Analysis and Risk Assessment (HARA)

**Document ID:** ASTR-SAFETY-HARA-V0.1
**Date:** 2026-04-20
**Standard:** ISO 26262-3:2018 §6 (Hazard analysis and risk assessment) + §8 (ASIL determination)
**Element:** AstraCore Neo NPU + Sensor-Fusion IP block (SEooC per `docs/safety/seooc_declaration_v0_1.md`)
**Status:** v0.1 — first formal release. Replaces the placeholder ASIL assumptions in SEooC §3.2 with derived numbers.
**Classification:** Internal — pre-engagement draft for TÜV SÜD India workshop and first NDA evaluation licensee.
**Author:** TBD (Track 2 lead) — currently founder + collaborator
**Reviewer:** TBD (independent reviewer per ISO 26262-2 §7 confirmation measures)
**Approver:** TBD (Safety Manager)

---

## 0. Purpose and framing

ISO 26262-3 §6 defines HARA as the work product that:

1. Identifies hazardous events at the **item level** (not the IP block level).
2. Classifies each hazardous event by **Severity (S)**, **Exposure (E)**, and **Controllability (C)** in the operational situation where it occurs.
3. Derives the **ASIL** from the (S, E, C) tuple per Table 4 of ISO 26262-3.
4. Defines **Safety Goals** as top-level safety requirements the item must satisfy to prevent or mitigate the hazardous events.

> **HARA is an item-level work product.** AstraCore is not the item — the licensee's vehicle program is the item. This document is therefore an **assumed HARA**: it documents the hazardous events and ASIL derivations we *assume* the licensee will arrive at for the three reference use cases the SEooC was developed against. The licensee is required by AoU-15 (SEooC §6.4) to perform their own item-level HARA and may revise these assumptions.
>
> **Why we still do this work:** the spec sheet's ASIL claims (rev 1.3 says "ASIL-D, certified") have no HARA backing today. An OEM evaluator's first DD pass *will* ask for the HARA. A defensible v0.1 — even if marked "assumed; licensee revises" — is the difference between a five-minute conversation and a six-month evaluation.

### 0.1 Companion documents

- `docs/safety/seooc_declaration_v0_1.md` — defines the assumed item context this HARA is performed against (§3 there)
- `docs/safety/functional_safety_concept_v0_1.md` — FSC v0.1 derives 25 Functional Safety Requirements (FSRs) from the 11 Safety Goals in this document; FSC §3 is the authoritative breakdown
- `docs/safety/iso26262_gap_analysis_v0_1.md` — Part 3 row tracks closure of HARA + ASIL determination (priority #3 in §2)
- `docs/safety/safety_manual_v0_1.md` — §10 indexes the AoUs that constrain how the licensee may use the assumptions here
- `docs/safety/findings_remediation_plan_v0_1.md` — F4 WPs trace back to safety goals derived in this document

### 0.2 Scope of this revision

This v0.1 covers three reference use cases drawn from the assumed item context (SEooC §3.2):

| # | Use case | AstraCore role |
|---|---|---|
| UC-1 | Forward perception → Forward Collision Warning + Automatic Emergency Braking + Lane Keep Assist | Camera + radar + lidar fusion → object detection + distance + lane positions → vehicle controller |
| UC-2 | Driver Monitoring System (drowsiness + distraction) | ToF/IR camera → gaze + head-pose tracking → fused attention level → vehicle / HMI |
| UC-3 | Surround view + Park Assist | 12 ultrasonics + 4 cameras + lidar → proximity alarm bands + obstacle annotated view → driver display + emergency brake |

Out of scope for v0.1 (deferred to v0.2 / v1.0):

- Highway pilot / traffic-jam pilot (SAE L3) — needs additional driver-takeover handover hazards
- Cabin emergency response (e.g., child-locked-in detection) — distinct sensor mix
- V2X-based hazards — depends on PHY licensee provides

---

## 1. Methodology

### 1.1 Severity (ISO 26262-3 §6.4.3, Table B.1)

| Class | Description | Examples |
|---|---|---|
| S0 | No injuries | Cosmetic damage |
| S1 | Light to moderate injuries | Bruising, sprains |
| S2 | Severe to life-threatening (survival probable) | Fractures, organ damage |
| S3 | Life-threatening to fatal (survival uncertain or impossible) | Multiple-vehicle collision at speed, pedestrian impact |

### 1.2 Exposure (ISO 26262-3 §6.4.4, Table B.2)

| Class | Frequency or duration |
|---|---|
| E0 | Incredible (< 0.001 % of operating time) |
| E1 | Very low (< 1 % of operating time, < once a year per vehicle) |
| E2 | Low (1–10 % of operating time, few times a year) |
| E3 | Medium (10–50 % of operating time, monthly) |
| E4 | High (> 50 % of operating time, daily) |

### 1.3 Controllability (ISO 26262-3 §6.4.5, Table B.3)

| Class | Description |
|---|---|
| C0 | Controllable in general (drivers reliably mitigate) |
| C1 | Simply controllable (≥ 99 % of average drivers) |
| C2 | Normally controllable (≥ 90 % of average drivers) |
| C3 | Difficult to control or uncontrollable (< 90 % can avoid the hazard) |

### 1.4 ASIL determination matrix (ISO 26262-3 §6.4.6, Table 4)

|     | C1 | C2 | C3 |
|----|---|---|---|
| **S1, E1** | QM | QM | QM |
| **S1, E2** | QM | QM | QM |
| **S1, E3** | QM | QM | A |
| **S1, E4** | QM | A  | B |
| **S2, E1** | QM | QM | QM |
| **S2, E2** | QM | QM | A |
| **S2, E3** | QM | A  | B |
| **S2, E4** | A  | B  | C |
| **S3, E1** | QM | QM | A |
| **S3, E2** | QM | A  | B |
| **S3, E3** | A  | B  | C |
| **S3, E4** | B  | C  | **D** |

(QM = Quality-Managed, no ISO 26262 ASIL requirement applies. ASIL-D = highest integrity level.)

### 1.5 Hazard identification approach

For each use case we systematically enumerate AstraCore-IP-related malfunctions and trace each to the worst-case vehicle-level hazardous event. The malfunction catalogue is generic (false-positive, false-negative, late, intermittent, wrong-attribute) applied per output type:

| Malfunction class | Meaning |
|---|---|
| **FP** (false positive) | Report something that isn't there |
| **FN** (false negative) | Miss something that is there |
| **L**  (late)            | Correct output delivered after FTTI |
| **W**  (wrong attribute) | Detected but wrong class / distance / velocity / lane |
| **I**  (intermittent)    | Output flickers between correct and incorrect |

### 1.6 Worst-credible-case principle

For each (malfunction × operational situation) pair we select the worst credible combination of S/E/C. "Credible" means we exclude scenarios where another independent vehicle subsystem would catch the malfunction first (e.g., a parking-speed FN that the brake-assist also catches via independent ultrasonic). This avoids over-claiming ASIL-D where item-level redundancy already mitigates.

---

## 2. UC-1 — Forward perception (FCW + AEB + LKA)

### 2.1 Item-level function (assumed)

> The vehicle uses AstraCore-derived forward-camera + radar + lidar fusion to detect objects, lane markings, and distances. The vehicle controller consumes this stream to: (a) warn the driver of imminent collision (FCW), (b) autonomously brake (AEB) when collision is unavoidable by driver reaction, (c) maintain lane position (LKA) at highway speeds.

### 2.2 Operational situations (selected)

| Situation | Speed envelope | Notes |
|---|---|---|
| OS-1.A | Highway cruise | 80–130 km/h, multilane, mixed traffic |
| OS-1.B | Urban arterial | 30–60 km/h, pedestrians + cyclists |
| OS-1.C | Stop-and-go traffic | 0–30 km/h |
| OS-1.D | Adverse weather (rain, fog, low sun) | Sensor performance degraded |

### 2.3 Hazardous events

| ID | Malfunction × situation | Hazardous event | S | E | C | ASIL | Rationale |
|---|---|---|:---:|:---:|:---:|:---:|---|
| HE-1.1 | FN object × OS-1.A | AEB does not trigger; high-closing-speed collision with stopped/slow vehicle ahead | S3 | E4 | C3 | **D** | Closing rate > 80 km/h gives < 2 s reaction window; uncontrollable for most drivers; happens daily during commute (E4). |
| HE-1.2 | FN pedestrian × OS-1.B | AEB does not trigger; pedestrian impact at 30–60 km/h | S3 | E3 | C3 | **C** | Pedestrian fatality ≥ 50 % at 50 km/h impact (S3); urban routes daily (E3 not E4 because not every drive has a pedestrian-conflict event); driver attention is split with other urban tasks (C3). |
| HE-1.3 | FP object × OS-1.A | Spurious AEB activation at highway speed → rear-end collision by following vehicle | S2 | E3 | C2 | **A** | Following vehicles typically maintain headway; severity bounded by multi-vehicle pile-up risk (S2 not S3 because the host-vehicle occupants are restrained); driver of following vehicle has ~1 s reaction window, > 90 % avoid (C2). |
| HE-1.4 | W class (truck mis-classified as small object) × OS-1.A | Wrong AEB calibration applied; insufficient deceleration → collision | S3 | E2 | C3 | **B** | Class-misclassification is rarer than full-FN; less common scenario reduces E to E2; severity and controllability identical to HE-1.1. |
| HE-1.5 | L (object detected late, after FTTI) × OS-1.A | AEB triggers too late to fully arrest closing speed → collision at reduced (but still injurious) speed | S2 | E3 | C3 | **B** | Severity reduced because partial braking happens; events common (E3); uncontrollable by driver in last seconds (C3). |
| HE-1.6 | W lane (drift detected wrongly) × OS-1.A | LKA steers vehicle into adjacent lane (occupied) → side collision | S3 | E2 | C2 | **A** | LKA torque is overridable by attentive driver (C2); severity full because side-impact at highway speed; rarer because requires both wrong-lane detection AND no driver counter-steer (E2). |
| HE-1.7 | FN lane × OS-1.A | LKA disengages silently; vehicle drifts off road | S2 | E3 | C2 | **A** | Off-road departure usually avoidable by driver; bounded by adjacent-terrain severity. |
| HE-1.8 | FN object × OS-1.D (adverse weather) | AEB miss in degraded sensor conditions | S3 | E2 | C3 | **B** | Adverse weather E reduced to E2; otherwise same as HE-1.1. (Item-level safety case may require AEB graceful-degradation or override.) |

### 2.4 Safety goals derived

| ID | Safety Goal | Maps to AstraCore role |
|---|---|---|
| SG-1.1 | The system shall detect on-path obstacles within the FTTI sufficient for AEB to bring the vehicle to a controlled stop in OS-1.A. | Object detection latency budget on AstraCore inference path. **ASIL-D.** |
| SG-1.2 | The system shall correctly classify on-path objects to within the granularity required for AEB calibration in OS-1.A. | Detection class output integrity. **ASIL-B.** |
| SG-1.3 | The system shall not report spurious obstacles on-path in OS-1.A more than once per 10⁵ km of mission. | False-positive rate budget. **ASIL-A.** |
| SG-1.4 | The system shall correctly identify lane markings or report a degraded-confidence flag if it cannot. | Lane-detection confidence integrity. **ASIL-A.** |

Highest item-level ASIL for UC-1: **ASIL-D** (SG-1.1).

---

## 3. UC-2 — Driver Monitoring System (drowsiness + distraction)

### 3.1 Item-level function (assumed)

> The vehicle uses AstraCore-derived ToF/IR camera fusion (`dms_fusion` module) to estimate driver attention state across four levels: ATTENTIVE, DROWSY, DISTRACTED, CRITICAL. CRITICAL or persistent DROWSY triggers escalating warnings; sustained CRITICAL hands over to a brake-to-stop response on a roadside.

### 3.2 Operational situations

| Situation | Notes |
|---|---|
| OS-2.A | Highway cruise (60+ min, low workload) — peak drowsiness window |
| OS-2.B | Urban driving (high workload) — distraction-relevant |
| OS-2.C | Night driving — IR sensor relied on; sun glare absent but condensation possible |

### 3.3 Hazardous events

| ID | Malfunction × situation | Hazardous event | S | E | C | ASIL | Rationale |
|---|---|---|:---:|:---:|:---:|:---:|---|
| HE-2.1 | FN drowsy × OS-2.A | DMS misses drowsy driver; driver falls asleep, vehicle drifts | S3 | E3 | C3 | **C** | Highway-speed off-road / lane-departure with unconscious driver; uncontrollable (C3); E3 because driver-drowsiness episodes are common across the fleet during long drives. |
| HE-2.2 | FN critical × OS-2.A | DMS misses eyes-closed > 2 s; no warning issued | S3 | E2 | C3 | **B** | Eyes-closed > 2 s rarer per-trip than general drowsiness; otherwise same as HE-2.1. |
| HE-2.3 | FP critical × OS-2.B | Spurious CRITICAL warning startles driver in urban traffic | S1 | E3 | C2 | **QM** | Warning is informational; driver controllably reacts. Item-level QM is acceptable; AstraCore *targets* better than QM via TMR + plausibility but no item-level ASIL claim. |
| HE-2.4 | L attention level × OS-2.A | DMS detects DROWSY but reports it after the FTTI; warning arrives after onset of inattention episode | S2 | E3 | C2 | **A** | Late warning still useful; only partial degradation. |
| HE-2.5 | I (flicker between ATTENTIVE / DROWSY) × OS-2.A | Inconsistent state confuses HMI; driver loses trust → ignores future warnings | S2 | E3 | C2 | **A** | Cumulative degradation over many trips; "cried wolf" scenario. |
| HE-2.6 | Sensor-fail (camera stuck) × OS-2.A | DMS goes blind without notification; driver is unmonitored | S3 | E2 | C3 | **B** | Watchdog (`dms_fusion`) catches sensor stuck and asserts SENSOR_FAIL; if THAT mechanism also fails, item is silently unmonitored. The hazardous event presumes the watchdog itself fails — rare (E2) but uncontrollable (C3). |

### 3.4 Safety goals derived

| ID | Safety Goal | Maps to AstraCore role |
|---|---|---|
| SG-2.1 | The system shall detect driver drowsiness within FTTI sufficient to issue an alert before driver loses control. | `dms_fusion` PERCLOS + continuous-closed paths. **ASIL-C.** |
| SG-2.2 | The system shall detect driver eyes-closed > 2 s critical state. | `dms_fusion` CRITICAL state. **ASIL-B.** |
| SG-2.3 | The system shall detect a stuck or failed driver-monitoring camera and signal SENSOR_FAIL within FTTI. | `dms_fusion` watchdog. **ASIL-B.** |
| SG-2.4 | The system shall produce attention-state output free of single-cycle flicker. | IIR temporal smoother in `dms_fusion`. **ASIL-A.** |

Highest item-level ASIL for UC-2: **ASIL-C** (SG-2.1).

---

## 4. UC-3 — Surround view + park assist

### 4.1 Item-level function (assumed)

> The vehicle uses AstraCore-derived 12-ultrasonic + 4-camera + low-range-lidar fusion (`ultrasonic_proximity_alarm` reference example) to drive proximity alarm bands and a stitched surround-view display during low-speed manoeuvring. WARNING level inhibits accelerator; CRITICAL triggers emergency low-speed brake.

### 4.2 Operational situations

| Situation | Notes |
|---|---|
| OS-3.A | Parking lot manoeuvre at < 10 km/h — children, pedestrians, low bollards in blind spot |
| OS-3.B | Reverse out of driveway at < 5 km/h — high probability of pedestrian behind |
| OS-3.C | Hitching trailer / tight industrial space at < 5 km/h |

### 4.3 Hazardous events

| ID | Malfunction × situation | Hazardous event | S | E | C | ASIL | Rationale |
|---|---|---|:---:|:---:|:---:|:---:|---|
| HE-3.1 | FN low obstacle (child, bollard) × OS-3.B | Reverse impact with pedestrian behind vehicle | S3 | E3 | C3 | **C** | Child fatality probable at any reverse speed (S3); reverse-out is daily (E3); driver cannot see directly behind (C3). |
| HE-3.2 | FN low obstacle × OS-3.A | Bumper/door-handle damage on bollard or wall | S1 | E4 | C2 | **A** | Cosmetic / minor structural; events frequent; controllable but mistakes happen. |
| HE-3.3 | FP obstacle × OS-3.A | Spurious WARNING locks accelerator; vehicle stops in cross-traffic lane | S2 | E2 | C2 | **QM** | Cross-traffic at parking-lot speed; bounded severity; driver overrides via continued accelerator request. |
| HE-3.4 | L (alarm fires after impact would occur) × OS-3.B | Effective FN, see HE-3.1 | S3 | E3 | C3 | **C** | Same as HE-3.1 — late vs absent alarm is equivalent for the hazardous event. |
| HE-3.5 | W band (CAUTION reported when CRITICAL warranted) × OS-3.B | Insufficient brake response → impact | S3 | E2 | C3 | **B** | Wrong-band rarer than full FN (cross-sensor disagreement reduces E); same severity/controllability as HE-3.1. |

### 4.4 Safety goals derived

| ID | Safety Goal | Maps to AstraCore role |
|---|---|---|
| SG-3.1 | The system shall detect on-path obstacles within FTTI sufficient for emergency-brake to prevent contact at reverse speeds < 5 km/h. | Cross-sensor proximity fusion (`ultrasonic_proximity_alarm` example pattern). **ASIL-C.** |
| SG-3.2 | The system shall correctly map detected obstacles to alarm bands per the configured speed envelope. | Speed-scaled threshold logic. **ASIL-B.** |
| SG-3.3 | The system shall not report spurious WARNING/CRITICAL bands at rates that erode driver trust. | Cross-sensor confirmation rules. **ASIL-A.** |

Highest item-level ASIL for UC-3: **ASIL-C** (SG-3.1).

---

## 5. Aggregate ASIL determination

| Use case | Highest item-level ASIL | Driver |
|---|:---:|---|
| UC-1 (FCW + AEB + LKA) | **ASIL-D** | SG-1.1 — on-path obstacle detection within FTTI at highway speeds |
| UC-2 (DMS) | **ASIL-C** | SG-2.1 — drowsiness detection FTTI |
| UC-3 (Surround / parking) | **ASIL-C** | SG-3.1 — reverse-speed obstacle detection FTTI |

**Item-level ASIL for the SEooC-bounded set: ASIL-D**, driven entirely by UC-1.

This is consistent with the SEooC §3.2 placeholder ("ASIL-D" for FCW/AEB/LKA item-level functions) — but now backed by an explicit S/E/C derivation rather than a flat assertion.

### 5.1 What this means for AstraCore IP claims

The AstraCore IP block **does not directly hold an ASIL** — ISO 26262 ASIL is an item attribute, not an IP attribute. What AstraCore IP can credibly claim is **"developed for ASIL-D as a Safety Element out of Context"**, meaning:

- Safety mechanisms are designed to support an ASIL-D safety case at the item level
- FMEDA targets are set at the ASIL-D thresholds (SPFM ≥ 99 %, LFM ≥ 90 %, PMHF ≤ 10 FIT)
- The licensee can use the IP to build an ASIL-D item without additional safety-mechanism additions, *provided* the FMEDA + fault-injection + safety-case work in `docs/safety/` reaches those thresholds

Today's measured FMEDA shows we are not yet at ASIL-D (npu_top SPFM 2.08 % vs target 99 %). The remediation plan (`docs/safety/findings_remediation_plan_v0_1.md`) targets **ASIL-B by W12** and **ASIL-D by W18-20**.

### 5.2 Spec sheet rev 1.4 wording (proposed)

Replace:
> ISO 26262 ASIL-D, certified

With:
> **ISO 26262 — Safety Element out of Context (SEooC), designed for ASIL-D.** Item-level ASIL determined by the licensee's HARA per ISO 26262-3 §6; assumed-HARA reference use cases (FCW/AEB/LKA, DMS, surround-view) yield ASIL-D / ASIL-C / ASIL-C respectively. ASIL-B safety case targeted Q4 2026 with TÜV SÜD India pre-engagement underway. ASIL-D safety case extension and licensee-side certification are in scope per the IP Safety Manual and Development Interface Agreement.

---

## 6. Safety Goal → AstraCore IP requirement traceability

This table maps each derived Safety Goal to the AstraCore Assumed Safety Requirement (ASR) in SEooC §4.1 that addresses it. Forms the basis of the safety-case traceability matrix.

| Safety Goal | Item-level ASIL | AstraCore ASR addressed | AstraCore mechanism |
|---|:---:|---|---|
| SG-1.1 (FCW/AEB on-path detection FTTI) | D | ASR-HW-01, ASR-HW-04, ASR-HW-09, ASR-HW-10 | TMR voter + ECC + FMEDA-validated DC |
| SG-1.2 (object class integrity) | B | ASR-HW-01, ASR-HW-09, ASR-SW-02 | TMR + bit-exact compiler/quantiser |
| SG-1.3 (FP object on-path) | A | ASR-SW-02, plausibility checker | Quantiser bit-exactness + range gating |
| SG-1.4 (lane integrity / degraded flag) | A | ASR-HW-07, ASR-HW-05 | Plausibility checker + safe-state |
| SG-2.1 (drowsiness FTTI) | C | ASR-HW-01, ASR-HW-08 | `dms_fusion` PERCLOS path + watchdog |
| SG-2.2 (eyes-closed > 2 s) | B | ASR-HW-01 | `dms_fusion` CRITICAL state, TMR-voted |
| SG-2.3 (SENSOR_FAIL within FTTI) | B | ASR-HW-08 | `dms_fusion` watchdog |
| SG-2.4 (no flicker) | A | ASR-SW-02 | IIR temporal smoother |
| SG-3.1 (reverse obstacle FTTI) | C | ASR-HW-01, ASR-HW-07 | Cross-sensor fusion + plausibility |
| SG-3.2 (alarm band correctness) | B | ASR-SW-03 | YAML-declared safety policies + `astracore.apply` |
| SG-3.3 (FP rate budget) | A | ASR-HW-07 | Plausibility checker |

Reverse traceability (ASR → SG) is tooling-derivable and lives in the FMEDA tool's `--trace` mode (deferred to W11 alongside Safety Case v0.1).

---

## 7. Open items for v0.2

These items must close before this HARA can be approved for external assessment:

1. **Named Track 2 lead, independent reviewer, Safety Manager / Approver** — currently TBD per §0
2. **Workshop with first NDA evaluation licensee** to validate the assumed item context against their actual vehicle program
3. **Quantitative FTTI numbers** per safety goal (currently qualitative — "within FTTI"); needs the licensee's actuator-chain latency model
4. **Operational situations expansion** — v0.1 lists 3 OS per UC; ISO 26262-3 §6.4.1 expects a more systematic enumeration. Cover: weather, time of day, road class, traffic density, geographic region (left-vs-right driving).
5. **Justification documents per S/E/C cell** — v0.1 has one-line rationale; full HARA practice requires per-cell evidence (accident statistics, driver-reaction studies, sensor-degradation curves)
6. **Coverage of UC outside this scope:** highway pilot (L3 with handover), cabin emergency response, V2X
7. **Severe-environment HARA** — extreme heat / cold / vibration profiles per the assumed mission profile (SEooC §3.1)

---

## 8. Verification of HARA (ISO 26262-3 §6.4.5.2 confirmation review)

Required before this document moves from v0.1 to v1.0:

- [ ] Independent reviewer named (per ISO 26262-2 §7)
- [ ] Confirmation review per ISO 26262-2 §7 Table 1 — recommended for ASIL-A, highly recommended for ASIL-B, mandatory for ASIL-C/D
- [ ] Cross-check of S/E/C cells against historical accident data (Euro NCAP / IIHS / NHTSA reports)
- [ ] Cross-check of FTTI assumptions against representative actuator-chain timing (vehicle bus + brake actuator + driver reaction)
- [ ] Sign-off by Safety Manager
- [ ] Distribution to first NDA evaluation licensee for their item-level confirmation

---

## 9. Document control

| Field | Value |
|---|---|
| Document ID | ASTR-SAFETY-HARA-V0.1 |
| Revision | 0.1 |
| Revision date | 2026-04-20 |
| Author | TBD (Track 2 lead) — currently founder + collaborator |
| Reviewer | TBD |
| Approver | TBD |
| Distribution | Internal + TÜV SÜD India + first NDA evaluation licensee |
| Retention | 15 years post-product-discontinuation per ISO 26262-8 §10 |
| Supersedes | None — this is v0.1; replaces the placeholder ASIL claim in SEooC §3.2 |

### 9.1 Revision triggers (parallel to SEooC §9.1)

This HARA is re-issued (with revision bump) on any of:

1. New reference use case added to scope
2. New hazardous event identified during fault-injection or field monitoring
3. S/E/C revision driven by accident statistics review
4. FTTI revision driven by licensee actuator-chain freeze
5. Safety Goal addition or merge
6. Confirmation review feedback that changes any (S, E, C, ASIL) cell
