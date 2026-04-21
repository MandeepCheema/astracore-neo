# Functional Safety Concept (FSC)

**Document ID:** ASTR-SAFETY-FSC-V0.1
**Date:** 2026-04-20
**Standard:** ISO 26262-3:2018 §7 (Functional safety concept)
**Element:** AstraCore Neo NPU + Sensor-Fusion IP block (SEooC)
**Status:** v0.1 — first formal release. Derives FSRs from the 11 Safety Goals enumerated in `docs/safety/hara_v0_1.md` §2.4 / §3.4 / §4.4.
**Classification:** Internal — pre-engagement draft for TÜV SÜD India workshop and first NDA evaluation licensee.
**Author:** TBD (Track 2 lead) — currently founder + collaborator
**Reviewer:** TBD (independent reviewer per ISO 26262-2 §7)
**Approver:** TBD (Safety Manager)

---

## 0. Purpose and framing

ISO 26262-3 §7 requires, after HARA + Safety Goals (`docs/safety/hara_v0_1.md`), the derivation of **Functional Safety Requirements (FSRs)** — the high-level "what the system must do" requirements that satisfy each Safety Goal at functional level, before architectural decomposition into hardware and software.

> **Like HARA, the FSC is an item-level work product.** AstraCore is not the item — the licensee's vehicle program is the item. This document is therefore an **assumed FSC**: it documents the FSRs we *assume* the licensee will derive from the SGs, based on the SEooC reference use cases. The licensee is required by AoU-15 (SEooC §6.4) to perform their own FSC and may revise these assumptions.

### 0.1 Companion documents

- `docs/safety/hara_v0_1.md` — HARA + 11 Safety Goals (this FSC's input)
- `docs/safety/seooc_declaration_v0_1.md` — SEooC + Assumed Safety Requirements (ASRs); FSC's FSRs map down to ASRs
- `docs/safety/iso26262_gap_analysis_v0_1.md` — Part 4 §6 + §7 closure
- `docs/safety/findings_remediation_plan_v0_1.md` — F4 WPs trace from FSRs through ASRs to RTL fixes
- `docs/safety/safety_manual_v0_1.md` — licensee user-guide carrying the AoUs that constrain how this FSC may be applied

### 0.2 What this document is NOT

- **Not a Technical Safety Concept (TSC).** ISO 26262-4 §6 is the next work product after FSC and decomposes FSRs into hardware/software technical safety requirements. The TSC is scheduled for W12-W14 in `docs/best_in_class_design.md` §7.2 once Phase A/B RTL hardening lands.
- **Not architecture.** ISO 26262-4 §7 covers system architecture. This FSC documents only allocation to existing architectural elements (AstraCore IP block / vehicle controller / brake actuator / etc.).
- **Not a vehicle-level safety case.** The vehicle OEM owns that.

---

## 1. Methodology (ISO 26262-3 §7.4)

For each Safety Goal we derive one or more FSRs. Each FSR carries the **inherited ASIL** of its parent SG (per ISO 26262-9 §5 ASIL inheritance) and is specified with the following attributes:

| Attribute | ISO 26262-3 §7.4.2 reference | What it means |
|---|---|---|
| **Operating mode** | §7.4.2.1 a | Vehicle / system mode in which the FSR applies (e.g., highway driving, parking) |
| **Fault Tolerant Time Interval (FTTI)** | §7.4.2.1 b | Maximum time from fault occurrence to safe state without hazard manifesting |
| **Safe state** | §7.4.2.1 c | Defined system state the item enters on FSR violation (e.g., "AEB-disabled with driver-alert active") |
| **Emergency operation interval (FRTI)** | §7.4.2.1 d | Time the item may continue in emergency operation after fault before forced safe-state entry |
| **Functional redundancy strategy** | §7.4.2.2 | Decomposition pattern (e.g., ASIL-D = ASIL-B(D) + ASIL-B(D); single-channel; lockstep) |
| **Allocation** | §7.4.3 | Which architectural element implements (or shares) the FSR (AstraCore IP / vehicle controller / brake / HMI / driver) |
| **Verification approach** | §7.4.4 | How the FSR is shown to be satisfied (FMEDA, fault injection, formal proof, integration test, vehicle test) |

### 1.1 ASIL decomposition convention

Where an FSR derives from an ASIL-D SG and the licensee chooses to apply ASIL decomposition (ISO 26262-9 §5), the convention used in §3 below is:

> **ASIL-D = ASIL-B(D) + ASIL-B(D)** — two independent ASIL-B channels, both qualified for ASIL-D context, summing to ASIL-D for the parent SG.

This decomposition is the **default option** for AstraCore IP integration patterns where the licensee instantiates two AstraCore IP blocks in lockstep (`lockstep_compare_in` boundary signal in SEooC §2.3) or pairs AstraCore with a diverse independent perception stack. Single-channel ASIL-D FSRs are flagged as such.

### 1.2 FTTI placeholders

Quantitative FTTI numbers depend on the licensee's actuator-chain latency model (vehicle bus + brake actuator + driver-reaction time). v0.1 carries **placeholder FTTI values** derived from common automotive-industry references (Bosch ABS specs, NHTSA AEB guidelines, Euro NCAP TR-2024) — the licensee revises with their own actuator profile in their item-level FSC.

| Symbol | Placeholder value | Source |
|---|---|---|
| FTTI_AEB | 100 ms | Total time-budget for FCW/AEB chain (sensor + fusion + decision + brake-actuator). AstraCore detection latency budget is ~30 ms within this envelope. |
| FTTI_LKA | 200 ms | Lane-keep-assist torque-correction chain |
| FTTI_DMS | 500 ms | Drowsiness alert from physiological onset to first warning |
| FTTI_PROX | 50 ms | Reverse-speed obstacle detection (low closing rate, short range) |
| FRTI_AEB | 0 ms (immediate) | AEB has no emergency operation — fault forces immediate safe state |
| FRTI_LKA | 1 s | LKA may continue with degraded confidence for one steering correction cycle |
| FRTI_DMS | 5 s | DMS may continue in degraded mode while SENSOR_FAIL escalates |

---

## 2. Architectural element catalogue

FSRs are allocated to one or more of the following architectural elements. Identifiers are used in the §3 allocation column.

| ID | Element | Owned by |
|---|---|---|
| **A1** | AstraCore NPU compute datapath (`npu_top` + AFUs + SRAM + DMA) | AstraCore IP |
| **A2** | AstraCore sensor-fusion engines (`dms_fusion`, `lane_fusion`, custom-fusion via SDK plugin registries) | AstraCore IP |
| **A3** | AstraCore safety mechanisms (`tmr_voter`, `ecc_secded`, `safe_state_controller`, `plausibility_checker`, `fault_predictor`, watchdog) | AstraCore IP |
| **A4** | AstraCore SDK toolchain (compiler, quantiser, runtime, configuration apply) | AstraCore IP |
| **L1** | Licensee SoC integration (memory PHY, package, clock distribution, supervisor MCU) | Licensee |
| **L2** | Licensee vehicle controller (consumes AstraCore output, makes brake/steer decisions) | Licensee |
| **L3** | Licensee actuator chain (brake-by-wire, steer-by-wire, throttle) | Licensee |
| **L4** | Licensee driver HMI (cluster, audible alerts, haptic seat) | Licensee |
| **D1** | Driver (assumed reaction within nominal envelope) | Vehicle OEM / human-factors design |

---

## 3. Functional Safety Requirements (FSRs)

For brevity, attributes shared across multiple FSRs of the same Safety Goal (e.g., operating mode, allocation) are tabulated once per group.

### 3.1 SG-1.1 (FCW/AEB on-path detection within FTTI) — ASIL-D

**Common attributes for FSR-1.1.\***

| Attribute | Value |
|---|---|
| Operating mode | Highway cruise (OS-1.A) and adverse weather (OS-1.D) per HARA §2.2 |
| FTTI | FTTI_AEB = 100 ms (placeholder) |
| Safe state | AEB pre-armed; vehicle controller transitions to driver-alert + AEB-active or driver-alert + AEB-disabled depending on which FSR fired |
| FRTI | FRTI_AEB = 0 ms (no emergency operation for AEB) |
| Functional redundancy | **ASIL-D = ASIL-B(D) + ASIL-B(D)** via dual-channel: AstraCore camera+radar+lidar fusion (channel A) + licensee independent radar-only AEB fallback (channel B). Single-channel ASIL-D acceptable if licensee qualifies AstraCore alone via FMEDA + fault-injection campaign. |

| FSR ID | Statement | ASIL | Allocation | Verification |
|---|---|:---:|---|---|
| FSR-1.1.1 | The system shall detect on-path obstacles meeting the licensee-defined size + range threshold within FTTI_AEB. | D (or 2× B(D)) | A1 + A2 + L2 | FMEDA on detection path + integration test on labelled OS-1.A dataset |
| FSR-1.1.2 | When AstraCore output confidence drops below the licensee-defined threshold, the system shall report a degraded-perception flag to the vehicle controller within 1 frame. | D | A2 + L2 | Integration test forcing low-confidence output |
| FSR-1.1.3 | On detection of any internal AstraCore fault that affects the AEB perception path, the system shall assert `safe_state_active` (SEooC §2.3) to the licensee supervisor within FDTI documented in the Safety Manual. | D | A3 + L1 | Fault-injection campaigns (`docs/safety/fault_injection/`) |
| FSR-1.1.4 | The system shall maintain detection performance per FSR-1.1.1 across the operational temperature range AoU-1 (–40 °C to +125 °C junction). | D | L1 (silicon characterisation) | Licensee silicon thermal characterisation |
| FSR-1.1.5 | The system shall not report a *false-negative* on-path obstacle for more than two consecutive frames in OS-1.A; the second consecutive miss shall escalate to safe state. | D | A1 + A2 + L2 | Integration test with synthetic occlusion patterns |

**FSR → ASR traceability:** FSR-1.1.1 → ASR-HW-01, ASR-HW-04; FSR-1.1.2 → ASR-HW-07, ASR-SW-04; FSR-1.1.3 → ASR-HW-05, ASR-HW-09, ASR-HW-10; FSR-1.1.4 → ASR-HW-06; FSR-1.1.5 → ASR-HW-09.

### 3.2 SG-1.2 (object class integrity for AEB calibration) — ASIL-B

| FSR ID | Statement | ASIL | Allocation | Verification |
|---|---|:---:|---|---|
| FSR-1.2.1 | The class output of every detected object shall be the same as the bit-exact AstraCore reference compiler output, verified against the Python mirror in `tools/npu_ref/`. | B | A1 + A4 | Bit-exact CI gate (`tests/test_*` pre-existing) + cocotb mirror tests |
| FSR-1.2.2 | The class output shall include a confidence value; the licensee vehicle controller shall reject classes with confidence < licensee-defined threshold and treat as "unknown obstacle" (more conservative AEB calibration). | B | A2 + L2 | Integration test with low-confidence inputs |
| FSR-1.2.3 | A single-event upset on the class-output register shall be detected within 1 cycle. | B | A3 (TMR voter on dms_fusion-style classified outputs) | Fault-injection campaign (`tmr_voter_seu_1k`) |

**FSR → ASR:** FSR-1.2.1 → ASR-SW-02; FSR-1.2.2 → ASR-HW-07; FSR-1.2.3 → ASR-HW-01.

### 3.3 SG-1.3 (FP rate budget on-path) — ASIL-A

| FSR ID | Statement | ASIL | Allocation | Verification |
|---|---|:---:|---|---|
| FSR-1.3.1 | The system shall not report obstacles with range/velocity outside the plausibility-checker rules (`docs/architecture.md` §plausibility). | A | A2 (`plausibility_checker`) | Plausibility-checker FMEDA + cocotb tests |
| FSR-1.3.2 | The integrated false-positive rate over a calibration corpus shall be ≤ 1 per 10⁵ km of mission profile. | A | A2 + L2 | Vehicle-test corpus on licensee programme |

**FSR → ASR:** FSR-1.3.1 → ASR-HW-07; FSR-1.3.2 → integration test scope.

### 3.4 SG-1.4 (lane integrity / degraded flag) — ASIL-A

| FSR ID | Statement | ASIL | Allocation | Verification |
|---|---|:---:|---|---|
| FSR-1.4.1 | When AstraCore lane-detection confidence drops below threshold (e.g., faded markings, glare), the system shall assert a degraded-confidence flag to the licensee LKA controller within FTTI_LKA. | A | A2 (`lane_fusion`) + L2 | Adverse-weather integration test |
| FSR-1.4.2 | A single-event upset in `lane_fusion` outputs shall be detected and reported via the SEooC §2.3 `fault_detected[]` bitfield. | A | A3 (TMR-style on lane_fusion outputs — F4 hardening WP candidate) | Fault-injection campaign (lane_fusion follow-up; not in W3-W4 batch) |

**FSR → ASR:** FSR-1.4.1 → ASR-HW-07, ASR-HW-05; FSR-1.4.2 → ASR-HW-01.

### 3.5 SG-2.1 (drowsiness FTTI) — ASIL-C

| FSR ID | Statement | ASIL | Allocation | Verification |
|---|---|:---:|---|---|
| FSR-2.1.1 | The PERCLOS path in `dms_fusion` shall accumulate drowsy-frame counts and assert DROWSY level within FTTI_DMS = 500 ms. | C | A2 (`dms_fusion`) | Integration test with synthetic eye-closure pattern; unit test `test_dms_fusion_drowsy_path` (existing) |
| FSR-2.1.2 | Drowsy-state output shall be temporally smoothed (IIR ~0.75 prev + 0.25 new) so single-frame perturbations do not trip a level transition. | C | A2 (IIR in `dms_fusion`) | FMEDA mechanism `iir_self_correcting` validated by `dms_fusion_inj_5k` campaign |

**FSR → ASR:** FSR-2.1.1 → ASR-HW-01, ASR-HW-08; FSR-2.1.2 → ASR-SW-02.

### 3.6 SG-2.2 (eyes-closed > 2 s) — ASIL-B

| FSR ID | Statement | ASIL | Allocation | Verification |
|---|---|:---:|---|---|
| FSR-2.2.1 | The cont_closed counter shall trigger CRITICAL state within `CLOSED_CRIT_FRAMES` (default 60 frames @ 30 fps = 2 s). | B | A2 | Integration test with sustained closed-eye pattern |
| FSR-2.2.2 | A single-event upset on the CRITICAL state output (TMR-voted) shall be masked by the TMR voter; lane-fault diagnostic shall route to the licensee supervisor via the `tmr_disagree_count` boundary signal. | B | A3 (`tmr_voter` + boundary signal per SEooC §2.3) | TMR fault-injection campaign |

**FSR → ASR:** FSR-2.2.1 → ASR-HW-01; FSR-2.2.2 → ASR-HW-01, ASR-HW-09.

### 3.7 SG-2.3 (SENSOR_FAIL within FTTI) — ASIL-B

| FSR ID | Statement | ASIL | Allocation | Verification |
|---|---|:---:|---|---|
| FSR-2.3.1 | Per-sensor watchdog timeout shall be set such that watchdog → SENSOR_FAIL latency ≤ FTTI_DMS − licensee-defined signaling latency. | B | A2 (per-sensor `WATCHDOG_CYCLES` parameter) + L2 | Watchdog timeout test + licensee actuator-chain timing model |
| FSR-2.3.2 | SENSOR_FAIL state shall route via the SEooC §2.3 `safe_state_active` boundary signal to the licensee supervisor. | B | A3 + L1 | Integration test with stuck sensor input |

**FSR → ASR:** FSR-2.3.1 → ASR-HW-08; FSR-2.3.2 → ASR-HW-05.

### 3.8 SG-2.4 (no flicker) — ASIL-A

| FSR ID | Statement | ASIL | Allocation | Verification |
|---|---|:---:|---|---|
| FSR-2.4.1 | The IIR temporal smoother shall attenuate single-frame perturbations of the fused score by ≥ 75 % per gaze_valid pulse. | A | A2 (IIR in `dms_fusion`) | Cocotb test on IIR convergence (existing) |

**FSR → ASR:** FSR-2.4.1 → ASR-SW-02.

### 3.9 SG-3.1 (reverse obstacle FTTI) — ASIL-C

| FSR ID | Statement | ASIL | Allocation | Verification |
|---|---|:---:|---|---|
| FSR-3.1.1 | The cross-sensor (US + lidar) confirmation rule (per `examples/ultrasonic_proximity_alarm.py` reference) shall raise CRITICAL within FTTI_PROX = 50 ms when both sensors report obstacle within configured proximity. | C | A2 (custom-fusion via SDK plugin registry) + L2 | Integration test with synthetic obstacle convergence |
| FSR-3.1.2 | The CRITICAL band shall be speed-independent (always raised when ANY sensor reports obstacle within ~0.3 m, regardless of vehicle speed). | C | A4 (SDK config) | Unit test on the proximity-alarm reference logic (`tests/test_ultrasonic_proximity_alarm.py` follow-up) |

**FSR → ASR:** FSR-3.1.1 → ASR-HW-01, ASR-HW-07; FSR-3.1.2 → ASR-SW-03.

### 3.10 SG-3.2 (alarm band correctness) — ASIL-B

| FSR ID | Statement | ASIL | Allocation | Verification |
|---|---|:---:|---|---|
| FSR-3.2.1 | Alarm bands (OFF / CAUTION / WARNING / CRITICAL) shall use speed-scaled thresholds per the `safety_policies` declared in the licensee's YAML configuration applied via `astracore configure --apply`. | B | A4 + L2 | Unit + integration tests on the apply path (`tests/test_apply.py`) |
| FSR-3.2.2 | The licensee shall not be permitted to configure thresholds that violate the safety-policy validation rules in `astracore.config`. | B | A4 | Schema-validation unit tests |

**FSR → ASR:** FSR-3.2.1 → ASR-SW-03; FSR-3.2.2 → ASR-SW-03 (policy enforcement).

### 3.11 SG-3.3 (FP rate budget — proximity alarms) — ASIL-A

| FSR ID | Statement | ASIL | Allocation | Verification |
|---|---|:---:|---|---|
| FSR-3.3.1 | WARNING and CRITICAL bands shall require cross-sensor confirmation (US + lidar agreement within 0.5 m) before assertion. | A | A2 (custom-fusion logic) | Integration test with single-sensor false echo |
| FSR-3.3.2 | The plausibility-checker `MIN_CONFIDENCE` parameter shall reject low-confidence detections from contributing to alarm-band escalation. | A | A2 (`plausibility_checker`) | FMEDA on plausibility_checker + integration test |

**FSR → ASR:** FSR-3.3.1 → ASR-HW-07; FSR-3.3.2 → ASR-HW-07.

---

## 4. Aggregate FSR → ASR coverage matrix

This matrix verifies that every Assumed Safety Requirement (ASR) declared in SEooC §4.1 has at least one FSR that exercises it. ASRs that are **not yet** addressed by any FSR are flagged for v0.2 follow-up.

| ASR (from SEooC §4.1) | Addressed by FSR(s) | ASIL claim |
|---|---|---|
| ASR-HW-01 (single-bit transient SEU within FDTI ≤ 10 µs) | FSR-1.1.1, FSR-1.2.3, FSR-1.4.2, FSR-2.1.1, FSR-2.2.2, FSR-3.1.1 | D |
| ASR-HW-02 (SECDED single-bit correct in SRAM) | (covered by F4-A-1.1 npu_top integration; FSR follow-up needed in v0.2 once F4-A-1.1 lands) | D |
| ASR-HW-03 (SECDED double-bit detect) | (same as ASR-HW-02; v0.2 follow-up) | D |
| ASR-HW-04 (stuck-at coverage in datapath) | FSR-1.1.1, FSR-1.1.5 | B (LBIST is Phase B WP F4-B-3) |
| ASR-HW-05 (drive IP block to safe state on aggregated fault) | FSR-1.1.3, FSR-1.4.1 (degraded flag), FSR-2.3.2 | D |
| ASR-HW-06 (clock loss / glitch / freq-out-of-bounds) | FSR-1.1.4 | B (clock monitor RTL is gap noted in SEooC §5; F4 follow-up) |
| ASR-HW-07 (implausible sensor inputs) | FSR-1.1.2, FSR-1.3.1, FSR-1.4.1, FSR-3.1.1, FSR-3.3.1, FSR-3.3.2 | B |
| ASR-HW-08 (sensor-stuck within configured watchdog timeout) | FSR-2.1.1, FSR-2.3.1 | B |
| ASR-HW-09 (SPFM ≥ 90 % aggregate) | FSR-1.1.3, FSR-1.1.5, FSR-2.2.2 | B (validated by aggregate FMEDA roll-up at W10) |
| ASR-HW-10 (LFM ≥ 60 % aggregate) | FSR-1.1.3 | B (same as ASR-HW-09) |
| ASR-HW-11 (PMHF ≤ 100 FIT ASIL-B / ≤ 10 FIT ASIL-D) | (aggregate FMEDA roll-up at W10; not addressed by an FSR — this is a property of the aggregate FMEDA) | B / D |
| ASR-HW-12 (secure boot for weights + config) | (Track 3 OpenTitan integration FSR follow-up in v0.2) | B |
| ASR-HW-13 (weight tampering detection) | (Track 3 OpenTitan integration; v0.2 FSR follow-up) | B |
| ASR-SW-01 (compiler shall not silently change semantics) | (covered by Tool Confidence Level evaluation per SEooC §4.3 ASR-PROC-04; not addressed by an FSR) | QM |
| ASR-SW-02 (quantiser bit-exactness) | FSR-1.2.1, FSR-2.1.2, FSR-2.4.1 | B |
| ASR-SW-03 (runtime enforces safety policies) | FSR-3.1.2, FSR-3.2.1, FSR-3.2.2 | B |
| ASR-SW-04 (safety-relevant event logging) | FSR-1.1.2 (degraded-perception flag) | B |
| ASR-SW-05 (C++ runtime per MISRA-C + ISO 26262-6) | (Track 1 future deliverable; v0.2 FSR follow-up) | B |
| ASR-PROC-01 to ASR-PROC-06 (process requirements) | (organisational; not addressed by an FSR) | n/a |

**ASRs without FSR coverage in v0.1** (must add in v0.2):
- ASR-HW-02, ASR-HW-03 — wait for F4-A-1.1 npu_top + ECC wrapper integration
- ASR-HW-12, ASR-HW-13 — wait for Track 3 OpenTitan integration
- ASR-SW-05 — wait for C++ runtime work package (F1-B3)

---

## 5. Preliminary architecture (textual)

ISO 26262-3 §7.4.5 requires a preliminary architecture sketch showing how FSRs are realised. The detailed architecture lives in `docs/astracore_v2_npu_architecture.md` and `docs/sensor_fusion_architecture.md`; this section indexes the safety-relevant subset.

```
              ┌──────────────────────────────────────┐
              │  Sensors (camera/radar/lidar/IMU/    │
              │   US/GNSS/ToF)  — licensee L1        │
              └──────┬─────────────────┬─────────────┘
                     │                 │
            ┌────────▼────────┐  ┌─────▼──────┐
            │  Sensor I/O      │  │ Plausi-    │  AstraCore A2
            │  (RTL)           │  │ bility     │
            └────────┬─────────┘  │ checker    │
                     │            └─────┬──────┘
            ┌────────▼─────────┐        │
            │  Fusion (DMS,    │◄───────┘
            │  lane, custom)   │
            └────────┬─────────┘
                     │
            ┌────────▼─────────┐
            │  NPU compute     │  AstraCore A1
            │  (npu_top)       │
            └────────┬─────────┘
                     │
            ┌────────▼─────────┐
            │  Safety mech.    │  AstraCore A3
            │  (TMR, ECC,      │
            │  safe_state,     │
            │  watchdogs)      │
            └────────┬─────────┘
              fault_detected[15:0]
              safe_state_active
              ECC counters
              TMR disagree counter
                     │
            ┌────────▼─────────┐
            │  Licensee        │  L1 + L2
            │  supervisor MCU  │
            │  (SEooC §2.3     │
            │  boundary)       │
            └────────┬─────────┘
                     │
            ┌────────▼─────────┐
            │  Vehicle         │  L2
            │  controller      │
            │  (AEB / LKA /    │
            │   driver alert)  │
            └────────┬─────────┘
                     │
                     ▼
            Actuators L3 + HMI L4
```

Key safety boundary: the **SEooC §2.3 boundary signals** are the
contract between AstraCore IP and the licensee SoC. Every FSR
allocated to A1/A2/A3 ultimately surfaces fault information to the
licensee via this boundary; the licensee's supervisor implements the
item-level safety case.

---

## 6. Open items for v0.2

These items must close before this FSC can be approved for external assessment:

1. **Named Track 2 lead, independent reviewer, Safety Manager / Approver** — currently TBD per §0
2. **Workshop with first NDA evaluation licensee** to validate the assumed item context and FTTI placeholders against their actual vehicle program
3. **Quantitative FTTI numbers** per FSR — currently placeholders in §1.2
4. **FSRs for ASR-HW-02, ASR-HW-03** — wait for F4-A-1.1 (`npu_top` instantiation swap to ECC wrapper) to land
5. **FSRs for ASR-HW-12, ASR-HW-13** — wait for Track 3 OpenTitan crypto integration
6. **FSR for ASR-SW-05** — wait for C++ runtime WP (F1-B3)
7. **Integration test plan** ✅ shipped 2026-04-20 at `docs/safety/integration_test_plan_v0_1.md` — 25 ITs across the 25 FSRs; 9 READY, 7 STUBBED, 5 BLOCKED on F4 work, 4 LICENSEE-allocated
8. **TSC kickoff** ✅ shipped 2026-04-20 at `docs/safety/technical_safety_concept_v0_1.md` — 38 TSRs derived; 15 READY / 8 STUBBED / 9 BLOCKED on F4 / 6 LICENSEE+performance

---

## 7. Verification of FSC (ISO 26262-2 §7 confirmation review)

Required before this document moves from v0.1 to v1.0:

- [ ] Independent reviewer named (per ISO 26262-2 §7)
- [ ] Confirmation review per ISO 26262-2 §7 Table 1 — recommended for ASIL-A, highly recommended for ASIL-B, mandatory for ASIL-C/D (we have ASIL-D FSRs → mandatory)
- [ ] Cross-check that every Safety Goal in HARA §2.4/§3.4/§4.4 has at least one FSR
- [ ] Cross-check that every FSR is allocated to at least one architectural element
- [ ] Cross-check that every FSR has a verification approach
- [ ] Cross-check that the FSR → ASR coverage matrix (§4) accounts for every ASR in SEooC §4.1
- [ ] Sign-off by Safety Manager
- [ ] Distribution to first NDA evaluation licensee for their item-level confirmation

---

## 8. Document control

| Field | Value |
|---|---|
| Document ID | ASTR-SAFETY-FSC-V0.1 |
| Revision | 0.1 |
| Revision date | 2026-04-20 |
| Author | TBD (Track 2 lead) — currently founder + collaborator |
| Reviewer | TBD |
| Approver | TBD |
| Distribution | Internal + TÜV SÜD India + first NDA evaluation licensee |
| Retention | 15 years post-product-discontinuation per ISO 26262-8 §10 |
| Supersedes | None — this is v0.1; closes ISO 26262 gap analysis Part 4 §6 + §7 priority items |

### 8.1 Revision triggers

This FSC is re-issued (with revision bump) on any of:

1. New Safety Goal added to HARA → must spawn new FSR(s)
2. New ASR added to SEooC §4.1 → must check FSR coverage in §4 matrix
3. FTTI revision driven by licensee actuator-chain freeze
4. New architectural element added or removed (e.g., crypto block lands → A3 expands)
5. F4 remediation phase milestone closes that affects an FSR's verification approach (e.g., F4-A-1.1 lands → ASR-HW-02/03 FSRs added)
6. Confirmation review feedback that changes any FSR statement, attribute, or allocation
7. TSC (ISO 26262-4 §6) elaborates FSRs into HW/SW technical safety requirements — FSC is then rev-bumped to align with TSC v1.0
