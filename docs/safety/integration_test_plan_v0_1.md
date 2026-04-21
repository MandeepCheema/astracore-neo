# Integration Test Plan (per-FSR)

**Document ID:** ASTR-SAFETY-ITP-V0.1
**Date:** 2026-04-20
**Standard:** ISO 26262-4:2018 §9 (System integration & testing) + ISO 26262-3:2018 §7.4.4 (FSR verification approach)
**Element:** AstraCore Neo NPU + Sensor-Fusion IP block (SEooC)
**Status:** v0.1 — first formal release. Closes the open item flagged in `docs/safety/functional_safety_concept_v0_1.md` §6.7. Companion to Safety Manual §11 (licensee-side verification).
**Classification:** Internal — pre-engagement draft for TÜV SÜD India workshop and first NDA evaluation licensee.
**Author:** TBD (Track 2 lead) — currently founder + collaborator
**Reviewer:** TBD (independent reviewer per ISO 26262-2 §7)
**Approver:** TBD (Safety Manager)

---

## 0. Purpose and framing

Each Functional Safety Requirement (FSR) in `docs/safety/functional_safety_concept_v0_1.md` §3 carries a one-line "Verification approach" column. This document operationalises that column: for each FSR, it specifies one or more **integration tests** with concrete setup, stimulus, expected response, and pass/fail criteria.

> **Two layers of test coverage** at the safety case:
>
> 1. **Unit / module tests** — the 1352 Python tests + cocotb gates that validate per-module RTL semantics. These run pre-integration. Already in place; see `tests/`.
> 2. **Integration tests** — the tests in *this* document. These run on the integrated SoC (FPGA + cocotb today; silicon at licensee tape-out). They verify that the FSRs hold *across module boundaries*, not just within a single module.
>
> The FMEDA + fault-injection campaigns are evidence; integration tests are claim verification.

### 0.1 Companion documents

- `docs/safety/functional_safety_concept_v0_1.md` — 25 FSRs this test plan operationalises
- `docs/safety/safety_manual_v0_5.md` §11 — licensee-side minimum verification activities (this doc is the per-FSR breakdown)
- `docs/safety/seooc_declaration_v0_1.md` §2.3 — boundary signals each test exercises
- `docs/safety/findings_remediation_plan_v0_1.md` — F4 WPs that gate certain integration tests
- `sim/fault_injection/campaigns/` — fault-injection campaigns providing stimulus for many tests

---

## 1. Methodology

### 1.1 Per-FSR test attributes

Each integration test (IT) carries:

| Attribute | What it captures |
|---|---|
| **ID** | `IT-<FSR>.<seq>` — e.g., IT-1.1.1.1 maps to the first integration test of FSR-1.1.1 |
| **Name** | Short human-readable label |
| **Parent FSR** | The FSC FSR this test verifies (one parent only; multi-FSR tests get a separate IT per parent) |
| **ASIL** | Inherited from parent FSR |
| **Setup / preconditions** | What must be in place before stimulus (RTL baseline, model loaded, configuration applied) |
| **Stimulus** | What the test driver does (input pattern, fault injection, configuration change) |
| **Expected response** | What AstraCore must do — typically observable on a SEooC §2.3 boundary signal or a tracked counter |
| **Pass/fail criteria** | Quantitative thresholds (latency in ms, count, bit-exact equality) |
| **Environment** | One of: `SIM` (Verilator + cocotb), `FPGA` (AWS F1 or Artix-7), `SILICON` (post-tape-out at Licensee), or `SDK` (Python-only on host) |
| **Required tools** | What's needed to run (e.g., cocotb 2.0.1, Verilator 5.030, fault-injection harness, custom testbench) |
| **Allocation** | One of: `S` (Supplier ships the test + reference; Licensee can re-run), `L` (Licensee authors per their item), `J` (Joint — Supplier ships skeleton; Licensee fills in item-specific values) |
| **Status** | `READY` (test exists today and runs), `STUBBED` (sketch exists; needs filling out), `BLOCKED` (waits for an F4 WP), `LICENSEE` (Licensee writes from scratch) |

### 1.2 Test environment summary

| Environment | When used | Tooling |
|---|---|---|
| **SIM** | Per-PR regression; immediate developer feedback; bit-exact comparisons | Verilator 5.030 + cocotb 2.0.1 (WSL) |
| **FPGA** | Silicon-scale validation pre-tape-out | AWS F1 VU9P (4096 MACs cap) per F1-F1..F3 WPs |
| **SILICON** | Post-tape-out validation at Licensee | Licensee's silicon program; AstraCore advisory |
| **SDK** | Python-only validation; runs on Windows / Linux / macOS without cocotb | pytest + numpy + onnx |

### 1.3 Pass/fail aggregation

Each test in §3 has independent pass/fail criteria. **A FSR is verified** if all of its child integration tests pass; **a Safety Goal is verified** if all of its child FSRs are verified.

The integration test result feeds the **safety case verification matrix** (Safety Case v1.0, planned W11). Failed integration tests block the corresponding FSR's verification status, which blocks the parent SG, which blocks the overall ASIL claim.

---

## 2. Test environment requirements

### 2.1 Hardware

| Item | Required for | Provided by |
|---|---|---|
| WSL Ubuntu 22.04 VM (or native Linux) | SIM environment | Developer / CI |
| Verilator 5.030 + cocotb 2.0.1 + PyYAML | SIM environment | `tools/wsl_install_verilator.sh` |
| AWS F1 instance (f1.2xlarge or larger) | FPGA environment | AstraCore F1-F1..F3 program |
| Artix-7 dev board (XC7A100T or larger) | FPGA fallback | Optional |
| Licensee target silicon | SILICON environment | Licensee tape-out |

### 2.2 Software baseline

| Item | Version | Reference |
|---|---|---|
| Python | 3.10 / 3.11 / 3.12 | `pyproject.toml` |
| pytest | ≥ 9.0 | `pyproject.toml [optional-dependencies]` |
| numpy | ≥ 1.26 | `pyproject.toml` |
| onnx | ≥ 1.14 | `pyproject.toml` |
| onnxruntime | ≥ 1.16 | `pyproject.toml` |
| pyyaml | ≥ 6.0 | `pyproject.toml` |

---

## 3. Per-FSR integration tests

Tests below cover each of the 25 FSRs in FSC §3. Where multiple tests exist for a single FSR, each is listed separately.

### 3.1 SG-1.1 — FCW/AEB on-path detection (ASIL-D)

#### IT-1.1.1.1 — On-path obstacle detection within FTTI

| Field | Value |
|---|---|
| Parent FSR | FSR-1.1.1 |
| ASIL | D |
| Setup | Load yolov8n model via `astracore.compile`; configure `astracore configure --apply examples/tier1_adas.yaml --backend npu-sim`; load reference video clip from OS-1.A (highway cruise) |
| Stimulus | Replay the clip through the integrated pipeline at 30 fps |
| Expected response | Detection latency from frame ingest to `bounding_box_out` ≤ 30 ms (AstraCore budget within FTTI_AEB = 100 ms) |
| Pass/fail | Latency P99 ≤ 30 ms across all frames; recall ≥ 95 % on labelled obstacles |
| Environment | SIM (today); FPGA (post F1-F3); SILICON (post tape-out) |
| Required tools | `astracore` SDK + cocotb harness + labelled OS-1.A test clip |
| Allocation | S — ships clip + harness; L re-runs on integrated SoC |
| Status | STUBBED — labelled OS-1.A clip needs assembly (tracked as IT-asset WP) |

#### IT-1.1.2.1 — Degraded-perception flag on low confidence

| Field | Value |
|---|---|
| Parent FSR | FSR-1.1.2 |
| ASIL | D |
| Setup | Same as IT-1.1.1.1 |
| Stimulus | Inject a sequence of frames with adverse weather / occlusion that drives confidence below the licensee-defined threshold (default 0.40) |
| Expected response | `degraded_perception` flag bit asserted in `fault_detected[]` within 1 frame (~33 ms @ 30 fps) |
| Pass/fail | Flag asserts on every frame where confidence < threshold; deasserts on next frame above threshold; no false positives during clear-weather frames |
| Environment | SIM; FPGA |
| Required tools | `astracore` SDK + adverse-weather clip |
| Allocation | S |
| Status | STUBBED |

#### IT-1.1.3.1 — Internal fault → safe_state_active

| Field | Value |
|---|---|
| Parent FSR | FSR-1.1.3 |
| ASIL | D |
| Setup | Integrated SoC running yolov8n; `safe_state_active` wired to monitored output |
| Stimulus | Inject TMR triple-disagree on `dms_fusion` lanes via fault-injection harness (`sim/fault_injection/campaigns/tmr_voter_seu_1k.yaml` or extension) |
| Expected response | `safe_state_active` asserts within FDTI = 1 clk + propagation delay through `safe_state_controller` |
| Pass/fail | Assertion observed within 10 clks; `fault_detected[0]` set; `tmr_disagree_count` increments |
| Environment | SIM (cocotb fault-injection); FPGA (manual lane perturbation harness) |
| Required tools | `sim/fault_injection/runner.py` + custom multi-lane injection campaign |
| Allocation | S — ships campaign; L re-runs on integrated SoC |
| Status | BLOCKED on multi-lane injection extension to fault-injection harness (W4 follow-up) |

#### IT-1.1.4.1 — Detection performance across temperature range

| Field | Value |
|---|---|
| Parent FSR | FSR-1.1.4 |
| ASIL | D |
| Setup | Silicon thermal chamber; load yolov8n |
| Stimulus | Sweep junction temperature across –40 °C → +125 °C; replay reference clip at each temperature |
| Expected response | Detection latency + recall maintained per IT-1.1.1.1 thresholds across all temperatures |
| Pass/fail | Same thresholds as IT-1.1.1.1, valid at every sampled temperature |
| Environment | SILICON (Licensee silicon program; AstraCore IP cannot self-validate temperature) |
| Required tools | Licensee thermal chamber + telemetry harness |
| Allocation | L — Licensee runs as part of silicon characterisation |
| Status | LICENSEE |

#### IT-1.1.5.1 — Two-frame consecutive miss escalates to safe state

| Field | Value |
|---|---|
| Parent FSR | FSR-1.1.5 |
| ASIL | D |
| Setup | Same as IT-1.1.1.1 |
| Stimulus | Inject two consecutive frames with synthetic occlusion such that detection misses for both |
| Expected response | After 2nd consecutive miss, `safe_state_active` asserts within 1 frame |
| Pass/fail | `fault_detected[8]` (NPU compute fault aggregated) bit set; `safe_state_active` asserted |
| Environment | SIM; FPGA |
| Required tools | `astracore` SDK + synthetic occlusion patterns |
| Allocation | S |
| Status | BLOCKED on F4-A Phase A (npu_top fault aggregation must populate fault_detected[8]) |

### 3.2 SG-1.2 — Object class integrity (ASIL-B)

#### IT-1.2.1.1 — Class output bit-exactness vs Python mirror

| Field | Value |
|---|---|
| Parent FSR | FSR-1.2.1 |
| ASIL | B |
| Setup | Load yolov8n; run inference on labelled COCO-128 eval set |
| Stimulus | For each input frame, run via cocotb (Verilator) AND via Python reference in parallel |
| Expected response | Class outputs bit-exactly identical between the two paths |
| Pass/fail | Zero mismatches across the eval set (28 frames currently; expand to 500+ at production) |
| Environment | SIM |
| Required tools | cocotb + Verilator + Python reference (`tools/npu_ref/`) |
| Allocation | S — already runs; existing test `tests/test_npu_top_yolo_match.py` |
| Status | READY ✅ |

#### IT-1.2.2.1 — Low-confidence rejection by vehicle controller

| Field | Value |
|---|---|
| Parent FSR | FSR-1.2.2 |
| ASIL | B |
| Setup | Integrated SoC; vehicle-controller stub configured with confidence threshold |
| Stimulus | Inject frames with detection confidence below threshold |
| Expected response | Vehicle controller treats detection as "unknown obstacle" (more conservative AEB calibration) |
| Pass/fail | Vehicle controller's downstream decision matches the conservative path |
| Environment | FPGA / SILICON |
| Required tools | Licensee vehicle-controller stub |
| Allocation | L |
| Status | LICENSEE |

#### IT-1.2.3.1 — Single-event upset on class-output detected within 1 cycle

| Field | Value |
|---|---|
| Parent FSR | FSR-1.2.3 |
| ASIL | B |
| Setup | Run `tmr_voter_seu_1k` campaign on the dms_fusion or class-output module |
| Stimulus | Per-bit SEU on lane registers (per `sim/fault_injection/campaigns/tmr_voter_seu_1k.yaml`) |
| Expected response | `tmr_fault` asserts within 1 clk; voted output remains correct |
| Pass/fail | ≥ 99 % detection coverage |
| Environment | SIM |
| Required tools | Fault-injection harness |
| Allocation | S |
| Status | READY (campaign shipped; needs WSL run for measurement) |

### 3.3 SG-1.3 — On-path FP rate (ASIL-A)

#### IT-1.3.1.1 — Plausibility-checker rejection of out-of-range obstacles

| Field | Value |
|---|---|
| Parent FSR | FSR-1.3.1 |
| ASIL | A |
| Setup | Integrated SoC with `plausibility_checker` enabled per default policy |
| Stimulus | Inject synthesised obstacles with range/velocity outside the policy bounds |
| Expected response | `plausibility_checker.check_ok = 0` for each violator; obstacle dropped from downstream stream |
| Pass/fail | 100 % rejection of in-violation obstacles; ≤ 0.1 % false rejection of compliant obstacles |
| Environment | SIM; FPGA |
| Required tools | Synthetic obstacle generator |
| Allocation | S |
| Status | STUBBED — synthetic obstacle generator needs implementation |

#### IT-1.3.2.1 — Integrated FP rate over calibration corpus

| Field | Value |
|---|---|
| Parent FSR | FSR-1.3.2 |
| ASIL | A |
| Setup | Integrated SoC + Licensee vehicle-test corpus |
| Stimulus | Replay corpus through full perception pipeline |
| Expected response | False-positive on-path obstacle rate ≤ 1 per 10⁵ km of mission profile |
| Pass/fail | Threshold met across corpus |
| Environment | SILICON (real vehicle test; AstraCore IP can simulate but FP rate is item-level metric) |
| Required tools | Licensee vehicle-test harness |
| Allocation | L |
| Status | LICENSEE |

### 3.4 SG-1.4 — Lane integrity (ASIL-A)

#### IT-1.4.1.1 — Degraded-confidence flag on faded lane markings

| Field | Value |
|---|---|
| Parent FSR | FSR-1.4.1 |
| ASIL | A |
| Setup | Load yolov8n + lane-detection extension |
| Stimulus | Replay clip with progressive lane-marking fade (synthetic alpha blend or natural fading) |
| Expected response | Lane-confidence flag drops; LKA degraded-confidence bit asserts within FTTI_LKA = 200 ms |
| Pass/fail | Flag asserts within budget on every faded frame |
| Environment | SIM; FPGA |
| Required tools | Lane-fade clip generator |
| Allocation | S |
| Status | STUBBED |

#### IT-1.4.2.1 — SEU on lane_fusion outputs detected via fault_detected[]

| Field | Value |
|---|---|
| Parent FSR | FSR-1.4.2 |
| ASIL | A |
| Setup | `lane_fusion` instantiated in test harness; fault-injection harness extended to lane_fusion outputs |
| Stimulus | SEU bit-flip on lane_fusion output regs |
| Expected response | Corresponding fault_detected[] bit asserts within 1 clk |
| Pass/fail | ≥ 95 % detection coverage |
| Environment | SIM |
| Required tools | Fault-injection harness extended to lane_fusion |
| Allocation | S |
| Status | BLOCKED on lane_fusion fault-injection campaign (W6+ per FMEDA schedule) |

### 3.5 SG-2.1 — Drowsiness detection FTTI (ASIL-C)

#### IT-2.1.1.1 — PERCLOS path triggers DROWSY within FTTI_DMS

| Field | Value |
|---|---|
| Parent FSR | FSR-2.1.1 |
| ASIL | C |
| Setup | Drive `dms_fusion` with synthetic eye-closure pattern (eye_state=CLOSED for sustained duration) |
| Stimulus | Hold eye_state=CLOSED for 30 frames at 30 fps (1 s) |
| Expected response | `driver_attention_level` transitions to DROWSY within FTTI_DMS = 500 ms after pattern onset |
| Pass/fail | Transition observed within 500 ms ± 1 frame |
| Environment | SIM |
| Required tools | `tb_dms_fusion_fi.sv` testbench (already shipped); cocotb stimulus driver |
| Allocation | S |
| Status | READY (testbench shipped; needs WSL cocotb run) |

#### IT-2.1.2.1 — IIR temporal smoother attenuates single-frame perturbations

| Field | Value |
|---|---|
| Parent FSR | FSR-2.1.2 |
| ASIL | C |
| Setup | Drive `dms_fusion` with steady ATTENTIVE pattern |
| Stimulus | Inject single-frame eye_state=CLOSED |
| Expected response | `driver_attention_level` does NOT transition to DROWSY (IIR attenuates the single-frame perturbation by ~75 %) |
| Pass/fail | Level remains ATTENTIVE for the duration; existing test `tests/test_dms_fusion_iir_smooth.py` covers |
| Environment | SIM |
| Required tools | Existing test infrastructure |
| Allocation | S |
| Status | READY ✅ |

### 3.6 SG-2.2 — Eyes-closed > 2 s (ASIL-B)

#### IT-2.2.1.1 — cont_closed counter triggers CRITICAL at threshold

| Field | Value |
|---|---|
| Parent FSR | FSR-2.2.1 |
| ASIL | B |
| Setup | Drive dms_fusion with eye_state=CLOSED for ≥ CLOSED_CRIT_FRAMES (60 frames default) |
| Stimulus | Hold eye_state=CLOSED for 70 frames |
| Expected response | `driver_attention_level` transitions to CRITICAL at frame 60 ± 1 |
| Pass/fail | Transition observed |
| Environment | SIM |
| Required tools | Existing test infrastructure |
| Allocation | S |
| Status | READY ✅ |

#### IT-2.2.2.1 — TMR voter masks single-lane SEU on CRITICAL output

| Field | Value |
|---|---|
| Parent FSR | FSR-2.2.2 |
| ASIL | B |
| Setup | Drive dms_fusion to CRITICAL state; arm fault-injection on dal_a/b/c lanes |
| Stimulus | Inject SEU on dal_a per `tmr_voter_seu_1k` |
| Expected response | Voted `driver_attention_level` remains CRITICAL; `tmr_disagree_count` increments; `fault_detected[1]` asserts (rate-based) |
| Pass/fail | Voted output stable; counter increments; fault flag asserted within 1 clk |
| Environment | SIM |
| Required tools | Fault-injection harness; `dms_fusion_inj_5k.yaml` campaign |
| Allocation | S |
| Status | READY (campaign shipped) |

### 3.7 SG-2.3 — SENSOR_FAIL within FTTI (ASIL-B)

#### IT-2.3.1.1 — Watchdog timeout asserts SENSOR_FAIL within budget

| Field | Value |
|---|---|
| Parent FSR | FSR-2.3.1 |
| ASIL | B |
| Setup | Drive dms_fusion with steady gaze_valid pulses; configure WATCHDOG_CYCLES |
| Stimulus | Stop driving gaze_valid pulses |
| Expected response | `dms_fusion.sensor_fail` asserts within WATCHDOG_CYCLES + 1 clk; `fault_detected[4]` asserts |
| Pass/fail | Assertion observed within budget |
| Environment | SIM |
| Required tools | Existing test infrastructure |
| Allocation | S |
| Status | READY ✅ |

#### IT-2.3.2.1 — SENSOR_FAIL routes to safe_state_active

| Field | Value |
|---|---|
| Parent FSR | FSR-2.3.2 |
| ASIL | B |
| Setup | Integrated SoC with safe_state_controller wired to dms_fusion fault output |
| Stimulus | Trigger SENSOR_FAIL per IT-2.3.1.1 |
| Expected response | `safe_state_active` asserts; safe_state ladder advances per §3.4 of Safety Manual |
| Pass/fail | Assertion observed; ladder advances per ALERT_TIME_MS / DEGRADE_TIME_MS budgets |
| Environment | SIM |
| Required tools | Integrated cocotb harness combining dms_fusion + safe_state_controller |
| Allocation | S |
| Status | STUBBED — integrated harness needs assembly |

### 3.8 SG-2.4 — No flicker (ASIL-A)

#### IT-2.4.1.1 — IIR attenuates single-frame perturbation by ≥ 75 %

(Equivalent to IT-2.1.2.1 — same test serves both FSRs)

| Field | Value |
|---|---|
| Parent FSR | FSR-2.4.1 |
| ASIL | A |
| Status | READY ✅ (cross-references IT-2.1.2.1) |

### 3.9 SG-3.1 — Reverse obstacle FTTI (ASIL-C)

#### IT-3.1.1.1 — Cross-sensor confirmation raises CRITICAL within FTTI_PROX

| Field | Value |
|---|---|
| Parent FSR | FSR-3.1.1 |
| ASIL | C |
| Setup | `examples/ultrasonic_proximity_alarm.py` reference fusion engine; configure with default thresholds |
| Stimulus | Inject US + lidar both reporting obstacle within ~0.3 m |
| Expected response | CRITICAL band raised within FTTI_PROX = 50 ms; alarm signalled to vehicle controller |
| Pass/fail | Transition observed within budget |
| Environment | SIM (today); FPGA (with custom-fusion plugin) |
| Required tools | `examples/ultrasonic_proximity_alarm.py` + cocotb stimulus driver |
| Allocation | S |
| Status | READY ✅ (existing example demo) |

#### IT-3.1.2.1 — CRITICAL band is speed-independent

| Field | Value |
|---|---|
| Parent FSR | FSR-3.1.2 |
| ASIL | C |
| Setup | Same as IT-3.1.1.1 |
| Stimulus | Sweep CAN-reported speed from 0 → 30 km/h; inject obstacle at < 0.3 m at each speed |
| Expected response | CRITICAL raised at every speed |
| Pass/fail | CRITICAL signal asserted at every sampled speed |
| Environment | SIM |
| Required tools | Speed-sweep extension to existing example |
| Allocation | S |
| Status | STUBBED — speed-sweep test needs assembly |

### 3.10 SG-3.2 — Alarm band correctness (ASIL-B)

#### IT-3.2.1.1 — Speed-scaled thresholds applied per YAML config

| Field | Value |
|---|---|
| Parent FSR | FSR-3.2.1 |
| ASIL | B |
| Setup | `astracore configure --apply` with `safety_policies` declaring custom speed-scaled thresholds |
| Stimulus | Inject obstacles at varied (range, speed) pairs |
| Expected response | Alarm band (OFF/CAUTION/WARNING/CRITICAL) matches the policy table at every (range, speed) |
| Pass/fail | All transitions match policy |
| Environment | SDK |
| Required tools | `astracore.apply` + policy validator + test fixtures |
| Allocation | S |
| Status | READY (existing test `tests/test_apply.py` covers config validation; band-correctness test needs explicit assertion) |

#### IT-3.2.2.1 — Schema validation rejects invalid policy

| Field | Value |
|---|---|
| Parent FSR | FSR-3.2.2 |
| ASIL | B |
| Setup | `astracore configure --apply` with deliberately malformed policy |
| Stimulus | Submit policy violating schema (e.g., negative threshold, threshold inversion) |
| Expected response | Apply call rejects with descriptive error; no inference begins |
| Pass/fail | Rejection observed; no partial application |
| Environment | SDK |
| Required tools | Existing test infrastructure |
| Allocation | S |
| Status | READY ✅ |

### 3.11 SG-3.3 — FP rate on proximity alarms (ASIL-A)

#### IT-3.3.1.1 — Cross-sensor confirmation required for WARNING/CRITICAL

| Field | Value |
|---|---|
| Parent FSR | FSR-3.3.1 |
| ASIL | A |
| Setup | `examples/ultrasonic_proximity_alarm.py` |
| Stimulus | Inject single-sensor false echo (US fires; lidar quiet) at multiple speeds |
| Expected response | Band stays at CAUTION (single-sensor); never escalates to WARNING/CRITICAL on a single-sensor signal |
| Pass/fail | No false WARNING/CRITICAL across the corpus |
| Environment | SDK |
| Required tools | Single-sensor-echo test fixtures |
| Allocation | S |
| Status | READY (existing example demo provides framework) |

#### IT-3.3.2.1 — Plausibility checker MIN_CONFIDENCE rejects low-confidence detections

| Field | Value |
|---|---|
| Parent FSR | FSR-3.3.2 |
| ASIL | A |
| Setup | `plausibility_checker` with default MIN_CONFIDENCE=64 |
| Stimulus | Inject detections with confidence sweeping 0..255 |
| Expected response | Detections with confidence < 64 rejected; ≥ 64 passed |
| Pass/fail | Rejection threshold matches MIN_CONFIDENCE |
| Environment | SIM |
| Required tools | `plausibility_inj_2k` campaign (per FMEDA schedule W9) |
| Allocation | S |
| Status | BLOCKED on plausibility_checker campaign authoring |

---

## 4. Aggregate test status

| Status | Count | % |
|---|---:|---:|
| READY | 9 | 36 % |
| STUBBED | 7 | 28 % |
| BLOCKED (waits for F4 or campaign WP) | 5 | 20 % |
| LICENSEE | 4 | 16 % |
| **Total** | **25** | **100 %** |

(Counts may double-count where one IT serves multiple FSRs — e.g., IT-2.4.1.1 ≡ IT-2.1.2.1.)

### 4.1 By FSR allocation

| Allocation | Count |
|---|---:|
| S (Supplier-shipped) | 18 |
| L (Licensee-authored) | 4 |
| J (Joint) | 0 |
| Cross-reference (no own test) | 1 |
| Mixed/None | 2 |

### 4.2 By environment

| Environment | Count |
|---|---:|
| SIM (cocotb / Verilator) | 16 |
| FPGA | 5 |
| SILICON (Licensee-only) | 3 |
| SDK (Python-only on host) | 5 |

(Some tests run in multiple environments — e.g., IT-1.1.1.1 runs in SIM today, FPGA after F1-F3, SILICON after Licensee tape-out.)

---

## 5. Test asset roadmap

The STUBBED + BLOCKED tests above need supporting test assets to move to READY. This section enumerates the assets needed.

| Asset | Used by | Source / WP | Effort |
|---|---|---|---|
| Labelled OS-1.A highway clip | IT-1.1.1.1, IT-1.1.2.1, IT-1.1.5.1 | Public dataset (BDD100K, KITTI) + manual labelling for ASIL-relevant subsets | 5 days |
| Synthetic adverse-weather frame generator | IT-1.1.2.1 | Custom Python (3D rendering + degradation models) | 4 days |
| Multi-lane TMR injection extension to fault-injection harness | IT-1.1.3.1 | Extension to `sim/fault_injection/runner.py` to support concurrent injections | 2 days |
| Synthetic obstacle generator (range, velocity, class) | IT-1.3.1.1 | Custom Python | 3 days |
| Lane-fade clip generator | IT-1.4.1.1 | Synthetic alpha-blend on labelled clips | 2 days |
| `lane_fusion` fault-injection campaign | IT-1.4.2.1 | New campaign YAML + testbench wrapper | 2 days |
| Integrated dms_fusion + safe_state_controller harness | IT-2.3.2.1 | Cocotb wrapper combining the two modules | 2 days |
| Speed-sweep test for proximity alarm | IT-3.1.2.1 | Extension to `examples/ultrasonic_proximity_alarm.py` | 1 day |
| `plausibility_inj_2k` campaign | IT-3.3.2.1 | New campaign YAML + testbench wrapper (per FMEDA schedule W9) | 3 days |
| **Total asset effort** | | | **~24 days** |

WPs to track:
- **F4-IT-1** through **F4-IT-9** (one per asset above) — proposed for Phase B/C alongside the F4-A through F4-D RTL/test work

---

## 6. Open items for v0.2

These items must close before this Integration Test Plan can be approved for external assessment:

1. **Named Track 2 lead, independent reviewer, Safety Manager / Approver** — currently TBD per §0
2. **Quantitative pass/fail thresholds** for all tests where today's threshold is qualitative (e.g., "P99 latency ≤ 30 ms" — confirmed against benchmark data)
3. **Licensee-test allocation** — confirm with first NDA evaluation licensee that the L-allocated tests (IT-1.1.4.1, IT-1.2.2.1, IT-1.3.2.1, IT-1.4.1.1's silicon variant) are acceptable as licensee responsibility
4. **Asset WPs F4-IT-1..9** opened in remediation plan
5. **Integration test summary report template** — define the format the licensee uses to report integration test results back to AstraCore (per DIA §5.2)
6. **Safety Case verification matrix** — at W11 when Safety Case v0.1 is drafted, this Integration Test Plan feeds the matrix that maps tests → FSRs → SGs → ASIL claim

---

## 7. Document control

| Field | Value |
|---|---|
| Document ID | ASTR-SAFETY-ITP-V0.1 |
| Revision | 0.1 |
| Revision date | 2026-04-20 |
| Author | TBD (Track 2 lead) — currently founder + collaborator |
| Reviewer | TBD |
| Approver | TBD |
| Distribution | Internal + TÜV SÜD India + first NDA evaluation licensee |
| Retention | 15 years post-product-discontinuation per ISO 26262-8 §10 |
| Supersedes | None — this is v0.1; closes the FSC §6.7 v0.2 open item for ITP |

### 7.1 Revision triggers

This Integration Test Plan is re-issued (with revision bump) on any of:

1. New FSR added to FSC → must spawn new IT(s)
2. FSR pass/fail threshold quantified or revised
3. Test asset moves from STUBBED/BLOCKED → READY (status column update; not necessarily a doc rev)
4. F4 phase milestone closes that unblocks a BLOCKED test
5. New test environment added (e.g., post-tape-out silicon validation kit)
6. Confirmation review feedback that changes any test specification
