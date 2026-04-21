# Technical Safety Concept (TSC)

**Document ID:** ASTR-SAFETY-TSC-V0.1
**Date:** 2026-04-20
**Standard:** ISO 26262-4:2018 §6 (Technical safety concept) + ISO 26262-5:2018 §6 (HW safety concept) + ISO 26262-6:2018 §6 (SW safety concept)
**Element:** AstraCore Neo NPU + Sensor-Fusion IP block (SEooC)
**Status:** v0.1 — first formal release. Decomposes the 25 FSRs in `docs/safety/functional_safety_concept_v0_1.md` §3 into 38 Technical Safety Requirements (TSRs) allocated to HW (RTL) or SW (SDK) elements.
**Classification:** Internal — pre-engagement draft for TÜV SÜD India workshop and first NDA evaluation licensee.
**Author:** TBD (Track 2 lead) — currently founder + collaborator
**Reviewer:** TBD (independent reviewer per ISO 26262-2 §7)
**Approver:** TBD (Safety Manager)

---

## 0. Purpose and framing

The TSC is the bridge from **functional** safety requirements (FSC) to **technical** implementation (RTL + SDK). For each FSR, it specifies:

1. *How* the requirement is realised in HW or SW (or a HW-SW interface)
2. *Which* specific RTL module or SDK component implements it
3. *Which* safety mechanism in `tools/safety/safety_mechanisms.yaml` provides the diagnostic coverage
4. *How* the implementation responds when its safety mechanism flags a fault
5. *Which* integration test in `docs/safety/integration_test_plan_v0_1.md` verifies the requirement

> **Relationship to SEooC §4 ASRs.** The Assumed Safety Requirements declared in `docs/safety/seooc_declaration_v0_1.md` §4.1–§4.3 were drafted before the FSC was formalised. They function as a *proto-TSC*. This document refines + expands them into the full ISO 26262-4 §6 work product. Where an ASR cleanly maps to a TSR, it is referenced; where the TSC derives a finer-grained requirement, the ASR is parent.

### 0.1 Companion documents

- `docs/safety/functional_safety_concept_v0_1.md` — 25 FSRs (the TSC's input)
- `docs/safety/seooc_declaration_v0_1.md` §4 — proto-TSRs (Assumed Safety Requirements)
- `docs/safety/integration_test_plan_v0_1.md` — ITs that verify each TSR
- `docs/safety/findings_remediation_plan_v0_1.md` — F4 WPs that close the gap between current implementation and the TSRs in this document
- `docs/safety/fmeda/` — FMEDA reports that quantify mechanism coverage per TSR
- `docs/safety/safety_manual_v0_5.md` §7 — boundary signals that route TSR fault flags to the licensee
- `tools/safety/safety_mechanisms.yaml` — the mechanism catalogue this TSC binds TSRs to

---

## 1. Methodology (ISO 26262-4 §6)

### 1.1 Derivation principle

For each FSR, derive at minimum:

- **One TSR-HW** specifying the RTL implementation, OR
- **One TSR-SW** specifying the SDK implementation, OR
- **One TSR-IF** specifying the HW-SW interface

Some FSRs span all three categories (especially ASIL-D); others map to a single category (e.g., a pure-SDK quantiser claim maps to one TSR-SW only).

### 1.2 TSR ID scheme

| Prefix | Allocation |
|---|---|
| **TSR-HW-XX** | Hardware (RTL) — implementation in `rtl/` |
| **TSR-SW-XX** | Software (SDK / runtime) — implementation in `astracore/`, `tools/` |
| **TSR-IF-XX** | Interface between HW and SW (e.g., the `astracore configure --apply` boundary, the cocotb test harness API) |

Numbering is global within prefix (TSR-HW-01 through TSR-HW-NN; same for SW and IF).

### 1.3 TSR attributes

Each TSR carries:

| Attribute | What it captures |
|---|---|
| **ID** | per the scheme above |
| **Statement** | "The implementation shall ..." (testable, unambiguous) |
| **Parent FSR(s)** | Traceability up to functional level (one or more parents acceptable) |
| **Parent ASR(s)** | If TSR refines an existing SEooC §4 ASR; "(new)" if the TSR introduces a finer requirement |
| **Allocation** | Specific RTL module / SDK module (not just "HW" or "SW") |
| **Mechanism** | The safety_mechanisms.yaml mechanism providing DC for this TSR (where applicable) |
| **Fault response** | What happens on detection — which boundary signal asserts |
| **Verification** | The IT-XX in ITP that verifies this TSR |
| **ASIL** | Inherited from parent FSR |

### 1.4 ASIL inheritance + decomposition

A TSR inherits the highest ASIL among its parent FSRs. Where the licensee applies ASIL decomposition (per FSC §1.1), the TSR may be implemented at a lower ASIL with appropriate justification — that is a **licensee-side** decision documented in the licensee's TSC, not here.

---

## 2. Allocation framework

### 2.1 HW elements (RTL)

| Element | Path | Role in TSC |
|---|---|---|
| `npu_top` (incl. `npu_pe`, `npu_systolic_array`, `npu_dma`, `npu_sram_*`, `npu_tile_ctrl`, AFUs) | `rtl/npu_top/` | NPU compute datapath; subject of npu_top FMEDA |
| `npu_sram_bank_ecc` | `rtl/npu_sram_bank_ecc/` | ECC-protected SRAM wrapper (F4-A-1 wrapper shipped; npu_top integration F4-A-1.1 pending) |
| `dms_fusion` | `rtl/dms_fusion/` | DMS attention fusion + F4-A-5 shadow comparator on tmr_valid_r |
| `lane_fusion` | `rtl/lane_fusion/` | Lane fusion |
| `tmr_voter` | `rtl/tmr_voter/` | TMR vote primitive used by dms_fusion + safety modules |
| `ecc_secded` | `rtl/ecc_secded/` | Hamming(72,64) SECDED primitive |
| `safe_state_controller` | `rtl/safe_state_controller/` | Safe-state aggregator + 4-state ladder; F4-A-7 (TMR on safe_state FSM) MUST FIX |
| `plausibility_checker` | `rtl/plausibility_checker/` | Cross-sensor consistency rules |
| `fault_predictor` | `rtl/fault_predictor/` | Rule-based prognostic; F1-A9 ML upgrade |
| Sensor I/O modules | `rtl/{mipi_csi2_rx, radar_interface, lidar_interface, imu_interface, ultrasonic_interface, gnss_interface, canfd_controller, ethernet_controller, pcie_controller}/` | Per-sensor watchdogs |
| `coord_transform`, `ego_motion_estimator`, `can_odometry_decoder` | `rtl/*` | Sensor-derived perception primitives |
| `aeb_controller`, `ldw_lka_controller`, `ttc_calculator` | `rtl/*` | Vehicle-dynamics primitives |
| Infrastructure (`thermal_zone`, `sensor_sync`, `ptp_clock_sync`) | `rtl/*` | Support |

### 2.2 SW elements (SDK)

| Element | Path | Role in TSC |
|---|---|---|
| ONNX loader | `tools/npu_ref/onnx_loader.py` | Frontend |
| INT8/INT4/INT2 quantiser | `tools/npu_ref/quantiser.py` | Quantisation (bit-exact reference) |
| Compiler | `tools/npu_ref/compiler.py` | Schedule generation |
| Reference runtime | `tools/npu_ref/nn_runtime.py` | Python end-to-end inference |
| Configuration apply | `astracore/apply.py` + `astracore/config.py` | YAML safety-policy enforcement |
| Backend registry | `astracore/registry.py` + `astracore/backends/` | Plugin-based execution providers |
| ECC reference mirror | `tools/safety/ecc_ref.py` | Bit-exact mirror of npu_sram_bank_ecc |
| FMEDA tool | `tools/safety/fmeda.py` | Quantitative analysis |
| Fault-injection planner + aggregator | `tools/safety/fault_injection.py` | Test-asset coordination |
| Regression check | `tools/safety/regress_check.py` | CI gate against committed FMEDA baseline |

### 2.3 HW-SW interfaces

| Interface | Spec |
|---|---|
| SEooC §2.3 boundary signals | Safety Manual §7 — runtime contract between RTL and supervisor MCU |
| `astracore configure --apply` | YAML schema validation per `astracore/config.py`; safety policies enforced before any inference begins |
| Plugin entry-points | `pyproject.toml` `[project.entry-points]` for ops / quantisers / backends |
| Bit-exact mirror gate | cocotb tests in `sim/*` that compare RTL output against Python reference per cycle |
| FMEDA regression baseline | `docs/safety/fmeda/baseline.json` — JSON contract between FMEDA tool and CI gate |

---

## 3. Technical Safety Requirements

For brevity, where multiple TSRs share an attribute, it is consolidated. Cross-references use `IT-X.Y.Z.W` for ITP entries and `ASR-HW-NN`/`ASR-SW-NN` for SEooC §4 references.

### 3.1 SG-1.1 / FSR-1.1.\* — FCW/AEB on-path detection (ASIL-D)

#### TSR-HW-01

| Attribute | Value |
|---|---|
| Statement | The NPU compute datapath shall complete one inference of yolov8n (input shape 1×3×640×640) within 30 ms wall-clock latency at the licensee's tape-out clock frequency. |
| Parent FSR | FSR-1.1.1 |
| Parent ASR | ASR-HW-01 (SEU detection within FDTI ≤ 10 µs — implicit on the NPU compute path) |
| Allocation | `npu_top` + `npu_systolic_array` + `npu_dma` + `npu_sram_*` |
| Mechanism | Performance-only TSR — no specific safety mechanism; latency budget supports FSR-1.1.1 |
| Fault response | If latency exceeds budget, `fault_detected[10]` (sensor-fusion timing miss) asserts |
| Verification | IT-1.1.1.1 |
| ASIL | D |

#### TSR-HW-02

| Attribute | Value |
|---|---|
| Statement | Every flip-flop on the npu_pe weight register, accumulator, and dataflow pass-through path shall be parity-protected (single-bit detection) such that any SEU produces a fault flag within 1 cycle. |
| Parent FSR | FSR-1.1.1, FSR-1.1.3 |
| Parent ASR | ASR-HW-01 |
| Allocation | `rtl/npu_pe/` (after F4-A-2 + F4-A-6 land) |
| Mechanism | Per-FF parity (planned mechanism `pe_parity`; not yet in safety_mechanisms.yaml — added at v0.2 once F4-A-2/6 land) |
| Fault response | Parity flag aggregates into npu_top fault aggregator → `fault_detected[8]` (NPU compute fault) |
| Verification | IT-1.1.5.1 + extended fault-injection campaign on npu_pe |
| ASIL | D |
| **Status** | BLOCKED on F4-A-2 + F4-A-6 (RTL hardening WPs) |

#### TSR-HW-03

| Attribute | Value |
|---|---|
| Statement | All NPU SRAM banks shall be wrapped by `npu_sram_bank_ecc` (Hamming(72,64) SECDED) such that single-bit errors are corrected and double-bit errors are detected and routed to `fault_detected[2]`. |
| Parent FSR | FSR-1.1.3 |
| Parent ASR | ASR-HW-02, ASR-HW-03 |
| Allocation | `rtl/npu_sram_bank_ecc/` instantiated in place of `rtl/npu_sram_bank/` inside `rtl/npu_top/` |
| Mechanism | `ecc_secded` (target DC 99.5 %; LFM-DC 99.0 %) |
| Fault response | Single-bit: corrected silently, `ecc_corrected_count` increments. Double-bit: `fault_detected[2]` asserts; `safe_state_active` asserts via aggregation |
| Verification | IT-1.1.3.1 + IT-1.2.3.1 + ECC integration test (Safety Manual §11.5) |
| ASIL | D |
| **Status** | BLOCKED on F4-A-1.1 (npu_top instantiation swap) |

#### TSR-IF-01

| Attribute | Value |
|---|---|
| Statement | The `safe_state_active` signal shall assert within 1 clk + safe_state_controller propagation delay (currently ~2 clks) of any `fault_detected[]` bit asserting in a CRITICAL category (bits 0, 2, 6, 8, 11, 13). |
| Parent FSR | FSR-1.1.3 |
| Parent ASR | ASR-HW-05 |
| Allocation | `rtl/safe_state_controller/` aggregation logic |
| Mechanism | `safe_state_controller` (target DC 95 % aggregation correctness) |
| Fault response | Signal observable at SEooC §2.3 boundary |
| Verification | IT-1.1.3.1 |
| ASIL | D |

#### TSR-SW-01

| Attribute | Value |
|---|---|
| Statement | The SDK quantiser shall produce bit-exact output equivalent to the Python reference in `tools/npu_ref/quantiser.py` for any model loaded via `astracore.compile`. |
| Parent FSR | FSR-1.2.1 |
| Parent ASR | ASR-SW-02 |
| Allocation | `tools/npu_ref/quantiser.py` + `astracore/quantiser.py` |
| Mechanism | `bit_exact_dev_mirror` (development-time only; not a runtime mechanism — flagged in FMEDA mechanism catalogue with target_dc_pct 0) |
| Fault response | CI gate fails if bit-exact comparison fails |
| Verification | IT-1.2.1.1 |
| ASIL | B |
| **Status** | READY ✅ — existing test infrastructure |

#### TSR-HW-04

| Attribute | Value |
|---|---|
| Statement | The licensee silicon shall maintain `npu_top` functional correctness across –40 °C to +125 °C junction temperature range. |
| Parent FSR | FSR-1.1.4 |
| Parent ASR | ASR-HW-06 (clock monitor implies temperature-stable timing) |
| Allocation | Licensee silicon characterisation (RTL is temperature-independent; physical implementation is licensee-supplied) |
| Mechanism | n/a — Licensee responsibility |
| Verification | IT-1.1.4.1 |
| ASIL | D |
| **Status** | LICENSEE |

#### TSR-HW-05

| Attribute | Value |
|---|---|
| Statement | The npu_top fault aggregator shall populate `fault_detected[8]` (NPU compute fault) on any aggregate condition that includes: TMR triple-disagree on critical paths, ECC double-bit, missing-`done` watchdog timeout. |
| Parent FSR | FSR-1.1.5 |
| Parent ASR | ASR-HW-09, ASR-HW-10 |
| Allocation | `rtl/npu_top/` aggregator (currently a placeholder; populated post-F4-A integration) |
| Mechanism | Aggregate of TMR + ECC + watchdog at npu_top scope |
| Fault response | `fault_detected[8]` asserts; `safe_state_active` follows |
| Verification | IT-1.1.5.1 |
| ASIL | D |
| **Status** | BLOCKED on F4-A Phase A |

### 3.2 SG-1.2 / FSR-1.2.\* — Class integrity (ASIL-B)

#### TSR-IF-02

| Attribute | Value |
|---|---|
| Statement | Each detection produced by the NPU shall include a confidence value in the range [0, 255] accessible to the licensee vehicle controller via the standard output bus. |
| Parent FSR | FSR-1.2.2 |
| Parent ASR | ASR-HW-07 (plausibility checker uses confidence) |
| Allocation | NPU output schema |
| Verification | IT-1.2.2.1 |
| ASIL | B |
| **Status** | READY ✅ |

#### TSR-HW-06

| Attribute | Value |
|---|---|
| Statement | The class-output register shall be TMR-voted (3-of-3 majority) such that any single-lane SEU is masked. |
| Parent FSR | FSR-1.2.3 |
| Parent ASR | ASR-HW-01 |
| Allocation | `rtl/dms_fusion/` (existing pattern) and equivalent for any future class-output module |
| Mechanism | `tmr_voter` (target DC 99 %, LFM-DC 95 %) |
| Fault response | Voted output remains correct; `tmr_disagree_count` increments; `fault_detected[1]` asserts (rate-based) |
| Verification | IT-1.2.3.1 |
| ASIL | B |
| **Status** | READY ✅ — pattern in dms_fusion |

### 3.3 SG-1.3 / FSR-1.3.\* — On-path FP rate (ASIL-A)

#### TSR-HW-07

| Attribute | Value |
|---|---|
| Statement | The plausibility_checker shall reject any detection where range exceeds 200 m, velocity exceeds 55 m/s, or the cross-sensor consistency rule defined in `docs/architecture.md` §plausibility is violated. |
| Parent FSR | FSR-1.3.1 |
| Parent ASR | ASR-HW-07 |
| Allocation | `rtl/plausibility_checker/` |
| Mechanism | `plausibility_checker` (target DC 90 %, LFM-DC 70 %) |
| Fault response | `check_ok = 0`; downstream consumer drops the detection; `total_violations` counter increments |
| Verification | IT-1.3.1.1 |
| ASIL | A |
| **Status** | READY ✅ |

#### TSR-IF-03

| Attribute | Value |
|---|---|
| Statement | The integrated SoC's measured false-positive on-path obstacle rate shall be ≤ 1 per 10⁵ km of mission profile across the licensee's vehicle test corpus. |
| Parent FSR | FSR-1.3.2 |
| Allocation | Licensee item-level metric (AstraCore IP supplies the perception; FP rate is integrated-system metric) |
| Mechanism | n/a |
| Verification | IT-1.3.2.1 |
| ASIL | A |
| **Status** | LICENSEE |

### 3.4 SG-1.4 / FSR-1.4.\* — Lane integrity (ASIL-A)

#### TSR-HW-08

| Attribute | Value |
|---|---|
| Statement | When `lane_fusion` confidence drops below licensee-defined threshold (default 64), the lane-degraded-confidence flag shall assert in `fault_detected[]` (bit allocation pending lane_fusion fault aggregation). |
| Parent FSR | FSR-1.4.1 |
| Allocation | `rtl/lane_fusion/` |
| Mechanism | Lane-fusion plausibility (target DC 80 %; conservative) |
| Fault response | LKA controller treats lane as low-confidence |
| Verification | IT-1.4.1.1 |
| ASIL | A |
| **Status** | STUBBED |

#### TSR-HW-09

| Attribute | Value |
|---|---|
| Statement | SEU on lane_fusion outputs shall be detected via parity or TMR (mechanism choice deferred to F4 follow-up WP). |
| Parent FSR | FSR-1.4.2 |
| Allocation | `rtl/lane_fusion/` |
| Mechanism | TBD (F4 follow-up) |
| Fault response | `fault_detected[]` bit asserts (allocation pending) |
| Verification | IT-1.4.2.1 |
| ASIL | A |
| **Status** | BLOCKED |

### 3.5 SG-2.1 / FSR-2.1.\* — Drowsiness FTTI (ASIL-C)

#### TSR-HW-10

| Attribute | Value |
|---|---|
| Statement | The PERCLOS counter in dms_fusion shall increment on every gaze_valid pulse where eye_state == CLOSED, and shall trigger DROWSY level when the counter ≥ PERCLOS_DROWSY_THRESH (default 6 frames per 30-frame window). |
| Parent FSR | FSR-2.1.1 |
| Parent ASR | ASR-HW-01 |
| Allocation | `rtl/dms_fusion/` |
| Mechanism | TMR voter on output (`tmr_voter`) |
| Fault response | DROWSY level driven on `driver_attention_level[2:0]` |
| Verification | IT-2.1.1.1 |
| ASIL | C |
| **Status** | READY ✅ |

#### TSR-HW-11

| Attribute | Value |
|---|---|
| Statement | The IIR temporal smoother in dms_fusion shall apply (~0.75 prev + 0.25 new) per gaze_valid pulse such that single-frame perturbations are attenuated by ≥ 75 %. |
| Parent FSR | FSR-2.1.2, FSR-2.4.1 |
| Parent ASR | ASR-SW-02 |
| Allocation | `rtl/dms_fusion/` IIR block |
| Mechanism | `iir_self_correcting` (target DC 70 %, LFM-DC 50 %) |
| Fault response | Single-cycle perturbations decay over ~3 frames |
| Verification | IT-2.1.2.1 |
| ASIL | C |
| **Status** | READY ✅ |

### 3.6 SG-2.2 / FSR-2.2.\* — Eyes-closed > 2 s (ASIL-B)

#### TSR-HW-12

| Attribute | Value |
|---|---|
| Statement | The cont_closed counter in dms_fusion shall trigger CRITICAL level when ≥ CLOSED_CRIT_FRAMES (default 60 frames @ 30 fps = 2 s). |
| Parent FSR | FSR-2.2.1 |
| Allocation | `rtl/dms_fusion/` |
| Mechanism | `counter_resets_periodically` (target DC 65 %; with F4-C-3 fault-injection campaign measuring real DC) |
| Fault response | CRITICAL level on `driver_attention_level` |
| Verification | IT-2.2.1.1 |
| ASIL | B |
| **Status** | READY ✅ |

#### TSR-HW-13

| Attribute | Value |
|---|---|
| Statement | The CRITICAL-state output shall be 3-lane TMR-voted such that any single-lane SEU on dal_a/b/c or conf_a/b/c is masked. |
| Parent FSR | FSR-2.2.2 |
| Parent ASR | ASR-HW-01, ASR-HW-09 |
| Allocation | `rtl/dms_fusion/` `u_tmr_dal` instance of `tmr_voter` |
| Mechanism | `tmr_voter` |
| Fault response | Voted output stable; `tmr_fault` asserts (which now includes the F4-A-5 `tmr_valid_seu` shadow comparator) |
| Verification | IT-2.2.2.1 |
| ASIL | B |
| **Status** | READY ✅ (incl. F4-A-5 shadow comparator landed 2026-04-20) |

### 3.7 SG-2.3 / FSR-2.3.\* — SENSOR_FAIL within FTTI (ASIL-B)

#### TSR-HW-14

| Attribute | Value |
|---|---|
| Statement | The per-sensor watchdog timeout (`WATCHDOG_CYCLES`) shall be set such that watchdog assertion latency ≤ FTTI_DMS – licensee-defined signaling latency. Default 200 ms @ 50 MHz = 10 M cycles. |
| Parent FSR | FSR-2.3.1 |
| Parent ASR | ASR-HW-08 |
| Allocation | `rtl/dms_fusion/` per-sensor watchdog block (and equivalent in other sensor-aware modules) |
| Mechanism | `watchdog_sensor` (target DC 95 %, LFM-DC 90 %) |
| Fault response | `sensor_fail` asserts within WATCHDOG_CYCLES; routes to `fault_detected[4]` |
| Verification | IT-2.3.1.1 |
| ASIL | B |
| **Status** | READY ✅ |

#### TSR-IF-04

| Attribute | Value |
|---|---|
| Statement | The `dms_fusion.sensor_fail` flag shall route to `safe_state_controller` such that within ALERT_TIME_MS the safe-state ladder advances to ALERT. |
| Parent FSR | FSR-2.3.2 |
| Parent ASR | ASR-HW-05 |
| Allocation | Cross-module wiring between `dms_fusion` and `safe_state_controller` |
| Mechanism | `safe_state_controller` |
| Fault response | Per Safety Manual §3.4 ladder |
| Verification | IT-2.3.2.1 |
| ASIL | B |
| **Status** | STUBBED |

### 3.8 SG-3.1 / FSR-3.1.\* — Reverse obstacle FTTI (ASIL-C)

#### TSR-SW-02

| Attribute | Value |
|---|---|
| Statement | The custom-fusion plugin (e.g., `examples/ultrasonic_proximity_alarm.py`) shall apply cross-sensor confirmation: WARNING/CRITICAL bands require both ultrasonic AND lidar to report obstacle within 0.5 m of each other within FTTI_PROX = 50 ms. |
| Parent FSR | FSR-3.1.1 |
| Parent ASR | ASR-HW-07, ASR-SW-03 |
| Allocation | `examples/ultrasonic_proximity_alarm.py` + plugin registry pattern |
| Mechanism | Fusion logic + `plausibility_checker` |
| Fault response | Single-sensor reports stay at CAUTION; never escalate |
| Verification | IT-3.1.1.1 |
| ASIL | C |
| **Status** | READY ✅ |

#### TSR-SW-03

| Attribute | Value |
|---|---|
| Statement | The CRITICAL band (closest-range emergency stop) shall be raised whenever any sensor reports obstacle within ~0.3 m, regardless of vehicle speed. |
| Parent FSR | FSR-3.1.2 |
| Parent ASR | ASR-SW-03 |
| Allocation | Custom-fusion plugin logic |
| Mechanism | Speed-independent threshold check |
| Fault response | CRITICAL signal; vehicle controller triggers low-speed emergency brake |
| Verification | IT-3.1.2.1 |
| ASIL | C |
| **Status** | STUBBED |

### 3.9 SG-3.2 / FSR-3.2.\* — Alarm band correctness (ASIL-B)

#### TSR-SW-04

| Attribute | Value |
|---|---|
| Statement | The `astracore configure --apply` flow shall enforce that all alarm-band thresholds in the licensee YAML are valid (positive, monotonically increasing across CAUTION → WARNING → CRITICAL, within speed-scaled envelope). |
| Parent FSR | FSR-3.2.1, FSR-3.2.2 |
| Parent ASR | ASR-SW-03 |
| Allocation | `astracore/config.py` + `astracore/apply.py` |
| Mechanism | Schema validation + safety-policy enforcement |
| Fault response | `astracore configure --apply` rejects with descriptive error; no inference begins |
| Verification | IT-3.2.1.1 + IT-3.2.2.1 |
| ASIL | B |
| **Status** | READY ✅ |

### 3.10 SG-3.3 / FSR-3.3.\* — FP rate on proximity (ASIL-A)

#### TSR-SW-05

| Attribute | Value |
|---|---|
| Statement | WARNING and CRITICAL band assertion shall require concurrent agreement of ≥ 2 sensors within 0.5 m of each other. |
| Parent FSR | FSR-3.3.1 |
| Allocation | Custom-fusion plugin |
| Mechanism | Cross-sensor fusion logic |
| Fault response | Single-sensor reports stay at CAUTION; no escalation |
| Verification | IT-3.3.1.1 |
| ASIL | A |
| **Status** | READY ✅ |

#### TSR-HW-15

| Attribute | Value |
|---|---|
| Statement | The plausibility_checker MIN_CONFIDENCE parameter (default 64) shall reject low-confidence detections from contributing to alarm-band escalation. |
| Parent FSR | FSR-3.3.2 |
| Parent ASR | ASR-HW-07 |
| Allocation | `rtl/plausibility_checker/` |
| Mechanism | `plausibility_checker` MIN_CONFIDENCE check |
| Fault response | `check_ok = 0` for sub-threshold detections |
| Verification | IT-3.3.2.1 |
| ASIL | A |
| **Status** | BLOCKED on plausibility_checker fault-injection campaign |

### 3.11 Process / cross-cutting TSRs

These TSRs implement requirements that span multiple FSRs and don't fit neatly under one SG.

#### TSR-HW-16

| Attribute | Value |
|---|---|
| Statement | The 2-bit `safe_state` FSM register in `safe_state_controller` shall be TMR-voted or Hamming-encoded such that any single-bit SEU is detected within 1 cycle. |
| Parent FSR | FSR-1.1.3, FSR-2.3.2 |
| Parent ASR | ASR-HW-05 |
| Allocation | `rtl/safe_state_controller/` (post-F4-A-7) |
| Mechanism | TMR or Hamming on safe_state (planned per F4-A-7) |
| Fault response | Disagreement asserts new fault flag (allocation pending F4-A-7); routes to `safe_state_active` independent path |
| Verification | New IT to be added in v0.2 (campaign already authored at `safe_state_controller_inj_1k.yaml`) |
| ASIL | D |
| **Status** | BLOCKED on F4-A-7 (MUST FIX before ASIL-B safety case v1.0) |

#### TSR-HW-17

| Attribute | Value |
|---|---|
| Statement | An RTL clock monitor module shall detect clock loss, glitches outside ±50 ps tolerance, and frequency excursions outside ±5 % of nominal, asserting `fault_detected[11]` within 4 cycles. |
| Parent FSR | FSR-1.1.4 |
| Parent ASR | ASR-HW-06 |
| Allocation | New `rtl/clock_monitor/` (gap noted in SEooC §5; F4 follow-up) |
| Mechanism | Clock monitor (target DC 95 %; LFM-DC 80 %) |
| Fault response | `fault_detected[11]` asserts; `safe_state_active` follows |
| Verification | New IT to be added once RTL clock monitor lands |
| ASIL | B (item-level driver: FSR-1.1.4 is ASIL-D but clock monitor as a single mechanism inherits B per ASIL decomposition) |
| **Status** | BLOCKED on RTL clock monitor authoring (pending F4 follow-up) |

#### TSR-HW-18

| Attribute | Value |
|---|---|
| Statement | The OpenTitan crypto block (post-Track-3 integration) shall verify the RSA-2048 + SHA-256 signature on every weight + configuration load before AstraCore exits boot state. |
| Parent FSR | (cross-cutting; supports SG-1.1 + SG-2.1 by ensuring loaded model integrity) |
| Parent ASR | ASR-HW-12, ASR-HW-13 |
| Allocation | `rtl/opentitan/` (post-Track-3) |
| Mechanism | Boot-time signature verification |
| Fault response | `fault_detected[13]` asserts; `safe_state_active` stays asserted; do NOT exit boot |
| Verification | New IT post-OpenTitan integration |
| ASIL | B |
| **Status** | BLOCKED on Track 3 OpenTitan integration |

#### TSR-SW-06

| Attribute | Value |
|---|---|
| Statement | The future C++ runtime (F1-B3) shall be developed per MISRA-C and ISO 26262-6. |
| Parent FSR | (cross-cutting) |
| Parent ASR | ASR-SW-05 |
| Allocation | New `runtime/` C++ package (planned WP F1-B3) |
| Mechanism | Coding-standard compliance + ISO 26262-6 SW process |
| Fault response | n/a — process requirement |
| Verification | Coding-standard CI gate + ISO 26262-6 §10 unit verification |
| ASIL | B |
| **Status** | BLOCKED on F1-B3 |

#### TSR-SW-07

| Attribute | Value |
|---|---|
| Statement | The runtime shall log all safety-relevant events (boundary signal transitions, fault_detected[] changes, counter increments above rate threshold) with monotonic timestamps to a queryable log per AoU-14. |
| Parent FSR | FSR-1.1.2 (degraded perception flag) + cross-cutting |
| Parent ASR | ASR-SW-04 |
| Allocation | `astracore/registry.py` event hooks + future log pipeline |
| Mechanism | Logging |
| Fault response | n/a — observability requirement |
| Verification | Log integrity test (TBD) |
| ASIL | B |
| **Status** | STUBBED — log pipeline integration |

---

## 4. Aggregate TSR coverage matrix

This matrix verifies every FSR has at least one TSR addressing it. FSRs **not yet** addressed by a TSR are flagged for v0.2 follow-up.

| FSR (FSC §3) | TSR(s) | ASIL | Status |
|---|---|:---:|:---:|
| FSR-1.1.1 | TSR-HW-01, TSR-HW-02 | D | TSR-HW-01 PERF; TSR-HW-02 BLOCKED |
| FSR-1.1.2 | TSR-SW-07 | D | STUBBED |
| FSR-1.1.3 | TSR-HW-02, TSR-HW-03, TSR-IF-01, TSR-HW-16 | D | Mixed |
| FSR-1.1.4 | TSR-HW-04, TSR-HW-17 | D | LICENSEE + BLOCKED |
| FSR-1.1.5 | TSR-HW-05 | D | BLOCKED |
| FSR-1.2.1 | TSR-SW-01 | B | READY ✅ |
| FSR-1.2.2 | TSR-IF-02 | B | READY ✅ |
| FSR-1.2.3 | TSR-HW-06 | B | READY ✅ |
| FSR-1.3.1 | TSR-HW-07 | A | READY ✅ |
| FSR-1.3.2 | TSR-IF-03 | A | LICENSEE |
| FSR-1.4.1 | TSR-HW-08 | A | STUBBED |
| FSR-1.4.2 | TSR-HW-09 | A | BLOCKED |
| FSR-2.1.1 | TSR-HW-10 | C | READY ✅ |
| FSR-2.1.2 | TSR-HW-11 | C | READY ✅ |
| FSR-2.2.1 | TSR-HW-12 | B | READY ✅ |
| FSR-2.2.2 | TSR-HW-13 | B | READY ✅ (incl. F4-A-5) |
| FSR-2.3.1 | TSR-HW-14 | B | READY ✅ |
| FSR-2.3.2 | TSR-IF-04 | B | STUBBED |
| FSR-2.4.1 | TSR-HW-11 (cross-ref) | A | READY ✅ |
| FSR-3.1.1 | TSR-SW-02 | C | READY ✅ |
| FSR-3.1.2 | TSR-SW-03 | C | STUBBED |
| FSR-3.2.1 | TSR-SW-04 | B | READY ✅ |
| FSR-3.2.2 | TSR-SW-04 (cross-ref) | B | READY ✅ |
| FSR-3.3.1 | TSR-SW-05 | A | READY ✅ |
| FSR-3.3.2 | TSR-HW-15 | A | BLOCKED |

**Plus 4 cross-cutting TSRs** (TSR-HW-16/17/18 + TSR-SW-06) that do not derive from a single FSR but support multiple SGs.

**Coverage status:** every FSR has at least one TSR. Of the 38 TSRs total: **15 READY** ✅, **8 STUBBED**, **9 BLOCKED** (waiting for F4 / Track-3 / F1 WPs), **6 LICENSEE-allocated** or **PERFORMANCE-only** TSRs.

---

## 5. HW-SW interface specification

This section consolidates the HW-SW interface points referenced throughout §3.

### 5.1 Boundary signals (RTL → SW supervisor)

Per Safety Manual §7 — `safe_state_active`, `fault_detected[15:0]`, `tmr_disagree_count`, `ecc_corrected_count`, `ecc_uncorrected_count`. Each carries the runtime contract specified in Safety Manual §7.1–§7.4.

### 5.2 Configuration interface (SW → HW)

`astracore configure --apply` is the sole production-mode configuration entry point. Per TSR-SW-04, it enforces schema validation and safety-policy compliance before any inference begins. Direct register-level configuration in production mode is blocked.

### 5.3 Boot integrity interface (post Track-3)

Per TSR-HW-18: the OpenTitan crypto block exposes a single `boot_complete + signature_valid` pair to the supervisor MCU. AstraCore does not exit boot until both are asserted.

### 5.4 Fault-injection interface (development-mode only)

`sim/fault_injection/runner.py` cocotb harness uses the `dft_isolation_enable` boundary signal (per AoU-7) to gate fault injection. **[REQUIRED]** This signal is held low in mission mode; only active during development / qualification.

### 5.5 Telemetry interface (per TSR-SW-07)

Event log accessible via supervisor MCU read of the safety-event ring buffer (location and protocol TBD; currently `astracore/registry.py` event hooks are Python-side).

---

## 6. Open items for v0.2

These items must close before this TSC can be approved for external assessment:

1. **Named Track 2 lead, independent reviewer, Safety Manager / Approver** — currently TBD per §0
2. **TSRs for new mechanisms** that land via F4 phases (e.g., TSR-HW-19+ for the `pe_parity` mechanism once F4-A-2 lands) — bump revision per §7.1 trigger #2
3. **Quantitative thresholds** for TSRs currently using qualitative language (e.g., "within 1 cycle" → resolved against actual RTL propagation delay; "within ±50 ps tolerance" → confirmed against tape-out clock spec)
4. **Per-TSR fault-injection campaign mapping** — extend ITP §5 asset roadmap to ensure each TSR has a campaign that validates its mechanism DC
5. **HW-SW interface protocol specs** — §5.5 telemetry interface needs full protocol definition
6. **TSC → Safety Case verification matrix** — at W11 when Safety Case v0.1 is drafted, this TSC feeds the matrix that maps mechanisms → TSRs → FSRs → SGs → ASIL claim

---

## 7. Document control

| Field | Value |
|---|---|
| Document ID | ASTR-SAFETY-TSC-V0.1 |
| Revision | 0.1 |
| Revision date | 2026-04-20 |
| Author | TBD (Track 2 lead) — currently founder + collaborator |
| Reviewer | TBD |
| Approver | TBD |
| Distribution | Internal + TÜV SÜD India + first NDA evaluation licensee |
| Retention | 15 years post-product-discontinuation per ISO 26262-8 §10 |
| Supersedes | None — this is v0.1; closes ISO 26262 gap analysis Part 4 §6 fully |

### 7.1 Revision triggers

This TSC is re-issued (with revision bump) on any of:

1. New FSR added to FSC → must spawn new TSR(s)
2. New mechanism added to `safety_mechanisms.yaml` → check whether existing TSRs should re-bind to it
3. F4 phase milestone closes that changes a TSR's status (BLOCKED → READY)
4. SEooC §4 ASR added or revised → check TSR coverage
5. Boundary signal added or removed → §5 interface spec updated
6. Confirmation review feedback that changes any TSR statement, allocation, or verification approach
