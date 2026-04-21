# Safety Element out of Context (SEooC) Declaration

**Document ID:** ASTR-SAFETY-SEOOC-V0.1
**Date:** 2026-04-20
**Standard:** ISO 26262:2018 — Part 10 §9 (Safety Element out of Context) and Part 11 §4.6 (Application of ISO 26262 to semiconductors)
**IP element:** AstraCore Neo NPU + Sensor-Fusion IP block
**Element revision:** Aligned to repo HEAD on 2026-04-20 + Spec Sheet rev 1.4 (in preparation)
**Classification:** Pre-engagement draft — to be shared with TÜV SÜD India and first NDA evaluation licensee
**Author:** AstraCore safety team (Track 2)
**Reviewers:** TBD (named at W2 process kickoff)
**Status:** v0.1 — first formal release. Successor revisions track FMEDA/HARA/Safety-Manual completion.

---

## 1. Purpose

This document declares **AstraCore Neo** (the NPU + sensor-fusion IP block) as a **Safety Element out of Context (SEooC)** per ISO 26262-10 §9. It defines:

1. The **assumed item context** within which AstraCore Neo is intended to be integrated.
2. The **assumed safety requirements** (functional and technical) the licensee will derive at item level.
3. The **safety mechanisms** AstraCore Neo provides, their assumed diagnostic coverage, and the residual responsibilities placed on the licensee.
4. The **assumptions of use** the licensee must respect for the AstraCore safety case to apply to their item.

Per ISO 26262-10 §9.1, an SEooC "is a safety-related element which is not developed in the context of a specific item." This document is the **contractual interface** between AstraCore (IP supplier) and the licensee (SoC integrator + vehicle OEM).

---

## 2. Element scope

### 2.1 In scope

The SEooC comprises the following deliverable IP and artefacts:

| Category | Item | Repo location |
|---|---|---|
| RTL | NPU compute datapath: `npu_pe`, `npu_systolic_array`, `npu_tile_ctrl`, `npu_top`, `npu_dma`, `npu_sram_bank`, `npu_sram_ctrl`, `npu_softmax`, `npu_layernorm`, `npu_activation`, `npu_fp` | `rtl/npu_*` |
| RTL | Sensor I/O: `mipi_csi2_rx`, `radar_interface`, `lidar_interface`, `imu_interface`, `ultrasonic_interface`, `gnss_interface`, `canfd_controller`, `ethernet_controller`, `pcie_controller` | `rtl/*_interface`, `rtl/canfd_controller`, `rtl/mipi_csi2_rx`, `rtl/ethernet_controller`, `rtl/pcie_controller` |
| RTL | Fusion: `dms_fusion`, `lane_fusion` | `rtl/dms_fusion`, `rtl/lane_fusion` |
| RTL | Perception primitives: `gaze_tracker`, `head_pose_tracker`, `cam_detection_receiver`, `det_arbiter`, `object_tracker` | `rtl/*` |
| RTL | Decoding/transform: `can_odometry_decoder`, `coord_transform`, `ego_motion_estimator` | `rtl/*` |
| RTL | Safety mechanisms: `tmr_voter`, `ecc_secded`, `safe_state_controller`, `plausibility_checker`, `fault_predictor` | `rtl/*` |
| RTL | Vehicle dynamics: `aeb_controller`, `ldw_lka_controller`, `ttc_calculator` | `rtl/*` |
| RTL | Infrastructure: `inference_runtime`, `mac_array`, `sensor_sync`, `ptp_clock_sync`, `thermal_zone` | `rtl/*` |
| RTL | Top: `astracore_top`, `astracore_system_top`, `astracore_fusion_top` | `rtl/astracore_*` |
| Toolchain | ONNX 2.0 loader, INT8/INT4/INT2 quantiser, fusion passes, im2col conv2d, simulator runtime | `tools/npu_ref/`, `astracore/` |
| Toolchain | YAML configuration (`astracore configure`), apply + bench + multistream + replay | `astracore/` |
| Toolchain | Plugin registries (ops, quantisers, backends) per `pyproject.toml` entry-points | `astracore/registry.py`, `pyproject.toml` |
| Safety artefact | This SEooC declaration | `docs/safety/seooc_declaration_v0_1.md` |
| Safety artefact | ISO 26262 gap analysis | `docs/safety/iso26262_gap_analysis_v0_1.md` |
| Safety artefact | Safety Manual (in preparation, v0.1 by W2) | `docs/safety/safety_manual_v0_1.md` |
| Safety artefact | FMEDA reports (in preparation, per-module from W2-W10) | `docs/safety/fmeda/` |
| Safety artefact | Fault-injection campaign reports (in preparation, from W3) | `docs/safety/fault_injection/` |
| Safety artefact | Tool Confidence Level (TCL) evaluations (in preparation, W7) | `docs/safety/tcl/` |

### 2.2 Out of scope (licensee-supplied)

The following are **explicitly outside** the AstraCore SEooC. The licensee provides them and is responsible for their safety cases:

| Out-of-scope item | Rationale | Licensee responsibility |
|---|---|---|
| Memory PHY (LPDDR5X, HBM3, DDR4) | Silicon-only IP; AstraCore exposes AXI/AHB interface | Licensee selects + integrates PHY IP; covers in their item-level safety case |
| Image Signal Processor (ISP) | Not in AstraCore RTL; partner-supplied | Licensee integrates Sony/OmniVision/their own ISP |
| UCIe chiplet interconnect | Silicon + package IP | Licensee licenses UCIe IP (Synopsys, Cadence) |
| C-V2X / cellular modem | Not in AstraCore RTL | Licensee integrates Qualcomm / Autotalks / etc. |
| Power management IC (PMIC) | Board-level | Licensee's reference design |
| Package / thermal solution | Silicon-only | Licensee + foundry |
| Final 7 nm tape-out | Silicon-only | Licensee or foundry partner |
| External crypto IP beyond OpenTitan-supplied | Out of OpenTitan scope | Licensee adds (e.g., HSM, eMRAM) |
| Item-level HARA on the licensee's vehicle | Licensee owns item definition | Licensee derives item-level safety goals |
| Vehicle-level safety case | Vehicle OEM owns | Vehicle OEM |

### 2.3 Boundary signals

The IP block exposes the following safety-relevant interface signals to the licensee SoC. The licensee must connect these to their item-level safety architecture.

| Signal | Direction | Purpose | Required licensee handling |
|---|---|---|---|
| `safe_state_active` | Out | AstraCore has entered safe state | Licensee SoC must respond per its item-level safety concept (e.g., disable downstream actuators, alert vehicle controller) |
| `fault_detected[15:0]` | Out | Fault category bitfield | Licensee logs + responds per category |
| `external_safe_state_request` | In | Force AstraCore to safe state from outside | Licensee asserts on external fault (e.g., supply, package thermal) |
| `lockstep_compare_in[N-1:0]` | In/Out | Optional dual-rail / lockstep with sibling instance | Licensee may instantiate two AstraCore IPs in lockstep for ASIL-D decomposition |
| `tmr_disagree_count[7:0]` | Out | Cumulative TMR disagreements (saturating) | Licensee uses for prognostic alerting |
| `ecc_corrected_count[15:0]` | Out | Cumulative ECC single-bit corrections | Licensee uses for prognostic alerting |
| `ecc_uncorrected_count[7:0]` | Out | Cumulative ECC double-bit detections | Licensee escalates per ASIL |
| `watchdog_kick` | In | Periodic kick from licensee SoC supervisor | Licensee must service per documented timing in Safety Manual |
| `dft_isolation_enable` | In | Disable DFT during normal operation | Licensee asserts low during mission mode |
| `clock_monitor_alert` | Out | Internal clock monitor flagged out-of-bounds | Licensee may also monitor independently |

These boundary signals form the contract between AstraCore IP and the licensee's safety architecture. The Safety Manual will document timing constraints, polarities, and reset behavior in detail.

---

## 3. Assumed item context

Per ISO 26262-10 §9.3, an SEooC must declare assumptions about the item in which it will be integrated. The AstraCore SEooC is developed under the following **assumed item**:

> **Assumed item:** *Automotive AI inference and sensor-fusion subsystem in a zonal or domain Electronic Control Unit (ECU) integrated into a passenger vehicle E/E architecture compliant with ISO 26262:2018.*

### 3.1 Assumed operational context

| Aspect | Assumption |
|---|---|
| Vehicle class | Passenger car or light commercial vehicle (M1, N1 per ECE) |
| Operational design domain (ODD) | On-road driving, including urban / highway / parking. Not certified for off-road, racing, or military use. |
| Driver presence | SAE Level 2 to Level 4 driver-assistance and automated-driving functions |
| Environmental temperature (junction) | Assumed –40 °C to +125 °C operating range; licensee's package + thermal design enforces |
| Vibration | Per ISO 16750-3 |
| EMC | Per CISPR 25 / ISO 11452 |
| Mission profile | 15-year vehicle life, 8000 hr operation, 240 thermal cycles |
| Power | Single-rail digital VDD provided by licensee PMIC at nominal 0.75 V (7 nm assumption); licensee handles PVT margining |

Any deviation from these assumptions invalidates downstream safety arguments and requires re-analysis.

### 3.2 Assumed item-level safety functions

The licensee is assumed to integrate AstraCore Neo as part of an item realising one or more of the following automotive safety functions. ASIL classification is the licensee's responsibility (per item-level HARA), but AstraCore Neo is **designed to support** these classifications.

**Backing:** the assumed ASILs in the table below are derived from a formal HARA on three reference use cases per ISO 26262-3 §6, documented at `docs/safety/hara_v0_1.md`. The HARA explicitly enumerates hazardous events with S/E/C ratings and derives ASIL via Table 4 of ISO 26262-3. Until v0.1 of this declaration the ASILs were placeholders; the HARA replaces that with derivation evidence.

| Item-level safety function | Assumed ASIL | AstraCore role |
|---|---|---|
| Forward Collision Warning + Automatic Emergency Braking | ASIL-D | Object detection, sensor fusion (radar+camera+lidar), TTC calculation |
| Lane Departure Warning + Lane Keep Assist | ASIL-B | Lane fusion, perception |
| Driver Monitoring + drowsiness detection | ASIL-B | DMS fusion, gaze + head-pose tracking |
| Surround View / Park Assist | QM (initial), ASIL-B (with safety override) | Ultrasonic + camera fusion |
| Cabin Voice Command | QM | Microphone array I/O (no safety function) |
| In-cabin person occupancy detection | ASIL-A to ASIL-B | Camera / ToF fusion |

For ASIL-D item-level functions, the licensee may apply ASIL decomposition (per ISO 26262-9 §5) using AstraCore's TMR / ECC / lockstep mechanisms or by integrating two AstraCore IP instances in dual-channel architecture. The `lockstep_compare_in` interface is provided for this purpose.

### 3.3 Assumed item-level safety goals

The licensee will derive item-level safety goals from their HARA. The AstraCore SEooC is developed to **support** safety goals of the following pattern:

> *"The system shall not produce a perception, fusion, or decision output that would cause a hazardous event in the assumed ODD without notification to the supervising controller within [licensee-defined Fault Tolerant Time Interval]."*

AstraCore guarantees fault detection and signaling within an internal Fault Detection Time Interval (FDTI) documented in the Safety Manual; the licensee's FTTI must accommodate FDTI + their downstream signaling latency.

**Derived safety goals.** `docs/safety/hara_v0_1.md` §2.4, §3.4, §4.4 enumerate eleven SGs across the three reference use cases, with ASIL ranging from QM to ASIL-D. SG → ASR (Assumed Safety Requirement) traceability lives in `docs/safety/hara_v0_1.md` §6 and maps each SG to one or more ASRs in §4.1 of this document.

---

## 4. Safety requirements assumed at item level

Per ISO 26262-10 §9.4, the SEooC must declare the safety requirements assumed to be derived at item level. AstraCore Neo is developed against the following assumed safety requirements (ASR = Assumed Safety Requirement):

**FSR derivation from these ASRs.** Each ASR below is referenced from one
or more **Functional Safety Requirements (FSRs)** in
`docs/safety/functional_safety_concept_v0_1.md` §3. The FSC §4 coverage
matrix verifies every ASR has at least one FSR addressing it (open
items: ASR-HW-02/03 wait for F4-A-1.1; ASR-HW-12/13 wait for OpenTitan
crypto integration; ASR-SW-05 waits for C++ runtime).

### 4.1 Hardware safety requirements (assumed)

| ID | Requirement | Assumed ASIL | Mechanism |
|---|---|---|---|
| ASR-HW-01 | Detect and signal single-bit transient faults in compute datapath within FDTI ≤ 10 µs | ASIL-D | TMR voter on critical paths + ECC on register files |
| ASR-HW-02 | Detect and correct single-bit faults in SRAM within one access cycle | ASIL-D | SECDED ECC (`ecc_secded`) |
| ASR-HW-03 | Detect double-bit faults in SRAM and signal upstream | ASIL-D | SECDED ECC double-bit detect |
| ASR-HW-04 | Detect stuck-at faults in datapath within mission profile coverage | ASIL-B | Periodic LBIST + TMR voter coverage |
| ASR-HW-05 | Drive IP block to safe state on aggregated fault | ASIL-D | `safe_state_controller` |
| ASR-HW-06 | Detect clock loss / glitch / out-of-bounds frequency | ASIL-B | Internal clock monitor (RTL counterpart to be added per gap analysis Part 5 §11) |
| ASR-HW-07 | Detect implausible sensor inputs (range / rate / cross-sensor) | ASIL-B | `plausibility_checker` |
| ASR-HW-08 | Detect sensor-stuck condition within configured watchdog timeout | ASIL-B | Per-sensor watchdog (e.g., `WATCHDOG_CYCLES` in `dms_fusion`) |
| ASR-HW-09 | Provide diagnostic coverage ≥ 90 % for single-point faults (SPFM) | ASIL-B | Aggregate FMEDA target |
| ASR-HW-10 | Provide diagnostic coverage ≥ 60 % for latent faults (LFM) | ASIL-B | Aggregate FMEDA target |
| ASR-HW-11 | Provide PMHF ≤ 100 FIT (ASIL-B) / ≤ 10 FIT (ASIL-D) | ASIL-B/D | Aggregate FMEDA result |
| ASR-HW-12 | Support secure boot of weights and configuration | ASIL-B | OpenTitan AES + RSA + SHA-256 (Track 3) |
| ASR-HW-13 | Detect tampering with model weights | ASIL-B | Boot-time RSA-2048 + SHA-256 signature verification (static integrity); SECDED ECC on weight SRAM (runtime integrity) |

### 4.2 Software safety requirements (assumed)

| ID | Requirement | Assumed ASIL | Mechanism |
|---|---|---|---|
| ASR-SW-01 | Compiler shall not silently change model semantics | QM (development tool) | Tool Confidence Level evaluation per Part 8 §11 |
| ASR-SW-02 | Quantiser shall produce bit-exact output vs documented reference | ASIL-B | Test suite + bit-exact mirror (`tools/npu_ref/`) |
| ASR-SW-03 | Runtime shall enforce safety policies declared in YAML config | ASIL-B | `astracore/apply.py` + `safety_policies` schema |
| ASR-SW-04 | Runtime shall log all safety-relevant events with timestamp | ASIL-B | `astracore/registry.py` event hooks |
| ASR-SW-05 | C++ runtime (when delivered) shall be developed per MISRA-C and ISO 26262-6 | ASIL-B | Track 1 future deliverable |

### 4.3 Process safety requirements (assumed)

| ID | Requirement |
|---|---|
| ASR-PROC-01 | Configuration management per ISO 26262-8 §7 — git + tagged baselines |
| ASR-PROC-02 | Change management per ISO 26262-8 §8 — change-impact analysis on safety-relevant files |
| ASR-PROC-03 | Documentation control per ISO 26262-8 §10 — document IDs, revision history, review records |
| ASR-PROC-04 | Tool qualification per ISO 26262-8 §11 — TCL evaluations for Verilator, Yosys, OpenROAD, ASAP7, OpenTitan |
| ASR-PROC-05 | Coding guidelines per ISO 26262-6 §8.4.5 + ISO 26262-11 §6 — MISRA-C / MISRA-Python + RTL style guide |
| ASR-PROC-06 | Confirmation measures per ISO 26262-2 §7 — independent reviewer + audit + assessment |

---

## 5. Safety mechanisms — declared coverage (preliminary)

> Numbers in this section are **preliminary targets**, not yet measured. They will be replaced with measured values from the FMEDA campaign (W2-W10) and fault-injection campaign (W3-W9). Each row is a placeholder for a measured number with a deadline.

| Mechanism | Module | Failure mode covered | Target DC | Measurement deadline |
|---|---|---|---|---|
| TMR voter | `tmr_voter`, used in `dms_fusion` and `safe_state_controller` | Stuck-at, single-event upset (SEU) on voted nets | ≥ 99 % | W4 fault-injection |
| SECDED (72,64) | `ecc_secded`, all SRAM banks | Single-bit error: correct; double-bit: detect | ≥ 99 % single, ≥ 99.9 % double | W8 formal proof |
| Safe-state controller | `safe_state_controller` | Aggregated fault → defined safe state | ≥ 95 % (correct entry) | W9 fault-injection |
| Plausibility checker | `plausibility_checker` | Implausible sensor inputs (range, rate, cross-sensor) | ≥ 90 % | W9 fault-injection |
| Watchdog (per sensor) | inline in `dms_fusion`, `gaze_tracker`, etc. | Sensor stuck / lost | ≥ 95 % | W4-W9 fault-injection |
| Fault predictor | `fault_predictor` | Pattern-based prognostic | Diagnostic only (not safety-claim) | n/a |
| Clock monitor | (RTL gap — to be added) | Clock loss, glitch, freq out of bounds | ≥ 95 % | W6 RTL + W9 fault-injection |
| LBIST (logic BIST) | (RTL gap — to be considered) | Stuck-at coverage at boot / periodic | ≥ 90 % @ boot | Possibly post-W16 |
| Boot-time integrity check | OpenTitan SHA-256 + RSA-2048 | Weight / config tampering | ≥ 99.9 % (cryptographic) | W7 (Track 3) |

---

## 6. Assumptions of use (binding on the licensee)

Per ISO 26262-11 §4.7, an SEooC must enumerate **assumptions of use (AoU)** that the licensee is contractually required to respect. If any AoU is violated, the AstraCore safety case does not transfer to the licensee's item-level safety case without re-analysis.

### 6.1 Operational environment

- **AoU-1.** Junction temperature shall be kept within –40 °C to +125 °C by licensee package + cooling.
- **AoU-2.** Supply voltage shall be regulated within ±5 % of nominal by licensee PMIC.
- **AoU-3.** Clock source shall meet jitter spec documented in Safety Manual §4 (TBD).
- **AoU-4.** Reset signal shall meet asynchronous-assert / synchronous-deassert protocol per Safety Manual §5 (TBD).

### 6.2 Functional integration

- **AoU-5.** Licensee shall connect `safe_state_active` to a downstream safe-state actuator (e.g., disable inference output to vehicle controller) per their item-level safety concept.
- **AoU-6.** Licensee shall service `watchdog_kick` within the period documented in Safety Manual §6.
- **AoU-7.** Licensee shall not drive `dft_isolation_enable` high during mission mode.
- **AoU-8.** Licensee shall route `ecc_uncorrected_count` and `tmr_disagree_count` to a supervisor that escalates per their item-level ASIL.

### 6.3 Software / data integration

- **AoU-9.** Models loaded into AstraCore for inference shall be cryptographically signed by a key under the licensee's control; the AstraCore secure-boot flow verifies the signature.
- **AoU-10.** Models shall be quantised using the documented AstraCore quantiser flow (or an equivalent flow that produces bit-exact output verifiable against the AstraCore reference); models quantised by other flows are out of scope of the safety case.
- **AoU-11.** Sensor inputs shall be calibrated per the manufacturer's procedure; AstraCore plausibility checker assumes calibrated inputs.
- **AoU-12.** Licensee shall not modify AstraCore RTL or Python toolchain without re-running the safety verification suite and updating the safety case.

### 6.4 Process

- **AoU-13.** Licensee shall sign a Development Interface Agreement (DIA) per ISO 26262-8 §5 with AstraCore before integration.
- **AoU-14.** Licensee shall report any anomalies discovered during integration to AstraCore within 30 days for inclusion in the AstraCore field-monitoring program.
- **AoU-15.** Licensee shall produce an item-level HARA and derive item-level safety goals; AstraCore's assumed safety requirements (§4) are subordinate to and must align with licensee's derived requirements.

### 6.5 Configuration

- **AoU-16.** AstraCore parameters shall be set within ranges documented in Safety Manual §3:
  - `N_ROWS × N_COLS` MAC array size: between 4×4 (development) and 48×512 (full spec); intermediate values must be characterised by licensee.
  - `WEIGHT_DEPTH` shall match `N_ROWS × N_COLS` per the latent-issue caveat documented in Safety Manual §3.2 (per `pre_awsf1_gaps_complete.md`).
  - Watchdog timeouts shall match licensee's FTTI minus signaling latency.
- **AoU-17.** Sparsity engine (when enabled) shall be exercised with QAT-trained models only; PTQ models on sparsity engine are out of scope of the safety case until characterised.

---

## 7. Verification & validation strategy

Per ISO 26262-10 §9.5, the SEooC must declare its V&V strategy.

### 7.1 Verification activities

| Activity | Coverage | Repo evidence | Status |
|---|---|---|---|
| Unit tests (Python) | 1140 collected, ~792+ pass before unrelated failure | `tests/`, `pytest --collect-only` | ✅ existing |
| Bit-exact RTL ↔ Python mirror tests (cocotb) | 6/6 npu_top, 5/5 systolic, 10/10 softmax/layernorm, 5/5 FP MAC sim-gate | `sim/`, `tools/run_verilator_*.sh` | ✅ existing |
| OpenLane sky130 synthesis batch | 32/32 modules pass | `asic/runs/` | ✅ existing |
| Vivado FPGA build (Artix-7) | 100 MHz target, end-to-end fusion | `constraints/`, `fpga/` | ✅ existing |
| AWS F1 VU9P build (planned) | 64×64 = 4096 MACs | (pending — Track 1 W1-6) | 🟡 planned |
| Verilator full-array simulation (planned) | 48×512 = 24,576 MACs | (pending — Track 1 W2-10) | 🟡 planned |
| ASAP7 7nm synthesis projection (planned) | timing + area at 2.5 GHz target | (pending — Track 1 W5-13) | 🟡 planned |
| Formal verification — TMR + ECC properties | Symbiyosys / Yosys-SBY proofs | (pending — Track 2 W7-9) | 🟡 planned |
| Fault-injection campaign | ≥10K injections per safety-critical module | `docs/safety/fault_injection/` | 🟡 planned (W3-9) |
| MISRA-C/SystemVerilog static analysis | Track 1 RTL + future C++ runtime | (pending — Track 2 W3) | 🟡 planned |

### 7.2 Validation activities (item-level — licensee-led, AstraCore supports)

- Vehicle-level HARA and derivation of item-level safety goals: **licensee**.
- Vehicle-level fault tree analysis incorporating AstraCore as a branch: **licensee**.
- Vehicle-level operational testing per ISO 26262-4 §11: **licensee + vehicle OEM**.
- Field monitoring per ISO 26262-7 §6: **licensee**, with AstraCore providing field-failure-mode database template.

---

## 8. Confirmation measures

Per ISO 26262-2 §7, the following confirmation measures are planned:

| Measure | Target | Owner | Schedule |
|---|---|---|---|
| Confirmation review of HARA + safety concept | Internal independent reviewer (named at W2) | Track 2 lead | W4 |
| Functional safety audit | TÜV SÜD India (pre-engagement W6) | TÜV | W14 (post safety-case-v0.1) |
| Functional safety assessment | TÜV SÜD India | TÜV | W26 (post safety-case-v1.0) |

---

## 9. Document control

| Field | Value |
|---|---|
| Document ID | ASTR-SAFETY-SEOOC-V0.1 |
| Revision | 0.1 |
| Revision date | 2026-04-20 |
| Author | TBD — Track 2 lead to be named at W1 hiring/assignment. Currently authored by founder + collaborator. |
| Reviewer | TBD — independent reviewer to be named at W2 sign-off (Part 2 §7 confirmation measures) |
| Approver | TBD — Safety Manager role to be created and assigned at W1; founder approves until then |
| Distribution | Internal + TÜV SÜD India + first NDA evaluation licensee |
| Retention | 15 years post-product-discontinuation per ISO 26262-8 §10 |
| Supersedes | None — this is v0.1 |

### 9.1 Revision triggers

This SEooC declaration must be re-issued (with revision bump) when **any** of the following occurs:

1. HARA on a reference use case (W3-4) yields a safety goal not previously assumed → update §3 and §4.
2. FMEDA result (W2-10) shows measured DC differs by >5 % from declared target → update §5.
3. Fault-injection campaign (W3-9) shows uncovered failure mode → update §5 and possibly add safety mechanism.
4. New safety mechanism added (e.g., RTL clock monitor at W6) → update §5 and §6.
5. AoU is added, removed, or materially changed → update §6.
6. Boundary signal added/removed → update §2.3.
7. Spec sheet rev 1.4 wording final → align §1 and reissue.

### 9.2 Open items at v0.1

Items deferred to v0.2:

- Specific FDTI numbers per ASR-HW-01 etc. (need clock spec freeze first)
- Specific watchdog timing values per AoU-6 (need licensee profile)
- Named internal reviewer for confirmation measures
- Named Track 2 lead, Safety Manager, and Approver per §9
- Cross-references to FMEDA reports once produced (`docs/safety/fmeda/`)
- Cross-reference to Safety Manual v1.0 once released (`docs/safety/safety_manual_*.md`)
- Soft Error Rate (SER) analysis result per ISO 26262-11 §7 (W10 deliverable)
- RTL clock monitor module — flagged as gap in §5; needs to be added to `rtl/` then re-cataloged in §2.1
- LBIST coverage decision — flagged in §5 as "to be considered"; if added, update §5 + §7.1

---

## Appendix A — Mapping of this SEooC to ISO 26262 clauses

| ISO 26262 clause | This document |
|---|---|
| 26262-2 §5 (Safety culture) | §9 (revision triggers, document control) — and forthcoming Safety Policy doc |
| 26262-2 §6 (FSM during development) | §8 (confirmation measures) |
| 26262-2 §7 (Confirmation measures) | §8 |
| 26262-3 §5 (Item definition) | §3 (assumed item context) |
| 26262-3 §6 (HARA) | §3.3 (assumed safety goals) — to be replaced with measured HARA at W3-4 |
| 26262-3 §7 (Functional safety concept) | §4 (assumed safety requirements) |
| 26262-3 §8 (ASIL determination) | §3.2 (assumed item-level ASIL) |
| 26262-4 §6 (Technical safety concept) | §4.1, §4.2 (HW + SW safety requirements) |
| 26262-4 §7 (System architectural design) | §2.3 (boundary signals) |
| 26262-5 §7 (HW safety requirements specification) | §4.1 |
| 26262-5 §8 (HW design) | Repo: `rtl/`. Per-module safety analysis in FMEDA reports |
| 26262-5 §9 (Safety analyses) | Per-module FMEDA in `docs/safety/fmeda/` (W2-10) |
| 26262-5 §10 (Verification of HW safety analyses) | §7.1, formal verification W7-9 |
| 26262-5 §11 (HW integration & testing) | §7.1, fault-injection W3-9 |
| 26262-6 §7 (SW safety requirements) | §4.2 |
| 26262-6 §8 (SW architectural design) | `astracore/__init__.py` (plugin registries) + repo |
| 26262-8 §5 (Distributed development → DIA) | §6.5 AoU-13, DIA template W6 |
| 26262-8 §6 (Safety case management) | This document is part 1 of the safety case |
| 26262-8 §7 (Configuration management) | §4.3 ASR-PROC-01 |
| 26262-8 §8 (Change management) | §4.3 ASR-PROC-02; revision triggers §9.1 |
| 26262-8 §10 (Documentation management) | §9 (document control) |
| 26262-8 §11 (Tool qualification) | §4.3 ASR-PROC-04, TCL evaluations W7 |
| 26262-9 §5 (ASIL decomposition) | §3.2 (lockstep option for ASIL-D) |
| 26262-9 §6 (Criticality analysis) | Per-module criticality W5 |
| 26262-9 §7 (Dependent failure analysis) | CCF analysis W14-15 |
| 26262-9 §8 (Safety analysis FMEA/FTA) | W4-8 |
| 26262-10 §9 (SEooC) | This entire document |
| 26262-11 §4.6 (IP supplier responsibilities) | This entire document |
| 26262-11 §4.7 (Assumptions on use) | §6 |
| 26262-11 §6 (Tool confidence in HW design tools) | §4.3 ASR-PROC-04 |
| 26262-11 §7 (Soft errors) | §5 (TMR + ECC), SER calculation W10 |
