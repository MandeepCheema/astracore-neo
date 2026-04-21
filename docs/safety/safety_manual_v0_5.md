# AstraCore Neo IP — Safety Manual

**Document ID:** ASTR-SAFETY-MANUAL-V0.5
**Date:** 2026-04-20
**Status:** v0.5 — licensee-critical sections filled out (§3.4 safe-state, §4 clock/reset, §5 init sequencing, §6 watchdog, §7 fault signalling, §8 secure boot, §10 AoU index, §11 licensee verification). Remaining TBDs: §2.1 safety-annotated block diagram, §9 diagnostic services, §12 field anomalies. Targets v1.0 release at W12 for TÜV interim review.
**Standard:** ISO 26262-10 §9 + ISO 26262-11 §4.7 (IP supplier safety manual)
**Supersedes:** ASTR-SAFETY-MANUAL-V0.1 (`docs/safety/safety_manual_v0_1.md` — kept for revision history)

**Companion documents:**
- `docs/safety/seooc_declaration_v0_1.md` — SEooC declaration (the contractual element)
- `docs/safety/hara_v0_1.md` — Hazard Analysis & Risk Assessment + 11 Safety Goals
- `docs/safety/functional_safety_concept_v0_1.md` — 25 Functional Safety Requirements derived from the Safety Goals
- `docs/safety/iso26262_gap_analysis_v0_1.md` — process gap analysis
- `docs/safety/findings_remediation_plan_v0_1.md` — F4 RTL hardening plan
- `docs/best_in_class_design.md` — strategic context (§7 founder direction)

> **Purpose.** The Safety Manual is the document a licensee uses to integrate AstraCore Neo IP into their item-level safety case. It tells the licensee **what to do, what not to do, what assumptions to respect, and what diagnostics to wire** in order to inherit the AstraCore safety case.
>
> The SEooC declaration is the *contract*; the Safety Manual is the *user guide*. Both are required deliverables to a licensee.

---

## 1. Introduction

### 1.1 Scope

This Safety Manual covers the AstraCore Neo NPU + sensor-fusion IP block as licensed (refer to SEooC §2 for the in-scope artefact list). It does **not** cover:

- The licensee's SoC integration (memory PHY, package, supervisor MCU, clock distribution)
- The licensee's vehicle controller, actuator chain, or driver HMI
- The vehicle-level safety case (vehicle OEM responsibility)
- Item-level HARA on the licensee's vehicle program (replaces the assumed HARA at `docs/safety/hara_v0_1.md`)

Out-of-scope items are tracked exhaustively in spec sheet rev 1.4 §12 (`docs/spec_sheet_rev_1_4.md`).

### 1.2 Intended audience

- Licensee functional-safety manager
- Licensee SoC integration engineers
- Licensee verification engineers
- Vehicle OEM safety reviewer (read-only)
- TÜV / external assessor (audit reference)

### 1.3 How to read this manual

Each section pairs **what AstraCore provides** with **what the licensee must do**. Mandatory steps are marked **[REQUIRED]**; recommended steps are **[RECOMMENDED]**.

Cross-references use the convention `<Doc>:§<Section>` — e.g., `SEooC:§2.3` means SEooC declaration section 2.3.

---

## 2. IP element overview

### 2.1 Block diagram
TBD — link to `docs/architecture.md` once a safety-annotated version exists. Plan: produce safety-annotated block diagram by W4 with TMR/ECC/safe-state nets highlighted. Until then, FSC:§5 has a textual diagram of the safety-relevant subset.

### 2.2 Lifecycle

| Phase | AstraCore role | Licensee role |
|---|---|---|
| Development | Ship RTL + SDK + safety-case docs at a tagged baseline | Receive baseline; perform item-level HARA |
| Integration | Provide integration support per Development Interface Agreement (DIA) | Integrate IP into SoC RTL; synthesize at target node; perform item-level FMEDA + fault injection |
| Operation | Field-monitoring database template; AoU-14 anomaly intake | Deploy to vehicle; report anomalies per AoU-14 |
| Decommissioning | Long-tail support per DIA terms | Document EOL per ISO 26262-7 §6 |

### 2.3 Versioning and configuration management

- Repo HEAD on date of release is the baseline.
- Safety-relevant changes follow change-impact analysis per ISO 26262-8 §8.
- Each licensee receives a tagged baseline; field updates require licensee re-verification.
- Baseline integrity check: every shipped baseline includes a manifest listing all RTL + tooling + safety-doc files with SHA-256. **[REQUIRED]** licensee verifies the manifest before integration.

---

## 3. Configuration parameters

### 3.1 MAC array sizing

- `N_ROWS × N_COLS` valid range: 4×4 (development) ↔ 48×512 (full spec).
- Performance / power / area scale documented in `docs/best_in_class_design.md` §7.4 once Track 1 measurements complete.
- **[REQUIRED]** Set `WEIGHT_DEPTH = N_ROWS × N_COLS` per the latent issue documented in `memory/pre_awsf1_gaps_complete.md`. Failure to do so produces silent data corruption at compile time. The default value in `rtl/npu_top/npu_top.v` auto-derives, but any caller overriding the parameter must respect this constraint.

### 3.2 Sensor I/O selection

The licensee may instantiate any subset of the sensor-I/O modules listed in SEooC:§2.1. Disabled interfaces should have their AXI / control inputs tied off per the conventional Verilog practice.

| Interface | Source | Disable pattern |
|---|---|---|
| `mipi_csi2_rx` | `rtl/mipi_csi2_rx/` | Hold `rx_enable` low; tie data inputs to 0 |
| `radar_interface` | `rtl/radar_interface/` | Hold `enable` low |
| `lidar_interface` | `rtl/lidar_interface/` | Hold `enable` low |
| `ultrasonic_interface` | `rtl/ultrasonic_interface/` | Hold `enable` low |
| `imu_interface` | `rtl/imu_interface/` | Hold `enable` low |
| `gnss_interface` | `rtl/gnss_interface/` | Hold `enable` low |
| `canfd_controller` | `rtl/canfd_controller/` | Hold `enable` low |
| `ethernet_controller` | `rtl/ethernet_controller/` | Hold `enable` low |
| `pcie_controller` | `rtl/pcie_controller/` | Hold `enable` low |

**[REQUIRED]** disabled interfaces shall not contribute to `fault_detected[]`. Tie-off pattern is licensee's responsibility.

### 3.3 Safety mechanism enables

| Mechanism | Default | Configurable? |
|---|---|---|
| TMR voter (`tmr_voter`) | always-on for safety-critical paths | NO — compiled-in |
| SECDED ECC (`ecc_secded`) | RTL primitive available; combinational ECC wrapper at `rtl/npu_sram_bank_ecc/` ✅. **Today npu_top still instantiates bare `npu_sram_bank` (no ECC); npu_top swap to the wrapper is F4-A-1.1 in `docs/safety/findings_remediation_plan_v0_1.md` Phase A — pending WSL cocotb regression cycle.** | YES per-bank instantiation post-F4-A-1.1; **[REQUIRED]** for ASIL-B and above |
| Plausibility checker (`plausibility_checker`) | per-sensor enable | YES via instantiation |
| Per-sensor watchdog | per-sensor timeout configurable | YES — see `WATCHDOG_CYCLES` parameters |
| Safe-state controller (`safe_state_controller`) | always-on aggregator | NO — compiled-in |
| Fault predictor (`fault_predictor`) | always-on rule-based today; ML upgrade per F1-A9 | NO |

### 3.4 Safe-state behaviour

The `safe_state_controller` aggregates fault flags from every safety mechanism and drives the chip through the four-state ladder:

| State | Encoding | Conditions | Outputs |
|---|:---:|---|---|
| NORMAL | 2'd0 | No fault for ≥ RECOVER_TIME_MS | `safe_state_active`=0; max_speed = 130 km/h |
| ALERT | 2'd1 | warning_faults set, OR critical_faults set briefly | `alert_driver`=1; max_speed = 130 km/h |
| DEGRADE | 2'd2 | critical_faults sustained ≥ ALERT_TIME_MS (default 2000 ms) | `alert_driver`=1, `limit_speed`=1; max_speed = 60 km/h |
| MRC (Minimal Risk Condition) | 2'd3 | critical_faults sustained ≥ DEGRADE_TIME_MS (default 3000 ms) | `mrc_pull_over`=1; max_speed = 5 km/h; **absorbing — only operator_reset returns to NORMAL** |

**[REQUIRED]** licensee shall connect `safe_state_active` to a downstream actuator that responds per the licensee item-level safety concept. Typical responses by state:

| State | Typical licensee response (item-level — adjust per HARA) |
|---|---|
| NORMAL | Normal vehicle operation; no driver alert |
| ALERT | Visual + audible driver alert; full driving envelope retained |
| DEGRADE | Limit ADAS features; reduce max-allowed setpoint; audible warning |
| MRC | Trigger safe pull-over manoeuvre; brake to stop on roadside; emergency hazard lights |

**[REQUIRED]** licensee shall NOT bypass `safe_state_active` in mission mode. Bypassing requires DIA amendment and re-validation of the safety case.

**[RECOMMENDED]** licensee implements an independent supervisor watchdog that forces vehicle to safe state if AstraCore's `safe_state_active` line is itself stuck (CCF mitigation per ISO 26262-9 §7).

#### 3.4.1 `safe_state_active` deassert

- Auto-recovery from ALERT or DEGRADE → one level lower per RECOVER_TIME_MS (default 5000 ms) of cleared faults
- MRC does NOT auto-recover; **[REQUIRED]** licensee provides `operator_reset` 1-cycle pulse to clear

---

## 4. Clock and reset

### 4.1 Clock specification

| Parameter | Value | Notes |
|---|---|---|
| Nominal clock frequency | Per licensee's tape-out target (e.g., 100 MHz on Artix-7 FPGA, 2.5–3.2 GHz target on 7 nm silicon) | AstraCore RTL is fully synchronous to `clk` |
| Clock domain | Single primary clock (`clk`) | Cross-domain crossings are out of scope of AstraCore IP |
| Maximum jitter (RMS) | < 1 % of clock period | Tighter for safety-critical paths; verify via clock-monitor module (see SEooC §5 clock-monitor RTL gap; F4 follow-up) |
| Duty cycle | 45–55 % | Standard CMOS digital |
| PLL bypass | **[REQUIRED]** must be available | Allows defeat-tested mode where the safety case relies on a known-good external reference clock |

### 4.2 Reset protocol

- Reset signal: `rst_n` (active-low, single net common to every AstraCore module)
- Assert: **asynchronous** — may be asserted on any cycle
- Deassert: **synchronous to clk** — driver must release rst_n on a rising edge of clk after a stable cycle
- Minimum reset width: **16 clk cycles** (allows internal counters to clear and FSM to settle)
- Post-reset stabilisation: **1 cycle** before any control input is sampled (most modules use `if (!rst_n)` in their always blocks, so one synchronised cycle is sufficient)

### 4.3 What goes wrong if you violate clock/reset

| Violation | Consequence | Detection |
|---|---|---|
| Glitch on clock | FF metastability, wrong vote outcomes | Clock-monitor RTL (per SEooC §5 — gap, F4 follow-up); meanwhile licensee provides external clock-fault detection |
| Clock loss > 1 cycle | All FFs hold previous value; safety mechanisms inert | Same — clock monitor or external watchdog |
| Asynchronous reset deassert | Multiple FFs sample mid-glitch, become metastable; some safety mechanisms may stick at wrong value | Reset-controller best practice — synchronise deassert via dedicated reset-bridge |
| Reset width < 16 cycles | Counters and FSMs may not fully clear; downstream behaviour undefined | Licensee enforces in reset controller |

---

## 5. Reset and initialization sequencing

### 5.1 Power-on sequence

1. **POR (Power-On Reset).** Licensee hardware shall hold `rst_n=0` until VDD is stable per the silicon-vendor's POR specification.
2. **Reset width.** Hold `rst_n=0` for ≥ 16 clk cycles after VDD stable.
3. **Synchronous deassert.** Release `rst_n=1` on a rising edge of clk.
4. **Boot integrity verification.** (Track 3 OpenTitan integration — see §8.) Until OpenTitan lands, this step is performed by host-side software at the SDK layer.
5. **Mission-mode handshake.** AstraCore asserts `safe_state_active=0` (NORMAL state) when boot completes successfully. Licensee supervisor MUST wait for `safe_state_active=0` before issuing any inference work or accepting AstraCore output as safety-relevant.
6. **Mission mode.** Licensee supervisor begins issuing `watchdog_kick` per §6.

### 5.2 What "boot complete" means today

In the absence of secure boot (Track 3 not yet integrated), boot-complete means: rst_n is high, AstraCore outputs `safe_state_active=0`, and the supervisor has issued the first valid `watchdog_kick`. Track 3 OpenTitan integration adds explicit signature verification before this transition.

---

## 6. Watchdog handling

### 6.1 `watchdog_kick` interface

`watchdog_kick` is an INPUT to AstraCore from the licensee supervisor MCU. It is a 1-cycle high pulse that resets AstraCore's watchdog counter.

| Parameter | Value | Notes |
|---|---|---|
| Kick period (default) | 200 ms | Per `WATCHDOG_CYCLES` in `dms_fusion` and similar modules |
| Kick period range | 50 ms – 1000 ms | Configurable per licensee item-level FTTI budget |
| Late-kick window | ±10 % of nominal period | Outside this window the kick is ignored and the watchdog continues counting |
| Pulse width | 1 clk cycle | Longer pulses are tolerated but only the rising edge is sampled |

### 6.2 Watchdog escalation

| Condition | AstraCore response | `fault_detected[]` bit |
|---|---|---|
| Late kick (within tolerance window) | Counter reset; INFO log to telemetry | none |
| Missed kick (counter saturates at WATCHDOG_CYCLES) | SENSOR_FAIL state asserted in `dms_fusion`; equivalent path in other modules | bit 4 |
| 3 consecutive missed kicks | `safe_state_active` asserted (CRITICAL) via safe_state_controller aggregation | bit 4 + bit 6 (MRC if sustained) |

### 6.3 Watchdog timing constraint

**[REQUIRED]** licensee item-level FTTI must accommodate:

```
FTTI ≥ kick_period + 1 clk + downstream signaling latency + actuator latency
```

Default 200 ms kick period gives ~210 ms internal latency budget; licensee adds their actuator chain and confirms the result satisfies item-level FSC FTTI placeholders (FSC:§1.2).

### 6.4 What happens if licensee never kicks

After WATCHDOG_CYCLES of clk with no kick, AstraCore enters SENSOR_FAIL → escalates through ALERT → DEGRADE → MRC per §3.4. From a cold boot with no kicks at all, AstraCore takes (kick_period + ALERT_TIME_MS + DEGRADE_TIME_MS) = ~5.2 s default to reach MRC.

---

## 7. Fault signalling

This is the contract between AstraCore IP and the licensee SoC. The full set of safety-relevant boundary signals is enumerated in SEooC:§2.3; this section gives the runtime semantics.

### 7.1 `safe_state_active` semantics

| Aspect | Specification |
|---|---|
| Direction | Output (AstraCore → licensee supervisor) |
| Type | Single-bit, registered, active-high |
| Assert latency | ≤ FDTI (Fault Detection Time Interval) + 1 clk; FDTI is mechanism-specific (see §7.4) |
| Deassert | Per §3.4.1 |
| Stuck-high mitigation | **[RECOMMENDED]** licensee implements independent supervisor watchdog (CCF mitigation) |
| Stuck-low mitigation | Cross-monitored via `fault_detected[]` aggregate; if fault_detected[] reports faults but safe_state_active stays 0, licensee supervisor flags AstraCore-internal fault |

### 7.2 `fault_detected[15:0]` bitfield

| Bit | Fault category | Source / RTL | Severity | Recommended licensee response |
|:---:|---|---|---|---|
| 0 | TMR voter triple-disagree | `tmr_voter.triple_fault` | CRITICAL | Force safe state immediately; do not trust voted output |
| 1 | TMR voter single-lane disagree (rate-based) | `tmr_disagree_count` exceeds threshold | WARNING | Log + alert; trend over time indicates ageing or radiation environment |
| 2 | ECC SECDED uncorrectable (double-bit) | `ecc_uncorrected_count` increment | CRITICAL | Force safe state; SRAM data unsafe to consume |
| 3 | ECC SECDED single-bit corrected (rate-based) | `ecc_corrected_count` increment rate | INFO → WARNING if rising | Log; rising rate signals impending double-bit failure |
| 4 | DMS sensor watchdog timeout | `dms_fusion.sensor_fail` asserted | WARNING | Driver alert; degrade DMS-dependent functions |
| 5 | Plausibility checker rejection (rate-based) | `plausibility_checker.check_ok=0` rate above threshold | WARNING | Log; sustained rejections indicate sensor calibration drift |
| 6 | Safe-state controller reached MRC | `safe_state == 2'd3` | CRITICAL | Execute MRC: pull over and stop |
| 7 | DMS attention CRITICAL | `dms_alert` asserted | depends on use case | Driver alert + ADAS handover preparation |
| 8 | NPU compute fault (aggregated) | npu_top fault aggregator (post-F4-A integration; placeholder today) | CRITICAL | Force safe state; do not trust inference output |
| 9 | DMA error | npu_dma fault flag (placeholder; F4 follow-up) | WARNING | Log + degrade |
| 10 | Sensor fusion timing miss | fusion-engine timing watchdog (placeholder; F4 follow-up) | WARNING | Log + degrade |
| 11 | Clock monitor alert | RTL clock monitor (gap noted in SEooC:§5; F4 follow-up) | CRITICAL | Force safe state; clock unreliable |
| 12 | Thermal threshold exceeded | `rtl/thermal_zone` rule-based; F1-A9 ML upgrade | WARNING → CRITICAL at higher threshold | Throttle compute; if persistent, force safe state |
| 13 | Cryptographic boot signature mismatch | OpenTitan integration (Track 3 W4-W7) | CRITICAL | Refuse to enter mission mode; require operator intervention |
| 14 | Reserved (future ML fault predictor) | F1-A9 | INFO | Log for trend analysis |
| 15 | Generic SEU detected (catch-all) | Aggregate of single-bit detectors not fitting above slots | INFO → WARNING if rising | Log |

**[REQUIRED]** licensee supervisor MUST handle bits 0, 2, 6, 8, 11, 13 as CRITICAL — escalate to safe state immediately. Other bits per licensee item-level safety concept.

**[RECOMMENDED]** licensee logs the 16-bit value at every state transition for field-monitoring per AoU-14.

### 7.3 Counter signals

| Signal | Width | Type | Behaviour |
|---|:---:|---|---|
| `tmr_disagree_count` | 8 bits | Saturating | Increments on every detected single-lane TMR disagreement; saturates at 0xFF; clears only on cold reset |
| `ecc_corrected_count` | 16 bits | Saturating | Increments on every ECC single-bit correction; saturates at 0xFFFF |
| `ecc_uncorrected_count` | 8 bits | Saturating | Increments on every ECC double-bit detection; saturates at 0xFF; **[REQUIRED]** licensee escalates per item-level ASIL on every increment (not just on saturation) |

**[RECOMMENDED]** licensee samples counters at fixed intervals (e.g., once per second) and computes derivative; rising counter rate is a stronger signal than the absolute value.

### 7.4 Per-mechanism Fault Detection Time Interval (FDTI)

| Mechanism | FDTI | Notes |
|---|---|---|
| TMR voter detection | 1 clk | Single-cycle detection; voter samples on the same edge |
| SECDED single-bit detection + correction | 1 clk after read | ECC is combinational; output asserts on next clk |
| SECDED double-bit detection | 1 clk after read | Same path as single-bit |
| Per-sensor watchdog | WATCHDOG_CYCLES (configurable; default 200 ms @ 50 MHz = 10 M cycles) | Configurable per FSC:§1.2 FTTI budget |
| Plausibility checker | 1 clk after `check_valid` | Combinational rule eval + 1-cycle output register |
| Safe-state escalation NORMAL→ALERT | 1 ms (1 tick_1ms) | Immediate on first critical fault |
| Safe-state escalation ALERT→DEGRADE | ALERT_TIME_MS (default 2000 ms) | Only if sustained |
| Safe-state escalation DEGRADE→MRC | DEGRADE_TIME_MS (default 3000 ms) | Only if sustained |

**[REQUIRED]** licensee FSC FTTI budget per safety goal must accommodate the AstraCore FDTI plus the licensee's own signaling and actuator latency. See FSC:§1.2 for placeholder FTTIs and the formula in §6.3 above.

---

## 8. Boot integrity (secure boot)

### 8.1 Status

**Today (pre-Track-3):** AstraCore relies on host-side software cryptography for boot integrity. Boot signature verification is performed by the SDK at the licensee's compile/load step (`astracore.apply` configuration loader). RTL-level boot integrity is **not yet** present.

**Post-Track-3 (W4-W7 per remediation plan):** OpenTitan crypto IP is integrated into `rtl/`. Boot flow becomes:

1. Boot ROM (in `rtl/`) holds the public key hash, eFuse-locked at licensee tape-out
2. Bootloader fetches weights + configuration from external NVM (interface licensee-supplied)
3. **[REQUIRED]** RSA-2048 + SHA-256 signature verification using OpenTitan primitives (per ASR-HW-12 in SEooC:§4.1)
4. On verification success: load weights into `npu_sram_bank_ecc` instances; release safe_state_active
5. On verification failure: assert `fault_detected[13]`; do **not** enter mission mode; halt

### 8.2 Public key provisioning (AoU-9 expansion)

**[REQUIRED]** licensee provisions the model-signing public key at silicon manufacture time via eFuse / OTP. Recommended pattern: dual-public-key (production + revocation) for field-update support.

**[RECOMMENDED]** key management process per ISO 21434 (cybersecurity-engineering) — separate process from this safety manual.

### 8.3 Pre-Track-3 software-only boot

For licensee evaluations before OpenTitan integration ships:

1. Models are signed by AstraCore SDK at compile time; verification is a Python step on the host before loading into AstraCore.
2. Boot integrity is **NOT** ASIL-relevant in this configuration; treat as QM until Track 3 lands.
3. **[REQUIRED]** licensees evaluating in this configuration MUST mark the safety case explicitly as "pending RTL secure boot" and MUST NOT publish ASIL-B+ claims dependent on boot integrity.

---

## 9. Diagnostic services

### 9.1 Built-in diagnostics

TBD — full enumeration at v1.0. Currently:
- TMR fault counters (always-on; readable via `tmr_disagree_count` etc.)
- ECC counters (always-on)
- DMS attention level + sensor-fail flag (always-on per `dms_fusion`)
- Plausibility checker statistics (always-on per `plausibility_checker`)

### 9.2 On-demand diagnostics

TBD — LBIST trigger interface (F4-B-3 follow-up); MBIST coverage (F4-B-4 follow-up).

### 9.3 Field diagnostic data export

TBD — interface to read out fault counters + history for field monitoring per ISO 26262-7 §6. Today the counters are accessible via the SoC's standard register-read mechanism; the licensee provides the field-extraction telemetry path.

---

## 10. Assumptions of use (AoU)

This is a denormalized index of the AoUs declared in `docs/safety/seooc_declaration_v0_1.md` §6. The licensee must respect every entry in this table.

### 10.1 Operational environment (SEooC:§6.1)

| AoU | Statement | Verification by licensee |
|---|---|---|
| AoU-1 | Junction temperature shall be kept within –40 °C to +125 °C by licensee package + cooling. | Silicon thermal characterisation report |
| AoU-2 | Supply voltage shall be regulated within ±5 % of nominal by licensee PMIC. | PMIC qualification per AEC-Q100 / equivalent |
| AoU-3 | Clock source shall meet jitter spec documented in §4.1 above. | Clock-tree characterisation; jitter bench measurement |
| AoU-4 | Reset signal shall meet asynchronous-assert / synchronous-deassert protocol per §4.2 above. | Reset-controller verification |

### 10.2 Functional integration (SEooC:§6.2)

| AoU | Statement | Verification by licensee |
|---|---|---|
| AoU-5 | Licensee shall connect `safe_state_active` to a downstream safe-state actuator per their item-level safety concept. | Integration test + safety case review |
| AoU-6 | Licensee shall service `watchdog_kick` within the period documented in §6.1 above. | Watchdog timing test + supervisor MCU code review |
| AoU-7 | Licensee shall not drive `dft_isolation_enable` high during mission mode. | Synthesis / netlist check + integration test |
| AoU-8 | Licensee shall route `ecc_uncorrected_count` and `tmr_disagree_count` to a supervisor that escalates per their item-level ASIL. | Integration test + safety case review |

### 10.3 Software / data integration (SEooC:§6.3)

| AoU | Statement | Verification by licensee |
|---|---|---|
| AoU-9 | Models loaded into AstraCore for inference shall be cryptographically signed by a key under the licensee's control; the AstraCore secure-boot flow verifies the signature. | Key-provisioning audit + boot-flow test (Track 3 dependent) |
| AoU-10 | Models shall be quantised using the documented AstraCore quantiser flow (or an equivalent flow that produces bit-exact output verifiable against the AstraCore reference); models quantised by other flows are out of scope of the safety case. | Bit-exact reference comparison gate in licensee CI |
| AoU-11 | Sensor inputs shall be calibrated per the manufacturer's procedure; AstraCore plausibility checker assumes calibrated inputs. | Sensor calibration audit |
| AoU-12 | Licensee shall not modify AstraCore RTL or Python toolchain without re-running the safety verification suite and updating the safety case. | Change-impact analysis per ISO 26262-8 §8 |

### 10.4 Process (SEooC:§6.4)

| AoU | Statement | Verification by licensee |
|---|---|---|
| AoU-13 | Licensee shall sign a Development Interface Agreement (DIA) per ISO 26262-8 §5 with AstraCore before integration. | Signed DIA on file |
| AoU-14 | Licensee shall report any anomalies discovered during integration to AstraCore within 30 days for inclusion in the AstraCore field-monitoring program. | Reporting log + AstraCore acknowledgement |
| AoU-15 | Licensee shall produce an item-level HARA and derive item-level safety goals; AstraCore's assumed safety requirements (SEooC:§4) are subordinate to and must align with licensee's derived requirements. | Item-level HARA document on file |

### 10.5 Configuration (SEooC:§6.5)

| AoU | Statement | Verification by licensee |
|---|---|---|
| AoU-16 | AstraCore parameters shall be set within ranges documented in §3 above. `WEIGHT_DEPTH = N_ROWS × N_COLS`. Watchdog timeouts ≤ FTTI – signaling latency. | Parameter-freeze review + integration test |
| AoU-17 | Sparsity engine (when enabled) shall be exercised with QAT-trained models only; PTQ models on sparsity engine are out of scope of the safety case until characterised. | Model-flow audit (compile-time gate) |

---

## 11. Verification activities required by licensee

This section specifies the **minimum** integration test set every licensee must execute on their integrated SoC to claim AstraCore IP coverage. Beyond the minimum, licensees may add tests per their item-level FSC FSRs (`docs/safety/functional_safety_concept_v0_1.md` §3, "Verification" column).

**See also:** `docs/safety/integration_test_plan_v0_1.md` — per-FSR integration test specifications with concrete setup, stimulus, expected response, and pass/fail criteria for each of the 25 FSRs derived in FSC §3. The §11 sections below remain the *minimum* — the full per-FSR breakdown lives in the ITP.

### 11.1 Boundary signal integration tests

For each boundary signal in SEooC:§2.3, the licensee shall provide a test that:

1. Drives the upstream condition that should cause the signal to assert
2. Verifies the signal asserts within the FDTI documented in §7.4
3. Verifies the signal deasserts when the condition clears (where applicable)

| Signal | Test stimulus | Expected response |
|---|---|---|
| `safe_state_active` | Inject TMR triple-disagree via fault-injection harness | Assert within 1 clk |
| `fault_detected[0]` | Same | Assert with safe_state_active |
| `fault_detected[2]` | Inject ECC double-bit error | Assert within 1 clk after read |
| `fault_detected[4]` | Hold sensor input stuck for WATCHDOG_CYCLES | Assert; safe_state_active follows after escalation |
| `fault_detected[6]` | Force critical_faults sustained for ALERT_TIME_MS + DEGRADE_TIME_MS | Assert |
| `fault_detected[11]` | Force clock glitch | Assert (post-clock-monitor RTL gap closure) |
| `fault_detected[13]` | Boot with bad signature | Assert; safe_state_active stays asserted until reset (post-Track-3) |
| `tmr_disagree_count` | Inject single-lane TMR perturbation | Counter increments by 1 |
| `ecc_corrected_count` | Inject SRAM single-bit error | Counter increments by 1 |
| `ecc_uncorrected_count` | Inject SRAM double-bit error | Counter increments by 1; **CRITICAL — licensee escalates immediately** |

### 11.2 Watchdog timing test

- Configure licensee supervisor MCU to issue `watchdog_kick` at the AstraCore-configured period
- Run for at least 1 hour of mission mode
- Verify zero false SENSOR_FAIL assertions
- Force a deliberate late kick (outside ±10 % window); verify SENSOR_FAIL asserts within WATCHDOG_CYCLES

### 11.3 Safe-state entry test

- From NORMAL state, inject sustained critical fault
- Verify NORMAL → ALERT within 1 ms tick
- Verify ALERT → DEGRADE within ALERT_TIME_MS (± 1 tick)
- Verify DEGRADE → MRC within DEGRADE_TIME_MS (± 1 tick)
- Verify `mrc_pull_over` asserts in MRC
- Verify operator_reset returns to NORMAL only from MRC

### 11.4 Reset and initialization test

- Power-on with VDD ramp per silicon-vendor spec
- Verify rst_n is held throughout VDD ramp
- Verify safe_state_active stays high until first watchdog_kick + boot complete
- Verify mission-mode handshake per §5.1

### 11.5 ECC integration test (post-F4-A-1.1)

- Boot the SoC; verify ecc_corrected_count = 0 and ecc_uncorrected_count = 0
- Inject a single-bit error on every SRAM bank in turn (using the fault-injection harness or licensee equivalent)
- Verify ecc_corrected_count increments to N (where N = number of banks)
- Inject a double-bit error
- Verify ecc_uncorrected_count increments + fault_detected[2] asserts + safe_state_active asserts

### 11.6 TMR validation test

- Use the fault-injection harness `sim/fault_injection/campaigns/tmr_voter_seu_1k.yaml` (or licensee equivalent) on the integrated SoC
- Verify ≥ 99 % detection coverage of single-lane perturbations matches the declared mechanism DC

### 11.7 Reporting

Every test result above (pass/fail + measured numbers) is part of the licensee's item-level safety case. **[REQUIRED]** results are stored alongside the SafetyCase document at the licensee under the documentation control regime per ISO 26262-8 §10.

---

## 12. Reporting field anomalies

TBD — AstraCore field-monitoring contact, severity classification, response SLA. Required by AoU-14. Placeholder until the AstraCore production-support process is defined (post first NDA evaluation engagement).

---

## 13. Document control

| Field | Value |
|---|---|
| Document ID | ASTR-SAFETY-MANUAL-V0.5 |
| Revision | 0.5 |
| Revision date | 2026-04-20 |
| Author | TBD (Track 2 lead) — currently founder + collaborator |
| Reviewer | TBD |
| Approver | TBD |
| Distribution | Internal + TÜV SÜD India + first NDA evaluation licensee |
| Retention | 15 years post-product-discontinuation per ISO 26262-8 §10 |
| Supersedes | ASTR-SAFETY-MANUAL-V0.1 (skeleton) |

### 13.1 Completion plan

| Wk | Section completed |
|---|---|
| 2 | §1, §2.3, §3.1 (already partial), §10 AoU index ✅ |
| 4 | §2.1 safety-annotated block diagram (pending), §3.4 safe-state ✅, §4 clock/reset ✅ |
| 4 | §5 init sequence ✅, §6 watchdog ✅, §7 fault signalling ✅ |
| 4 | §11 licensee verification ✅ (this revision) |
| 7 | §8 secure boot reference flow expansion (with Track 3 OpenTitan integration) — currently partial |
| 9 | §9 diagnostic services |
| 11 | §12 field anomalies |
| 12 | v1.0 release for TÜV interim review |

### 13.2 Revision triggers

This Safety Manual is re-issued (with revision bump) on any of:

1. SEooC §6 AoU added, removed, or materially changed → §10 must be re-synced
2. New SEooC §2.3 boundary signal added or removed → §7 re-synced
3. New `fault_detected[]` bit allocated → §7.2 table updated
4. Track 3 OpenTitan integration milestone (changes §8 from partial to complete)
5. F4-A-1.1 (npu_top ECC integration) lands → §3.3 + §7.4 + §11.5 update with measured numbers
6. F4-A-7 (TMR on safe_state) lands → §3.4 update with new fault flag wiring
7. New WATCHDOG_CYCLES default driven by licensee characterisation → §6.1 update
