# FMEDA — aggregate

**Document ID:** ASTR-FMEDA-AGGREGATE-V0.1  
**Generated:** 2026-04-20  
**Standard:** ISO 26262-5 §7.4.5, §8.4 + ISO 26262-11 §4.6  
**Module:** `rtl/aggregate/`  
**ASIL target:** ASIL-B  
**Status:** v0.1 — failure-rate baselines are placeholders sourced from IEC 62380 / SN29500 scaled to 7 nm digital-logic assumptions. Diagnostic-coverage values are *targets* until validated by the fault-injection campaign documented in `docs/safety/fault_injection/`.

## 1. Per-failure-mode table

| ID | Sub-part | Failure mode | λ_FM (FIT) | Class | Mechanism | DC (%) | λ_S | λ_DD | λ_DU | λ_LF |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|
| `dms_fusion.wdog_cnt.seu` | wdog_cnt (24-bit watchdog counter) | SEU on counter bit → premature/late SENSOR_FAIL | 0.0102 | dangerous | Per-sensor watchdog | 95.0 | 0 | 0.0097 | 5.1000e-04 | 5.1000e-05 |
| `dms_fusion.sensor_fail.seu` | sensor_fail (1-bit flag) | SEU sets/clears sensor_fail spuriously | 4.5000e-04 | dangerous | TMR voter (3-of-3 majority) | 99.0 | 0 | 4.4550e-04 | 4.5000e-06 | 2.2500e-07 |
| `dms_fusion.cont_closed.seu` | cont_closed (7-bit saturating counter) | SEU on counter bit → spurious CRITICAL or missed CRITICAL | 0.0030 | dangerous | Counter periodic reset | 65.0 | 0 | 0.0019 | 0.0010 | 4.6856e-04 |
| `dms_fusion.cont_distracted.seu` | cont_distracted (7-bit saturating counter) | SEU on counter bit → spurious DISTRACTED or missed DISTRACTED | 0.0030 | dangerous | Counter periodic reset | 65.0 | 0 | 0.0019 | 0.0010 | 4.6856e-04 |
| `dms_fusion.blink_frame_cnt.seu` | blink_frame_cnt (6-bit window counter) | SEU shortens or lengthens the blink-rate measurement window | 0.0024 | dangerous | Blink window resampling | 70.0 | 0 | 0.0017 | 7.2000e-04 | 2.8800e-04 |
| `dms_fusion.blink_snapshot.seu` | blink_snapshot (16-bit snapshot of blink_count) | SEU corrupts snapshot used for delta calculation | 0.0068 | dangerous | Blink window resampling | 70.0 | 0 | 0.0048 | 0.0020 | 8.1600e-04 |
| `dms_fusion.blink_elevated.seu` | blink_elevated (1-bit flag) | SEU sets blink_elevated spuriously (or clears it) | 4.5000e-04 | dangerous | Blink window resampling | 70.0 | 0 | 3.1500e-04 | 1.3500e-04 | 5.4000e-05 |
| `dms_fusion.score_filt.seu` | score_filt_x4 (9-bit IIR accumulator) | SEU on filtered score bit | 0.0038 | dangerous | IIR temporal smoother (self-correcting) | 70.0 | 0 | 0.0027 | 0.0011 | 5.7375e-04 |
| `dms_fusion.raw_score.stuck` | raw_score combinational priority encoder | stuck-at on intermediate threshold comparators | 0.0014 | dangerous | Priority encoder safe-default | 60.0 | 0 | 8.4000e-04 | 5.6000e-04 | 3.3600e-04 |
| `dms_fusion.dal_lane.seu` | dal_a/b/c (3 × 3-bit TMR lanes for driver_attention_level) | SEU on one of three TMR lanes | 0.0043 | dangerous | TMR voter (3-of-3 majority) | 99.0 | 0 | 0.0042 | 4.2750e-05 | 2.1375e-06 |
| `dms_fusion.conf_lane.seu` | conf_a/b/c (3 × 8-bit TMR lanes for dms_confidence) | SEU on one of three TMR lanes | 0.0114 | dangerous | TMR voter (3-of-3 majority) | 99.0 | 0 | 0.0113 | 1.1400e-04 | 5.7000e-06 |
| `dms_fusion.dal_lane.stuck` | dal_a/b/c (TMR lanes) | stuck-at on single TMR lane | 8.5500e-04 | dangerous | TMR voter (3-of-3 majority) | 99.0 | 0 | 8.4645e-04 | 8.5500e-06 | 4.2750e-07 |
| `dms_fusion.conf_lane.stuck` | conf_a/b/c (TMR lanes) | stuck-at on single TMR lane | 0.0023 | dangerous | TMR voter (3-of-3 majority) | 99.0 | 0 | 0.0023 | 2.2800e-05 | 1.1400e-06 |
| `dms_fusion.tmr_valid.seu` | tmr_valid_r (1-bit valid flag to voter; shadow added F4-A-5) | SEU sets/clears voter valid input | 4.2500e-04 | dangerous | Shadow-register duplicate-and-compare on a 1-bit critical flag | 99.0 | 0 | 4.2075e-04 | 4.2500e-06 | 2.1250e-07 |
| `dms_fusion.gaze_input.stuck_lo` | gaze_valid input (boundary) | stuck-low → no valid pulses ever arrive | 5.0000e-04 | dangerous | Per-sensor watchdog | 95.0 | 0 | 4.7500e-04 | 2.5000e-05 | 2.5000e-06 |
| `dms_fusion.gaze_input.stuck_hi` | gaze_valid input (boundary) | stuck-high → continuous "valid" pulses | 5.0000e-04 | safe | — | 0 | 5.0000e-04 | 0 | 0 | 0 |
| `ecc_secded.data_out.seu` | data_out output register (64-bit) | SEU corrupts decoded output AFTER ECC has corrected it | 0.0266 | dangerous | — | 0 | 0 | 0 | 0.0266 | 0.0266 |
| `ecc_secded.flag_regs.seu` | single_err / double_err / corrected output regs | SEU spuriously sets or clears an error flag | 0.0013 | dangerous | Aggregate fault flag → safe_state_controller | 40.0 | 0 | 5.1000e-04 | 7.6500e-04 | 5.3550e-04 |
| `ecc_secded.syndrome_logic.stuck` | combinational syndrome computation (h_syndrome 7 bits + overall_recv_parity) | stuck-at on syndrome bit → wrong error position computed | 0.0030 | dangerous | — | 0 | 0 | 0 | 0.0030 | 0.0030 |
| `ecc_secded.parity_out.seu` | parity_out output register (8-bit) | SEU corrupts encoded parity written to SRAM | 0.0033 | dangerous | — | 0 | 0 | 0 | 0.0033 | 0.0033 |
| `npu_top.pe_weight_reg.seu` | npu_pe.weight_reg (16 PEs × 8-bit weight register) | SEU on held weight → wrong MACs until next load_w | 0.0531 | dangerous | — | 0 | 0 | 0 | 0.0531 | 0.0531 |
| `npu_top.pe_weight_reg.stuck` | npu_pe.weight_reg | stuck-at on weight bit → systematic MAC error | 0.0109 | dangerous | — | 0 | 0 | 0 | 0.0109 | 0.0109 |
| `npu_top.pe_acc.seu` | npu_pe.acc (16 PEs × 32-bit accumulator) | SEU on partial sum → wrong tile output | 0.2125 | dangerous | — | 0 | 0 | 0 | 0.2125 | 0.2125 |
| `npu_top.pe_acc.stuck` | npu_pe.acc | stuck-at on accumulator bit | 0.0435 | dangerous | — | 0 | 0 | 0 | 0.0435 | 0.0435 |
| `npu_top.pe_dataflow.seu` | npu_pe.{a_out, a_valid_out, sparse_skip_out} pass-through (16 PEs × 10 bits) | SEU on inter-PE activation flow | 0.0680 | dangerous | — | 0 | 0 | 0 | 0.0680 | 0.0680 |
| `npu_top.pe_mul_tree.stuck` | combinational multiplier (INT8 + INT4 + INT2 trees, all 16 PEs) | stuck-at on multiplier partial product | 0.0216 | dangerous | — | 0 | 0 | 0 | 0.0216 | 0.0216 |
| `npu_top.systolic_drain.seu` | systolic_array drain mux + c_valid strobe | SEU on output capture → spurious or missed c_valid | 0.0085 | dangerous | Aggregate fault flag → safe_state_controller | 40.0 | 0 | 0.0034 | 0.0051 | 0.0036 |
| `npu_top.tile_ctrl_fsm.seu` | npu_tile_ctrl state register (FSM, ~5 states) | SEU jumps FSM to wrong state | 0.0013 | dangerous | Aggregate fault flag → safe_state_controller | 40.0 | 0 | 5.1000e-04 | 7.6500e-04 | 5.3550e-04 |
| `npu_top.tile_ctrl_cfg.seu` | latched config (cfg_k, cfg_ai_base, cfg_ao_base, cfg_afu_mode_r) | SEU on captured config word → wrong tile parameters | 0.0225 | dangerous | — | 0 | 0 | 0 | 0.0225 | 0.0225 |
| `npu_top.dma_fsm.seu` | npu_dma FSM + address counters | SEU on DMA state or address → wrong transfer | 0.0170 | dangerous | Aggregate fault flag → safe_state_controller | 40.0 | 0 | 0.0068 | 0.0102 | 0.0071 |
| `npu_top.sram_data.seu` | SRAM banks WA/AI/AO/SCRATCH data array (4 banks × 256 entries × ~64 bits avg) | SEU bit-flip in SRAM cell | 0.0760 | dangerous | — | 0 | 0 | 0 | 0.0760 | 0.0760 |
| `npu_top.sram_addr.stuck` | SRAM address decoder | stuck-at on address bit → wrong row accessed | 0.0045 | dangerous | — | 0 | 0 | 0 | 0.0045 | 0.0045 |
| `npu_top.busy_done.seu` | top-level busy / done handshake registers | SEU sets/clears busy or done spuriously | 0.0014 | dangerous | Aggregate fault flag → safe_state_controller | 40.0 | 0 | 5.4000e-04 | 8.1000e-04 | 5.6700e-04 |
| `npu_top.precision_mode.stuck` | cfg precision_mode broadcast to all PEs | stuck-at → wrong precision applied silently | 6.8000e-04 | dangerous | — | 0 | 0 | 0 | 6.8000e-04 | 6.8000e-04 |
| `plausibility_checker.check_ok.seu` | check_ok output register (1-bit) | SEU spuriously passes a violating detection or rejects a clean one | 4.2500e-04 | dangerous | Aggregate fault flag → safe_state_controller | 40.0 | 0 | 1.7000e-04 | 2.5500e-04 | 1.7850e-04 |
| `plausibility_checker.asil_degrade.seu` | asil_degrade[7:0] output register | SEU corrupts ASIL routing instruction | 0.0034 | dangerous | Aggregate fault flag → safe_state_controller | 40.0 | 0 | 0.0014 | 0.0020 | 0.0014 |
| `plausibility_checker.violation.seu` | check_violation[2:0] output register | SEU corrupts violation code | 0.0013 | dangerous | Aggregate fault flag → safe_state_controller | 40.0 | 0 | 5.1000e-04 | 7.6500e-04 | 5.3550e-04 |
| `plausibility_checker.rule_logic.stuck` | combinational redundancy-rule evaluation | stuck-at on rule output → wrong cross-sensor decision | 0.0024 | dangerous | — | 0 | 0 | 0 | 0.0024 | 0.0024 |
| `plausibility_checker.counters.seu` | total_checks / total_violations counters (32 bits total) | SEU on counter | 0.0136 | no-effect | — | 0 | 0.0136 | 0 | 0 | 0 |
| `safe_state_controller.state.seu` | safe_state[1:0] output register (2-bit FSM) | SEU jumps state — could go to safer (NORMAL→ALERT) or less safe (DEGRADE→NORMAL) | 9.0000e-04 | dangerous | — | 0 | 0 | 0 | 9.0000e-04 | 9.0000e-04 |
| `safe_state_controller.timer.seu` | escalation/recovery timer counter (~16 bits) | SEU advances or rewinds escalation timer | 0.0068 | dangerous | Counter periodic reset | 65.0 | 0 | 0.0044 | 0.0024 | 0.0011 |
| `safe_state_controller.max_speed.seu` | max_speed_kmh[7:0] output register | SEU spuriously raises or lowers commanded max speed | 0.0034 | dangerous | — | 0 | 0 | 0 | 0.0034 | 0.0034 |
| `safe_state_controller.transition_logic.stuck` | combinational state-transition logic | stuck-at on transition condition → can't escalate, or can't recover | 0.0016 | dangerous | — | 0 | 0 | 0 | 0.0016 | 0.0016 |
| `tmr_voter.voted_reg.seu` | voted output register (32-bit) | SEU on voted output bit | 0.0133 | dangerous | Aggregate fault flag → safe_state_controller | 40.0 | 0 | 0.0053 | 0.0080 | 0.0056 |
| `tmr_voter.fault_flags.seu` | fault_a / fault_b / fault_c / triple_fault output regs | SEU spuriously sets or clears a fault flag | 0.0017 | dangerous | Aggregate fault flag → safe_state_controller | 40.0 | 0 | 6.8000e-04 | 0.0010 | 7.1400e-04 |
| `tmr_voter.agreement.seu` | agreement output register (1-bit) | SEU sets/clears agreement spuriously | 4.2500e-04 | dangerous | Aggregate fault flag → safe_state_controller | 40.0 | 0 | 1.7000e-04 | 2.5500e-04 | 1.7850e-04 |
| `tmr_voter.vote_logic.stuck` | ab_eq / ac_eq / bc_eq combinational comparators (32-bit each) | stuck-at on a comparator output line → wrong majority detection | 0.0043 | dangerous | — | 0 | 0 | 0 | 0.0043 | 0.0043 |

All λ values in FIT (failures per 10⁹ hours).

## 2. Aggregates

| Quantity | Value (FIT) |
|---|---:|
| λ_total (all failure modes) | 0.6807 |
| λ_S (safe) | 0.0141 |
| λ_dangerous (DD + DU) | 0.6666 |
| λ_DD (dangerous-detected) | 0.0682 |
| λ_DU (dangerous-undetected, SPF) | 0.5984 |
| λ_LF (latent-fault residual) | 0.5838 |
| λ_DPF (estimated dual-point) | 2.5570e-06 |

## 3. Metrics vs ASIL target

| Metric | Computed | Target (ASIL-B) | Pass? |
|---|---:|---:|:---:|
| SPFM | 10.23 % | ≥ 90 % | ❌ |
| LFM  | 0.00 % | ≥ 60 % | ❌ |
| PMHF | 0.5984 FIT | ≤ 100 FIT | ✅ |

**Overall:** ❌ fails ASIL-B target.

## 4. Findings and next actions

Module-level FMEDA does **not** meet the ASIL-B target (SPFM 10.23 % below target 90 %; LFM 0.00 % below target 60 %).

Top dangerous-undetected contributors (drive the SPFM gap):

- `npu_top.pe_acc.seu` — λ_DU = 0.2125 FIT (no mechanism, DC = 0.0 %)
- `npu_top.sram_data.seu` — λ_DU = 0.0760 FIT (no mechanism, DC = 0.0 %)
- `npu_top.pe_dataflow.seu` — λ_DU = 0.0680 FIT (no mechanism, DC = 0.0 %)

Closure options:
1. Add a module-level mechanism for any uncovered (`no mechanism`) row. The most common pattern is parity or duplication on a single critical FF.
2. Improve the declared DC of the named mechanism by running the fault-injection campaign and demonstrating higher actual coverage than the conservative target.
3. Demonstrate aggregate coverage via the `safe_state_controller` cross-module roll-up — a module-level SPF that escalates to safe-state within the FTTI may be reclassified as DD at aggregate scope.

## 5. Cross-references

- `docs/safety/seooc_declaration_v0_1.md` §5 (declared mechanism coverage)
- `docs/safety/iso26262_gap_analysis_v0_1.md` Part 5 (FMEDA gap closure)
- `docs/safety/fault_injection/` (campaigns that will validate DC numbers)
- `tools/safety/failure_modes.yaml` (input catalog)
- `tools/safety/safety_mechanisms.yaml` (mechanism catalog)

## 6. Reproduce

```bash
python -m tools.safety.fmeda \
    --module aggregate \
    --asil ASIL-B \
    --output docs/safety/fmeda/aggregate_fmeda.md
```
