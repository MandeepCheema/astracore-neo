# Fault-Injection Campaign — dms_fusion_inj_5k

**Document ID:** ASTR-FI-DMS_FUSION_INJ_5K-V0.1  
**Generated:** 2026-04-20  
**Standard:** ISO 26262-5 §11 + ISO 26262-11 §7  
**Target module:** `rtl/dms_fusion/`  
**Oracle signal:** `tb_dms_fusion_fi.tmr_fault`  
**Expected safe response:** tmr_fault asserts within 1-2 cycles of any SEU that perturbs an output (driver_attention_level, dms_confidence, dms_alert). The TMR voter catches per-lane SEUs on dal/conf; the F4-A-5 shadow comparator catches SEUs on tmr_valid_r; counter/IIR SEUs may not trip tmr_fault but should self-correct within the documented bounded window — those rows have expected_detection set per the mechanism's nature.


## 1. Aggregate

| Metric | Value |
|---|---:|
| Planned injections | 21 |
| Runs completed | 21 |
| Perturbed an output | 3 |
| Detected by oracle | 3 |
| Missed by oracle | 0 |
| Benign (no output change) | 18 |
| False positives | 0 |
| **Diagnostic coverage** | **100.00 %** |

## 2. Per-injection results (first 50)

| # | Target | Detected? | Cycle | Perturbed? |
|---|---|:---:|---:|:---:|
| 0 | `tb_dms_fusion_fi.u_dut.tmr_valid_r` | ✅ | 500 | yes |
| 1 | `tb_dms_fusion_fi.u_dut.tmr_valid_r_shadow` | ✅ | 600 | yes |
| 2 | `tb_dms_fusion_fi.u_dut.dal_a` | ❌ | — | no |
| 3 | `tb_dms_fusion_fi.u_dut.dal_a` | ❌ | — | no |
| 4 | `tb_dms_fusion_fi.u_dut.dal_a` | ❌ | — | no |
| 5 | `tb_dms_fusion_fi.u_dut.dal_b` | ❌ | — | no |
| 6 | `tb_dms_fusion_fi.u_dut.dal_c` | ❌ | — | no |
| 7 | `tb_dms_fusion_fi.u_dut.conf_a` | ✅ | 1001 | yes |
| 8 | `tb_dms_fusion_fi.u_dut.conf_b` | ❌ | — | no |
| 9 | `tb_dms_fusion_fi.u_dut.conf_c` | ❌ | — | no |
| 10 | `tb_dms_fusion_fi.u_dut.cont_closed` | ❌ | — | no |
| 11 | `tb_dms_fusion_fi.u_dut.cont_distracted` | ❌ | — | no |
| 12 | `tb_dms_fusion_fi.u_dut.score_filt_x4` | ❌ | — | no |
| 13 | `tb_dms_fusion_fi.u_dut.score_filt_x4` | ❌ | — | no |
| 14 | `tb_dms_fusion_fi.u_dut.blink_frame_cnt` | ❌ | — | no |
| 15 | `tb_dms_fusion_fi.u_dut.blink_snapshot` | ❌ | — | no |
| 16 | `tb_dms_fusion_fi.u_dut.blink_elevated` | ❌ | — | no |
| 17 | `tb_dms_fusion_fi.u_dut.sensor_fail` | ❌ | — | no |
| 18 | `tb_dms_fusion_fi.u_dut.wdog_cnt` | ❌ | — | no |
| 19 | `tb_dms_fusion_fi.u_dut.dal_a` | ❌ | — | no |
| 20 | `tb_dms_fusion_fi.u_dut.dal_a` | ❌ | — | no |

## 3. FMEDA traceability

This campaign validates the diagnostic-coverage assumption for the following failure-mode rows in `tools/safety/failure_modes.yaml`:

- `dms_fusion.dal_lane.seu`
- `dms_fusion.conf_lane.seu`
- `dms_fusion.tmr_valid.seu`
- `dms_fusion.cont_closed.seu`
- `dms_fusion.cont_distracted.seu`
- `dms_fusion.score_filt.seu`
- `dms_fusion.blink_snapshot.seu`
- `dms_fusion.blink_frame_cnt.seu`
- `dms_fusion.blink_elevated.seu`
- `dms_fusion.sensor_fail.seu`

After this report is filed, update the corresponding mechanism `target_dc_pct` in `tools/safety/safety_mechanisms.yaml` to the measured **100.00 %** if it differs from the declared target by more than 5 percentage points (per SEooC §9.1 revision trigger #2).

## 4. Reproduce

```bash
# (WSL Ubuntu 22.04, Verilator 5.030, cocotb 2.0.1)
cd sim/fault_injection
make CAMPAIGN=dms_fusion_inj_5k
python -m tools.safety.fault_injection \
    --campaign campaigns/dms_fusion_inj_5k.yaml \
    --results out/dms_fusion_inj_5k.jsonl \
    --output ../../docs/safety/fault_injection/dms_fusion_inj_5k.md
```

