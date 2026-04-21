# Fault-Injection Campaign — tmr_voter_seu_1k

**Document ID:** ASTR-FI-TMR_VOTER_SEU_1K-V0.1  
**Generated:** 2026-04-20  
**Standard:** ISO 26262-5 §11 + ISO 26262-11 §7  
**Target module:** `rtl/tmr_voter/`  
**Oracle signal:** `tb_tmr_voter_fi.fault_a`  
**Expected safe response:** fault_<lane> asserts within 1 cycle of the voted disagreement, and the voted output continues to match the majority (uncorrupted lanes).


## 1. Aggregate

| Metric | Value |
|---|---:|
| Planned injections | 17 |
| Runs completed | 17 |
| Perturbed an output | 10 |
| Detected by oracle | 10 |
| Missed by oracle | 0 |
| Benign (no output change) | 7 |
| False positives | 0 |
| **Diagnostic coverage** | **100.00 %** |

## 2. Per-injection results (first 50)

| # | Target | Detected? | Cycle | Perturbed? |
|---|---|:---:|---:|:---:|
| 0 | `tb_tmr_voter_fi.lane_a_reg` | ✅ | 101 | yes |
| 1 | `tb_tmr_voter_fi.lane_a_reg` | ✅ | 101 | yes |
| 2 | `tb_tmr_voter_fi.lane_a_reg` | ✅ | 101 | yes |
| 3 | `tb_tmr_voter_fi.lane_a_reg` | ✅ | 101 | yes |
| 4 | `tb_tmr_voter_fi.lane_a_reg` | ✅ | 101 | yes |
| 5 | `tb_tmr_voter_fi.lane_a_reg` | ✅ | 101 | yes |
| 6 | `tb_tmr_voter_fi.lane_a_reg` | ✅ | 101 | yes |
| 7 | `tb_tmr_voter_fi.lane_a_reg` | ✅ | 101 | yes |
| 8 | `tb_tmr_voter_fi.lane_b_reg` | ❌ | — | no |
| 9 | `tb_tmr_voter_fi.lane_b_reg` | ❌ | — | no |
| 10 | `tb_tmr_voter_fi.lane_b_reg` | ❌ | — | no |
| 11 | `tb_tmr_voter_fi.lane_b_reg` | ❌ | — | no |
| 12 | `tb_tmr_voter_fi.lane_c_reg` | ❌ | — | no |
| 13 | `tb_tmr_voter_fi.lane_c_reg` | ❌ | — | no |
| 14 | `tb_tmr_voter_fi.lane_a_reg` | ✅ | 201 | yes |
| 15 | `tb_tmr_voter_fi.lane_a_reg` | ❌ | — | no |
| 16 | `tb_tmr_voter_fi.lane_a_reg` | ✅ | 401 | yes |

## 3. FMEDA traceability

This campaign validates the diagnostic-coverage assumption for the following failure-mode rows in `tools/safety/failure_modes.yaml`:

- `dms_fusion.dal_lane.seu`
- `dms_fusion.conf_lane.seu`

After this report is filed, update the corresponding mechanism `target_dc_pct` in `tools/safety/safety_mechanisms.yaml` to the measured **100.00 %** if it differs from the declared target by more than 5 percentage points (per SEooC §9.1 revision trigger #2).

## 4. Reproduce

```bash
# (WSL Ubuntu 22.04, Verilator 5.030, cocotb 2.0.1)
cd sim/fault_injection
make CAMPAIGN=tmr_voter_seu_1k
python -m tools.safety.fault_injection \
    --campaign campaigns/tmr_voter_seu_1k.yaml \
    --results out/tmr_voter_seu_1k.jsonl \
    --output ../../docs/safety/fault_injection/tmr_voter_seu_1k.md
```

