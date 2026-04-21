# Fault-Injection Campaign — safe_state_controller_inj_1k

**Document ID:** ASTR-FI-SAFE_STATE_CONTROLLER_INJ_1K-V0.1  
**Generated:** 2026-04-20  
**Standard:** ISO 26262-5 §11 + ISO 26262-11 §7  
**Target module:** `rtl/safe_state_controller/`  
**Oracle signal:** `tb_safe_state_controller_fi.safe_state_seu_detected`  
**Expected safe response:** Pre-F4-A-7: most SEUs on safe_state register are NOT detected (no internal mechanism). The campaign records the baseline. Post-F4-A-7: every SEU on the 2-bit safe_state register must be detected within 1 cycle by the new TMR / Hamming check, which asserts tb_safe_state_controller_fi.safe_state_seu_detected.


## 1. Aggregate

| Metric | Value |
|---|---:|
| Planned injections | 9 |
| Runs completed | 9 |
| Perturbed an output | 0 |
| Detected by oracle | 0 |
| Missed by oracle | 0 |
| Benign (no output change) | 9 |
| False positives | 0 |
| **Diagnostic coverage** | **100.00 %** |

## 2. Per-injection results (first 50)

| # | Target | Detected? | Cycle | Perturbed? |
|---|---|:---:|---:|:---:|
| 0 | `tb_safe_state_controller_fi.u_dut.safe_state` | ❌ | — | no |
| 1 | `tb_safe_state_controller_fi.u_dut.safe_state` | ❌ | — | no |
| 2 | `tb_safe_state_controller_fi.u_dut.safe_state` | ❌ | — | no |
| 3 | `tb_safe_state_controller_fi.u_dut.safe_state` | ❌ | — | no |
| 4 | `tb_safe_state_controller_fi.u_dut.max_speed_kmh` | ❌ | — | no |
| 5 | `tb_safe_state_controller_fi.u_dut.max_speed_kmh` | ❌ | — | no |
| 6 | `tb_safe_state_controller_fi.u_dut.latched_faults` | ❌ | — | no |
| 7 | `tb_safe_state_controller_fi.u_dut.latched_faults` | ❌ | — | no |
| 8 | `tb_safe_state_controller_fi.critical_faults_reg` | ❌ | — | no |

## 3. FMEDA traceability

This campaign validates the diagnostic-coverage assumption for the following failure-mode rows in `tools/safety/failure_modes.yaml`:

- `safe_state_controller.state.seu`
- `safe_state_controller.timer.seu`
- `safe_state_controller.max_speed.seu`
- `safe_state_controller.transition_logic.stuck`

After this report is filed, update the corresponding mechanism `target_dc_pct` in `tools/safety/safety_mechanisms.yaml` to the measured **100.00 %** if it differs from the declared target by more than 5 percentage points (per SEooC §9.1 revision trigger #2).

## 4. Reproduce

```bash
# (WSL Ubuntu 22.04, Verilator 5.030, cocotb 2.0.1)
cd sim/fault_injection
make CAMPAIGN=safe_state_controller_inj_1k
python -m tools.safety.fault_injection \
    --campaign campaigns/safe_state_controller_inj_1k.yaml \
    --results out/safe_state_controller_inj_1k.jsonl \
    --output ../../docs/safety/fault_injection/safe_state_controller_inj_1k.md
```

