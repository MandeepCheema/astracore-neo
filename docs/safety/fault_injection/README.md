# Fault Injection Campaign Reports

**Purpose.** Cocotb-based fault-injection harness output, per ISO 26262-5 §11 (Hardware integration & testing) and ISO 26262-11 §7 (Soft errors).

**Status:** 4 campaigns + testbench wrappers shipped 2026-04-20; results land after WSL run.

## Index

| Target module | Campaign | Injections | Report | Wk | Status |
|---|---|---|---|---|---|
| `tmr_voter` | Stuck-at + bit-flip on voted nets | 1,000 | `tmr_voter_seu_1k.md` | W4 | 🟡 YAML + tb_tmr_voter_fi.sv shipped; 17 sample injections |
| `ecc_secded` | Single + double bit-flip on data + parity inputs | 10,000 | `ecc_secded_bf_10k.md` | W4 | 🟡 YAML + tb_ecc_secded_fi.sv shipped; 21 sample injections covering data-bit single, parity-bit single (incl. F4-D-6 alias positions), stuck-at, triple-bit |
| `dms_fusion` | SEU on internal state + F4-A-5 fix validation | 5,000 | `dms_fusion_inj_5k.md` | W9 | 🟡 YAML + tb_dms_fusion_fi.sv shipped; 21 injections covering all 10 catalogued FMEDA rows incl. tmr_valid_r + tmr_valid_r_shadow |
| `safe_state_controller` | SEU on safe_state FSM + timer (pre-F4-A-7 baseline) | 1,000 | `safe_state_controller_inj_1k.md` | W9 | 🟡 YAML + tb_safe_state_controller_fi.sv shipped; oracle is placeholder pre-F4-A-7 — same campaign re-runs after F4-A-7 lands and oracle wires to new TMR/Hamming flag, expect 0 % → ~100 % swing |
| `plausibility_checker` | Out-of-range / stuck inputs | 2,000 | `plausibility_inj_2k.md` | W9 | pending |
| Cross-module aggregate | Random injection across all safety-critical modules | 50,000+ | `aggregate_campaign_v0_1.md` | W10 | pending |

## Harness

Lives at `sim/fault_injection/` — scaffold delivered 2026-04-20, expanded with 4 campaigns + testbench wrappers same day. Built on cocotb 2.0.1; uses Verilator 5.030 backend per existing flow (`tools/run_verilator_*.sh`).

| File | Purpose |
|---|---|
| `sim/fault_injection/README.md` | Layout + end-to-end flow |
| `sim/fault_injection/Makefile` | Per-campaign run target — supports tmr_voter / ecc_secded / dms_fusion / safe_state_controller |
| `sim/fault_injection/runner.py` | cocotb test entrypoint |
| `sim/fault_injection/tb_tmr_voter_fi.sv` + `tb_ecc_secded_fi.sv` + `tb_dms_fusion_fi.sv` + `tb_safe_state_controller_fi.sv` | Testbench wrappers (module names follow `tb_<dut>_fi` convention) |
| `sim/fault_injection/campaigns/*.yaml` | 4 shipped campaigns; full counts (1k/5k/10k) are scripted sweep expansions in WSL |
| `tools/safety/fault_injection.py` | Planner + result aggregator (host-side, Windows-runnable) |
| `tests/test_fault_injection_planner.py` | 27 unit + integration tests (all pass), incl. `test_all_shipped_campaigns_parse` and `test_all_campaign_fmeda_ids_exist_in_failure_modes_yaml` cross-traceability gates |

**Two-half split** so authoring + interpretation works on Windows; only the simulator runs in WSL. See `tools/safety/fault_injection.py` module docstring for the rationale.

Injection types:
- **Stuck-at (0/1):** force a net to constant for N cycles.
- **Bit-flip (transient):** flip a register bit on one cycle.
- **Multiple-bit upset:** flip K bits in a codeword (for ECC double-bit detection coverage).
- **Clock domain transient:** mimic glitch on clock input.

## Coverage metric

For each campaign:
- **Detection rate** = (injections that triggered `fault_detected` or correct safe-state entry) / (total injections that perturbed an output).
- **False-positive rate** = injections that triggered fault-detect on a quiescent input.
- Both feed the FMEDA Diagnostic Coverage column.

After a campaign produces measured DC, `tools/safety/safety_mechanisms.yaml` is updated only when the measurement differs from the declared target by more than 5 percentage points (per SEooC §9.1 revision trigger #2). The affected FMEDAs are then re-rendered and the regression baseline (`docs/safety/fmeda/baseline.json`) regenerated.

## Reference

ISO 26262-5 §11; ISO 26262-11 §7. Verilator force/release caveat documented in `memory/feedback_verilator_force.md` — `force()` does not reach port-constant drivers; use internal nets only.
