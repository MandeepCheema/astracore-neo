# FMEDA — plausibility_checker

**Document ID:** ASTR-FMEDA-PLAUSIBILITY_CHECKER-V0.1  
**Generated:** 2026-04-20  
**Standard:** ISO 26262-5 §7.4.5, §8.4 + ISO 26262-11 §4.6  
**Module:** `rtl/plausibility_checker/`  
**ASIL target:** ASIL-B  
**Status:** v0.1 — failure-rate baselines are placeholders sourced from IEC 62380 / SN29500 scaled to 7 nm digital-logic assumptions. Diagnostic-coverage values are *targets* until validated by the fault-injection campaign documented in `docs/safety/fault_injection/`.

## 1. Per-failure-mode table

| ID | Sub-part | Failure mode | λ_FM (FIT) | Class | Mechanism | DC (%) | λ_S | λ_DD | λ_DU | λ_LF |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|
| `plausibility_checker.check_ok.seu` | check_ok output register (1-bit) | SEU spuriously passes a violating detection or rejects a clean one | 4.2500e-04 | dangerous | Aggregate fault flag → safe_state_controller | 40.0 | 0 | 1.7000e-04 | 2.5500e-04 | 1.7850e-04 |
| `plausibility_checker.asil_degrade.seu` | asil_degrade[7:0] output register | SEU corrupts ASIL routing instruction | 0.0034 | dangerous | Aggregate fault flag → safe_state_controller | 40.0 | 0 | 0.0014 | 0.0020 | 0.0014 |
| `plausibility_checker.violation.seu` | check_violation[2:0] output register | SEU corrupts violation code | 0.0013 | dangerous | Aggregate fault flag → safe_state_controller | 40.0 | 0 | 5.1000e-04 | 7.6500e-04 | 5.3550e-04 |
| `plausibility_checker.rule_logic.stuck` | combinational redundancy-rule evaluation | stuck-at on rule output → wrong cross-sensor decision | 0.0024 | dangerous | — | 0 | 0 | 0 | 0.0024 | 0.0024 |
| `plausibility_checker.counters.seu` | total_checks / total_violations counters (32 bits total) | SEU on counter | 0.0136 | no-effect | — | 0 | 0.0136 | 0 | 0 | 0 |

All λ values in FIT (failures per 10⁹ hours).

## 2. Aggregates

| Quantity | Value (FIT) |
|---|---:|
| λ_total (all failure modes) | 0.0211 |
| λ_S (safe) | 0.0136 |
| λ_dangerous (DD + DU) | 0.0075 |
| λ_DD (dangerous-detected) | 0.0020 |
| λ_DU (dangerous-undetected, SPF) | 0.0055 |
| λ_LF (latent-fault residual) | 0.0045 |
| λ_DPF (estimated dual-point) | 1.9894e-08 |

## 3. Metrics vs ASIL target

| Metric | Computed | Target (ASIL-B) | Pass? |
|---|---:|---:|:---:|
| SPFM | 27.20 % | ≥ 90 % | ❌ |
| LFM  | 0.00 % | ≥ 60 % | ❌ |
| PMHF | 0.0055 FIT | ≤ 100 FIT | ✅ |

**Overall:** ❌ fails ASIL-B target.

## 4. Findings and next actions

Module-level FMEDA does **not** meet the ASIL-B target (SPFM 27.20 % below target 90 %; LFM 0.00 % below target 60 %).

Top dangerous-undetected contributors (drive the SPFM gap):

- `plausibility_checker.rule_logic.stuck` — λ_DU = 0.0024 FIT (no mechanism, DC = 0.0 %)
- `plausibility_checker.asil_degrade.seu` — λ_DU = 0.0020 FIT (Aggregate fault flag → safe_state_controller, DC = 40.0 %)
- `plausibility_checker.violation.seu` — λ_DU = 7.6500e-04 FIT (Aggregate fault flag → safe_state_controller, DC = 40.0 %)

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
    --module plausibility_checker \
    --asil ASIL-B \
    --output docs/safety/fmeda/plausibility_checker_fmeda.md
```
