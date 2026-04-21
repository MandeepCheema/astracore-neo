# FMEDA — tmr_voter

**Document ID:** ASTR-FMEDA-TMR_VOTER-V0.1  
**Generated:** 2026-04-20  
**Standard:** ISO 26262-5 §7.4.5, §8.4 + ISO 26262-11 §4.6  
**Module:** `rtl/tmr_voter/`  
**ASIL target:** ASIL-B  
**Status:** v0.1 — failure-rate baselines are placeholders sourced from IEC 62380 / SN29500 scaled to 7 nm digital-logic assumptions. Diagnostic-coverage values are *targets* until validated by the fault-injection campaign documented in `docs/safety/fault_injection/`.

## 1. Per-failure-mode table

| ID | Sub-part | Failure mode | λ_FM (FIT) | Class | Mechanism | DC (%) | λ_S | λ_DD | λ_DU | λ_LF |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|
| `tmr_voter.voted_reg.seu` | voted output register (32-bit) | SEU on voted output bit | 0.0133 | dangerous | Aggregate fault flag → safe_state_controller | 40.0 | 0 | 0.0053 | 0.0080 | 0.0056 |
| `tmr_voter.fault_flags.seu` | fault_a / fault_b / fault_c / triple_fault output regs | SEU spuriously sets or clears a fault flag | 0.0017 | dangerous | Aggregate fault flag → safe_state_controller | 40.0 | 0 | 6.8000e-04 | 0.0010 | 7.1400e-04 |
| `tmr_voter.agreement.seu` | agreement output register (1-bit) | SEU sets/clears agreement spuriously | 4.2500e-04 | dangerous | Aggregate fault flag → safe_state_controller | 40.0 | 0 | 1.7000e-04 | 2.5500e-04 | 1.7850e-04 |
| `tmr_voter.vote_logic.stuck` | ab_eq / ac_eq / bc_eq combinational comparators (32-bit each) | stuck-at on a comparator output line → wrong majority detection | 0.0043 | dangerous | — | 0 | 0 | 0 | 0.0043 | 0.0043 |

All λ values in FIT (failures per 10⁹ hours).

## 2. Aggregates

| Quantity | Value (FIT) |
|---|---:|
| λ_total (all failure modes) | 0.0197 |
| λ_S (safe) | 0 |
| λ_dangerous (DD + DU) | 0.0197 |
| λ_DD (dangerous-detected) | 0.0062 |
| λ_DU (dangerous-undetected, SPF) | 0.0135 |
| λ_LF (latent-fault residual) | 0.0107 |
| λ_DPF (estimated dual-point) | 4.6954e-08 |

## 3. Metrics vs ASIL target

| Metric | Computed | Target (ASIL-B) | Pass? |
|---|---:|---:|:---:|
| SPFM | 31.35 % | ≥ 90 % | ❌ |
| LFM  | 0.00 % | ≥ 60 % | ❌ |
| PMHF | 0.0135 FIT | ≤ 100 FIT | ✅ |

**Overall:** ❌ fails ASIL-B target.

## 4. Findings and next actions

Module-level FMEDA does **not** meet the ASIL-B target (SPFM 31.35 % below target 90 %; LFM 0.00 % below target 60 %).

Top dangerous-undetected contributors (drive the SPFM gap):

- `tmr_voter.voted_reg.seu` — λ_DU = 0.0080 FIT (Aggregate fault flag → safe_state_controller, DC = 40.0 %)
- `tmr_voter.vote_logic.stuck` — λ_DU = 0.0043 FIT (no mechanism, DC = 0.0 %)
- `tmr_voter.fault_flags.seu` — λ_DU = 0.0010 FIT (Aggregate fault flag → safe_state_controller, DC = 40.0 %)

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
    --module tmr_voter \
    --asil ASIL-B \
    --output docs/safety/fmeda/tmr_voter_fmeda.md
```
