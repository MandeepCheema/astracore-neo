# FMEDA — safe_state_controller

**Document ID:** ASTR-FMEDA-SAFE_STATE_CONTROLLER-V0.1  
**Generated:** 2026-04-20  
**Standard:** ISO 26262-5 §7.4.5, §8.4 + ISO 26262-11 §4.6  
**Module:** `rtl/safe_state_controller/`  
**ASIL target:** ASIL-B  
**Status:** v0.1 — failure-rate baselines are placeholders sourced from IEC 62380 / SN29500 scaled to 7 nm digital-logic assumptions. Diagnostic-coverage values are *targets* until validated by the fault-injection campaign documented in `docs/safety/fault_injection/`.

## 1. Per-failure-mode table

| ID | Sub-part | Failure mode | λ_FM (FIT) | Class | Mechanism | DC (%) | λ_S | λ_DD | λ_DU | λ_LF |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|
| `safe_state_controller.state.seu` | safe_state[1:0] output register (2-bit FSM) | SEU jumps state — could go to safer (NORMAL→ALERT) or less safe (DEGRADE→NORMAL) | 9.0000e-04 | dangerous | — | 0 | 0 | 0 | 9.0000e-04 | 9.0000e-04 |
| `safe_state_controller.timer.seu` | escalation/recovery timer counter (~16 bits) | SEU advances or rewinds escalation timer | 0.0068 | dangerous | Counter periodic reset | 65.0 | 0 | 0.0044 | 0.0024 | 0.0011 |
| `safe_state_controller.max_speed.seu` | max_speed_kmh[7:0] output register | SEU spuriously raises or lowers commanded max speed | 0.0034 | dangerous | — | 0 | 0 | 0 | 0.0034 | 0.0034 |
| `safe_state_controller.transition_logic.stuck` | combinational state-transition logic | stuck-at on transition condition → can't escalate, or can't recover | 0.0016 | dangerous | — | 0 | 0 | 0 | 0.0016 | 0.0016 |

All λ values in FIT (failures per 10⁹ hours).

## 2. Aggregates

| Quantity | Value (FIT) |
|---|---:|
| λ_total (all failure modes) | 0.0127 |
| λ_S (safe) | 0 |
| λ_dangerous (DD + DU) | 0.0127 |
| λ_DD (dangerous-detected) | 0.0044 |
| λ_DU (dangerous-undetected, SPF) | 0.0083 |
| λ_LF (latent-fault residual) | 0.0070 |
| λ_DPF (estimated dual-point) | 3.0533e-08 |

## 3. Metrics vs ASIL target

| Metric | Computed | Target (ASIL-B) | Pass? |
|---|---:|---:|:---:|
| SPFM | 34.80 % | ≥ 90 % | ❌ |
| LFM  | 0.00 % | ≥ 60 % | ❌ |
| PMHF | 0.0083 FIT | ≤ 100 FIT | ✅ |

**Overall:** ❌ fails ASIL-B target.

## 4. Findings and next actions

Module-level FMEDA does **not** meet the ASIL-B target (SPFM 34.80 % below target 90 %; LFM 0.00 % below target 60 %).

Top dangerous-undetected contributors (drive the SPFM gap):

- `safe_state_controller.max_speed.seu` — λ_DU = 0.0034 FIT (no mechanism, DC = 0.0 %)
- `safe_state_controller.timer.seu` — λ_DU = 0.0024 FIT (Counter periodic reset, DC = 65.0 %)
- `safe_state_controller.transition_logic.stuck` — λ_DU = 0.0016 FIT (no mechanism, DC = 0.0 %)

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
    --module safe_state_controller \
    --asil ASIL-B \
    --output docs/safety/fmeda/safe_state_controller_fmeda.md
```
