# FMEDA — ecc_secded

**Document ID:** ASTR-FMEDA-ECC_SECDED-V0.1  
**Generated:** 2026-04-20  
**Standard:** ISO 26262-5 §7.4.5, §8.4 + ISO 26262-11 §4.6  
**Module:** `rtl/ecc_secded/`  
**ASIL target:** ASIL-B  
**Status:** v0.1 — failure-rate baselines are placeholders sourced from IEC 62380 / SN29500 scaled to 7 nm digital-logic assumptions. Diagnostic-coverage values are *targets* until validated by the fault-injection campaign documented in `docs/safety/fault_injection/`.

## 1. Per-failure-mode table

| ID | Sub-part | Failure mode | λ_FM (FIT) | Class | Mechanism | DC (%) | λ_S | λ_DD | λ_DU | λ_LF |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|
| `ecc_secded.data_out.seu` | data_out output register (64-bit) | SEU corrupts decoded output AFTER ECC has corrected it | 0.0266 | dangerous | — | 0 | 0 | 0 | 0.0266 | 0.0266 |
| `ecc_secded.flag_regs.seu` | single_err / double_err / corrected output regs | SEU spuriously sets or clears an error flag | 0.0013 | dangerous | Aggregate fault flag → safe_state_controller | 40.0 | 0 | 5.1000e-04 | 7.6500e-04 | 5.3550e-04 |
| `ecc_secded.syndrome_logic.stuck` | combinational syndrome computation (h_syndrome 7 bits + overall_recv_parity) | stuck-at on syndrome bit → wrong error position computed | 0.0030 | dangerous | — | 0 | 0 | 0 | 0.0030 | 0.0030 |
| `ecc_secded.parity_out.seu` | parity_out output register (8-bit) | SEU corrupts encoded parity written to SRAM | 0.0033 | dangerous | — | 0 | 0 | 0 | 0.0033 | 0.0033 |

All λ values in FIT (failures per 10⁹ hours).

## 2. Aggregates

| Quantity | Value (FIT) |
|---|---:|
| λ_total (all failure modes) | 0.0341 |
| λ_S (safe) | 0 |
| λ_dangerous (DD + DU) | 0.0341 |
| λ_DD (dangerous-detected) | 5.1000e-04 |
| λ_DU (dangerous-undetected, SPF) | 0.0336 |
| λ_LF (latent-fault residual) | 0.0334 |
| λ_DPF (estimated dual-point) | 1.4625e-07 |

## 3. Metrics vs ASIL target

| Metric | Computed | Target (ASIL-B) | Pass? |
|---|---:|---:|:---:|
| SPFM | 1.49 % | ≥ 90 % | ❌ |
| LFM  | 0.00 % | ≥ 60 % | ❌ |
| PMHF | 0.0336 FIT | ≤ 100 FIT | ✅ |

**Overall:** ❌ fails ASIL-B target.

## 4. Findings and next actions

Module-level FMEDA does **not** meet the ASIL-B target (SPFM 1.49 % below target 90 %; LFM 0.00 % below target 60 %).

Top dangerous-undetected contributors (drive the SPFM gap):

- `ecc_secded.data_out.seu` — λ_DU = 0.0266 FIT (no mechanism, DC = 0.0 %)
- `ecc_secded.parity_out.seu` — λ_DU = 0.0033 FIT (no mechanism, DC = 0.0 %)
- `ecc_secded.syndrome_logic.stuck` — λ_DU = 0.0030 FIT (no mechanism, DC = 0.0 %)

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
    --module ecc_secded \
    --asil ASIL-B \
    --output docs/safety/fmeda/ecc_secded_fmeda.md
```
