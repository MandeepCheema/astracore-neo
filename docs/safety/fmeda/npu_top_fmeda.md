# FMEDA — npu_top

**Document ID:** ASTR-FMEDA-NPU_TOP-V0.1  
**Generated:** 2026-04-20  
**Standard:** ISO 26262-5 §7.4.5, §8.4 + ISO 26262-11 §4.6  
**Module:** `rtl/npu_top/`  
**ASIL target:** ASIL-B  
**Status:** v0.1 — failure-rate baselines are placeholders sourced from IEC 62380 / SN29500 scaled to 7 nm digital-logic assumptions. Diagnostic-coverage values are *targets* until validated by the fault-injection campaign documented in `docs/safety/fault_injection/`.

## 1. Per-failure-mode table

| ID | Sub-part | Failure mode | λ_FM (FIT) | Class | Mechanism | DC (%) | λ_S | λ_DD | λ_DU | λ_LF |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|
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

All λ values in FIT (failures per 10⁹ hours).

## 2. Aggregates

| Quantity | Value (FIT) |
|---|---:|
| λ_total (all failure modes) | 0.5414 |
| λ_S (safe) | 0 |
| λ_dangerous (DD + DU) | 0.5414 |
| λ_DD (dangerous-detected) | 0.0113 |
| λ_DU (dangerous-undetected, SPF) | 0.5302 |
| λ_LF (latent-fault residual) | 0.5251 |
| λ_DPF (estimated dual-point) | 2.2999e-06 |

## 3. Metrics vs ASIL target

| Metric | Computed | Target (ASIL-B) | Pass? |
|---|---:|---:|:---:|
| SPFM | 2.08 % | ≥ 90 % | ❌ |
| LFM  | 0.00 % | ≥ 60 % | ❌ |
| PMHF | 0.5302 FIT | ≤ 100 FIT | ✅ |

**Overall:** ❌ fails ASIL-B target.

## 4. Findings and next actions

Module-level FMEDA does **not** meet the ASIL-B target (SPFM 2.08 % below target 90 %; LFM 0.00 % below target 60 %).

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
    --module npu_top \
    --asil ASIL-B \
    --output docs/safety/fmeda/npu_top_fmeda.md
```
