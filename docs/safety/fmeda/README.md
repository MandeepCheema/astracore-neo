# FMEDA Reports

**Purpose.** Per-module Failure Modes, Effects, and Diagnostic Analysis per ISO 26262-5 §7.4.5.

**Status:** Empty at v0.1. Reports populate during Track 2 W2-W10.

## Index (planned)

| Module | Report file | Wk | Status |
|---|---|---|---|
| `dms_fusion` | `dms_fusion_fmeda.md` | W2 | ✅ v0.1 (2026-04-20) — module-level SPFM 84.7 %, LFM 91.95 %, PMHF 0.008 FIT @ ASIL-D placeholder rates. Top SPF: `tmr_valid_r` SEU (uncovered at module scope; lifts at aggregate). |
| `npu_top` (with `npu_pe`, `npu_systolic_array`, `npu_dma`, `npu_sram_*`) | `npu_top_fmeda.md` | W3 | ✅ v0.1 (2026-04-20) — **SPFM 2.08 %, LFM 0 %, PMHF 0.53 FIT @ ASIL-B placeholder rates.** Largest uncovered exposures: PE accumulator + weight registers (no parity), SRAM data (ECC RTL exists but not wired in), tile_ctrl config registers. Drives the RTL hardening backlog in `docs/best_in_class_design.md` §7 Track 1. |
| `tmr_voter` | `tmr_voter_fmeda.md` | W4 | ✅ v0.1 (2026-04-20) — SPFM 31.35 % @ ASIL-B placeholder rates. Module-level closure requires F4-D-1 formal proofs (TMR voter is itself a safety mechanism with no internal protection). |
| `ecc_secded` | `ecc_secded_fmeda.md` | W4 | ✅ v0.1 (2026-04-20) — **SPFM 1.49 %** — `data_out` and `parity_out` output regs are fully uncovered. Closure: F4-D-2 formal proofs + parity-on-data_out fix. Same primitive-with-no-internal-protection pattern as tmr_voter. |
| `safe_state_controller` | `safe_state_controller_fmeda.md` | W5 | ✅ v0.1 (2026-04-20) — SPFM 34.80 %. Top action item: **F4-A-7** — TMR or Hamming-encode the 2-bit `safe_state` FSM register (cheapest, biggest single-module impact; flagged MUST FIX before ASIL-B safety case v1.0). |
| `plausibility_checker` | `plausibility_checker_fmeda.md` | W5 | ✅ v0.1 (2026-04-20) — SPFM 27.20 %. Counters classified no-effect (telemetry only). Top dangerous: rule_logic stuck-at and asil_degrade SEU; closure via formal proof of valid-output-set + dual-rail rule eval. |
| `lane_fusion` | `lane_fusion_fmeda.md` | W6 | pending |
| `gaze_tracker`, `head_pose_tracker` | `dms_perception_fmeda.md` | W7 | pending |
| Sensor interfaces (camera/radar/lidar/IMU/US/GNSS/CAN/Ethernet/PCIe) | `sensor_io_fmeda.md` | W8 | pending |
| `aeb_controller`, `ldw_lka_controller`, `ttc_calculator` | `vehicle_dynamics_fmeda.md` | W9 | pending |
| Aggregate DC/LFM/SPFM/PMHF v0.1 | `aggregate_fmeda_v0_1.md` | W10 | pending |

## Baseline + regression gate

`baseline.json` (next to this README) is the committed FMEDA snapshot
across all catalogued modules. CI runs:

```bash
python -m tools.safety.regress_check --baseline docs/safety/fmeda/baseline.json
```

and exits non-zero if any module's SPFM/LFM drops > 1 percentage point
or PMHF rises > 0.001 FIT vs the baseline. After an intentional fix
that changes metrics, regenerate with `--emit-baseline` and commit.

## Tooling

FMEDA tool (Python) lives at `tools/safety/fmeda.py` (to be created at W2). Reads:
- `tools/safety/failure_modes.yaml` — per-module failure-mode catalog
- `tools/safety/safety_mechanisms.yaml` — mechanism coverage table
- RTL module list (auto-discovered from `rtl/`)

Outputs: per-module markdown report + aggregate roll-up.

## Reference

ISO 26262-5 §7.4.5; ISO 26262-11 §4.6.

See also: `docs/safety/iso26262_gap_analysis_v0_1.md` Part 5 row.
