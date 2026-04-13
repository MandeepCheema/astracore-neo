# AstraCore Neo — Master Build Tracker

> **Session Protocol:** Read this file first. Python sim is COMPLETE (842/842). Next work is RTL — find the next PENDING Verilog module in the RTL Status table, write it, run cocotb, update this file.

## Project Phases

| Phase | Status | Notes |
|-------|--------|-------|
| Python behavioral simulation | ✅ COMPLETE | 842/842 tests, all 11 modules |
| RTL (Verilog) + cocotb verification | ✅ COMPLETE | 108/108 tests, all 11 modules |
| FPGA implementation (Vivado) | 🔄 IN PROGRESS | Top-level done, synthesis pending |
| ASIC (OpenLane + sky130) | ⏳ NOT STARTED | After FPGA validated |

## Module Status

| # | Module | Status | Session Date | Tests | Pass Rate | Log |
|---|--------|--------|-------------|-------|-----------|-----|
| 1 | hal | DONE | 2026-03-31 | 79 | 79/79 (100%) | logs/test_hal.log |
| 2 | memory | DONE | 2026-03-31 | 73 | 73/73 (100%) | logs/test_memory.log |
| 3 | compute | DONE | 2026-03-31 | 91 | 91/91 (100%) | logs/test_compute.log |
| 4 | inference | DONE | 2026-03-31 | 65 | 65/65 (100%) | logs/test_inference.log |
| 5 | perception | DONE | 2026-04-02 | 83 | 83/83 (100%) | logs/test_perception.log |
| 6 | safety | DONE | 2026-04-02 | 92 | 92/92 (100%) | logs/test_safety.log |
| 7 | security | DONE | 2026-04-02 | 75 | 75/75 (100%) | logs/test_security.log |
| 8 | telemetry | DONE | 2026-04-02 | 76 | 76/76 (100%) | logs/test_telemetry.log |
| 9 | dms | DONE | 2026-04-02 | 78 | 78/78 (100%) | logs/test_dms.log |
| 10 | connectivity | DONE | 2026-04-02 | 75 | 75/75 (100%) | logs/test_connectivity.log |
| 11 | models | DONE | 2026-04-02 | 55 | 55/55 (100%) | logs/test_models.log |

## RTL Status (Verilog + cocotb)

Run any sim: `cd sim/<module> && PATH=../../.venv/bin:$PATH make`

| # | Module | Verilog | cocotb Tests | Result |
|---|--------|---------|--------------|--------|
| 1 | gaze_tracker | rtl/gaze_tracker/gaze_tracker.v | sim/gaze_tracker/ | ✅ 11/11 PASS |
| 2 | thermal_zone | rtl/thermal_zone/thermal_zone.v | sim/thermal_zone/ | ✅ 10/10 PASS |
| 3 | canfd_controller | rtl/canfd_controller/canfd_controller.v | sim/canfd_controller/ | ✅ 9/9 PASS |
| 4 | ecc_secded | rtl/ecc_secded/ecc_secded.v | sim/ecc_secded/ | ✅ 9/9 PASS |
| 5 | tmr_voter | rtl/tmr_voter/tmr_voter.v | sim/tmr_voter/ | ✅ 9/9 PASS |
| 6 | fault_predictor | rtl/fault_predictor/fault_predictor.v | sim/fault_predictor/ | ✅ 11/11 PASS |
| 7 | head_pose_tracker | rtl/head_pose_tracker/head_pose_tracker.v | sim/head_pose_tracker/ | ✅ 11/11 PASS |
| 8 | pcie_controller | rtl/pcie_controller/pcie_controller.v | sim/pcie_controller/ | ✅ 8/8 PASS |
| 9 | ethernet_controller | rtl/ethernet_controller/ethernet_controller.v | sim/ethernet_controller/ | ✅ 9/9 PASS |
| 10 | mac_array | rtl/mac_array/mac_array.v | sim/mac_array/ | ✅ 11/11 PASS |
| 11 | inference_runtime | rtl/inference_runtime/inference_runtime.v | sim/inference_runtime/ | ✅ 10/10 PASS |

## FPGA Status (Arty A7-35T — xc7a35ticsg324-1L)

Run lint check: `python fpga/scripts/synth_check.py`
Run Vivado build: `vivado -mode batch -source fpga/scripts/build.tcl`

| # | Deliverable | File | Status |
|---|-------------|------|--------|
| 1 | Top-level integration | rtl/astracore_top/astracore_top.v | ✅ DONE |
| 2 | Pin constraints (XDC) | fpga/constraints/arty_a7_35t.xdc | ✅ DONE |
| 3 | Vivado build script | fpga/scripts/build.tcl | ✅ DONE |
| 4 | Pre-synth lint check | fpga/scripts/synth_check.py | ✅ PASS (0 warnings, 12 files) |
| 5 | Vivado synthesis | fpga/reports/utilization.rpt | ⏳ Needs Vivado install |
| 6 | Timing closure | fpga/reports/timing.rpt | ⏳ After synthesis |
| 7 | Bitstream | fpga/output/astracore_neo.bit | ⏳ After timing closure |

### Top-Level AXI4-Lite Register Map

| Offset | Name | Direction | Description |
|--------|------|-----------|-------------|
| 0x00 | CTRL | W | [0]=mod_valid (pulse) [1]=sw_rst |
| 0x04 | GAZE | W | [7:0]=left_ear [15:8]=right_ear |
| 0x08 | THERMAL | W | [7:0]=temp_in |
| 0x0C | CANFD | W | [0]=tx_success [1]=tx_error [2]=rx_error [3]=bus_off_recovery |
| 0x10/14 | ECC_LO/HI | W | 64-bit data_in |
| 0x18 | ECC_CTRL | W | [0]=mode [15:8]=parity_in |
| 0x1C/20/24 | TMR_A/B/C | W | 32-bit TMR lane inputs |
| 0x28 | FAULT | W | [15:0]=fault_value |
| 0x2C | HEAD_POSE | W | [7:0]=yaw [15:8]=pitch [23:16]=roll |
| 0x30–3C | PCIE_* | W | link_up/down, TLP type/start, req_id, addr, length |
| 0x40 | ETH | W | [0]=rx_valid [1]=rx_last [9:2]=rx_byte |
| 0x44 | MAC | W | [0]=valid [1]=clear [9:2]=a [17:10]=b |
| 0x48 | INF_CTRL | W | load/run/abort/done strobes |
| 0x80 | GAZE_ST | R | eye_state, perclos_num, blink_count |
| 0x84 | THERM_ST | R | state, throttle_en, shutdown_req |
| 0x88 | CANFD_ST | R | tec[8:0], rec[7:0], bus_state[1:0] |
| 0x8C–94 | ECC_ST/D | R | single/double/corrected, err_pos, data_out |
| 0x98–9C | TMR_* | R | voted result, fault flags |
| 0xA0 | FAULT_ST | R | risk, alarm, rolling_mean |
| 0xA4 | HEAD_ST | R | in_zone, distracted_count |
| 0xA8–B4 | PCIE_* | R | link_state, tlp_done, 96-bit tlp_hdr |
| 0xB8–BC | ETH_ST | R | frame_ok/err, types, byte_count, ethertype |
| 0xC0 | MAC_RES | R | 32-bit accumulated result |
| 0xC4 | INF_ST | R | state, busy, session_done, error |

## Dependency Graph

```
hal
├── memory (needs hal)
├── compute (needs hal, memory)
│   └── inference (needs compute, memory)
├── perception (needs hal, memory)
│   └── dms (needs compute, perception)
├── safety (needs hal, compute)
│   └── telemetry (needs hal, safety)
├── security (needs hal)
└── connectivity (needs hal)
models (needs inference, perception)
```

## Build Order

1. hal → 2. memory → 3. compute → 4. inference → 5. perception → 6. safety → 7. security → 8. telemetry → 9. dms → 10. connectivity → 11. models

## Per-Module Checklist

Before marking DONE, confirm all 4 deliverables exist:
- [ ] `src/<module>/` — implementation files
- [ ] `tests/test_<module>.py` — testbench
- [ ] `logs/test_<module>.log` — captured test run output
- [ ] `docs/<module>.md` — documentation

## Session Log

| Date | Module | Notes |
|------|--------|-------|
| 2026-03-31 | — | Project initialized |
| 2026-03-31 | hal | 79/79 tests pass. Fixed: STATUS/CLK_STATUS are R/O to SW, use _hw_write() for internal device sim updates. |
| 2026-03-31 | memory | 73/73 tests pass. Fixed: INT4 test used bytes(range(128)) which clamps to nibble range — corrected to use i%16 data. |
| 2026-03-31 | compute | 91/91 tests pass. Fixed: identity matmul test must use FP32 (float [0,1) casts to 0 in INT8); layer norm variance tolerance loosened to 2e-4 (biased np.var). |
| 2026-03-31 | inference | 65/65 tests pass. Clean first run — no fixes needed. |
| 2026-04-02 | perception | 83/83 tests pass. Clean first run — code was pre-written, just needed venv setup and test run. |
| 2026-04-02 | safety | 92/92 tests pass. Fixed: SECDED overall-parity syndrome computation — P7 must be XOR of all codeword bits (data + P0-P6), not just data; flipping a data bit also flips its Hamming parity bit, cancelling out in a naive computation. Separated _compute_hamming() from _compute_parity(). |
| 2026-04-02 | security | 75/75 tests pass. Fixed: (1) TEE switch_to_normal must increment SMC counter; (2) OTA begin_update must allow COMPLETE state for sequential updates; (3) UpdatePackage.payload_hash must be stored at creation time (use __post_init__), not recomputed from payload; (4) tampered-payload test catches SecurityBaseError (not OTAError) since signature check fires after auto-hash. |
| 2026-04-02 | telemetry | 76/76 tests pass. Clean first run — no fixes needed. Three sub-systems: TelemetryLogger (ring buffer, LogLevel, per-level counters), ThermalMonitor/ThermalZone (5-state NOMINAL→SHUTDOWN machine, slope tracking), FaultPredictor/MetricTracker (rolling window stats, spike z-score detection, trend escalation). |
| 2026-04-02 | dms | 78/78 tests pass. Clean first run — no fixes needed. GazeTracker (EAR→EyeState, PERCLOS rolling window, blink counting), HeadPoseTracker (AttentionZone ±yaw/pitch/roll, distraction ratio), DMSAnalyzer (5-state: ALERT/DROWSY/DISTRACTED/MICROSLEEP/EMERGENCY), DMSMonitor (single process_frame() entry point). |
| 2026-04-02 | connectivity | 75/75 tests pass. Fixed: Ethernet test OTHER_MAC first byte 0x11 is odd (multicast bit set); changed to 0x22. CAN-FD, Ethernet, PCIe (BAR/TLP), V2X (DSRC channels, RSSI filter), ConnectivityManager. |
| 2026-04-02 | models | 55/55 tests pass. Clean first run — no fixes needed. ModelDescriptor (versioned metadata), HardwareSpec + ModelValidator (4 violation types), ModelCatalog (registry, filter, recommendation), 5 reference models all pass ASTRA_HW_SPEC validation. |
