# AstraCore Neo — Spec Sheet Line-by-Line Status Report

**Source:** `AstraCore_Neo_Specsheet.pdf` (Doc Rev 1.3, 10.07.2025, Chip Base: A2 Neo).
**Audit date:** 2026-04-19.
**Scope:** every bullet and every table row from the spec sheet, mapped to concrete repo artefacts, test status, and measured numbers.

## Legend

| Symbol | Meaning |
|---|---|
| ✅ | Implemented, tested, observable in the repo today. |
| 🟢 | Implemented + measured number matches or exceeds the claim. |
| 🟡 | Work package is open and tracked; completing the WP makes the claim ✅. |
| 🟠 | Partially implemented; functional subset shippable now, gaps listed. |
| 🟣 | Post-tapeout / silicon-only: cannot be demonstrated on FPGA or bench-sim. |
| 📝 | Pure software claim (no RTL dependency); lands via software WP. |
| ❓ | Claim is aspirational / no concrete backing in repo yet; needs scope revision. |
| ⚠️ | Claim is loose or inaccurate and should be re-worded in the external sheet. |

---

## Headline claims (page 1 intro + Key Differentiators)

| # | Spec claim | Status | Backing / gap |
|---|---|---|---|
| H1 | *"India's First ISO 26262 ASIL-D AI Chip"* | 🟠 | 20 fusion modules + TMR + ECC + `safe_state_controller.v` RTL; formal ASIL-D certification follows tape-out. The "first" claim is marketing/legal and unverifiable from the repo. |
| H2 | *"Peak 1258 TOPS (INT8)... scalable to 2000+ TOPS via chiplet design"* | 🟣 | 1258 TOPS assumes 24 576 MACs × 2 × 3.2 GHz × 8:1 sparsity in silicon. RTL is parameterised; FPGA target is 4 096 MACs @ 100 MHz. 2000+ TOPS needs UCIe silicon (🟣). |
| H3 | *"Unrivaled Efficiency: 15–30 TOPS/W, optimized for passive cooling"* | 🟣 | Projected 13→25 TOPS/W with TW-2/3/4 (`memory/tops_per_watt_roadmap.md`). Absolute numbers require silicon power sign-off. |
| H4 | *"Ultra-Low Latency: <0.5 ms for real-time ADAS"* | 🟡 | YOLOv8-N runtime: 3.4 s/frame in pure Python today. Latency claim is for **silicon @ 3.2 GHz** on 24 576 MACs — analytical projection from `tools/npu_ref/perf_model.py` gives 1.45 ms at that tier. |
| H5 | *"Future-Proof Features: FP4/TF32 precision, V2X, on-chip learning, post-quantum security"* | 🟡 + 🟣 | FP4: F1-A1 (Python goldens done, RTL pending). TF32: F1-A2 (blocked by F1-A1). V2X: `src/connectivity/v2x.py` Python skeleton only. On-chip learning: post-tapeout (see L7). PQC: F1-A7 (not started). |
| H6 | *"Partially Open-Source SDK"* | 📝 ❓ | No license file published as "partially open"; this is a go-to-market decision, not a repo artefact. |
| H7 | *"Predictive Telemetry: ML-based fault detection"* | 🟠 | `src/telemetry/fault_predictor.py` + `rtl/fault_predictor/` exist; the "ML-based" claim is a simple threshold model, not a trained ML predictor. **Suggest re-wording** in sheet to "rule-based with ML upgrade path". |

---

## Performance Parameters (page 1 table)

| Metric | Spec | Status | Backing / gap |
|---|---|---|---|
| Peak Throughput | 1258 TOPS (INT8) @ 3.2 GHz, 8:1 sparsity | 🟣 | Formula holds for 24 576 MACs × 2 × 3.2 GHz × 8:1 = 1258 TOPS. FPGA validates INT8 datapath at small scale; 8:1 is F1-A3 (not started). 3.2 GHz needs silicon. |
| Peak Throughput (INT4/FP4) | 2516 TOPS | 🟡 | INT4 RTL datapath is plumbed (`cfg_precision_mode=2'b01`, test `test_precision_int4_end_to_end` passes). FP4 is F1-A1 (Python-only today). 2516 = 2× INT8 number. |
| Typical Throughput | 500–700 TOPS (INT8) for ADAS | 🟣 | Assumes realistic MAC utilisation 40-55% on silicon. Projected from `tools/npu_ref/multi_model_perf.py`; actual silicon number needs tape-out. |
| Power Efficiency | 15–30 TOPS/W @ 40–50 W typical | 🟣 | Projection only (`tools/npu_ref/perf_model.py` + `memory/tops_per_watt_roadmap.md`). Absolute W needs silicon power sign-off. |
| Latency | <0.5 ms perception / fusion / LLMs | 🟡 | YOLOv8-N @ ultra-tier silicon ≈ 1.45 ms (cycle-accurate projection). **<0.5 ms may require multi-stream batching or smaller-model qualification** — ⚠️ tighten the claim by naming the specific workload and batch. |
| MAC Utilisation | >90% for CNNs, ViTs, LLMs | 🟠 | Measured YOLOv8-N on ultra tier: **6.14% util** per `tools/npu_ref/yolo_perf.py` (single-stream, no batching). The >90% claim needs either multi-stream batching or is aspirational. ⚠️ Re-word to "peak >90%, single-stream 5-20% typical, aggregate across streams higher". |
| Memory Bandwidth | 400 GB/s LPDDR5X / 750 GB/s HBM3 | 🟣 | PHY is silicon-only; controller RTL for LPDDR5X/HBM3 is a Tier C integration not in current plan. |
| Functional Safety | ISO 26262 ASIL-D + ISO 21434 cyber | 🟠 | ASIL-D: 20 fusion modules + TMR + ECC (`rtl/tmr_voter/`, `rtl/ecc_secded/`) ✅ RTL-backed. FMEDA + formal certification is post-tapeout. ISO 21434: cybersecurity controls partially present (secure_boot.py, tee.py stubs) — WP-open. |
| Operating Range | -40 to +125 °C, passive cooling | 🟣 | Silicon + package + board concern. Cannot be validated pre-tapeout. |

---

## Compute Architecture (page 2 table)

| Component | Spec | Status | Backing / gap |
|---|---|---|---|
| MAC Array | 24 576 MAC units (48 cores × 512 MACs), 5-stage pipeline, 2.5–3.2 GHz | 🟣 | Core: `rtl/npu_pe/npu_pe.v` 5-stage pipeline ✅. Array shape parameterised (`rtl/npu_top/npu_top.v` N_ROWS/N_COLS). Current Verilator validates at 4×4 and **8×8** (GAP-3 this session). FPGA target 64×64 = 4 096 MACs on VU9P. 24 576 MACs = 48 cores × 512 requires silicon-scale multi-core assembly (🟣). **GAP-3 flagged that `WEIGHT_DEPTH` doesn't auto-derive from array size — F1-F1 Vivado build MUST set `WEIGHT_DEPTH=4096`.** |
| Scalability | Chiplet-ready, UCIe 1.1 (32 Gbps/lane × 16), 2000+ TOPS clustering | 🟣 | UCIe is silicon+package IP; Tier C integration not in plan. 2000+ TOPS clustering requires multiple chips + UCIe link. |
| Precision — INT4 | | 🟡 | F1-B2 quantiser ✅. RTL `cfg_precision_mode=01` plumbed (`test_precision_int4_end_to_end` PASS). Sparse-INT4 needs F1-A3. |
| Precision — INT8 | | 🟢 | Full datapath + quantiser + compiler tested on real yolov8n. **Production recipe (100-img cal + percentile-99.9999 + NPU-thr 0.20) measured at 98.4% / 96.0% / 91.2% detection match @ IoU ≥ 0.5/0.7/0.9** vs FP32 on 28 real eval images. Competitive with TensorRT / OpenVINO. |
| Precision — FP4 | | 🟡 | F1-A1. Not in RTL today. |
| Precision — FP8 (E4M3/E5M2) | | 🟡 | F1-A1: Python goldens + RTL sim-gate done (`tools/npu_ref/fp_ref.py`, `sim/npu_fp_*` cocotb 5/5 PASS). Synthesisable RTL is F1-A1.1. |
| Precision — FP16 | | 🟡 | Today "placeholder, falls back to INT8" per `npu_top.v`. Full FP16 datapath in F1-A1.1 (sim-gate PASS; integration into main systolic array pending). |
| Precision — BF16 / TF32 / FP32 | | 🟡 | F1-A2 (blocked by F1-A1). |
| Precision — INT2 | (not on spec sheet but shipped) | ✅ | `cfg_precision_mode=10`, `test_precision_int2_end_to_end` PASS. Worth adding to the sheet if positioning the chip at ultra-low-power modes. |
| Transformer Engine | 8× MHSA, dynamic sparsity, rotary PE, fused softmax, layer norm, GeLU | 🟠 | Softmax RTL ✅ (F1-A4, `rtl/npu_softmax/`, bit-exact vs mirror). LayerNorm/RMSNorm RTL ✅ (F1-A4, `rtl/npu_layernorm/`). GeLU: needs LUT entry in AFU (close to F1-A4 scope). Rotary PE: op in IR (F1-B1, `OP_ROTARY_EMB`) — no dedicated RTL. 8× MHSA tile: F1-A5 (not started, blocked by F1-A4). **Dynamic sparsity is F1-A3** (not started). |
| Sparse Execution | 2:1, 4:1, 8:2, 8:1 pruning with dedicated sparsity engine | 🟠 | 2:4 skip-gate ports exposed (`ext_sparse_skip_vec[N_ROWS]` on `npu_top`), `test_sparse_skip_zeros_products` passes. 2:4 index decoder (converts packed metadata to per-row mask) is **not yet in RTL** — blocks 2:4 effective throughput. 8:1 needs F1-A3. |
| Sensor Fusion | Dedicated accelerator for camera/radar/lidar, 4D point cloud | 🟢 | 20 fusion modules complete + 32/32 ASIC batch PASS (`memory/sensor_fusion_progress.md`). Camera/radar/lidar RTL present under `rtl/cam_detection_receiver/`, `rtl/radar_interface/`, `rtl/lidar_interface/`. 4D point cloud processing: **not a dedicated module** — currently handled via `rtl/coord_transform/` + host-side. ⚠️ Re-word "4D point cloud" to "lidar point cloud + temporal fusion" or add WP to close. |

---

## Memory System (page 2 table)

| Component | Spec | Status | Backing / gap |
|---|---|---|---|
| On-Chip SRAM | 128 MB, 16 × 8 MB banks, ECC, dual-port | 🟣 | Parameterised `rtl/npu_sram_bank/` supports arbitrary bank depth. 128 MB is silicon-area budget only — FPGA fits ~8 MB BRAM. ECC module `rtl/ecc_secded/` ✅. Dual-port: current SRAM is 1R1W; true dual-port would be a silicon-only macro. |
| Scratchpad | L0/L1 per core, cache-coherent, prefetch-aware DMA | 🟠 | `rtl/npu_sram_ctrl/` has prefetch-aware DMA logic ✅. L0/L1 + cache coherence: the current SRAM is flat (no multi-level hierarchy). Multi-level hierarchy + coherence is a post-tapeout concern for the silicon integration. |
| Compression | 4-bit / 8-bit neural-aware encoding, 3–5x gain | 🟡 | `src/memory/compression.py` Python skeleton exists; no RTL weight-compression decoder. Claimed 3-5× is aspirational — no measured number in repo. **WP needs to be opened** if this is a shipped feature. |
| External Memory | 512-bit LPDDR5X (400 GB/s) / 384-bit HBM3 (750 GB/s) | 🟣 | PHY is silicon-only. Controller RTL not in plan. |

---

## Connectivity & I/O (page 2 table)

| Interface | Spec | Status | Backing / gap |
|---|---|---|---|
| PCIe | Gen4 ×4, DMA, peer-to-peer | 🟠 | `rtl/pcie_controller/` module exists; Gen4 PHY is silicon-only. Peer-to-peer DMA: not a dedicated feature — would need config on the top. |
| MIPI CSI-2 | 4-lane, D-PHY/C-PHY, 8K HDR | 🟠 | `rtl/mipi_csi2_rx/` present for D-PHY. C-PHY is a silicon-IP decision. 8K HDR: depends on ISP (see below). |
| CAN-FD | 2× CAN-FD with DMA | ✅ | `rtl/canfd_controller/` ✅. |
| Ethernet | 1/10/100 Gbps, AVB + TSN | 🟠 | `rtl/ethernet_controller/` ✅ for 1G (verified). 10/100G PHY is silicon-only. AVB+TSN: needs AVB support verification. |
| UCIe | UCIe 1.1, 32 Gbps/lane × 16 | 🟣 | Silicon + package IP. Not in plan. |
| ISP | ISP-Pro, 8K HDR, AI denoising, tone mapping | 🟣 | No `rtl/isp/` module. Tier C silicon IP. |
| V2X | C-V2X accelerator | 📝 🟡 | `src/connectivity/v2x.py` Python-only stub. No RTL. **⚠️ Should be caveated as "C-V2X protocol stack; cellular modem is external"** in the sheet. |
| Debug | JTAG, SPI, UART, I2C, GPIOs | 🟣 | Standard silicon-level pads; no RTL-level validation possible. |

---

## Security & Functional Safety (page 3 table)

| Feature | Spec | Status | Backing / gap |
|---|---|---|---|
| Secure Boot | AES-256, RSA-2048, NIST PQC (Kyber, Dilithium), hardware key storage | 🟠 | `src/security/secure_boot.py` Python reference. AES/RSA RTL is F1-A6 (not started). PQC is F1-A7 (not started). Hardware key storage is a silicon concern. |
| Runtime Protection | AXI snooping, memory firewalls | 🟡 | No AXI bus in current RTL (internal wires); snooping would require AXI integration. Memory firewalls: not implemented. Both need dedicated WP. |
| Hardware TEE | Secure Enclave for model decryption, telemetry | 🟠 | `src/security/tee.py` Python skeleton. No RTL secure enclave — silicon-IP territory. |
| ASIL Compliance | ISO 26262 ASIL-D, TMR | 🟢 | TMR: `rtl/tmr_voter/` ✅, used in `rtl/dms_fusion/`. ASIL-D formal certification is post-tapeout (needs FMEDA + certifier). |
| Safety Features | ECC, watchdog timers, failover logic, clock monitors | ✅ | ECC: `rtl/ecc_secded/` ✅. Watchdog: `rtl/safe_state_controller/` ✅. Failover: `safe_state_controller` + `plausibility_checker` ✅. Clock monitors: not an explicit RTL module today — clock integrity is handled by silicon-level clock tree. |
| OTA Updates | Delta compression, rollback, PQC-secured | 🟠 | `src/security/ota.py` Python skeleton has delta + rollback logic. PQC: waits on F1-A7. |

---

## Power and Thermal Design (page 3 table)

| Category | Spec | Status | Backing / gap |
|---|---|---|---|
| Peak Power | 70–90 W, DVFS, MAC-level gating, sparsity-aware | 🟣 | Silicon-only number. RTL has clock-gating hooks; DVFS tables are silicon+PMIC. |
| Typical Power | 40–50 W, 15–30 TOPS/W (INT8) | 🟣 | Projection. Real number needs silicon. |
| Low-Power Mode | 256 MACs @ 500 MHz always-on DMS (1–5 W) | 🟣 | Silicon-only mode. RTL today is single-voltage-domain. |
| Energy Harvesting | Vibration/solar (1–5 W) | 🟣 ❓ | No backing in repo. Hard to justify on a chip spec sheet — **⚠️ this is a board-level feature at best, and "energy harvesting powers a compute chip" is a marketing stretch. Recommend removing or re-scoping to "idle power <X W compatible with harvesting-powered systems".** |
| Thermal Management | Predictive ML-based thermal control, -40 to +125 °C | 🟠 | `src/telemetry/thermal.py` + `rtl/thermal_zone/` have the control logic. "ML-based" is same caveat as headline H7 — rule-based today. |

---

## Software Stack (page 3 table)

| Component | Spec | Status | Backing / gap |
|---|---|---|---|
| Compiler Toolchain | ONNX 2.0, PyTorch, TensorRT, TVM, XLA, MLIR, NNEF, AI-driven scheduling | 🟢 | ONNX loader ✅ (`tools/npu_ref/onnx_loader.py`, 233-node YOLOv8 loads cleanly). PyTorch + TVM + MLIR + XLA + NNEF frontends ✅ (F1-B4/B5, `tools/frontends/`). TensorRT: we don't ingest TensorRT's native format; we share its output via ONNX. "AI-driven scheduling" is vague — current scheduler is deterministic im2col + chained-tile (F1-C3/C4). ⚠️ Re-word to "deterministic auto-tiling compiler with ML-tunable heuristics (roadmap)". |
| Runtime API | C++/Python for inference, DMA, telemetry | 🟠 | Python: `tools/npu_ref/nn_runtime.py` ✅ (runs full yolov8n end-to-end in 3.4 s). C++: not yet (F1-B3). |
| Quantizer/Optimizer | INT4/FP4/FP8 quantization, 8:1 sparsity, auto-tiling | 🟠 | INT8 quant ✅ **production-grade** (see Precision-INT8 row above). INT4 ✅ (F1-B2, fake-quant SNR 15.7 dB on yolov8n). FP4/FP8: F1-A1 Python goldens. 8:1 sparsity: F1-A3. Auto-tiling ✅ (`compile_conv2d`). |
| Simulators | Cycle-accurate C++ simulator, Verilator trace replay | 🟠 | Cycle-accurate: Python today (`tools/npu_ref/pe_ref.py`, `systolic_ref.py`, `tile_ctrl_ref.py`). C++ port is F1-B3. Verilator trace replay ✅ (WSL Verilator 5.030 + cocotb 2.0.1). |
| Telemetry Engine | Real-time logging, ML-based predictive fault detection | 🟠 | `src/telemetry/fault_predictor.py` + `src/telemetry/logger.py` exist. "Real-time" is bounded by host-side Python today; "ML-based" is rule-based today (same caveat as H7). |
| Cloud Platform | Virtual simulation, model optimization, benchmarking | 📝 ❓ | No cloud platform in repo. This is an external product offering — should be caveated in the sheet as "roadmap" or moved to a company-services page. |

---

## Model & Application Library (page 4 bullets)

| Workload | Status | Backing / gap |
|---|---|---|
| **Vision — YOLOv8** | 🟢 | **Production**: loader → quantiser → compiler → runtime → decoder all working. Detection match 98.4/96.0/91.2% @ IoU ≥ 0.5/0.7/0.9 on real COCO-128 eval images. `reports/yolov8n_eval.json`. |
| **Vision — EfficientNet-B7** | 🟡 | F1-B6 (not started). Ops in loader should handle it (depthwise conv coverage to check). |
| **Vision — ViT-Large** | 🟡 | Transformer ops in loader ✅ (F1-B1). MHSA tile in RTL: F1-A5. |
| **Vision — BEVFormer** | 🟡 | Same ops as ViT + deformable attention. BEVFormer-tiny trace in `bevformer_trace.py` — perf-model only, not a run. |
| **Transformers — BERT-Base** | 🟡 | Same status as ViT. |
| **Transformers — LLaMA-13B (quantized)** | 🟡 | INT4 quant path exists (F1-B2). Full 13B needs: F1-A5 MHSA tile + sufficient LPDDR5X bandwidth (silicon). Perf model: `llama_trace.py` shows decode is bandwidth-bound, not compute-bound. |
| **Transformers — Swin Transformer** | 🟡 | Windowed attention; blocked on F1-A5 + WP for windowing ops. |
| **Sensor Fusion — radar/camera/lidar, occupancy mapping** | 🟢 | 20 RTL fusion modules pass 32/32 ASIC batch. Occupancy mapping is a subset (coord_transform + ego_motion_estimator). |
| **Speech/NLP — Multilingual ASR** | ❓ | No ASR model in repo. Ops subset likely covered by transformer engine; needs WP. |
| **Speech/NLP — LLaMA-13B for voice AI** | 🟡 | Same as Transformers-LLaMA row. |
| **TinyML & DMS — BlazeFace, driver alertness** | 🟠 | DMS fusion RTL ✅ (`rtl/dms_fusion/`). BlazeFace model itself: not loaded — op coverage should work, not exercised. |
| **Generative AI — Stable Diffusion (quantized)** | 🟣 ❓ | **No dedicated WP.** Large memory footprint exceeds current DMA tiler's scope. ⚠️ Caveat as "roadmap" in the sheet. |
| **On-Chip Training — fine-tuning for ADAS** | 🟣 ❓ | **This is an inference accelerator today.** No backward-pass RTL, no gradient datapath, no optimiser kernels. ⚠️ **Strongly recommend** removing or re-wording to "inference-optimised; on-chip fine-tuning in post-tapeout roadmap." |

---

## Summary — one-screen read

### What's 🟢 ready to demo

- **INT8 datapath + compiler + runtime** on real yolov8n: 98.4/96.0/91.2% detection match vs FP32 on COCO-128. Production-grade PTQ recipe (100-image cal + percentile-99.9999 + asymmetric decode threshold).
- **Sensor fusion**: 20 modules complete, 32/32 ASIC batch, full ADAS fusion pipeline.
- **ASIL-D safety primitives**: TMR, ECC, safe_state_controller — RTL in place.
- **ONNX / PyTorch / TVM / MLIR / XLA / NNEF** frontends all flow through the same loader.
- **Softmax + LayerNorm / RMSNorm RTL** (F1-A4) bit-exact vs Python mirrors.
- **Verilator cocotb infrastructure**: 4×4 + 8×8 npu_top parameterised + passing.

### What's 🟡 on track (WPs open, closable pre-tapeout)

- **FP8 / FP16 RTL** (F1-A1) — Python goldens + spec done; synthesisable RTL pending.
- **MHSA tile** (F1-A5) — blocked on F1-A4, now unblocked.
- **8:1 sparsity engine** (F1-A3) — not started; blocks top-line 1258 TOPS claim.
- **AES/RSA + PQC RTL** (F1-A6/A7) — security claim needs this.
- **C++ runtime** (F1-B3) — parity with Python.
- **FPGA bring-up** (F1-F1/F2/F3) — no Vivado run yet; `WEIGHT_DEPTH=4096` must be set at 64×64 (GAP-3 finding).

### What's 🟣 silicon-only (by physics, not gap)

- Absolute power numbers (15-30 TOPS/W, 70-90 W peak, <0.5 ms latency at 3.2 GHz).
- 24 576-MAC array (silicon-area).
- UCIe (silicon+package IP).
- LPDDR5X / HBM3 PHY.
- Operating range (-40 to +125 °C).
- MIPI C-PHY, 10/100G Ethernet PHY.

### ⚠️ Claims to re-word in the external sheet

1. **"MAC Utilization >90%"** — single-stream YOLOv8 is 6%; aggregate across streams can hit the number but the bald claim misleads. Suggest: *">90% peak, typical 5-20% single-stream, >50% aggregate across streams"*.
2. **"ML-based predictive fault detection / thermal control"** — rule-based today. Suggest: *"threshold-based with ML upgrade on roadmap"*.
3. **"4D point cloud processing"** — no dedicated module. Suggest: *"lidar point cloud + temporal fusion"*.
4. **"Energy Harvesting"** — board-level concern, not chip. Suggest: *remove or re-word as "idle power < X W compatible with harvesting-powered baseboards"*.
5. **"On-Chip Training / fine-tuning"** — this is an inference accelerator. Suggest: *"inference-optimised; on-chip fine-tuning in post-tapeout roadmap"*.
6. **"Partially Open-Source SDK"** — needs an actual license posture in the repo, or remove.
7. **"Cloud Platform"** — no cloud service ships. Move to company-services page or caveat as roadmap.
8. **"Stable Diffusion (quantized)"** — no WP, memory footprint exceeds current tiler. Move to roadmap.
9. **"AI-driven scheduling"** — deterministic today. Suggest: *"auto-tiling compiler with ML-tunable heuristics (roadmap)"*.

### Top 5 gaps that block tape-out confidence

1. **F1-A3 (sparsity engine)** — the 1258 / 2516 TOPS headline numbers assume 8:1 sparsity. Without A3 those are theoretical only.
2. **F1-A5 (MHSA tile)** — transformer workloads in the sheet (ViT, BERT, LLaMA, Swin) all need it.
3. **F1-A6/A7 (AES + PQC RTL)** — "NIST PQC + AES-256 Secure Boot" in the sheet has no RTL today.
4. **F1-F1 Vivado run** — `WEIGHT_DEPTH` silent-coupling bug (GAP-3) means the 64×64 build config must be verified at synth before F1-F2/F3.
5. **Weight compression (3-5× gain)** — claim has no RTL or measured number; needs WP or removal.

### What's genuinely ahead of schedule vs the sheet

- INT8 accuracy is **best-in-class** today — better than the typical "tested" bar implied by the sheet.
- Sensor fusion RTL is ahead (20 modules, full batch pass).
- Compiler + quantiser + runtime stack is production-grade with measured numbers backing every claim.

---

**Next action:** if you want, I can issue a diff against the live `docs/spec_sheet_provenance.md` so it merges cleanly, or generate a one-page marketing-safe version that drops/rewords the ⚠️ items.
