# AstraCore Neo — IP Datasheet (Rev 1.4)

**Document ID:** ASTR-DATASHEET-V1.4
**Date:** 2026-04-20
**Element:** AstraCore Neo NPU + Sensor-Fusion IP Block
**Target node:** TSMC N7 (licensee tape-out)
**Classification:** Proprietary & Confidential — for licensee evaluation under NDA
**Supersedes:** Rev 1.3 (10.07.2025) chip-datasheet framing; Rev 1.4 DRAFT (`docs/spec_sheet_rev_1_4_draft.md`) chip-datasheet rewrite

---

## 0. Reading guide

This is an **IP datasheet**, not a chip datasheet. AstraCore licenses the
NPU + sensor-fusion IP block; the licensee integrates it into their SoC
on their tape-out program. Each spec table separates three buckets:

| Bucket | Meaning |
|---|---|
| ✅ **AstraCore IP delivers** | In the licensed RTL + toolchain today; demonstrable in the repo via cocotb / Verilator / pytest |
| 🟡 **Roadmap (pre-tape-out WP)** | Work package open in the licensed IP track; closes on FPGA without a 7 nm run |
| 🟪 **Reference SoC integration** | Out of AstraCore IP scope; supplied by the licensee or external IP partner (PHY, package, foundry SRAM compiler, certain crypto, ISP, UCIe, modems). Not a deficiency — this is standard IP-licensing practice |

The ⚠️ marker carried from rev 1.3 → rev 1.4 indicates a claim that
has been **reworded** to match the repo. No rev 1.3 wording survives
unmodified where it overstated capability.

**Companion documents** (the safety case the AstraCore IP brings to
the licensee's item-level safety case):

- `docs/safety/seooc_declaration_v0_1.md` — Safety Element out of Context per ISO 26262-10 §9 + 26262-11 §4.6
- `docs/safety/hara_v0_1.md` — Hazard Analysis & Risk Assessment per ISO 26262-3 §6 + §8 (3 reference use cases)
- `docs/safety/iso26262_gap_analysis_v0_1.md` — process gap analysis vs all 11 parts
- `docs/safety/safety_manual_v0_1.md` — licensee user-guide
- `docs/safety/fmeda/` — per-module FMEDA reports + baseline
- `docs/safety/findings_remediation_plan_v0_1.md` — quantified F4 remediation backlog
- `docs/best_in_class_design.md` — strategic positioning + 16-week plan
- `docs/spec_sheet_provenance.md` — claim-by-claim provenance to repo artefacts

---

## 1. Executive positioning

**AstraCore Neo** is a small, safety-shaped **automotive AI inference IP block** purpose-built for the **zonal / domain-ECU tier** of automotive E/E architectures. The same envelope works for industrial PLCs and medical edge.

**The IP wedge:**

| Dimension | AstraCore Neo IP |
|---|---|
| Compute IP | INT8 / INT4 / INT2 systolic-array NPU + FP8 / FP16 sim-gate (synthesizable RTL pre-tape-out WP); transformer AFUs (softmax, LayerNorm, RMSNorm, GeLU) RTL-backed |
| Sensor fusion IP | 48 RTL modules covering camera / radar / lidar / IMU / ultrasonic / GNSS / CAN-FD / Ethernet / PCIe; DMS, lane, and object-tracking fusion engines |
| Safety mechanisms | TMR voter, SECDED ECC (Hamming 72,64), safe-state controller, plausibility checker, fault predictor — all RTL-backed |
| Process | ISO 26262 Safety Element out of Context (SEooC) v0.1; HARA v0.1 (item-level ASIL-D); ASIL-B safety case Q4 2026 with TÜV SÜD India pre-engagement; ASIL-D extension follows |
| SDK | `pip install astracore-sdk`: ONNX 2.0 loader, INT8/INT4/INT2 quantiser, Python runtime, plug-in registries (ops / quantisers / backends), Tier-1 ADAS YAML harness, custom-fusion examples |
| Licensing | RTL under evaluation licence; SDK Apache-2.0; licensee provides their own tape-out + PHY + package |

**What the licensee gets** — ready-to-integrate RTL plus a credible safety-case starting point. **What the licensee provides** — the SoC integration, foundry SRAM compiler with optional ECC wrapping, memory PHY, package thermal, item-level HARA on their vehicle program, ISO 26262 confirmation review on their integrated item.

---

## 2. Performance Parameters

| Metric | Validated today | Tape-out target | Licensee-supplied |
|---|---|---|---|
| Peak Throughput (INT8) | 16 MAC default array; 64 MAC validated 8×8 — bit-exact vs Python mirror | 1258 TOPS at 24,576 MACs × 3.2 GHz × 8:1 sparsity (silicon target; AWS-F1 4096-MAC bring-up Q3 2026 measures intermediate point) | 7 nm timing closure |
| Peak Throughput (INT4 / FP4) | INT4 datapath validated; SNR 15.7 dB / cos 0.986 on yolov8n vs FP32 | 2516 TOPS projection (2× INT8) | Same as above |
| Typical Throughput (ADAS mix) | Multi-stream scaling measured on host CPU + ONNX Runtime backend (1.48–1.90×); silicon multiplier projection in `tools/npu_ref/perf_model.py` | 500–700 TOPS aggregate, ADAS workload mix | Mission-profile measurement |
| Power Efficiency | 13 TOPS/W projected at INT2 dense per `memory/tops_per_watt_roadmap.md`; trajectory to 20–25 TOPS/W with TW-2/3/4 work packages | 15–30 TOPS/W typical | Silicon power sign-off |
| Latency | YOLOv8-N projected 1.45 ms on full-spec ultra-tier silicon (cycle-accurate model in `tools/npu_ref/perf_model.py`) | < 0.5 ms achievable via multi-stream batching or smaller-model configurations on full-spec silicon. ⚠️ Rev 1.3's flat *"<0.5 ms"* removed | Final measurement on silicon |
| MAC Utilisation | Single-stream typical 5–20 %; aggregate multi-stream higher per `reports/benchmark_sweep/multistream_*.md` | Peak > 90 % aggregate; ⚠️ rev 1.3's flat *">90 % for CNN/ViT/LLM"* tightened to "peak >90%, single-stream 5–20% typical" | n/a |
| Memory Bandwidth | n/a — interface only | 400 GB/s LPDDR5X / 750 GB/s HBM3 (architecture-side AXI/CHI; bandwidth depends on PHY) | 🟪 **PHY: licensee-supplied** (Synopsys, Cadence, Rambus). AstraCore exposes AXI/CHI; bandwidth is licensee's PHY decision. |
| Functional Safety | SEooC v0.1; TMR + ECC + safe-state RTL; HARA v0.1 deriving item-level ASIL-D | ASIL-B safety case Q4 2026; ASIL-D extension follows. ⚠️ Rev 1.3's *"ASIL-D, certified"* reworded to "designed for ASIL-D as SEooC; cert path documented" | Item-level HARA; on-vehicle confirmation review |
| Operating Range | n/a | –40 °C to +125 °C, passive-cooling target | 🟪 Silicon + package + board characterisation by licensee |

---

## 3. Compute Architecture

| Component | Validated today | Tape-out target | Licensee-supplied |
|---|---|---|---|
| MAC Array | `npu_pe` 5-stage pipeline ✅; systolic array parameterised; default 4×4 / max-validated 8×8 via Verilator + cocotb | 24,576 MACs (48 cores × 512 MACs); intermediate AWS F1 64×64 = 4096 MACs validation Q3 2026 | 7 nm assembly + array clocking |
| Scalability | n/a (multi-die is silicon) | UCIe 1.1 chiplet target → ≥ 2000 TOPS clustering | 🟪 **UCIe IP: licensee-supplied**; AstraCore exposes AXI bridge |
| Precision — INT8 | Per-channel weight + per-tensor percentile-99.9999 activation PTQ; **98.4 / 96.0 / 91.2 % detection match vs FP32** on yolov8n @ IoU ≥ 0.5 / 0.7 / 0.9 (28-image COCO-128 eval); competitive with TensorRT / OpenVINO | Same recipe at scale | n/a |
| Precision — INT4 | Datapath plumbed; SNR 15.7 dB / cos 0.986 on yolov8n; QAT pipeline in flight | Sparse-INT4 throughput 2× via 8:1 sparsity engine (F1-A3) | n/a |
| Precision — INT2 | Plumbed + bit-exact vs Python mirror; representation wall without QAT noted | QAT lifts INT2 detection accuracy | n/a |
| Precision — FP8 (E4M3, E5M2) | Sim-gate PASS (5/5 cocotb); RTL fidelity vs Python mirror bit-exact for E5M2, < 1e-2 error for FP16 | 🟡 F1-A1.1 — synthesizable RTL wired into systolic array | n/a |
| Precision — FP16 | Sim-gate PASS; current systolic array falls back to INT8 on `precision_mode=11` per `npu_top.v` "current gaps" comment | 🟡 F1-A1.1 — wire-in to main array | n/a |
| Precision — FP4 / BF16 / TF32 / FP32 | Reference RTL exists in `rtl/npu_fp/`; not in main datapath | 🟡 F1-A1.2 / F1-A2 | n/a |
| Transformer AFUs | Softmax ✅, LayerNorm ✅, RMSNorm ✅, GeLU ✅ (LUT in `npu_activation`); bit-exact vs Python mirrors (10/10 cocotb) | 🟡 8× MHSA tile (F1-A5); rotary positional embedding RTL (today as IR op only) | n/a |
| Sparse Execution | 2:4 skip-gate ports plumbed on `npu_top` (`ext_sparse_skip_vec[N_ROWS]`); `test_sparse_skip_zeros_products` PASS | 🟡 F1-A3 — index decoder + 8:1 throughput + QAT pipeline | n/a |
| Sensor Fusion Accelerator | 48 RTL modules: 9 sensor I/O (camera/radar/lidar/IMU/US/GNSS/CAN-FD/Ethernet/PCIe), 3 fusion top, 5 perception primitives, 5 safety mechanisms, 3 vehicle-dynamics, 5 infrastructure, 3 top-level wrappers; 32/32 OpenLane sky130 ASIC batch PASS. ⚠️ Rev 1.3's *"4D point cloud processing"* reworded to "lidar point-cloud + temporal fusion via `coord_transform` + host-side tooling" | Item-level fusion algorithm extensions configurable via SDK plugin registries | n/a |

---

## 4. Memory System

| Component | Validated today | Tape-out target | Licensee-supplied |
|---|---|---|---|
| On-Chip SRAM | `rtl/npu_sram_bank/` parameterised primitive ✅; ECC SECDED RTL (`rtl/ecc_secded/`) ✅; **new ECC wrapper `rtl/npu_sram_bank_ecc/`** combinational SECDED preserves 1-cycle read latency (F4-A-1, 2026-04-20) | 128 MB total, 16 × 8 MB banks, ECC SECDED on all banks (npu_top instantiation swap to ECC wrapper = follow-up WP F4-A-1.1) | 🟪 **Foundry SRAM compiler** at licensee's target node; some compilers ship integrated ECC (use AstraCore wrapper as reference) |
| Scratchpad / DMA | `rtl/npu_dma/` prefetch-aware ✅; `rtl/npu_sram_ctrl/` ✅ | L0/L1 hierarchy + cache coherence post-tape-out | 🟪 Multi-level cache controllers if needed |
| Compression | n/a | 🟡 F1-A8 — 4-bit / 8-bit neural-aware encoding, 3–5× target. ⚠️ Rev 1.3 claim was unbacked; tracked as WP | n/a |
| External Memory PHY | n/a | n/a | 🟪 **LPDDR5X (400 GB/s) / HBM3 (750 GB/s) PHY: licensee-supplied** (Synopsys, Cadence, Rambus). AstraCore exposes AXI/CHI interface |

---

## 5. Connectivity & I/O

| Interface | Validated today | Tape-out target | Licensee-supplied |
|---|---|---|---|
| PCIe | `rtl/pcie_controller/` link-training scaffolding ✅ | Gen4 ×4 with DMA + P2P | 🟪 **Gen4 PHY: licensee-supplied** |
| MIPI CSI-2 | `rtl/mipi_csi2_rx/` packet-layer ✅; 4-lane D-PHY support in RTL | 4-lane D-PHY at production rates; C-PHY support via licensed IP | 🟪 **D-PHY / C-PHY: licensee-supplied**; 8K HDR ISP licensee-supplied |
| CAN-FD | `rtl/canfd_controller/` ✅ with DMA | 2× CAN-FD with DMA | 🟪 CAN transceiver |
| Ethernet | `rtl/ethernet_controller/` 1 Gbps scaffolding ✅ | 🟡 10 / 100 Gbps + AVB / TSN extensions | 🟪 **MAC PHY: licensee-supplied**; high-rate PHY licensed-IP |
| UCIe | n/a | n/a | 🟪 **UCIe IP: licensee-supplied**; AstraCore exposes AXI bridge if chiplet integration is in licensee's scope |
| ISP | n/a | n/a | 🟪 **ISP-Pro / 8K HDR / AI denoising: licensee-supplied or partner-supplied** (Sony, OmniVision, in-house ISP). Rev 1.3's ISP-Pro claim is removed from AstraCore IP scope |
| V2X | n/a | n/a | 🟪 **C-V2X modem: licensee-supplied** |
| Debug | JTAG / SPI / UART / I²C / GPIO scaffolds in fusion infrastructure | Standard debug pads | 🟪 Pads + level-shifters at licensee's PVT |

---

## 6. Security & Functional Safety

| Feature | Validated today | Tape-out target | Licensee-supplied |
|---|---|---|---|
| Secure Boot | Reference flow in SDK (`astracore.apply` + ASR-HW-12) | 🟡 F4-A (Phase B/C, 2026-04-20 plan): integrate **OpenTitan** AES-256 / RSA-2048 / SHA-256 / TRNG (Apache-2.0; production-grade per Google Titan) | 🟪 Hardware key storage at licensee's tape-out (eFuse / OTP) |
| NIST PQC (Kyber, Dilithium) | n/a | 🟡 W17+ or Series-B scope | 🟪 PQC algorithm acceleration if not in OpenTitan scope |
| Runtime Protection | `plausibility_checker` ✅; SEooC §2.3 boundary signals (`safe_state_active`, `fault_detected[15:0]`, ECC counters, TMR disagree counter) | AXI memory firewalls — 🟡 prerequisite WP for AXI substrate | 🟪 Bus arbitration policy at SoC level |
| Hardware TEE | n/a | 🟡 OpenTitan secure-enclave path | 🟪 Boot ROM authority |
| ASIL Compliance | **SEooC v0.1** designed for ASIL-D per `docs/safety/seooc_declaration_v0_1.md`. Safety mechanisms RTL-backed: TMR voter, SECDED ECC, safe-state controller, plausibility checker, fault predictor, watchdog. **HARA v0.1** derives item-level ASIL-D for FCW/AEB/LKA reference use case (`docs/safety/hara_v0_1.md` §5). **F4-A-5** shadow-register fix applied to dms_fusion (2026-04-20). Module-level FMEDA today: dms_fusion SPFM 85.5 % LFM 93.0 % PMHF 0.008 FIT; npu_top SPFM 2.08 % (drives the F4 RTL hardening backlog). | **ASIL-B safety case Q4 2026** with TÜV SÜD India pre-engagement; ASIL-D extension follows post-formal-verification + post-fault-injection campaigns | 🟪 Item-level safety case (vehicle program), confirmation review per ISO 26262-2 §7, on-silicon fault-injection campaign at licensee tape-out |
| ECC, watchdog, failover, clock monitors | ECC SECDED ✅; per-sensor watchdogs ✅; failover via safe-state controller ✅; clock monitor RTL is a tracked gap noted in SEooC §5 (Python `clock_monitor.py` exists; RTL counterpart pending) | RTL clock monitor module added per SEooC §9.2 open items | 🟪 Clock tree at licensee PVT |
| OTA Updates | n/a | 🟡 Reference flow lands with OpenTitan integration (W4-W7 per remediation plan) | 🟪 OTA orchestration at licensee firmware layer |

---

## 7. Power and Thermal Design

| Category | Validated today | Tape-out target | Licensee-supplied |
|---|---|---|---|
| Peak Power | n/a — RTL primitive only | 70–90 W with DVFS + MAC-level gating + sparsity-aware throttling | 🟪 PMIC + DVFS ladder by licensee |
| Typical Power | n/a | 40–50 W at 15–30 TOPS/W (INT8) projected | 🟪 Silicon + package measurement |
| Low-Power Mode | DMS fusion path RTL ✅ — natural always-on subset | 256 MACs @ 500 MHz always-on DMS, 1–5 W silicon-only mode | 🟪 Power-island floorplan |
| Thermal Management | `rtl/thermal_zone/` rule-based controller ✅; `rtl/fault_predictor/` rule-based ✅ | 🟡 F1-A9 — ML-based predictive upgrade. ⚠️ Rev 1.3's *"ML-based predictive fault detection"* reworded to "rule-based today; ML upgrade tracked as F1-A9" | 🟪 Thermal interface materials, package, cooling solution |
| ~~Energy Harvesting~~ | **Removed from AstraCore IP scope** (board-level feature) | n/a | 🟪 PMIC / harvester selection if used |

---

## 8. Software Stack (SDK — Apache-2.0)

| Component | Validated today | Tape-out target | Licensee-supplied |
|---|---|---|---|
| Compiler Frontends | ONNX 2.0 ✅ (233-node yolov8n loads cleanly); PyTorch / TVM / XLA / MLIR / NNEF via shared ONNX path ✅. ⚠️ TensorRT removed (it's a runtime, not a frontend) | Per-frontend optimisation | n/a |
| Quantiser | INT8 production-grade ✅ (per-channel + percentile-99.9999); INT4 ✅; production recipe documented | 🟡 FP4 / FP8 (F1-A1.x); 8:1 sparsity (F1-A3); QAT pipeline (Phase C) | n/a |
| Auto-tiling / Compiler | Im2col conv2d → matmul ✅; K-chunking + N-splitting ✅; chained partial sums ✅; SiLU fusion pass ✅ | 🟡 ML-tunable scheduling on roadmap | n/a |
| Runtime | Python end-to-end ✅ (`tools/npu_ref/nn_runtime.py`); C++ runtime stub planned (F1-B3) | 🟡 C++ runtime parity with Python; CI gates against `tools/safety/regress_check.py` baseline | 🟪 Driver shim at licensee SoC level |
| Backends (Execution Providers) | Built-in: `npu-sim` ✅, `ort` ✅; plug-in registry pattern (`astracore.backends` entry-point) ✅ — licensees register custom-silicon backends without forking | Additional first-party backends as licensee silicon ships | 🟪 Licensee-specific backends (TensorRT, SNPE, target NPU) |
| Plugin Registries | `astracore.register_op`, `astracore.register_quantiser`, `astracore.register_backend` decorators + setuptools entry-points ✅ | More extension points (custom calibration strategies, custom schedulers) | n/a |
| Telemetry / Profiler | `astracore bench`, `astracore multistream`, `astracore configure --apply` ✅ produce reproducible reports | 🟡 Per-op cycle-count profiler (F1 follow-up) | 🟪 SoC-level telemetry pipeline |
| ~~Cloud Platform~~ | **Removed from AstraCore IP scope** (services offering, not a chip feature) | n/a | n/a |

---

## 9. Model & Application Library

Curated model zoo at `astracore zoo` covers eight ONNX models today;
licensees extend via the `astracore.zoo` entry-point.

| Workload | Validated today | Roadmap |
|---|---|---|
| Vision detection — YOLOv8-N | End-to-end INT8 PTQ + AWS F1 bring-up gate met (4/5 detections @ IoU≥0.7 on bus.jpg); production recipe yields 98.4/96.0/91.2 % detection match | YOLOv8-S / -M / -L same recipe |
| Vision classification — ResNet50, MobileNetV2, ShuffleNetV2, SqueezeNet, EfficientNet-Lite4 | Bench + multi-stream measured on host CPU + ORT backend; reports under `reports/benchmark_sweep/` | Same suite on AWS F1 + silicon backend |
| Transformers — BERT, GPT-2 / LLaMA-family | Ops in loader ✅ (GELU / LayerNorm / RMSNorm / RotaryEmbedding / MHA); GPT-2 bench measured | 🟡 8× MHSA tile (F1-A5); transformer perf at scale |
| Sensor fusion — DMS (driver attention) | `rtl/dms_fusion/` ✅ + Python mirror; F4-A-5 shadow-register fix applied; FMEDA SPFM 85.5 % LFM 93.0 % | F4 remediation lifts to ASIL-B by W12 |
| Sensor fusion — Lane | `rtl/lane_fusion/` ✅ | FMEDA pending W6 per `docs/safety/fmeda/README.md` |
| Sensor fusion — Custom (US + lidar + CAN proximity alarm) | `examples/ultrasonic_proximity_alarm.py` ✅ — 4-band alarm with speed-scaled thresholds + cross-sensor confirmation | Reference for licensee-authored fusion engines |
| Tier-1 ADAS rig | `examples/tier1_adas.yaml` ✅ — 4 cameras + 1 lidar + 6 radars + 12 ultrasonics + 1 thermal + 1 event + 1 ToF + 2 CAN + GNSS + IMU + 3 models + safety policies, drives `astracore configure --apply` end-to-end | Reference for licensee SoC integration |
| ~~Generative AI — Stable Diffusion~~ | **Moved to V2 roadmap** (memory footprint exceeds current DMA tiler) | 🟡 Post-Phase B compiler work |
| ~~On-Chip Training~~ | **Removed from AstraCore IP scope** (no backward-pass RTL; this is an inference accelerator) | 🟡 Future post-tape-out program if licensee demand materialises |

---

## 10. Validation Snapshot (reproducible numbers)

The numbers any external reviewer can verify on a fresh checkout:

| What | Today's number | Command |
|---|---|---|
| Python suite | **1352 pass, 0 fail** (verified twice 2026-04-20, ~10 min runtime) | `pytest --ignore=tests/test_pytorch_frontend.py` |
| Safety tooling tests (subset of above) | **206 pass** | `pytest tests/test_fmeda.py tests/test_fault_injection_planner.py tests/test_ecc_ref.py tests/test_regress_check.py` |
| RTL cocotb (4×4 NPU top) | 6 / 6 PASS | WSL: `bash tools/run_verilator_npu_top.sh` |
| RTL cocotb (8×8 NPU top) | 2 / 2 PASS | WSL: `bash tools/run_verilator_npu_top_8x8.sh` |
| F1-A4 softmax + LayerNorm / RMSNorm RTL | 10 / 10 PASS bit-exact vs Python mirror; SNR 30.93 / 58.14 dB vs FP32 | WSL: `bash tools/run_verilator_npu_softmax.sh`; `bash tools/run_verilator_npu_layernorm.sh` |
| F1-A1 FP MAC sim-gate | 5 / 5 PASS; bit-exact E5M2; < 1e-2 FP16 error | WSL: `bash tools/run_verilator_npu_fp_mac.sh` |
| ASIC PPA | 32 / 32 modules close on sky130 130 nm via OpenLane | `asic/runs/` |
| YOLOv8-N detection eval | 98.4 / 96.0 / 91.2 % match @ IoU ≥ 0.5 / 0.7 / 0.9 (28-image COCO-128) | `reports/yolov8n_eval.json` |
| Multi-stream scaling | 1.48× (yolov8n) / 1.57× (mobilenetv2) / 1.90× (shufflenet) on 8 streams vs 1 | `reports/benchmark_sweep/multistream_*.md` |
| FMEDA — `dms_fusion` | SPFM 85.52 % / LFM 92.99 % / PMHF 0.008 FIT @ ASIL-D placeholder rates | `python -m tools.safety.fmeda --module dms_fusion --asil ASIL-D` |
| FMEDA — `npu_top` | SPFM 2.08 % / LFM 0 % / PMHF 0.53 FIT @ ASIL-B placeholder rates — drives the F4 RTL hardening backlog (Phase A → ~75 %, Phase B → ~92 %) | `python -m tools.safety.fmeda --module npu_top --asil ASIL-B` |
| FMEDA — safety primitives | `tmr_voter` SPFM 31.35 %, `ecc_secded` SPFM 1.49 %, `safe_state_controller` SPFM 34.80 %, `plausibility_checker` SPFM 27.20 %. These are *safety mechanisms* with no internal protection; module-level closure via F4-D-1/D-2 formal proofs + F4-A-7 (TMR on `safe_state` FSM) | `python -m tools.safety.fmeda --module <name> --asil ASIL-B` |
| FMEDA regression gate | Baseline committed (6 modules); CI fails on SPFM/LFM drop > 1 pp or PMHF rise > 0.001 FIT | `python -m tools.safety.regress_check` |

---

## 11. Roadmap (open work packages)

| WP | Scope | Estimated close |
|---|---|---|
| F1-A1.1 | Synthesizable FP8 / FP16 datapath wired into systolic array (sim-gate already 5/5 PASS) | Q3 2026 |
| F1-A2 | BF16 / TF32 / FP32 (blocked on F1-A1.1) | Q4 2026 |
| F1-A3 | 8:1 structured sparsity engine + QAT pipeline | Q4 2026 |
| F1-A5 | 8× MHSA tile for transformers | Q1 2027 |
| F1-A8 | Weight compression 3–5× | Q1 2027 |
| F1-A9 | ML-based thermal / fault predictor (currently rule-based) | Q1 2027 |
| F1-B3 | C++ runtime parity with Python | Q4 2026 |
| F1-B6 | Broader model library validation (BEVFormer, BERT-Large, Llama-3) | Q1 2027 |
| F1-F1..F3 | AWS F1 VU9P bring-up, 64×64 = 4096 MACs (16 % of full-spec) — first measured silicon-scale latency | Q3 2026 |
| F4-A-1.1 | npu_top instantiation swap to ECC wrapper (RTL `npu_sram_bank_ecc/` shipped 2026-04-20; integration follow-up) | W4 (per remediation plan) |
| F4-A-2..6 | PE weight / dataflow / config / precision parity | W3-W4 |
| F4-B-1..6 | PE accumulator parity + duplicated tile_ctrl FSM + LBIST + MBIST + busy/done TMR + drain parity | W5-W8 |
| F4-C-1..4 | Fault-injection campaigns (tmr_voter / ecc_secded / dms_fusion / safe_state_controller) | W7-W10 |
| F4-D-1..6 | ASIL-D extension: formal proofs, ECC upgrade, CCF analysis, SER analysis, interleaved-Hamming layout | W13-W18 |

Full execution detail in `docs/safety/findings_remediation_plan_v0_1.md` and `docs/best_in_class_design.md` §7.2.

---

## 12. What is NOT in this datasheet (scoping clarity)

To prevent ambiguity that has historically tripped IP licensing conversations, here is what **AstraCore Neo IP does not deliver** and the licensee or partner is responsible for:

1. **Memory PHY** (LPDDR5X, HBM3, DDR4, GDDR) — license from Synopsys / Cadence / Rambus
2. **High-speed serdes** (PCIe Gen4 PHY, MIPI C-PHY, 10/100 G Ethernet PHY) — license from PHY vendors at target node
3. **UCIe chiplet PHY + package** — license + assembly
4. **Image Signal Processor** (ISP-Pro, 8K HDR, AI denoising) — partner with Sony, OmniVision, or in-house
5. **Cellular / V2X modem** — partner (Qualcomm, Autotalks, MediaTek)
6. **Power Management IC** (PMIC), DVFS controllers — board-level
7. **Thermal solution** — package + cooling
8. **Foundry SRAM compiler** — at licensee's target node (use `npu_sram_bank_ecc` wrapper as ECC integration reference)
9. **Final 7 nm tape-out, packaging, validation silicon** — licensee program
10. **Item-level HARA** on the licensee's vehicle program — AstraCore HARA is *assumed* per SEooC §3
11. **ISO 26262 confirmation review on the integrated item** — independent reviewer per ISO 26262-2 §7
12. **Vehicle-level safety case** — vehicle OEM
13. **Cloud / services platform, OTA orchestration, fleet telemetry** — licensee firmware + cloud team
14. **On-chip training / fine-tuning** — out of scope; AstraCore Neo is an **inference** accelerator
15. **Energy-harvesting subsystem** — board-level if used
16. **Stable Diffusion-class generative models** — out of current memory-tiling scope; V2 program

This list is intentionally explicit. Standard IP-licensing practice (Synopsys ARC NPX, Arm Ethos, Imagination NN-A, Cadence Tensilica) follows the same boundary; over-promising scope at the IP level destroys licensee trust on first integration audit.

---

## 13. Document control

| Field | Value |
|---|---|
| Document ID | ASTR-DATASHEET-V1.4 |
| Revision | 1.4 |
| Revision date | 2026-04-20 |
| Author | TBD (Track 2 + product) — currently founder + collaborator |
| Reviewer | TBD (independent reviewer per ISO 26262-2 §7) |
| Approver | TBD (Safety Manager + founder) |
| Distribution | Internal + first NDA evaluation licensee; external publication after spec sheet revision-control sign-off |
| Retention | 15 years post-product-discontinuation per ISO 26262-8 §10 |
| Supersedes | Rev 1.3 (10.07.2025); Rev 1.4 DRAFT (`docs/spec_sheet_rev_1_4_draft.md`) — retained for revision history |

### 13.1 Revision triggers

This datasheet is re-issued (rev 1.5+) on any of:

1. New FMEDA pass that materially changes module-level SPFM / LFM / PMHF (>1 pp or >0.001 FIT)
2. F4-A / F4-B / F4-C / F4-D phase milestone closes
3. AWS F1 measurement produces silicon-scale latency / throughput / bandwidth numbers — replaces "tape-out target" projections with measured intermediate values
4. HARA revision driven by new use case or accident-statistics review
5. SEooC declaration revision (per `docs/safety/seooc_declaration_v0_1.md` §9.1)
6. New licensee-supplied scope item identified during NDA evaluation (e.g., licensee asks AstraCore to absorb a previously licensee-scoped item, or vice versa)
7. TÜV SÜD pre-engagement workshop output that changes ASIL-B / ASIL-D timeline
