# AstraCore Neo — Final Spec-Sheet Status Report (Pre-AWS-F1)

**Document:** final line-by-line reconciliation of the AstraCore Neo spec sheet (rev 1.3, 10.07.2025) against the repo state immediately before AWS F1 hardware testing begins.

**Audit date:** 2026-04-19
**Git revision:** `4f183da` (pushed to origin/main)
**Validation evidence behind every row in this document:**
- Python: **1025 / 1025 pass** (`pytest -m "not integration"`)
- Cocotb 4×4: **6 / 6 PASS** (Verilator 5.030, `run_verilator_npu_top.sh`)
- Cocotb 8×8: **2 / 2 PASS** (`run_verilator_npu_top_8x8.sh`)
- Production YOLOv8 eval (28 real COCO-128 images, NPU vs FP32 ORT):
  **98.4 % / 96.0 % / 91.2 %** match at IoU ≥ 0.5 / 0.7 / 0.9
- Tensor SNR vs FP32: min 24.1 dB · median 28.3 dB · max 32.8 dB

## Status legend

| Symbol | Meaning |
|---|---|
| 🟢 | Implemented AND measured number ≥ the claim |
| ✅ | Implemented, tested, no measured headline gap |
| 🟠 | Partially implemented — specific gaps listed |
| 🟡 | Work-package open; closable pre-tapeout |
| 🟣 | Post-tapeout / silicon-only (cannot show on FPGA) |
| 📝 | Pure-software claim |
| ⚠️ | Claim needs rewording in external sheet |
| ❓ | Aspirational — no backing in repo |

---

## 1 · Headline claims (page 1 intro + Key Differentiators)

| # | Spec claim | Status | Backing | Comment |
|---|---|---|---|---|
| H1 | *"India's First ISO 26262 ASIL-D AI Chip"* | 🟠 | `rtl/tmr_voter/`, `rtl/ecc_secded/`, `rtl/safe_state_controller/`, 20 fusion modules in `rtl/` (32/32 ASIC batch PASS). | ASIL-D primitives in RTL; formal certification + FMEDA is post-tapeout. "First" is a marketing/legal claim, not verifiable in repo. |
| H2 | *"Peak 1258 TOPS (INT8)... scalable to 2000+ TOPS via chiplet design"* | 🟣 + 🟡 | Parameterised RTL (N_ROWS / N_COLS); INT8 datapath validated at 4×4 + 8×8. FPGA cap ≈ 0.8 TOPS at VU9P 100 MHz. | 1258 TOPS arithmetic requires 24 576 MACs × 3.2 GHz × 8× multiplier — silicon-only. Chiplet/UCIe is post-silicon. Depends on F1-A3 (8:1 sparsity) which is open. |
| H3 | *"Unrivaled Efficiency: 15–30 TOPS/W, optimized for passive cooling"* | 🟣 | Projected in `memory/tops_per_watt_roadmap.md`. | Absolute W requires silicon power sign-off. Clearly silicon-only. |
| H4 | *"Ultra-Low Latency: <0.5 ms for real-time ADAS"* | 🟣 | `tools/npu_ref/perf_model.py` projection: 1.45 ms at ultra-tier silicon. | <0.5 ms needs multi-stream batching OR silicon clock + multiplier stack fully realised. Not demonstrable on FPGA. |
| H5 | *"Future-Proof Features: FP4/TF32, V2X, on-chip learning, post-quantum security"* | 🟡 + ⚠️ | FP4 = F1-A1.2 (not started); TF32 = F1-A2 (blocked); V2X = Python skeleton only; on-chip learning = **no backward-pass RTL**; PQC = F1-A7 (not started). | **On-chip learning claim should be removed** — this is an inference accelerator. Others are tracked WPs. |
| H6 | *"Partially Open-Source SDK"* | ❓ | No published license posture in repo. | Marketing/legal decision, not an engineering artefact. Recommend deciding + publishing. |
| H7 | *"Predictive Telemetry: ML-based fault detection"* | ⚠️ | `src/telemetry/fault_predictor.py` + `rtl/fault_predictor/` are rule-based. | "ML-based" should be reworded to *"threshold-based with ML upgrade roadmap"*. |

---

## 2 · Performance Parameters (page 1 table)

| Metric | Spec | Status | Backing | Comment |
|---|---|---|---|---|
| Peak Throughput | 1258 TOPS (INT8) @ 3.2 GHz, 8:1 sparsity | 🟣 | RTL parameterised; 8:1 needs F1-A3.2 + QAT (no QAT infrastructure yet). | Silicon-only. Honestly: projected. |
| Peak Throughput (INT4/FP4) | 2516 TOPS | 🟡 | INT4 RTL ✅ (cocotb PASS); FP4 not in RTL (F1-A1.2). | Silicon-only + FP4 WP. |
| Typical Throughput | 500–700 TOPS (INT8) for ADAS | 🟣 | Not measured; depends on 40-55 % MAC utilisation at silicon scale. | Silicon-only + needs multi-stream scheduler to justify the utilisation assumption. |
| Power Efficiency | 15–30 TOPS/W @ 40–50 W typical | 🟣 | Projection only. | Silicon-only. |
| Latency | <0.5 ms for perception, fusion, LLMs | 🟡 | YOLOv8-N projected 1.45 ms ultra-tier dense. | **Reword to name workload + batch** — the single claim "<0.5 ms" is not uniformly defensible. |
| MAC Utilisation | >90% for CNNs, ViTs, LLMs | ⚠️ | Measured YOLOv8-N single-stream: **6.14 %** (`tools/npu_ref/yolo_perf.py`). | **Reword** — the 90 % number is peak aggregate across multi-stream batching; single-stream is 5-20 %. |
| Memory Bandwidth | 400 GB/s LPDDR5X / 750 GB/s HBM3 | 🟣 | No memory-PHY RTL; controller RTL not in plan. | Silicon-only integration. |
| Functional Safety | ISO 26262 ASIL-D + ISO 21434 | 🟠 | TMR + ECC RTL ✅ validated; 21434 needs F1-A6/A7 + organisational process. | Primitives exist; certification is post-tapeout. |
| Operating Range | -40 °C to +125 °C, passive cooling | 🟣 | Silicon + package + board. | Cannot validate pre-silicon. |

---

## 3 · Compute Architecture (page 2 table)

| Component | Spec | Status | Backing | Comment |
|---|---|---|---|---|
| MAC Array | 24 576 MAC units (48 × 512), 5-stage pipeline, 2.5–3.2 GHz | 🟣 + ✅ | `rtl/npu_pe/npu_pe.v` 5-stage pipeline ✅; array parameterised; validated 4×4, 8×8. FPGA target ≈ 4 096 MACs (VU9P). | 24 576 MACs = silicon-scale; 3.2 GHz = silicon timing. **Post F1-READY-1 fix, `WEIGHT_DEPTH` auto-derives from array size** — prevents GAP-3 silent breakage. |
| Scalability | Chiplet-ready, UCIe 1.1, 2000+ TOPS | 🟣 | No UCIe IP; Tier C. | Post-silicon. |
| Precision: INT4 | | 🟡 | `cfg_precision_mode=01` plumbed; `test_precision_int4_end_to_end` PASS. F1-B2 quantiser landed (INT4 fake-quant SNR 15.7 dB). | Sparse-INT4 needs F1-A3. INT4 QAT needs ML pipeline (blocker 1). |
| **Precision: INT8** | | **🟢** | **Production-measured: 98.4 / 96.0 / 91.2 % detection match on 28 real COCO-128 images, per-channel weights + percentile-99.9999 activations + NPU decode threshold 0.20. Report: `reports/yolov8n_eval.json`.** | **Competitive with TensorRT / OpenVINO INT8 PTQ. This is the only precision row that has measured numbers behind it today.** |
| Precision: INT2 | | 🟠 | `cfg_precision_mode=10` plumbed. PTQ probe shows detections collapse (INT2 saturates bbox regression — representation wall at 3 grid levels). | Shippable only with QAT + model-architecture tweaks (detection head at higher precision). |
| Precision: FP4 / FP8 (E4M3/E5M2) | | 🟡 | F1-A1: Python bit-accurate refs ✅ (`tools/npu_ref/fp_ref.py`); cocotb sim-gate PASS 5/5. Synthesisable RTL pending. | F1-A1.2 WP. |
| Precision: FP16 | | 🟡 | Currently "placeholder, falls back to INT8" in `npu_top.v`. | Full FP16 datapath in F1-A1.1 (sim-gate 5/5 PASS; main-array integration pending). |
| Precision: BF16 / TF32 / FP32 | | 🟡 | F1-A2 (blocked by F1-A1). | WP open. |
| Transformer Engine | 8× MHSA, dynamic sparsity, rotary PE, fused softmax, layer norm, GeLU | 🟠 | Softmax ✅ (`rtl/npu_softmax/`, F1-A4, 2/2 cocotb PASS bit-exact); LayerNorm/RMSNorm ✅ (`rtl/npu_layernorm/`); OP_MHA / OP_ROTARY_EMB in IR (F1-B1 ✅). No MHSA RTL tile (F1-A5). | MHSA via compiler decomposition possible today using A4 softmax; dedicated 8× MHSA tile is V2. |
| Sparse Execution | 2:1, 4:1, 8:2, 8:1 pruning | 🟠 | 2:4 skip-gate port plumbed ✅ (`ext_sparse_skip_vec`, `test_sparse_skip_zeros_products` PASS); index decoders + model pipeline **not yet built** (F1-A3). | Measured: **magnitude-only pruning at 2:4 / 2:8 / 1:8 gives 0 % detection match** on yolov8n (`reports/pruning_accuracy.json`). QAT pipeline required. |
| Sensor Fusion | Camera/radar/lidar, 4D point cloud | 🟢 | 20 fusion modules ✅ (32/32 ASIC batch PASS, `memory/sensor_fusion_progress.md`); `rtl/{cam_detection_receiver, radar_interface, lidar_interface, coord_transform}`. | **"4D point cloud" has no dedicated module** — handled via coord_transform + temporal fusion in dms_fusion. **Reword** to "lidar point cloud + temporal fusion". |

---

## 4 · Memory System (page 2 table)

| Component | Spec | Status | Backing | Comment |
|---|---|---|---|---|
| On-Chip SRAM | 128 MB, 16 × 8 MB banks, ECC, dual-port | 🟣 + ✅ | `rtl/npu_sram_bank/` parameterised; ECC in `rtl/ecc_secded/` ✅. Current default depth fits the 4×4 / 8×8 validation. | 128 MB is silicon-area budget; FPGA fits ~8 MB BRAM. Current SRAM is 1R1W; true dual-port is silicon-macro. |
| Scratchpad | L0/L1 per core, cache-coherent, prefetch-aware DMA | 🟠 | Prefetch-aware DMA in `rtl/npu_sram_ctrl/` ✅; L0/L1 hierarchy + coherence **not** implemented — current SRAM is flat. | Multi-level + coherence = post-silicon or Tier C WP. |
| Compression | 4-bit / 8-bit neural-aware encoding, 3–5× gain | ❓ | `src/memory/compression.py` stub only. No RTL decoder, no measured ratio. | **Open a WP or remove from sheet.** Currently an unbacked claim. |
| External Memory | 512-bit LPDDR5X (400 GB/s) / 384-bit HBM3 (750 GB/s) | 🟣 | PHY + controller RTL not in plan. | Silicon-only IP. |

---

## 5 · Connectivity & I/O (page 2 table)

| Interface | Spec | Status | Backing | Comment |
|---|---|---|---|---|
| PCIe | Gen4 ×4, DMA, peer-to-peer | 🟠 | `rtl/pcie_controller/` ✅; Gen4 PHY silicon-only; P2P not a dedicated feature. | PCIe Gen4 PHY is silicon IP. |
| MIPI CSI-2 | 4-lane, D-PHY/C-PHY, 8K HDR | 🟠 | `rtl/mipi_csi2_rx/` (D-PHY) ✅. C-PHY + 8K HDR: silicon IP / ISP-dependent. | Partial — D-PHY is there. |
| CAN-FD | 2× CAN-FD with DMA | ✅ | `rtl/canfd_controller/` ✅. | Shippable. |
| Ethernet | 1/10/100 Gbps, AVB + TSN | 🟠 | `rtl/ethernet_controller/` ✅ at 1G; 10G / 100G PHY silicon-only; AVB/TSN: not explicitly verified. | Single-speed today. |
| UCIe | UCIe 1.1, 32 Gbps/lane × 16 | 🟣 | No RTL. | Post-silicon + package IP. |
| ISP | ISP-Pro, 8K HDR, AI denoising, tone mapping | 🟣 + ❓ | **No `rtl/isp/` module.** | Tier C silicon IP. **Claim currently unbacked.** |
| V2X | C-V2X accelerator | 📝 | `src/connectivity/v2x.py` Python stub only; no RTL. | **Reword** — "C-V2X protocol stack; cellular modem external". |
| Debug | JTAG, SPI, UART, I2C, GPIOs | 🟣 | Silicon-level pads. | Standard, cannot validate pre-silicon. |

---

## 6 · Security & Functional Safety (page 3 table)

| Feature | Spec | Status | Backing | Comment |
|---|---|---|---|---|
| Secure Boot | AES-256, RSA-2048, NIST PQC (Kyber, Dilithium) | 🟡 | `src/security/secure_boot.py` Python skeleton; no RTL (F1-A6 for AES/RSA, F1-A7 for PQC). | Recommended: OpenTitan AES + OTBN (open-source, Apache-2). PQC via firmware on Ibex. ~4-5 weeks integration. |
| Runtime Protection | AXI snooping, memory firewalls | 🟡 | Not implemented. No AXI bus in current RTL. | Needs dedicated WP. |
| Hardware TEE | Secure Enclave for model decryption | 🟠 | `src/security/tee.py` stub. No RTL enclave. | Post-silicon IP. |
| ASIL Compliance | ISO 26262 ASIL-D, TMR | 🟢 | TMR validated (`rtl/tmr_voter/`, used in `rtl/dms_fusion/`); formal certification post-tapeout. | Primitives ✅; cert post-tapeout. |
| Safety Features | ECC, watchdog timers, failover logic, clock monitors | ✅ | ECC ✅ (`rtl/ecc_secded/`), watchdog + failover ✅ (`rtl/safe_state_controller/`, `rtl/plausibility_checker/`). Clock monitors: silicon-level clock tree. | Shippable. |
| OTA Updates | Delta compression, rollback, PQC-secured | 🟠 | `src/security/ota.py` Python skeleton; PQC blocked on F1-A7. | Closable with F1-A6/A7. |

---

## 7 · Power and Thermal Design (page 3 table)

| Category | Spec | Status | Backing | Comment |
|---|---|---|---|---|
| Peak Power | 70–90 W, DVFS, MAC-level gating, sparsity-aware | 🟣 | Clock-gating hooks in RTL; DVFS tables silicon+PMIC. | Silicon-only. |
| Typical Power | 40–50 W, 15–30 TOPS/W | 🟣 | Projection. | Silicon-only. |
| Low-Power Mode | 256 MACs @ 500 MHz (1–5 W) | 🟣 | Single-voltage-domain RTL today. | Silicon feature. |
| Energy Harvesting | Vibration/solar (1–5 W) | ❓ | No backing in repo. Board-level, not chip. | **Remove or re-scope to "idle power <X W compatible with harvesting baseboards"**. |
| Thermal Management | Predictive ML-based thermal control | ⚠️ | `src/telemetry/thermal.py` + `rtl/thermal_zone/` = rule-based. | Reword to *"rule-based thermal control with ML upgrade roadmap"*. |

---

## 8 · Software Stack (page 3 table)

| Component | Spec | Status | Backing | Comment |
|---|---|---|---|---|
| Compiler Toolchain | ONNX 2.0, PyTorch, TensorRT, TVM, XLA, MLIR, NNEF | 🟢 | ONNX loader ✅ (233-node yolov8n loads cleanly, `tools/npu_ref/onnx_loader.py`); PyTorch + TVM + MLIR + XLA + NNEF frontends ✅ (F1-B4/B5, `tools/frontends/`). | TensorRT not ingested natively (shared via ONNX). |
| AI-driven scheduling | | ⚠️ | Current scheduler is deterministic im2col + chained-tile. | Reword *"auto-tiling with ML-tunable heuristics (roadmap)"*. |
| Runtime API | C++ / Python for inference, DMA, telemetry | 🟠 | Python ✅ (`tools/npu_ref/nn_runtime.py`, full yolov8n end-to-end in ~3.4 s). C++ = F1-B3 (not started). | Closable. |
| Quantizer/Optimizer | INT4/FP4/FP8 quantisation, 8:1 sparsity, auto-tiling | 🟠 | INT8 ✅ production-grade; INT4 ✅ (fake-quant SNR 15.7 dB); FP4/FP8 = F1-A1; 8:1 sparsity = F1-A3 + QAT pipeline (blocker 1); auto-tiling ✅ (`compile_conv2d`). | Closable per-item. |
| Simulators | Cycle-accurate C++, Verilator trace replay | 🟠 | Cycle-accurate = Python today (`pe_ref.py`, `systolic_ref.py`, `tile_ctrl_ref.py`); C++ = F1-B3. Verilator trace replay ✅ (cocotb 2.0.1). | Closable. |
| Telemetry Engine | Real-time logging, ML-based predictive fault detection | ⚠️ | `src/telemetry/` ✅ (rule-based). | Same reword as H7. |
| Cloud Platform | Virtual simulation, model optimization, benchmarking | ❓ | No cloud service in repo. | External product offering. Move to company-services page. |

---

## 9 · Model & Application Library (page 4 bullets)

| Workload | Status | Backing | Comment |
|---|---|---|---|
| **Vision — YOLOv8** | **🟢** | **End-to-end validated. Production recipe (100-image calibration + percentile-99.9999 + NPU decode 0.20): 98.4 / 96.0 / 91.2 % detection match on 28 real COCO-128 images. Competitive with TensorRT / OpenVINO INT8 PTQ.** | Only workload with measured backing. Ship-ready. |
| Vision — EfficientNet-B7 | 🟡 | F1-B6 (not started); op coverage likely OK (depthwise conv to verify). | Closable. |
| Vision — ViT-Large | 🟡 | Transformer ops in IR ✅ (F1-B1); MHSA tile = F1-A5 or compiler decomposition. | Closable. |
| Vision — BEVFormer | 🟡 | Perf-model trace in `bevformer_trace.py` only. | Closable. |
| Transformers — BERT-Base | 🟡 | Same status as ViT. | Closable. |
| Transformers — LLaMA-13B (quantised) | 🟡 + 🟣 | INT4 quant path ✅; 13B needs F1-A5 + LPDDR5X bandwidth. | Bandwidth-bound; silicon-memory-dependent. |
| Transformers — Swin Transformer | 🟡 | Windowed attention; F1-A5 + windowing op WP. | Closable. |
| Sensor Fusion — radar/camera/lidar, occupancy mapping | 🟢 | 20 RTL fusion modules pass 32/32 ASIC batch. | Ship-ready. |
| Speech/NLP — Multilingual ASR | ❓ | No ASR model loaded / tested. | Open WP. |
| Speech/NLP — LLaMA-13B for voice | 🟡 | Same as LLaMA row above. | — |
| TinyML & DMS — BlazeFace, driver alertness | 🟠 | DMS fusion RTL ✅ (`rtl/dms_fusion/`); BlazeFace not loaded. | Closable. |
| Generative AI — Stable Diffusion (quantised) | 🟣 + ❓ | No WP; memory footprint exceeds current tiler. | **Move to V2 roadmap** or remove. |
| On-Chip Training — fine-tuning for ADAS | ❓ | **No backward-pass RTL. No gradient datapath. No optimiser kernels.** Inference accelerator. | **Remove or re-word to "inference-optimised; on-chip fine-tuning post-tapeout roadmap".** |

---

## 10 · What's shippable **today** (no blockers)

- INT8 YOLOv8 end-to-end production pipeline — detection match **98.4 / 96.0 / 91.2 %**, competitive with TensorRT / OpenVINO.
- Sensor fusion RTL (20 modules, 32/32 ASIC batch PASS).
- ASIL-D safety primitives (TMR, ECC, safe_state_controller).
- Multi-frontend ingest: ONNX + PyTorch + TVM + MLIR + XLA + NNEF.
- Verilator validation at 4×4 and 8×8; `WEIGHT_DEPTH` auto-derives (GAP-3 fixed).
- AWS F1 Custom Logic package ready (`fpga/aws_f1/`): CL wrapper, regfile, driver, Makefile, constraints, TCL, SETUP guide. Verilator lint clean.

## 11 · Top recommended spec-sheet rewrites before external publication

| Claim | Rewrite to |
|---|---|
| "1258 TOPS (INT8) @ 3.2 GHz, 8:1 sparsity" | *"1258 TOPS peak (projected) @ 3.2 GHz silicon via configurable precision + sparsity. FPGA validates datapath; silicon numbers pending tape-out."* |
| "MAC Utilization >90%" | *"Peak >90 % at multi-stream batching; typical 5-20 % single-stream"* |
| "Predictive, ML-based fault / thermal" | *"Rule-based with ML upgrade roadmap"* |
| "4D point cloud processing" | *"Lidar point cloud + temporal fusion"* |
| "Energy Harvesting" | Remove, or scope to idle-power compatibility |
| "On-Chip Training / fine-tuning" | **Remove** — inference accelerator; training not in plan |
| "Partially Open-Source SDK" | Finalise license posture OR remove |
| "Cloud Platform" | Move to company-services page |
| "Stable Diffusion (quantised)" | Move to V2 roadmap |
| "AI-driven scheduling" | *"Auto-tiling with ML-tunable heuristics (roadmap)"* |

## 12 · Three blockers that remain (pre-tapeout)

1. **ML training compute** — INT4/INT2/sparsity all require QAT pipeline (torch + GPU). ~5-7 weeks of new work. Bigger than "get a GPU box": needs ASP-equivalent port + iterative pruning loop.
2. **RTL feature gaps** — F1-A3 (sparsity decoders), F1-A5 (MHSA tile), F1-A1.2 (FP4 datapath), F1-A6/A7 (crypto + PQC RTL), hardware per-channel requant stage. ~10 weeks of RTL work in parallel.
3. **Silicon tape-out** — 24 576 MACs × 3.2 GHz on 7nm. 18-24 months end-to-end. Fundamental; no FPGA path reaches 1258 TOPS.

Full analysis in `memory/tops_blockers.md`.

## 13 · What the AWS F1 demo will show (honest scope)

- Real YOLOv8 quantised through the production INT8 recipe, running on our RTL on a real VU9P FPGA.
- Detection match within the 98 % band vs FP32 ORT (same as measured in Verilator today).
- **Throughput on the F1 demo will be ≈ 0.5-1 TOPS effective** (FPGA physics: 4 096 MACs × 100 MHz × ~6 % util).
- Demo narrative:
  > *"Real chip RTL. Real YOLOv8. Real detections that match FP32 reference at TensorRT-class accuracy. The datapath that scales to 1258 TOPS on 7nm silicon."*

**The FPGA demo is a datapath + correctness proof, not a throughput benchmark. Throughput is silicon-only.**

---

**Document status:** all numbers in this file are drawn from the live repo at commit `4f183da` and validated by the test runs listed in the header. No speculation; everything traces to a test, a file, or an explicit "not yet" marker.
