# AstraCore Neo — Spec Sheet Provenance & Post-Tapeout Caveats

**Purpose.** The AstraCore Neo spec sheet (rev 1.3, 10.07.2025) is the
product-level claim surface for the A2 Neo chip. This document is its
engineering-provenance companion: for each headline claim, it records
whether the capability is (a) RTL-backed today, (b) being built out as
a tracked work package, or (c) a post-tapeout / silicon-only claim that
cannot be demonstrated on FPGA.

Every public or investor-facing version of the spec sheet should either
inline this provenance or carry an explicit "post-tapeout" caveat
beside the items flagged below. Internal and marketing versions should
not diverge on the caveat text.

---

## Provenance legend

- **✅ RTL-backed** — Implemented, tested, runs today on Verilator +
  cocotb. Observable in the repo.
- **🟡 WP-tracked** — Work package is open in the task list; completing
  the WP makes the claim RTL-backed.
- **🟣 Post-tapeout** — Cannot be demonstrated on FPGA or bench silicon
  alone; depends on tape-out, packaging, or external silicon IP that
  only exists in the production node. Explicitly caveated in the
  external sheet.
- **📝 Software** — Pure-software or toolchain claim; lands via a
  software WP with no silicon dependency.

---

## Performance Parameters

| Sheet claim | Status | Backing / work package |
|---|---|---|
| Peak 1258 TOPS (INT8) @ 3.2 GHz, 8:1 sparsity | 🟡 WP-tracked | Needs F1-A3 (8:1 sparsity). MAC count requires Tier D silicon. |
| 2516 TOPS (INT4/FP4) | 🟡 WP-tracked | Needs F1-A1 (FP4 datapath) + F1-A3 (8:1). |
| Typical 500–700 TOPS for ADAS | 🟣 Post-tapeout | Depends on typical-workload MAC utilisation at silicon clocks. |
| 15–30 TOPS/W at 40–50 W typical | 🟡 + 🟣 | Projected 13 → 20–25 with TW-2/TW-3/TW-4. Absolute W figures need silicon power sign-off (post-tapeout). |
| Latency < 0.5 ms | 🟡 WP-tracked | Depends on F1-A4/A5 (transformer engine) + final clock. Achievable for YOLOv8 path once F1-C5 lands; transformer workloads need A4/A5/B6. |
| MAC Utilisation > 90% | 🟡 WP-tracked | Compiler-scheduler claim. Measurable after F1-C5 (YOLOv8 end-to-end) and F1-B6 (transformer models). |
| 400 GB/s LPDDR5X / 750 GB/s HBM3 | 🟣 Post-tapeout | Memory PHY is silicon-only. Controller RTL is a Tier C integration task not yet in plan. |
| ISO 26262 ASIL-D | ✅ RTL-backed | 20 fusion modules + TMR + ECC + safe_state_controller. Formal certification follows tape-out. |
| ISO 21434 cybersecurity-ready | 🟡 WP-tracked | Closes fully with F1-A6 (AES/RSA/TRNG) + F1-A7 (PQC). Process compliance is organisational, not RTL. |
| –40 °C to +125 °C, passive cooling | 🟣 Post-tapeout | Thermal envelope is silicon + package + board-level; cannot be demonstrated on FPGA. |

## Compute Architecture

| Sheet claim | Status | Backing / WP |
|---|---|---|
| **24,576 MAC units** (48 cores × 512 MACs) | 🟣 Post-tapeout | FPGA bring-up targets 64×64 = 4,096 MACs on VU9P. Full 24,576 fits only in silicon. |
| 5-stage PE pipeline | ✅ RTL-backed | `rtl/npu_pe/npu_pe.v`. |
| 2.5–3.2 GHz | 🟣 Post-tapeout | Clock closure at this frequency needs silicon timing. FPGA runs at ~300-500 MHz. |
| Chiplet-ready, UCIe 1.1 (32 Gbps/lane × 16) | 🟣 Post-tapeout | UCIe IP is Tier C (not in F1 plan); physical link is silicon + package. |
| 2000+ TOPS clustering | 🟣 Post-tapeout | Requires UCIe silicon. |
| Precision: INT4 | 🟡 WP-tracked | F1-B2 for quantiser; RTL INT4 path partially present (needs F1-A3 for sparse-INT4). |
| Precision: INT8 | ✅ RTL-backed | Datapath + quantiser + compiler tested on yolov8n end-to-end. Production quant recipe (100-image COCO-128 calibration, per-channel weights, per-tensor percentile-99.9999 activations, NPU-side decode threshold 0.20) measured at **98.4% / 96.0% / 91.2% detection match @ IoU≥0.5/0.7/0.9** vs FP32 on 28 held-out eval images — competitive with TensorRT / OpenVINO INT8 PTQ. Full report: `reports/yolov8n_eval.json`. |
| Precision: INT2 | ✅ RTL-backed | Plumbed in `npu_top.v:111`. |
| Precision: FP4 / FP8 (E4M3/E5M2) / FP16 | 🟡 WP-tracked | F1-A1. Today FP16 is a "placeholder, falls back to INT8". |
| Precision: BF16 / TF32 / FP32 | 🟡 WP-tracked | F1-A2 (blocked by F1-A1). |
| Transformer Engine: 8×MHSA | 🟡 WP-tracked | F1-A5 (blocked by F1-A4). |
| Transformer Engine: rotary PE | 🟡 WP-tracked | F1-A5. |
| Transformer Engine: fused softmax | 🟡 WP-tracked | F1-A4. `npu_activation.v:44` defers to V2. |
| Transformer Engine: layer norm | 🟡 WP-tracked | F1-A4. |
| Transformer Engine: GeLU | ✅ RTL-backed | `npu_activation` AFU LUT (with SiLU + Sigmoid). |
| Sparsity: 2:1 / 4:1 / 8:2 / 8:1 | 🟡 WP-tracked | F1-A3 (consolidates TW-1). |
| Sensor Fusion accelerator (camera/radar/lidar, 4D) | ✅ RTL-backed | 20 modules in `rtl/`: lidar/radar/camera + lane_fusion + object_tracker + dms_fusion. |

## Memory System

| Sheet claim | Status | Backing / WP |
|---|---|---|
| On-Chip SRAM: 128 MB, 16 × 8 MB banks, ECC, dual-port | 🟣 Post-tapeout | SRAM bank + controller RTL exist (`npu_sram_bank`, `npu_sram_ctrl`); 128 MB capacity is silicon-only. ECC is RTL-backed. |
| Scratchpad L0/L1 per core, prefetch-aware DMA | ✅ RTL-backed | `npu_dma` + tile-local SRAM. |
| Compression: 4/8-bit neural-aware, 3–5× | 🟡 WP-tracked | F1-A8. |
| External memory: 512-bit LPDDR5X / 384-bit HBM3 | 🟣 Post-tapeout | Needs licensed PHY + controller IP. Tier C. |

## Connectivity & I/O

| Sheet claim | Status | Backing / WP |
|---|---|---|
| PCIe Gen4 ×4, DMA, P2P | 🟡 + 🟣 | `rtl/pcie_controller/` stub exists; full Gen4 PHY is silicon. |
| MIPI CSI-2, 4-lane, D-PHY/C-PHY, 8K HDR | 🟡 + 🟣 | `rtl/mipi_csi2_rx/` exists; 8K HDR + C-PHY PHY are silicon. |
| 2× CAN-FD with DMA | ✅ RTL-backed | `rtl/canfd_controller/`. |
| Ethernet 1/10/100 Gbps, AVB + TSN | 🟡 WP-tracked | `rtl/ethernet_controller/` exists; AVB/TSN extension is a Tier C sub-task. |
| UCIe 1.1, 32 Gbps/lane × 16 | 🟣 Post-tapeout | IP + silicon. |
| ISP-Pro, 8K HDR, AI denoising, tone mapping | 🟣 Post-tapeout | No `rtl/isp/`; Tier C. |
| V2X / C-V2X accelerator | 🟣 Post-tapeout | No `rtl/v2x/`; depends on modem IP. |
| Debug: JTAG, SPI, UART, I2C, GPIOs | ✅ RTL-backed | Standard debug infrastructure in sensor-fusion path. |

## Security & Functional Safety

| Sheet claim | Status | Backing / WP |
|---|---|---|
| Secure Boot: AES-256, RSA-2048, hardware key storage | 🟡 WP-tracked | F1-A6. |
| NIST PQC (Kyber, Dilithium) | 🟡 WP-tracked | F1-A7. |
| Runtime Protection: AXI snooping, memory firewalls | 🟡 WP-tracked | Partially present via `plausibility_checker`; AXI firewall is F1-A6's runtime extension. |
| Hardware TEE (secure enclave) | 🟡 WP-tracked | F1-A6. |
| ASIL-D, TMR | ✅ RTL-backed | `tmr_voter` + instances in `dms_fusion`, `safe_state_controller`, etc. |
| ECC, watchdog, failover, clock monitors | ✅ RTL-backed | `ecc_secded`, thermal/watchdog modules. |
| OTA: delta compression, rollback, PQC-secured | 🟡 WP-tracked | Needs F1-A6 + F1-A7 + software OTA state machine (new WP to be opened after A6/A7). |

## Power and Thermal Design

| Sheet claim | Status | Backing / WP |
|---|---|---|
| Peak Power 70–90 W, DVFS, MAC-level gating, sparsity-aware | 🟣 Post-tapeout | Power figures are silicon-only. MAC gating itself is RTL (partially in TW-2). |
| Typical 40–50 W, 15–30 TOPS/W | 🟣 Post-tapeout | Silicon measurement. |
| Low-Power Mode: 256 MACs @ 500 MHz, 1–5 W always-on DMS | 🟣 Post-tapeout | DMS RTL path exists (`dms_fusion`); power envelope is silicon. |
| Energy Harvesting (vibration/solar, 1–5 W) | 🟣 Post-tapeout + **aspirational** | Board-level feature; no RTL commitment. Recommended to clarify in the sheet as "supported via companion PMIC" or remove. |
| Predictive ML-based thermal control | 🟡 WP-tracked | F1-A9 (thermal_zone RTL exists; ML classifier is WP). |

## Software Stack

| Sheet claim | Status | Backing / WP |
|---|---|---|
| Compiler: ONNX 2.0 | ✅ RTL-backed | F1-C1 + F1-C1b + F1-C1c. |
| Compiler: PyTorch | 📝 Software | F1-B4. |
| Compiler: TensorRT | N/A | TensorRT is a runtime, not a front-end. Remove from the list. |
| Compiler: TVM, XLA, MLIR, NNEF | 📝 Software | F1-B5. |
| Compiler: AI-driven scheduling | 🟡 | Planned post-F1-C4/C5; tracked in compiler epic, not yet a separate WP. |
| Runtime API (C++/Python) | 📝 Software | Python exists (`tools/npu_ref/`); C++ port is F1-B3. |
| Quantizer INT4/FP4/FP8 + 8:1 sparsity + auto-tiling | 🟡 WP-tracked | F1-B2 landed INT4 (fake-quant SNR 15.7 dB / cos 0.986 on yolov8n). INT8 quantiser is production-recipe today: per-channel weights + per-tensor percentile-99.9999 activation calibration, configurable via `tools/npu_ref/quantiser.py:CALIB_PERCENTILE`. 8:1 sparsity needs F1-A3; FP4/FP8 needs F1-A1. |
| Cycle-accurate C++ simulator | 🟡 WP-tracked | F1-B3. Today the reference is Python. |
| Verilator trace replay | ✅ RTL-backed | Verilator 5.030 + cocotb 2.0.1 working. |
| Telemetry engine | 🟡 WP-tracked | F1-A9. |
| Cloud platform | 🟣 Post-tapeout + organisational | Not a chip feature; belongs in a separate product doc. |

## Model & Application Library

| Sheet model | Status | Backing / WP |
|---|---|---|
| YOLOv8 | ✅ RTL-backed (in progress) | F1-C5 will land end-to-end cocotb run. |
| EfficientNet-B7 | 🟡 WP-tracked | F1-B1 (ops) + F1-B6. |
| ViT-Large | 🟡 WP-tracked | F1-A4/A5 + F1-B1 + F1-B6. |
| BEVFormer | 🟡 WP-tracked | F1-B1 + F1-B6. |
| BERT-Base | 🟡 WP-tracked | F1-B1 + F1-B6. |
| LLaMA-13B (quantized) | 🟡 WP-tracked | F1-A5 + F1-B2 (INT4) + F1-B6. |
| Swin Transformer | 🟡 WP-tracked | F1-B1 + F1-B6. |
| BlazeFace | 🟡 WP-tracked | F1-B6 (depends only on existing ops). |
| Stable Diffusion (quantized) | 🟣 Aspirational | No dedicated WP yet; needs large-model memory footprint handling beyond current DMA tiler. |
| On-Chip Training / fine-tuning | 🟣 Post-tapeout | This is an **inference** accelerator today. Training path is not in the F1 plan and would need a separate program. Recommended to add "inference-optimized; on-chip fine-tuning in roadmap" caveat to the sheet. |

---

## Items recommended for spec-sheet revision

The sheet's external credibility improves significantly with three
minor edits:

1. **Add a "Status" column** to each table with ✅ / 🟡 / 🟣 glyphs
   matching this provenance doc.
2. **Footnote Tier D claims** (24,576 MACs, 128 MB SRAM, 3.2 GHz,
   LPDDR5X, HBM3, UCIe, V2X, ISP-Pro, passive cooling, 40–50 W)
   as "Post-tapeout — measured on A2 Neo silicon sample."
3. **Remove TensorRT** from the compiler front-end list (it's a runtime,
   not a front-end) and **qualify on-chip training** as "fine-tuning
   in roadmap" rather than present-tense.

These edits do not shrink the sheet's ambition — they separate
"committed and demonstrable today" from "committed and in-flight" from
"post-tapeout measured on sample silicon", which is the convention
every serious silicon datasheet uses.

---

## Cross-reference to active WPs

- **Tier A (RTL extensions)**: F1-A1 through F1-A9 (tasks #116–124)
- **Tier B (software breadth)**: F1-B1 through F1-B6 (tasks #125–130)
- **Tier C (integration, not yet in plan)**: UCIe link, LPDDR5X
  controller, ISP-Pro, C-V2X, Ethernet TSN extension. Open as Tier C
  when Tier A/B are substantially complete.
- **Tier D (post-tapeout)**: 24,576 MACs, 128 MB SRAM, 3.2 GHz, clock
  closure, passive-cooling thermals, power envelope measurement,
  chiplet clustering. These close at tape-out.
