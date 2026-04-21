# AstraCore Neo — Spec Sheet Rev 1.4 (DRAFT, audit-cleaned)

> **SUPERSEDED 2026-04-20** by `docs/spec_sheet_rev_1_4.md` — the IP-datasheet
> reframe (per `docs/best_in_class_design.md` §7 founder direction: Path A IP
> licensing) with HARA-derived ASILs, F4 remediation status, and `docs/safety/`
> companion-document references. This draft is retained as revision history.

**Draft status.** This is the internal buyer-safe rewrite of rev 1.3. Every
row carries one of the following flags; external marketing copy should
respect them before publication.

- ✅ **Validated** — demonstrable in the repo today (unit tests, cocotb, RTL-measured).
- 🟡 **Target (pre-tape-out WP)** — work package open; closable on FPGA without a 7 nm run.
- 🟣 **Target (post-tape-out)** — silicon-only; cannot be demonstrated on FPGA or bench.
- ⚠️ **Rewording** — rev 1.3 claim is stronger than the repo supports; reworded here.

Cross-reference: `docs/spec_sheet_provenance.md`, `docs/buyer_dd_findings_2026_04_19.md`.

---

## Document header

**Chip:** A2 Neo. **Target node:** TSMC N7. **Revision:** 1.4 DRAFT. **Date:** 2026-04-19.
**Classification:** Proprietary & Confidential.

---

## 1 — Key Differentiators (page 1)

- **ISO 26262 ASIL-D architecture** — TMR voter + ECC SECDED + safe-state
  controller + plausibility checker are RTL-backed (✅). FMEDA + formal
  certification follow tape-out (🟣).
- **Peak 1258 TOPS (INT8) projected on 7 nm silicon** — 24 576 MACs × 3.2 GHz
  × 8:1 structured sparsity multiplier. 🟣 post-tape-out projection; FPGA
  validates the INT8 datapath at 4×4 / 8×8 array scale.
- **Efficiency 15–30 TOPS/W (projected)** — silicon-only, requires power
  sign-off at tape-out. 🟣
- **Ultra-Low Latency** — YOLOv8-N projected 1.45 ms on silicon (ultra
  tier, dense INT8). ⚠️ rev 1.3's *"< 0.5 ms"* claim is reworded as
  *"< 0.5 ms achievable via multi-stream batching or reduced-model
  configurations; single-stream YOLOv8-N projected 1.45 ms."*
- **Future-proof precision** — INT8 / INT4 / INT2 validated; FP8 / FP16
  sim-gate PASS (🟡 F1-A1.1 synthesizable RTL pending); FP4 / BF16 / TF32 /
  FP32 🟡 F1-A1.2 / F1-A2.
- **Open Python SDK, source-available RTL** — Apache-2.0 covers `src/` +
  `tools/`; evaluation licence covers `rtl/`. See `LICENSE`.
- **Threshold-based fault / thermal supervisor (ML upgrade on roadmap)** —
  ⚠️ rev 1.3's *"ML-based predictive fault detection"* is reworded; current
  code is a rule-based threshold predictor with an ML upgrade WP (F1-A9).

---

## 2 — Performance Parameters

| Metric | Rev 1.4 text | Flag |
|---|---|---|
| Peak Throughput | 1258 TOPS (INT8) projected @ 3.2 GHz / 8:1 sparsity on 7 nm silicon; 2516 TOPS (INT4/FP4) | 🟣 |
| Typical Throughput | Multi-stream aggregate 500-700 TOPS (INT8) ADAS mix; single-stream 5-20 % MAC utilisation typical | ⚠️ + 🟣 |
| Power Efficiency | 15-30 TOPS/W projected at 40-50 W typical | 🟣 |
| Latency | < 0.5 ms multi-stream batched; YOLOv8-N single-stream projected 1.45 ms on ultra-tier silicon | ⚠️ + 🟣 |
| MAC Utilisation | Peak > 90 % aggregate multi-stream; 5-20 % typical single-stream | ⚠️ |
| Memory Bandwidth | 400 GB/s LPDDR5X / 750 GB/s HBM3 (requires licensed PHY + controller IP) | 🟣 |
| Functional Safety | ISO 26262 ASIL-D architecture ✅; formal certification + FMEDA post-tape-out | ✅ / 🟣 |
| Operating Range | -40 °C to +125 °C, passive cooling (silicon + package characterisation) | 🟣 |

---

## 3 — Compute Architecture

| Component | Rev 1.4 text | Flag |
|---|---|---|
| MAC Array | 24 576 MAC units (48 cores × 512 MACs), 2.5-3.2 GHz on 7 nm silicon. **Array parameterised; FPGA validates 4×4 and 8×8 today; silicon target 24 576 MACs.** | 🟣 |
| Scalability | Chiplet-ready via UCIe 1.1 (32 Gbps/lane × 16) target; ≥ 2000 TOPS clustering on silicon | 🟣 |
| Precision — INT8 | Per-channel weight + per-tensor percentile-99.9999 activation PTQ; 98.4 / 96.0 / 91.2 % detection match vs FP32 on yolov8n @ IoU ≥ 0.5 / 0.7 / 0.9 (28-image COCO-128 eval) | ✅ |
| Precision — INT4 | Fake-quant SNR 15.7 dB / cos 0.986 on yolov8n; RTL datapath validated | ✅ |
| Precision — INT2 | Plumbed + bit-exact vs Python mirror; detection collapses without QAT (representation wall) | ✅ |
| Precision — FP8 (E4M3/E5M2) | Sim-gate PASS (5/5 cocotb); synthesizable RTL pending | 🟡 F1-A1.1 |
| Precision — FP16 | Sim-gate PASS; main systolic array currently falls back to INT8 on precision_mode=11 | 🟡 F1-A1.1 |
| Precision — FP4 / BF16 / TF32 / FP32 | Target set; no current RTL | 🟡 F1-A1.2 / F1-A2 |
| Transformer Engine | Softmax ✅, LayerNorm / RMSNorm ✅, GeLU ✅ (AFU LUT); 8× MHSA tile target | 🟡 F1-A5 |
| Sparse Execution — 2:4 structured | PE-level skip-gate port plumbed ✅; 2:4 index decoder pending | 🟡 F1-A3 |
| Sparse Execution — 4:1 / 8:2 / 8:1 | Target; requires QAT pipeline | 🟡 F1-A3 |
| Sensor Fusion Accelerator | 20 RTL modules for camera / radar / lidar / IMU / GNSS / DMS; 32/32 OpenLane sky130 ASIC batch PASS; lidar point-cloud + temporal fusion (⚠️ "4D point cloud processing" reworded) | ✅ |

---

## 4 — Memory System

| Component | Rev 1.4 text | Flag |
|---|---|---|
| On-Chip SRAM | 128 MB, 16 × 8 MB banks, ECC SECDED, target silicon layout. RTL bank parameterised; ECC ✅ | 🟣 + ✅ |
| Scratchpad / DMA | Prefetch-aware DMA ✅; L0/L1 hierarchy + cache coherence target post-tape-out | ✅ / 🟣 |
| Compression | 4-bit / 8-bit neural-aware encoding, 3-5× gain — **WP target, no RTL decoder or measured ratio today** | 🟡 F1-A8 |
| External Memory | 512-bit LPDDR5X (400 GB/s) / 384-bit HBM3 (750 GB/s) via licensed PHY + controller IP | 🟣 |

---

## 5 — Connectivity & I/O

| Interface | Rev 1.4 text | Flag |
|---|---|---|
| PCIe | Gen4 ×4 target; frame + link-training RTL scaffolding ✅; Gen4 PHY silicon-only | ✅ / 🟣 |
| MIPI CSI-2 | 4-lane D-PHY target; packet-layer RTL ✅; C-PHY and 8K HDR require silicon IP | ✅ / 🟣 |
| CAN-FD | 2× CAN-FD with DMA ✅ (`rtl/canfd_controller/`) | ✅ |
| Ethernet | 1 Gbps RTL scaffolding ✅; 10 / 100 Gbps PHY + AVB/TSN extensions target | ✅ / 🟡 |
| UCIe | UCIe 1.1, 32 Gbps/lane × 16 — licensed IP + package | 🟣 |
| ISP | ISP-Pro, 8K HDR, AI denoising, tone mapping — **Tier C IP**, no RTL today | 🟣 |
| V2X | C-V2X protocol stack (cellular modem external) — Python reference; no RTL | 🟡 |
| Debug | JTAG / SPI / UART / I²C / GPIO — standard silicon pads | 🟣 |

---

## 6 — Security & Functional Safety

| Feature | Rev 1.4 text | Flag |
|---|---|---|
| Secure Boot | AES-256, RSA-2048, NIST PQC (Kyber, Dilithium), hardware key storage — **target; Python reference only; no crypto RTL today** | 🟡 F1-A6 / A7 |
| Runtime Protection | Memory firewalls + bus-monitor target — **AXI bus substrate is a prerequisite WP**; no AXI in current RTL | 🟡 |
| Hardware TEE | Secure enclave for model decryption — Python skeleton; RTL target post-Ibex integration | 🟡 |
| ASIL Compliance | ISO 26262 ASIL-D architecture ✅ (TMR voter, ECC SECDED, safe-state controller, plausibility checker); FMEDA + certification post-tape-out | ✅ / 🟣 |
| Safety Features | ECC ✅, watchdog ✅, failover ✅, clock monitors (silicon-level clock tree) | ✅ / 🟣 |
| OTA Updates | Delta compression ✅, rollback ✅, PQC-secured (blocked on F1-A7) | ✅ / 🟡 |

---

## 7 — Power and Thermal Design

| Category | Rev 1.4 text | Flag |
|---|---|---|
| Peak Power | 70-90 W with DVFS + MAC-level gating (silicon measurement) | 🟣 |
| Typical Power | 40-50 W, 15-30 TOPS/W (INT8) projected | 🟣 |
| Low-Power Mode | 256 MACs @ 500 MHz always-on DMS (~1-5 W, silicon-only mode) | 🟣 |
| ~~Energy Harvesting~~ | **Removed from spec.** Board-level feature outside the chip's scope. | ⚠️ |
| Thermal Management | Rule-based thermal zone controller ✅; ML predictive upgrade on roadmap (F1-A9) | ✅ / 🟡 |

---

## 8 — Software Stack

| Component | Rev 1.4 text | Flag |
|---|---|---|
| Compiler Frontends | ONNX 2.0 ✅, PyTorch ✅, TVM ✅, XLA ✅, MLIR ✅, NNEF ✅ (all via shared ONNX path). **TensorRT removed** — it is a runtime, not a front-end; we share models via ONNX export | ✅ |
| Auto-tiling | Deterministic im2col + chained-tile ✅; ML-tunable heuristics on roadmap (⚠️ *"AI-driven scheduling"* reworded) | ✅ |
| Runtime API | Python ✅ (`tools/npu_ref/nn_runtime.py`, full yolov8n end-to-end in ~3.4 s); C++ port target (F1-B3) | ✅ / 🟡 |
| Quantiser | INT8 production-grade ✅; INT4 ✅; FP4 / FP8 🟡 F1-A1; 8:1 sparsity 🟡 F1-A3; auto-tiling ✅ | ✅ / 🟡 |
| Simulators | Python cycle-accurate ✅ (`pe_ref.py`, `systolic_ref.py`, `tile_ctrl_ref.py`); C++ port target (F1-B3); Verilator trace replay ✅ | ✅ / 🟡 |
| Telemetry Engine | Rule-based logging + fault predictor ✅; ML predictive upgrade roadmap (F1-A9) | ✅ / 🟡 |
| ~~Cloud Platform~~ | **Removed from spec** — services offering, not a chip feature. See company services page. | ⚠️ |

---

## 9 — Model & Application Library

| Workload | Rev 1.4 text | Flag |
|---|---|---|
| Vision — YOLOv8-N | End-to-end INT8 PTQ validated (98.4 / 96.0 / 91.2 % match vs FP32, 28-image eval) | ✅ |
| Vision — EfficientNet-B7 / ViT-Large / BEVFormer | Ops in loader ✅; end-to-end validation target | 🟡 F1-B6 |
| Transformers — BERT-Base / LLaMA-13B / Swin | Transformer ops ✅; 8× MHSA tile target | 🟡 F1-A5 + F1-B6 |
| Sensor Fusion — radar/camera/lidar + occupancy | 20 RTL modules, 32/32 ASIC batch PASS | ✅ |
| Speech / NLP — ASR | Target; op subset covered by transformer engine | 🟡 |
| TinyML & DMS — BlazeFace, driver alertness | DMS fusion RTL ✅; BlazeFace target | ✅ / 🟡 |
| ~~Generative AI — Stable Diffusion (quantised)~~ | **Moved to V2 roadmap.** Memory footprint exceeds current DMA tiler. | ⚠️ |
| ~~On-Chip Training — fine-tuning for ADAS~~ | **Removed from spec.** This is an inference accelerator; no backward-pass RTL. Re-introduced as an optional post-tape-out roadmap item. | ⚠️ |

---

## 10 — Validation Snapshot (NEW section)

Added in rev 1.4 so external reviewers can anchor the text above to
reproducible numbers.

- **Python suite:** `pytest -m "not integration"` → 1025 pass, 1 skip, 7 deselected
- **RTL cocotb:** `run_verilator_npu_top.sh` → 6 / 6 PASS at 4×4; `run_verilator_npu_top_8x8.sh` → 2 / 2 PASS at 8×8
- **F1-A4 RTL:** softmax + layernorm + RMSNorm bit-exact vs Python mirrors (10 / 10 PASS)
- **YOLOv8 eval:** `reports/yolov8n_eval.json` (28-image eval; ≥ 500-image run scheduled before external publication)
- **ASIC PPA:** 32 / 32 modules close on sky130 130 nm via OpenLane; 7 nm PPA post-tape-out

---

## 11 — Editor notes (strip before external publication)

- Remove every **~~strikethrough~~**; these are rev 1.3 claims removed from rev 1.4.
- Keep the Validation Snapshot section — it is what makes this sheet credible vs a pure-marketing datasheet.
- Any claim that shows ⚠️ here should be the **only** place the reworded text appears in any external comms.
- "ISP-Pro" row: decide whether to (a) remove, (b) caveat as "via licensed ISP IP integration", or (c) open a dedicated WP.
- License posture (see `LICENSE`): confirm with counsel before external publication.
- The stray `|` typo in rev 1.3 page 3 Typical Power row has been cleaned.
