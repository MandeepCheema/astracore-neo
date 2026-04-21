# AstraCore Neo — Software-First Execution Plan (2026-04-19)

**Business model.** AstraCore sells a **software/firmware SDK** to automotive Tier-1 suppliers who deploy it on their own silicon (Qualcomm SA8650P, NVIDIA Orin/Thor, Ambarella CV3, Mobileye EyeQ Ultra, or custom). The chip spec sheet's 1258 TOPS / 15-30 TOPS/W are **targets the software must help customer silicon achieve**, not standalone hardware claims.

**Deliverable language.** C/C++ for runtime (OEM firmware). Python for tooling (compiler, quantiser, evaluation harness).

**North-star metric.** TOPS and TOPS/W delivered on the customer's silicon. Everything else — model coverage, sensor integration, extensibility — exists in service of those two numbers.

---

## The TOPS multiplier chain

```
Silicon peak TOPS (dense INT8)
 × quantisation multiplier   (INT8 → INT4 = 2× ;  INT2 = 4× ;  FP4 = 2×)
 × sparsity multiplier       (2:4 = 2× ;  4:1 = 4× ;  8:1 = 8×)
 × MAC utilisation factor    (0.05 today single-stream → 0.90 target multi-stream)
 = delivered TOPS

TOPS/W  =  delivered TOPS ÷ delivered Watts
        ↑ software levers: sparsity, quantisation, tiling, fusion (less energy per op)
```

For the spec sheet to hold on customer silicon:
- **1258 TOPS** = 24576 MACs × 2 × 3.2 GHz × **8** (sparsity) ÷ 10¹² → software must deliver the 8× sparsity
- **15-30 TOPS/W** = ~5 TOPS/W silicon floor × 3-6× software efficiency → software must deliver the 3-6× via sparsity + quantisation + utilisation

## Software contribution per lever (current status)

| Lever | Status | TOPS multiplier | TOPS/W multiplier | Blocker |
|---|---|---|---|---|
| INT8 quantiser | ✅ 98.4 % match on YOLOv8 | 1× (baseline) | 1× (baseline) | — |
| INT4 quantiser | ✅ SNR 15.7 dB | 2× | ~1.7× | QAT for accuracy |
| INT2 quantiser | 🟠 representation wall | 4× | ~3× | QAT + head-at-higher-precision |
| FP4 quantiser | ❌ not implemented | 2× | ~1.7× | F1-A1.2 |
| 2:4 structured sparsity | 🟠 PE skip-gate port | 2× | ~1.8× | Metadata decoder RTL |
| 4:1 / 8:1 sparsity + QAT | ❌ | 4-8× | 3-6× | F1-A3 + QAT pipeline |
| MAC-util 6 % → 90 % | 🟠 6.14 % measured | 15× effective | 15× effective | Compiler + multi-stream batching |
| Operator fusion | ✅ SiLU fusion in F1-C1c | 1.1-1.3× | 1.2-1.5× | Partial; expandable |
| C++ runtime | ❌ Python only | — | — | F1-B3 |

**Critical path to spec-sheet numbers:** MAC-util (×15) + sparsity+QAT (×8) = combined multiplier gets you from 1 TOPS effective to 120 TOPS effective on the same silicon without any new MAC. That's the thesis the software sale rests on.

---

## Three-phase program

### Phase A — Python test framework (3-4 weeks)

**Goal.** Prove the three pillars (extensibility, model coverage, sensor I/O) work as a **framework**. Back-end here is still the internal NPU simulator. No silicon or C++ yet.

**Why first.** Lets us validate the plugin contract, operator coverage, and dataset integration with fast iteration. When C++ lands in Phase B, the test harness is already proven.

**Deliverables:**
- `pyproject.toml` + pip-installable (`pip install -e .` works)
- Plugin API via setuptools entry-points (`@register_op`, `@register_quantizer`, `@register_backend`)
- Backend abstraction protocol (NPU-sim today; F1, Orin, CPU later)
- `astracore bench --model X.onnx` CLI → standardised JSON + markdown report
- Model zoo: 5-8 ONNX models in `data/models/zoo/` with `scripts/fetch_model_zoo.sh`
- Benchmark matrix: per-model {load OK, INT8 Δ vs FP32, MAC util, latency}
- Extensibility test (`tests/test_extensibility.py`) — passes without importing anything private
- Dataset connector (nuScenes subset, KITTI optional) — replay one scene, dump tracks
- Multi-stream MAC-util measurement (4-8 concurrent YOLOv8 streams)

**Success criteria:**
1. OEM can `pip install astracore-sdk` and run `astracore bench --model their-model.onnx`
2. OEM plugin package registers a custom op without editing our source → confirmed by test
3. ≥ 5 models have published accuracy + MAC-util numbers
4. Multi-stream MAC util ≥ 40 % on YOLOv8 × 4 streams (4-6× improvement over single-stream)

### Phase B — C++ runtime + F1 hardware validation (4-6 weeks)

**Goal.** Deliver the runtime in C++ (the form OEMs actually deploy). Validate on AWS F1 (64×64 = 4096 MACs) as first real-silicon proof.

**Why F1 first (not Orin).** F1 runs OUR RTL → cycle-accurate extrapolation to 7 nm silicon. Orin runs TensorRT → demonstrates compiler quality but not our stack's silicon contribution.

**Deliverables:**
- `runtime/cpp/` with CMake build
- Host-side XRT driver (port of `fpga/aws_f1/host/f1_npu_driver.py`)
- `astracore_bench` C++ binary (same report shape as Phase A's Python)
- AWS F1 bitstream compile (Vivado synth + P&R)
- F1 deploy + first runs: YOLOv8 + at least 3 model-zoo members
- Measured: {MAC util, wall-clock latency, FPGA power, delivered TOPS dense INT8}

**Success criteria:**
1. C++ binary produces bit-identical output to Python reference on YOLOv8
2. F1 runs end-to-end YOLOv8 at ≥ 50 % MAC util
3. Measured F1 TOPS extrapolate cleanly to silicon-scale projection (≥ 90 % of theoretical)
4. FPGA power measured, TOPS/W on F1 reported (will be small — F1 is 14 nm FPGA)

**Honest framing of F1 numbers.** F1 = VU9P, 14 nm, 4096 MACs @ 250 MHz → peak ~2 TOPS dense INT8. This is **6× fewer MACs and 13× slower clock** than silicon target. F1 numbers are *ratios* (MAC util, sparsity gain, BW efficiency), not absolute TOPS — those extrapolate to silicon.

### Phase C — Sparsity engine + QAT pipeline (4-5 weeks)

**Goal.** Unlock the 8× sparsity multiplier. Without this, the 1258 TOPS math does not close on any silicon.

**Deliverables:**
- F1-A3 RTL: sparsity metadata decoder + compaction pipeline (2:4, 4:1, 8:1 structured)
- QAT pipeline (PyTorch-based): iterative prune + fine-tune on YOLOv8 + 2-3 zoo models
- Measured accuracy at each sparsity ratio (target: ≤ 2 % mAP drop vs dense INT8)
- Re-benchmark on F1 with sparsity enabled → measure effective TOPS multiplier
- Updated `reports/pruning_accuracy.json` replacing the current 0 %-baseline with QAT numbers

**Success criteria:**
1. 2:4 structured sparsity: ≤ 1 % mAP drop, measured 1.8-2.0× TOPS gain on F1
2. 4:1 structured sparsity: ≤ 2 % mAP drop, measured 3.5-4.0× TOPS gain on F1
3. 8:1 sparsity (aspirational): measured viable on at least one model family

---

## OEM-evaluation readiness checklist (end of Phase C)

This is what an OEM technical-DD review expects to see:

- [x] Python tests pass (1025 today)
- [x] RTL cocotb pass (6/6 today)
- [ ] `pip install astracore-sdk` works
- [ ] `astracore bench --model <their-model>.onnx` produces report
- [ ] 5-8 model benchmark matrix with published numbers
- [ ] Extensibility demonstrated via external plugin package
- [ ] nuScenes (or KITTI) end-to-end scenario replay
- [ ] C++ runtime parity test with Python
- [ ] F1 silicon run with measured MAC util, power, TOPS
- [ ] Sparsity + QAT with measured accuracy at each ratio
- [ ] Integration guide + API docs published
- [ ] Commercial licence decided (NOT Apache-2.0 for the SDK if selling)

---

## Not in this plan (explicit descope)

Items from the spec sheet we are **deliberately not** attempting in this program, because they are hardware-only or out of software scope:

- 7 nm silicon tape-out
- Memory PHY (LPDDR5X / HBM3)
- UCIe chiplet link
- MIPI C-PHY, 10/100 G Ethernet PHY, PCIe Gen4 PHY
- ISP-Pro silicon IP
- AES / RSA / PQC RTL (left as Python reference implementations for customer firmware port)
- ASIL-D FMEDA + formal certification (customer's silicon's concern)
- 24576-MAC scale validation (F1 caps at 4096; full scale is post-tape-out)

When talking to OEMs, these are re-framed as *"we deliver software; your silicon and memory subsystem are yours"* — which is the point of a software-SDK sale.

---

## Schedule estimate (calendar weeks)

| Phase | Optimistic | Realistic | Gating |
|---|---|---|---|
| A — Python framework | 3 | 4 | Dataset connector effort on nuScenes/KITTI |
| B — C++ + F1 | 4 | 6 | Vivado synth iterations + AWS deploy learning curve |
| C — Sparsity + QAT | 4 | 5 | QAT training time on GPU |
| **Total to OEM demo** | **11** | **15** | — |

Target OEM-pitch-ready: **15 weeks from today (2026-04-19)**. That lands around end-Q3 / early-Q4 2026.

---

## Today's decision

**Moving forward with Phase A — Python-first.** C++ and F1 wait until Phase A's framework is proven. Execution starts with the plugin API + pip packaging so the extensibility contract is established before any downstream work depends on it.
