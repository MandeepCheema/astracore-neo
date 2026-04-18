# AstraCore v2 — AI-Forward NPU Architecture
# Rev 1.0 | 2026-04-16
# Target: Programmable AI Inference + ASIL-D Sensor Fusion on One Die

---

## Executive Summary

AstraCore v2 evolves from a sensor fusion co-processor into a **full-stack
autonomous driving SoC** — combining a programmable Neural Processing Unit
(NPU) for real-time AI perception with the proven ASIL-D sensor fusion
pipeline for safety-critical decision-making.

The key insight: AstraCore v1's fusion pipeline already defines the exact
interface a perception engine must satisfy (`camera_detection_t` structs).
The NPU is a drop-in replacement for the external CNN assumption — bringing
perception on-chip without changing a single line of the fusion pipeline.

**Competitive position:** A single-chip solution with hardware-verified
ISO 26262 ASIL-D safety alongside programmable AI — something neither
pure-NPU startups (Hailo, Blaize, Tenstorrent) nor GPU vendors (NVIDIA,
Qualcomm) currently offer in a single tightly-coupled die.

---

## 1. Architecture Overview

### 1.1 Three-Subsystem SoC

```
                   AstraCore v2 SoC
 ┌──────────────────────────────────────────────────────────┐
 │                                                          │
 │  ┌─────────────────────┐        ┌─────────────────────┐ │
 │  │   NPU Subsystem      │        │  Fusion Pipeline     │ │
 │  │                       │ det_t  │  (20 modules,        │ │
 │  │  Systolic Array       │───────►│   ASIL-D verified)   │ │
 │  │  Activation Unit      │        │                      │ │
 │  │  SRAM Scratchpad      │        │  sensor_sync         │ │
 │  │  DMA Engine           │        │  coord_transform     │ │
 │  │  Tile Controller      │        │  object_tracker      │ │
 │  │                       │        │  plausibility_check  │ │
 │  └──────────┬────────────┘        │  ttc_calculator      │ │
 │             │ AXI                  │  aeb_controller      │ │
 │             │                      │  safe_state_ctrl     │ │
 │  ┌──────────▼────────────┐        └──────────┬──────────┘ │
 │  │  AXI4 Interconnect    │◄──────────────────┘            │
 │  └──┬──────────┬─────────┘                                │
 │     │          │                                          │
 │  ┌──▼──┐  ┌───▼──────────┐                               │
 │  │RISCV│  │ Memory Ctrl   │                               │
 │  │CPU  │  │ DDR4/LPDDR5   │                               │
 │  └─────┘  └───────────────┘                               │
 └──────────────────────────────────────────────────────────┘
```

### 1.2 Design Philosophy

| Principle | Rationale |
|-----------|-----------|
| **Perception is programmable** | Models evolve yearly; silicon must support OTA weight updates |
| **Safety is fixed-function** | ASIL-D certification requires deterministic, verifiable hardware |
| **Clean interface boundary** | NPU outputs `camera_detection_t` structs — same interface the fusion pipeline already consumes |
| **Tiled compute** | Systolic array processes one tile at a time; SRAM double-buffers tiles for zero-stall pipelining |
| **Memory-bandwidth aware** | Dataflow architecture minimizes off-chip memory accesses — the #1 bottleneck in AI inference |

---

## 2. NPU Subsystem — Detailed Architecture

### 2.1 Systolic Array (Compute Core)

The systolic array is a 2D grid of Processing Elements (PEs), each containing
one MAC unit identical to the existing `mac_array.v` module. Data flows
through the grid in a weight-stationary pattern:

```
Weight-Stationary Dataflow (NxN array):

Activations broadcast across columns (flow right):
      a[0]  a[1]  a[2]  ...  a[N-1]
        │     │     │           │
        ▼     ▼     ▼           ▼
w[0] ─►[PE]─►[PE]─►[PE]─ ... ─►[PE]──► psum[0]
w[1] ─►[PE]─►[PE]─►[PE]─ ... ─►[PE]──► psum[1]
w[2] ─►[PE]─►[PE]─►[PE]─ ... ─►[PE]──► psum[2]
  :     :     :     :           :        :
w[N] ─►[PE]─►[PE]─►[PE]─ ... ─►[PE]──► psum[N-1]

Each PE:
  ┌──────────────────┐
  │  weight_reg (W)  │  ← loaded once per tile
  │                  │
  │  acc += A_in * W │  ← computed every cycle
  │                  │
  │  A_out = A_in    │  ← pass activation to next PE (1-cycle delay)
  └──────────────────┘

One NxN tile computes: C[N×N] += A[N×K] × B[K×N]
over K cycles (one column of A per cycle).
```

**Precision support:**

| Format | Bits | Use Case | MAC output |
|--------|------|----------|------------|
| INT8 × INT8 | 8×8 → 32 acc | CNN inference (YOLO, MobileNet) | 32-bit accumulator |
| INT4 × INT4 | 4×4 → 32 acc | Quantized LLMs (LLaMA INT4) | 32-bit accumulator, 2× throughput |
| FP16 × FP16 | 16×16 → 32 acc | Fine-tuning, high-precision inference | 32-bit FP accumulator |
| INT8 × INT4 | Mixed | Activation INT8 × Weight INT4 | Flexible quantization |

**Why weight-stationary?**
- Weights are loaded once per tile, reused across all activations
- Minimizes weight memory bandwidth (the bottleneck for transformers)
- Simple control logic — load weights, stream activations, drain outputs
- Same dataflow works for CNN convolutions AND transformer attention (Q×K^T, Attn×V)

### 2.2 Activation Function Unit (AFU)

Transformers and modern CNNs require non-linear activation functions that
cannot be computed in the systolic array. The AFU is a pipelined unit that
processes the partial sum output of the systolic array.

```
Systolic Array Output ──► AFU Pipeline ──► SRAM Writeback

AFU Pipeline (4-stage):
  Stage 1: Mode select (ReLU / GELU / SiLU / Softmax / LayerNorm / bypass)
  Stage 2: Lookup + interpolation (piecewise-linear approximation)
  Stage 3: Normalization arithmetic (for LayerNorm / Softmax)
  Stage 4: Quantize + saturate (INT32 → INT8 re-quantization)
```

**Supported operations:**

| Function | Used By | Implementation |
|----------|---------|----------------|
| ReLU | YOLOv8, CNNs | `max(0, x)` — trivial, 1 cycle |
| GELU | ViT, BEVFormer, LLaMA | Piecewise-linear LUT (16 segments), <0.1% error |
| SiLU (Swish) | YOLOv8 v5+, EfficientNet | `x * sigmoid(x)` — LUT for sigmoid, 1 multiply |
| Softmax | Attention mechanism (all transformers) | Log-domain: `exp(x - max) / sum(exp)`, streaming accumulator |
| LayerNorm | ViT, LLaMA, BEVFormer | Two-pass: compute mean+var, then normalize. Needs SRAM buffering |
| BatchNorm | YOLOv8 backbone | Fused into conv weights at compile time — zero runtime cost |

**Softmax hardware detail:**
```
Streaming Softmax (avoids storing all values):
  Pass 1: Find max value across vector (streaming comparator)
  Pass 2: Compute exp(x_i - max) and running sum (LUT + accumulator)
  Pass 3: Divide each exp(x_i - max) by sum (reciprocal LUT + multiply)

Total latency: 3N cycles for N-element vector
SRAM needed: one row buffer (N × 16 bits)
```

### 2.3 SRAM Scratchpad

The scratchpad is the on-chip working memory for the NPU. It holds weight
tiles, activation tiles, and intermediate results. Double-buffering allows
compute and data loading to overlap.

```
SRAM Scratchpad Layout:

┌─────────────────────────────────────┐
│  Bank A: Weight Buffer              │  ← DMA loads next tile's weights
│  (N×N × 8 bits = N² bytes per tile) │     while systolic array uses Bank B
├─────────────────────────────────────┤
│  Bank B: Weight Buffer              │  ← Systolic array reads current tile
│  (N×N × 8 bits)                     │
├─────────────────────────────────────┤
│  Bank C: Activation Input           │  ← Input feature map tile
│  (N×K × 8 bits per tile)            │
├─────────────────────────────────────┤
│  Bank D: Activation Output          │  ← Partial sums / AFU output
│  (N×N × 32 bits accumulator)        │
├─────────────────────────────────────┤
│  Bank E: Scratch / KV Cache         │  ← Transformer attention workspace
│  (variable)                         │
└─────────────────────────────────────┘
```

**Sizing by tier:**

| Tier | SRAM | Largest layer it can tile | Technology |
|------|------|--------------------------|------------|
| Demo (sky130) | 64 KB | MobileNetV2 depthwise conv (32×32×32) | OpenRAM macros |
| Starter (28nm) | 512 KB | YOLOv8-N C2f block (128×128×64) | Foundry SRAM |
| Mid (28nm) | 2 MB | ViT-B attention head (768×768 Q×K^T) | Foundry SRAM |
| Pro (12nm) | 8 MB | BEVFormer spatial cross-attention | Foundry SRAM |
| Ultra (7nm) | 32 MB | LLaMA-7B single-layer KV cache | Foundry SRAM |

**OpenRAM for sky130:**
Sky130 does not include foundry SRAM macros. OpenRAM (open-source SRAM
compiler) generates custom SRAM arrays for sky130:
- Density: ~250 bits/cell on sky130 → 64KB ≈ 2.1 mm²
- Read/write: single-port synchronous, 1 cycle latency
- Proven: multiple tapeouts on sky130 (Efabless Chipignite, Google MPW)

### 2.4 DMA Engine

The DMA engine moves data between external memory (DDR/SRAM) and the
on-chip scratchpad. It supports tiled transfers with stride and padding.

```
DMA Transfer Descriptor:
┌────────────────────────────────────────┐
│  src_addr    [31:0]  — external memory │
│  dst_addr    [15:0]  — scratchpad bank │
│  tile_h      [15:0]  — tile height     │
│  tile_w      [15:0]  — tile width      │
│  src_stride  [15:0]  — row stride      │
│  pad_top/bot [3:0]   — zero-padding    │
│  pad_left/rt [3:0]   — zero-padding    │
│  done_irq    [0:0]   — interrupt on    │
│                         completion     │
└────────────────────────────────────────┘
```

**Double-buffering flow:**
```
Time ────────────────────────────────────────────►

DMA:    [Load Tile 0] [Load Tile 1] [Load Tile 2] ...
                  ↕ swap       ↕ swap       ↕ swap
Compute:         [Compute T0] [Compute T1] [Compute T2] ...

Result: DMA and compute fully overlap after initial fill.
        Effective throughput = max(DMA time, compute time).
```

### 2.5 Tile Controller / Sequencer

The tile controller orchestrates the NPU by executing a pre-compiled
schedule of tile operations. Each instruction describes one tile of
matrix multiplication.

```
Tile Instruction Format (64 bits):
┌──────────────────────────────────────────────────┐
│ [63:60] opcode     — LOAD_W / COMPUTE / ACTIVATE │
│                      / STORE / SYNC / NOP         │
│ [59:48] tile_m     — output tile row count        │
│ [47:36] tile_n     — output tile col count        │
│ [35:24] tile_k     — reduction dimension          │
│ [23:20] act_func   — ReLU/GELU/SiLU/Softmax/none │
│ [19:16] precision  — INT8/INT4/FP16               │
│ [15:0]  sram_bank  — which scratchpad bank        │
└──────────────────────────────────────────────────┘
```

**Two control approaches by tier:**

| Tier | Controller | Flexibility | Complexity |
|------|-----------|-------------|------------|
| Demo/Starter | Microcode ROM | Fixed model, fast boot | ~300 lines RTL |
| Mid/Pro/Ultra | RISC-V RV32IMC | Any model via firmware | ~5000 lines RTL (PicoRV32) |

For the RISC-V approach, the CPU runs a thin firmware that:
1. Parses a compiled model graph (stored in external memory)
2. Programs DMA descriptors for each layer's tiles
3. Configures the systolic array precision and AFU mode
4. Synchronizes on DMA-done and compute-done events
5. Triggers the next layer

---

## 3. Model Compatibility Analysis

### 3.1 YOLOv8 (Object Detection — Primary Use Case)

```
YOLOv8-N Architecture:
  Backbone: CSPDarknet (Conv + C2f blocks)
  Neck:     PANet (feature pyramid)
  Head:     Decoupled head (cls + box + obj)

Layer breakdown:
  Conv2D:       ~85% of compute → systolic array (native)
  Batch Norm:   Fused into Conv at compile time → zero cost
  SiLU:         ~5% of compute → AFU
  Concat/Route: Data movement only → DMA
  Upsample:     Nearest-neighbor → simple address generator

Compute requirements:
  YOLOv8-N: 8.7 GFLOPs per frame @ 640×640
  YOLOv8-S: 28.6 GFLOPs
  YOLOv8-M: 78.9 GFLOPs
  YOLOv8-X: 257.8 GFLOPs

Performance projections (INT8):
  ┌──────────┬───────┬──────────┬─────────────────────┐
  │ Tier     │ TOPS  │ YOLOv8-N │ YOLOv8-X            │
  ├──────────┼───────┼──────────┼─────────────────────┤
  │ Demo     │ 0.026 │ 3 fps    │ not feasible        │
  │ Starter  │ 0.4   │ 46 fps   │ 1.5 fps             │
  │ Mid      │ 4.1   │ 471 fps  │ 16 fps              │
  │ Pro      │ 32.8  │ >1000fps │ 127 fps             │
  │ Ultra    │ 196   │ >1000fps │ 760 fps             │
  └──────────┴───────┴──────────┴─────────────────────┘

  Automotive requirement: 30 fps per camera × 6 cameras = 180 fps
  YOLOv8-N @ 180 fps → needs ~0.35 TOPS → Starter tier sufficient
  YOLOv8-M @ 180 fps → needs ~3.2 TOPS → Mid tier sufficient
```

### 3.2 BEVFormer (Bird's Eye View Transformer — SOTA Perception)

```
BEVFormer Architecture:
  Multi-camera images → Backbone (ResNet/Swin) → BEV Queries
  → Spatial Cross-Attention (deformable) → Temporal Self-Attention
  → BEV Feature Map → Detection Head

Key operations:
  Backbone CNN:         ~60% compute → systolic array (standard conv)
  Attention (Q×K^T):    ~20% compute → systolic array (matrix multiply)
  Softmax:              ~2% compute  → AFU (streaming softmax unit)
  Deformable sampling:  ~8% compute  → DMA + interpolation unit
  LayerNorm:            ~3% compute  → AFU
  FFN (Linear+GELU):    ~7% compute  → systolic array + AFU

Compute: ~150-300 GFLOPs per frame (6 cameras)
Memory:  Temporal features need ~50MB buffer (previous frame's BEV)

Minimum viable tier: Mid (4 TOPS, 2MB SRAM) at ~15 fps with INT8
Comfortable tier: Pro (32 TOPS, 8MB SRAM) at ~100+ fps
```

### 3.3 Vision Transformer (ViT)

```
ViT-B/16 Architecture:
  Image → Patch Embedding (Conv 16×16) → 12× Transformer Blocks
  Each block: LayerNorm → Multi-Head Attention → LayerNorm → FFN

Key operations:
  Patch Embedding:  Single large Conv2D → systolic array
  Q/K/V Projection: Linear layers → systolic array
  Attention:        Q×K^T matrix multiply → systolic array
  Softmax:          Per-head softmax → AFU
  LayerNorm:        Mean + variance + normalize → AFU
  FFN:              Two linear layers + GELU → systolic array + AFU

Compute: 17.6 GFLOPs (ViT-B), 61.6 GFLOPs (ViT-L)
SRAM for attention: 768×197 ×2 × 12 heads ≈ 4.5 MB (ViT-B)

Minimum viable tier: Mid (4 TOPS) at ~230 fps for ViT-B INT8
```

### 3.4 LLaMA (Large Language Model — In-Cabin AI)

```
LLaMA-7B Architecture:
  32 Transformer layers × (Attention + FFN)
  Hidden dim: 4096, Heads: 32, FFN dim: 11008

Two phases:
  Prefill:  Process full prompt in parallel (compute-bound)
  Decode:   Generate one token at a time (memory-bound)

Key bottleneck: MEMORY BANDWIDTH, not compute
  Each token decode reads ~7GB of weights (INT8) or ~3.5GB (INT4)
  At 30 tokens/sec: need 3.5GB × 30 = 105 GB/s bandwidth

Memory interface requirements:
  LPDDR5-6400: ~51 GB/s → ~15 tokens/sec (INT4) — acceptable for voice
  HBM2e:       ~460 GB/s → ~130 tokens/sec (INT4) — fluid conversation

KV Cache per layer: 2 × seq_len × 4096 × 2 bytes = 16KB per token
  2048 token context: 32 layers × 2048 × 4096 × 2 = 512 MB
  → must be in external memory, streamed through SRAM

Minimum viable tier: Pro (32 TOPS, 8MB SRAM, DDR4) at ~5 tok/s INT4
Comfortable tier: Ultra (196 TOPS, 32MB SRAM, LPDDR5) at ~30 tok/s
```

---

## 4. Feasibility by Process Node

### 4.1 Sky130 (130nm) — Proof of Concept

| Parameter | Value | Notes |
|-----------|-------|-------|
| Array size | 16×16 | 256 MACs |
| Clock | 50 MHz | Conservative for sky130 |
| Throughput | 25.6 GOPS (0.026 TOPS) | INT8 |
| SRAM | 64 KB | OpenRAM, ~2.1 mm² |
| Memory interface | SPI/QSPI to external SRAM | ~10 MB/s, weight pre-load |
| NPU die area | ~15-20 mm² | Array ~8mm², SRAM ~2mm², control ~2mm², routing ~5mm² |
| Fusion pipeline | ~7 mm² | Existing 20 modules (from current OpenLane runs) |
| Total SoC | ~30-35 mm² | NPU + Fusion + base modules |
| Power | ~200-500 mW | Estimated at 1.8V, 130nm |
| Models | MobileNetV2-tiny, custom small CNNs | 3 fps on MobileNet |
| Tape-out cost | ~$10K (Efabless Chipignite) | Shuttle run |

**Feasibility: HIGH.** All components exist in open-source: OpenRAM for SRAM,
PicoRV32 for CPU, OpenLane for ASIC flow. The systolic array is a straightforward
extension of the existing `mac_array.v`. Main risk: SRAM yield on sky130.

**What it proves:**
- End-to-end AI inference on custom silicon
- NPU → fusion pipeline integration works in hardware
- Architecture is sound before committing to expensive advanced nodes

### 4.2 28nm — First Product (6×6mm die, 36 mm²)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Array size | 64×64 | 4,096 MACs per core |
| Clock | 500 MHz - 1 GHz | 28nm HPC process |
| Throughput | 4-8 TOPS (INT8), 8-16 (INT4) | With sparsity: up to 32 effective |
| SRAM | 4-8 MB | Foundry SRAM macros |
| Memory | DDR4-3200 | ~25 GB/s, sufficient for CNN models |
| **Die size** | **6×6 mm (36 mm²)** | Sized for 4-core expansion via metal respin |
| v1 utilization | ~38% | 1 NPU core, 13 mm² headroom |
| v2 (metal respin) | ~55% | 2 NPU cores, 16 TOPS |
| v3 (metal respin) | ~75% | 4 NPU cores, 32 TOPS |
| Power | 3-8 W (v1), 12-20 W (v3) | Automotive thermal budget |
| Models | YOLOv8 (real-time), ViT-B, BEVFormer-tiny | v3 adds LLaMA-7B INT4 |
| Tape-out cost | €30-50K MPW, $50-100K metal respin | Via Europractice |

**Die sizing rationale:** 6×6mm allows scaling from 4 TOPS (v1) to 32 TOPS (v3)
on a SINGLE die through metal-only respins ($50-100K each vs $500K full respin).
No tile link needed below 32 TOPS. Per-chip cost vs 5×5mm is only ~$0.60 extra.
Mobileye EyeQ4 was ~100mm² on the same node — AstraCore at 36mm² is modest.

**EDA toolchain:** Cadence (Genus + Innovus + Tempus + Voltus + JasperGold).

**Competitive position at 4-32 TOPS:**
- Mobileye EyeQ4 (2018): 2.5 TOPS, 28nm — AstraCore v1 already exceeds this
- Hailo-8 (2019): 26 TOPS, 16nm — AstraCore v3 matches on larger node
- AstraCore advantage: integrated ASIL-D HW safety + single-die scaling + lower cost

### 4.3 12nm / 7nm — Competitive Product

| Parameter | Value (12nm) | Value (7nm) |
|-----------|-------------|-------------|
| Array size | 128×128 | 128×128 × 4 cores |
| MACs | 16,384 | 65,536 |
| Clock | 1 GHz | 1.5 GHz |
| Throughput | 32.8 TOPS | 196 TOPS |
| INT4 throughput | 65.6 TOPS | 393 TOPS |
| SRAM | 8 MB | 32 MB |
| Memory | LPDDR5 | LPDDR5 or HBM2e |
| Models | BEVFormer, YOLOv8-X, ViT-L | + LLaMA-7B INT4, multi-model |
| Die area | ~80-120 mm² | ~200-350 mm² |
| Power | 15-25 W | 30-60 W |
| Tape-out cost | $5-15M | $30-80M |

**Feasibility: Requires significant funding ($20M+ for 12nm, $100M+ for 7nm).**
This is the tier where AstraCore competes directly with Mobileye EyeQ6,
NVIDIA Orin, and Qualcomm Ride.

### 4.4 Reaching 1250+ TOPS

To reach the 1250 TOPS target:

```
Option A: Brute force (single die)
  MACs needed at 2 GHz: 312,500 (INT8) or 156,250 (INT4 with 2x throughput)
  Array: 560×560 (INT8) or 395×395 (INT4)
  Node: 5nm or 4nm
  Die: ~400-600 mm²
  Cost: $50-100M tape-out
  Power: 80-150 W

Option B: Sparsity + structured pruning (effective TOPS)
  Real MACs: 65,536 (128×128 × 4 cores)
  Raw: 196 TOPS at 1.5 GHz
  2:4 structured sparsity: 2× effective throughput = 392 effective TOPS
  INT4 precision: another 2× = 784 effective TOPS
  With 60% model sparsity: 784 / 0.4 = 1960 effective TOPS
  This is how NVIDIA reports "2000 TOPS" on Thor

  Node: 7nm or 5nm
  Die: ~250-350 mm²
  Cost: $30-80M
  This is achievable and honest

Option C: Multi-chip module (chiplet)
  4× AstraCore dies on interposer, each 196 TOPS
  Total: 784 TOPS raw, ~1500+ effective with sparsity
  Node: 7nm per chiplet
  Cost per chiplet: $15-20M, interposer: $10M
  Lower risk than monolithic, scales naturally
```

**Recommendation:** Option B (sparsity-aware architecture on 7nm) is the
most realistic path to 1250+ effective TOPS. This requires adding:
- 2:4 structured sparsity support in the systolic array (skip zero weights)
- INT4 compute mode (pack two INT4 ops per INT8 MAC cycle)
- These are both ~200 lines of RTL changes to the PE, not architectural rewrites

---

## 5. Integration with Existing Fusion Pipeline

### 5.1 The Interface is Already Defined

The NPU connects to the fusion pipeline through the exact interface that
`astracore_fusion_top` already exposes for external CNN detections:

```verilog
// These ports ALREADY EXIST on astracore_fusion_top:
input  wire        cam_det_valid,
input  wire [15:0] cam_det_class_id,     // YOLO class ID
input  wire [15:0] cam_det_confidence,   // 0-1000 fixed-point
input  wire [15:0] cam_det_bbox_x,       // bounding box
input  wire [15:0] cam_det_bbox_y,
input  wire [15:0] cam_det_bbox_w,
input  wire [15:0] cam_det_bbox_h,
input  wire [31:0] cam_det_timestamp_us, // frame timestamp
input  wire [7:0]  cam_det_camera_id,    // which camera
```

The NPU's post-processing stage (YOLO NMS / BEV decoder) formats its
output into this struct and asserts `cam_det_valid`. The fusion pipeline
processes it identically whether the detection came from an external GPU
or the on-chip NPU.

### 5.2 What Changes in astracore_system_top

```verilog
// v1: external CNN feeds fusion directly
assign cam_det_valid = external_cam_det_valid;   // from off-chip

// v2: on-chip NPU feeds fusion
assign cam_det_valid = npu_det_valid;            // from npu_top
// (external port still available as fallback / debug)
```

This is a one-line wiring change in `astracore_system_top.v`.

### 5.3 Data Flow: Camera Frame to Safety Decision

```
Step 1: MIPI CSI-2 → mipi_csi2_rx (existing) → raw frame to SRAM
Step 2: DMA loads frame tile from SRAM → NPU scratchpad
Step 3: NPU runs YOLOv8 backbone (Conv + C2f blocks)
Step 4: NPU runs detection head → bounding boxes + class + confidence
Step 5: Post-processing (NMS) filters overlapping detections
Step 6: Detection structs written to cam_det_* interface
Step 7: Fusion pipeline takes over (sensor_sync → coord_transform →
        object_tracker → plausibility_checker → ttc_calculator →
        aeb_controller)
Step 8: Brake/steer/alert commands output on CAN-FD

End-to-end latency budget:
  MIPI capture:    2 ms (one frame at 30 fps = 33ms, pipelined)
  NPU inference:   5-15 ms (depends on model and tier)
  Fusion pipeline:  1-2 ms (fixed-function, deterministic)
  CAN-FD output:   <1 ms
  Total:           8-20 ms (well within 100ms ASIL-D budget)
```

---

## 6. NPU Module Inventory (New RTL)

| Module | Purpose | Est. Lines | Complexity |
|--------|---------|-----------|------------|
| `npu_pe.v` | Single processing element (MAC + weight reg + activation pass) | ~80 | Low |
| `npu_systolic_array.v` | NxN PE grid with data routing | ~300 | Medium |
| `npu_activation.v` | AFU: ReLU/GELU/SiLU/Softmax/LayerNorm | ~500 | Medium-High |
| `npu_sram_ctrl.v` | Scratchpad bank controller, double-buffer logic | ~250 | Medium |
| `npu_dma.v` | Tiled DMA engine with stride/padding | ~400 | Medium |
| `npu_tile_ctrl.v` | Microcode sequencer / tile scheduler | ~300 | Medium |
| `npu_postproc.v` | NMS / detection formatting → cam_det_t interface | ~200 | Medium |
| `npu_top.v` | Top-level integration + AXI interface | ~400 | Medium |
| **Total** | | **~2,400** | |

For comparison: the entire fusion pipeline is ~4,200 lines across 20 modules.
The NPU is comparable in RTL complexity but much larger in silicon area
(due to the systolic array and SRAM).

---

## 7. Memory Bandwidth Analysis

Memory bandwidth is the #1 performance limiter for AI inference, especially
for transformers and LLMs. The NPU architecture must be designed around it.

### 7.1 Bandwidth Requirements by Model

| Model | Weights | Activations | Total BW @ 30fps | Min Interface |
|-------|---------|-------------|-------------------|---------------|
| MobileNetV2 (INT8) | 3.4 MB | ~2 MB | 162 MB/s | SPI @ 40 MHz |
| YOLOv8-N (INT8) | 3.2 MB | ~8 MB | 336 MB/s | QSPI / parallel |
| YOLOv8-M (INT8) | 25.9 MB | ~30 MB | 1.7 GB/s | DDR4-1600 |
| ViT-B (INT8) | 86 MB | ~15 MB | 3.0 GB/s | DDR4-3200 |
| BEVFormer (INT8) | ~120 MB | ~200 MB | 9.6 GB/s | DDR4-3200 ×2 |
| LLaMA-7B (INT4) | 3.5 GB | ~50 MB/token | 105 GB/s @ 30tok/s | LPDDR5 / HBM |

### 7.2 On-Chip Data Reuse Strategy

The key to high performance with limited bandwidth is **maximizing data reuse**
— keeping data on-chip as long as possible.

```
Data reuse hierarchy:
  1. Weight reuse (weight-stationary): Load weights once, reuse across
     all spatial positions in a feature map. Reuse factor = H×W.
     For 640×640 YOLO input: weight reuse = 40×40 = 1600×

  2. Activation reuse (row-stationary): Each activation is used by
     multiple output channels. Reuse factor = C_out.
     For 256-channel conv: activation reuse = 256×

  3. Output reuse (tiling): Accumulate partial sums on-chip across
     the K dimension. Only write final result to memory.

Combined effect: Actual bandwidth ≈ Theoretical / (weight_reuse × act_reuse)
  YOLOv8-N: 8.7 GFLOPS theoretical, but with reuse: ~50 MB/s actual BW
  → Even SPI can handle the demo tier!
```

---

## 8. Power Analysis

| Component | Sky130 (130nm) | 28nm | 7nm |
|-----------|---------------|------|-----|
| Systolic array (dynamic) | 100-300 mW | 1-3 W | 5-15 W |
| SRAM leakage | 50-100 mW | 200-500 mW | 1-3 W |
| SRAM dynamic | 30-50 mW | 100-300 mW | 500 mW-2 W |
| DMA + control | 10-30 mW | 50-100 mW | 200-500 mW |
| Memory PHY (DDR) | N/A | 500 mW-1 W | 1-2 W |
| Fusion pipeline | 50-100 mW | 20-50 mW | 5-15 mW |
| RISC-V CPU | 20-50 mW | 10-30 mW | 5-10 mW |
| **Total SoC** | **300-700 mW** | **2-5 W** | **8-25 W** |
| | | | |
| TOPS/W | 0.04-0.09 | 0.8-4.0 | 8-25 |

**Automotive thermal budget:** Typically 5-15W for ADAS processors (fanless,
under-hood mounting up to 105C ambient). The 28nm tier fits comfortably.
The 7nm tier needs careful thermal design but is within production norms
(NVIDIA Orin runs at 40W with active cooling).

---

## 9. Software Toolchain (Critical Path)

Hardware without software is a paperweight. The NPU needs a compilation
flow that converts trained models to executable tile schedules.

### 9.1 Compilation Pipeline

```
PyTorch Model (.pt)
       │
       ▼
ONNX Export (.onnx)
       │
       ▼
Graph Optimizer (operator fusion, constant folding)
       │
       ▼
Quantizer (FP32 → INT8/INT4 with calibration dataset)
       │
       ▼
Tiler (partition each layer into tiles that fit in SRAM)
       │
       ▼
Scheduler (order tiles to maximize data reuse, minimize DMA stalls)
       │
       ▼
Code Generator (emit tile instructions / DMA descriptors)
       │
       ▼
Binary (.acbin) — loaded into NPU microcode ROM or CPU firmware
```

### 9.2 Build vs. License

| Component | Build In-House | License/Adapt Open-Source |
|-----------|---------------|-------------------------|
| ONNX parser | Use existing `onnx` Python lib | onnxruntime |
| Quantizer | Use existing tools | ONNX quantization toolkit |
| Graph optimizer | Medium effort (~2 months) | TVM / MLIR (Apache 2.0) |
| Tiler + Scheduler | Core differentiator — build | ~ |
| Code generator | Specific to our ISA — build | ~ |
| Runtime (on-chip) | Simple firmware — build | ~ |

**Recommendation:** Use TVM as the graph-level optimizer and build the
tiler/scheduler/codegen as a TVM backend. This is ~3-6 months of
software engineering for a 2-person team.

---

## 10. Competitive Landscape

| Product | TOPS | Node | Safety | AI Models | NPU + Fusion | Price |
|---------|------|------|--------|-----------|-------------|-------|
| Mobileye EyeQ4 | 2.5 | 28nm | ASIL-B (SW) | CNN only | No — separate chips | ~$50 |
| Mobileye EyeQ6H | 34 | 7nm | ASIL-D (SW) | CNN + RNN | Partial (SW fusion) | ~$100 |
| NVIDIA Orin | 275 | 8nm | ASIL-D (SW) | Any (GPU) | No — SW stack | ~$250 |
| NVIDIA Thor | 2000 | 4nm | ASIL-D (SW) | Any (GPU+DLA) | No — SW stack | ~$500 |
| Tesla FSD | 144 | 14nm | Proprietary | CNN (custom) | HW fusion (custom) | N/A |
| Hailo-8 | 26 | 16nm | None | CNN only | No fusion | ~$30 |
| **AstraCore v2 (28nm)** | **4-8** | **28nm** | **ASIL-D (HW)** | **CNN+ViT** | **Yes — HW fusion** | **~$20-40** |
| **AstraCore v2 (7nm)** | **50-200** | **7nm** | **ASIL-D (HW)** | **Any** | **Yes — HW fusion** | **~$80-150** |

### 10.1 AstraCore's Unfair Advantage

1. **Hardware-verified ASIL-D fusion** — competitors do safety in software
   (harder to certify, higher latency, more power). AstraCore's 20-module
   fusion pipeline is fixed-function, deterministic, formally verifiable.

2. **Deterministic worst-case latency** — the fusion pipeline has bounded,
   predictable latency (no OS jitter, no cache misses, no context switches).
   This is critical for ASIL-D timing guarantees.

3. **Lower cost at entry tier** — a 28nm chip with 4-8 TOPS + HW safety is
   viable at $20-40 BOM. Competitors either lack safety (Hailo) or cost
   5-10× more (Mobileye, NVIDIA).

4. **Open architecture** — OEMs can inspect, modify, and extend the RTL.
   No vendor lock-in. This matters for OEMs pursuing functional safety
   certification (they need full design visibility).

---

## 11. Phased Development Roadmap

### Phase 1: Architecture Proof (sky130) — 3-6 months

**Goal:** Tape out a working NPU + fusion SoC on sky130 via Chipignite.

| Deliverable | Description | Timeline |
|-------------|-------------|----------|
| `npu_pe.v` | Single PE with INT8 MAC + weight register | Week 1-2 |
| `npu_systolic_array.v` | 16×16 PE grid | Week 2-4 |
| `npu_activation.v` | ReLU + quantize (GELU deferred) | Week 4-6 |
| `npu_sram_ctrl.v` | 64KB OpenRAM interface + banking | Week 6-8 |
| `npu_dma.v` | Simple linear DMA (tiling deferred) | Week 8-10 |
| `npu_tile_ctrl.v` | Microcode ROM sequencer | Week 10-12 |
| `npu_postproc.v` | YOLO output → cam_det_t formatter | Week 12-14 |
| `npu_top.v` | Integration + cocotb verification | Week 14-16 |
| System integration | Wire into astracore_system_top | Week 16-18 |
| OpenLane ASIC flow | Synthesis → P&R → GDS-II | Week 18-22 |
| Tape-out | Chipignite shuttle submission | Week 22-24 |

**Cost:** ~$10K tape-out + compute resources for simulation.
**Team:** 1-2 engineers.

### Phase 2: First Product (28nm) — 12-18 months

**Goal:** Production-grade 4-8 TOPS ADAS SoC.

| Deliverable | Description |
|-------------|-------------|
| Scale array to 64×64 | 4,096 MACs |
| Full AFU | GELU, Softmax, LayerNorm |
| 2MB SRAM | Foundry macros |
| DDR4 controller | Licensed PHY + open controller |
| RISC-V CPU | PicoRV32 or similar |
| AXI4 interconnect | Open-source crossbar |
| TVM compiler backend | Model → tile schedule |
| ISO 26262 pre-certification | FMEDA, DFA, safety manual |

**Cost:** $2-5M (tape-out $500K-2M, tools $500K, team $1-2M).
**Team:** 5-8 engineers.

### Phase 3: Scale (7nm) — 24-36 months

**Goal:** 50-200 TOPS competitive ADAS SoC.

| Deliverable | Description |
|-------------|-------------|
| Multi-core NPU | 4× 128×128 systolic cores |
| Sparsity engine | 2:4 structured sparsity |
| INT4 mode | 2× effective throughput |
| 32MB SRAM | Multi-bank with NoC |
| LPDDR5 controller | High-bandwidth memory |
| Full compiler | TVM backend + auto-tiling |
| ISO 26262 ASIL-D certification | Full qualification |

**Cost:** $30-80M.
**Team:** 20-40 engineers.
**Funding:** Series A/B ($50-100M raise).

---

## 12. Post-Silicon Configurability & Runtime Flexibility

A chip that can only run one configuration is a chip that needs a respin
for every new customer, vehicle, or sensor vendor. AstraCore v2 is designed
to be deeply configurable AFTER fabrication — the same silicon adapts to
different sensor suites, safety thresholds, AI models, and vehicle platforms.

### 12.1 Fusion Pipeline Configurability (AXI Register-Mapped)

Every safety-critical threshold and decision boundary in the fusion pipeline
is promoted from hardcoded `parameter`/`localparam` to **runtime-writable AXI
registers with power-on defaults**. A companion MCU or the on-chip RISC-V
programs these at boot time. No respin needed for per-vehicle calibration.

**Configurable thresholds by module:**

| Module | Configurable Parameters | Default | Why Configurable |
|--------|------------------------|---------|------------------|
| `dms_fusion` | PERCLOS_DROWSY_THRESH, PERCLOS_CRIT_THRESH, CLOSED_CRIT_FRAMES, DISTRACTED_CRIT_FRAMES, WATCHDOG_CYCLES | 20%/50%/2s/3s/200ms | Different driver monitoring standards per region (EU vs US vs China) |
| `object_tracker` | GATE_THRESH (association distance), AGE_PRUNE (track timeout), NUM_TRACKS | 5m/10frames/8 | Different sensor FOV and density → different tracking parameters |
| `ttc_calculator` | WARN_MS, PREP_MS, BRAKE_MS (TTC thresholds) | 3000/1500/700 | OEM-specific braking strategy and driver warning timing |
| `plausibility_checker` | Sensor redundancy rules (which sensors required per class) | CAM+RAD for vehicles | Sensor suite varies by vehicle trim level |
| `safe_state_controller` | ALERT_TICKS, DEGRADE_TICKS, MRC_TICKS (escalation timing) | Per ASIL-D spec | Different MRC strategies (pull-over vs slow-stop) |
| `aeb_controller` | MIN_BRAKE_MS, CLEAR_TICKS | 500ms/100 | Brake system response time varies by vehicle |
| `ldw_lka_controller` | WARN_OFFSET_MM, INTERVENE_OFFSET_MM, K_GAIN | 600/900mm | Lane width and steering feel vary by vehicle class |
| `sensor_sync` | FUSION_WINDOW_US, STALE_TIMEOUT_US | 100us/200ms | Sensor timing characteristics vary by vendor |
| `coord_transform` | 3×3 rotation + 3×1 translation per sensor (calibration regs) | Identity | Each vehicle has different sensor mounting positions |

**Register map architecture:**
```
AXI-Lite Address Space (0x1000 - 0x1FFF): Fusion Config Registers
  0x1000: dms_fusion config (thresholds, watchdog)
  0x1020: object_tracker config (gate, pruning, track count)
  0x1040: ttc_calculator config (warning/prep/brake thresholds)
  0x1060: plausibility_checker config (sensor mask rules)
  0x1080: safe_state_controller config (escalation timing)
  0x10A0: aeb_controller config (brake hold, clear timing)
  0x10C0: ldw_lka_controller config (offsets, gain)
  0x10E0: sensor_sync config (window, stale timeout)
  0x1100-0x11FF: coord_transform calibration matrices (per sensor)

All registers have hardware-reset defaults matching current hardcoded values.
Boot sequence: RISC-V reads vehicle config from flash → writes registers → 
releases fusion pipeline from reset.
```

### 12.2 Sensor Profile Register Bank

Different sensor vendors output different byte formats, even for the same
sensor type. The sensor profile register bank lets the same silicon work
with any sensor without RTL changes.

```
Per-sensor-input profile (programmed at boot):
  ┌─────────────────────────────────────────────────┐
  │  header_length    [7:0]   — bytes before payload │
  │  payload_offset   [7:0]   — first data byte      │
  │  endianness       [0:0]   — 0=big, 1=little      │
  │  valid_polarity   [0:0]   — 0=active-high        │
  │  frame_length     [15:0]  — expected frame size   │
  │  checksum_type    [1:0]   — none/XOR/CRC8/CRC16  │
  │  checksum_offset  [7:0]   — where checksum lives  │
  └─────────────────────────────────────────────────┘

Affected modules: imu_interface, radar_interface, ultrasonic_interface,
                  lidar_interface, cam_detection_receiver

The Sensor Abstraction and Conditioning Layer (SACL) on the companion
MCU translates vendor-specific protocols to AstraCore's input format.
The profile registers tell the hardware what to expect.
```

### 12.3 NPU Runtime Reconfigurability

The NPU is inherently programmable — it runs whatever model you load.
But beyond model selection, the hardware supports runtime flexibility:

**a) Model hot-swap (no reset required):**
```
Running YOLOv8 for highway driving
  → Enter urban zone
  → RISC-V loads pedestrian-optimized model weights via DMA
  → NPU switches models in <10ms (one DMA transfer)
  → No pipeline stall — old model runs until new weights are loaded

Use case: Highway model (fast, fewer classes) vs City model (pedestrian-focused)
```

**b) Runtime precision switching:**
```
Register: NPU_PRECISION_CTRL
  [1:0] = 00: INT8×INT8 (default, max accuracy)
  [1:0] = 01: INT4×INT4 (2× throughput, lower accuracy)
  [1:0] = 10: INT8×INT4 (mixed, compromise)
  [1:0] = 11: FP16×FP16 (highest accuracy, lowest throughput)

Use case: Switch to INT4 when thermal throttling or power-constrained
          Switch to FP16 for safety-critical scenarios (school zone)
```

**c) Power gating unused compute:**
```
Register: NPU_POWER_CTRL
  [3:0] core_enable  — bitmask, enable/disable individual systolic cores
  [4]   afu_enable   — gate activation unit when running ReLU-only models
  [5]   dma_idle_gate — auto-gate DMA clock when idle

Use case: Parking mode uses 1 core (low power), highway uses all 4
```

**d) Configurable post-processing:**
```
Register: NPU_POSTPROC_CTRL
  [7:0]   nms_iou_thresh    — IoU threshold for NMS (0-255 → 0.0-1.0)
  [15:8]  confidence_thresh — minimum detection confidence
  [23:16] max_detections    — cap output count
  [31:24] class_filter_mask — which classes to output (pedestrian/vehicle/etc)

Use case: Night mode raises confidence threshold to reduce false positives
          Parking mode enables only proximity classes
```

### 12.4 BIST Controller (Built-In Self-Test)

ASIL-D requires periodic self-test of safety-critical logic. The BIST
controller is a small state machine that validates the entire fusion
pipeline at every ignition cycle.

```
BIST Sequence (triggered via AXI register write):
  1. Isolate fusion pipeline from live sensor inputs (mux to BIST source)
  2. Inject known stimulus pattern:
     - Camera detection: vehicle at 50m, confidence 800, camera_id 0
     - Radar object: range 5000cm, velocity -1000cm/s
     - IMU: zero rotation, 1g downward
  3. Clock pipeline for N cycles (deterministic)
  4. Compare outputs against hardcoded expected values:
     - object_tracker: track_id=0, x=50000mm, class=vehicle
     - ttc_calculator: TTC warning asserted
     - aeb_controller: brake_level >= PRECHARGE
  5. Assert BIST_PASS or BIST_FAIL flag
  6. Reconnect live sensor inputs

Duration: ~1000 cycles = 20 us @ 50 MHz
Area: ~200 cells (~2500 um^2 on sky130)
Runs: every ignition cycle + periodic (configurable interval)
```

### 12.5 Debug / Trace Port

An 8-bit + valid output port muxed to observe any internal fusion signal,
selected by a register. Essential for bench bring-up and field diagnosis.

```
Register: DEBUG_MUX_SEL [7:0]
  0x00: sensor_sync.window_release + sensor_valid[3:0]
  0x01: coord_transform pipeline stage (pre/post)
  0x02: object_tracker.match_found + track_id[2:0] + confidence
  0x03: plausibility_checker.asil_level + sensor_mask
  0x04: ttc_calculator.warn/prep/brake flags + active_track
  0x05: aeb_controller.state + brake_level
  0x06: safe_state_controller.state + fault_vector
  0x07: dms_fusion.driver_attention_level + confidence
  0x08-0x0F: NPU internal (tile_state, DMA progress, SRAM bank)
  0x10-0x1F: Raw sensor interface signals

Output: debug_data[7:0], debug_valid (directly to chip pad)
Can be captured by logic analyzer or routed to JTAG scan chain.
```

### 12.6 Configurable Fusion Routing Matrix

A register-programmable crossbar between Layer 2 outputs and Layer 3
inputs. Adapts to different sensor suites without respin.

```
Default routing (full sensor suite):
  coord_transform → object_tracker → ttc_calculator → aeb_controller
  coord_transform → object_tracker → lane_fusion → ldw_lka_controller

Reduced routing (camera + radar only, no LiDAR):
  coord_transform → object_tracker → ttc_calculator → aeb_controller
  (lane_fusion bypassed, ldw_lka disabled)

Routing register: FUSION_ROUTE_CTRL [15:0]
  [0] enable_radar_path
  [1] enable_lidar_path
  [2] enable_ultrasonic_path
  [3] enable_camera_path
  [4] enable_dms_path
  [5] enable_lane_fusion
  [6] enable_ldw_lka
  [7] force_safe_state (manual override)
  [8] bypass_plausibility (debug only, never in production)
  [15:9] reserved

Hardware: MUX gates on valid signals at each routing point.
When a path is disabled, its sensor_mask bit is cleared in
plausibility_checker, preventing false ASIL violations.
```

### 12.7 Chip Infrastructure for Flexibility

| Feature | Purpose | Area |
|---------|---------|------|
| **Chip ID block** | 128-bit read-only register (lot/wafer/die fields) for traceability and anti-cloning | ~100 cells |
| **OTP fuse block** | Per-die feature enable/disable — lock out NPU on safety-only SKUs, enable/disable interfaces | ~500 cells |
| **Chip-to-chip link** | Source-synchronous parallel interface for multi-die lockstep (redundant safety) or split compute (NPU on one die, fusion on another) | ~2000 cells |

**OTP SKU management:**
```
Fuse map:
  [0]    npu_enable        — gate entire NPU subsystem
  [1]    lidar_enable      — gate LiDAR interface
  [2]    pcie_enable       — gate PCIe controller
  [3]    dms_enable        — gate DMS camera path
  [7:4]  max_tracks        — limit object_tracker entries (cost tiering)
  [15:8] chip_revision     — hardcoded at fab
  [127:16] chip_serial     — unique per die

Same mask set → different products:
  AstraCore Lite:  NPU disabled, 4 tracks, no LiDAR     ($10)
  AstraCore Pro:   NPU enabled, 8 tracks, full sensor    ($30)
  AstraCore Ultra: NPU enabled, 128 tracks, multi-die    ($80)
```

### 12.7.1 I/O Pin Budget (28nm, 6×6mm die, ~380 usable pads)

```
ALLOCATED (v1 product):                                    ~218 pads
  Power/ground:      60    (VDD/VSS/VDDIO, many for IR drop)
  Clock/reset/PLL:    4
  Sensor interfaces: 31    (MIPI, CAN, SPI×2, ETH, UART, GNSS)
  Safety outputs:    12    (brake/steer CAN, safe_state, DMS, IRQ)
  DDR4 interface:    80    (32-bit data bus + addr + ctrl)
  SPI flash:          4    (boot + model weight storage)
  JTAG + debug:       7
  AXI external:      20    (host CPU or external bus master)

RESERVED FOR FUTURE (on die, not all bonded in v1):        ~102 pads
  Tile link:         22    (ASIC-to-ASIC scaling or ASIC-to-FPGA data)
  FPGA companion:    16    (SPI + GPIO + IRQ for simple FPGA connection)
  Second camera:      6    (MIPI 2-lane for rear/side camera)
  Second CAN:         4    (V2X / fleet telemetry, non-safety)
  I2C general:        2    (PMIC, EEPROM, misc sensors)
  Analog test:        4    (temp sensor, voltage monitor, ATE test)
  General GPIO:       8    (RISC-V controlled, OEM-assignable)
  Unbonded spare:    40    (on die but not in v1 package — available in v2)

PACKAGING STRATEGY:
  v1: QFN/BGA-280  — bonds 280 pads, 40 unbonded (lowest cost package)
  v2: BGA-320      — bonds all 320 pads (same die, bigger package)
                     Enables: 2nd tile link, 2nd camera, all GPIO
                     NO die respin needed — only package change
```

### 12.7.2 Power Scaling Provisions

```
VOLTAGE/FREQUENCY SCALING:
  PLL supports 3 modes (register-selectable):
    Eco mode:    300 MHz  (40% less power, parking/idle)
    Normal mode: 500 MHz  (nominal)
    Boost mode:  650 MHz  (15% more compute, 30% more power)
  
  Register: CLK_CTRL [1:0] freq_select, [2] pll_bypass

POWER DOMAIN GATING (on 28nm, foundry header/footer cells):
  Always-on:   fusion pipeline + safety + clock/reset (never gated)
  Gatable:     NPU cores (per-core power switch)
  Gatable:     DDR controller (sleep when not inferencing)
  Gatable:     non-critical sensor interfaces

DYNAMIC VOLTAGE SCALING (SoM-level):
  ASIC requests voltage via 2-pin interface to external PMIC:
    1.0V nominal:  full speed, full power
    0.9V reduced:  ~80% speed, ~65% power
    0.8V eco:      ~60% speed, ~40% power
```

### 12.8 Configurability Summary

```
Layer 0 (Fab time):     OTP fuses → SKU, chip ID, feature gates
Layer 1 (Boot time):    Flash → thresholds, calibration, sensor profiles, model weights
Layer 2 (Runtime):      AXI registers → precision, power gating, routing, debug mux
Layer 3 (Model swap):   DMA → new model weights, hot-swap without reset
Layer 4 (Field update): OTA → new firmware (RISC-V) + new model weights → CAN-FD/Ethernet

No respin needed for:
  ✓ Different sensor vendors (sensor profile registers)
  ✓ Different vehicle platforms (calibration matrices + thresholds)
  ✓ Different safety standards (threshold registers)
  ✓ Different AI models (weight loading + microcode)
  ✓ Different product tiers (OTP fuse SKU management)
  ✓ Field improvements (OTA model + firmware updates)
  ✓ Debug and diagnosis (trace port + BIST)
```

---

## 13. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SRAM yield on sky130 | Medium | High | Use conservative bitcell, add redundancy |
| Timing closure on large systolic array | Medium | Medium | Pipeline aggressively, accept lower clock |
| DDR4 PHY integration (28nm) | Medium | High | License proven IP, budget 6 months |
| Model accuracy loss from INT8 quantization | Low | Medium | Quantization-aware training, INT4 fallback |
| Compiler complexity underestimated | High | High | Start with fixed model support, generalize later |
| Competitor releases at lower price point | Medium | Medium | Differentiate on safety (HW ASIL-D) |
| Automotive qualification timeline | High | Medium | Start FMEDA early, engage safety consultancy |

---

## 14. Future-Proofing: Designing for 2028-2030

Silicon takes 18-24 months from design to production. Whatever we build
today ships into a world where the AI landscape has moved on. This section
identifies the trends that matter and how the architecture stays relevant.

### 14.1 Trend: End-to-End Neural Driving

**What's coming:** Instead of separate detect → track → plan steps, one
neural network goes directly from camera pixels to steering/brake commands.
Tesla FSD v12+ already does this. By 2028, most L2+ systems will use some
form of end-to-end model.

**Impact on AstraCore:**
```
Naive reaction:    "The fusion pipeline is obsolete!"
Correct reaction:  "The fusion pipeline becomes the SAFETY SUPERVISOR."

End-to-end model:  Camera → Neural Net → "steer left 5°, brake 20%"
                                              │
                                              ▼
                    Fusion pipeline:  "Is this command sane?"
                                     → radar confirms obstacle? ✓
                                     → TTC still safe? ✓
                                     → steering within limits? ✓
                                     → ALLOW command

No regulator will certify a pure neural net for ASIL-D braking
by 2028. The hardware fusion pipeline is the WATCHDOG that catches
when the AI hallucinates. This makes it MORE valuable, not less.
```

**Design implication:** Add a `neural_command_input` port to the fusion
pipeline that accepts steering/brake commands from the NPU's end-to-end
model, validates them against sensor reality, and either passes them
through or overrides to safe state. ~100 lines of new RTL.

### 14.2 Trend: World Models and Prediction

**What's coming:** Models that don't just detect what IS in the scene but
predict what WILL BE in 2-5 seconds. "That pedestrian is walking toward
the road and will be in my lane in 3 seconds."

**Impact on AstraCore:**
```
Current ttc_calculator:  "Object is X meters away, closing at Y m/s"
                         → simple linear projection

Future world model:      "Object will be at position Z in 3 seconds
                          with 85% probability, considering their
                          walking direction and the crosswalk"
                         → learned prediction, much more accurate
```

**Design implication:** The NPU already handles this — world models are
just larger neural networks. The key requirement is **more SRAM** to hold
temporal state (what the scene looked like in previous frames). Design
the SRAM controller to support a **temporal buffer ring** — a circular
buffer of past BEV features that the model can attend to.

### 14.3 Trend: State Space Models (Mamba) Replacing Transformers

**What's coming:** Transformers scale quadratically with sequence length
(O(n²) attention). State Space Models (Mamba, RWKV, RetNet) scale linearly
(O(n)). By 2028 they may dominate for sequence tasks.

**Impact on AstraCore NPU:**
```
Transformer attention:  Q × K^T → Softmax → × V
  → needs: matrix multiply + softmax + matrix multiply
  → our systolic array + AFU handles this ✓

State Space Model:      x(t) = A·x(t-1) + B·u(t),  y(t) = C·x(t)
  → needs: matrix multiply + element-wise multiply + addition
  → our systolic array handles the matmul ✓
  → element-wise ops need a SIMD lane (NEW)
```

**Design implication:** Add a **vector ALU lane** alongside the systolic
array. Simple element-wise operations (multiply, add, compare) on vectors.
~200 lines of RTL. This also accelerates:
- Residual connections (add two tensors)
- Skip connections
- Feature concatenation
- Any future architecture that mixes matmul with element-wise ops

### 14.4 Trend: Extreme Quantization (INT4, INT2, Binary)

**What's coming:** By 2028, INT4 inference is standard. INT2 and even
binary (1-bit) neural networks are viable for edge deployment with
acceptable accuracy loss.

**Impact on AstraCore NPU:**
```
INT8 MAC:    a[7:0] × b[7:0] → 1 operation per cycle
INT4 MAC:    pack 2 INT4 pairs into one INT8 MAC → 2× throughput
INT2 MAC:    pack 4 INT2 pairs → 4× throughput
Binary:      XNOR + popcount → 8× throughput (different circuit)

Effective TOPS scaling (same hardware):
  INT8:    4 TOPS
  INT4:    8 TOPS (2×)
  INT2:   16 TOPS (4×)
  Binary: 32 TOPS (8×) ← free performance from the same silicon
```

**Design implication:** The PE must support sub-byte packing from day one.
Don't build an INT8-only MAC and try to retrofit INT4 later — design the
PE with a configurable precision splitter:

```verilog
// Future-proof PE: one INT8 MAC or two INT4 MACs or four INT2 MACs
always @(*) begin
  case (precision_mode)
    2'b00: result = a_i8 * b_i8;                          // 1× INT8
    2'b01: result = {a_i4_hi*b_i4_hi, a_i4_lo*b_i4_lo};  // 2× INT4
    2'b10: result = {a_i2[3]*b_i2[3], a_i2[2]*b_i2[2],   // 4× INT2
                     a_i2[1]*b_i2[1], a_i2[0]*b_i2[0]};
    2'b11: result = popcount(~(a_bin ^ b_bin));            // 8× Binary
  endcase
end
```

**This is critical.** If you only build INT8 today, your chip is 2× slower
than competitors by 2028. If you build INT4+INT2 support, the same
silicon stays competitive for years.

### 14.5 Trend: Sparsity is Free Performance

**What's coming:** Modern models are 50-80% zeros after pruning. Hardware
that can skip zero-weight multiplications gets 2-4× speedup for free.

**Two types:**
```
Unstructured sparsity:  Random zeros scattered everywhere
  → Hard to exploit in hardware (irregular memory access)
  → Skip: not worth the complexity

Structured 2:4 sparsity (NVIDIA Ampere+):
  For every 4 weights, exactly 2 are zero
  → Hardware stores only the 2 non-zero values + 2-bit index
  → 2× throughput, 50% memory savings
  → NVIDIA trains models this way; by 2028 it's standard

  Example:
    Dense:    [0.5, 0, 0.3, 0] [0, 0.7, 0, 0.1]
    Stored:   [0.5, 0.3, idx=0b01] [0.7, 0.1, idx=0b10]
    → Only multiply the non-zero pairs → 2× speed
```

**Design implication:** Add a **sparse index decoder** before the systolic
array input. ~150 lines of RTL. The decoder reads the 2-bit index per
pair and routes the non-zero values to the correct PE inputs. When
sparsity is disabled, it passes data through unchanged.

### 14.6 Trend: Multi-Modal Fusion in Neural Networks

**What's coming:** Instead of processing each sensor separately and then
fusing in hardware (our current approach), one neural network takes ALL
sensor data simultaneously — camera + radar + LiDAR → single model.
Examples: BEVFusion, TransFusion, UniAD.

**Impact:**
```
Current:  Camera → NPU → detections ─┐
          Radar  → HW interface ──────┤→ HW fusion → decisions
          LiDAR  → HW interface ──────┘

Future:   Camera ─┐
          Radar  ─┤→ NPU (unified model) → commands → HW safety check
          LiDAR  ─┘
```

**Design implication:** The NPU needs **multi-input DMA channels** — load
camera features AND radar points AND LiDAR points into different SRAM
banks simultaneously, so the neural network can attend to all modalities
at once. This is a DMA enhancement (~100 lines), not an architectural
change.

The hardware fusion pipeline doesn't disappear — it becomes the safety
validator that checks the neural network's decisions are physically
plausible (radar confirms what the neural net claims to see).

### 14.7 Architecture Decisions That Stay Relevant in 2030

| Decision | Why it survives |
|----------|----------------|
| Systolic array (matrix multiply) | Every AI architecture = matmul at core |
| Weight-stationary dataflow | Memory bandwidth stays the bottleneck |
| Multi-precision PE (INT8/4/2) | Quantization only gets more aggressive |
| Large SRAM scratchpad | Models get bigger, on-chip reuse matters more |
| Hardware fusion pipeline | Regulators won't certify neural nets for ASIL-D |
| Register-configurable thresholds | Standards and OEM requirements keep changing |
| OTA model update | Models improve monthly, silicon lives 10+ years |
| Sparsity support | 2:4 sparsity is already industry standard (NVIDIA) |

### 14.8 What to build NOW vs LATER

```
BUILD NOW (sky130 demo — proves architecture):
  ✓ Multi-precision PE (INT8 + INT4 at minimum)
  ✓ Systolic array with weight-stationary dataflow
  ✓ Double-buffered SRAM with DMA
  ✓ Activation unit (ReLU + GELU + Softmax)
  ✓ Fusion pipeline safety supervisor mode
  ✓ Configurable everything via registers

ADD FOR 28nm PRODUCT:
  + Structured 2:4 sparsity decoder
  + Vector ALU lane (element-wise ops for SSMs)
  + Multi-input DMA (multi-modal fusion models)
  + Temporal buffer ring (world models)
  + INT2 / binary precision mode
  + DDR4/LPDDR5 controller

DEFER TO 7nm:
  - Multi-core NPU (just replicate the proven single core)
  - HBM controller
  - Chiplet interface
  - Hardware-accelerated NMS / post-processing
```

---

## 15. Key Metrics to Track

| Metric | Demo (sky130) | Product (28nm) | Scale (7nm) |
|--------|--------------|----------------|-------------|
| TOPS (INT8) | 0.026 | 4-8 | 50-200 |
| TOPS/W | 0.05 | 1-4 | 8-25 |
| TOPS/$ (die cost) | 0.005 | 0.2-0.8 | 0.5-2.0 |
| Frames/sec (YOLOv8-N) | 3 | 460+ | >1000 |
| Frames/sec (BEVFormer) | N/A | 10-15 | 50-100 |
| Fusion latency (worst case) | 2 ms | 2 ms | 2 ms |
| End-to-end latency | 50 ms | 12 ms | 5 ms |
| Power | 0.5 W | 3-5 W | 15-25 W |
| Die area | 30 mm² | 25 mm² | 200 mm² |
| Safety level | ASIL-D (HW) | ASIL-D (HW) | ASIL-D (HW) |

---

## Appendix A: Existing AstraCore v1 Modules (Unchanged in v2)

All 20 sensor fusion modules carry over unchanged. The NPU is additive.

**Layer 1 — Sensor Interfaces (10 modules):**
mipi_csi2_rx, imu_interface, gnss_interface, ptp_clock_sync,
can_odometry_decoder, radar_interface, ultrasonic_interface,
cam_detection_receiver, lidar_interface, det_arbiter

**Layer 2 — Fusion Processing (6 modules):**
sensor_sync, coord_transform, ego_motion_estimator, object_tracker,
lane_fusion, plausibility_checker

**Layer 3 — Decision/Output (4 modules):**
ttc_calculator, aeb_controller, ldw_lka_controller, safe_state_controller

**Base Infrastructure (11 modules):**
gaze_tracker, head_pose_tracker, dms_fusion, canfd_controller,
ethernet_controller, mac_array, inference_runtime, tmr_voter,
ecc_secded, fault_predictor, thermal_zone, pcie_controller

**Total proven RTL: 31 modules, ~200+ cocotb tests, all passing.**

---

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| TOPS | Tera Operations Per Second (10^12 INT8 MACs × 2) |
| Systolic Array | 2D grid of MACs where data flows through in a regular pattern |
| Weight-Stationary | Dataflow where weights stay in PEs, activations stream through |
| BEVFormer | Bird's Eye View Transformer — SOTA multi-camera 3D perception |
| ViT | Vision Transformer — applies transformer architecture to images |
| NMS | Non-Maximum Suppression — filters overlapping bounding boxes |
| KV Cache | Key-Value cache — stores attention state for autoregressive LLMs |
| OpenRAM | Open-source SRAM compiler for custom memory macros |
| FMEDA | Failure Modes, Effects, and Diagnostic Analysis (ISO 26262) |
| DFA | Dependent Failure Analysis (ISO 26262) |
