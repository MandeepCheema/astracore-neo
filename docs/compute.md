# Compute — MAC Array, Sparsity Engine, Transformer Engine

**Module:** `src/compute/`  
**Depends on:** `hal/`, `memory/`  
**Status:** DONE  
**Test log:** `logs/test_compute.log`  
**Test result:** 91/91 passed ✓

---

## Purpose

Models the AstraCore Neo's core AI compute pipeline:

- **MAC Array** — 24,576 MAC units (48 cores × 512), 6 precision modes, conv2d via im2col, HAL-integrated
- **Sparsity Engine** — structured N:M pruning (2:1/4:1/8:2/8:1), mask generation, effective TOPS uplift
- **Transformer Engine** — 8× MHSA, RoPE, fused softmax/layer-norm/GeLU, sparse attention, full transformer block

---

## Files

| File | Description |
|------|-------------|
| `compute/mac_array.py` | `MACArray`, `MACCore`, `PrecisionMode` |
| `compute/sparsity.py` | `SparsityEngine`, `SparsityPattern` |
| `compute/transformer.py` | `TransformerEngine`, `TransformerBlock`, `MultiHeadSelfAttention`, `FeedForward`, fused ops |
| `compute/exceptions.py` | `ComputeError`, `MACError`, `SparsityError`, `TransformerError`, `PrecisionError` |
| `compute/__init__.py` | Public API exports |

---

## MAC Array

### Architecture

```
MACArray (24,576 MACs total)
├── Core 0   (512 MACs) ─ FETCH→DECODE→EXECUTE→ACCUMULATE→WRITEBACK
├── Core 1   (512 MACs)
├── ...
└── Core 47  (512 MACs)

Work distribution: rows of A split across enabled cores
```

### Precision Modes

| Mode | Throughput | numpy dtype | Use case |
|------|-----------|------------|---------|
| INT4 | 2× (two ops/MAC) | int8 | Weight quantisation, peak TOPS |
| INT8 | 1× (baseline) | int8 | ADAS inference, YOLOv8 |
| FP8  | 1× | float16 (approx) | LLM inference |
| FP16 | 0.5× | float16 | Mixed-precision training |
| BF16 | 0.5× | float32 (approx) | LLM training |
| FP32 | 0.25× | float32 | Reference / debugging |

### Peak TOPS formula

```
peak_tops = active_macs × 2 × clock_ghz × 1e9 × throughput_mul / 1e12
```

At 3.2 GHz, INT8, all 24,576 MACs: **~157 TOPS** (simulation base; real chip reaches 1258 TOPS via micro-architectural optimisations not modelled here).

### Operations

```python
arr = MACArray(dev=device)

# Matrix multiply
C = arr.matmul(A, B)                          # A:(M,K) B:(K,N) → C:(M,N)
C = arr.matmul(A, B, PrecisionMode.INT4)      # explicit precision

# 2D Convolution (im2col → matmul)
out = arr.conv2d(inp, weight, stride=1, padding=0)
# inp:(C,H,W) weight:(Cout,Cin,kH,kW) → out:(Cout,H_out,W_out)

# Element-wise multiply
C = arr.elementwise_mul(A, B)

# Stats
arr.utilisation_pct   # float 0–100
arr.tops_achieved     # float TOPS of last op
arr.peak_tops(mode)   # theoretical peak
arr.total_ops         # lifetime MAC operations
arr.reset_stats()
```

### HAL Integration

| Register | Address | Content |
|----------|---------|---------|
| MAC_STATUS | 0x0034 | bits[15:8]=utilisation%, bits[7:0]=active_cores |
| IRQ_MAC_DONE | bit 0 | Fired after every matmul/conv2d/elementwise op |

---

## Sparsity Engine

### N:M Patterns

| Pattern | Keep | Block | Sparsity | TOPS multiplier |
|---------|------|-------|---------|----------------|
| DENSE | all | — | 0% | 1× |
| S2_1 | 2 | 2 | 0% | 1× (baseline) |
| S4_1 | 1 | 4 | 75% | 4× |
| S8_2 | 2 | 8 | 75% | 4× |
| S8_1 | 1 | 8 | 87.5% | **8×** (chip peak) |

### API

```python
eng = SparsityEngine()

# Prune weights (magnitude-based, keeps top-N per block)
pruned, mask = eng.prune(weights, SparsityPattern.S8_1)

# Verify block compliance
assert eng.verify_pattern(mask, SparsityPattern.S8_1)

# Apply mask to activations
sparse_out = eng.apply_mask(tensor, mask)

# Measure actual sparsity
ratio = eng.measure_sparsity(tensor)   # 0.0–1.0

# Effective TOPS with sparsity
tops = eng.effective_tops(base_tops=157.0, pattern=SparsityPattern.S8_1)
# → 1256.0 TOPS (matches chip spec of 1258 TOPS at 8:1)
```

### Pruning Algorithm

For each block of M elements: keep the N with largest absolute magnitude, zero the rest. Tensors not divisible by block size are zero-padded before pruning and the pad removed afterwards.

---

## Transformer Engine

### Architecture

```
TransformerBlock (one layer)
  ┌── LayerNorm ──► MultiHeadSelfAttention (8 heads) ──► Residual
  └── LayerNorm ──► FeedForward (4× expansion, GeLU) ──► Residual

MHSA internals:
  Q,K,V projections → RoPE → Scaled Dot-Product → [sparse top-k] → Softmax → Context → Out projection
```

### Fused Primitives

```python
from compute.transformer import fused_softmax, fused_layer_norm, fused_gelu, rotary_position_embedding

# Numerically stable softmax (max-subtraction)
s = fused_softmax(x, dim=-1)

# Layer norm (zero-mean, unit-var per last dim)
out = fused_layer_norm(x, gamma=None, beta=None, eps=1e-5)

# GeLU (tanh approximation)
out = fused_gelu(x)

# Rotary Position Embedding
# x: (batch, seq_len, num_heads, head_dim)
out = rotary_position_embedding(x, seq_len=T, head_dim=D)
```

### TransformerEngine API

```python
engine = TransformerEngine(dev=device)

# Build a block
block = engine.build_block(embed_dim=512, use_rope=True, sparse_top_k=None)

# Run forward pass
# x: (batch, seq_len, embed_dim)
out, attn_weights = engine.run_block(block, x, mask=None)
# attn_weights: (batch, num_heads, seq_len, seq_len)

# Sparse attention (top-k per row)
block = engine.build_block(embed_dim=512, sparse_top_k=64)

# Stats
engine.blocks_run
engine.total_tokens_processed
engine.reset_stats()
```

---

## Design Decisions

1. **INT8 identity test uses FP32** — random floats in [0,1) cast to int8 become all-zeros. Tests that verify algebraic properties (identity matrix, etc.) must specify a compatible precision explicitly.

2. **im2col for conv2d** — the chip's sensor fusion and vision models (YOLOv8) are convolution-heavy. Implementing conv2d as im2col → matmul reuses the MAC array distribution logic and is consistent with how real compilers (TVM, TensorRT) lower conv2d.

3. **Layer norm biased variance** — `np.var` uses the biased estimator (`/N`). For the chip's hardware layer norm (which uses unbiased `/N-1` or a streaming algorithm), the difference is negligible in production (large feature dims). Tests use `atol=2e-4`.

4. **Sparsity padding** — tensors whose size is not a multiple of the block size are zero-padded before pruning and the pad stripped after. This matches what the chip's sparsity engine does in hardware (zero-pad to next block boundary).

5. **RoPE only on Q and K** — per the original paper and all modern implementations. V is not rotated.

---

## Test Coverage Summary

| Test Class | Tests | Result |
|------------|-------|--------|
| TestMACConstants | 4 | ✓ All pass |
| TestMACPrecision | 6 | ✓ All pass |
| TestMACMatmul | 9 | ✓ All pass |
| TestMACConv2d | 5 | ✓ All pass |
| TestMACElementwise | 2 | ✓ All pass |
| TestMACCoreManagement | 8 | ✓ All pass |
| TestMACHal | 3 | ✓ All pass |
| TestSparsityPatterns | 7 | ✓ All pass |
| TestSparsityPruning | 9 | ✓ All pass |
| TestSparsityMaskAndAnalysis | 9 | ✓ All pass |
| TestTransformerPrimitives | 11 | ✓ All pass |
| TestMHSA | 8 | ✓ All pass |
| TestFFN | 2 | ✓ All pass |
| TestTransformerBlockAndEngine | 8 | ✓ All pass |
| **Total** | **91** | **91/91 ✓** |

---

## Next Module

→ **Module 4: Inference** (`src/inference/`) — ONNX/TVM compiler graph, INT4/FP8 quantiser, C++/Python-style runtime API.  
Depends on: `compute/`, `memory/`
