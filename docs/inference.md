# Inference — Compiler, Quantizer, Runtime

**Module:** `src/inference/`  
**Depends on:** `compute/`, `memory/`  
**Status:** DONE  
**Test log:** `logs/test_inference.log`  
**Test result:** 65/65 passed ✓

---

## Purpose

Models the AstraCore Neo's AI inference pipeline — the software layer that sits between model files and the compute hardware:

- **Compiler** — parses computation graphs, fuses operators, tiles large ops, topologically schedules, estimates TOPS
- **Quantizer** — calibrates per-tensor / per-channel statistics, applies INT4/INT8/FP8 quantization, dequantizes for validation
- **Runtime** — manages inference sessions, dispatches each graph node to MACArray / TransformerEngine, profiles execution

---

## Files

| File | Description |
|------|-------------|
| `inference/compiler.py` | `AstraCoreCompiler`, `CompiledModel`, `GraphNode`, `OpType`, `CompilerTarget` |
| `inference/quantizer.py` | `Quantizer`, `QuantConfig`, `QuantPrecision`, `QuantizedTensor`, `CalibStats` |
| `inference/runtime.py` | `InferenceRuntime`, `InferenceSession`, `RunResult`, `NodeProfile` |
| `inference/exceptions.py` | `InferenceError`, `CompilerError`, `QuantizationError`, `TilingError`, `FusionError` |
| `inference/__init__.py` | Public API exports |

---

## Compiler

### Pipeline

```
node_defs (list of dicts)
    │
    ▼  parse()
GraphNode list  ──→  validate (cycle detection, unknown ops, duplicate ids)
    │
    ▼  fuse()
Fused node list  ──→  conv+relu → FUSED_CONV_RELU
                       matmul+elemwise → FUSED_MATMUL_ADD
                       layernorm+gelu → FUSED_LAYERNORM_GELU
                       attention+softmax → FUSED_ATTENTION_SOFTMAX
    │
    ▼  tile()
Tiled nodes  ──→  mark nodes whose output.numel() > tile_size
    │
    ▼  topological_sort() (Kahn's algorithm)
CompiledModel schedule  ──→  TOPS estimate + memory estimate
```

### Supported Op Types

`matmul`, `conv2d`, `elemwise`, `relu`, `gelu`, `sigmoid`, `tanh`, `layernorm`, `batchnorm`, `softmax`, `reshape`, `transpose`, `concat`, `split`, `maxpool`, `avgpool`, `attention`, `load`, `store` + 4 fused variants.

### API

```python
compiler = AstraCoreCompiler(tile_size=256*1024, enable_fusion=True, enable_tiling=True)

# Define graph
node_defs = [
    {"id": "n0", "op": "conv2d",  "inputs": ["img"],    "outputs": ["feat"],
     "shape_in": (3,224,224), "shape_out": (64,112,112)},
    {"id": "n1", "op": "relu",    "inputs": ["feat"],   "outputs": ["act"]},
    {"id": "n2", "op": "matmul",  "inputs": ["act"],    "outputs": ["logits"]},
]

nodes = compiler.parse(node_defs)
model = compiler.compile(nodes, name="yolov8", target=CompilerTarget.INT8,
                         input_names=["img"], output_names=["logits"])

print(model.node_count)        # 2 (n0+n1 fused)
print(model.fused_nodes)       # 1
print(model.estimated_tops)    # float
print(model.memory_bytes)      # estimated activation memory
```

### CompilerTarget TOPS Multipliers

| Target | Multiplier |
|--------|-----------|
| INT4 | 2.0× |
| INT8 | 1.0× (baseline) |
| FP8 | 1.0× |
| FP16 | 0.5× |
| FP32 | 0.25× |

---

## Quantizer

### Workflow

```
1. calibrate(name, data)   — accumulate min/max/mean/std from representative batches
2. quantize(name, data)    — compute scale+zero_point, clamp, round
3. dequantize(qt)          — recover float32 from QuantizedTensor
```

### Precision ranges

| Precision | Min | Max |
|-----------|-----|-----|
| INT4 | -8 | 7 |
| INT8 | -128 | 127 |
| FP8 (E4M3 approx) | -448 | 448 |

### API

```python
from inference import Quantizer, QuantConfig, QuantPrecision, QuantGranularity

# Per-tensor symmetric INT8 (default)
q = Quantizer()
q.calibrate("weights", weight_batch_1)
q.calibrate("weights", weight_batch_2)   # accumulates
qt = q.quantize("weights", weights)
recovered = q.dequantize(qt)

# Per-channel INT4
q4 = Quantizer(QuantConfig(
    precision=QuantPrecision.INT4,
    granularity=QuantGranularity.PER_CHANNEL,
    symmetric=True,
))

# One-shot (uses tensor's own min/max)
qt = q.quantize_uncalibrated(tensor, QuantPrecision.INT8)
```

### Scale / zero-point computation

**Symmetric:**
```
scale     = max(|min|, |max|) / qmax
zero_point = 0
```

**Asymmetric:**
```
scale     = (max - min) / (qmax - qmin)
zero_point = qmin - min / scale
```

---

## Runtime

### Session lifecycle

```
InferenceRuntime
  └── load_model(compiled_model) → InferenceSession  (LOADED)
        ├── bind_input(name, tensor)
        ├── run() → RunResult                          (RUNNING → DONE)
        └── (reusable — bind new inputs, call run() again)
  └── unload_session(session_id)
```

### Operator dispatch table

| Op | Dispatched to |
|----|--------------|
| MATMUL, FUSED_MATMUL_ADD | `MACArray.matmul()` |
| CONV2D, FUSED_CONV_RELU | `MACArray.conv2d()` |
| ELEMWISE | `MACArray.elementwise_mul()` |
| RELU | `np.maximum(0, x)` |
| GELU, FUSED_LAYERNORM_GELU | `fused_gelu()` / `fused_layer_norm()` |
| SIGMOID | `1 / (1 + exp(-x))` |
| TANH | `np.tanh()` |
| LAYERNORM | `fused_layer_norm()` |
| SOFTMAX | `fused_softmax()` |
| ATTENTION, FUSED_ATTENTION_SOFTMAX | `TransformerEngine.run_block()` |
| RESHAPE | `np.reshape()` |
| TRANSPOSE | `np.transpose()` |
| Others | passthrough |

### API

```python
rt = InferenceRuntime(mac_array=arr, transformer=engine, sparsity=sp_eng)

# Load and run
session = rt.load_model(compiled_model)
result  = rt.run(session, {"img": input_tensor})

print(result.latency_ms)
print(result.total_tops)
print(result.slowest_node.node_id)
print(result.node_profiles)         # per-node latency breakdown

# Multiple runs with new inputs
result2 = rt.run(session, {"img": another_tensor})
```

### RunResult fields

| Field | Description |
|-------|-------------|
| `outputs` | Dict[name → np.ndarray] |
| `latency_ms` | End-to-end wall-clock ms |
| `node_profiles` | List of NodeProfile (id, op, latency, TOPS) |
| `total_tops` | Sum of per-node TOPS contributions |
| `session_id` | Which session produced this result |
| `fastest_node` | NodeProfile with min latency |
| `slowest_node` | NodeProfile with max latency |

---

## Design Decisions

1. **Compiler stores no weights** — the compiler only models the graph structure and schedule. Weights are bound at runtime through input binding, matching how real compilers (TVM, TensorRT) work: compile once, run many times with different weights/inputs.

2. **Fusion is single-pass adjacent pairs** — real compilers do multi-pass pattern matching across arbitrary subgraphs. Single-pass is sufficient to demonstrate the concept and produces correct fused ops for the common cases (conv+relu, matmul+add) the chip targets.

3. **Quantization uses float32 storage internally** — QuantizedTensor stores quantized values as float32 arrays for numpy compatibility. A real chip would store INT4/INT8 as packed bits; the scale/zero_point metadata would be in a side-channel register.

4. **Runtime dispatch uses numpy** — all op implementations delegate to numpy / compute module functions. This means latency_ms is dominated by Python overhead, not actual arithmetic cost. In production, dispatch would call compiled kernels.

5. **Session is reusable** — after `run()` returns DONE, you can bind new inputs and call `run()` again. This models the chip's persistent session model where model weights stay loaded in SRAM.

---

## Test Coverage Summary

| Test Class | Tests | Result |
|------------|-------|--------|
| TestCompilerParse | 7 | ✓ All pass |
| TestCompilerFusion | 8 | ✓ All pass |
| TestCompilerTilingScheduling | 9 | ✓ All pass |
| TestQuantizerCalibration | 5 | ✓ All pass |
| TestQuantizerQuantize | 13 | ✓ All pass |
| TestRuntimeSession | 13 | ✓ All pass |
| TestRuntimeDispatch | 10 | ✓ All pass |
| **Total** | **65** | **65/65 ✓** |

---

## Next Module

→ **Module 5: Perception** (`src/perception/`) — Camera (MIPI CSI-2, ISP-Pro), Lidar (4D point cloud), Radar, sensor fusion accelerator.  
Depends on: `hal/`, `memory/`
