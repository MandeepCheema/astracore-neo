# Module 11 — Models

## Overview

The Models subsystem is the AstraCore Neo **model library**: a versioned catalog of pre-validated AI model descriptors for the five canonical ADAS inference tasks. It does not store weights or run inference — it describes *what a model is* and *whether it can run on the hardware*.

Four components:

| Component | Class | Purpose |
|-----------|-------|---------|
| Descriptor | `ModelDescriptor` | Metadata schema for one model |
| Validator | `ModelValidator` | Checks descriptor against `HardwareSpec` |
| Catalog | `ModelCatalog` | Versioned registry with search and recommendation |
| Reference library | `reference_models` | 5 pre-validated canonical models |

---

## ModelDescriptor

### TensorSpec

```python
TensorSpec(name="image", shape=[1, 3, 416, 416], dtype="uint8")
```

`numel()` returns the product of all shape dimensions. `size_bytes()` maps dtype to bytes (float32→4, float16→2, int8/uint8→1, int4→1).

### ModelDescriptor Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Unique model name |
| `version` | str | Semantic version string |
| `task` | `ModelTask` | Task category enum |
| `precision` | `ModelPrecision` | Dominant weight/activation precision |
| `input_specs` | list[TensorSpec] | One entry per model input |
| `output_specs` | list[TensorSpec] | One entry per model output |
| `params_M` | float | Parameter count in millions |
| `ops_G` | float | Giga-ops per inference |
| `memory_mb` | float | Peak activation memory in MB |
| `max_latency_ms` | float | Target latency in ms (0 = no constraint) |

`model_id` property returns `"name@version"`.

### ModelTask Enum

`OBJECT_DETECTION`, `LANE_DETECTION`, `DEPTH_ESTIMATION`, `DRIVER_MONITORING`, `OCCUPANCY_GRID`, `CLASSIFICATION`, `SEGMENTATION`

### ModelPrecision Enum

`FP32`, `FP16`, `INT8`, `INT4`

---

## HardwareSpec

```python
hw = HardwareSpec(
    compute_tops=1258.0,
    memory_mb=128.0,
    max_latency_ms=16.0,
    supported_precisions=frozenset({ModelPrecision.INT8, ModelPrecision.FP16}),
)
```

Represents the target hardware capabilities. Default `supported_precisions` includes all four precision levels.

---

## ModelValidator

Validates a `ModelDescriptor` against a `HardwareSpec`. Returns a `ValidationResult(passed, violations)`.

### Violation Types

| Condition | Violation Description |
|-----------|----------------------|
| Precision not in `hw.supported_precisions` | "Precision X not supported by hardware" |
| `descriptor.memory_mb > hw.memory_mb` | "Model requires X MB; hardware provides Y MB" |
| `descriptor.max_latency_ms > hw.max_latency_ms` (both > 0) | "Model latency target X ms exceeds hardware limit Y ms" |
| Required TOPS to meet latency > `hw.compute_tops` | "Model requires X TOPS to meet latency; hardware provides Y TOPS" |

`ValidationResult.__bool__()` returns `passed`, so `if validator.validate(desc):` works naturally.

```python
validator = ModelValidator(ASTRA_HW_SPEC)
result = validator.validate(ASTRA_DETECT_V1)
assert result.passed
```

---

## ModelCatalog

### Key Methods

| Method | Description |
|--------|-------------|
| `register(descriptor)` | Add to catalog; raises `ModelRegistrationError` if duplicate |
| `unregister(name, version)` | Remove; raises `ModelNotFoundError` if absent |
| `find(name, version)` | Exact lookup; raises `ModelNotFoundError` if absent |
| `find_latest(name)` | Lexicographically latest version of a named model |
| `filter_by_task(task)` | All models for a task |
| `filter_by_precision(precision)` | All models of a precision |
| `filter_by_task_and_precision(task, precision)` | Intersection filter |
| `get_recommended(task, validator=None)` | Best model for task; lowest latency target; validator filters |
| `"name@version" in catalog` | Membership test |
| `len(catalog)` | Total number of registered models |

`get_recommended()` sorts by `(max_latency_ms, ops_G)` ascending — lowest-latency model first. If a `validator` is provided, only passing models are considered.

---

## Reference Model Library

Five pre-validated descriptors ship with the AstraCore Neo SDK:

| Model | Task | Precision | Params | Ops | Memory | Latency |
|-------|------|-----------|--------|-----|--------|---------|
| `astra_detect_v1@1.0.0` | Object Detection | INT8 | 3.2 M | 8.7 G | 12 MB | 10 ms |
| `astra_lane_v1@1.0.0` | Lane Detection | INT8 | 1.8 M | 4.2 G | 8 MB | 8 ms |
| `astra_depth_v1@1.0.0` | Depth Estimation | INT8 | 6.5 M | 14.1 G | 18 MB | 14 ms |
| `astra_dms_v1@1.0.0` | Driver Monitoring | INT8 | 0.9 M | 0.6 G | 2 MB | 4 ms |
| `astra_occgrid_v1@1.0.0` | Occupancy Grid | FP16 | 14.2 M | 35.0 G | 52 MB | 15 ms |

All five pass validation against `ASTRA_HW_SPEC` (1258 TOPS, 128 MB, 16 ms).

```python
from models import build_default_catalog, ASTRA_HW_SPEC
from models import ModelValidator, ModelTask

catalog = build_default_catalog()
validator = ModelValidator(ASTRA_HW_SPEC)

# Find the fastest validated object detector
rec = catalog.get_recommended(ModelTask.OBJECT_DETECTION, validator=validator)
print(rec.model_id)   # astra_detect_v1@1.0.0
```

### Connection to DMS Module

`astra_dms_v1` is designed to feed the DMS module directly:
- Input: 64×64 grayscale face crop
- Output 1: 68 facial landmarks (x, y) pairs
- Output 2: head pose [yaw, pitch, roll] — feeds `HeadPoseTracker.update()`
- Output 3: EAR [left, right] — feeds `GazeTracker.update()`

---

## Exception Hierarchy

```
ModelsBaseError
├── ModelRegistrationError  — duplicate or invalid registration
├── ModelNotFoundError      — lookup miss
└── ModelValidationError    — constraint violation or invalid descriptor field
```

---

## Test Coverage (55/55)

| Category | Tests |
|----------|-------|
| TensorSpec numel and size_bytes | 4 |
| ModelDescriptor construction and validation | 7 |
| HardwareSpec construction | 3 |
| ModelValidator — passing | 3 |
| ModelValidator — violations (4 types + multiple + validate_all) | 6 |
| ModelCatalog basic (register, find, unregister, contains, all_models) | 9 |
| ModelCatalog filter (by task, precision, latest) | 6 |
| ModelCatalog recommendation | 4 |
| Reference models (tasks, precisions, validation, catalog) | 13 |
| **Total** | **55** |

---

## RTL Notes

The Models module is pure software — it has no hardware RTL equivalent. It acts as the **compiler/SDK side** of the chip deployment flow:

1. Designer selects a model from the catalog
2. `ModelValidator` checks hardware fit
3. The chosen `ModelDescriptor` is handed to the **inference** module's `AstraCoreCompiler` for tiling, quantization, and graph compilation
4. The compiled model runs on the chip via `InferenceRuntime`

The `astra_dms_v1` descriptor is the bridge between the Models and DMS modules, specifying exactly what the on-chip face-landmark model produces and what the DMS pipeline consumes.
