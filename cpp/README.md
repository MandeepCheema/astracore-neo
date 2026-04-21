# AstraCore Neo — C++ runtime (v0.1)

## What this is

Production-grade C++ implementation of the same `Backend` protocol the
Python SDK ships. Same model files, same EP plumbing, same numerical
output. The Python `OrtBackend` is the developer-facing artefact;
this library is the **deployment-grade** artefact OEM firmware
links against. They share one ONNX Runtime under the hood, so output
is bit-identical (proven by the cross-runtime conformance test once
both ends are present — see `tests/test_cpp_runtime.py`).

## Layout

```
cpp/
├── include/astracore/
│   └── runtime.hpp                   # Public C++ API (Backend, Tensor, Report)
├── src/
│   └── runtime.cpp                   # OrtBackend impl wrapping ONNX Runtime C++ API
├── python/
│   └── bindings.cpp                  # pybind11 binding → astracore_runtime.so
├── third_party/onnxruntime/          # Vendored ORT headers (1.23.2)
│   ├── onnxruntime_c_api.h
│   ├── onnxruntime_cxx_api.h
│   ├── onnxruntime_cxx_inline.h
│   └── onnxruntime_float16.h
├── setup.py                          # pybind11-driven build (Linux/macOS/WSL)
└── README.md                         # this file
```

## Status

| Capability | v0.1 (today) | Phase B v1.0 |
|---|---|---|
| `astracore::Backend` interface | ✅ | ✅ |
| `OrtBackend` wrapping ORT 1.23 | ✅ | ✅ |
| `Tensor` non-owning input view + owning output | ✅ | ✅ |
| EP short-name aliases (cpu/cuda/tensorrt/openvino/qnn/coreml/dml/rocm) | ✅ | ✅ + per-EP options dict |
| pybind11 binding | ✅ | ✅ |
| Output-shape inference | ✅ via ORT | ✅ |
| Stable C ABI (`extern "C"`) | — | ✅ |
| Cross-compiler linking guarantee | — | ✅ |
| Direct silicon backends (TensorRT plan loader, SNPE, F1 XRT) | — | ✅ (per target) |
| Quantisation API | — (use Python `astracore quantise` then load .int8.onnx via OrtBackend) | ✅ in-process |

The v0.1 ↔ v1.0 gap is the C++ runtime track of Phase B. v0.1 is the
minimum surface needed to claim "OEM firmware can integrate AstraCore
in C++ today via pybind11" — which is what unblocks MLPerf submissions
and licensee-on-customer-silicon demos.

## Build prerequisites

| OS | Toolchain |
|---|---|
| **Linux / WSL Ubuntu 22.04** | `g++ ≥ 9` + `python3-dev` (`sudo apt install python3-dev`) + `pip install pybind11 onnxruntime` |
| **macOS** | Xcode CLI tools + `pip install pybind11 onnxruntime` |
| **Windows MSVC** | Visual Studio 2022 Build Tools (cl.exe) + `pip install pybind11 onnxruntime` |

The setup.py auto-detects the ONNX Runtime shared library inside the
pip wheel and links against it; no separate ORT install needed.

## Build

```bash
cd cpp
python setup.py build_ext --inplace
```

Produces `astracore_runtime.cpython-<py>-<arch>.so` (Linux/macOS) or
`.pyd` (Windows) **in this directory**. Add `cpp/` to `sys.path` to
import.

## Smoke test (Python, after build)

```python
import sys; sys.path.insert(0, "cpp")
import numpy as np
import astracore_runtime as ar

print("astracore_runtime version:", ar.version())
be = ar.make_backend("onnxruntime", ["cpu"])
program = be.compile("data/models/yolov8n.onnx")
x = np.random.standard_normal((1, 3, 640, 640)).astype(np.float32)
out = be.run(program, {"images": x})
print("YOLO output shape:", next(iter(out.values())).shape)
print("Wall ms / inf:", be.report_last().wall_ms_per_inference)
```

Cross-validation against Python `OrtBackend`:

```python
from astracore.backends.ort import OrtBackend
py_be = OrtBackend(providers=["cpu"])
py_program = py_be.compile("data/models/yolov8n.onnx")
py_out = py_be.run(py_program, {"images": x})

# Same model + same input + same backend → bit-identical output.
np.testing.assert_array_equal(
    next(iter(out.values())),
    next(iter(py_out.values())),
)
```

## Why pybind11 (not raw ctypes / cffi)?

- pybind11 maps numpy arrays to `astracore::Tensor` views with no copy
  (zero-overhead — same pointer, same shape).
- Type-safe: dtype mismatches raise at the binding boundary, not as
  segfaults.
- The same C++ source compiles directly into a customer's firmware
  build — no wrapper rewrite. The pybind11 binding is just one extra
  translation unit in the Python build, not a separate API surface.

## Why vendor the ORT headers?

ONNX Runtime's pip wheel ships the shared library but not the C/C++
headers. Vendoring `onnxruntime_c_api.h` + `_cxx_api.h` + `_cxx_inline.h`
+ `_float16.h` (4 files, ~14k lines total) at a pinned ORT version
(1.23.2 today) decouples our build from the customer's ORT install
choice. The runtime loads the wheel-bundled `libonnxruntime.so` —
header version and lib version must be ABI-compatible (ORT guarantees
this within a major version).

To bump ORT: re-fetch the four headers from the matching tag at
`https://github.com/microsoft/onnxruntime/tree/v<X.Y.Z>/include/onnxruntime/core/session/`.

## What's NOT in v0.1 (intentionally)

- **Per-EP options** (TRT workspace size, CUDA device_id, OpenVINO
  device_type). v0.1 honours the EP short-name list; per-EP option
  dicts ship in v0.2. Workaround today: set ORT environment variables
  before instantiating the backend.
- **Stable C ABI**. The C++ ABI is implicit; cross-compiler linking
  (mix MSVC C++ host with libc++ extension) is not yet promised.
  Phase B work.
- **Hot-swap silicon backends**. v0.1 is ORT-only. Direct TensorRT plan
  loader, SNPE, F1 XRT are separate backends (each ~3-4 weeks).
- **Reference-count / lifetime hardening**. The Python binding takes
  ownership of the `Program` raw pointer through pybind11's
  `take_ownership` policy — works correctly today, but the explicit
  `unique_ptr` boundary will be enforced more strictly in v0.2.

## Forward (Phase B follow-on, by priority)

1. **Per-EP options** — ~2 days. Lift the Python `_normalise_providers`
   logic into C++. Closes the gap with the Python OrtBackend feature
   parity.
2. **Cross-runtime conformance test** — ~1 day. SHA-256 fingerprint of
   the C++ output must match the Python output for every zoo model.
   Locks "C++ and Python paths produce bit-identical numerics".
3. **Stable C ABI shim** (`extern "C" astracore_*` functions) — ~3 days.
   Lets a customer link from a different C++ compiler / standard library.
4. **TensorRT plan loader backend** — ~3 weeks. Direct .plan file load,
   bypasses ORT EP overhead.
5. **F1 XRT backend** — Phase B already-scheduled (F1-F1/F2/F3).
6. **MLPerf loadgen integration** — ~1 week (CPU-runnable today; real
   submission needs cloud GPU).
