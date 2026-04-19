"""FP32 reference execution for NnGraph models via onnxruntime.

Two consumers:

  1. F1-C2 (quantiser) — needs per-tensor activation statistics on a
     representative calibration set. This module runs the original
     .onnx end-to-end and yields named intermediate tensors so the
     quantiser can fit scales without re-implementing the graph.

  2. F1-C5 (end-to-end cocotb) — needs the golden output vector on a
     test image. The NPU run is compared against this, accounting for
     INT8 quantisation error.

Why onnxruntime and not a from-scratch reference interpreter: ORT is
the canonical definition of what the .onnx file means. Rewriting the
whole op library in numpy to cross-check would multiply the surface
area we have to trust.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclass
class ReferenceRun:
    """One forward pass captured as a name → tensor dict.

    Graph-level outputs are in `outputs`. Intermediate tensors requested
    via `intermediate_names` are in `activations`. All arrays are
    float32.
    """
    outputs: Dict[str, np.ndarray]
    activations: Dict[str, np.ndarray]


def _ort_session(onnx_path: str):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    # CPU-only, no graph-level rewrites — the ORT "BASIC" optimisation
    # level still folds some constants but leaves the op boundaries
    # intact, which matters because the intermediate tensor names we
    # fetch must match the NnGraph layer outputs.
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])


def _augment_outputs(onnx_path: str, keep_tensors: Iterable[str]) -> str:
    """Produce a temp .onnx where `keep_tensors` are also promoted to
    graph-level outputs, so ORT will return them alongside the real
    outputs. Returns the path to the augmented file.

    Why we rewrite rather than hook ORT: ORT has no first-class API for
    "give me this intermediate tensor"; the supported pattern is to add
    the tensor name to graph.output.
    """
    import onnx
    import tempfile

    keep = set(keep_tensors)
    if not keep:
        return onnx_path

    model = onnx.load(onnx_path)
    existing = {o.name for o in model.graph.output}
    vi_by_name = {vi.name: vi for vi in model.graph.value_info}
    for name in keep:
        if name in existing:
            continue
        if name in vi_by_name:
            model.graph.output.append(vi_by_name[name])
        else:
            # Fall back to a bare value_info — ORT will fill shape at
            # runtime. If this ever fails, add shape inference upstream.
            model.graph.output.append(
                onnx.helper.make_empty_tensor_value_info(name))

    fd, fd_path = tempfile.mkstemp(suffix=".onnx")
    import os as _os
    _os.close(fd)  # Windows EACCES if the fd stays open while onnx.save reopens the path.
    onnx.save(model, fd_path)
    return fd_path


def run_reference(onnx_path: str,
                  inputs: Dict[str, np.ndarray],
                  *,
                  intermediate_names: Optional[List[str]] = None,
                  ) -> ReferenceRun:
    """Run one FP32 forward pass through onnxruntime.

    Args:
        onnx_path: path to the .onnx.
        inputs: graph-input name → float32 ndarray with the shape the
            graph expects.
        intermediate_names: optional list of tensor names (typically
            NnLayer.outputs entries) to also capture.

    Returns:
        ReferenceRun with `outputs` (graph outputs) and `activations`
        (the requested intermediates).
    """
    import onnxruntime as ort
    path = _augment_outputs(onnx_path, intermediate_names or [])
    sess = _ort_session(path)

    # Validate the caller gave us the right input names and dtypes —
    # silent-broadcast bugs here would pollute downstream quant scales.
    expected = {i.name: i for i in sess.get_inputs()}
    if set(expected) != set(inputs):
        raise ValueError(
            f"input name mismatch: model expects {sorted(expected)}, "
            f"got {sorted(inputs)}"
        )
    feed = {}
    for name, arr in inputs.items():
        if arr.dtype != np.float32:
            raise TypeError(f"input {name!r}: must be float32, got {arr.dtype}")
        feed[name] = arr

    out_names = [o.name for o in sess.get_outputs()]
    results = sess.run(out_names, feed)

    wanted_intermediate = set(intermediate_names or [])
    outputs: Dict[str, np.ndarray] = {}
    activations: Dict[str, np.ndarray] = {}
    for name, val in zip(out_names, results):
        if name in wanted_intermediate:
            activations[name] = val
        else:
            outputs[name] = val
    return ReferenceRun(outputs=outputs, activations=activations)


def make_seeded_input(shape, seed: int = 0,
                       low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Deterministic float32 input for regression tests. The exact
    distribution doesn't matter for a structural smoke test — it just
    has to be reproducible and non-trivial."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=shape).astype(np.float32)
