"""F1-C1 acceptance test — cross-check yolov8n.onnx against yolo_trace.

Gate: the loader produces a structurally-plausible graph for the real
Ultralytics YOLOv8-N model, with conv counts and total-MAC totals that
agree with `yolo_trace.build_yolov8n()` within tolerance.

The test is skipped (not failed) if the .onnx artifact is missing —
acquiring it requires a separate venv with ultralytics (see
scripts/export_yolov8n_onnx.py). CI should run the export once and
cache the artifact before invoking this test.

The MAC-count check is a tight ±3% gate. After the 2026-04-18 fix to
yolo_trace's detection-head channel widths (G2 in the F1-C1 audit),
yolo_trace reconstructs the real ultralytics export to within 0.01%.
Any future drift past ±3% is the loader or yolo_trace breaking — not
acceptable floating-point or spec wobble.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from tools.npu_ref.nn_graph import OP_CONV
from tools.npu_ref.onnx_loader import load_onnx
from tools.npu_ref.onnx_reference import make_seeded_input, run_reference
from tools.npu_ref.yolo_trace import build_yolov8n

REPO = Path(__file__).resolve().parent.parent
ONNX_PATH = REPO / "data" / "models" / "yolov8n.onnx"
MANIFEST = REPO / "data" / "models" / "yolov8n.manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_artifact():
    if not ONNX_PATH.exists():
        pytest.skip(
            f"{ONNX_PATH} not present. Run "
            f"scripts/export_yolov8n_onnx.py from a venv with "
            f"ultralytics installed to produce it."
        )
    if MANIFEST.exists():
        m = json.loads(MANIFEST.read_text())
        actual = _sha256(ONNX_PATH)
        expected = m.get("onnx_sha256")
        if expected and actual != expected:
            pytest.fail(
                f"{ONNX_PATH} SHA256 mismatch: got {actual}, "
                f"manifest says {expected}. Rerun the export script."
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _conv_fingerprint_onnx(L) -> Tuple:
    """Stable fingerprint for an NnLayer conv; compare against the
    yolo_trace fingerprint on matching fields."""
    in_shape = L.primary_input_shape
    out_shape = L.primary_output_shape
    if not in_shape or not out_shape:
        return None
    # ONNX NCHW: (1, C_in, H_in, W_in) → (1, C_out, H_out, W_out)
    c_in = in_shape[1] if len(in_shape) == 4 else None
    c_out = out_shape[1] if len(out_shape) == 4 else None
    h_out = out_shape[2] if len(out_shape) == 4 else None
    w_out = out_shape[3] if len(out_shape) == 4 else None
    return (c_in, c_out, h_out, w_out,
            L.attrs.get("kernel"), L.attrs.get("stride"),
            L.attrs.get("groups", 1))


def _conv_fingerprint_trace(layer) -> Tuple:
    # yolo_trace.Layer: in_shape = (C, H, W), out_shape = (C, H, W)
    c_in, _, _ = layer.in_shape
    c_out, h_out, w_out = layer.out_shape
    k = (layer.kernel, layer.kernel)
    s = (layer.stride, layer.stride)
    return (c_in, c_out, h_out, w_out, k, s, layer.groups)


def _conv_macs(c_in, c_out, h_out, w_out, kernel, groups) -> int:
    k_h, k_w = kernel
    return h_out * w_out * k_h * k_w * (c_in // groups) * c_out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_loader_accepts_real_model():
    _require_artifact()
    g = load_onnx(str(ONNX_PATH))
    assert len(g) > 0
    # Every conv layer must have resolved shapes after shape inference
    # — otherwise quantiser/tiler downstream will break.
    for L in g.layers_of(OP_CONV):
        assert L.primary_input_shape is not None, \
            f"conv {L.name!r}: input shape unresolved"
        assert L.primary_output_shape is not None, \
            f"conv {L.name!r}: output shape unresolved"
        assert all(d > 0 for d in L.primary_input_shape), \
            f"conv {L.name!r}: input has dynamic dim {L.primary_input_shape}"
        assert L.weights is not None, \
            f"conv {L.name!r}: weights missing"


def test_conv_count_in_expected_range():
    _require_artifact()
    g = load_onnx(str(ONNX_PATH))
    trace_convs = [L for L in build_yolov8n() if L.kernel > 0]
    onnx_convs = g.layers_of(OP_CONV)
    # YOLOv8-N has on the order of 60–80 convs; require ONNX and trace
    # to be within 25% of each other.
    ratio = len(onnx_convs) / len(trace_convs)
    assert 0.75 <= ratio <= 1.25, (
        f"conv count mismatch: ONNX={len(onnx_convs)}, "
        f"trace={len(trace_convs)}, ratio={ratio:.2f}"
    )


def test_total_macs_within_tolerance():
    """Tight agreement between the real export and yolo_trace. After
    the G2 head-channel fix (2026-04-18) the two agree to within
    0.01%; any drift past ±3% indicates a loader regression or a
    yolo_trace vs. ultralytics divergence that needs investigation
    before we keep trusting the perf-model output.
    """
    _require_artifact()
    g = load_onnx(str(ONNX_PATH))

    onnx_macs = 0
    for L in g.layers_of(OP_CONV):
        in_shape = L.primary_input_shape
        out_shape = L.primary_output_shape
        c_in = in_shape[1]
        c_out, h_out, w_out = out_shape[1], out_shape[2], out_shape[3]
        onnx_macs += _conv_macs(
            c_in, c_out, h_out, w_out,
            L.attrs["kernel"], L.attrs.get("groups", 1),
        )

    trace_macs = sum(L.macs for L in build_yolov8n())
    ratio = onnx_macs / trace_macs
    print(f"total MACs  ONNX={onnx_macs:,}  trace={trace_macs:,}  "
          f"ratio={ratio:.4f}")
    assert 0.97 <= ratio <= 1.03, (
        f"MAC discrepancy: ONNX={onnx_macs:,}, trace={trace_macs:,}, "
        f"ratio={ratio:.4f} — drift past ±3% indicates the loader "
        f"silently dropped a conv, or yolo_trace's structural "
        f"reconstruction has drifted from the ultralytics spec."
    )


def test_io_shapes_look_like_yolov8():
    _require_artifact()
    g = load_onnx(str(ONNX_PATH))
    # One input: (1, 3, 640, 640).
    assert len(g.inputs) == 1
    in_shape = next(iter(g.inputs.values()))
    assert in_shape == (1, 3, 640, 640), (
        f"expected input (1,3,640,640), got {in_shape}"
    )
    # At least one output with a plausible detection head shape.
    assert len(g.outputs) >= 1


def test_fingerprint_intersection_is_substantial():
    """At least half the trace convs should find a match in the ONNX
    conv set (or vice versa). A low intersection indicates we're
    loading the wrong model or our shape-extraction is broken."""
    _require_artifact()
    g = load_onnx(str(ONNX_PATH))

    onnx_fps = {_conv_fingerprint_onnx(L) for L in g.layers_of(OP_CONV)}
    onnx_fps.discard(None)
    trace_fps = {_conv_fingerprint_trace(L) for L in build_yolov8n()
                 if L.kernel > 0}
    overlap = onnx_fps & trace_fps
    # Print for manual inspection — useful when adjusting yolo_trace.
    print(f"ONNX-only fingerprints: {len(onnx_fps - trace_fps)}")
    print(f"trace-only fingerprints: {len(trace_fps - onnx_fps)}")
    print(f"shared: {len(overlap)}")
    # After G2, every trace fingerprint should map to an ONNX
    # fingerprint (≥95% overlap). A dip below this signals either a
    # loader bug or a yolo_trace regression.
    shared_ratio = len(overlap) / max(len(trace_fps), 1)
    assert shared_ratio >= 0.95, (
        f"fingerprint intersection {shared_ratio:.1%} "
        f"({len(overlap)}/{len(trace_fps)}) below 95% — loader or "
        f"yolo_trace has regressed."
    )


def test_loader_to_ort_round_trip():
    """G4 integration gate — load yolov8n.onnx AND run it through
    onnx_reference in one test. Unit tests cover the loader and the
    reference runner separately but never together; integration tests
    catch bugs that unit tests can't (see memory/feedback on this).
    """
    _require_artifact()
    # Load and spot-check the loader's view of the graph.
    g = load_onnx(str(ONNX_PATH))
    assert len(g.layers_of(OP_CONV)) >= 60
    (in_name,) = g.inputs.keys()
    assert g.inputs[in_name] == (1, 3, 640, 640)

    # Now run the same .onnx through ORT. The loader's
    # concretise-batch step runs shape inference in-memory without
    # touching the source file, so ORT must accept the file unchanged.
    probe = make_seeded_input((1, 3, 640, 640), seed=7)
    run = run_reference(str(ONNX_PATH), {in_name: probe})
    assert len(run.outputs) == 1
    (out_name, out_tensor) = next(iter(run.outputs.items()))
    assert out_name in g.outputs
    assert out_tensor.shape == (1, 84, 8400)
    assert out_tensor.dtype.name == "float32"
    # Output is bounded: YOLOv8 writes class logits + bbox coords
    # (both finite). Pure sanity — no NaN / Inf leaking from the export.
    assert np.isfinite(out_tensor).all()
