"""Pin YOLOv8n per-image output fingerprints as a regression.

Rationale
---------
Every one of the 28 pre-packaged eval images produces a distinct
SHA-256 prefix over the YOLOv8n raw output (rounded to 3 dp) on the
host's onnxruntime CPU EP. If an ORT upgrade, a session-option
tweak, or a backend swap silently shifts the numerics, these hashes
diverge and the test fails. That's the drift signal we need before
adding a CUDA / TensorRT / QNN backend.

Lives under tests/ instead of reports/ because pytest picks it up
automatically and the baseline becomes part of the committed
artefact — exactly what we want for a regression.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parent.parent
BASELINE = Path(__file__).resolve().parent / "yolo_fingerprint_baseline.json"


def _fingerprint(arr: np.ndarray, *, nd: int = 3) -> str:
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(np.round(arr.astype(np.float64).ravel(), nd).tobytes())
    return h.hexdigest()[:16]


@pytest.fixture(scope="module")
def baseline():
    if not BASELINE.exists():
        pytest.skip(f"baseline file missing: {BASELINE}")
    return json.loads(BASELINE.read_text())


@pytest.fixture(scope="module")
def eval_images():
    p = REPO / "data" / "calibration" / "yolov8n_eval.npz"
    if not p.exists():
        pytest.skip(f"eval set missing: {p}")
    return np.load(p)["images"]


@pytest.fixture(scope="module")
def yolo_outputs(eval_images):
    """Run YOLOv8n on every eval image once; return raw outputs."""
    yolo_path = REPO / "data" / "models" / "yolov8n.onnx"
    if not yolo_path.exists():
        pytest.skip(f"yolov8n.onnx missing: {yolo_path}")
    from astracore.backends.ort import OrtBackend
    be = OrtBackend()
    program = be.compile(str(yolo_path))
    input_name = program.input_names[0]
    outputs = []
    for i in range(eval_images.shape[0]):
        x = eval_images[i:i + 1].astype(np.float32)
        out = be.run(program, {input_name: x})
        outputs.append(np.asarray(next(iter(out.values()))).squeeze())
    return outputs


def test_baseline_has_28_unique_fingerprints(baseline):
    fps = baseline["per_image_sha256_prefix"]
    assert len(fps) == 28
    assert len(set(fps)) == 28, "duplicate fingerprints in baseline"


def test_per_image_fingerprints_match_baseline(baseline, yolo_outputs):
    """The 28 fingerprints must match the committed baseline.

    If this fails, either:
    * ORT version changed numerically (check onnxruntime release notes)
    * someone tweaked a session option that affects graph-opt path
    * the eval set or the ONNX file moved

    In none of these cases should you just update the baseline — look
    at WHAT changed first. If the divergence is benign (new ORT build),
    regenerate via ``scripts/run_realworld_scenarios.py --only 2``.
    """
    expected = baseline["per_image_sha256_prefix"]
    observed = [_fingerprint(o) for o in yolo_outputs]
    mismatches = [(i, e, o) for i, (e, o) in enumerate(zip(expected, observed))
                  if e != o]
    if mismatches:
        msg = (
            f"{len(mismatches)} of {len(expected)} fingerprints drifted. "
            f"First 3: " + ", ".join(
                f"img{i}: {e} -> {o}" for i, e, o in mismatches[:3]
            )
        )
        raise AssertionError(msg)


def test_yolo_output_shape_and_dtype(yolo_outputs):
    """Sanity: every per-image output has the YOLOv8 detect shape."""
    for i, out in enumerate(yolo_outputs):
        assert out.ndim == 2, f"img{i}: expected 2D, got {out.shape}"
        assert out.shape[0] == 84, (
            f"img{i}: expected 84 rows (4 box + 80 classes), got {out.shape[0]}"
        )
        assert out.dtype in (np.float32, np.float64)
