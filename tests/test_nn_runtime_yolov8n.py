"""F1-C5 acceptance — end-to-end NPU pipeline on real YOLOv8-N.

Gate: `nn_runtime.run_graph` on a quantised yolov8n NnGraph must
produce the same output as F1-C2's fake-quant ONNX running under
onnxruntime (within tight numerical tolerance — these two paths
implement the same INT8 quant recipe so agreement should be sharp).

If this passes, the full F1-C compile→execute stack is numerically
correct end-to-end: loader (F1-C1) → SiLU fusion (F1-C1c) →
quantiser (F1-C2) → layer-by-layer execution engine (F1-C5) → same
output as the reference fake-quant ONNX.

Full mAP / Top-k / bbox-IoU gates per the plan belong to F1-T2's
detection harness. F1-C5's bar is "our runtime agrees with the
proven-correct fake-quant ORT output".

Observed 2026-04-18 with 20-batch seeded-noise calibration:

    end-to-end SNR (runtime vs fake-quant)     43.6 dB
    end-to-end cosine                          0.999978
    runtime wall time                          ~3.4s (640×640 input)

Thresholds below include margin above the observed values so seed
or numerical-order drift doesn't flake the test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.npu_ref.fake_quant_model import build_fake_quant_model
from tools.npu_ref.fusion import fuse_silu
from tools.npu_ref.nn_runtime import run_graph
from tools.npu_ref.onnx_loader import load_onnx
from tools.npu_ref.onnx_reference import make_seeded_input, run_reference
from tools.npu_ref.quantiser import (
    make_seeded_calibration_set,
    quantise_model,
)

REPO = Path(__file__).resolve().parent.parent
ONNX_PATH = REPO / "data" / "models" / "yolov8n.onnx"


def _require_artifact():
    if not ONNX_PATH.exists():
        pytest.skip(
            f"{ONNX_PATH} not present. Run scripts/export_yolov8n_onnx.py."
        )


def _snr_db(ref: np.ndarray, meas: np.ndarray) -> float:
    err = ref - meas
    num = float((ref ** 2).mean())
    den = max(float((err ** 2).mean()), 1e-30)
    return 10.0 * np.log10(num / den)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    num = float(a.ravel() @ b.ravel())
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    return num / den


@pytest.fixture(scope="module")
def e2e_setup(tmp_path_factory):
    _require_artifact()
    g = load_onnx(str(ONNX_PATH))
    fuse_silu(g)
    cal = make_seeded_calibration_set(
        "images", (1, 3, 640, 640), n_batches=20, seed=0)
    quantise_model(g, str(ONNX_PATH), cal)
    fq_path = tmp_path_factory.mktemp("fq") / "yolov8n.fq.onnx"
    build_fake_quant_model(g, str(ONNX_PATH), out_path=str(fq_path))
    return g, str(fq_path)


def test_runtime_matches_fake_quant_ort(e2e_setup):
    """The F1-C5 gate: nn_runtime output ≈ fake-quant ORT output
    on a fixed seeded input. Both paths model the same INT8 quant
    recipe, so agreement should be tight (≥40 dB SNR in practice —
    the 35 dB floor below is conservative)."""
    g, fq_path = e2e_setup
    probe = make_seeded_input((1, 3, 640, 640), seed=99)

    # Golden: fake-quant ONNX through ORT.
    fq_out = run_reference(str(ONNX_PATH), {"images": probe})  # not used
    fq_out = run_reference(fq_path, {"images": probe}).outputs["output0"]

    # Measured: our layer-by-layer engine.
    rt_out = run_graph(g, {"images": probe})["output0"]

    snr = _snr_db(fq_out, rt_out)
    cos = _cosine(fq_out, rt_out)
    print(f"runtime vs fake-quant  SNR={snr:.2f} dB  cosine={cos:.6f}")

    assert rt_out.shape == fq_out.shape == (1, 84, 8400)
    assert snr >= 35.0, (
        f"nn_runtime SNR {snr:.2f} dB vs fake-quant ORT below 35 dB "
        f"floor — the two paths should produce nearly identical "
        f"results because they model the same quantisation recipe. "
        f"A big divergence indicates a dispatch / dequant / order-of-"
        f"ops bug."
    )
    assert cos >= 0.999, (
        f"cosine {cos:.6f} below 0.999 floor — output structure "
        f"diverged, not just a magnitude drift"
    )


def test_runtime_and_fake_quant_within_rounding_of_fp32(e2e_setup):
    """Sanity that the engine path gives similar magnitude-level
    agreement with FP32 reference as fake-quant ORT does (same bar
    as F1-C2's acceptance). Indirectly confirms the engine's quant
    chain is correct."""
    g, _ = e2e_setup
    probe = make_seeded_input((1, 3, 640, 640), seed=99)
    fp32_out = run_reference(str(ONNX_PATH),
                              {"images": probe}).outputs["output0"]
    rt_out = run_graph(g, {"images": probe})["output0"]

    snr = _snr_db(fp32_out, rt_out)
    cos = _cosine(fp32_out, rt_out)
    print(f"runtime vs FP32        SNR={snr:.2f} dB  cosine={cos:.6f}")
    # Same thresholds as F1-C2's fake-quant-vs-FP32 test.
    assert snr >= 33.0
    assert cos >= 0.998
