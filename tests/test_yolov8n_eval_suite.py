"""F1-T2 production-scale acceptance — batched detection evaluation
across the COCO-128 eval split (real images).

Compared to `test_yolov8n_detection.py` (single image, sanity gate)
this runs the full NPU pipeline side-by-side with the FP32 ORT
reference across **every eval-split image** and reports distributional
metrics. Produces a JSON artefact at `reports/yolov8n_eval.json`
that's committed as the production-baseline fingerprint for
regression tracking.

Production recipe (post-GAP-1, 2026-04-18):
  - 100-image calibration subset of COCO-128.
  - `CALIB_PERCENTILE` at `99.9999` percentile (1 in 1M clipped).
    Measured best-in-class across a 4-variant sweep. See
    `scripts/compare_calibration_methods.py`.
  - FP32 decode at default threshold 0.25, **NPU decode at 0.20**
    (asymmetric: INT8 systematically compresses scores by ~0.01-0.05,
    so borderline FP32 detections fall just below 0.25 on the NPU
    side. 0.20 recovers them without introducing false positives —
    sweep in GAP-1 showed 0.18 plateau, 0.20 is conservative).

Expensive: ~4–5 minutes of CPU on a typical dev box. Marked
`integration` so it can be skipped in fast-feedback pytest runs
(`pytest -m "not integration"`).

Gates observed 2026-04-18 with the full production recipe on the
28-image eval split:

    aggregate match @ IoU≥0.5     99.2 %
    aggregate match @ IoU≥0.7     98.4 %
    aggregate match @ IoU≥0.9     95.2 %
    median tensor SNR vs FP32     ~29 dB

Historical progression (all numbers on the same 28-image eval set):

    recipe                                          IoU≥0.5  IoU≥0.7  IoU≥0.9
    20-img calib, max-abs, sym thr 0.25               87.4%   86.6%   78.2%
    100-img calib, max-abs, sym thr 0.25              90.4%   88.8%   82.4%
    100-img calib, percentile-99.9999, sym thr 0.25   93.6%   92.0%   90.4%
    100-img calib, percentile-99.9999, NPU thr 0.20   99.2%   98.4%   95.2%  ← production

The gates below are set ~3 pp below observed to tolerate seed /
numerical drift without flaking.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from tools.npu_ref.detection_eval import evaluate_on_image_set
from tools.npu_ref.fusion import fuse_silu
from tools.npu_ref.onnx_loader import load_onnx
from tools.npu_ref.quantiser import CALIB_PERCENTILE, quantise_model

REPO = Path(__file__).resolve().parent.parent
ONNX_PATH = REPO / "data" / "models" / "yolov8n.onnx"
CALIB_NPZ = REPO / "data" / "calibration" / "yolov8n_calib.npz"
EVAL_NPZ  = REPO / "data" / "calibration" / "yolov8n_eval.npz"
REPORT_DIR = REPO / "reports"
REPORT_PATH = REPORT_DIR / "yolov8n_eval.json"

pytestmark = pytest.mark.integration  # opt-in: pytest -m integration


def _require_artifacts():
    missing = []
    for p in (ONNX_PATH, CALIB_NPZ, EVAL_NPZ):
        if not p.exists():
            missing.append(str(p.relative_to(REPO)))
    if missing:
        pytest.skip(
            f"missing artifacts ({', '.join(missing)}). Run "
            f"scripts/export_yolov8n_onnx.py and "
            f"scripts/fetch_yolo_calibration.py from the side venv."
        )


@pytest.fixture(scope="module")
def eval_result():
    _require_artifacts()
    cal = np.load(CALIB_NPZ)["images"]
    real_cal = [{"images": cal[i:i + 1]} for i in range(cal.shape[0])]

    g = load_onnx(str(ONNX_PATH))
    fuse_silu(g)
    t0 = time.perf_counter()
    quantise_model(g, str(ONNX_PATH), real_cal,
                    calibration_method=CALIB_PERCENTILE)  # 99.9999-ile default
    print(f"calibration wall: {time.perf_counter()-t0:.1f}s")

    images = np.load(EVAL_NPZ)["images"]
    print(f"running eval on {images.shape[0]} images...")
    result = evaluate_on_image_set(
        g, str(ONNX_PATH), images,
        iou_thresholds=(0.5, 0.7, 0.9),
        score_threshold=0.25,       # FP32 reference at YOLOv8 default
        npu_score_threshold=0.20,   # GAP-1 score-calibration
        progress=True,
    )
    print(f"eval wall: {result.wall_s_total:.1f}s "
          f"({result.wall_s_total / result.n_images:.2f}s/img)")

    # Write artefact before any assertion — so a gate failure still
    # leaves the evidence on disk for triage.
    REPORT_DIR.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    print(f"wrote report: {REPORT_PATH}")
    return result


def test_aggregate_match_rate_at_iou_05(eval_result):
    """Best-in-class at IoU≥0.5 (COCO standard). Production recipe
    observed 98-99%. 94% floor gives margin for reservoir variance."""
    rate = eval_result.match_rate(0.5)
    assert rate >= 0.94, (
        f"aggregate match rate @ IoU≥0.5 is {rate:.1%} — below the "
        f"94% best-in-class floor. Indicates calibration or score-"
        f"threshold regression."
    )


def test_aggregate_match_rate_at_iou_07(eval_result):
    """Best-in-class at IoU≥0.7. Production recipe observed 97-98%.
    The F1-C5 plan acceptance was 80% — we're ≥15 pp above, which is
    where TensorRT / OpenVINO INT8 PTQ typically lands on COCO."""
    rate = eval_result.match_rate(0.7)
    assert rate >= 0.93, (
        f"aggregate match rate @ IoU≥0.7 is {rate:.1%} — below the "
        f"93% best-in-class floor."
    )


def test_bbox_geometry_tight_at_iou_09(eval_result):
    """IoU≥0.9 — matched bboxes should be pixel-tight. Observed
    88-95% with the production recipe (reservoir sampling in the
    percentile path produces a few pp of run-to-run variance,
    concentrated at IoU≥0.9 where bbox precision is most sensitive
    to quant scale fluctuations). Floor 85% is below the observed
    low end but well above the 78.2% v1 (max-abs) baseline."""
    rate = eval_result.match_rate(0.9)
    assert rate >= 0.85, (
        f"aggregate match rate @ IoU≥0.9 is {rate:.1%} — below the "
        f"85% floor. Indicates bbox geometry has degraded below the "
        f"percentile-calibration baseline."
    )


def test_no_catastrophic_accuracy_loss(eval_result):
    """Production recipe typically hits ≥98% at IoU≥0.5 per-image.
    Anything below 93% is a regression worth investigating."""
    rates = eval_result.per_image_match_rates(0.5)
    assert len(rates) > 0, "no images with reference detections"
    nonzero = sum(1 for r in rates if r > 0)
    ratio = nonzero / len(rates)
    assert ratio >= 0.93, (
        f"only {ratio:.1%} of images found any detection at IoU≥0.5 — "
        f"{len(rates) - nonzero} complete failures out of {len(rates)}"
    )


def test_per_image_distribution_majority_pass_80(eval_result):
    """With the production recipe, most images hit near-100% match.
    85% floor is conservative against the observed ~93%+ rate."""
    rates = eval_result.per_image_match_rates(0.7)
    n_pass = sum(1 for r in rates if r >= 0.80)
    ratio = n_pass / len(rates) if rates else 0.0
    assert ratio >= 0.85, (
        f"only {ratio:.1%} of eval images hit ≥80% per-image match "
        f"rate at IoU≥0.7"
    )


def test_tensor_snr_distribution_within_floor(eval_result):
    """Per-image tensor SNR under percentile-99.9999 calibration.
    Unaffected by score-threshold tuning (which only changes decode)."""
    stats = eval_result.tensor_snr_stats()
    print(f"tensor SNR distribution: {stats}")
    assert stats["min"] >= 20.0, (
        f"worst-image tensor SNR {stats['min']:.1f} dB below 20 dB floor"
    )
    assert stats["median"] >= 25.0, (
        f"median tensor SNR {stats['median']:.1f} dB below 25 dB floor"
    )


def test_report_artefact_written(eval_result):
    """The JSON report is committed as the baseline fingerprint."""
    assert REPORT_PATH.exists()
    report = json.loads(REPORT_PATH.read_text())
    assert report["n_images"] == eval_result.n_images
    assert "aggregate_match_rate" in report
    assert "per_image" in report and len(report["per_image"]) == eval_result.n_images
