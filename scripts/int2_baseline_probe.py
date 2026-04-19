"""OPT-D-1 — INT2 baseline probe on yolov8n without QAT.

Fast signal on whether Option D (INT2 + 2:4) has a chance before we
commit 5-7 weeks to the QAT infrastructure. Three-way comparison on
the 28-image COCO-128 eval split:

    INT8  — current production recipe (percentile-99.9999 + NPU thr 0.20)
    INT4  — F1-B2's already-validated path
    INT2  — aggressive; this is what Option D needs

Uses fake_quant_model + ORT (no nn_runtime changes required) — the
QDQ-ONNX path is precision-aware via quantise_model's precision arg.

If INT2 alone loses 50+ pp vs FP32, QAT is climbing a very steep hill.
If ≤20 pp, combined INT2+2:4 with QAT is realistic.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import onnxruntime as ort

from tools.npu_ref.fake_quant_model import build_fake_quant_model
from tools.npu_ref.fusion import fuse_silu
from tools.npu_ref.onnx_loader import load_onnx
from tools.npu_ref.quantiser import CALIB_PERCENTILE, quantise_model
from tools.npu_ref.yolo_decode import decode_yolov8_output, match_detections

DENSE_ONNX = REPO / "data" / "models" / "yolov8n.onnx"
CALIB_NPZ = REPO / "data" / "calibration" / "yolov8n_calib.npz"
EVAL_NPZ = REPO / "data" / "calibration" / "yolov8n_eval.npz"
REPORT_PATH = REPO / "reports" / "int2_baseline.json"


def _sess(path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(path, sess_options=opts,
                                 providers=["CPUExecutionProvider"])


def eval_variant(label: str, fq_path: str,
                  dense_sess: ort.InferenceSession,
                  images: np.ndarray,
                  fp32_thr: float = 0.25,
                  npu_thr: float = 0.20) -> dict:
    sess = _sess(fq_path)
    totals = {t: [0, 0] for t in (0.5, 0.7, 0.9)}
    t0 = time.perf_counter()
    zero_det_imgs = 0
    max_score = 0.0
    for i in range(images.shape[0]):
        x = images[i:i + 1]
        fp32 = dense_sess.run(["output0"], {"images": x})[0]
        fq = sess.run(["output0"], {"images": x})[0]
        max_score = max(max_score, float(fq[0, 4:].max()))
        ref = decode_yolov8_output(fp32, score_threshold=fp32_thr)
        pred = decode_yolov8_output(fq, score_threshold=npu_thr)
        if len(pred) == 0 and len(ref) > 0:
            zero_det_imgs += 1
        for thr in totals:
            m, _, t = match_detections(ref, pred, iou_threshold=thr)
            totals[thr][0] += m
            totals[thr][1] += t
    wall = time.perf_counter() - t0
    agg = {f"iou>={t:.2f}": totals[t][0] / totals[t][1] if totals[t][1] else 1.0
           for t in totals}
    return {
        "label": label,
        "fq_path": fq_path,
        "wall_s": wall,
        "agg": agg,
        "zero_det_images": zero_det_imgs,
        "max_class_score_any_image": max_score,
    }


def run_precision(precision: str, calib_batches, eval_images,
                   dense_sess, tmp_dir: Path) -> dict:
    print(f"\n--- precision={precision} ---")
    g = load_onnx(str(DENSE_ONNX))
    fuse_silu(g)
    t0 = time.perf_counter()
    quantise_model(g, str(DENSE_ONNX), calib_batches,
                    precision=precision,
                    calibration_method=CALIB_PERCENTILE)
    print(f"  calibration: {time.perf_counter() - t0:.1f}s")
    fq_path = tmp_dir / f"yolov8n.{precision}.onnx"
    t0 = time.perf_counter()
    build_fake_quant_model(g, str(DENSE_ONNX), out_path=str(fq_path))
    print(f"  fake-quant build: {time.perf_counter() - t0:.1f}s")
    result = eval_variant(precision, str(fq_path),
                            dense_sess, eval_images)
    for k, v in result["agg"].items():
        print(f"  {k}: {v:.1%}")
    print(f"  zero-detection images: {result['zero_det_images']}/"
          f"{eval_images.shape[0]}  "
          f"max class score seen: {result['max_class_score_any_image']:.4f}")
    return result


def main() -> int:
    for p in (DENSE_ONNX, CALIB_NPZ, EVAL_NPZ):
        if not p.exists():
            print(f"ERROR: {p} missing", file=sys.stderr)
            return 2
    cal = np.load(CALIB_NPZ)["images"]
    eval_imgs = np.load(EVAL_NPZ)["images"]
    real_cal = [{"images": cal[i:i + 1]} for i in range(cal.shape[0])]
    dense_sess = _sess(str(DENSE_ONNX))
    tmp_dir = REPO / "reports" / "int2_probe_artefacts"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for precision in ("int8", "int4", "int2"):
        all_results.append(
            run_precision(precision, real_cal, eval_imgs, dense_sess, tmp_dir)
        )

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps({"results": all_results}, indent=2) + "\n")
    print(f"\nwrote {REPORT_PATH.relative_to(REPO)}")

    print("\n=== summary ===")
    print(f"{'precision':<10s} {'IoU>=0.5':<10s} {'IoU>=0.7':<10s} "
          f"{'IoU>=0.9':<10s} {'zero-det':<10s} {'max-score':<10s}")
    for r in all_results:
        print(f"{r['label']:<10s} "
              f"{r['agg']['iou>=0.50']:<10.1%} "
              f"{r['agg']['iou>=0.70']:<10.1%} "
              f"{r['agg']['iou>=0.90']:<10.1%} "
              f"{r['zero_det_images']:<10d} "
              f"{r['max_class_score_any_image']:<10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
