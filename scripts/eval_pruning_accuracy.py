"""SPARSE-2 — evaluate magnitude-pruned yolov8n variants against the
dense FP32 reference.

For each (n, m) pruning pattern produced by prune_yolov8n_nm.py:
  1. Run dense ONNX through ORT on 28 COCO-128 eval images → ref dets.
  2. Run pruned ONNX through ORT on same images → pruned dets.
  3. Compare via detection_match_rate.

Isolates pruning-only accuracy loss (no quantisation in this loop).
This is the **fastest possible signal** on whether 1:8 is tractable:
if magnitude-only 1:8 loses ≥20 pp vs FP32, QAT/retraining will have
a hard time closing it. ≤10 pp suggests QAT is a viable path.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.npu_ref.yolo_decode import decode_yolov8_output, match_detections

DENSE_ONNX = REPO / "data" / "models" / "yolov8n.onnx"
PRUNED_DIR = REPO / "data" / "models" / "pruned"
EVAL_NPZ = REPO / "data" / "calibration" / "yolov8n_eval.npz"
REPORT_PATH = REPO / "reports" / "pruning_accuracy.json"


def _make_session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts,
                                 providers=["CPUExecutionProvider"])


def eval_variant(model_path: Path, dense_sess: ort.InferenceSession,
                  images: np.ndarray, score_threshold: float = 0.25,
                  iou_thresholds=(0.5, 0.7, 0.9)) -> dict:
    sess = _make_session(model_path)
    totals = {t: [0, 0] for t in iou_thresholds}
    per_image = []
    t0 = time.perf_counter()
    for i in range(images.shape[0]):
        x = images[i:i + 1]
        fp32 = dense_sess.run(["output0"], {"images": x})[0]
        pruned = sess.run(["output0"], {"images": x})[0]
        ref_dets = decode_yolov8_output(fp32, score_threshold=score_threshold)
        pr_dets = decode_yolov8_output(pruned, score_threshold=score_threshold)
        row = {"i": i, "n_ref": len(ref_dets),
               "n_pred": len(pr_dets), "matched": {}}
        for thr in iou_thresholds:
            m, _, t = match_detections(ref_dets, pr_dets, iou_threshold=thr)
            row["matched"][f"{thr:.2f}"] = m
            totals[thr][0] += m
            totals[thr][1] += t
        per_image.append(row)
    wall = time.perf_counter() - t0
    agg = {}
    for thr in iou_thresholds:
        m, t = totals[thr]
        agg[f"iou>={thr:.2f}"] = m / t if t else 1.0
    return {
        "model": str(model_path.relative_to(REPO).as_posix()),
        "wall_s": wall,
        "aggregate_match_rate": agg,
        "per_image": per_image,
    }


def main() -> int:
    if not DENSE_ONNX.exists() or not EVAL_NPZ.exists():
        print("missing artefacts — run the export + fetch scripts first",
              file=sys.stderr)
        return 2
    images = np.load(EVAL_NPZ)["images"]
    dense_sess = _make_session(DENSE_ONNX)

    # Sanity: dense-vs-dense → 100 % match. Guards against decoder drift.
    self_check = eval_variant(DENSE_ONNX, dense_sess, images)
    print(f"self-check dense-vs-dense: "
          f"{self_check['aggregate_match_rate']}")

    manifest = json.loads((PRUNED_DIR / "pruning_manifest.json").read_text())
    results = {"eval_images": int(images.shape[0]),
               "dense_self_check": self_check["aggregate_match_rate"],
               "variants": []}
    for variant in manifest["variants"]:
        path = REPO / variant["path"]
        print(f"\n--- {variant['n']}:{variant['m']} ({path.name}) ---")
        r = eval_variant(path, dense_sess, images)
        print(f"  wall: {r['wall_s']:.1f}s")
        for k, v in r["aggregate_match_rate"].items():
            print(f"  {k}: {v:.1%}")
        r["n"] = variant["n"]
        r["m"] = variant["m"]
        r["sparsity"] = variant["overall_sparsity"]
        results["variants"].append(r)

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {REPORT_PATH.relative_to(REPO)}")

    # One-line summary.
    print("\n=== summary ===")
    print(f"{'pattern':<10s} {'sparsity':<10s} "
          f"{'IoU>=0.5':<10s} {'IoU>=0.7':<10s} {'IoU>=0.9':<10s}")
    for v in results["variants"]:
        tag = f"{v['n']}:{v['m']}"
        print(f"{tag:<10s} {v['sparsity']:<10.3f} "
              f"{v['aggregate_match_rate']['iou>=0.50']:<10.1%} "
              f"{v['aggregate_match_rate']['iou>=0.70']:<10.1%} "
              f"{v['aggregate_match_rate']['iou>=0.90']:<10.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
