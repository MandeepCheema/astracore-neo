"""Side-by-side quantiser acceptance comparison: max-abs vs percentile
calibration on the same eval split (BIC-3).

Runs the full detection eval twice using the same graph + eval images,
once per calibration method, and writes two JSON report artefacts
plus a printed summary diff.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# The script is run from the repo root via `python scripts/...` —
# add the repo to sys.path so `tools.npu_ref.*` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.npu_ref.detection_eval import evaluate_on_image_set
from tools.npu_ref.fusion import fuse_silu
from tools.npu_ref.onnx_loader import load_onnx
from tools.npu_ref.quantiser import (
    CALIB_MAX_ABS,
    CALIB_PERCENTILE,
    quantise_model,
)

REPO = Path(__file__).resolve().parent.parent
ONNX_PATH = REPO / "data" / "models" / "yolov8n.onnx"
CALIB_NPZ = REPO / "data" / "calibration" / "yolov8n_calib.npz"
EVAL_NPZ = REPO / "data" / "calibration" / "yolov8n_eval.npz"
REPORT_DIR = REPO / "reports"


def run_variant(label: str, method: str,
                 calib_batches, eval_images,
                 percentile: float = 99.99) -> dict:
    print(f"\n=== {label} ({method}) ===")
    g = load_onnx(str(ONNX_PATH))
    fuse_silu(g)
    t0 = time.perf_counter()
    quantise_model(g, str(ONNX_PATH), calib_batches,
                    calibration_method=method,
                    percentile=percentile)
    print(f"  calibration wall: {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    result = evaluate_on_image_set(
        g, str(ONNX_PATH), eval_images,
        iou_thresholds=(0.5, 0.7, 0.9),
        score_threshold=0.25,
        progress=True,
    )
    print(f"  eval wall: {result.wall_s_total:.1f}s "
          f"({result.wall_s_total / result.n_images:.2f}s/img)")

    report = result.to_dict()
    report["label"] = label
    report["calibration_method"] = method
    if method == CALIB_PERCENTILE:
        report["percentile"] = percentile
    report["n_calibration_batches"] = len(calib_batches)

    out_path = REPORT_DIR / f"yolov8n_eval_{label}.json"
    REPORT_DIR.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"  wrote {out_path}")
    return report


def _match(report, thr: float) -> float:
    return report["aggregate_match_rate"][f"iou>={thr:.2f}"]


def _snr(report) -> dict:
    return report["tensor_snr_db"]


def main():
    cal = np.load(CALIB_NPZ)["images"]
    ev = np.load(EVAL_NPZ)["images"]
    calib = [{"images": cal[i:i + 1]} for i in range(cal.shape[0])]
    print(f"calibration: {cal.shape[0]} images   evaluation: {ev.shape[0]} images")

    # Load already-computed variants if present (saves ~3 min each).
    def _load_or_run(label, method, p=99.99):
        path = REPORT_DIR / f"yolov8n_eval_{label}.json"
        if path.exists():
            print(f"(cached) {label}")
            return json.loads(path.read_text())
        return run_variant(label, method, calib, ev, percentile=p)

    max_abs = _load_or_run("100cal_maxabs", CALIB_MAX_ABS)

    variants = [
        ("100cal_pct9999",   99.99),
        ("100cal_pct99999",  99.999),
        ("100cal_pct999999", 99.9999),
    ]
    percentile_reports = []
    for label, p in variants:
        r = _load_or_run(label, CALIB_PERCENTILE, p=p)
        percentile_reports.append((label, p, r))

    print("\n=== diff ===")
    header = f"{'metric':<40s} {'max_abs':>10s}"
    for label, _, _ in percentile_reports:
        header += f"  {label[8:]:>10s}"
    print(header)
    for thr in (0.5, 0.7, 0.9):
        row = f"{'agg match @ IoU>=' + str(thr):<40s} {_match(max_abs, thr):>9.1%} "
        for _, _, r in percentile_reports:
            row += f"  {_match(r, thr):>9.1%} "
        print(row)
    for key in ("min", "median", "mean"):
        row = f"{'tensor SNR dB (' + key + ')':<40s} {_snr(max_abs)[key]:>10.2f}"
        for _, _, r in percentile_reports:
            row += f"  {_snr(r)[key]:>10.2f}"
        print(row)


if __name__ == "__main__":
    main()
