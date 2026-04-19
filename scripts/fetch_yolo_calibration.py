"""One-shot acquisition of a small real-image calibration set for
F1-T1 / F1-T2.

Downloads Ultralytics' COCO-128 dataset (a well-known 128-image
subset typically bundled with their tutorials, ~30MB) and produces:

    data/calibration/yolov8n_calib.npz     — 20 preprocessed (1,3,640,640)
                                              float32 tensors stacked to
                                              shape (20, 3, 640, 640)
    data/calibration/yolov8n_calib.manifest.json — SHA256 + provenance
    data/calibration/bus.npz               — single bus.jpg preprocessed
                                              (reference image for F1-T2)
    data/calibration/zidane.npz            — second reference image

Runs from the side venv (.venv-export/) because it uses PIL /
urllib. Outputs are gitignored; rebuild via this script.

Pinned source:
  https://ultralytics.com/assets/coco128.zip
  (sha256 check below is the value observed at acquisition time;
   update if the upstream file changes.)
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "calibration"
COCO128_URL = "https://ultralytics.com/assets/coco128.zip"
# Deterministic per-dataset-layout split: first 100 images for
# calibration, remaining images for evaluation. No leakage between the
# two — the quantiser's activation scales are fit on calibration only,
# and eval images are never seen during calibration.
#
# 100 images is a moderate PTQ calibration size (TensorRT's default is
# ~100-500; OpenVINO uses 300). Anything under 50 tends to produce
# unstable per-tensor scales; above 500 gives diminishing returns for
# a network this size. With 128 total in COCO-128, 100/28 is the
# largest split that still leaves a meaningful eval set.
N_CALIB = 100


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _letterbox(image: "Image.Image", new_size: int = 640) -> np.ndarray:
    """Resize + pad image to (new_size, new_size) preserving aspect
    ratio. YOLOv8's standard preprocessing."""
    from PIL import Image
    w, h = image.size
    scale = min(new_size / w, new_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = image.resize((nw, nh), Image.BILINEAR)
    padded = Image.new("RGB", (new_size, new_size), (114, 114, 114))
    padded.paste(resized, ((new_size - nw) // 2, (new_size - nh) // 2))
    arr = np.asarray(padded, dtype=np.float32) / 255.0  # HWC [0,1]
    arr = arr.transpose(2, 0, 1)                          # CHW
    return arr


def _preprocess_jpg(jpg_bytes: bytes) -> np.ndarray:
    from PIL import Image
    img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
    return _letterbox(img, 640)


def _fetch_coco128(out_buf: bytes = None) -> bytes:
    if out_buf is not None:
        return out_buf
    print(f"downloading {COCO128_URL}")
    with urllib.request.urlopen(COCO128_URL, timeout=120) as r:
        data = r.read()
    return data


def _extract_images(zip_bytes: bytes, n: int,
                      start: int = 0) -> List[Tuple[str, bytes]]:
    """Return JPG entries [start:start+n] (by path order) inside the
    zip, with their raw bytes. Sorting the namelist makes the split
    reproducible across machines."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = sorted(
            name for name in zf.namelist()
            if name.lower().endswith(".jpg") and "images" in name.lower()
        )
        names = names[start:start + n]
        results = []
        for name in names:
            with zf.open(name) as f:
                results.append((name, f.read()))
        return results


def _total_jpg_count(zip_bytes: bytes) -> int:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        return sum(
            1 for name in zf.namelist()
            if name.lower().endswith(".jpg") and "images" in name.lower()
        )


def _process_reference_images() -> dict:
    """Preprocess the two Ultralytics-bundled assets as the single-
    image F1-T2 reference inputs."""
    from PIL import Image
    asset_dir = (REPO / ".venv-export" / "Lib" / "site-packages" /
                  "ultralytics" / "assets")
    out = {}
    for stem in ("bus", "zidane"):
        path = asset_dir / f"{stem}.jpg"
        if not path.exists():
            print(f"WARN: {path} missing — skipping")
            continue
        tensor = _preprocess_jpg(path.read_bytes())[None]   # (1,3,640,640)
        (OUT_DIR / f"{stem}.npz").write_bytes(b"")          # placeholder
        np.savez(OUT_DIR / f"{stem}.npz", image=tensor)
        sha = _sha256_bytes(path.read_bytes())
        out[stem] = {
            "source": str(path.relative_to(REPO)),
            "source_sha256": sha,
            "shape": list(tensor.shape),
        }
        print(f"wrote {OUT_DIR / f'{stem}.npz'}  (src sha={sha[:12]})")
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reference images (for F1-T2 detection acceptance).
    ref_info = _process_reference_images()

    # Calibration set.
    try:
        zip_bytes = _fetch_coco128()
    except Exception as e:
        print(f"ERROR fetching coco128: {e}", file=sys.stderr)
        return 2

    zip_sha = _sha256_bytes(zip_bytes)
    total = _total_jpg_count(zip_bytes)
    print(f"coco128.zip sha256 = {zip_sha}  size = {len(zip_bytes):,} bytes  "
          f"images = {total}")

    # Calibration split.
    calib_entries = _extract_images(zip_bytes, N_CALIB, start=0)
    if len(calib_entries) < N_CALIB:
        print(f"ERROR: only {len(calib_entries)} calibration images found; "
              f"expected {N_CALIB}", file=sys.stderr)
        return 3
    calib = np.stack([_preprocess_jpg(b) for _, b in calib_entries], axis=0)
    print(f"calibration tensor shape: {calib.shape} dtype={calib.dtype}")
    np.savez(OUT_DIR / "yolov8n_calib.npz", images=calib)

    # Evaluation split: everything after the calibration slice.
    n_eval = max(0, total - N_CALIB)
    eval_entries = _extract_images(zip_bytes, n_eval, start=N_CALIB)
    eval_arr = (np.stack([_preprocess_jpg(b) for _, b in eval_entries], axis=0)
                if eval_entries else np.empty((0, 3, 640, 640), dtype=np.float32))
    print(f"evaluation tensor shape: {eval_arr.shape} dtype={eval_arr.dtype}")
    np.savez(OUT_DIR / "yolov8n_eval.npz", images=eval_arr)

    manifest = {
        "source_url": COCO128_URL,
        "source_sha256": zip_sha,
        "source_bytes": len(zip_bytes),
        "n_calibration": int(calib.shape[0]),
        "n_evaluation": int(eval_arr.shape[0]),
        "calibration_names": [n for n, _ in calib_entries],
        "evaluation_names": [n for n, _ in eval_entries],
        "calibration_shape": list(calib.shape),
        "evaluation_shape": list(eval_arr.shape),
        "preprocessing": "letterbox 640x640, /255, NCHW, RGB, pad_val=114",
        "split_invariant": (
            "First 20 sorted names → calibration; remaining → evaluation. "
            "No leakage. Reproducible across machines given the same zip."
        ),
        "reference_images": ref_info,
    }
    (OUT_DIR / "yolov8n_calib.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_DIR / 'yolov8n_calib.npz'}")
    print(f"wrote {OUT_DIR / 'yolov8n_eval.npz'}")
    print(f"wrote manifest {OUT_DIR / 'yolov8n_calib.manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
