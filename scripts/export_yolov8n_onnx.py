"""One-shot reproducible export of YOLOv8-N to ONNX.

Run from a dedicated venv that has `ultralytics` installed (see
scripts/export_setup.md or the F1-C1 task notes). The core project venv
(.venv/) deliberately does NOT carry ultralytics + torch so the dev
loop stays lean.

Usage:
    .venv-export/Scripts/python.exe scripts/export_yolov8n_onnx.py

Output:
    data/models/yolov8n.onnx             — the exported model
    data/models/yolov8n.manifest.json    — provenance record

The manifest is the gate the rest of the compiler checks against. F1-C1
acceptance test (tests/test_onnx_yolov8n.py) refuses to run if the
on-disk SHA256 doesn't match the manifest, so the demo is always
reproducible from this script.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO / "data" / "models"
ONNX_PATH = MODELS_DIR / "yolov8n.onnx"
PT_PATH = MODELS_DIR / "yolov8n.pt"
MANIFEST = MODELS_DIR / "yolov8n.manifest.json"

# Export parameters — pinned so every rebuild is bit-identical given the
# same ultralytics + torch versions.
IMGSZ = 640
OPSET = 17
DYNAMIC = False
SIMPLIFY = True
BATCH = 1


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as e:
        print(
            "ERROR: ultralytics / torch missing from this venv.\n"
            "This script is meant to run from a side venv dedicated to\n"
            "model export, e.g.:\n"
            "    python -m venv .venv-export\n"
            "    .venv-export/Scripts/pip install ultralytics\n"
            "    .venv-export/Scripts/python scripts/export_yolov8n_onnx.py\n"
            f"\nOriginal ImportError: {e}",
            file=sys.stderr,
        )
        return 2

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Ultralytics downloads yolov8n.pt to CWD by default; we want it in
    # data/models/ so it can be cleaned up in one place.
    cwd_before = os.getcwd()
    os.chdir(MODELS_DIR)
    try:
        print(f"Loading yolov8n.pt (ultralytics {ultralytics.__version__}, "
              f"torch {torch.__version__})")
        model = YOLO("yolov8n.pt")

        print(f"Exporting to ONNX (opset={OPSET}, imgsz={IMGSZ}, "
              f"simplify={SIMPLIFY}, dynamic={DYNAMIC}, batch={BATCH})")
        exported_path = model.export(
            format="onnx",
            imgsz=IMGSZ,
            opset=OPSET,
            simplify=SIMPLIFY,
            dynamic=DYNAMIC,
            batch=BATCH,
        )
    finally:
        os.chdir(cwd_before)

    # Ultralytics returns a relative path ("yolov8n.onnx") relative to
    # the CWD at export time (MODELS_DIR). Resolve against MODELS_DIR
    # explicitly — we've since chdir'd back.
    exported_path = Path(exported_path)
    if not exported_path.is_absolute():
        exported_path = MODELS_DIR / exported_path.name
    if not exported_path.exists():
        raise FileNotFoundError(
            f"ultralytics reported export to {exported_path!s} but the "
            f"file is missing."
        )
    if exported_path.resolve() != ONNX_PATH.resolve():
        shutil.move(str(exported_path), str(ONNX_PATH))

    onnx_hash = _sha256(ONNX_PATH)
    pt_hash = _sha256(PT_PATH) if PT_PATH.exists() else None

    manifest = {
        "model": "yolov8n",
        "source": "ultralytics",
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "export_params": {
            "imgsz": IMGSZ,
            "opset": OPSET,
            "dynamic": DYNAMIC,
            "simplify": SIMPLIFY,
            "batch": BATCH,
        },
        "onnx_path": str(ONNX_PATH.relative_to(REPO).as_posix()),
        "onnx_sha256": onnx_hash,
        "onnx_bytes": ONNX_PATH.stat().st_size,
        "pt_sha256": pt_hash,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {ONNX_PATH}  ({ONNX_PATH.stat().st_size:,} bytes)")
    print(f"wrote {MANIFEST}")
    print(f"sha256 = {onnx_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
