"""YOLOv8 object detection demo.

Uses the project's existing decoder in ``tools/npu_ref/yolo_decode.py``
so this demo shares the same post-processing path as the production
eval (``reports/yolov8n_eval.json``).
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

import numpy as np

from astracore.demo.base import DemoError, DemoResult, register_demo_family


COCO80_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)


@register_demo_family("vision-detection")
def run(zoo_entry, onnx_path: Path, *,
        input_spec: Optional[str] = None,
        backend_name: str = "onnxruntime") -> DemoResult:
    import astracore.backends  # noqa: F401
    from astracore.registry import get_backend
    from tools.npu_ref.yolo_decode import decode_yolov8_output  # lazy

    spec = input_spec or "bus"
    known = {"bus", "zidane"}
    if spec in known:
        npz_path = Path("data/calibration") / f"{spec}.npz"
    else:
        npz_path = Path(spec)
    if not npz_path.exists():
        raise DemoError(f"test image not found: {npz_path}")

    data = np.load(npz_path)
    key = "image" if "image" in data.files else data.files[0]
    img = data[key].astype(np.float32)
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    if img.ndim == 3:
        img = img[None, ...]
    if img.shape[1] != 3:   # HWC to NCHW
        img = np.transpose(img, (0, 3, 1, 2))

    import onnx
    model = onnx.load(str(onnx_path))
    be_cls = get_backend(backend_name)
    be = be_cls() if isinstance(be_cls, type) else be_cls

    init_names = {t.name for t in model.graph.initializer}
    real_inputs = [inp for inp in model.graph.input if inp.name not in init_names]
    input_name = real_inputs[0].name

    try:
        program = be.compile(model, concrete_shapes={input_name: tuple(img.shape)})
    except TypeError:
        program = be.compile(model)

    t0 = time.perf_counter()
    out = be.run(program, {input_name: img})
    wall_ms = (time.perf_counter() - t0) * 1e3

    raw = next(iter(out.values()))
    raw = np.asarray(raw)
    detections = decode_yolov8_output(raw, score_threshold=0.25)

    predictions = []
    for det in detections[:10]:
        cls_id = int(det.class_id)
        cls_name = COCO80_NAMES[cls_id] if cls_id < len(COCO80_NAMES) else f"cls{cls_id}"
        x1, y1, x2, y2 = det.bbox_xyxy
        predictions.append({
            "class_id": cls_id,
            "label": cls_name,
            "score": float(det.score),
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
        })

    if predictions:
        top = predictions[0]
        summary = (
            f"{len(predictions)} detections; "
            f"top: {top['label']} (score {top['score']:.2%}, "
            f"bbox=[{top['bbox_xyxy'][0]:.0f}, {top['bbox_xyxy'][1]:.0f}, "
            f"{top['bbox_xyxy'][2]:.0f}, {top['bbox_xyxy'][3]:.0f}])"
        )
    else:
        summary = "no detections above threshold"

    return DemoResult(
        model=zoo_entry.name,
        family=zoo_entry.family,
        backend=backend_name,
        input_source=spec,
        wall_ms=wall_ms,
        predictions=predictions,
        summary=summary,
        raw_shape=list(raw.shape),
    )
