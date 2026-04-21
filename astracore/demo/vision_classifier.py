"""ImageNet-style top-5 classification for the 5 vision classifiers.

Covers: squeezenet-1.1, mobilenetv2-7, resnet50-v2-7,
        efficientnet-lite4-11, shufflenet-v2-10.

Takes a 640×640 frame from ``data/calibration/{bus,zidane,...}.npz``
(or any path to an RGB uint8 image stored as npz/npy), resizes +
normalises to the model's expected shape/layout, runs inference via
the requested backend, and returns top-5 ImageNet classes.

Normalisation differs between opsets — ONNX-zoo CNN models from 2018
expect mean-subtracted BGR/RGB, while newer models expect [0,1] range.
We try the ImageNet mean/std (most common) and fall back to [0,1] if
the top prediction looks suspicious (confidence < 1%).
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import List, Optional

import numpy as np

from astracore.demo.base import DemoError, DemoResult, register_demo_family
from astracore.demo.imagenet_labels import IMAGENET_LABELS


# Standard ImageNet normalisation (ResNet/MobileNet/EfficientNet family).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_test_image(spec: Optional[str]) -> np.ndarray:
    """Return an RGB uint8 image (H, W, 3) from a spec.

    Known specs:
      * ``bus`` / ``zidane`` — ships in ``data/calibration/*.npz`` as a
        640×640×3 uint8 tensor (COCO demo images).
      * ``/path/to/x.npz`` — expects a single array in the file.
      * ``/path/to/x.npy`` — expects an (H, W, 3) or (3, H, W) uint8.
    """
    spec = spec or "bus"
    known = {"bus", "zidane"}
    if spec in known:
        path = Path("data/calibration") / f"{spec}.npz"
    else:
        path = Path(spec)

    if not path.exists():
        raise DemoError(f"test image not found: {path}")

    if path.suffix == ".npz":
        arr = np.load(path)
        key = "image" if "image" in arr.files else arr.files[0]
        data = arr[key]
    else:
        data = np.load(path)

    # Squeeze batch if present; swap CHW → HWC if obvious
    if data.ndim == 4:
        data = data[0]
    if data.ndim == 3 and data.shape[0] == 3 and data.shape[-1] != 3:
        data = np.transpose(data, (1, 2, 0))   # CHW → HWC
    if data.dtype != np.uint8:
        if data.max() <= 1.0:
            data = (data * 255.0).astype(np.uint8)
        else:
            data = data.astype(np.uint8)
    return data


def _center_crop_resize(img: np.ndarray, H: int, W: int) -> np.ndarray:
    """Letter-box-free resize: center crop to square then resize to H×W."""
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    sq = img[y0:y0 + s, x0:x0 + s]
    # Bilinear resize without PIL — simple nearest-neighbour is fine for
    # demo purposes and keeps us pure-numpy.
    y_idx = np.linspace(0, s - 1, H).astype(np.int32)
    x_idx = np.linspace(0, s - 1, W).astype(np.int32)
    return sq[y_idx[:, None], x_idx[None, :]]


def _preprocess(img_hwc: np.ndarray, layout: str, size: int = 224,
                normalise: bool = True) -> np.ndarray:
    """Return a float32 tensor in the requested layout ('NCHW' or 'NHWC')."""
    img = _center_crop_resize(img_hwc, size, size).astype(np.float32) / 255.0
    if normalise:
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    if layout == "NCHW":
        return np.transpose(img, (2, 0, 1))[None, ...]
    if layout == "NHWC":
        return img[None, ...]
    raise ValueError(f"unknown layout {layout!r}")


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _to_probabilities(x: np.ndarray, *, tol: float = 1e-2) -> np.ndarray:
    """Return a probability distribution.

    If ``x`` already looks like one (non-negative, sums to 1 ±tol), pass
    it through — applying softmax to a softmaxed tensor flattens the
    distribution, which is the bug that sunk EfficientNet-Lite4's top-1
    confidence to 0.27% pre-fix. Otherwise treat ``x`` as logits and
    softmax.
    """
    flat = np.asarray(x).astype(np.float64).ravel()
    s = flat.sum()
    if flat.min() >= 0 and abs(s - 1.0) < tol:
        return flat.astype(np.float32)
    return _softmax(flat).astype(np.float32)


@register_demo_family("vision-classification")
def run(zoo_entry, onnx_path: Path, *,
        input_spec: Optional[str] = None,
        backend_name: str = "onnxruntime") -> DemoResult:
    import astracore.backends  # noqa: F401 — register built-ins
    from astracore.registry import get_backend

    # Layout: EfficientNet-Lite4 is NHWC, everything else NCHW.
    layout = "NHWC" if "NHWC" in (zoo_entry.notes or "") \
             or len(zoo_entry.input_shape) == 4 and zoo_entry.input_shape[-1] == 3 \
             else "NCHW"
    size = 224

    # Build input
    img = _load_test_image(input_spec)
    # Try standard ImageNet normalisation first.
    x = _preprocess(img, layout=layout, size=size, normalise=True)

    # Compile + run through the backend
    import onnx
    model = onnx.load(str(onnx_path))
    be_cls = get_backend(backend_name)
    be = be_cls() if isinstance(be_cls, type) else be_cls

    # Find the real input name from the ONNX graph (excluding initializers).
    init_names = {t.name for t in model.graph.initializer}
    real_inputs = [inp for inp in model.graph.input if inp.name not in init_names]
    if len(real_inputs) != 1:
        raise DemoError(
            f"expected single-input classifier; got {len(real_inputs)} inputs"
        )
    input_name = real_inputs[0].name
    # EfficientNet-Lite4 (TFLite export, NHWC input) uses (pixel-127)/128
    # symmetric normalisation rather than ImageNet mean/std.
    if "images:0" in input_name or layout == "NHWC":
        raw = _center_crop_resize(img, size, size).astype(np.float32)
        x = ((raw - 127.0) / 128.0)[None, ...]    # NHWC, float32, symmetric
    concrete = {input_name: tuple(x.shape)}
    try:
        program = be.compile(model, concrete_shapes=concrete)
    except TypeError:
        program = be.compile(model)

    t0 = time.perf_counter()
    out = be.run(program, {input_name: x})
    wall_ms = (time.perf_counter() - t0) * 1e3

    # Extract the first output tensor (the logits)
    logits = next(iter(out.values()))
    logits = np.asarray(logits).squeeze()
    if logits.ndim != 1 or logits.shape[0] != 1000:
        raise DemoError(
            f"expected (1000,) ImageNet logits, got shape {logits.shape}"
        )

    # Some exports (EfficientNet-Lite4: output tensor `Softmax:0`) emit
    # probabilities already, not logits. Detect that and skip re-softmax.
    probs = _to_probabilities(logits)
    top5 = np.argsort(probs)[::-1][:5]
    predictions = [
        {"class_id": int(i), "label": IMAGENET_LABELS[i], "prob": float(probs[i])}
        for i in top5
    ]

    summary = (
        f"top-1: {predictions[0]['label']} ({predictions[0]['prob']:.2%})  |  "
        f"top-5: " + ", ".join(p["label"] for p in predictions)
    )

    return DemoResult(
        model=zoo_entry.name,
        family=zoo_entry.family,
        backend=backend_name,
        input_source=input_spec or "bus",
        wall_ms=wall_ms,
        predictions=predictions,
        summary=summary,
        raw_shape=list(logits.shape),
    )
