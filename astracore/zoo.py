"""Model zoo — curated ONNX models for OEM evaluation.

Each entry describes one public ONNX file the SDK can load + benchmark.
Entries are pure metadata; downloads are orchestrated by
``scripts/fetch_model_zoo.py`` or ``astracore zoo fetch``.

OEMs pointing at their own model library can extend this registry via
the ``astracore.zoo`` entry-point (planned for Phase A.5.1); for now,
drop a second manifest JSON into ``data/models/zoo/`` and the fetcher
picks it up.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ZooModel:
    name: str                           # short stable ID (filename-safe)
    display_name: str                   # pretty name for reports
    family: str                         # "vision-classification", "vision-detection", ...
    url: Optional[str]                  # upstream download URL; None ⇒ local-only
    sha256: Optional[str]               # expected SHA-256 of the ONNX file
    size_bytes: Optional[int]           # approximate file size
    input_name: str                     # first input tensor name
    input_shape: Tuple[int, ...]        # NCHW or equivalent
    opset: int                          # ONNX opset version
    notes: str = ""

    @property
    def local_path(self) -> Path:
        return Path("data/models/zoo") / f"{self.name}.onnx"


# ---------------------------------------------------------------------------
# Curated zoo — 6 entries covering vision classification + detection.
# Sizes kept modest so the whole zoo downloads in < 5 min on a typical link.
# ---------------------------------------------------------------------------

_ONNX_ZOO = "https://github.com/onnx/models/raw/main/validated"

ZOO: List[ZooModel] = [
    ZooModel(
        name="squeezenet-1.1",
        display_name="SqueezeNet 1.1",
        family="vision-classification",
        url=f"{_ONNX_ZOO}/vision/classification/squeezenet/model/squeezenet1.1-7.onnx",
        sha256=None,   # populated on first fetch; see scripts/fetch_model_zoo.py
        size_bytes=4_956_208,
        input_name="data",
        input_shape=(1, 3, 224, 224),
        opset=7,
        notes="Tiny CNN; smoke-test baseline",
    ),
    ZooModel(
        name="mobilenetv2-7",
        display_name="MobileNet V2",
        family="vision-classification",
        url=f"{_ONNX_ZOO}/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
        sha256=None,
        size_bytes=13_964_571,
        input_name="data",
        input_shape=(1, 3, 224, 224),
        opset=7,
        notes="Depthwise-separable convs; mobile-optimised",
    ),
    ZooModel(
        name="resnet50-v2-7",
        display_name="ResNet-50 v2",
        family="vision-classification",
        url=f"{_ONNX_ZOO}/vision/classification/resnet/model/resnet50-v2-7.onnx",
        sha256=None,
        size_bytes=102_149_934,
        input_name="data",
        input_shape=(1, 3, 224, 224),
        opset=7,
        notes="Residual CNN; ImageNet benchmark baseline",
    ),
    ZooModel(
        name="efficientnet-lite4-11",
        display_name="EfficientNet-Lite4",
        family="vision-classification",
        url=f"{_ONNX_ZOO}/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
        sha256=None,
        size_bytes=52_459_323,
        input_name="images:0",
        input_shape=(1, 224, 224, 3),
        opset=11,
        notes="Mobile/edge-optimised; NHWC input",
    ),
    ZooModel(
        name="shufflenet-v2-10",
        display_name="ShuffleNet V2",
        family="vision-classification",
        url=f"{_ONNX_ZOO}/vision/classification/shufflenet/model/shufflenet-v2-10.onnx",
        sha256=None,
        size_bytes=9_227_219,
        input_name="input",
        input_shape=(1, 3, 224, 224),
        opset=10,
        notes="Low-compute group convs",
    ),
    ZooModel(
        name="yolov8n",
        display_name="YOLOv8-N",
        family="vision-detection",
        url=None,           # already shipped in-repo at data/models/yolov8n.onnx
        sha256=None,
        size_bytes=None,
        input_name="images",
        input_shape=(1, 3, 640, 640),
        opset=17,
        notes="Object detection; production reference workload",
    ),
    ZooModel(
        name="bert-squad-10",
        display_name="BERT-Squad",
        family="nlp-encoder-transformer",
        url=f"{_ONNX_ZOO}/text/machine_comprehension/bert-squad/model/bertsquad-10.onnx",
        sha256=None,
        size_bytes=435_900_000,
        input_name="unique_ids_raw_output___9:0",
        input_shape=(1,),     # placeholder — see notes, exercises attention
        opset=10,
        notes=("Encoder transformer (MHA, LayerNorm, GELU). Multi-input "
               "(input_ids, input_mask, segment_ids); benchmark harness "
               "populates the remaining inputs with zeros."),
    ),
    ZooModel(
        name="gpt-2-10",
        display_name="GPT-2 (117M)",
        family="nlp-decoder-transformer",
        url=f"{_ONNX_ZOO}/text/machine_comprehension/gpt-2/model/gpt2-10.onnx",
        sha256=None,
        size_bytes=548_200_000,
        input_name="input1",
        input_shape=(1, 1, 8),   # (batch, seq, token-id) — small prompt
        opset=10,
        notes=("Decoder-only transformer with causal attention — same "
               "architecture family as LLaMA. Proves LLaMA-style "
               "transformer inference end-to-end."),
    ),
]


# Lookup helpers ------------------------------------------------------------

_BY_NAME = {m.name: m for m in ZOO}


def get(name: str) -> ZooModel:
    if name not in _BY_NAME:
        raise KeyError(f"Unknown zoo model {name!r}; known: {sorted(_BY_NAME)}")
    return _BY_NAME[name]


def all_models() -> List[ZooModel]:
    return list(ZOO)


def local_paths() -> Dict[str, Path]:
    """Return {name: path} for every zoo entry — existence not guaranteed."""
    out: Dict[str, Path] = {}
    for m in ZOO:
        if m.name == "yolov8n":
            out[m.name] = Path("data/models/yolov8n.onnx")
        else:
            out[m.name] = m.local_path
    return out


def available() -> List[ZooModel]:
    """Subset of the zoo whose ONNX file is actually on disk."""
    paths = local_paths()
    return [m for m in ZOO if paths[m.name].exists()]


def as_manifest_dicts() -> List[Dict]:
    return [asdict(m) for m in ZOO]
