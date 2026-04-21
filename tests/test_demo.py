"""Tests for the demo dispatcher + per-family handlers.

Each test asserts the demo produces a semantically sensible output,
not just runs without error — that's the whole point of the demo
(vs the zoo benchmark which only measures speed).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astracore import zoo
from astracore.demo import DemoResult, run_demo
from astracore.demo.base import register_demo_family, get_demo_handler, DemoError


def _require(name: str) -> Path:
    path = zoo.local_paths()[name]
    if not path.exists():
        pytest.skip(f"{name} zoo entry not downloaded")
    return path


# ---------------------------------------------------------------------------
# Vision classifiers — bus image must produce a vehicle-class top-1
# ---------------------------------------------------------------------------

_VEHICLE_LABELS = {
    "minibus", "police van", "trolleybus", "tram", "amphibious vehicle",
    "jeep", "minivan", "recreational vehicle", "school bus", "passenger car",
    "moving van", "ambulance", "fire engine", "garbage truck", "tow truck",
    "pickup", "taxi",
}


@pytest.mark.parametrize("model_name", [
    "squeezenet-1.1",
    "mobilenetv2-7",
    "resnet50-v2-7",
    "efficientnet-lite4-11",
    "shufflenet-v2-10",
])
def test_vision_classifier_identifies_bus(model_name):
    """Feed the canonical COCO bus image; top-5 should include a
    vehicle-like class from ImageNet."""
    path = _require(model_name)
    result = run_demo(zoo.get(model_name), path, input_spec="bus")
    assert result.ok, f"demo failed: {result.error}"
    assert isinstance(result, DemoResult)
    assert result.raw_shape == [1000], f"expected 1000 logits; got {result.raw_shape}"
    top5_labels = {p["label"] for p in result.predictions}
    overlap = top5_labels & _VEHICLE_LABELS
    assert overlap, (
        f"{model_name}: top-5 {top5_labels} contains no vehicle class — "
        f"preprocessing likely wrong"
    )


# ---------------------------------------------------------------------------
# Vision detector — YOLOv8 should find a person OR a bus
# ---------------------------------------------------------------------------

def test_yolov8_detects_something_in_bus_image():
    path = _require("yolov8n")
    result = run_demo(zoo.get("yolov8n"), path, input_spec="bus")
    assert result.ok, f"demo failed: {result.error}"
    assert len(result.predictions) >= 1, "YOLOv8 found 0 detections"
    labels = {p["label"] for p in result.predictions}
    # The canonical bus.jpg is known to contain people and a bus.
    assert labels & {"person", "bus", "car", "truck"}, \
        f"unexpected labels: {labels}"
    # Every detection must have a valid bbox
    for p in result.predictions:
        x1, y1, x2, y2 = p["bbox_xyxy"]
        assert 0 <= x1 < x2 and 0 <= y1 < y2
        assert 0.0 <= p["score"] <= 1.0


# ---------------------------------------------------------------------------
# BERT — canned Q+A must produce structurally valid span
# ---------------------------------------------------------------------------

def test_bert_squad_produces_valid_span():
    path = _require("bert-squad-10")
    result = run_demo(zoo.get("bert-squad-10"), path)
    assert result.ok, f"demo failed: {result.error}"
    pred = result.predictions[0]
    assert 0 <= pred["start_token_idx"] < 256
    assert 0 <= pred["end_token_idx"] < 256
    assert 1 <= pred["span_len"] <= 10


# ---------------------------------------------------------------------------
# GPT-2 — canned prompt must surface " Paris" (token 6342) in a sensible rank
# ---------------------------------------------------------------------------

def test_gpt2_predicts_paris_semantically():
    """'The capital of France is' should rank token 6342 (' Paris')
    high in the next-token distribution. This is a real semantic
    check — proves the model produces meaningful output, not gibberish."""
    path = _require("gpt-2-10")
    result = run_demo(zoo.get("gpt-2-10"), path)
    assert result.ok, f"demo failed: {result.error}"
    # Parse the summary for the Paris rank
    summary = result.summary
    assert "Paris" in summary, f"unexpected summary: {summary}"
    # If the Paris rank is reported, it should be in top 100
    # (out of 50257 vocab — so being in top-100 proves semantic working)
    import re
    m = re.search(r"rank (\d+)/", summary)
    if m:
        rank = int(m.group(1))
        assert rank <= 100, (
            f"GPT-2 ranked ' Paris' at {rank}/50257 — model not producing "
            f"semantically sensible output"
        )


# ---------------------------------------------------------------------------
# Registry plumbing — custom family handler
# ---------------------------------------------------------------------------

def test_custom_demo_family_registration():
    """OEMs can register demos for their own zoo entries' families."""
    @register_demo_family("test-oem-family")
    def _handler(zoo_entry, onnx_path, **kwargs):
        return DemoResult(
            model=zoo_entry.name, family="test-oem-family",
            backend="mock", input_source="none", wall_ms=0.0,
            summary="custom handler ran",
        )
    h = get_demo_handler("test-oem-family")
    assert h is _handler
    # Cleanup: remove the registration so state doesn't leak
    from astracore.demo.base import _HANDLERS
    _HANDLERS.pop("test-oem-family", None)


def test_unknown_demo_family_raises():
    with pytest.raises(DemoError):
        get_demo_handler("not-a-real-family")


# ---------------------------------------------------------------------------
# Dispatcher — zoo entry's family field routes correctly
# ---------------------------------------------------------------------------

def test_dispatcher_routes_by_family_field():
    """Changing a zoo entry's family should route to the matching
    handler. Using a fabricated ZooModel + the existing handlers."""
    from astracore.zoo import ZooModel
    from astracore.demo import run_demo

    entry = ZooModel(
        name="squeezenet-1.1",
        display_name="SqueezeNet 1.1",
        family="vision-classification",
        url=None, sha256=None, size_bytes=None,
        input_name="data", input_shape=(1, 3, 224, 224),
        opset=7, notes="",
    )
    path = _require("squeezenet-1.1")
    result = run_demo(entry, path)
    assert result.ok
    assert result.family == "vision-classification"
