"""Multi-stream harness — framework tests.

Fast assertions on the measurement contract. Actual scaling numbers
depend on host CPU/GPU and are measured in
``reports/multistream.json``; tests here only verify the report
*shape* and invariants that must hold on any hardware.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astracore import zoo
from astracore.multistream import (
    MultiStreamReport,
    StreamSlice,
    run_multistream,
)


def _small_model() -> Path:
    """Pick the smallest downloaded zoo model for speed."""
    paths = zoo.local_paths()
    for name in ("squeezenet-1.1", "shufflenet-v2-10", "yolov8n"):
        p = paths.get(name)
        if p and p.exists():
            return p
    pytest.skip("no zoo model downloaded; run scripts/fetch_model_zoo.py")


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

def test_report_shape_is_stable():
    """Run 2 tiny slices; verify the report is well-formed."""
    model = _small_model()
    # Use the model's own declared shape — SqueezeNet expects 224×224.
    m = next(m for m in zoo.all_models() if zoo.local_paths()[m.name] == model)
    rep = run_multistream(
        model,
        backend="onnxruntime",
        n_streams_list=(1, 2),
        duration_s=0.5,
        warmup_s=0.2,
        input_shape=m.input_shape,
    )
    assert isinstance(rep, MultiStreamReport)
    assert rep.backend == "onnxruntime"
    assert len(rep.slices) == 2
    assert all(isinstance(s, StreamSlice) for s in rep.slices)
    assert rep.slices[0].n_streams == 1
    assert rep.slices[1].n_streams == 2


def test_report_single_stream_is_baseline():
    """At n_streams=1, scale-factor and util must be exactly 1.0."""
    model = _small_model()
    m = next(m for m in zoo.all_models() if zoo.local_paths()[m.name] == model)
    rep = run_multistream(
        model,
        n_streams_list=(1,),
        duration_s=0.3,
        warmup_s=0.1,
        input_shape=m.input_shape,
    )
    s = rep.slices[0]
    assert s.scaling_vs_single == 1.0
    assert s.util_vs_single == 1.0


def test_invariants_hold_for_every_slice():
    model = _small_model()
    m = next(m for m in zoo.all_models() if zoo.local_paths()[m.name] == model)
    rep = run_multistream(
        model,
        n_streams_list=(1, 2),
        duration_s=0.4,
        warmup_s=0.1,
        input_shape=m.input_shape,
    )
    for s in rep.slices:
        assert s.n_inferences_total > 0
        assert s.wall_s > 0
        assert s.throughput_ips > 0
        assert s.p50_latency_ms > 0
        assert s.p99_latency_ms >= s.p50_latency_ms


def test_markdown_table_contains_all_slices():
    model = _small_model()
    m = next(m for m in zoo.all_models() if zoo.local_paths()[m.name] == model)
    rep = run_multistream(
        model,
        n_streams_list=(1, 2, 4),
        duration_s=0.3,
        warmup_s=0.1,
        input_shape=m.input_shape,
    )
    md = rep.as_markdown()
    for n in (1, 2, 4):
        assert f"| {n} |" in md


def test_report_serialisable():
    """JSON round-trip works."""
    import json
    model = _small_model()
    m = next(m for m in zoo.all_models() if zoo.local_paths()[m.name] == model)
    rep = run_multistream(
        model,
        n_streams_list=(1,),
        duration_s=0.2,
        warmup_s=0.05,
        input_shape=m.input_shape,
    )
    encoded = json.dumps(rep.as_dict())
    back = json.loads(encoded)
    assert back["backend"] == "onnxruntime"
    assert len(back["slices"]) == 1


def test_mac_ops_populated_for_conv_models():
    """The ONNX MAC estimator should populate non-zero ops for any of the
    downloaded zoo models (shape-inference fallback must fire)."""
    model = _small_model()
    m = next(m for m in zoo.all_models() if zoo.local_paths()[m.name] == model)
    rep = run_multistream(
        model,
        n_streams_list=(1,),
        duration_s=0.1,
        warmup_s=0.05,
        input_shape=m.input_shape,
    )
    assert rep.mac_ops_per_inference > 0, \
        "MAC estimator returned 0 — ORT shape-inference path regressed"
