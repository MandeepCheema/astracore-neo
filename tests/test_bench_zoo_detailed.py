"""Smoke tests for scripts/bench_zoo_detailed.py.

Keeps the research harness honest: import cleanly, run against one
small model end-to-end, emit valid JSON, and produce a fingerprint.
Full zoo sweep is too slow for CI (~150 s); one model is ~5 s.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "bench_zoo_detailed.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location(
        "bench_zoo_detailed", SCRIPT,
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["bench_zoo_detailed"] = m
    spec.loader.exec_module(m)
    return m


def test_latency_stats_handles_empty(mod):
    s = mod.LatencyStats.from_list([])
    assert s.n_samples == 0
    assert s.mean_ms == 0


def test_latency_stats_computes_percentiles(mod):
    s = mod.LatencyStats.from_list([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s.n_samples == 5
    assert s.mean_ms == pytest.approx(3.0)
    assert s.p50_ms == pytest.approx(3.0)
    # p99 of a 5-element uniform list is close to max (4.96).
    assert s.p99_ms == pytest.approx(4.96)
    assert s.max_ms == pytest.approx(5.0)


def test_supports_batch_sweep_detection(mod):
    """Models with dim_value=1 or dynamic first dim support batch sweep."""
    import onnx
    from astracore import zoo as zoo_mod
    paths = zoo_mod.local_paths()
    shufflenet = paths.get("shufflenet-v2-10")
    if not shufflenet or not shufflenet.exists():
        pytest.skip("shufflenet ONNX not on disk")
    m = onnx.load(str(shufflenet))
    assert mod._model_supports_batch_sweep(m) is True


def test_fingerprint_is_deterministic(mod):
    import numpy as np
    outputs = {
        "out_a": np.array([1.001, 2.002, 3.003], dtype=np.float32),
        "out_b": np.array([[4.1, 5.2], [6.3, 7.4]], dtype=np.float32),
    }
    fp1 = mod._fingerprint_outputs(outputs)
    fp2 = mod._fingerprint_outputs({k: v.copy() for k, v in outputs.items()})
    assert fp1 == fp2
    assert len(fp1) == 16


def test_fingerprint_diverges_on_real_drift(mod):
    import numpy as np
    a = {"out": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    b = {"out": np.array([1.0, 2.0, 99.0], dtype=np.float32)}  # real change
    assert mod._fingerprint_outputs(a) != mod._fingerprint_outputs(b)


def test_fingerprint_tolerates_sub_mdp_jitter(mod):
    """A sub-0.001 perturbation should not flip the fingerprint
    (we round to 3 dp before hashing)."""
    import numpy as np
    a = {"out": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    b = {"out": np.array([1.00001, 2.00001, 3.00001], dtype=np.float32)}
    assert mod._fingerprint_outputs(a) == mod._fingerprint_outputs(b)


def test_end_to_end_single_model(mod, tmp_path):
    """Run one fast model through run_suite and verify the shape of the
    result. 5-6 s total. Uses shufflenet because it's the smallest."""
    from astracore import zoo as zoo_mod
    paths = zoo_mod.local_paths()
    target = "shufflenet-v2-10"
    if not paths.get(target) or not paths[target].exists():
        pytest.skip(f"{target} ONNX not on disk")

    report = mod.run_suite(
        providers=[],
        batch_sizes=[1],
        thread_counts=[],
        graph_opt_levels=["all"],
        warmup=1, n_timed=3,
        only=[target],
    )
    assert report.n_models == 1
    m = report.models[0]
    assert m.model == target
    # "base" scenario + "gopt=all"
    assert len(m.scenarios) >= 2
    base = next(s for s in m.scenarios if s.scenario == "base")
    assert not base.failed, base.error
    assert base.steady.n_samples == 3
    assert base.output_fingerprint  # non-empty hex
    assert len(base.output_fingerprint) == 16
    assert base.providers_active


def test_markdown_renders_valid(mod):
    """Hand-crafted SuiteReport → render_markdown must not raise."""
    host = {"platform": "test", "cpu": "fake", "cpu_count": 4,
            "python": "3.12", "ort_available_providers": ["CPUExecutionProvider"]}
    ls = mod.LatencyStats.from_list([1.0, 1.1, 1.2])
    base = mod.ScenarioResult(
        scenario="base", backend="onnxruntime",
        providers_requested=[], providers_active=["CPUExecutionProvider"],
        batch_size=1, intra_op_threads=None, graph_opt_level="default",
        warmup_ms=2.0, steady=ls,
        output_fingerprint="deadbeefdeadbeef",
    )
    mr = mod.ModelReport(
        model="tinymodel", family="vision-classification",
        onnx_path="/dev/null", input_name="x", input_shape=[1, 3, 224, 224],
        opset=7, gmacs=0.1, scenarios=[base],
    )
    sr = mod.SuiteReport(
        host=host, backend="onnxruntime", providers_requested=[],
        wall_s_total=0.5, n_models=1, models=[mr],
    )
    md = mod.render_markdown(sr)
    assert "tinymodel" in md
    assert "deadbeefdeadbeef" in md
    assert "Tail-latency ratios" in md
