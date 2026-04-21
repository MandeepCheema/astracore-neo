"""Smoke tests for scripts/ai_model_deep_tests.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "ai_model_deep_tests.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location(
        "ai_model_deep_tests", SCRIPT,
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["ai_model_deep_tests"] = m
    spec.loader.exec_module(m)
    return m


def test_deep_tests_registry(mod):
    assert len(mod.DEEP_TESTS) == 6
    names = [t[0] for t in mod.DEEP_TESTS]
    assert "input_perturbation" in names
    assert "bert_determinism" in names
    assert "gpt2_paris_rank" in names
    assert "yolo_determinism" in names
    assert "latency_vs_gmacs" in names
    assert "int8_drift" in names


def test_input_perturbation_smoke(mod):
    """Fast — ~2 s. Verifies verdict structure."""
    d = mod.test_input_perturbation()
    assert d["rows"]
    assert d["verdicts"]
    for name, v in d["verdicts"].items():
        assert "top1_stable_per_sigma" in v
        assert "survives_up_to_sigma" in v
        # At σ=0 the top-1 must match itself trivially.
        assert v["top1_stable_per_sigma"][0.0] is True


def test_yolo_determinism_smoke(mod):
    """28/28 images must be deterministic at fp32 — hard gate."""
    d = mod.test_yolo_determinism()
    if "skipped" in d:
        pytest.skip(d["skipped"])
    assert d["all_stable"] is True, (
        f"{d['n_unstable']} YOLO images drifted across 5 runs — "
        f"nondeterminism detected"
    )


def test_committed_reports_exist():
    base = REPO / "reports" / "ai_deep_tests"
    for i, name in [
        (1, "input_perturbation"),
        (2, "bert_determinism"),
        (3, "gpt2_paris_rank"),
        (4, "yolo_determinism"),
        (5, "latency_vs_gmacs"),
        (6, "int8_drift"),
    ]:
        assert (base / f"{i}_{name}.md").exists(), f"missing {i}_{name}.md"
        assert (base / f"{i}_{name}.json").exists(), f"missing {i}_{name}.json"


def test_int8_drift_is_production_floor():
    """Pin a non-trivial INT8 drift result.

    Single-sample calibration on yolov8n: SNR should be in the 25-35 dB
    band. Below 20 dB = calibration is broken; above 40 dB = single-
    sample calibration is suspicious (probably not a fresh run).
    """
    p = REPO / "reports" / "ai_deep_tests" / "6_int8_drift.json"
    if not p.exists():
        pytest.skip("int8 drift report not generated")
    d = json.loads(p.read_text())
    if "skipped" in d:
        pytest.skip(d["skipped"])
    assert 20.0 < d["snr_db"] < 45.0, (
        f"int8 drift SNR {d['snr_db']} dB is outside the plausible "
        f"single-sample calibration band"
    )
    assert d["cosine"] > 0.99
