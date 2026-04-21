"""Tests for astracore.quantise + astracore quantise CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
YOLO = REPO / "data" / "models" / "yolov8n.onnx"


def _require_yolo():
    if not YOLO.exists():
        pytest.skip(f"yolov8n.onnx missing: {YOLO}")


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------

def test_quantise_returns_manifest(tmp_path):
    _require_yolo()
    from astracore.quantise import quantise
    out = tmp_path / "yolov8n.int8.onnx"
    m = quantise(
        model_path=YOLO, output_path=out,
        cal_samples=10,      # tight for CI
    )
    assert out.exists()
    assert m.source_onnx.endswith("yolov8n.onnx")
    assert m.precision == "int8"
    assert m.granularity == "per_channel"
    # 10-sample calibration should still clear ~25 dB on yolov8n.
    assert m.snr_db > 20.0, f"SNR {m.snr_db} too low — calibration broken?"
    assert 0.99 < m.cosine <= 1.0001
    assert m.output_bytes > 0


def test_quantise_writes_manifest_json(tmp_path):
    _require_yolo()
    from astracore.quantise import quantise, write_manifest
    out = tmp_path / "yolov8n.int8.onnx"
    manifest_path = tmp_path / "yolov8n.int8.onnx.json"
    m = quantise(model_path=YOLO, output_path=out, cal_samples=8)
    write_manifest(m, manifest_path)
    assert manifest_path.exists()
    doc = json.loads(manifest_path.read_text())
    for field in ("snr_db", "cosine", "max_abs_err", "calibration_samples",
                  "input_name", "output_tensor", "source_sha256"):
        assert field in doc


def test_quantise_manifest_round_trip(tmp_path):
    """A manifest should be stable enough that re-running produces the
    same fake-quant ONNX bytes (same seed, same samples) → same SHA."""
    _require_yolo()
    from astracore.quantise import quantise
    out1 = tmp_path / "a.int8.onnx"
    out2 = tmp_path / "b.int8.onnx"
    m1 = quantise(model_path=YOLO, output_path=out1, cal_samples=5,
                  cal_seed=42)
    m2 = quantise(model_path=YOLO, output_path=out2, cal_samples=5,
                  cal_seed=42)
    # Same inputs -> identical outputs, byte-for-byte.
    assert m1.output_sha256 == m2.output_sha256, (
        "same calibration seed produced different fake-quant bytes — "
        "quantiser has hidden nondeterminism"
    )
    # SNR should also match since the probe seed is fixed.
    assert m1.snr_db == m2.snr_db


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_quantise_end_to_end(tmp_path):
    _require_yolo()
    out = tmp_path / "y.int8.onnx"
    r = subprocess.run(
        [sys.executable, "-m", "astracore.cli", "quantise",
         "--model", str(YOLO),
         "--out", str(out),
         "--cal-samples", "6",
         "--cal-seed", "0"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert out.exists()
    manifest = Path(str(out) + ".json")
    assert manifest.exists()
    assert "SNR:" in r.stdout
    assert "cosine:" in r.stdout


def test_cli_quantise_rejects_missing_model():
    r = subprocess.run(
        [sys.executable, "-m", "astracore.cli", "quantise",
         "--model", "/nonexistent/model.onnx"],
        cwd=REPO, capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 2
    assert "not found" in r.stderr.lower()


def test_cli_quantise_help_lists_flags():
    r = subprocess.run(
        [sys.executable, "-m", "astracore.cli", "quantise", "--help"],
        cwd=REPO, capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0
    for flag in ("--model", "--out", "--manifest", "--cal-samples",
                 "--precision", "--granularity", "--method"):
        assert flag in r.stdout, f"flag {flag} missing from --help"
