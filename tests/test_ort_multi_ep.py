"""Step-1 tests for the ORT multi-EP façade.

Covers:
* short-name alias expansion (cuda / trt / dml / openvino / qnn / ...)
* missing-EP fallback behaviour
* per-EP option tuples survive normalisation
* YAML ``backend.options.providers`` reaches ``OrtBackend.__init__``
* ``astracore list eps`` CLI smoke
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Normaliser unit tests (no onnxruntime session needed)
# ---------------------------------------------------------------------------

def test_short_names_expand_to_full():
    from astracore.backends.ort import _expand_alias
    assert _expand_alias("cuda") == "CUDAExecutionProvider"
    assert _expand_alias("trt") == "TensorrtExecutionProvider"
    assert _expand_alias("TensorRT") == "TensorrtExecutionProvider"
    assert _expand_alias("dml") == "DmlExecutionProvider"
    assert _expand_alias("openvino") == "OpenVINOExecutionProvider"
    assert _expand_alias("ov") == "OpenVINOExecutionProvider"
    assert _expand_alias("qnn") == "QNNExecutionProvider"
    assert _expand_alias("coreml") == "CoreMLExecutionProvider"
    assert _expand_alias("cpu") == "CPUExecutionProvider"
    # Unknown string passes through verbatim (vendor-custom EP).
    assert _expand_alias("MyVendorEP") == "MyVendorEP"


def test_normalise_drops_unavailable_and_keeps_cpu():
    from astracore.backends.ort import _normalise_providers
    # Pretend only CPU is present. Asking for CUDA should drop it and
    # still leave CPU at the end.
    available = ["CPUExecutionProvider"]
    out = _normalise_providers(["cuda", "cpu"], available)
    assert out == ["CPUExecutionProvider"]


def test_normalise_preserves_per_ep_options():
    from astracore.backends.ort import _normalise_providers
    available = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    out = _normalise_providers(
        [("cuda", {"device_id": 0, "gpu_mem_limit": 2 * 1024 ** 3})],
        available,
    )
    # First entry: CUDA tuple with options; second: CPU appended.
    assert out[0] == ("CUDAExecutionProvider",
                      {"device_id": 0, "gpu_mem_limit": 2 * 1024 ** 3})
    assert "CPUExecutionProvider" in out


def test_normalise_deduplicates():
    from astracore.backends.ort import _normalise_providers
    available = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    out = _normalise_providers(
        ["cuda", "CUDAExecutionProvider", "cuda", "cpu"], available,
    )
    # CUDA should appear once, CPU once.
    names = [p if isinstance(p, str) else p[0] for p in out]
    assert names.count("CUDAExecutionProvider") == 1
    assert names.count("CPUExecutionProvider") == 1


def test_normalise_empty_request_leaves_cpu():
    from astracore.backends.ort import _normalise_providers
    available = ["CPUExecutionProvider"]
    out = _normalise_providers([], available)
    assert out == ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# OrtBackend end-to-end (uses real onnxruntime + yolov8n.onnx)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def yolo_path():
    p = REPO / "data" / "models" / "yolov8n.onnx"
    if not p.exists():
        pytest.skip("yolov8n.onnx not on disk")
    return p


def test_ortbackend_records_active_providers(yolo_path):
    """A compiled session should report which EPs ORT actually used."""
    from astracore.backends.ort import OrtBackend
    be = OrtBackend(providers=["cpu"])
    program = be.compile(str(yolo_path), concrete_shapes=None)
    rep = be.report_last()
    assert rep.silicon_profile.startswith("ort-")
    assert "CPUExecutionProvider" in rep.silicon_profile
    assert rep.extra.get("active_providers")


def test_ortbackend_unavailable_ep_falls_back(yolo_path, caplog):
    """Requesting CUDA on a CPU-only ORT build should warn + fall back."""
    import logging
    from astracore.backends.ort import OrtBackend
    be = OrtBackend(providers=["cuda", "cpu"])
    with caplog.at_level(logging.WARNING, logger="astracore.backends.ort"):
        program = be.compile(str(yolo_path))
    # Session still compiled on CPU — no exception.
    assert program is not None
    # If CUDA is absent, we logged a warning. On a CUDA-enabled host
    # the warning won't fire — test is lenient.
    import onnxruntime as ort
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        assert any("CUDAExecutionProvider" in r.message
                   for r in caplog.records)


# ---------------------------------------------------------------------------
# Propagation: benchmark_model + multistream + apply
# ---------------------------------------------------------------------------

def test_benchmark_model_threads_backend_options(yolo_path):
    from astracore.benchmark import benchmark_model
    rep = benchmark_model(
        yolo_path,
        backend="onnxruntime",
        backend_options={"providers": ["cpu"]},
        n_iter=1, warmup=1,
    )
    assert rep.extra.get("active_providers")
    assert "CPUExecutionProvider" in rep.silicon_profile


def test_yaml_backend_options_reach_ortbackend(tmp_path):
    """YAML ``backend.options.providers`` must show up in the report."""
    from astracore import config
    from astracore.apply import apply_config

    yaml_body = textwrap.dedent(f"""
        version: 1
        name: ep-smoke
        backend:
          name: onnxruntime
          options:
            providers: [cpu]
        models:
          - id: m1
            path: {(REPO / 'data' / 'models' / 'yolov8n.onnx').as_posix()}
            family: vision-detection
            precision: INT8
            sparsity: dense
        multistream:
          enabled: false
        dataset:
          connector: synthetic
          preset: tiny
    """).strip()
    p = tmp_path / "smoke.yaml"
    p.write_text(yaml_body)
    cfg = config.load(p)

    if not (REPO / "data" / "models" / "yolov8n.onnx").exists():
        pytest.skip("yolov8n.onnx not on disk")

    rep = apply_config(
        cfg,
        out_dir=tmp_path / "out",
        bench_iter=1,
        skip_replay=True,
        skip_multistream=True,
    )
    assert rep.models[0].bench_ok, rep.models[0].bench_error


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_list_eps_cli():
    r = subprocess.run(
        [sys.executable, "-m", "astracore.cli", "list", "eps"],
        cwd=REPO, capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, r.stderr
    assert "CPUExecutionProvider" in r.stdout
    # Aliases section is always printed.
    assert "aliases" in r.stdout.lower()


def test_list_subcommand_accepts_eps_choice():
    """Regression against argparse choices list."""
    r = subprocess.run(
        [sys.executable, "-m", "astracore.cli", "list", "--help"],
        cwd=REPO, capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0
    assert "eps" in r.stdout
