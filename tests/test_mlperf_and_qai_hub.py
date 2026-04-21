"""Smoke tests for astracore.mlperf + the Qualcomm AI Hub scaffold."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
YOLO = REPO / "data" / "models" / "yolov8n.onnx"


# ---------------------------------------------------------------------------
# astracore.mlperf
# ---------------------------------------------------------------------------

def test_mlperf_module_imports():
    from astracore.mlperf import Scenario, run_scenario, run_all_scenarios
    assert Scenario.SINGLE_STREAM.value == "single_stream"
    assert callable(run_scenario)
    assert callable(run_all_scenarios)


def test_mlperf_scenario_enum_has_four_values():
    from astracore.mlperf import Scenario
    vals = {s.value for s in Scenario}
    assert vals == {"single_stream", "multi_stream", "offline", "server"}


def test_mlperf_stats_helper_math():
    from astracore.mlperf import _stats_from_ms
    s = _stats_from_ms([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["mean_ms"] == pytest.approx(3.0)
    assert s["p50_ms"] == pytest.approx(3.0)
    assert s["max_ms"] == pytest.approx(5.0)


@pytest.mark.skipif(not YOLO.exists(), reason="yolov8n.onnx missing")
def test_mlperf_single_stream_smoke():
    """Run a 1s SingleStream against yolov8n — must produce a report."""
    from astracore.backends.ort import OrtBackend
    from astracore.mlperf import Scenario, run_scenario
    be = OrtBackend(providers=["cpu"])
    r = run_scenario(
        backend=be, model_path=str(YOLO),
        scenario=Scenario.SINGLE_STREAM, duration_s=1.0,
    )
    assert r.n_queries > 0, "expected ≥1 query in 1s"
    assert r.throughput_qps > 0
    assert r.latency_stats_ms["p99_ms"] > 0
    assert r.scenario == "single_stream"


@pytest.mark.skipif(not YOLO.exists(), reason="yolov8n.onnx missing")
def test_mlperf_offline_scenario_runs_n_samples_exactly():
    from astracore.backends.ort import OrtBackend
    from astracore.mlperf import Scenario, run_scenario
    be = OrtBackend(providers=["cpu"])
    r = run_scenario(
        backend=be, model_path=str(YOLO),
        scenario=Scenario.OFFLINE, n_samples=5,
    )
    assert r.n_queries == 5
    assert r.extra["target_qps"] is None


def test_mlperf_cli_help_lists_scenarios():
    r = subprocess.run(
        [sys.executable, "-m", "astracore.cli", "mlperf", "--help"],
        cwd=REPO, capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0
    for sc in ("single_stream", "multi_stream", "offline", "server", "all"):
        assert sc in r.stdout


@pytest.mark.skipif(not YOLO.exists(), reason="yolov8n.onnx missing")
def test_mlperf_cli_single_stream_runs(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "astracore.cli", "mlperf",
         "--model", str(YOLO),
         "--scenario", "single_stream",
         "--duration", "1",
         "--out", str(tmp_path / "mlperf")],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stderr
    # Produced the JSON + MD artefacts.
    assert (tmp_path / "mlperf" / "single_stream.json").exists()
    assert (tmp_path / "mlperf" / "single_stream.md").exists()


# ---------------------------------------------------------------------------
# Qualcomm AI Hub scaffold
# ---------------------------------------------------------------------------

QAI_SCRIPT = REPO / "scripts" / "submit_to_qualcomm_ai_hub.py"


def test_qai_hub_script_exists():
    assert QAI_SCRIPT.exists()


def test_qai_hub_script_has_dry_run_flag():
    body = QAI_SCRIPT.read_text(encoding="utf-8")
    assert "--dry-run" in body
    assert "Snapdragon" in body


def test_qai_hub_dry_run_smoke():
    if not YOLO.exists():
        pytest.skip("yolov8n.onnx missing")
    r = subprocess.run(
        [sys.executable, str(QAI_SCRIPT),
         "--model", str(YOLO),
         "--device", "Snapdragon 8 Gen 3",
         "--dry-run"],
        cwd=REPO, capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout


def test_qai_hub_rejects_missing_model():
    r = subprocess.run(
        [sys.executable, str(QAI_SCRIPT),
         "--model", "/nonexistent/model.onnx",
         "--dry-run"],
        cwd=REPO, capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 2


# ---------------------------------------------------------------------------
# ONNX prep helper
# ---------------------------------------------------------------------------

PREP_SCRIPT = REPO / "scripts" / "prep_onnx_for_ai_hub.py"
BATCH_SCRIPT = REPO / "scripts" / "submit_zoo_to_ai_hub.py"


def test_prep_script_exists():
    assert PREP_SCRIPT.exists()


def test_prep_script_documents_all_fixes():
    body = PREP_SCRIPT.read_text(encoding="utf-8")
    # Each known fixup must be mentioned in the docstring so future
    # contributors know why each step is there.
    for snippet in ("opset upgrade", "value_info", "initializer",
                    "IR-version", "shape_inference", "Pow"):
        assert snippet.lower() in body.lower(), (
            f"prep script docstring missing '{snippet}'")


def test_pow_rewrite_helper_exists_and_rewrites_pow_3():
    """The Pow(x,3) -> x*x*x rewrite must be reachable as a module."""
    import importlib.util as _iu
    p = REPO / "scripts" / "fix_bert_pow.py"
    assert p.exists()
    spec = _iu.spec_from_file_location("fix_bert_pow", p)
    mod = _iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "rewrite_pow_as_mul")

    # Build a tiny ONNX model with Pow(x, 3) and verify rewrite.
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    import numpy as np
    exp_init = numpy_helper.from_array(np.array(3.0, dtype=np.float32), name="exp")
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
    node = helper.make_node("Pow", ["x", "exp"], ["y"], name="pow_3")
    graph = helper.make_graph([node], "t", [x_info], [y_info], [exp_init])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 7
    n = mod.rewrite_pow_as_mul(m)
    assert n == 1
    # After rewrite: 2 Mul ops, 0 Pow ops.
    ops = [n.op_type for n in m.graph.node]
    assert ops.count("Mul") == 2
    assert ops.count("Pow") == 0


def test_prep_helper_roundtrips_a_small_model(tmp_path):
    """Round-trip any zoo model that's on disk through the prep helper.
    Lightweight — squeezenet is ~5 MB and preps in < 2s."""
    from scripts.prep_onnx_for_ai_hub import prep_onnx_for_ai_hub
    src = REPO / "data" / "models" / "zoo" / "squeezenet-1.1.onnx"
    if not src.exists():
        pytest.skip("squeezenet-1.1.onnx missing")
    dst = tmp_path / "sq.aihub.onnx"
    log = prep_onnx_for_ai_hub(src, dst)
    assert dst.exists()
    assert log["steps"], "prep helper did nothing?"
    # Must end up opset ≥ 13 and IR ≥ 7.
    import onnx
    m = onnx.load(str(dst))
    assert m.ir_version >= 7
    for oi in m.opset_import:
        if oi.domain in ("", "ai.onnx"):
            assert oi.version >= 13
    # Initializers must NOT be in graph.input (modern convention).
    init_names = {i.name for i in m.graph.initializer}
    input_names = {i.name for i in m.graph.input}
    assert not (init_names & input_names), (
        "initializer-inputs remain after prep — AI Hub will reject")


def test_batch_script_exists_and_lists_cnn_zoo():
    assert BATCH_SCRIPT.exists()
    body = BATCH_SCRIPT.read_text(encoding="utf-8")
    # Each CNN the leaderboard §6 relies on must be in the batch.
    for model in ("yolov8n", "squeezenet-1.1", "mobilenetv2-7",
                  "resnet50-v2-7", "shufflenet-v2-10",
                  "efficientnet-lite4-11"):
        assert model in body, f"batch script missing zoo entry: {model}"
