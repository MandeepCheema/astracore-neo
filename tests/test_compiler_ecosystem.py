"""Compiler-ecosystem tests — MLIR/IREE/TVM conformance + BYOC scaffold.

Covers two Track-3 compiler-ecosystem deliverables from
``docs/external_validation_options.md``:

  1. MLIR / IREE conformance — ``tools.frontends.iree_conformance``
     runs a canonical op suite through every normalised ONNX adapter
     (mlir-stablehlo, tvm-relay, jax-xla) and asserts each case lands
     with the expected op in our NnGraph IR.

  2. TVM BYOC — ``tools.frontends.tvm_byoc`` exposes the pattern table
     (``ASTRACORE_PATTERNS``), a ``describe_patterns`` JSON dump, and
     TVM-gated entry points that raise a clear ``ImportError`` when
     ``tvm`` is absent. The full Relay IRModule partitioning check is
     ``@pytest.mark.integration``-gated because TVM isn't a default
     SDK dep.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# MLIR / IREE conformance harness
# ---------------------------------------------------------------------------


def test_conformance_suite_runs_all_canonical_cases():
    """Runner exercises every case × every adapter and passes 100%."""
    from tools.frontends.iree_conformance import (
        run_conformance_suite, ADAPTERS, _suite,
    )
    suite = _suite()
    expected_total = len(suite) * len(ADAPTERS)

    report = run_conformance_suite()
    assert len(report.cases) == expected_total
    assert report.pass_count == expected_total, (
        f"conformance regressions: {[c for c in report.cases if not c.passed]}"
    )
    assert report.fail_count == 0


def test_conformance_suite_covers_expected_op_groups():
    """Sanity: the suite exercises arith + linalg + act + norm + shape + pool."""
    from tools.frontends.iree_conformance import _suite
    groups = {c.op_group for c in _suite()}
    assert {"arith", "linalg", "act", "norm", "shape", "pool"} <= groups


def test_conformance_adapter_restriction():
    """--adapter filter only runs the requested adapters."""
    from tools.frontends.iree_conformance import run_conformance_suite, _suite
    report = run_conformance_suite(adapters=["mlir-stablehlo"])
    assert report.adapters == ["mlir-stablehlo"]
    assert len(report.cases) == len(_suite())  # one adapter × N cases


def test_conformance_unknown_adapter_raises():
    from tools.frontends.iree_conformance import run_conformance_suite
    with pytest.raises(ValueError, match="unknown adapters"):
        run_conformance_suite(adapters=["does-not-exist"])


def test_conformance_report_serialises_cleanly(tmp_path):
    """JSON emitted by the runner is round-trip parseable and has the
    summary totals the CLI needs for its exit code."""
    from tools.frontends.iree_conformance import (
        run_conformance_suite, report_as_markdown,
    )
    report = run_conformance_suite()
    blob = report.to_dict()
    parsed = json.loads(json.dumps(blob))
    assert parsed["summary"]["passed"] == report.pass_count
    assert parsed["summary"]["total"] == len(report.cases)
    assert parsed["adapters"] == report.adapters

    md = report_as_markdown(report)
    assert "MLIR / IREE / TVM conformance" in md
    # Header shape — one row per case, one column per adapter + 2 meta.
    for case in ("add", "matmul", "conv", "softmax", "layernorm"):
        assert f"| {case} |" in md


def test_conformance_captures_failure_when_case_expects_wrong_op():
    """Inject a bad expected-op and prove the runner reports it rather
    than silently passing — this is the guard that keeps the suite
    honest if someone edits a case builder without updating its
    expected_ops."""
    from tools.frontends.iree_conformance import (
        ConformanceCase, _case_add, run_conformance_suite,
    )

    bad_case = ConformanceCase(
        name="add-mislabelled", description="deliberately wrong",
        op_group="arith", build=_case_add,
        expected_ops=("conv",),  # <-- wrong
    )
    report = run_conformance_suite(cases=[bad_case], adapters=["mlir-stablehlo"])
    assert report.fail_count == 1
    assert report.cases[0].passed is False
    assert "expected ops missing" in report.cases[0].error


# ---------------------------------------------------------------------------
# TVM BYOC scaffold
# ---------------------------------------------------------------------------


def test_tvm_byoc_pattern_table_has_core_ops():
    """Every op our canonical MLIR/IREE suite exercises must also
    appear in the TVM BYOC pattern table — if they drift, the
    compiler-ecosystem story has a hole."""
    from tools.frontends.tvm_byoc import ASTRACORE_PATTERNS
    composites = {p.composite for p in ASTRACORE_PATTERNS}
    for required in (
        "astracore.conv2d", "astracore.dense", "astracore.matmul",
        "astracore.add", "astracore.mul", "astracore.relu",
        "astracore.sigmoid", "astracore.softmax", "astracore.layernorm",
        "astracore.maxpool", "astracore.avgpool", "astracore.reshape",
        "astracore.transpose", "astracore.concat", "astracore.gelu",
    ):
        assert required in composites, f"missing BYOC pattern: {required}"


def test_tvm_byoc_describe_patterns_is_json_serialisable():
    from tools.frontends.tvm_byoc import describe_patterns
    blob = describe_patterns()
    # Round-trip through json — catches any stray frozenset / tuple
    # inside attrs that json.dumps can't handle.
    # tuples are allowed (json serialises them as arrays).
    roundtrip = json.loads(json.dumps(blob, default=list))
    assert len(roundtrip) == len(blob)
    for entry in roundtrip:
        assert {"relay_op", "composite", "attrs"} <= entry.keys()


def test_tvm_byoc_target_name_is_stable():
    """TVM identifies our backend by this exact string; changing it
    silently would break any user code that already calls
    ``relay.build(..., target='astracore')``."""
    from tools.frontends.tvm_byoc import ASTRACORE_TARGET_NAME
    assert ASTRACORE_TARGET_NAME == "astracore"


def test_tvm_byoc_raises_clear_error_without_tvm():
    """If TVM isn't installed, the codegen / partition entry points must
    raise a clear ImportError naming the package + install command."""
    from tools.frontends.tvm_byoc import (
        is_tvm_available, register_astracore_target,
        partition_graph_for_astracore, astracore_codegen,
    )
    if is_tvm_available():
        pytest.skip("tvm installed; skip missing-package path")

    for fn in (register_astracore_target, partition_graph_for_astracore,
               astracore_codegen):
        with pytest.raises(ImportError, match="TVM"):
            # All three gate on _require_tvm; call with dummy args that
            # will never reach real logic because the import check fails
            # first.
            if fn is partition_graph_for_astracore:
                fn(object())
            elif fn is astracore_codegen:
                fn(object())
            else:
                fn()


@pytest.mark.integration
def test_tvm_byoc_partition_smoke_when_tvm_available():
    """End-to-end smoke: if TVM is on the host, a minimal Relay graph
    with a single Conv2D should partition with the AstraCore target
    without raising. Skipped in the default suite."""
    from tools.frontends.tvm_byoc import (
        is_tvm_available, partition_graph_for_astracore,
    )
    if not is_tvm_available():
        pytest.skip("tvm not installed")

    import tvm
    from tvm import relay
    x = relay.var("x", shape=(1, 3, 8, 8), dtype="float32")
    w = relay.var("w", shape=(4, 3, 3, 3), dtype="float32")
    y = relay.nn.conv2d(x, w, channels=4, kernel_size=(3, 3),
                        padding=(1, 1))
    mod = tvm.IRModule.from_expr(relay.Function([x, w], y))
    partitioned = partition_graph_for_astracore(mod)
    # The partition pass replaces the Conv with a compiler-tagged
    # external function; the textual IR should mention our target.
    assert "astracore" in str(partitioned)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_cli_conformance_writes_reports(tmp_path):
    from astracore.cli import main
    rc = main(["conformance", "--out", str(tmp_path)])
    assert rc == 0

    js = tmp_path / "conformance.json"
    md = tmp_path / "conformance.md"
    assert js.exists() and md.exists()
    blob = json.loads(js.read_text())
    assert blob["summary"]["passed"] == blob["summary"]["total"]
    # CLI variant attaches the BYOC pattern table for a unified artefact.
    assert "tvm_byoc_patterns" in blob
    assert any(p["composite"] == "astracore.conv2d"
               for p in blob["tvm_byoc_patterns"])
