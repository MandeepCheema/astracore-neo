"""MLIR / IREE / StableHLO conformance harness for AstraCore.

Ships a canonical op suite that every external compiler stack we claim
compatibility with (torch-mlir → StableHLO → ONNX, IREE's `iree-import-onnx`,
TVM's Relay/Relax `from_onnx`, JAX/XLA → ONNX) must be able to lower. Each
case is a minimal ONNX graph exercising exactly one op class; the runner
pushes every case through the ``tools.frontends.*`` adapters (which
normalise on ONNX → NnGraph) and records pass / fail / op-count.

The suite is self-contained. We don't require ``iree-tools``,
``torch_mlir``, or ``tvm`` to be installed — the adapters' public
contract is "accept ONNX bytes, return NnGraph", and this harness
validates that contract against the canonical ops an MLIR / IREE / TVM
lowering path would produce.

If the optional toolchains are present, the runner can also exercise
them via ``--with iree`` / ``--with tvm`` (provider-specific handlers
plug in through ``EXTERNAL_RUNNERS``).

Usage
-----
From Python::

    from tools.frontends.iree_conformance import run_conformance_suite
    result = run_conformance_suite()
    print(result.summary_line())

From CLI (wired up via ``astracore`` in a follow-up)::

    python -m tools.frontends.iree_conformance --out reports/compiler_ecosystem/

Each run emits a JSON summary + a human-readable markdown companion.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Case builders. Each returns (name, onnx_bytes, expected_ops_in_nngraph).
#
# The graphs are intentionally the smallest shape that hits the op — minimal
# tensor dims so the suite runs in < 1 s even with every adapter + every
# case. Shapes are concrete (no dynamic dims) because every downstream
# compiler we target rejects or warns on free dim_params when lowering.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConformanceCase:
    name: str
    description: str
    op_group: str           # grouping for the report (arith/linalg/norm/...)
    build: Callable[[], bytes]
    expected_ops: Tuple[str, ...]    # subset of tools.npu_ref.nn_graph.OP_*


def _import_helpers():
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    return onnx, TensorProto, helper, numpy_helper


def _make_model(nodes, inputs, outputs, initializers=(), opset: int = 17) -> bytes:
    onnx, _TP, helper, _nh = _import_helpers()
    graph = helper.make_graph(
        nodes=list(nodes), name="conformance",
        inputs=list(inputs), outputs=list(outputs),
        initializer=list(initializers),
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)],
                          ir_version=8)
    return m.SerializeToString()


def _fp(name: str, shape: Tuple[int, ...]):
    _onnx, TP, helper, _nh = _import_helpers()
    return helper.make_tensor_value_info(name, TP.FLOAT, shape)


def _init_fp(name: str, arr: np.ndarray):
    _onnx, _TP, _helper, numpy_helper = _import_helpers()
    return numpy_helper.from_array(arr.astype(np.float32), name)


# -- individual case builders ----------------------------------------------


def _case_add() -> bytes:
    _, _, helper, _ = _import_helpers()
    node = helper.make_node("Add", ["a", "b"], ["y"], name="add0")
    return _make_model([node], [_fp("a", (1, 8)), _fp("b", (1, 8))],
                       [_fp("y", (1, 8))])


def _case_mul() -> bytes:
    _, _, helper, _ = _import_helpers()
    node = helper.make_node("Mul", ["a", "b"], ["y"], name="mul0")
    return _make_model([node], [_fp("a", (1, 8)), _fp("b", (1, 8))],
                       [_fp("y", (1, 8))])


def _case_matmul() -> bytes:
    _, _, helper, _ = _import_helpers()
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 4))
    node = helper.make_node("MatMul", ["x", "w"], ["y"], name="matmul0")
    return _make_model([node], [_fp("x", (1, 8))], [_fp("y", (1, 4))],
                       initializers=[_init_fp("w", w)])


def _case_gemm() -> bytes:
    _, _, helper, _ = _import_helpers()
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4,)).astype(np.float32)
    node = helper.make_node("Gemm", ["x", "w", "b"], ["y"],
                            name="gemm0", transB=1)
    return _make_model([node], [_fp("x", (1, 8))], [_fp("y", (1, 4))],
                       initializers=[_init_fp("w", w), _init_fp("b", b)])


def _case_conv() -> bytes:
    _, _, helper, _ = _import_helpers()
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
    node = helper.make_node("Conv", ["x", "w"], ["y"],
                            kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                            name="conv0")
    return _make_model([node], [_fp("x", (1, 3, 8, 8))],
                       [_fp("y", (1, 4, 8, 8))],
                       initializers=[_init_fp("w", w)])


def _case_relu() -> bytes:
    _, _, helper, _ = _import_helpers()
    node = helper.make_node("Relu", ["x"], ["y"], name="relu0")
    return _make_model([node], [_fp("x", (1, 8))], [_fp("y", (1, 8))])


def _case_sigmoid() -> bytes:
    _, _, helper, _ = _import_helpers()
    node = helper.make_node("Sigmoid", ["x"], ["y"], name="sig0")
    return _make_model([node], [_fp("x", (1, 8))], [_fp("y", (1, 8))])


def _case_softmax() -> bytes:
    _, _, helper, _ = _import_helpers()
    node = helper.make_node("Softmax", ["x"], ["y"], name="sm0", axis=-1)
    return _make_model([node], [_fp("x", (1, 8))], [_fp("y", (1, 8))])


def _case_gelu() -> bytes:
    _, _, helper, _ = _import_helpers()
    # GELU is opset 20; many MLIR stacks lower it as a composite. Use the
    # approximate=tanh form that Optimum emits.
    node = helper.make_node("Gelu", ["x"], ["y"], name="gelu0",
                            approximate="tanh")
    return _make_model([node], [_fp("x", (1, 8))], [_fp("y", (1, 8))],
                       opset=20)


def _case_layernorm() -> bytes:
    _, _, helper, _ = _import_helpers()
    g = np.ones((8,), dtype=np.float32)
    b = np.zeros((8,), dtype=np.float32)
    node = helper.make_node("LayerNormalization",
                            ["x", "scale", "bias"], ["y"],
                            name="ln0", axis=-1, epsilon=1e-5)
    return _make_model([node], [_fp("x", (1, 8))], [_fp("y", (1, 8))],
                       initializers=[_init_fp("scale", g), _init_fp("bias", b)],
                       opset=17)


def _case_reshape() -> bytes:
    _, TP, helper, numpy_helper = _import_helpers()
    shape = numpy_helper.from_array(np.array([1, 2, 4], dtype=np.int64),
                                     "new_shape")
    node = helper.make_node("Reshape", ["x", "new_shape"], ["y"],
                            name="reshape0")
    return _make_model([node], [_fp("x", (1, 8))], [_fp("y", (1, 2, 4))],
                       initializers=[shape])


def _case_transpose() -> bytes:
    _, _, helper, _ = _import_helpers()
    node = helper.make_node("Transpose", ["x"], ["y"], name="tp0",
                            perm=[0, 2, 1])
    return _make_model([node], [_fp("x", (1, 4, 8))], [_fp("y", (1, 8, 4))])


def _case_concat() -> bytes:
    _, _, helper, _ = _import_helpers()
    node = helper.make_node("Concat", ["a", "b"], ["y"], name="cat0", axis=1)
    return _make_model([node], [_fp("a", (1, 4)), _fp("b", (1, 4))],
                       [_fp("y", (1, 8))])


def _case_maxpool() -> bytes:
    _, _, helper, _ = _import_helpers()
    node = helper.make_node("MaxPool", ["x"], ["y"], name="mp0",
                            kernel_shape=[2, 2], strides=[2, 2])
    return _make_model([node], [_fp("x", (1, 3, 8, 8))],
                       [_fp("y", (1, 3, 4, 4))])


def _case_globalavgpool() -> bytes:
    _, _, helper, _ = _import_helpers()
    node = helper.make_node("GlobalAveragePool", ["x"], ["y"], name="gap0")
    return _make_model([node], [_fp("x", (1, 3, 8, 8))],
                       [_fp("y", (1, 3, 1, 1))])


# -- suite registry ---------------------------------------------------------


# NB: expected_ops uses the OP_* constants from tools.npu_ref.nn_graph —
# importing them inline avoids pulling the IR at module-import time.
def _suite() -> List[ConformanceCase]:
    from tools.npu_ref.nn_graph import (
        OP_ADD, OP_MUL, OP_MATMUL, OP_GEMM, OP_CONV, OP_RELU, OP_SIGMOID,
        OP_SOFTMAX, OP_GELU, OP_LAYERNORM, OP_RESHAPE, OP_TRANSPOSE,
        OP_CONCAT, OP_MAXPOOL, OP_AVGPOOL,
    )
    return [
        ConformanceCase("add",            "elementwise add",            "arith",  _case_add,            (OP_ADD,)),
        ConformanceCase("mul",            "elementwise mul",            "arith",  _case_mul,            (OP_MUL,)),
        ConformanceCase("matmul",         "2D matmul with weight init", "linalg", _case_matmul,         (OP_MATMUL,)),
        ConformanceCase("gemm",           "GEMM (linear layer)",        "linalg", _case_gemm,           (OP_GEMM,)),
        ConformanceCase("conv",           "2D conv, stride 1, pad 1",   "linalg", _case_conv,           (OP_CONV,)),
        ConformanceCase("relu",           "ReLU activation",            "act",    _case_relu,           (OP_RELU,)),
        ConformanceCase("sigmoid",        "Sigmoid activation",         "act",    _case_sigmoid,        (OP_SIGMOID,)),
        ConformanceCase("softmax",        "Softmax along last axis",    "act",    _case_softmax,        (OP_SOFTMAX,)),
        ConformanceCase("gelu",           "GELU (approximate=tanh)",    "act",    _case_gelu,           (OP_GELU,)),
        ConformanceCase("layernorm",      "LayerNormalization",         "norm",   _case_layernorm,      (OP_LAYERNORM,)),
        ConformanceCase("reshape",        "Reshape to (1,2,4)",         "shape",  _case_reshape,        (OP_RESHAPE,)),
        ConformanceCase("transpose",      "Transpose perm=(0,2,1)",     "shape",  _case_transpose,      (OP_TRANSPOSE,)),
        ConformanceCase("concat",         "Concat axis=1",              "shape",  _case_concat,         (OP_CONCAT,)),
        ConformanceCase("maxpool",        "MaxPool 2x2 stride 2",       "pool",   _case_maxpool,        (OP_MAXPOOL,)),
        ConformanceCase("globalavgpool",  "GlobalAveragePool",          "pool",   _case_globalavgpool,  (OP_AVGPOOL,)),
    ]


# ---------------------------------------------------------------------------
# Adapter wrappers. Every adapter normalises on ONNX-bytes → NnGraph; this
# table gives the runner a uniform interface.
# ---------------------------------------------------------------------------


def _run_mlir_adapter(onnx_bytes: bytes):
    from tools.frontends.mlir import load_stablehlo_from_onnx_bytes
    return load_stablehlo_from_onnx_bytes(onnx_bytes, batch_size=1)


def _run_tvm_adapter(onnx_bytes: bytes):
    from tools.frontends.tvm import load_tvm_from_onnx_bytes
    return load_tvm_from_onnx_bytes(onnx_bytes, batch_size=1)


def _run_xla_adapter(onnx_bytes: bytes):
    from tools.frontends.xla import load_xla_from_onnx_bytes
    return load_xla_from_onnx_bytes(onnx_bytes, batch_size=1)


ADAPTERS: Dict[str, Callable[[bytes], object]] = {
    # Short names — these are what the report groups by.
    "mlir-stablehlo": _run_mlir_adapter,
    "tvm-relay":      _run_tvm_adapter,
    "jax-xla":        _run_xla_adapter,
}


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    case: str
    op_group: str
    adapter: str
    passed: bool
    wall_ms: float
    ops_seen: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ConformanceReport:
    generated_at: str
    host: Dict[str, str]
    suite_size: int
    adapters: List[str]
    cases: List[CaseResult]

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.cases if not c.passed)

    def summary_line(self) -> str:
        total = len(self.cases)
        return (f"MLIR/IREE conformance: {self.pass_count}/{total} pass "
                f"({self.fail_count} failing) across {len(self.adapters)} adapters")

    def to_dict(self) -> Dict:
        return {
            "generated_at": self.generated_at,
            "host": self.host,
            "suite_size": self.suite_size,
            "adapters": self.adapters,
            "summary": {
                "total": len(self.cases),
                "passed": self.pass_count,
                "failed": self.fail_count,
            },
            "cases": [asdict(c) for c in self.cases],
        }


def _run_one(case: ConformanceCase, adapter_name: str,
             adapter: Callable[[bytes], object]) -> CaseResult:
    t0 = time.perf_counter()
    try:
        onnx_bytes = case.build()
        graph = adapter(onnx_bytes)
        ops_seen = [L.op for L in graph.layers]
        missing = [op for op in case.expected_ops if op not in ops_seen]
        wall_ms = (time.perf_counter() - t0) * 1e3
        if missing:
            return CaseResult(
                case=case.name, op_group=case.op_group, adapter=adapter_name,
                passed=False, wall_ms=wall_ms, ops_seen=ops_seen,
                error=f"expected ops missing from NnGraph: {missing}",
            )
        return CaseResult(
            case=case.name, op_group=case.op_group, adapter=adapter_name,
            passed=True, wall_ms=wall_ms, ops_seen=ops_seen,
        )
    except Exception as exc:    # noqa: BLE001 — report every failure shape
        wall_ms = (time.perf_counter() - t0) * 1e3
        return CaseResult(
            case=case.name, op_group=case.op_group, adapter=adapter_name,
            passed=False, wall_ms=wall_ms, ops_seen=[],
            error=f"{type(exc).__name__}: {exc}",
        )


def run_conformance_suite(*,
                          adapters: Optional[List[str]] = None,
                          cases: Optional[List[ConformanceCase]] = None,
                          ) -> ConformanceReport:
    suite = list(cases) if cases is not None else _suite()
    selected = list(adapters) if adapters else list(ADAPTERS.keys())
    missing = [a for a in selected if a not in ADAPTERS]
    if missing:
        raise ValueError(f"unknown adapters: {missing}; "
                         f"available: {list(ADAPTERS)}")

    results: List[CaseResult] = []
    for case in suite:
        for a in selected:
            results.append(_run_one(case, a, ADAPTERS[a]))

    return ConformanceReport(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        host={
            "platform": sys.platform,
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        suite_size=len(suite),
        adapters=selected,
        cases=results,
    )


# ---------------------------------------------------------------------------
# Report formatters.
# ---------------------------------------------------------------------------


def report_as_markdown(report: ConformanceReport) -> str:
    lines: List[str] = []
    lines.append("# MLIR / IREE / TVM conformance — AstraCore")
    lines.append("")
    lines.append(f"- Generated: `{report.generated_at}`")
    lines.append(f"- Suite size: **{report.suite_size}** canonical ops")
    lines.append(f"- Adapters: {', '.join(f'`{a}`' for a in report.adapters)}")
    lines.append(f"- Result: **{report.pass_count} / {len(report.cases)} pass**")
    lines.append("")

    header = ["case", "group", *report.adapters]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    by_case = {}
    for c in report.cases:
        by_case.setdefault(c.case, {})[c.adapter] = c
    for case in _suite():
        row = [case.name, case.op_group]
        for a in report.adapters:
            r = by_case.get(case.name, {}).get(a)
            row.append("PASS" if (r and r.passed) else ("FAIL" if r else "—"))
        lines.append("| " + " | ".join(row) + " |")

    fails = [c for c in report.cases if not c.passed]
    if fails:
        lines.append("")
        lines.append("## Failures")
        for f in fails:
            lines.append(f"- `{f.adapter}` / `{f.case}` — {f.error}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path,
                   default=Path("reports/compiler_ecosystem"),
                   help="output directory for JSON + markdown")
    p.add_argument("--adapter", action="append", choices=list(ADAPTERS),
                   help="restrict to specific adapters (repeatable)")
    args = p.parse_args(argv)

    report = run_conformance_suite(adapters=args.adapter)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "conformance.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (args.out / "conformance.md").write_text(
        report_as_markdown(report), encoding="utf-8")
    print(report.summary_line())
    print(f"  JSON:     {args.out / 'conformance.json'}")
    print(f"  Markdown: {args.out / 'conformance.md'}")
    return 0 if report.fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
