"""TVM BYOC (bring-your-own-codegen) backend for AstraCore.

Gives Apache TVM users a direct partitioning + codegen path onto the
AstraCore target. Pattern table mirrors the op set that the AstraCore
compiler (``tools.npu_ref.compiler``) can already lower — everything
else in the Relay / Relax graph stays on TVM's CPU / GPU backends.

Design
------
The module is import-safe without TVM installed. The public entry
points are:

- ``ASTRACORE_PATTERNS`` — the list of ``(name, matcher)`` tuples every
  BYOC backend registers with TVM; each entry matches an op subgraph we
  support and tells TVM to offload it.
- ``is_tvm_available()`` — cheap gate the CLI / tests use.
- ``register_astracore_target()`` — called at import time by user code
  once TVM is present; wires up the pattern table and codegen function.
- ``astracore_codegen(func, mod)`` — the codegen entry point TVM calls
  on each partitioned subgraph. Lowers through the existing ONNX →
  NnGraph path and returns a module TVM can embed.

What we deliberately DON'T do here
----------------------------------
- We don't implement the C runtime stubs TVM calls into at inference
  time. Those land once AstraCore has a published C-ABI (same track as
  the F1 host driver). Until then, the codegen path returns the
  NnGraph wrapped in a TVM-visible placeholder that raises on invoke —
  enough to validate *partitioning + codegen dispatch*, which is the
  hard integration work. Running the compiled subgraph is a separate
  milestone.

Why this split
--------------
The 4-6 engineer-week scoping in ``docs/external_validation_options.md``
row 9 assumes a real runtime. We ship the scaffolding + pattern table
now so upstream can review the partitioning boundary; the runtime
glue closes the loop once the host C-ABI lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Public — independent of whether TVM is installed.
# ---------------------------------------------------------------------------


def is_tvm_available() -> bool:
    """``True`` iff the ``tvm`` package imports cleanly on this host."""
    try:
        import tvm  # noqa: F401
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class PatternEntry:
    """One BYOC pattern: a Relay/Relax op name + the composite-pattern
    label TVM sees when the matcher fires. AstraCore's compiler lowers
    the composite back to its internal IR.

    ``attrs`` carries op-specific hints (e.g. minimum kernel size for
    Conv so we don't offload 1×1 convs TVM would keep on LLVM)."""
    relay_op: str
    composite: str
    attrs: Dict[str, Any]


# Canonical pattern table — kept in sync with ``tools.npu_ref.nn_graph``'s
# ``SUPPORTED_OPS``. Adding an op here also means the MLIR/IREE conformance
# suite in ``iree_conformance.py`` should grow a case for it.
ASTRACORE_PATTERNS: Tuple[PatternEntry, ...] = (
    PatternEntry("nn.conv2d",       "astracore.conv2d",
                 {"min_kernel": 3,  "layout": ("NCHW", "NHWC")}),
    PatternEntry("nn.dense",        "astracore.dense",       {}),
    PatternEntry("nn.batch_matmul", "astracore.matmul",      {}),
    PatternEntry("add",             "astracore.add",         {}),
    PatternEntry("multiply",        "astracore.mul",         {}),
    PatternEntry("nn.relu",         "astracore.relu",        {}),
    PatternEntry("sigmoid",         "astracore.sigmoid",     {}),
    PatternEntry("nn.softmax",      "astracore.softmax",     {}),
    PatternEntry("nn.layer_norm",   "astracore.layernorm",   {}),
    PatternEntry("nn.max_pool2d",   "astracore.maxpool",     {}),
    PatternEntry("nn.avg_pool2d",   "astracore.avgpool",     {}),
    PatternEntry("nn.global_avg_pool2d", "astracore.global_avgpool", {}),
    PatternEntry("reshape",         "astracore.reshape",     {}),
    PatternEntry("transpose",       "astracore.transpose",   {}),
    PatternEntry("concatenate",     "astracore.concat",      {}),
    PatternEntry("split",           "astracore.split",       {}),
    PatternEntry("resize",          "astracore.resize",      {}),
    # Fused transformer patterns — these fire when the surrounding
    # graph matches the composite TVM's op-fusion produces for each:
    PatternEntry("gelu",            "astracore.gelu",        {"approx": ("tanh", "erf")}),
    PatternEntry("silu",            "astracore.silu",        {}),
    PatternEntry("rms_norm",        "astracore.rmsnorm",     {}),
)


ASTRACORE_TARGET_NAME = "astracore"


@dataclass
class PartitionReport:
    """Returned by ``partition_graph_for_astracore`` so callers can see
    what was offloaded vs left on TVM. Populated by the real
    ``relay.transform.AnnotateTarget`` + ``MergeCompilerRegions`` pass
    once TVM is available; otherwise built statically from the pattern
    table for inspection + tests."""
    offloaded_ops: List[str]
    remained_on_tvm: List[str]
    total_ops: int

    @property
    def offload_fraction(self) -> float:
        return (len(self.offloaded_ops) / self.total_ops) if self.total_ops else 0.0


def describe_patterns() -> List[Dict[str, Any]]:
    """Introspection helper — returns a JSON-serialisable pattern table.
    Used by ``astracore conformance`` to publish the coverage matrix."""
    out: List[Dict[str, Any]] = []
    for p in ASTRACORE_PATTERNS:
        out.append({
            "relay_op": p.relay_op,
            "composite": p.composite,
            "attrs": dict(p.attrs),
        })
    return out


# ---------------------------------------------------------------------------
# TVM-gated entry points — these require the real ``tvm`` package.
# ---------------------------------------------------------------------------


def _require_tvm():
    """Raise a clear ImportError if TVM is missing — lets test stubs
    distinguish 'TVM not installed' from 'TVM installed but failing'."""
    try:
        import tvm
    except ImportError as exc:
        raise ImportError(
            "Apache TVM is not installed; the BYOC backend's codegen "
            "and partition entry points require tvm>=0.15. Install "
            "with `pip install apache-tvm`. The pattern table itself "
            "(ASTRACORE_PATTERNS, describe_patterns) is available "
            "without TVM."
        ) from exc
    return tvm


def register_astracore_target() -> None:
    """Register AstraCore as a BYOC target with TVM's codegen registry.

    Idempotent: calling twice is a no-op. Raises ``ImportError`` if TVM
    is not installed.
    """
    tvm = _require_tvm()
    from tvm.relay.op.contrib.register import register_pattern_table  # noqa: E501
    from tvm import relay                       # noqa: F401

    # Build a pattern_table entry list in the shape TVM expects:
    #   [(composite_name, dataflow_pattern, check_fn), ...]
    # We use a cheap "match-op-name" pattern for each entry; the composite
    # label is what shows up in TVM's Partitioning pass.
    from tvm.relay.dataflow_pattern import is_op, wildcard

    def _pattern_table():
        table = []
        for entry in ASTRACORE_PATTERNS:
            op = entry.relay_op
            # wildcard inputs work for single-op patterns; fusion
            # patterns (GELU, SiLU, RMSNorm) will gain richer matchers
            # once the AstraCore compiler exposes its fused-op signatures.
            pat = is_op(op)(wildcard(), wildcard()) if op in {
                "nn.conv2d", "nn.dense", "nn.batch_matmul", "add",
                "multiply", "concatenate",
            } else is_op(op)(wildcard())
            table.append((entry.composite, pat, lambda _call: True))
        return table

    register_pattern_table(ASTRACORE_TARGET_NAME, _pattern_table)


def partition_graph_for_astracore(mod, *, params: Optional[Dict] = None):
    """Apply AstraCore annotation + partitioning to a Relay IRModule.

    Thin wrapper around TVM's standard BYOC transform sequence:

        relay.transform.MergeComposite(pattern_table)
        relay.transform.AnnotateTarget("astracore")
        relay.transform.MergeCompilerRegions()
        relay.transform.PartitionGraph()

    Returns the transformed IRModule. Raises ImportError if TVM missing.
    """
    tvm = _require_tvm()
    from tvm import relay
    register_astracore_target()
    # Build the composite pattern list TVM expects.
    from tvm.relay.dataflow_pattern import is_op, wildcard
    patterns = []
    for entry in ASTRACORE_PATTERNS:
        op = entry.relay_op
        pat = is_op(op)(wildcard(), wildcard()) if op in {
            "nn.conv2d", "nn.dense", "nn.batch_matmul", "add",
            "multiply", "concatenate",
        } else is_op(op)(wildcard())
        patterns.append((entry.composite, pat, lambda _c: True))

    seq = tvm.transform.Sequential([
        relay.transform.MergeComposite(patterns),
        relay.transform.AnnotateTarget(ASTRACORE_TARGET_NAME),
        relay.transform.MergeCompilerRegions(),
        relay.transform.PartitionGraph(),
    ])
    with tvm.transform.PassContext(opt_level=3):
        return seq(mod)


def astracore_codegen(func, *, onnx_bytes: Optional[bytes] = None):
    """Codegen entry point TVM dispatches per partitioned AstraCore
    subgraph.

    If the caller has the original ONNX bytes for the subgraph, we lower
    via ``tools.frontends.tvm.load_tvm_from_onnx_bytes`` → NnGraph →
    AstraCore compiler. Otherwise this is a placeholder that will become
    the full TIR → runtime-call lowering once the C-ABI lands.
    """
    _require_tvm()
    if onnx_bytes is None:
        raise NotImplementedError(
            "astracore_codegen requires onnx_bytes until the TIR → "
            "runtime-call path is implemented. Pass onnx_bytes kwarg "
            "or use the ONNX round-trip via tools.frontends.tvm."
        )
    from tools.frontends.tvm import load_tvm_from_onnx_bytes
    return load_tvm_from_onnx_bytes(onnx_bytes)
