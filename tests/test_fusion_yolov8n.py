"""F1-C1c acceptance — SiLU fusion on real yolov8n.onnx.

Gate: the YOLOv8 backbone has 57 Sigmoid+Mul SiLU patterns that must
all fuse in one pass, leaving only the one standalone Sigmoid (final
classification activation) and the one standalone Mul (bbox-decode
scale) that genuinely aren't SiLUs.

If fusion count drops, F1-C5 will compile 57 redundant tiles — cheap
but wasteful. If it rises above this baseline, a new fusable pattern
was added to the export or the detector logic regressed. Either way
we want to notice.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from tools.npu_ref.fusion import fuse_silu
from tools.npu_ref.nn_graph import OP_MUL, OP_SIGMOID, OP_SILU
from tools.npu_ref.onnx_loader import load_onnx

REPO = Path(__file__).resolve().parent.parent
ONNX_PATH = REPO / "data" / "models" / "yolov8n.onnx"


def _require_artifact():
    if not ONNX_PATH.exists():
        pytest.skip(
            f"{ONNX_PATH} not present. Run scripts/export_yolov8n_onnx.py."
        )


@pytest.fixture(scope="module")
def fused_graph():
    _require_artifact()
    g = load_onnx(str(ONNX_PATH))
    # Snapshot before fusion for the count asserts.
    before = Counter(L.op for L in g.layers)
    fuse_silu(g)
    after = Counter(L.op for L in g.layers)
    return g, before, after


def test_silu_fusions_match_expected_count(fused_graph):
    g, before, after = fused_graph
    # YOLOv8n.onnx from ultralytics 8.4.38: 58 Sigmoid + 58 Mul nodes
    # in the raw graph. 57 of each fuse (the standalone Sigmoid and
    # Mul in the detection head stay intact).
    assert before[OP_SIGMOID] == 58
    assert before[OP_MUL] == 58
    assert after[OP_SILU] == 57
    assert after[OP_SIGMOID] == 1
    assert after[OP_MUL] == 1
    assert g.metadata["silu_fusions"] == 57


def test_layer_count_dropped_by_fusion_count(fused_graph):
    """Net effect: each fusion removes 2 layers (Sigmoid + Mul) and
    adds 1 (SiLU) — so total layer count should drop by exactly
    silu_fusions."""
    g, before, after = fused_graph
    before_total = sum(before.values())
    after_total = sum(after.values())
    assert before_total - after_total == g.metadata["silu_fusions"]


def test_fused_graph_has_no_dangling_references(fused_graph):
    """Every layer input must be produced by an earlier layer, or
    be a graph-level input, or be the name of a graph initializer.
    If fusion got tensor routing wrong this would fail."""
    g, _, _ = fused_graph
    graph_inputs = set(g.inputs)
    produced: set[str] = set()
    for L in g.layers:
        for inp in L.inputs:
            if inp in graph_inputs:
                continue
            assert inp in produced, (
                f"layer {L.name!r} references unproduced tensor {inp!r}"
            )
        for out in L.outputs:
            produced.add(out)


def test_unfused_sigmoid_and_mul_are_distinct_subgraphs(fused_graph):
    """The leftover Sigmoid's output must not flow to the leftover
    Mul (that would indicate fusion incorrectly skipped a real SiLU)."""
    g, _, _ = fused_graph
    sigmoid_layer = next(L for L in g.layers if L.op == OP_SIGMOID)
    mul_layer = next(L for L in g.layers if L.op == OP_MUL)
    sig_out = sigmoid_layer.outputs[0]
    assert sig_out not in mul_layer.inputs, (
        "leftover Sigmoid feeds the leftover Mul — this should have fused"
    )
