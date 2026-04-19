"""Unit tests for tools/npu_ref/fusion.py (F1-C1c).

Covers:
  - Basic SiLU pattern fuses correctly.
  - SiLU pattern but sigmoid output consumed elsewhere → does NOT fuse.
  - Mul with unrelated second operand → does NOT fuse.
  - Bare Sigmoid with no matching Mul → untouched.
  - Multiple independent SiLUs all fuse in one pass.
  - Fused graph remains topologically ordered.
  - Graph metadata records the fusion count.
"""

from __future__ import annotations

import os
import tempfile
from typing import List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tools.npu_ref.fusion import fuse_silu
from tools.npu_ref.nn_graph import (
    OP_MUL,
    OP_SIGMOID,
    OP_SILU,
    NnGraph,
    NnLayer,
)
from tools.npu_ref.onnx_loader import load_onnx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save(model: onnx.ModelProto) -> str:
    fd, path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    onnx.save(model, path)
    return path


def _graph_to_onnx(nodes, inputs, outputs, initializers=None):
    g = helper.make_graph(
        nodes=nodes, name="t",
        inputs=[helper.make_tensor_value_info(n, t, s) for n, t, s in inputs],
        outputs=[helper.make_tensor_value_info(n, t, s) for n, t, s in outputs],
        initializer=initializers or [],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)],
                          ir_version=8)
    return _save(m)


def _is_topological(graph: NnGraph) -> bool:
    """Every layer's inputs must be produced by an earlier layer, or
    be a graph input / initializer."""
    graph_inputs = set(graph.inputs)
    produced_at: dict[str, int] = {}
    for i, L in enumerate(graph.layers):
        for inp in L.inputs:
            if inp in graph_inputs:
                continue
            if inp not in produced_at:
                return False
        for out in L.outputs:
            produced_at[out] = i
    return True


# ---------------------------------------------------------------------------
# Positive fusion — basic SiLU
# ---------------------------------------------------------------------------
def test_basic_silu_fuses():
    """x → Sigmoid → s;  Mul(x, s) → y  — should fuse to SiLU."""
    sig = helper.make_node("Sigmoid", inputs=["x"], outputs=["s"], name="sig")
    mul = helper.make_node("Mul", inputs=["x", "s"], outputs=["y"], name="mul")
    path = _graph_to_onnx(
        [sig, mul],
        inputs=[("x", TensorProto.FLOAT, (1, 4, 8, 8))],
        outputs=[("y", TensorProto.FLOAT, (1, 4, 8, 8))],
    )
    g = load_onnx(path)
    assert [L.op for L in g.layers] == [OP_SIGMOID, OP_MUL]
    fuse_silu(g)
    assert [L.op for L in g.layers] == [OP_SILU]
    silu = g.layers[0]
    assert silu.inputs == ["x"]
    assert silu.outputs == ["y"]
    assert silu.in_shapes.get("x") == (1, 4, 8, 8)
    assert silu.out_shapes.get("y") == (1, 4, 8, 8)
    assert g.metadata["silu_fusions"] == 1


def test_silu_with_swapped_mul_operand_order_fuses():
    """Mul(s, x) is the same pattern as Mul(x, s). Must fuse."""
    sig = helper.make_node("Sigmoid", inputs=["x"], outputs=["s"], name="sig")
    mul = helper.make_node("Mul", inputs=["s", "x"], outputs=["y"], name="mul")
    path = _graph_to_onnx(
        [sig, mul],
        inputs=[("x", TensorProto.FLOAT, (1, 4))],
        outputs=[("y", TensorProto.FLOAT, (1, 4))],
    )
    g = load_onnx(path)
    fuse_silu(g)
    assert [L.op for L in g.layers] == [OP_SILU]


# ---------------------------------------------------------------------------
# Negative fusion — must NOT fuse
# ---------------------------------------------------------------------------
def test_does_not_fuse_when_sigmoid_output_shared():
    """Sigmoid output consumed by both Mul and another Add; fusing
    would break the Add. Must leave both Sigmoid and Mul intact."""
    sig = helper.make_node("Sigmoid", inputs=["x"], outputs=["s"], name="sig")
    mul = helper.make_node("Mul", inputs=["x", "s"], outputs=["y"], name="mul")
    add = helper.make_node("Add", inputs=["s", "x"], outputs=["z"], name="add")
    path = _graph_to_onnx(
        [sig, mul, add],
        inputs=[("x", TensorProto.FLOAT, (1, 4))],
        outputs=[("y", TensorProto.FLOAT, (1, 4)),
                 ("z", TensorProto.FLOAT, (1, 4))],
    )
    g = load_onnx(path)
    fuse_silu(g)
    assert OP_SIGMOID in [L.op for L in g.layers]
    assert OP_SILU not in [L.op for L in g.layers]
    assert g.metadata["silu_fusions"] == 0


def test_does_not_fuse_when_mul_has_different_second_operand():
    """sigmoid(x) * y (y != x) — not a SiLU. Leave intact."""
    sig = helper.make_node("Sigmoid", inputs=["x"], outputs=["s"], name="sig")
    mul = helper.make_node("Mul", inputs=["y", "s"], outputs=["z"], name="mul")
    path = _graph_to_onnx(
        [sig, mul],
        inputs=[("x", TensorProto.FLOAT, (1, 4)),
                ("y", TensorProto.FLOAT, (1, 4))],
        outputs=[("z", TensorProto.FLOAT, (1, 4))],
    )
    g = load_onnx(path)
    fuse_silu(g)
    assert [L.op for L in g.layers] == [OP_SIGMOID, OP_MUL]
    assert g.metadata["silu_fusions"] == 0


def test_bare_sigmoid_untouched():
    sig = helper.make_node("Sigmoid", inputs=["x"], outputs=["y"], name="sig")
    path = _graph_to_onnx(
        [sig],
        inputs=[("x", TensorProto.FLOAT, (1, 4))],
        outputs=[("y", TensorProto.FLOAT, (1, 4))],
    )
    g = load_onnx(path)
    fuse_silu(g)
    assert [L.op for L in g.layers] == [OP_SIGMOID]
    assert g.metadata["silu_fusions"] == 0


# ---------------------------------------------------------------------------
# Multiple SiLUs in one graph
# ---------------------------------------------------------------------------
def test_two_independent_silus_both_fuse():
    sig1 = helper.make_node("Sigmoid", inputs=["x"], outputs=["s1"], name="sig1")
    mul1 = helper.make_node("Mul", inputs=["x", "s1"], outputs=["a"], name="mul1")
    sig2 = helper.make_node("Sigmoid", inputs=["a"], outputs=["s2"], name="sig2")
    mul2 = helper.make_node("Mul", inputs=["a", "s2"], outputs=["y"], name="mul2")
    path = _graph_to_onnx(
        [sig1, mul1, sig2, mul2],
        inputs=[("x", TensorProto.FLOAT, (1, 4))],
        outputs=[("y", TensorProto.FLOAT, (1, 4))],
    )
    g = load_onnx(path)
    fuse_silu(g)
    assert [L.op for L in g.layers] == [OP_SILU, OP_SILU]
    assert g.metadata["silu_fusions"] == 2
    assert _is_topological(g)


def test_fused_graph_stays_topological():
    """SiLU between a Conv and another Conv. After fusion, the conv
    consuming SiLU's output must still find its input."""
    W1 = np.ones((4, 3, 1, 1), dtype=np.float32)
    W2 = np.ones((2, 4, 1, 1), dtype=np.float32)
    conv1 = helper.make_node("Conv", inputs=["x", "w1"], outputs=["c"],
                              kernel_shape=[1, 1], name="c1")
    sig = helper.make_node("Sigmoid", inputs=["c"], outputs=["s"], name="sig")
    mul = helper.make_node("Mul", inputs=["c", "s"], outputs=["h"], name="mul")
    conv2 = helper.make_node("Conv", inputs=["h", "w2"], outputs=["y"],
                              kernel_shape=[1, 1], name="c2")
    path = _graph_to_onnx(
        [conv1, sig, mul, conv2],
        inputs=[("x", TensorProto.FLOAT, (1, 3, 8, 8))],
        outputs=[("y", TensorProto.FLOAT, (1, 2, 8, 8))],
        initializers=[numpy_helper.from_array(W1, "w1"),
                      numpy_helper.from_array(W2, "w2")],
    )
    g = load_onnx(path)
    fuse_silu(g)
    ops = [L.op for L in g.layers]
    assert ops.count(OP_SILU) == 1
    assert OP_SIGMOID not in ops and OP_MUL not in ops
    assert _is_topological(g)
    # conv2 must consume the SiLU's output (tensor name "h").
    conv2_layer = g.layers[-1]
    assert conv2_layer.inputs == ["h"]
    silu_layer = g.layers[1]
    assert silu_layer.outputs == ["h"]
