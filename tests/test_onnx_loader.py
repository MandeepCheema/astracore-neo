"""Unit tests for tools/npu_ref/onnx_loader.py (F1-C1).

Each test builds a tiny single-op ONNX model via `onnx.helper`, runs it
through `load_onnx`, and asserts the resulting NnLayer carries exactly
the fields downstream passes (F1-C2 quantiser, F1-C3/C4 tiler) will read.

Tests here catch handler-level bugs. The yolov8n.onnx acceptance test
(tests/test_onnx_yolov8n.py) catches integration-level regressions.
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Tuple

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tools.npu_ref.nn_graph import (
    NnGraph,
    OP_ADD,
    OP_AVGPOOL,
    OP_CONCAT,
    OP_CONV,
    OP_DIV,
    OP_GELU,
    OP_GEMM,
    OP_LAYERNORM,
    OP_MATMUL,
    OP_MAXPOOL,
    OP_MHA,
    OP_MUL,
    OP_RELU,
    OP_RESHAPE,
    OP_RESIZE,
    OP_RMSNORM,
    OP_ROTARY_EMB,
    OP_SIGMOID,
    OP_SLICE,
    OP_SOFTMAX,
    OP_SPLIT,
    OP_SUB,
    OP_TRANSPOSE,
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


def _make_model(node: onnx.NodeProto,
                inputs: List[Tuple[str, int, Tuple[int, ...]]],
                outputs: List[Tuple[str, int, Tuple[int, ...]]],
                initializers: List[onnx.TensorProto] = None,
                opset: int = 17) -> str:
    """Wrap one or more nodes (pass a list via `node.op_type == 'SEQ'`
    by using _make_graph for multi-node) into a minimal runnable model."""
    return _make_graph([node], inputs, outputs, initializers, opset)


def _make_graph(nodes: List[onnx.NodeProto],
                inputs: List[Tuple[str, int, Tuple[int, ...]]],
                outputs: List[Tuple[str, int, Tuple[int, ...]]],
                initializers: List[onnx.TensorProto] = None,
                opset: int = 17) -> str:
    graph = helper.make_graph(
        nodes=nodes,
        name="test",
        inputs=[helper.make_tensor_value_info(n, t, s) for n, t, s in inputs],
        outputs=[helper.make_tensor_value_info(n, t, s) for n, t, s in outputs],
        initializer=initializers or [],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=8,
    )
    model.producer_name = "test_onnx_loader"
    return _save(model)


# ---------------------------------------------------------------------------
# Conv
# ---------------------------------------------------------------------------
def test_conv_basic():
    W = np.random.randn(16, 3, 3, 3).astype(np.float32)
    B = np.zeros((16,), dtype=np.float32)
    node = helper.make_node(
        "Conv", inputs=["x", "w", "b"], outputs=["y"],
        kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1], group=1,
        name="stem",
    )
    path = _make_model(
        node,
        inputs=[("x", TensorProto.FLOAT, (1, 3, 640, 640))],
        outputs=[("y", TensorProto.FLOAT, (1, 16, 320, 320))],
        initializers=[
            numpy_helper.from_array(W, "w"),
            numpy_helper.from_array(B, "b"),
        ],
    )
    g = load_onnx(path)
    assert len(g) == 1
    L = g.layers[0]
    assert L.op == OP_CONV
    assert L.name == "stem"
    assert L.inputs == ["x"]
    assert L.outputs == ["y"]
    assert L.attrs["kernel"] == (3, 3)
    assert L.attrs["stride"] == (2, 2)
    assert L.attrs["pad"] == (1, 1, 1, 1)
    assert L.attrs["groups"] == 1
    assert L.attrs["dilation"] == (1, 1)
    assert L.weights is not None and L.weights.shape == (16, 3, 3, 3)
    assert L.weights.dtype == np.float32
    assert L.bias is not None and L.bias.shape == (16,)
    assert L.in_shapes["x"] == (1, 3, 640, 640)
    assert L.out_shapes["y"] == (1, 16, 320, 320)


def test_conv_no_bias():
    W = np.random.randn(8, 4, 1, 1).astype(np.float32)
    node = helper.make_node(
        "Conv", inputs=["x", "w"], outputs=["y"], kernel_shape=[1, 1],
        name="pw",
    )
    path = _make_model(
        node,
        inputs=[("x", TensorProto.FLOAT, (1, 4, 32, 32))],
        outputs=[("y", TensorProto.FLOAT, (1, 8, 32, 32))],
        initializers=[numpy_helper.from_array(W, "w")],
    )
    g = load_onnx(path)
    L = g.layers[0]
    assert L.bias is None
    assert L.weights.shape == (8, 4, 1, 1)


def test_conv_weight_not_constant_raises():
    node = helper.make_node(
        "Conv", inputs=["x", "w"], outputs=["y"], kernel_shape=[3, 3],
    )
    path = _make_model(
        node,
        inputs=[("x", TensorProto.FLOAT, (1, 3, 8, 8)),
                ("w", TensorProto.FLOAT, (4, 3, 3, 3))],
        outputs=[("y", TensorProto.FLOAT, (1, 4, 6, 6))],
    )
    with pytest.raises(ValueError, match="not in initializers"):
        load_onnx(path)


# ---------------------------------------------------------------------------
# Gemm / MatMul
# ---------------------------------------------------------------------------
def test_gemm():
    W = np.random.randn(256, 128).astype(np.float32)
    B = np.zeros((256,), dtype=np.float32)
    node = helper.make_node(
        "Gemm", inputs=["x", "w", "b"], outputs=["y"],
        transB=1, alpha=1.0, beta=1.0,
    )
    path = _make_model(
        node,
        inputs=[("x", TensorProto.FLOAT, (1, 128))],
        outputs=[("y", TensorProto.FLOAT, (1, 256))],
        initializers=[
            numpy_helper.from_array(W, "w"),
            numpy_helper.from_array(B, "b"),
        ],
    )
    g = load_onnx(path)
    L = g.layers[0]
    assert L.op == OP_GEMM
    assert L.attrs["trans_b"] == 1
    assert L.attrs["alpha"] == pytest.approx(1.0)
    assert L.weights.shape == (256, 128)


def test_matmul_with_constant_rhs():
    W = np.random.randn(64, 32).astype(np.float32)
    node = helper.make_node("MatMul", inputs=["x", "w"], outputs=["y"])
    path = _make_model(
        node,
        inputs=[("x", TensorProto.FLOAT, (1, 64))],
        outputs=[("y", TensorProto.FLOAT, (1, 32))],
        initializers=[numpy_helper.from_array(W, "w")],
    )
    g = load_onnx(path)
    L = g.layers[0]
    assert L.op == OP_MATMUL
    assert L.weights.shape == (64, 32)
    assert L.inputs == ["x"]


# ---------------------------------------------------------------------------
# Element-wise
# ---------------------------------------------------------------------------
def test_add_two_activations():
    node = helper.make_node("Add", inputs=["a", "b"], outputs=["y"])
    path = _make_model(
        node,
        inputs=[("a", TensorProto.FLOAT, (1, 16, 8, 8)),
                ("b", TensorProto.FLOAT, (1, 16, 8, 8))],
        outputs=[("y", TensorProto.FLOAT, (1, 16, 8, 8))],
    )
    g = load_onnx(path)
    L = g.layers[0]
    assert L.op == OP_ADD
    assert L.weights is None
    assert L.inputs == ["a", "b"]


def test_mul_with_constant():
    C = np.array([2.0], dtype=np.float32)
    node = helper.make_node("Mul", inputs=["x", "c"], outputs=["y"])
    path = _make_model(
        node,
        inputs=[("x", TensorProto.FLOAT, (1, 4, 4))],
        outputs=[("y", TensorProto.FLOAT, (1, 4, 4))],
        initializers=[numpy_helper.from_array(C, "c")],
    )
    g = load_onnx(path)
    L = g.layers[0]
    assert L.op == OP_MUL
    assert L.weights is not None
    assert L.inputs == ["x"]


def test_sub_two_activations():
    node = helper.make_node("Sub", inputs=["a", "b"], outputs=["y"])
    path = _make_model(
        node,
        inputs=[("a", TensorProto.FLOAT, (1, 8)),
                ("b", TensorProto.FLOAT, (1, 8))],
        outputs=[("y", TensorProto.FLOAT, (1, 8))],
    )
    g = load_onnx(path)
    L = g.layers[0]
    assert L.op == OP_SUB
    assert L.inputs == ["a", "b"]
    assert L.weights is None


def test_div_with_constant_denominator():
    C = np.array([10.0], dtype=np.float32)
    node = helper.make_node("Div", inputs=["x", "c"], outputs=["y"])
    path = _make_model(
        node,
        inputs=[("x", TensorProto.FLOAT, (1, 8))],
        outputs=[("y", TensorProto.FLOAT, (1, 8))],
        initializers=[numpy_helper.from_array(C, "c")],
    )
    g = load_onnx(path)
    L = g.layers[0]
    assert L.op == OP_DIV
    assert L.inputs == ["x"]
    assert L.weights is not None


# ---------------------------------------------------------------------------
# Unary / shape ops
# ---------------------------------------------------------------------------
def test_sigmoid():
    node = helper.make_node("Sigmoid", inputs=["x"], outputs=["y"])
    path = _make_model(
        node,
        inputs=[("x", TensorProto.FLOAT, (1, 8))],
        outputs=[("y", TensorProto.FLOAT, (1, 8))],
    )
    g = load_onnx(path)
    assert g.layers[0].op == OP_SIGMOID


def test_relu():
    node = helper.make_node("Relu", inputs=["x"], outputs=["y"])
    path = _make_model(
        node,
        inputs=[("x", TensorProto.FLOAT, (1, 8))],
        outputs=[("y", TensorProto.FLOAT, (1, 8))],
    )
    g = load_onnx(path)
    assert g.layers[0].op == OP_RELU


def test_softmax():
    node = helper.make_node("Softmax", inputs=["x"], outputs=["y"], axis=-1)
    path = _make_model(
        node,
        inputs=[("x", TensorProto.FLOAT, (1, 10))],
        outputs=[("y", TensorProto.FLOAT, (1, 10))],
    )
    g = load_onnx(path)
    L = g.layers[0]
    assert L.op == OP_SOFTMAX
    assert L.attrs["axis"] == -1


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------
def test_maxpool():
    node = helper.make_node(
        "MaxPool", inputs=["x"], outputs=["y"],
        kernel_shape=[5, 5], strides=[1, 1], pads=[2, 2, 2, 2],
    )
    path = _make_model(
        node,
        inputs=[("x", TensorProto.FLOAT, (1, 32, 20, 20))],
        outputs=[("y", TensorProto.FLOAT, (1, 32, 20, 20))],
    )
    g = load_onnx(path)
    L = g.layers[0]
    assert L.op == OP_MAXPOOL
    assert L.attrs["kernel"] == (5, 5)
    assert L.attrs["pad"] == (2, 2, 2, 2)


def test_average_pool():
    """G6: AveragePool handler was in the dispatch table but never
    exercised by a test (YOLOv8n uses GlobalAveragePool + MaxPool).
    Keep the handler covered so the path is regression-safe for
    future models (MobileNet, ResNet-avg-pool) that hit it."""
    node = helper.make_node(
        "AveragePool", inputs=["x"], outputs=["y"],
        kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0],
    )
    path = _make_model(
        node,
        inputs=[("x", TensorProto.FLOAT, (1, 16, 8, 8))],
        outputs=[("y", TensorProto.FLOAT, (1, 16, 4, 4))],
    )
    g = load_onnx(path)
    L = g.layers[0]
    assert L.op == OP_AVGPOOL
    assert L.attrs["kernel"] == (2, 2)
    assert L.attrs["stride"] == (2, 2)
    assert L.attrs.get("global") is not True  # distinguish from GAP


def test_global_avg_pool():
    node = helper.make_node("GlobalAveragePool", inputs=["x"], outputs=["y"])
    path = _make_model(
        node,
        inputs=[("x", TensorProto.FLOAT, (1, 128, 7, 7))],
        outputs=[("y", TensorProto.FLOAT, (1, 128, 1, 1))],
    )
    g = load_onnx(path)
    L = g.layers[0]
    assert L.op == OP_AVGPOOL
    assert L.attrs["global"] is True
    assert L.attrs["kernel"] == (7, 7)


# ---------------------------------------------------------------------------
# Concat / Split
# ---------------------------------------------------------------------------
def test_concat():
    node = helper.make_node(
        "Concat", inputs=["a", "b", "c"], outputs=["y"], axis=1,
    )
    path = _make_model(
        node,
        inputs=[("a", TensorProto.FLOAT, (1, 8, 16, 16)),
                ("b", TensorProto.FLOAT, (1, 8, 16, 16)),
                ("c", TensorProto.FLOAT, (1, 8, 16, 16))],
        outputs=[("y", TensorProto.FLOAT, (1, 24, 16, 16))],
    )
    g = load_onnx(path)
    L = g.layers[0]
    assert L.op == OP_CONCAT
    assert L.attrs["axis"] == 1
    assert L.inputs == ["a", "b", "c"]


def test_split_with_sizes_input():
    sizes = np.array([8, 8], dtype=np.int64)
    node = helper.make_node(
        "Split", inputs=["x", "sizes"], outputs=["a", "b"], axis=1,
    )
    path = _make_model(
        node,
        inputs=[("x", TensorProto.FLOAT, (1, 16, 4, 4))],
        outputs=[("a", TensorProto.FLOAT, (1, 8, 4, 4)),
                 ("b", TensorProto.FLOAT, (1, 8, 4, 4))],
        initializers=[numpy_helper.from_array(sizes, "sizes")],
    )
    g = load_onnx(path)
    L = g.layers[0]
    assert L.op == OP_SPLIT
    assert L.attrs["axis"] == 1
    assert L.attrs["split"] == (8, 8)


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------
def test_resize_with_scales():
    roi = np.array([], dtype=np.float32)
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    node = helper.make_node(
        "Resize", inputs=["x", "roi", "scales"], outputs=["y"],
        mode="nearest",
    )
    path = _make_model(
        node,
        inputs=[("x", TensorProto.FLOAT, (1, 32, 20, 20))],
        outputs=[("y", TensorProto.FLOAT, (1, 32, 40, 40))],
        initializers=[
            numpy_helper.from_array(roi, "roi"),
            numpy_helper.from_array(scales, "scales"),
        ],
    )
    g = load_onnx(path)
    L = g.layers[0]
    assert L.op == OP_RESIZE
    assert L.attrs["mode"] == "nearest"
    assert L.attrs["scales"] == (1.0, 1.0, 2.0, 2.0)


# ---------------------------------------------------------------------------
# Reshape / Transpose / Slice
# ---------------------------------------------------------------------------
def test_reshape():
    shape = np.array([1, -1], dtype=np.int64)
    node = helper.make_node("Reshape", inputs=["x", "s"], outputs=["y"])
    path = _make_model(
        node,
        inputs=[("x", TensorProto.FLOAT, (1, 4, 8))],
        outputs=[("y", TensorProto.FLOAT, (1, 32))],
        initializers=[numpy_helper.from_array(shape, "s")],
    )
    g = load_onnx(path)
    L = g.layers[0]
    assert L.op == OP_RESHAPE
    assert L.attrs["shape"] == (1, -1)


def test_transpose():
    node = helper.make_node("Transpose", inputs=["x"], outputs=["y"],
                            perm=[0, 2, 3, 1])
    path = _make_model(
        node,
        inputs=[("x", TensorProto.FLOAT, (1, 4, 8, 8))],
        outputs=[("y", TensorProto.FLOAT, (1, 8, 8, 4))],
    )
    g = load_onnx(path)
    L = g.layers[0]
    assert L.op == OP_TRANSPOSE
    assert L.attrs["perm"] == (0, 2, 3, 1)


def test_slice():
    starts = np.array([0], dtype=np.int64)
    ends = np.array([4], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    node = helper.make_node(
        "Slice",
        inputs=["x", "starts", "ends", "axes"],
        outputs=["y"],
    )
    path = _make_model(
        node,
        inputs=[("x", TensorProto.FLOAT, (1, 8, 4, 4))],
        outputs=[("y", TensorProto.FLOAT, (1, 4, 4, 4))],
        initializers=[
            numpy_helper.from_array(starts, "starts"),
            numpy_helper.from_array(ends, "ends"),
            numpy_helper.from_array(axes, "axes"),
        ],
    )
    g = load_onnx(path)
    L = g.layers[0]
    assert L.op == OP_SLICE
    assert L.attrs["starts"] == (0,)
    assert L.attrs["ends"] == (4,)
    assert L.attrs["axes"] == (1,)


# ---------------------------------------------------------------------------
# Loader-level behaviour
# ---------------------------------------------------------------------------
def test_unsupported_op_strict():
    # Abs isn't in our handler table; strict mode should raise.
    node = helper.make_node("Abs", inputs=["x"], outputs=["y"])
    path = _make_model(
        node,
        inputs=[("x", TensorProto.FLOAT, (1, 4))],
        outputs=[("y", TensorProto.FLOAT, (1, 4))],
    )
    with pytest.raises(NotImplementedError, match="Abs"):
        load_onnx(path)


def test_unsupported_op_non_strict_skips():
    node = helper.make_node("Abs", inputs=["x"], outputs=["y"])
    path = _make_model(
        node,
        inputs=[("x", TensorProto.FLOAT, (1, 4))],
        outputs=[("y", TensorProto.FLOAT, (1, 4))],
    )
    g = load_onnx(path, strict=False)
    assert len(g) == 0
    assert g.metadata["unsupported_ops"] == ["Abs"]


def test_multi_op_graph_order():
    # Conv → Sigmoid → Mul — a YOLO-style SiLU fragment. Ensures layers
    # arrive in topological order.
    W = np.random.randn(4, 3, 1, 1).astype(np.float32)
    conv = helper.make_node("Conv", inputs=["x", "w"], outputs=["c"],
                            kernel_shape=[1, 1], name="conv")
    sig = helper.make_node("Sigmoid", inputs=["c"], outputs=["s"], name="sig")
    mul = helper.make_node("Mul", inputs=["c", "s"], outputs=["y"], name="mul")
    path = _make_graph(
        [conv, sig, mul],
        inputs=[("x", TensorProto.FLOAT, (1, 3, 8, 8))],
        outputs=[("y", TensorProto.FLOAT, (1, 4, 8, 8))],
        initializers=[numpy_helper.from_array(W, "w")],
    )
    g = load_onnx(path)
    assert [L.op for L in g.layers] == [OP_CONV, OP_SIGMOID, OP_MUL]
    assert [L.name for L in g.layers] == ["conv", "sig", "mul"]


def test_graph_io_excludes_initializers():
    # Older models list initializers in graph.input; our I/O map must
    # skip them.
    W = np.random.randn(4, 3, 1, 1).astype(np.float32)
    conv = helper.make_node("Conv", inputs=["x", "w"], outputs=["y"],
                            kernel_shape=[1, 1])
    path = _make_graph(
        [conv],
        inputs=[("x", TensorProto.FLOAT, (1, 3, 8, 8))],
        outputs=[("y", TensorProto.FLOAT, (1, 4, 8, 8))],
        initializers=[numpy_helper.from_array(W, "w")],
    )
    g = load_onnx(path)
    assert set(g.inputs.keys()) == {"x"}
    assert set(g.outputs.keys()) == {"y"}


def test_dynamic_batch_concretised():
    # Declare the batch dim as symbolic; loader should concretise to 1.
    W = np.random.randn(4, 3, 1, 1).astype(np.float32)
    conv = helper.make_node("Conv", inputs=["x", "w"], outputs=["y"],
                            kernel_shape=[1, 1])
    # Use a symbolic dim by passing a string.
    x_vi = helper.make_tensor_value_info(
        "x", TensorProto.FLOAT, ("N", 3, 8, 8))
    y_vi = helper.make_tensor_value_info(
        "y", TensorProto.FLOAT, ("N", 4, 8, 8))
    graph = helper.make_graph(
        nodes=[conv], name="dyn", inputs=[x_vi], outputs=[y_vi],
        initializer=[numpy_helper.from_array(W, "w")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    fd, path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    onnx.save(model, path)
    g = load_onnx(path, batch_size=1)
    assert g.inputs["x"] == (1, 3, 8, 8)
    assert g.layers[0].in_shapes["x"] == (1, 3, 8, 8)


# ---------------------------------------------------------------------------
# F1-B1: Transformer ops — GELU, LayerNorm, RMSNorm, RotaryEmbedding, MHA
# ---------------------------------------------------------------------------
def _save_single_node(node, inputs, outputs, initializer=None, opset=17):
    graph = helper.make_graph(
        nodes=[node], name="t", inputs=inputs, outputs=outputs,
        initializer=initializer or [],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=8)
    return _save(model)


def test_gelu_tanh_approx_loads():
    node = helper.make_node(
        "Gelu", inputs=["x"], outputs=["y"], approximate="tanh")
    path = _save_single_node(
        node,
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, (1, 8))],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, (1, 8))],
        opset=20,
    )
    g = load_onnx(path)
    L = g.layers[0]
    assert L.op == OP_GELU
    assert L.attrs["approximate"] == "tanh"


def test_gelu_default_erf_loads():
    node = helper.make_node("Gelu", inputs=["x"], outputs=["y"])
    path = _save_single_node(
        node,
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, (1, 8))],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, (1, 8))],
        opset=20,
    )
    g = load_onnx(path)
    assert g.layers[0].op == OP_GELU
    assert g.layers[0].attrs["approximate"] == "none"


def test_layernorm_captures_scale_bias_and_eps():
    scale = np.full((16,), 1.5, dtype=np.float32)
    bias = np.full((16,), 0.25, dtype=np.float32)
    node = helper.make_node(
        "LayerNormalization", inputs=["x", "s", "b"], outputs=["y"],
        axis=-1, epsilon=1e-3)
    path = _save_single_node(
        node,
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, (2, 16))],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, (2, 16))],
        initializer=[numpy_helper.from_array(scale, "s"),
                     numpy_helper.from_array(bias, "b")],
    )
    g = load_onnx(path)
    L = g.layers[0]
    assert L.op == OP_LAYERNORM
    np.testing.assert_allclose(L.weights, scale)
    np.testing.assert_allclose(L.bias, bias)
    assert L.attrs["epsilon"] == pytest.approx(1e-3)
    assert L.attrs["axis"] == -1
    # Scale and bias initializer names must not end up in the
    # activation input list — only "x" is an activation.
    assert L.inputs == ["x"]


def test_rmsnorm_captures_scale_and_eps():
    scale = np.ones((8,), dtype=np.float32)
    node = helper.make_node(
        "RMSNormalization", inputs=["x", "s"], outputs=["y"],
        axis=-1, epsilon=1e-6)
    path = _save_single_node(
        node,
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, (1, 8))],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, (1, 8))],
        initializer=[numpy_helper.from_array(scale, "s")],
        opset=23,
    )
    g = load_onnx(path)
    L = g.layers[0]
    assert L.op == OP_RMSNORM
    np.testing.assert_allclose(L.weights, scale)
    assert L.attrs["epsilon"] == pytest.approx(1e-6)
    assert L.inputs == ["x"]


def test_rotary_embedding_captures_cos_sin_caches():
    cos_cache = np.ones((32, 4), dtype=np.float32)
    sin_cache = np.zeros((32, 4), dtype=np.float32)
    # X shape (B, S, num_heads, head_dim); the RoPE op preserves it.
    node = helper.make_node(
        "RotaryEmbedding", inputs=["x", "cos", "sin"], outputs=["y"],
        interleaved=0, num_heads=4)
    path = _save_single_node(
        node,
        inputs=[helper.make_tensor_value_info(
            "x", TensorProto.FLOAT, (1, 32, 4, 8))],
        outputs=[helper.make_tensor_value_info(
            "y", TensorProto.FLOAT, (1, 32, 4, 8))],
        initializer=[numpy_helper.from_array(cos_cache, "cos"),
                     numpy_helper.from_array(sin_cache, "sin")],
        opset=23,
    )
    g = load_onnx(path)
    L = g.layers[0]
    assert L.op == OP_ROTARY_EMB
    np.testing.assert_allclose(L.weights, cos_cache)
    np.testing.assert_allclose(L.bias, sin_cache)
    assert L.attrs["num_heads"] == 4
    assert L.attrs["interleaved"] is False


def test_multi_head_attention_captures_heads_and_scale():
    # MultiHeadAttention is an ORT contrib op in the "com.microsoft"
    # domain. We skip the standard ONNX checker (doesn't know contrib
    # ops) and load the raw graph.
    node = helper.make_node(
        "MultiHeadAttention", inputs=["q", "k", "v"], outputs=["y"],
        domain="com.microsoft", num_heads=8, scale=0.125)
    graph = helper.make_graph(
        nodes=[node], name="t",
        inputs=[
            helper.make_tensor_value_info("q", TensorProto.FLOAT, (1, 16, 64)),
            helper.make_tensor_value_info("k", TensorProto.FLOAT, (1, 16, 64)),
            helper.make_tensor_value_info("v", TensorProto.FLOAT, (1, 16, 64)),
        ],
        outputs=[helper.make_tensor_value_info(
            "y", TensorProto.FLOAT, (1, 16, 64))],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=8,
    )
    fd, path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    onnx.save(model, path)
    g = load_onnx(path)
    L = g.layers[0]
    assert L.op == OP_MHA
    assert L.attrs["num_heads"] == 8
    assert L.attrs["scale"] == pytest.approx(0.125)
    assert L.inputs == ["q", "k", "v"]
