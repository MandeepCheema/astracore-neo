"""Unit tests for tools/frontends/{tvm,mlir,xla,nnef}.py (F1-B5).

Each adapter converges on the F1-C1 loader via a .onnx intermediate.
These tests exercise the byte-string entry point (which every adapter
provides) by:

    1. Building a tiny ONNX graph with `onnx.helper` (same helper the
       F1-C1 unit tests use).
    2. Serialising it to bytes.
    3. Calling each adapter's `load_*_from_onnx_bytes` entry point.
    4. Asserting the returned NnGraph has the expected structure.

This catches adapter-level bugs (tempfile handling, cleanup, arg
plumbing) without depending on the actual source-format toolchains
(tvm, torch-mlir, jax, nnef-tools) being installed.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tools.frontends.mlir import load_stablehlo_from_onnx_bytes
from tools.frontends.nnef import load_nnef_from_onnx_bytes
from tools.frontends.tvm import load_tvm_from_onnx_bytes
from tools.frontends.xla import load_xla_from_onnx_bytes
from tools.npu_ref.nn_graph import OP_CONV, OP_RELU


def _tiny_conv_relu_model_bytes() -> bytes:
    """Build a minimal (Conv + ReLU) ONNX model and return it as bytes."""
    W = np.random.default_rng(42).standard_normal(
        (4, 3, 3, 3)).astype(np.float32)
    conv = helper.make_node(
        "Conv", inputs=["x", "w"], outputs=["y_pre"],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="conv0",
    )
    relu = helper.make_node(
        "Relu", inputs=["y_pre"], outputs=["y"], name="relu0",
    )
    graph = helper.make_graph(
        nodes=[conv, relu], name="t",
        inputs=[helper.make_tensor_value_info(
            "x", TensorProto.FLOAT, (1, 3, 8, 8))],
        outputs=[helper.make_tensor_value_info(
            "y", TensorProto.FLOAT, (1, 4, 8, 8))],
        initializer=[numpy_helper.from_array(W, "w")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    return model.SerializeToString()


@pytest.mark.parametrize("adapter", [
    load_tvm_from_onnx_bytes,
    load_stablehlo_from_onnx_bytes,
    load_xla_from_onnx_bytes,
    load_nnef_from_onnx_bytes,
])
def test_adapter_round_trips_conv_relu(adapter):
    """Every B5 adapter must load a trivial Conv+ReLU ONNX byte string
    into an NnGraph with the two ops present."""
    onnx_bytes = _tiny_conv_relu_model_bytes()
    g = adapter(onnx_bytes, batch_size=1)
    ops = [L.op for L in g.layers]
    assert OP_CONV in ops, f"{adapter.__name__}: Conv missing, got {ops}"
    assert OP_RELU in ops, f"{adapter.__name__}: ReLU missing, got {ops}"
    # Input shape round-trips.
    assert g.inputs["x"] == (1, 3, 8, 8)


def test_nnef_in_process_conversion_raises_without_package():
    """The in-process NNEF converter (`load_nnef`) is only available
    when nnef-tools is installed. Without it, we must raise a clear
    ImportError, not swallow the missing dependency."""
    from tools.frontends.nnef import load_nnef

    # If nnef-tools IS installed, skip the test rather than fake-fail.
    try:
        import nnef_tools  # noqa: F401
        pytest.skip("nnef-tools installed; cannot test missing-package path")
    except ImportError:
        pass

    with pytest.raises(ImportError, match="nnef-tools"):
        load_nnef("/nonexistent/path.nnef")
