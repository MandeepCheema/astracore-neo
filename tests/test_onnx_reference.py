"""Unit tests for tools/npu_ref/onnx_reference.py.

Covers:
  - End-to-end FP32 execution matches hand-computed expected output.
  - Intermediate tensor capture via graph output augmentation.
  - Input name / dtype validation rejects bad calls.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tools.npu_ref.onnx_reference import make_seeded_input, run_reference


def _build_relu_model() -> str:
    """Identity-conv + ReLU. Output = max(0, input @ identity-kernel)."""
    # 1x1 identity conv on 1 channel
    W = np.ones((1, 1, 1, 1), dtype=np.float32)
    conv = helper.make_node("Conv", inputs=["x", "w"], outputs=["c"],
                            kernel_shape=[1, 1], name="conv")
    relu = helper.make_node("Relu", inputs=["c"], outputs=["y"], name="relu")
    graph = helper.make_graph(
        nodes=[conv, relu], name="t",
        inputs=[helper.make_tensor_value_info(
            "x", TensorProto.FLOAT, (1, 1, 4, 4))],
        outputs=[helper.make_tensor_value_info(
            "y", TensorProto.FLOAT, (1, 1, 4, 4))],
        initializer=[numpy_helper.from_array(W, "w")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    fd, path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    onnx.save(model, path)
    return path


def test_end_to_end_forward():
    path = _build_relu_model()
    x = np.array([[[[-1, 0, 1, 2],
                    [-2, -1, 3, 4],
                    [0, 0, 0, 0],
                    [5, -5, 2, -2]]]], dtype=np.float32)
    run = run_reference(path, {"x": x})
    assert "y" in run.outputs
    np.testing.assert_array_equal(run.outputs["y"], np.maximum(x, 0.0))


def test_intermediate_capture():
    path = _build_relu_model()
    x = np.array([[[[-3, 4]]]], dtype=np.float32).reshape(1, 1, 1, 2)
    # Reshape input to match (1,1,4,4) expected by the model
    x = np.zeros((1, 1, 4, 4), dtype=np.float32)
    x[0, 0, 0, 0] = -3.0
    x[0, 0, 0, 1] = 4.0
    run = run_reference(path, {"x": x}, intermediate_names=["c"])
    # Pre-ReLU tensor "c" should carry the negative value
    assert "c" in run.activations
    assert run.activations["c"][0, 0, 0, 0] == -3.0
    # Post-ReLU graph output should clip it
    assert run.outputs["y"][0, 0, 0, 0] == 0.0


def test_input_name_mismatch_raises():
    path = _build_relu_model()
    x = np.zeros((1, 1, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="input name mismatch"):
        run_reference(path, {"wrong_name": x})


def test_input_dtype_mismatch_raises():
    path = _build_relu_model()
    x = np.zeros((1, 1, 4, 4), dtype=np.float64)
    with pytest.raises(TypeError, match="must be float32"):
        run_reference(path, {"x": x})


def test_seeded_input_is_deterministic():
    a = make_seeded_input((2, 3), seed=42)
    b = make_seeded_input((2, 3), seed=42)
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.float32
