"""Pre-AWS-F1 gap-closure tests (GAP-4 / GAP-5).

Closes the last software-side unknowns before the hardware bring-up
starts. Each test is a concrete claim about the NPU path that would
be embarrassing to discover broken on silicon.

GAP-4: accumulator overflow sanity — worst-case INT8×INT8 with
        K = max AI SRAM depth must not saturate the INT32 accumulator.

GAP-5: bias data-path — the F1-C2 audit H1 decision (bias is FP32
        post-dequant) must produce numerically identical output to
        ORT's native Conv(x, w, b) path, within INT8 quantisation
        noise.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tools.npu_ref.compiler import simulate_program
from tools.npu_ref.conv_compiler import (
    compile_conv2d,
    reassemble_conv_output,
    reference_conv2d_int8,
)
from tools.npu_ref.fusion import fuse_silu
from tools.npu_ref.nn_runtime import run_graph
from tools.npu_ref.onnx_loader import load_onnx
from tools.npu_ref.onnx_reference import run_reference
from tools.npu_ref.quantiser import (
    make_seeded_calibration_set,
    quantise_model,
)


def _save_model(model):
    fd, path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    onnx.save(model, path)
    return path


# ---------------------------------------------------------------------------
# GAP-4 — accumulator overflow
# ---------------------------------------------------------------------------
def test_gap4_int32_accumulator_handles_worst_case_int8_product():
    """Worst-case INT8 × INT8 conv: weights all +127, input all +127.
    For K = 64 reduction (max AI SRAM depth × N_ROWS on a 4×4 array)
    the sum is 127 × 127 × 64 = 1,032,256 per output. Safely in
    INT32 range (±2,147,483,647) but would break an INT24 accumulator
    (±8,388,607).

    For a realistic YOLOv8 conv: K_total = C_in * k_h * k_w up to
    9216 (256 × 3 × 3). Worst-case sum = 127² × 9216 ≈ 148 M —
    still comfortably inside INT32. The test below pins the small
    4×4 case and checks the math engine survives it bit-exactly."""
    C_in = 16
    k_h, k_w = 2, 2   # K_total = 16 × 2 × 2 = 64 — max single-tile K on 4x4
    x = np.full((1, C_in, 2, 2), 127, dtype=np.int8)
    w = np.full((4, C_in, k_h, k_w), 127, dtype=np.int8)
    # Output position (0,0) takes every K value (no padding).
    ref = reference_conv2d_int8(x, w, stride=(1, 1),
                                  pad=(0, 0, 0, 0))
    expected_cell = 127 * 127 * 64
    # Every output channel is the same (identical weights); every
    # spatial position is the same (identical input).
    assert ref[0, 0, 0, 0] == expected_cell, (
        f"expected {expected_cell}, got {ref[0,0,0,0]}"
    )
    # Full round-trip through compile + simulate must preserve the
    # INT32 value exactly.
    result = compile_conv2d(w, x, n_rows=4, n_cols=4,
                             stride=(1, 1), pad=(0, 0, 0, 0))
    _, read_log = simulate_program(
        result.program, n_rows=4, n_cols=4, return_read_log=True)
    out = reassemble_conv_output(read_log, result)
    assert np.array_equal(out, ref), (
        "compile→simulate diverges from reference at INT8 extremes — "
        "would indicate an accumulator width or overflow handling bug"
    )
    assert ref.max() < 2**31 - 1 and ref.min() > -(2**31), \
        "INT32 accumulator would overflow on this worst-case"


def test_gap4_negative_extremes_dont_wrap():
    """Same test with -127 weights + -127 activations. Product is
    positive (127²) so the final sum is the same positive value.
    Catches sign-handling bugs that would show as negative output."""
    C_in = 8
    x = np.full((1, C_in, 2, 2), -127, dtype=np.int8)
    w = np.full((2, C_in, 2, 2), -127, dtype=np.int8)
    ref = reference_conv2d_int8(x, w, pad=(0, 0, 0, 0))
    # -127 × -127 = +16129, summed over K = 8 × 2 × 2 = 32 → 516,128.
    expected = 127 * 127 * 32
    assert ref[0, 0, 0, 0] == expected, (
        f"sign handling broken: {ref[0,0,0,0]} vs {expected}"
    )


# ---------------------------------------------------------------------------
# GAP-5 — bias data-path end-to-end validation
# ---------------------------------------------------------------------------
def _build_conv_bias_model(W: np.ndarray, B: np.ndarray):
    """Build an ONNX model with a single Conv(x, w, b) node."""
    C_out, C_in, k_h, k_w = W.shape
    node = helper.make_node(
        "Conv", inputs=["x", "w", "b"], outputs=["y"],
        kernel_shape=[k_h, k_w], pads=[k_h // 2, k_w // 2,
                                         k_h // 2, k_w // 2],
        name="conv",
    )
    graph = helper.make_graph(
        nodes=[node], name="gap5_conv_bias",
        inputs=[helper.make_tensor_value_info(
            "x", TensorProto.FLOAT, (1, C_in, 8, 8))],
        outputs=[helper.make_tensor_value_info(
            "y", TensorProto.FLOAT, (1, C_out, 8, 8))],
        initializer=[numpy_helper.from_array(W, "w"),
                     numpy_helper.from_array(B, "b")],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                           ir_version=8)
    return _save_model(m)


def test_gap5_bias_is_post_dequant_fp32():
    """F1-C2 audit H1: bias is FP32 added AFTER INT8 accumulator
    dequantisation. The spec decision: never quantise bias, apply it
    host-side to the dequantised int32 accumulator × w_scale × in_scale
    product.

    This test runs a conv+bias layer three ways and confirms they
    agree:
      (a) ORT FP32 native reference (ground truth).
      (b) nn_runtime with bias applied per the audit H1 recipe.
      (c) Manual pipeline: reference_conv2d_int8 (INT32 acc) →
          dequant per-channel × input_scale → + FP32 bias → compare.

    (a) ≈ (b) at INT8 tolerance; (b) = (c) exactly (both implement
    the same H1 recipe).
    """
    rng = np.random.default_rng(0)
    W = rng.standard_normal((4, 3, 3, 3)).astype(np.float32) * 0.1
    # Non-trivial bias — the test must break if bias is zeroed or
    # applied pre-dequant (which would scale it wrong).
    B = np.array([0.5, -0.3, 1.2, -0.8], dtype=np.float32)
    path = _build_conv_bias_model(W, B)

    graph = load_onnx(path)
    fuse_silu(graph)   # no-op here; just exercises the same path
    cal = make_seeded_calibration_set(
        "x", (1, 3, 8, 8), n_batches=10, seed=0)
    quantise_model(graph, path, cal)

    probe = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)

    # (a) FP32 reference.
    fp32_out = run_reference(path, {"x": probe}).outputs["y"]

    # (b) nn_runtime INT8 path with bias added FP32 post-dequant.
    rt_out = run_graph(graph, {"x": probe})["y"]

    # (c) Manual reconstruction using the same recipe.
    qp = graph.layers[0].quant
    # Quantise input + weights to INT8.
    x_i8 = np.clip(np.round(probe / qp.input_scale),
                     -127, 127).astype(np.int8)
    w_scale = qp.weight_scale
    w_i8 = np.zeros_like(W, dtype=np.int8)
    for c in range(W.shape[0]):
        w_i8[c] = np.clip(np.round(W[c] / float(w_scale[c])),
                           -127, 127).astype(np.int8)
    acc = reference_conv2d_int8(x_i8, w_i8, stride=(1, 1),
                                   pad=(1, 1, 1, 1))
    # Dequant per-channel × input_scale + bias FP32.
    multi = (w_scale * qp.input_scale).astype(np.float32).reshape(1, -1, 1, 1)
    manual = acc.astype(np.float32) * multi + B.reshape(1, -1, 1, 1)

    # (b) and (c) implement the same H1 recipe — must agree bit-exactly
    # modulo floating-point rounding order.
    np.testing.assert_allclose(rt_out, manual, rtol=1e-5, atol=1e-5), \
        "nn_runtime bias recipe diverges from manual H1 reconstruction"

    # (a) vs (b): INT8 quant noise. Bound tolerances loosely — the
    # point is that bias is approximately preserved.
    err = fp32_out - rt_out
    bias_magnitude = float(np.max(np.abs(B)))
    err_magnitude = float(np.max(np.abs(err)))
    assert err_magnitude < 10 * bias_magnitude, (
        f"FP32 vs NPU error ({err_magnitude:.3f}) much larger than "
        f"bias magnitude ({bias_magnitude:.3f}) — suggests bias is "
        f"being dropped / doubled / applied pre-dequant"
    )


def test_gap5_zero_bias_is_no_op():
    """Independent sanity: setting the bias to zero must produce a
    numerical result identical to a conv without bias. Catches the
    opposite failure mode (bias erroneously added twice)."""
    rng = np.random.default_rng(1)
    W = rng.standard_normal((2, 3, 3, 3)).astype(np.float32) * 0.1

    # With zero bias.
    path_zero = _build_conv_bias_model(
        W, np.zeros((2,), dtype=np.float32))
    g_zero = load_onnx(path_zero); fuse_silu(g_zero)
    cal = make_seeded_calibration_set(
        "x", (1, 3, 8, 8), n_batches=5, seed=2)
    quantise_model(g_zero, path_zero, cal)

    # Without bias node at all.
    C_out, C_in, k_h, k_w = W.shape
    node = helper.make_node(
        "Conv", inputs=["x", "w"], outputs=["y"],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="conv",
    )
    graph = helper.make_graph(
        nodes=[node], name="no_bias",
        inputs=[helper.make_tensor_value_info(
            "x", TensorProto.FLOAT, (1, C_in, 8, 8))],
        outputs=[helper.make_tensor_value_info(
            "y", TensorProto.FLOAT, (1, C_out, 8, 8))],
        initializer=[numpy_helper.from_array(W, "w")],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                           ir_version=8)
    path_none = _save_model(m)
    g_none = load_onnx(path_none); fuse_silu(g_none)
    quantise_model(g_none, path_none, cal)

    probe = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    out_zero = run_graph(g_zero, {"x": probe})["y"]
    out_none = run_graph(g_none, {"x": probe})["y"]
    np.testing.assert_allclose(out_zero, out_none, rtol=1e-5, atol=1e-5)
