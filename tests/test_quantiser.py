"""Unit tests for tools/npu_ref/quantiser.py (F1-C2).

Covers weight quant (per-channel + per-tensor), activation calibration
(running max-abs across batches, input/output scale propagation),
zero-range guards, idempotency, and error paths.

Synthetic ONNX models are built via onnx.helper so each test is
independent and fast.
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
    GRAN_PER_CHANNEL,
    GRAN_PER_TENSOR,
    OP_CONV,
    PRECISION_INT8,
    QuantParams,
)
from tools.npu_ref.onnx_loader import load_onnx
from tools.npu_ref.quantiser import (
    CALIB_MAX_ABS,
    CALIB_PERCENTILE,
    _per_channel_symmetric_scale,
    calibrate_activations,
    fake_quantise_activation,
    fake_quantise_weights,
    make_seeded_calibration_set,
    quantise_model,
    quantise_weights,
)


# ---------------------------------------------------------------------------
# Helpers: build tiny ONNX models + load them
# ---------------------------------------------------------------------------
def _save(model: onnx.ModelProto) -> str:
    fd, path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    onnx.save(model, path)
    return path


def _tiny_conv_model(W: np.ndarray, in_shape=(1, 1, 4, 4),
                      out_shape=(1, 1, 4, 4),
                      with_bias: bool = False) -> str:
    inits = [numpy_helper.from_array(W, "w")]
    node_inputs = ["x", "w"]
    if with_bias:
        B = np.zeros((W.shape[0],), dtype=np.float32)
        inits.append(numpy_helper.from_array(B, "b"))
        node_inputs.append("b")
    node = helper.make_node(
        "Conv", inputs=node_inputs, outputs=["y"],
        kernel_shape=list(W.shape[2:]),
        pads=[W.shape[2] // 2, W.shape[3] // 2] * 2,
        name="conv",
    )
    graph = helper.make_graph(
        nodes=[node], name="t",
        inputs=[helper.make_tensor_value_info(
            "x", TensorProto.FLOAT, in_shape)],
        outputs=[helper.make_tensor_value_info(
            "y", TensorProto.FLOAT, out_shape)],
        initializer=inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    return _save(model)


# ---------------------------------------------------------------------------
# Per-channel scale math
# ---------------------------------------------------------------------------
def test_per_channel_scale_computes_max_abs_per_channel():
    # 4 output channels, each with a different max-abs.
    w = np.zeros((4, 2, 3, 3), dtype=np.float32)
    w[0] = 2.0
    w[1] = 0.5
    w[2] = -1.27
    w[3] = 127.0  # edge: max value
    scale = _per_channel_symmetric_scale(w)
    np.testing.assert_allclose(scale[0], 2.0 / 127, rtol=1e-6)
    np.testing.assert_allclose(scale[1], 0.5 / 127, rtol=1e-6)
    np.testing.assert_allclose(scale[2], 1.27 / 127, rtol=1e-6)
    np.testing.assert_allclose(scale[3], 127.0 / 127, rtol=1e-6)


def test_per_channel_scale_zero_range_guard():
    # An all-zero channel must produce scale=1.0 (no div-by-zero).
    w = np.zeros((2, 1, 1, 1), dtype=np.float32)
    w[1] = 5.0
    scale = _per_channel_symmetric_scale(w)
    assert scale[0] == 1.0
    np.testing.assert_allclose(scale[1], 5.0 / 127, rtol=1e-6)


# ---------------------------------------------------------------------------
# quantise_weights
# ---------------------------------------------------------------------------
def test_quantise_weights_attaches_per_channel_params():
    W = np.array([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]],
                 dtype=np.float32)
    path = _tiny_conv_model(W)
    g = load_onnx(path)
    quantise_weights(g)
    layer = g.layers[0]
    assert layer.op == OP_CONV
    assert isinstance(layer.quant, QuantParams)
    assert layer.quant.precision == PRECISION_INT8
    assert layer.quant.granularity == GRAN_PER_CHANNEL
    assert layer.quant.weight_scale.shape == (4,)
    np.testing.assert_allclose(layer.quant.weight_scale,
                                [1/127, 2/127, 3/127, 4/127], rtol=1e-6)


def test_quantise_weights_per_tensor_mode():
    W = np.array([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]],
                 dtype=np.float32)
    path = _tiny_conv_model(W)
    g = load_onnx(path)
    quantise_weights(g, granularity=GRAN_PER_TENSOR)
    qp = g.layers[0].quant
    assert qp.granularity == GRAN_PER_TENSOR
    assert qp.weight_scale.shape == ()
    # max_abs = 4.0
    np.testing.assert_allclose(float(qp.weight_scale), 4.0 / 127, rtol=1e-6)


def test_quantise_weights_idempotent():
    W = np.ones((2, 1, 1, 1), dtype=np.float32)
    path = _tiny_conv_model(W)
    g = load_onnx(path)
    quantise_weights(g)
    layer = g.layers[0]
    layer.quant.input_scale = 0.123  # simulate post-calibration state
    quantise_weights(g)
    # input_scale must be preserved across re-quant.
    assert layer.quant.input_scale == 0.123


def test_quantise_weights_int4_supported():
    """INT4 was added in F1-B2; scale is max_abs/7."""
    W = np.array([[[[1.0]]], [[[2.0]]], [[[3.5]]], [[[7.0]]]],
                 dtype=np.float32)
    path = _tiny_conv_model(W)
    g = load_onnx(path)
    quantise_weights(g, precision="int4")
    qp = g.layers[0].quant
    assert qp.precision == "int4"
    assert qp.weight_scale.shape == (4,)
    np.testing.assert_allclose(
        qp.weight_scale, [1/7, 2/7, 3.5/7, 7/7], rtol=1e-6)


def test_quantise_weights_rejects_fp_for_now():
    """FP4/FP8 land with F1-A1; should raise a clear NotImplementedError."""
    W = np.ones((1, 1, 1, 1), dtype=np.float32)
    path = _tiny_conv_model(W)
    g = load_onnx(path)
    with pytest.raises(NotImplementedError, match="FP4/FP8"):
        quantise_weights(g, precision="fp8")


# ---------------------------------------------------------------------------
# calibrate_activations
# ---------------------------------------------------------------------------
def test_calibrate_single_batch_populates_scales():
    W = np.eye(1, dtype=np.float32).reshape(1, 1, 1, 1)
    path = _tiny_conv_model(W, in_shape=(1, 1, 4, 4), out_shape=(1, 1, 4, 4))
    g = load_onnx(path)
    quantise_weights(g)
    # Batch with max-abs = 2.5
    x = np.full((1, 1, 4, 4), 2.5, dtype=np.float32)
    calibrate_activations(g, path, [{"x": x}])
    assert "activation_scales" in g.metadata
    assert g.metadata["calibration_batches"] == 1
    # Input scale for "x" ≈ 2.5 / 127
    np.testing.assert_allclose(g.metadata["activation_scales"]["x"],
                                2.5 / 127, rtol=1e-5)
    qp = g.layers[0].quant
    assert qp.input_scale == pytest.approx(2.5 / 127, rel=1e-5)
    # The conv with identity-weight passes the input through, so
    # output scale should equal input scale.
    assert qp.output_scale == pytest.approx(2.5 / 127, rel=1e-3)


def test_calibrate_running_max_across_batches():
    W = np.eye(1, dtype=np.float32).reshape(1, 1, 1, 1)
    path = _tiny_conv_model(W)
    g = load_onnx(path)
    quantise_weights(g)
    batches = [
        {"x": np.full((1, 1, 4, 4), 1.0, dtype=np.float32)},
        {"x": np.full((1, 1, 4, 4), 3.0, dtype=np.float32)},  # largest
        {"x": np.full((1, 1, 4, 4), 2.0, dtype=np.float32)},
    ]
    calibrate_activations(g, path, batches)
    # Max-abs across all three is 3.0
    np.testing.assert_allclose(g.metadata["activation_scales"]["x"],
                                3.0 / 127, rtol=1e-5)
    assert g.metadata["calibration_batches"] == 3


def test_calibrate_empty_batches_raises():
    W = np.ones((1, 1, 1, 1), dtype=np.float32)
    path = _tiny_conv_model(W)
    g = load_onnx(path)
    quantise_weights(g)
    with pytest.raises(ValueError, match="calibration_inputs was empty"):
        calibrate_activations(g, path, [])


def test_calibrate_without_quantise_weights_raises():
    W = np.ones((1, 1, 1, 1), dtype=np.float32)
    path = _tiny_conv_model(W)
    g = load_onnx(path)
    x = np.ones((1, 1, 4, 4), dtype=np.float32)
    with pytest.raises(RuntimeError, match="run quantise_weights"):
        calibrate_activations(g, path, [{"x": x}])


def test_calibration_warns_on_missing_scale():
    """When a weight-bearing layer references a tensor that never
    flows through any calibration batch, the layer's scale defaults
    to 1.0 and the event is recorded in graph.metadata so downstream
    passes can notice."""
    W = np.ones((1, 1, 1, 1), dtype=np.float32)
    path = _tiny_conv_model(W)
    g = load_onnx(path)
    quantise_weights(g)
    # Simulate a layer whose input tensor was never produced by any
    # calibration batch — e.g. a constant-folded tensor or a layer
    # edit that introduced a dangling reference. Mutating L.inputs
    # in memory is safe because ORT runs from the on-disk .onnx.
    g.layers[0].inputs = ["never_seen_tensor"]
    x = np.ones((1, 1, 4, 4), dtype=np.float32)
    calibrate_activations(g, path, [{"x": x}])
    warnings = g.metadata.get("calibration_warnings", [])
    assert any("never_seen_tensor" in w for w in warnings), (
        f"expected a missing-scale warning for 'never_seen_tensor', "
        f"got {warnings}"
    )
    # Layer keeps its default input_scale (1.0) rather than silently
    # using a plausible-but-wrong neighbour's scale.
    assert g.layers[0].quant.input_scale == 1.0


# ---------------------------------------------------------------------------
# quantise_model convenience
# ---------------------------------------------------------------------------
def test_quantise_model_does_both_passes():
    W = np.ones((2, 1, 1, 1), dtype=np.float32)
    path = _tiny_conv_model(W, out_shape=(1, 2, 4, 4))
    g = load_onnx(path)
    x = np.full((1, 1, 4, 4), 1.5, dtype=np.float32)
    quantise_model(g, path, [{"x": x}])
    qp = g.layers[0].quant
    assert qp is not None
    assert qp.weight_scale.shape == (2,)
    assert qp.input_scale > 0
    assert qp.output_scale > 0


# ---------------------------------------------------------------------------
# fake-quant round trip
# ---------------------------------------------------------------------------
def test_fake_quantise_weights_per_channel_zero_error_at_full_scale():
    # With scale chosen as max_abs/127, a tensor whose channel max is
    # exactly representable quantises exactly. We pick values that
    # round cleanly through INT8.
    w = np.zeros((2, 1, 1, 1), dtype=np.float32)
    w[0, 0, 0, 0] = 127.0
    w[1, 0, 0, 0] = 64.0
    scale = _per_channel_symmetric_scale(w)
    round_trip = fake_quantise_weights(w, scale)
    np.testing.assert_allclose(round_trip, w, atol=1e-4)


def test_fake_quantise_weights_introduces_bounded_error():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
    scale = _per_channel_symmetric_scale(w)
    round_trip = fake_quantise_weights(w, scale)
    # Maximum error must be <= scale per channel (one LSB).
    for c in range(w.shape[0]):
        max_err = float(np.abs(round_trip[c] - w[c]).max())
        assert max_err <= float(scale[c]) + 1e-6, (
            f"channel {c}: max_err={max_err}, scale={float(scale[c])}"
        )


def test_fake_quantise_activation_zero_scale():
    a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    assert np.all(fake_quantise_activation(a, 0.0) == 0.0)


# ---------------------------------------------------------------------------
# F1-B2: INT4 extensions
# ---------------------------------------------------------------------------
def test_int4_per_channel_scale_uses_range_7():
    """INT4 symmetric range is ±7; scale = max_abs / 7."""
    w = np.zeros((3, 1, 1, 1), dtype=np.float32)
    w[0, 0, 0, 0] = 7.0
    w[1, 0, 0, 0] = 3.5
    w[2, 0, 0, 0] = 1.0
    scale = _per_channel_symmetric_scale(w, precision="int4")
    np.testing.assert_allclose(scale, [1.0, 0.5, 1.0 / 7], rtol=1e-6)


def test_int4_fake_quant_clips_to_7_grid():
    """Values outside the INT4 symmetric range clip to ±7 * scale."""
    w = np.array([-20.0, -7.0, 0.0, 3.0, 7.0, 20.0], dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)  # per-tensor
    rt = fake_quantise_weights(w, scale, precision="int4")
    # Both ±20.0 must clip to ±7.0 (one-LSB * 7).
    assert rt[0] == -7.0
    assert rt[-1] == 7.0
    # In-range values round to nearest integer.
    np.testing.assert_allclose(rt[1:-1], [-7.0, 0.0, 3.0, 7.0])


def test_int4_fake_quant_activation_grid_step():
    """With scale=0.5 and range ±7, INT4 grid is {-3.5,...,0,...,3.5}."""
    a = np.array([-4.0, -3.6, -3.5, 0.0, 3.5, 3.6, 4.0], dtype=np.float32)
    rt = fake_quantise_activation(a, 0.5, precision="int4")
    # -4.0 and +4.0 clip to ±3.5 (±7 * 0.5).
    assert rt[0] == -3.5
    assert rt[-1] == 3.5
    # 3.6 rounds to 3.5 (nearest grid point 7 * 0.5).
    assert rt[-2] == pytest.approx(3.5)


def test_int4_end_to_end_model_calibrates():
    """quantise_model + INT4 end-to-end on a tiny conv: scales populated,
    weight_scale length matches C_out, fake-quant is bounded by one LSB."""
    W = np.array([[[[2.0]]], [[[4.0]]]], dtype=np.float32)
    path = _tiny_conv_model(W, out_shape=(1, 2, 4, 4))
    g = load_onnx(path)
    x = np.full((1, 1, 4, 4), 1.4, dtype=np.float32)
    quantise_model(g, path, [{"x": x}], precision="int4")
    qp = g.layers[0].quant
    assert qp is not None
    assert qp.precision == "int4"
    assert qp.weight_scale.shape == (2,)
    np.testing.assert_allclose(qp.weight_scale, [2.0 / 7, 4.0 / 7], rtol=1e-6)
    assert qp.input_scale > 0
    # Fake-quant round-trip uses precision from QuantParams.
    fq = fake_quantise_weights(W, qp.weight_scale, precision=qp.precision)
    for c in range(W.shape[0]):
        max_err = float(np.abs(fq[c] - W[c]).max())
        assert max_err <= float(qp.weight_scale[c]) + 1e-6


def test_fp8_raises_until_f1_a1_lands():
    """FP4/FP8 need the F1-A1 RTL datapath. Until then, raise loudly."""
    with pytest.raises(NotImplementedError, match="FP4/FP8"):
        _per_channel_symmetric_scale(
            np.ones((2, 1, 1, 1), dtype=np.float32),
            precision="fp8_e4m3",
        )


def test_make_seeded_calibration_set_deterministic():
    a = make_seeded_calibration_set("x", (1, 3, 4, 4), n_batches=3, seed=42)
    b = make_seeded_calibration_set("x", (1, 3, 4, 4), n_batches=3, seed=42)
    assert len(a) == 3
    for ba, bb in zip(a, b):
        np.testing.assert_array_equal(ba["x"], bb["x"])


# ---------------------------------------------------------------------------
# Percentile calibration (F1-C2 audit H2 remediation)
# ---------------------------------------------------------------------------
def test_percentile_calibration_ignores_outliers():
    """With a tailed distribution (normal bulk + a few outliers),
    percentile calibration must produce a smaller scale than max-abs —
    that's the whole point: more quant resolution for the bulk, at
    the cost of clipping the outliers.

    Uses 1024 samples so the 99%-ile has enough bulk samples to
    robustly ignore 10 injected outliers."""
    W = np.eye(1, dtype=np.float32).reshape(1, 1, 1, 1)
    path = _tiny_conv_model(W, in_shape=(1, 1, 32, 32),
                              out_shape=(1, 1, 32, 32))

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=(1, 1, 32, 32)).astype(np.float32)
    # Inject 10 large outliers (~1% of 1024 samples).
    idx = rng.choice(1024, 10, replace=False)
    x.ravel()[idx] = 50.0

    g_max = load_onnx(path); quantise_weights(g_max)
    calibrate_activations(g_max, path, [{"x": x}],
                           calibration_method=CALIB_MAX_ABS)
    max_scale = g_max.metadata["activation_scales"]["x"]

    g_pct = load_onnx(path); quantise_weights(g_pct)
    calibrate_activations(g_pct, path, [{"x": x}],
                           calibration_method=CALIB_PERCENTILE,
                           percentile=99.0)
    pct_scale = g_pct.metadata["activation_scales"]["x"]

    # Max scale = 50/127 ≈ 0.394. Percentile at 99% ignores the ~1%
    # outliers; 99%-ile of |N(0,1)| is ~2.58 → scale ≈ 0.020.
    assert max_scale > 0.3, f"max-abs scale unexpectedly small: {max_scale}"
    assert pct_scale < 0.05, f"percentile scale should ignore outliers: {pct_scale}"
    assert max_scale > 5 * pct_scale  # clear separation


def test_percentile_calibration_metadata_recorded():
    W = np.ones((1, 1, 1, 1), dtype=np.float32)
    path = _tiny_conv_model(W)
    g = load_onnx(path); quantise_weights(g)
    x = np.full((1, 1, 4, 4), 2.0, dtype=np.float32)
    calibrate_activations(g, path, [{"x": x}],
                           calibration_method=CALIB_PERCENTILE,
                           percentile=99.9)
    assert g.metadata["calibration_method"] == CALIB_PERCENTILE
    assert g.metadata["calibration_percentile"] == 99.9


def test_percentile_calibration_zero_tensor_safe():
    """An all-zero tensor must produce scale=1.0, not 0/zero-division."""
    W = np.ones((1, 1, 1, 1), dtype=np.float32)
    path = _tiny_conv_model(W)
    g = load_onnx(path); quantise_weights(g)
    x = np.zeros((1, 1, 4, 4), dtype=np.float32)
    calibrate_activations(g, path, [{"x": x}],
                           calibration_method=CALIB_PERCENTILE)
    assert g.metadata["activation_scales"]["x"] == 1.0


def test_unknown_calibration_method_raises():
    W = np.ones((1, 1, 1, 1), dtype=np.float32)
    path = _tiny_conv_model(W)
    g = load_onnx(path); quantise_weights(g)
    x = np.ones((1, 1, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="unknown calibration_method"):
        calibrate_activations(g, path, [{"x": x}],
                               calibration_method="banana")
