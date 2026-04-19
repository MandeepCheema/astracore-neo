"""Unit tests for tools/npu_ref/softmax_ref.py and layernorm_ref.py (F1-A4).

These tests validate the Python golden references against numpy
baselines + ONNX semantics. Once the RTL lands in the follow-up
session, the same tests will be reused with the `_lut` paths
enabled to validate the fixed-point mirror.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.npu_ref.layernorm_ref import (
    layernorm_fp32,
    layernorm_lut,
    rmsnorm_fp32,
)
from tools.npu_ref.softmax_ref import softmax_fp32, softmax_lut


# ---------------------------------------------------------------------------
# softmax_fp32
# ---------------------------------------------------------------------------
def test_softmax_sums_to_one_along_axis():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 32)).astype(np.float32)
    y = softmax_fp32(x, axis=-1)
    np.testing.assert_allclose(y.sum(axis=-1), 1.0, atol=1e-6)


def test_softmax_uniform_on_zero_input():
    y = softmax_fp32(np.zeros(8, dtype=np.float32))
    np.testing.assert_allclose(y, 1.0 / 8, atol=1e-7)


def test_softmax_monotone_in_input():
    """Larger x_i must yield larger y_i under softmax."""
    x = np.array([-2.0, 0.0, 1.0, 3.0], dtype=np.float32)
    y = softmax_fp32(x)
    assert np.all(np.diff(y) > 0)


def test_softmax_numerically_stable_on_large_inputs():
    """softmax(x) == softmax(x + C) for any constant C — the max-subtract
    trick protects against exp overflow for x up to ~1e4."""
    x = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
    y = softmax_fp32(x)
    expected = softmax_fp32(np.array([0.0, 1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(y, expected, atol=1e-5)


def test_softmax_multidim_axis():
    """axis=0 normalises down the first dim; each column sums to 1."""
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    y = softmax_fp32(x, axis=0)
    np.testing.assert_allclose(y.sum(axis=0), 1.0, atol=1e-6)


def test_softmax_lut_without_tables_falls_back_to_fp32():
    """Without LUTs supplied, `softmax_lut` matches `softmax_fp32` —
    lets tests exercise the pipeline shape before the RTL session."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(16).astype(np.float32)
    y_lut = softmax_lut(x)
    y_fp32 = softmax_fp32(x)
    np.testing.assert_allclose(y_lut, y_fp32, atol=1e-6)


def test_softmax_lut_with_tables_is_not_yet_implemented():
    """Once RTL LUTs are supplied, the bit-exact mirror is a separate
    implementation path. Until F1-A4 RTL lands, supplying LUTs raises."""
    x = np.zeros(4, dtype=np.float32)
    fake_exp = np.ones(256, dtype=np.uint32)
    fake_recip = np.ones(1024, dtype=np.uint32)
    with pytest.raises(NotImplementedError, match="RTL"):
        softmax_lut(x, exp_lut=fake_exp, recip_lut=fake_recip)


# ---------------------------------------------------------------------------
# layernorm_fp32
# ---------------------------------------------------------------------------
def test_layernorm_mean_is_zero_and_var_is_one():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 128)).astype(np.float32)
    y = layernorm_fp32(x, axis=-1)
    np.testing.assert_allclose(y.mean(axis=-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(y.var(axis=-1), 1.0, atol=1e-3)


def test_layernorm_affine_shifts_predictably():
    """With γ=2, β=3: output mean should be β and var should be γ²."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 64)).astype(np.float32)
    scale = np.full(64, 2.0, dtype=np.float32)
    bias = np.full(64, 3.0, dtype=np.float32)
    y = layernorm_fp32(x, scale=scale, bias=bias, axis=-1)
    np.testing.assert_allclose(y.mean(axis=-1), 3.0, atol=1e-4)
    np.testing.assert_allclose(y.var(axis=-1), 4.0, atol=1e-2)


def test_layernorm_onnx_parity():
    """Match ONNX's LayerNormalization op (opset 17+) semantics:
    epsilon inside the sqrt, per-feature γ/β, reduction axis inclusive."""
    x = np.array([[1.0, 2.0, 3.0, 4.0],
                  [10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    bias = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    y = layernorm_fp32(x, scale=scale, bias=bias, axis=-1, epsilon=1e-5)
    # Expected: (x - mean) / sqrt(var + eps) with mean=2.5, var=1.25 for
    # row 0 and mean=25, var=125 for row 1.
    for row in range(2):
        r = x[row]
        m = r.mean()
        v = r.var()
        expected = (r - m) / np.sqrt(v + 1e-5)
        np.testing.assert_allclose(y[row], expected, rtol=1e-4)


def test_layernorm_lut_falls_back_to_fp32_without_rsqrt_lut():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((1, 32)).astype(np.float32)
    y1 = layernorm_lut(x, axis=-1)
    y2 = layernorm_fp32(x, axis=-1)
    np.testing.assert_allclose(y1, y2, atol=1e-6)


# ---------------------------------------------------------------------------
# rmsnorm_fp32 (LLaMA)
# ---------------------------------------------------------------------------
def test_rmsnorm_preserves_direction():
    """RMSNorm only normalises magnitude — direction (sign pattern) must
    be preserved per-element."""
    x = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32)
    y = rmsnorm_fp32(x)
    assert np.all(np.sign(y) == np.sign(x))


def test_rmsnorm_unit_norm_of_output():
    """With scale=None and all equal elements, RMSNorm output is all 1s."""
    x = np.full((1, 8), 2.5, dtype=np.float32)
    y = rmsnorm_fp32(x, axis=-1)
    np.testing.assert_allclose(y, 1.0, atol=1e-3)


def test_rmsnorm_scale_multiplies_uniformly():
    x = np.ones((1, 4), dtype=np.float32)
    scale = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    y = rmsnorm_fp32(x, scale=scale, axis=-1)
    np.testing.assert_allclose(y, 2.0, atol=1e-3)
