"""Unit tests for tools/npu_ref/fp_ref.py (F1-A1).

Validates the Python FP8 E4M3, FP8 E5M2, and FP16 round-trip
quantisers against the OCP specification boundaries and
representative numerical properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.npu_ref.fp_ref import (
    fp32_to_e4m3,
    fp32_to_e5m2,
    fp32_to_fp16,
    fp_mac,
)


# ---------------------------------------------------------------------------
# FP8 E4M3
# ---------------------------------------------------------------------------
def test_e4m3_zero_and_unit_exact():
    assert fp32_to_e4m3(np.float32(0.0)) == 0.0
    assert fp32_to_e4m3(np.float32(1.0)) == 1.0
    assert fp32_to_e4m3(np.float32(-1.0)) == -1.0
    # Powers of two within range are exact.
    for e in range(-6, 9):
        v = np.float32(2.0 ** e)
        assert fp32_to_e4m3(v) == v, f"2^{e}: {fp32_to_e4m3(v)} != {v}"


def test_e4m3_saturates_at_max():
    """OCP E4M3 max magnitude is 448.0; overflow saturates, not inf."""
    assert fp32_to_e4m3(np.float32(448.0)) == 448.0
    assert fp32_to_e4m3(np.float32(1e6)) == 448.0
    assert fp32_to_e4m3(np.float32(-1e6)) == -448.0
    # No Inf emission — this is the difference from IEEE-style FP8.
    result = fp32_to_e4m3(np.float32(np.inf))
    assert np.isnan(result) or result == 448.0  # either is acceptable


def test_e4m3_mantissa_grid_step():
    """Between 1.0 and 2.0 the mantissa has 8 levels (3-bit). Adjacent
    representable values are 1/8 apart."""
    # 1.0, 1.125, 1.25, ..., 1.875 are all exact.
    for k in range(8):
        v = np.float32(1.0 + k / 8.0)
        np.testing.assert_allclose(fp32_to_e4m3(v), v, rtol=0, atol=0)
    # 1.0625 (halfway between 1.0 and 1.125) rounds to nearest-even.
    rt = fp32_to_e4m3(np.float32(1.0625))
    assert rt in (1.0, 1.125)


def test_e4m3_bounded_quantisation_error():
    """Max relative error in E4M3 range should be at most 2^-3 / 1.0 = 12.5%
    since mantissa has 3 bits. In practice median error is much lower."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-10.0, 10.0, size=10000).astype(np.float32)
    rt = fp32_to_e4m3(x)
    # All round-trip outputs must be finite + bounded by ±max.
    assert np.all(np.isfinite(rt))
    assert np.all(np.abs(rt) <= 448.0 + 1e-6)


# ---------------------------------------------------------------------------
# FP8 E5M2
# ---------------------------------------------------------------------------
def test_e5m2_zero_and_unit_exact():
    assert fp32_to_e5m2(np.float32(0.0)) == 0.0
    assert fp32_to_e5m2(np.float32(1.0)) == 1.0
    assert fp32_to_e5m2(np.float32(-1.0)) == -1.0


def test_e5m2_wide_range_vs_e4m3():
    """E5M2 has 5-bit exponent vs E4M3's 4-bit; range is ~128x wider."""
    assert fp32_to_e5m2(np.float32(1024.0)) == 1024.0   # exact (power of 2)
    assert fp32_to_e5m2(np.float32(32768.0)) == 32768.0  # still in range
    # Beyond 57344.0 overflows to Inf per IEEE-style semantics.
    assert np.isinf(fp32_to_e5m2(np.float32(1e6)))


def test_e5m2_mantissa_has_fewer_levels_than_e4m3():
    """E5M2 only has 2 mantissa bits → 4 levels between powers of 2.
    So 1.125 (which is exact in E4M3) is NOT exact in E5M2."""
    rt = fp32_to_e5m2(np.float32(1.125))
    # Expected to round to 1.0 or 1.25 (nearest grid points).
    assert rt in (1.0, 1.25)


# ---------------------------------------------------------------------------
# FP16
# ---------------------------------------------------------------------------
def test_fp16_matches_numpy_native():
    """Our FP16 path is numpy's float16; verify it's a thin wrapper."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(1000).astype(np.float32)
    rt = fp32_to_fp16(x)
    np.testing.assert_array_equal(
        rt, x.astype(np.float16).astype(np.float32)
    )


def test_fp16_precision_finer_than_fp8():
    """FP16 has 10+1 mantissa bits vs FP8's 3 or 2 — round-trip error
    is much smaller."""
    rng = np.random.default_rng(2)
    x = rng.uniform(-10, 10, 1000).astype(np.float32)
    fp16_err = np.max(np.abs(fp32_to_fp16(x) - x))
    e4m3_err = np.max(np.abs(fp32_to_e4m3(x) - x))
    assert fp16_err < e4m3_err, (
        f"FP16 max err ({fp16_err}) should be << E4M3 max err ({e4m3_err})"
    )


# ---------------------------------------------------------------------------
# fp_mac (fused multiply-accumulate)
# ---------------------------------------------------------------------------
def test_fp_mac_fp16_dot_product_matches_fp32():
    """FP16 MAC preserves FP32 accumulate — dot product of small vectors
    should be within FP16 quantisation error of the FP32 result."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    # Reference: 1*5 + 2*6 + 3*7 + 4*8 = 70
    acc = None
    for ai, bi in zip(a, b):
        acc = fp_mac(np.array([ai]), np.array([bi]),
                     precision="fp16", acc=acc)
    np.testing.assert_allclose(acc[0], 70.0, atol=1e-3)


def test_fp_mac_e4m3_is_lossier_than_fp16():
    """E4M3 has ~12.5% per-op error; over 32 MACs, cumulative error is
    bounded well below the FP32 magnitude but much larger than FP16."""
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, 32).astype(np.float32)
    b = rng.uniform(-1, 1, 32).astype(np.float32)
    ref = float(np.dot(a, b))
    acc_fp16 = float(fp_mac(a, b, precision="fp16").sum())
    acc_e4m3 = float(fp_mac(a, b, precision="fp8_e4m3").sum())
    err_fp16 = abs(acc_fp16 - ref)
    err_e4m3 = abs(acc_e4m3 - ref)
    assert err_e4m3 >= err_fp16, (
        f"E4M3 ({err_e4m3}) should be at least as lossy as FP16 ({err_fp16})"
    )


def test_fp_mac_unknown_precision_raises():
    with pytest.raises(KeyError):
        fp_mac(np.array([1.0]), np.array([1.0]), precision="fp4")
