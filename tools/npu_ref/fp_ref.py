"""Python bit-accurate references for the F1-A1 FP datapath.

Covers the three floating-point modes landing in F1-A1:

  - FP8 E4M3 (OCP spec — sign / 4-bit exponent / 3-bit mantissa)
  - FP8 E5M2 (OCP spec — sign / 5-bit exponent / 2-bit mantissa)
  - FP16    (IEEE-754 half, numpy-native as `float16`)

The MAC datapath is FP8/FP16 multiply → FP32 accumulate. These
functions quantise a FP32 tensor to the chosen FP format and back
(lossy round-trip), and perform the MAC fused-multiply-add step
exactly the way the RTL will.

The FP8 encodings follow the OCP MX-FP8 specification:

  E4M3 : bias=7, max=448.0, min_normal=2^-6≈0.0156
         NaN at S.1111.111 ; no Inf (saturates to max)
  E5M2 : bias=15, max=57344.0, min_normal=2^-14
         IEEE-style NaN / Inf behaviour

The Python encodings below mirror the RTL's planned bit extraction.
They are *not* the same as `torch.float8_e4m3fn` (PyTorch) or
`ml_dtypes.float8_e4m3` (JAX) — we implement them directly to avoid
adding either dependency.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# FP8 E4M3 (OCP)
# ---------------------------------------------------------------------------
_E4M3_MAX = 448.0
_E4M3_MIN_NORMAL = 2.0 ** -6   # smallest normal
_E4M3_MIN_SUBNORMAL = 2.0 ** -9  # smallest subnormal (eps * min_normal)


def fp32_to_e4m3(x: np.ndarray) -> np.ndarray:
    """Quantise FP32 → FP8 E4M3 → FP32 (fake-quant round trip).

    Saturating semantics (matches the OCP MX spec used by NVIDIA H100):
    values with |x| > max saturate to ±max; NaN → NaN; no Inf encoding.
    """
    x = np.asarray(x, dtype=np.float32)
    signs = np.sign(x)
    abs_x = np.abs(x)

    # NaN passthrough
    nan_mask = np.isnan(x)

    # Saturate inputs above max. Technically +Inf encodes as S.1111.111,
    # but per OCP spec we saturate finite inputs to ±max and only encode
    # NaN on the all-ones mantissa slot.
    abs_x = np.minimum(abs_x, _E4M3_MAX)

    # Near-zero → exact 0 (no sub-normal ambiguity)
    zero_mask = abs_x < _E4M3_MIN_SUBNORMAL / 2.0

    # Convert to exponent + 3-bit mantissa grid.
    # log2(x) = e + m/8 for m in [0,8), exponent e in [-6, 8].
    with np.errstate(divide="ignore", invalid="ignore"):
        exponent_f = np.floor(np.log2(np.where(abs_x > 0, abs_x, 1.0)))
    exponent_f = np.clip(exponent_f, -6, 8).astype(np.int32)
    mantissa_f = abs_x / (2.0 ** exponent_f.astype(np.float32))  # in [1,2)
    # Round 1.mmm (3 fractional bits) to nearest-even.
    mantissa_q = np.round((mantissa_f - 1.0) * 8.0) / 8.0
    mantissa_q = np.clip(mantissa_q, 0.0, 7.0 / 8.0)

    value = (1.0 + mantissa_q) * (2.0 ** exponent_f.astype(np.float32))
    result = (signs * value).astype(np.float32)

    result = np.where(zero_mask, 0.0, result)
    result = np.where(nan_mask, np.float32(np.nan), result)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# FP8 E5M2 (OCP)
# ---------------------------------------------------------------------------
_E5M2_MAX = 57344.0
_E5M2_MIN_NORMAL = 2.0 ** -14
_E5M2_MIN_SUBNORMAL = 2.0 ** -16


def fp32_to_e5m2(x: np.ndarray) -> np.ndarray:
    """Quantise FP32 → FP8 E5M2 → FP32 (fake-quant round trip).

    IEEE-style semantics (matches OCP): supports ±Inf, NaN; on
    overflow returns ±Inf.
    """
    x = np.asarray(x, dtype=np.float32)
    signs = np.sign(x)
    abs_x = np.abs(x)

    nan_mask = np.isnan(x)
    inf_mask = np.isinf(x) | (abs_x > _E5M2_MAX)

    # Near-zero flushed to zero
    zero_mask = abs_x < _E5M2_MIN_SUBNORMAL / 2.0
    abs_x = np.minimum(abs_x, _E5M2_MAX)

    with np.errstate(divide="ignore", invalid="ignore"):
        exponent_f = np.floor(np.log2(np.where(abs_x > 0, abs_x, 1.0)))
    exponent_f = np.clip(exponent_f, -14, 15).astype(np.int32)
    mantissa_f = abs_x / (2.0 ** exponent_f.astype(np.float32))
    # 2-bit mantissa → 4 levels in [1, 2): 1.00, 1.25, 1.50, 1.75
    mantissa_q = np.round((mantissa_f - 1.0) * 4.0) / 4.0
    mantissa_q = np.clip(mantissa_q, 0.0, 3.0 / 4.0)

    value = (1.0 + mantissa_q) * (2.0 ** exponent_f.astype(np.float32))
    result = (signs * value).astype(np.float32)

    result = np.where(zero_mask, 0.0, result)
    result = np.where(inf_mask, signs * np.float32(np.inf), result)
    result = np.where(nan_mask, np.float32(np.nan), result)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# FP16 (IEEE-754 half)
# ---------------------------------------------------------------------------
def fp32_to_fp16(x: np.ndarray) -> np.ndarray:
    """FP32 → FP16 → FP32 round trip using numpy's native float16."""
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


# ---------------------------------------------------------------------------
# FP MAC (fused multiply-accumulate in FP32 domain)
# ---------------------------------------------------------------------------
def fp_mac(a: np.ndarray, b: np.ndarray, *, precision: str,
           acc: np.ndarray = None) -> np.ndarray:
    """One MAC step: acc += quant(a) * quant(b), accumulated in FP32.

    Mirrors the RTL's PE behaviour: FP8/FP16 inputs go through the
    quantiser *before* the multiply, the product is exact in FP32,
    and the accumulator stays in FP32. Matches what F1-A1 PE does per
    systolic tick.

    Args:
        a, b: FP32 input arrays (broadcastable).
        precision: "fp8_e4m3" / "fp8_e5m2" / "fp16".
        acc: FP32 accumulator. If None, starts at 0.

    Returns:
        Updated accumulator (FP32).
    """
    qa = _QUANTISERS[precision](a)
    qb = _QUANTISERS[precision](b)
    prod = qa * qb
    if acc is None:
        return prod.astype(np.float32)
    return (acc + prod).astype(np.float32)


_QUANTISERS = {
    "fp8_e4m3": fp32_to_e4m3,
    "fp8_e5m2": fp32_to_e5m2,
    "fp16":     fp32_to_fp16,
}


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # E4M3: 1.0 round-trips exactly (mantissa 000, exp 0).
    assert fp32_to_e4m3(np.float32(1.0)) == 1.0
    # E4M3: saturates at 448.0
    assert fp32_to_e4m3(np.float32(1e6)) == _E4M3_MAX
    assert fp32_to_e4m3(np.float32(-1e6)) == -_E4M3_MAX
    # E5M2: much wider range
    assert fp32_to_e5m2(np.float32(1024.0)) == 1024.0  # power of 2 exact
    # FP16 round trip on exact values
    assert fp32_to_fp16(np.float32(0.5)) == 0.5
    # MAC: simple dot product
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    dp = fp_mac(a, b, precision="fp16").sum()
    assert np.isclose(dp, 32.0, atol=1e-3)
    print("fp_ref self-check: PASS")
