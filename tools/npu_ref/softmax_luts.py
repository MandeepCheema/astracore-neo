"""F1-A4 softmax LUTs + bit-exact RTL mirror.

The RTL (`rtl/npu_softmax/npu_softmax.v`) operates on signed INT32 inputs
with `EXP_SCALE` fractional bits (inputs are interpreted as Q(32-F).F
fixed-point with F=log2(EXP_SCALE)).

Two LUTs:

  exp_lut  [256 entries, Q0.32 unsigned]
    Index idx = clip((m - x) >> 0, 0, 255).  Because the online-softmax
    algorithm guarantees m ≥ x_i for every element after pass 1 completes
    (and also during pass 1 at the moment of emit), (m − x) is always
    non-negative and a pure positive shift amount in units of 1/EXP_SCALE.
    Value at idx = exp(-idx / EXP_SCALE), encoded as
    uint32(round(value * 2**32)) with 1.0 saturating to 0xFFFFFFFF.

  recip_lut [1024 entries, Q0.32 unsigned]
    Index idx = clip(s_raw >> RECIP_SHIFT, 0, 1023) where s_raw is the
    Q(40-F).F internal sum (F=32 so s is Q8.32 over 40 bits; we take
    bits [37:28] so RECIP_SHIFT=28, giving top 10 bits of s in Q6.4
    steps: idx=16 ↔ s=1.0, idx=1023 ↔ s≈63.9).
    Value at idx = 1 / ((idx + 0.5) * 2**-RECIP_FRAC_BITS) in Q0.32,
    saturating to 0xFFFFFFFF for very small s (< recip lut's lowest slot).

The mirror `softmax_rtl_mirror(x_int)` replays pass 1 (online max+sum)
and pass 2 (divide) using the exact fixed-point math the RTL will do,
producing Q0.8 unsigned outputs bit-for-bit.

Design choices worth flagging:
  - One exp LUT; no separate "exp of shifted max" table.  The same LUT
    serves both the per-sample exp(x_i - m) and the online multiplier
    exp(m_old - x_new).
  - Sum accumulator is 40 bits (Q8.32).  Max possible sum is VEC_LEN=64
    entries × 1.0 = 64.0 < 2^7, so 8 integer bits gives headroom.
  - Reciprocal-index saturation: for s < 1.0 (pathological; shouldn't
    happen in real softmax), idx clamps to 16 so the LUT value is ~1.0
    and the output stays bounded.
"""

from __future__ import annotations

import numpy as np

EXP_LUT_DEPTH      = 256
EXP_SCALE          = 16        # 1 LUT step == 1/16 in real-number space
RECIP_LUT_DEPTH    = 1024
SUM_FRAC_BITS      = 32        # Q8.32 internal sum
RECIP_SHIFT        = 28        # top-10 bits of s == bits [37:28]
Q32                = 1 << 32
Q32_MAX            = Q32 - 1   # saturation value = ~1.0 in Q0.32
OUT_FRAC_BITS      = 8         # Q0.8 output


# ---------------------------------------------------------------------------
# LUT generation
# ---------------------------------------------------------------------------
def make_exp_lut() -> np.ndarray:
    """256-entry Q0.32 exp LUT: lut[idx] = exp(-idx / EXP_SCALE)."""
    idx = np.arange(EXP_LUT_DEPTH, dtype=np.float64)
    vals = np.exp(-idx / EXP_SCALE)
    # Saturate at Q0.32 max; clip for safety.
    quant = np.clip(np.round(vals * Q32), 0, Q32_MAX).astype(np.uint64)
    return quant.astype(np.uint32)


def make_recip_lut() -> np.ndarray:
    """1024-entry Q0.32 reciprocal LUT.

    Index idx corresponds to s_raw in [idx << RECIP_SHIFT, (idx+1) << RECIP_SHIFT).
    Use the midpoint (idx + 0.5) << RECIP_SHIFT for the LUT entry so average
    quantisation error is minimised.
    """
    out = np.zeros(RECIP_LUT_DEPTH, dtype=np.uint32)
    for idx in range(RECIP_LUT_DEPTH):
        # Real-number midpoint of this bucket: s = (idx + 0.5) / (1 << (SUM_FRAC_BITS - RECIP_SHIFT))
        # With SUM_FRAC_BITS=32 and RECIP_SHIFT=28, the divider is 1 << 4 = 16,
        # so s_mid = (idx + 0.5) / 16.
        divisor = 1 << (SUM_FRAC_BITS - RECIP_SHIFT)
        s_mid = (idx + 0.5) / divisor
        if s_mid <= 0.0:
            out[idx] = Q32_MAX
        else:
            recip = 1.0 / s_mid
            # Clamp to Q0.32 range [0, ~1.0]. For s_mid < 1 the recip > 1 and
            # saturates; in practice idx < 16 ↔ s < 1 which shouldn't arise
            # but we keep the LUT safe.
            q = int(round(recip * Q32))
            out[idx] = min(max(q, 0), Q32_MAX)
    return out


# ---------------------------------------------------------------------------
# Bit-exact RTL mirror
# ---------------------------------------------------------------------------
def softmax_rtl_mirror(x_int: np.ndarray,
                       *,
                       exp_lut: np.ndarray | None = None,
                       recip_lut: np.ndarray | None = None) -> np.ndarray:
    """Replay the RTL's two-pass fixed-point softmax on a 1-D INT32 vector.

    Args:
        x_int: shape (VEC_LEN,) signed INT32.  Caller controls the interpretation
               via a global scale (e.g. int(round(x_fp * EXP_SCALE)) if the
               softmax input should live on the LUT grid, or any other
               consistent scale since softmax is translation-invariant).
        exp_lut: 256-entry Q0.32 exp LUT; default = make_exp_lut().
        recip_lut: 1024-entry Q0.32 reciprocal LUT; default = make_recip_lut().

    Returns:
        shape (VEC_LEN,) uint8 Q0.8 softmax outputs.
    """
    if exp_lut is None:
        exp_lut = make_exp_lut()
    if recip_lut is None:
        recip_lut = make_recip_lut()
    assert x_int.ndim == 1
    assert exp_lut.shape == (EXP_LUT_DEPTH,)
    assert recip_lut.shape == (RECIP_LUT_DEPTH,)

    vec = x_int.astype(np.int64)  # use Python big-ish ints
    n = vec.shape[0]

    # ------------------- Pass 1 ------------------------------------------------
    # m: signed 32-bit running max. s: Q8.32 unsigned sum of exp(x_i - m).
    m = int(vec[0])
    s = int(exp_lut[0])           # exp(0) = 1.0 in Q0.32 → idx 0
    for i in range(1, n):
        x = int(vec[i])
        if x > m:
            # shift = clip(x - m, 0, 255)
            shift = max(0, min(x - m, EXP_LUT_DEPTH - 1))
            exp_shift = int(exp_lut[shift])    # exp(-(x-m)/EXP_SCALE) in Q0.32
            # s <- s * exp_shift + 1.0 (in Q0.32 units)
            # s is Q8.32, exp_shift is Q0.32 → product is Q8.64, shift right 32.
            s = (s * exp_shift) >> SUM_FRAC_BITS
            s += int(exp_lut[0])               # += 1.0 in Q0.32
            m = x
        else:
            shift = max(0, min(m - x, EXP_LUT_DEPTH - 1))
            s += int(exp_lut[shift])

    # ------------------- Reciprocal --------------------------------------------
    # idx = clip((s + half_bucket) >> RECIP_SHIFT, 0, 1023).
    # Round-to-nearest into bucket so that s at a bucket's lower edge maps
    # to the bucket whose midpoint is closest — this is needed because the
    # LUT value for bucket k is 1/((k+0.5)*step), so nearest-bucket indexing
    # cancels the systematic under-estimate that pure truncation produces.
    half_bucket = 1 << (RECIP_SHIFT - 1)
    r_idx = (s + half_bucket) >> RECIP_SHIFT
    if r_idx >= RECIP_LUT_DEPTH:
        r_idx = RECIP_LUT_DEPTH - 1
    inv_s = int(recip_lut[r_idx])            # Q0.32

    # ------------------- Pass 2 ------------------------------------------------
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        x = int(vec[i])
        shift = max(0, min(m - x, EXP_LUT_DEPTH - 1))
        exp_val = int(exp_lut[shift])          # Q0.32
        # product = exp_val * inv_s, Q0.64.  Output = top 8 fractional bits
        # rounded to nearest-even: add half-LSB before shifting.
        product = exp_val * inv_s
        shift = 2 * SUM_FRAC_BITS - OUT_FRAC_BITS
        y = (product + (1 << (shift - 1))) >> shift
        if y > 0xFF:
            y = 0xFF
        out[i] = y
    return out


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    exp_lut = make_exp_lut()
    recip_lut = make_recip_lut()
    print(f"exp_lut[0]   = 0x{exp_lut[0]:08x}  (~ 1.0 in Q0.32)")
    print(f"exp_lut[16]  = 0x{exp_lut[16]:08x}  (~ exp(-1) = 0.368)")
    print(f"exp_lut[255] = 0x{exp_lut[255]:08x}  (~ exp(-15.94))")
    print(f"recip_lut[16]  = 0x{recip_lut[16]:08x}  (~ 1/(16.5/16) ~= 0.97)")
    print(f"recip_lut[160] = 0x{recip_lut[160]:08x}  (~ 1/10)")

    # Sanity: uniform input → uniform output.
    x = np.zeros(64, dtype=np.int32)
    y = softmax_rtl_mirror(x)
    expected = round(1.0 / 64 * 256)
    assert abs(int(y[0]) - expected) <= 2, f"uniform: y={y[0]} expected≈{expected}"
    # Sum of Q0.8 outputs should be close to 256.
    assert abs(int(y.sum()) - 256) <= 8, f"sum = {y.sum()} (want ~256)"
    print(f"uniform-64 softmax: y[0]={y[0]}  sum={y.sum()}  PASS")

    # Compare against FP32 oracle on random vectors.
    from softmax_ref import softmax_fp32
    rng = np.random.default_rng(0)
    for trial in range(20):
        x_fp = rng.standard_normal(64).astype(np.float32) * 2.0
        x_int = np.round(x_fp * EXP_SCALE).astype(np.int32)
        y_rtl = softmax_rtl_mirror(x_int).astype(np.float32) / 256.0
        y_fp  = softmax_fp32(x_fp)
        noise = y_rtl - y_fp
        sig_pow = float(np.mean(y_fp ** 2))
        noise_pow = float(np.mean(noise ** 2))
        snr = 10.0 * np.log10(sig_pow / max(noise_pow, 1e-30))
        if trial < 3:
            print(f"trial {trial}: SNR = {snr:.2f} dB, sum(y_rtl)={y_rtl.sum():.3f}")
        assert snr >= 25.0, f"trial {trial} SNR {snr:.2f} dB below 25 dB"
    print("softmax_rtl_mirror vs FP32 oracle: 20 trials PASS (SNR >= 25 dB)")
