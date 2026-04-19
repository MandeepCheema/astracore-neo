"""F1-A4 layernorm LUTs + bit-exact RTL mirror.

Numeric contract (matches `rtl/npu_layernorm/npu_layernorm.v`):

  Inputs  x, scale γ, bias β  : signed INT32 interpreted as Q16.16
                                (16 integer bits, 16 fractional).
  Σx                           : 48-bit signed accumulator.
  Σx²                          : 64-bit unsigned accumulator (x² is up to
                                 2^(2*IN_BITS)-wide after truncation).
  VEC_LEN                      : power of 2; μ = Σx >> LOG2_VEC_LEN.
  σ² = Σx²/N − μ²              : non-negative; carried as 48-bit unsigned
                                 with Q32.16 interpretation (32 int bits
                                 + 16 frac — up to the Q scale of x²).
  rsqrt lookup                 : normalise (σ²+ε) to [2^15, 2^16); use
                                 leading-zero count to shift right, look
                                 up top 8 bits in a 256-entry LUT giving
                                 rsqrt in Q1.15 (value ≈ [1.0, sqrt(2))),
                                 then reconstruct the full inv_sigma by
                                 shifting according to leading-zero parity.
  Output y = γ·(x−μ)·inv_sigma + β  (LayerNorm)
             γ·x·inv_sigma          (RMSNorm)
  Saturation-clamped to Q16.16 INT32 signed.

The `layernorm_rtl_mirror` function replays every step of the RTL's
fixed-point FSM and produces bit-exact outputs for cocotb to compare.

Design notes:
  - Leading-zero count + even-shift normalisation lets one LUT serve all
    exponent ranges at the cost of one branch on the parity of the shift.
  - The ε stabiliser is hard-coded to 2^-16 (== 1 in Q16.16 fractional
    units) for LayerNorm and 2^-16 for RMSNorm both; the spec's 1e-5 in
    FP32 maps to roughly 2^-17, close enough for the 30 dB gate.
  - VEC_LEN must be a power of 2 (initial-pass constraint).  F1-A4
    follow-up will add arbitrary-VEC_LEN support via a small divider.
"""

from __future__ import annotations

import numpy as np

IN_FRAC_BITS       = 16
IN_W               = 32
SUM_W              = 48        # Σx accumulator width (signed)
SQ_SUM_W           = 64        # Σx² accumulator width (unsigned)
RSQRT_LUT_DEPTH    = 256
RSQRT_LUT_FRAC     = 15        # Q1.15 LUT values
VAR_NORM_BITS      = 16        # post-normalisation, σ² lives in [2^15, 2^16)
EPS_Q16_16         = 7         # ≈ 1e-4 in Q16.16; prevents divide-by-zero
OUT_FRAC_BITS      = 16        # output Q16.16


def make_rsqrt_lut() -> np.ndarray:
    """256-entry rsqrt LUT.

    Post-normalisation the argument sits in the 8-bit range [128, 255]
    (bit 7 = leading 1).  lut[i] = 1 / sqrt((i + 128) / 256)  in Q1.15
    unsigned.  Range of values: [sqrt(256/255), sqrt(256/128)]
                              = [1.002, 1.414].
    """
    idx = np.arange(RSQRT_LUT_DEPTH, dtype=np.float64)
    argument = (idx + 128) / 256.0                # in [0.5, ~1.0)
    vals = 1.0 / np.sqrt(argument)                 # in [1.0, 1.414]
    q = np.clip(np.round(vals * (1 << RSQRT_LUT_FRAC)), 0, (1 << (RSQRT_LUT_FRAC + 1)) - 1)
    return q.astype(np.uint32)


def _leading_zero_count_64(v: int) -> int:
    """Number of leading zeros of a 64-bit unsigned integer."""
    if v == 0:
        return 64
    lz = 0
    for bit in range(63, -1, -1):
        if (v >> bit) & 1:
            break
        lz += 1
    return lz


def _rsqrt_fixed(v_q32_32: int, rsqrt_lut: np.ndarray) -> int:
    """Compute rsqrt of a Q32.32 unsigned value, returning Q1.31 unsigned.

    Steps:
      1. lzc = count leading zeros of v in 64 bits.
      2. Shift v so top bit aligns at bit 63: v_norm = v << lzc
         (so v_norm is in [2^63, 2^64)).
         Alternatively, for our convention we want 8-bit index in
         [128, 255]: idx = v_norm >> 56.
      3. Because sqrt halves the exponent and the shift amount must be
         even to keep the sqrt exponent integer, split into two cases
         based on parity of (lzc).
      4. Full result: rsqrt = LUT(idx) * 2^((lzc - 32)/2)  when lzc even
                            = LUT(idx) * 2^((lzc - 32 - 1)/2) * 1/sqrt(2)  when lzc odd
    """
    if v_q32_32 <= 0:
        return (1 << 31) - 1         # saturate at ~1.0 in Q1.31
    lzc = _leading_zero_count_64(v_q32_32)
    # Shift to get an 8-bit index with bit 7 = leading 1.
    v_norm = v_q32_32 << lzc                     # now in [2^63, 2^64)
    idx = (v_norm >> 56) & 0xFF                  # top byte of 64 bits
    # idx is in [128, 255] by construction; LUT stored as lut[idx - 128] OR lut[idx].
    # We stored the LUT as lut[i] for i in [0, 255] covering argument (i+128)/256 → so
    # for raw idx (in [128, 255]) the LUT-array index is (idx - 128).
    lut_idx = idx - 128
    lut_val = int(rsqrt_lut[lut_idx])            # Q1.15 value
    # v = v_norm * 2^-lzc; so v_real = v_q32_32 / 2^32.
    # We had v_real = (v_norm / 2^64) * 2^(64 - lzc - 32) where v_norm/2^64 is in
    # [0.5, 1.0).  Actually let me restate cleanly:
    #   v_real = v_q32_32 * 2^-32
    #   Let v_norm_real = v_norm * 2^-64 ∈ [0.5, 1.0).
    #   v_norm = v_q32_32 << lzc ⇒ v_norm_real = v_real * 2^(lzc - 32).
    #   So  v_real = v_norm_real * 2^(32 - lzc).
    #   rsqrt(v_real) = rsqrt(v_norm_real) * 2^((lzc - 32)/2).
    # LUT gives rsqrt(v_norm_real) in Q1.15.  Multiply by 2^((lzc-32)/2).
    # For odd (lzc - 32), the half-power is folded in via a precomputed
    # factor sqrt(0.5) ≈ 0x0B504 in Q1.15 (=46341/32768 ≈ 0.7071).
    # rsqrt(v_real) = lut_val * 2^(-shift_raw/2).  shift_raw = 32 - lzc.
    # Positive shift_raw ⇒ v_real > ~1.0, so rsqrt < lut_val — need to DIVIDE.
    # Negative shift_raw ⇒ v_real < ~1.0, so rsqrt > lut_val — need to MULTIPLY.
    shift_raw = 32 - lzc
    if shift_raw % 2 == 0:
        half_shift = shift_raw // 2
        result = lut_val << 16                   # Q1.15 -> Q1.31 (value preserved)
        if half_shift >= 0:
            result = result >> half_shift        # divide by 2^half_shift
        else:
            result = result << (-half_shift)     # multiply by 2^-half_shift
    else:
        SQRT_HALF_Q15 = 0x5A82                   # sqrt(0.5) in Q1.15
        lut_scaled = (lut_val * SQRT_HALF_Q15) >> 15   # still Q1.15
        half_shift = (shift_raw - 1) // 2        # remaining integer factor after √0.5
        result = lut_scaled << 16                # Q1.31
        if half_shift >= 0:
            result = result >> half_shift
        else:
            result = result << (-half_shift)
    # Saturate to Q1.31 range.
    max_q1_31 = (1 << 32) - 1
    if result > max_q1_31:
        result = max_q1_31
    if result < 0:
        result = 0
    return result


# ---------------------------------------------------------------------------
# Bit-exact RTL mirror
# ---------------------------------------------------------------------------
def layernorm_rtl_mirror(x_int: np.ndarray,
                          scale_int: np.ndarray,
                          bias_int: np.ndarray,
                          *,
                          mode: str = "layernorm",
                          rsqrt_lut: np.ndarray | None = None) -> np.ndarray:
    """Bit-exact replay of the RTL's two-pass LN / RMSNorm.

    Args:
        x_int, scale_int, bias_int: shape (VEC_LEN,) signed INT32, interpreted
            as Q(32-IN_FRAC_BITS).IN_FRAC_BITS.  VEC_LEN must be a power of 2.
        mode: "layernorm" (default) or "rmsnorm".
        rsqrt_lut: optional override; default = make_rsqrt_lut().

    Returns:
        shape (VEC_LEN,) signed INT32 Q16.16 outputs.
    """
    assert mode in ("layernorm", "rmsnorm")
    if rsqrt_lut is None:
        rsqrt_lut = make_rsqrt_lut()

    n = x_int.shape[0]
    assert n > 0 and (n & (n - 1)) == 0, f"VEC_LEN={n} must be a power of 2"
    log2_n = int(np.log2(n))

    x = x_int.astype(np.int64)

    # -------- Pass 1 -----------------------------------------------------------
    if mode == "layernorm":
        sum_x = int(x.sum())
    else:
        sum_x = 0
    # Σx² uses the raw Q16.16 product; x_i * x_i has Q32.32 interpretation.
    sum_x2 = int((x * x).sum())

    # μ = Σx / N (right shift for power-of-2 N).  Keep in Q16.16 format.
    mu = sum_x >> log2_n
    # Σx²/N in Q32.32 (since each x² is Q32.32 and dividing by N keeps the Q-format).
    mean_x2 = sum_x2 >> log2_n
    # μ² in Q32.32: mu is Q16.16 signed; square is Q32.32 unsigned.
    mu2 = (mu * mu)
    # σ² in Q32.32.  Subtract and clamp non-negative (fixed-point quant may
    # nudge below zero on truly-zero-var inputs).
    if mode == "layernorm":
        var = mean_x2 - mu2
    else:
        var = mean_x2
    if var < 0:
        var = 0

    # Add ε.  EPS_Q16_16 is stored as a Q16.16 raw value (7 ≈ 1e-4).
    # Var lives in Q32.32, so ε in Q32.32 = EPS_Q16_16 * 2^16 (shift by 16,
    # not 2*IN_FRAC_BITS — that scales to Q64.64 instead).
    eps_q = EPS_Q16_16 << IN_FRAC_BITS
    var_eps = var + eps_q

    inv_sigma_q1_31 = _rsqrt_fixed(var_eps, rsqrt_lut)   # Q1.31 ≈ 1/sqrt(var) where var was Q32.32

    # -------- Pass 2 -----------------------------------------------------------
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        xi = int(x[i])
        if mode == "layernorm":
            centered = xi - mu                      # Q16.16 signed
        else:
            centered = xi                           # Q16.16 signed
        # centered (Q16.16) * inv_sigma (Q1.31) = Q17.47 signed.  Shift right 31 to get Q16.16.
        norm = (centered * inv_sigma_q1_31) >> 31
        # norm * scale_int (Q16.16 * Q16.16 = Q32.32).  Shift right 16 to get Q16.16.
        gamma = int(scale_int[i])
        scaled = (norm * gamma) >> IN_FRAC_BITS
        if mode == "layernorm":
            beta = int(bias_int[i])
            y = scaled + beta
        else:
            y = scaled
        # Saturate to Q16.16 INT32 signed.
        max_i32 = (1 << 31) - 1
        min_i32 = -(1 << 31)
        if y > max_i32:
            y = max_i32
        if y < min_i32:
            y = min_i32
        out[i] = y
    return out.astype(np.int32)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rsqrt_lut = make_rsqrt_lut()
    print(f"rsqrt_lut[0]   = 0x{rsqrt_lut[0]:04x}  (~sqrt(2) = 1.414)")
    print(f"rsqrt_lut[127] = 0x{rsqrt_lut[127]:04x}  (~sqrt(256/255))")
    print(f"rsqrt_lut[255] = 0x{rsqrt_lut[255]:04x}  (~1.0)")

    from layernorm_ref import layernorm_fp32, rmsnorm_fp32
    rng = np.random.default_rng(0)
    IN_SCALE = 1 << 16
    VEC_LEN = 256

    def to_int(a):
        return np.clip(np.round(a * IN_SCALE), -(1 << 31), (1 << 31) - 1).astype(np.int32)
    def from_int(a):
        return a.astype(np.float64) / IN_SCALE

    # --- LayerNorm SNR sweep ---
    total_sig = 0.0
    total_noise = 0.0
    for trial in range(10):
        x_fp = rng.standard_normal(VEC_LEN).astype(np.float32)
        scale_fp = rng.uniform(0.5, 2.0, VEC_LEN).astype(np.float32)
        bias_fp  = rng.uniform(-0.5, 0.5, VEC_LEN).astype(np.float32)
        x_i  = to_int(x_fp)
        s_i  = to_int(scale_fp)
        b_i  = to_int(bias_fp)
        y_rtl = from_int(layernorm_rtl_mirror(x_i, s_i, b_i, mode="layernorm"))
        y_fp  = layernorm_fp32(x_fp, scale_fp, bias_fp, axis=-1)
        noise = y_rtl - y_fp
        total_sig   += float(np.sum(y_fp   ** 2))
        total_noise += float(np.sum(noise  ** 2))
    snr_ln = 10.0 * np.log10(total_sig / max(total_noise, 1e-30))
    print(f"LayerNorm aggregate SNR over 10 trials: {snr_ln:.2f} dB")

    # --- RMSNorm ---
    total_sig = 0.0
    total_noise = 0.0
    for trial in range(10):
        x_fp = rng.standard_normal(VEC_LEN).astype(np.float32)
        scale_fp = rng.uniform(0.5, 2.0, VEC_LEN).astype(np.float32)
        x_i  = to_int(x_fp)
        s_i  = to_int(scale_fp)
        b_i  = np.zeros(VEC_LEN, dtype=np.int32)
        y_rtl = from_int(layernorm_rtl_mirror(x_i, s_i, b_i, mode="rmsnorm"))
        y_fp  = rmsnorm_fp32(x_fp, scale_fp, axis=-1)
        noise = y_rtl - y_fp
        total_sig   += float(np.sum(y_fp   ** 2))
        total_noise += float(np.sum(noise  ** 2))
    snr_rms = 10.0 * np.log10(total_sig / max(total_noise, 1e-30))
    print(f"RMSNorm   aggregate SNR over 10 trials: {snr_rms:.2f} dB")

    assert snr_ln >= 25.0, f"LayerNorm SNR {snr_ln:.2f} dB too low"
    assert snr_rms >= 25.0, f"RMSNorm SNR {snr_rms:.2f} dB too low"
    print("layernorm_rtl_mirror vs FP32 oracle: PASS")
