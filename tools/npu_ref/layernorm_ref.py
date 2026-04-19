"""Python golden reference for rtl/npu_layernorm/npu_layernorm.v (F1-A4).

Fused two-pass layernorm with Welford-style accumulation:

    pass 1:  μ = Σ_i x_i / N
             σ² = Σ_i (x_i − μ)² / N
    pass 2:  y_i = scale_i · (x_i − μ) / √(σ² + ε) + bias_i

Like softmax, the RTL realises this as a two-pass streaming state
machine. The reciprocal-square-root uses a LUT over the normalised
variance; μ and σ² are accumulated in wide fixed-point and truncated
only at the emit stage.

F1-A4 deliverable boundary this session: FP32 oracle + unit tests.
Fixed-point LUT version + RTL + cocotb gate land in the RTL session.

RTL interface contract (for the follow-up session):

    module npu_layernorm #(
        parameter VEC_LEN        = 1024,   // feature dim (ViT token width)
        parameter IN_W           = 32,
        parameter OUT_W          = 32,
        parameter RSQRT_LUT_DEPTH = 256,
        parameter EPS_Q           = 32'h3F800000  // 1.0 as FP32; cfg'd via CSR
    ) (
        input  wire                       clk,
        input  wire                       rst_n,
        input  wire                       start,
        input  wire                       in_valid,
        input  wire signed [IN_W-1:0]     in_data,
        input  wire signed [IN_W-1:0]     in_scale,   // γ_i (stream #2)
        input  wire signed [IN_W-1:0]     in_bias,    // β_i (stream #3)
        output reg                        out_valid,
        output reg  signed [OUT_W-1:0]    out_data,
        output reg                        done
    );

Pass 1: in_valid + in_data for VEC_LEN cycles — accumulates Σx and Σx²
concurrently, computes μ and σ² at the boundary. Pass 2: same VEC_LEN
cycles with in_scale + in_bias streams; emits out_valid + out_data.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Oracle: pure-numpy FP32 layernorm (golden reference)
# ---------------------------------------------------------------------------
def layernorm_fp32(x: np.ndarray,
                   scale: Optional[np.ndarray] = None,
                   bias: Optional[np.ndarray] = None,
                   *,
                   axis: int = -1,
                   epsilon: float = 1e-5) -> np.ndarray:
    """Standard LayerNorm (opset-17 semantics) along `axis`.

    Matches `onnx.op.LayerNormalization` within FP32 rounding so the
    fake-quant validation flow can compare layer-by-layer for transformer
    workloads.

    Args:
        x: input tensor.
        scale: per-feature affine gain γ. If None, γ=1.
        bias: per-feature affine shift β. If None, β=0.
        axis: normalisation axis (or axes) — matches ONNX's axis semantics.
        epsilon: variance stabiliser.

    Returns:
        Normalised tensor, same shape as x, dtype float32.
    """
    x = np.asarray(x, dtype=np.float32)
    # Accept a single axis (ONNX convention) — layernorm normalises over
    # all dims >= axis.
    ndim = x.ndim
    ax = axis if axis >= 0 else axis + ndim
    reduce_axes = tuple(range(ax, ndim))

    mean = x.mean(axis=reduce_axes, keepdims=True)
    centered = x - mean
    var = (centered ** 2).mean(axis=reduce_axes, keepdims=True)
    norm = centered / np.sqrt(var + epsilon)

    if scale is not None:
        scale = np.asarray(scale, dtype=np.float32)
        norm = norm * scale
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float32)
        norm = norm + bias
    return norm.astype(np.float32)


# ---------------------------------------------------------------------------
# RMSNorm (LLaMA-style) — cheap variant that skips mean subtraction
# ---------------------------------------------------------------------------
def rmsnorm_fp32(x: np.ndarray,
                 scale: Optional[np.ndarray] = None,
                 *,
                 axis: int = -1,
                 epsilon: float = 1e-6) -> np.ndarray:
    """RMS-only variant; LLaMA-family models use this.

        y_i = scale_i · x_i / √(Σx²/N + ε)

    Shares the reciprocal-square-root LUT with full LayerNorm, so the
    RTL module will support both via a mode bit.
    """
    x = np.asarray(x, dtype=np.float32)
    ndim = x.ndim
    ax = axis if axis >= 0 else axis + ndim
    reduce_axes = tuple(range(ax, ndim))
    ms = (x ** 2).mean(axis=reduce_axes, keepdims=True)
    norm = x / np.sqrt(ms + epsilon)
    if scale is not None:
        norm = norm * np.asarray(scale, dtype=np.float32)
    return norm.astype(np.float32)


# ---------------------------------------------------------------------------
# Fixed-point mirror (stub for the RTL pass)
# ---------------------------------------------------------------------------
def layernorm_lut(x: np.ndarray,
                  scale: Optional[np.ndarray] = None,
                  bias: Optional[np.ndarray] = None,
                  *,
                  rsqrt_lut: Optional[np.ndarray] = None,
                  axis: int = -1,
                  epsilon: float = 1e-5) -> np.ndarray:
    """Fixed-point mirror of the RTL's two-pass layernorm.

    When `rsqrt_lut` is omitted, falls back to `layernorm_fp32` so
    unit tests still exercise the full pipeline shape. The bit-exact
    LUT-based path lands with the RTL session.
    """
    if rsqrt_lut is None:
        return layernorm_fp32(x, scale, bias, axis=axis, epsilon=epsilon)
    raise NotImplementedError(
        "layernorm_lut with explicit rsqrt_lut lands with F1-A4 RTL."
    )


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 16)).astype(np.float32)
    # Output mean ≈ 0, var ≈ 1 along the normalisation axis.
    y = layernorm_fp32(x, axis=-1)
    assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(y.var(axis=-1), 1.0, atol=1e-3)
    # With scale=2 and bias=3 the output mean/var shift predictably.
    y2 = layernorm_fp32(x, scale=np.full(16, 2.0, dtype=np.float32),
                           bias=np.full(16, 3.0, dtype=np.float32), axis=-1)
    assert np.allclose(y2.mean(axis=-1), 3.0, atol=1e-4)
    assert np.allclose(y2.var(axis=-1), 4.0, atol=1e-3)
    # RMSNorm: x=1s -> y=1s (scaled by scale if given).
    r = rmsnorm_fp32(np.ones((1, 8), dtype=np.float32), axis=-1)
    assert np.allclose(r, 1.0, atol=1e-3)
    print("layernorm_ref self-check: PASS")
