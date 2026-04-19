"""Python golden reference for rtl/npu_softmax/npu_softmax.v (F1-A4).

Fused two-pass softmax with numerical-stability guarantees:

    pass 1:  m = max(x)                        # row-max scan
             s = Σ_i exp(x_i - m)              # shifted-exp sum
    pass 2:  y_i = exp(x_i - m) / s            # normalised output

The RTL will realise this as a two-pass streaming state machine over
one row (or one attention head's K vector). The Python reference
provides two implementations:

  - `softmax_fp32(x)` — the oracle: pure numpy FP32, the output we
    want the RTL to match to within LUT-quantisation error.
  - `softmax_lut(x, exp_lut, recip_lut)` — the bit-exact mirror of
    what the RTL will do once the LUT tables + intermediate widths
    are specified in F1-A4 RTL. Stubbed in this session with the
    interface contract + wiring; full fixed-point math lands with
    the RTL.

F1-A4 deliverable boundary: Python reference + unit tests this session;
RTL + cocotb fidelity gate in the follow-up RTL session. The interface
contract below is what the RTL will implement.

RTL interface contract (for the follow-up session):

    module npu_softmax #(
        parameter VEC_LEN          = 64,       // one attention K length
        parameter IN_W             = 32,       // INT32 accumulator-width in
        parameter OUT_W            = 8,        // Q0.8 softmax output
        parameter EXP_LUT_DEPTH    = 256,      // exp((-128..127)/scale)
        parameter RECIP_LUT_DEPTH  = 1024,     // 1/s for 10-bit s
    ) (
        input  wire                       clk,
        input  wire                       rst_n,
        input  wire                       start,        // pass 1 kickoff
        input  wire                       in_valid,
        input  wire signed [IN_W-1:0]     in_data,      // streamed x[i]
        output reg                        out_valid,
        output reg  signed [OUT_W-1:0]    out_data,     // streamed y[i]
        output reg                        done
    );

The module takes VEC_LEN cycles of in_valid to accumulate pass 1 (tracks
m and s concurrently via online-softmax-friendly update), then another
VEC_LEN cycles to emit y[i] = exp(x_i - m) * (1/s) where both lookups
are free-running LUT reads.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Oracle: pure-numpy FP32 softmax (golden reference)
# ---------------------------------------------------------------------------
def softmax_fp32(x: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along `axis`. The RTL's output is
    validated against this oracle within the LUT-quantisation floor
    (~40 dB SNR with 256-entry exp LUT + 1024-entry reciprocal LUT)."""
    x = np.asarray(x, dtype=np.float32)
    m = np.max(x, axis=axis, keepdims=True)
    shifted = x - m
    e = np.exp(shifted)
    s = np.sum(e, axis=axis, keepdims=True)
    # Divide-safe: if an entire axis is -inf (shouldn't happen in real
    # attention but we guard anyway) the sum is 0 and the output is
    # uniform over the axis.
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.where(s > 0, e / s, 1.0 / x.shape[axis])
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Fixed-point mirror (stub for the RTL pass)
# ---------------------------------------------------------------------------
def softmax_lut(x: np.ndarray,
                *,
                exp_lut: Optional[np.ndarray] = None,
                recip_lut: Optional[np.ndarray] = None,
                in_scale: float = 1.0,
                axis: int = -1) -> np.ndarray:
    """Fixed-point mirror of the RTL's two-pass softmax.

    LUTs are optional — when omitted this routes to `softmax_fp32` so
    unit tests that don't care about LUT quantisation still pass. The
    LUT+widths are finalised in the F1-A4 RTL session; once those are
    locked this function becomes the bit-exact mirror and gains its
    own acceptance gate (SNR ≥ 30 dB vs softmax_fp32).

    Args:
        x: INT32-style input array; `in_scale` gives the FP-equivalent.
        exp_lut: 256-entry LUT mapping q8.0 (−128..127) → Q0.32 exp.
        recip_lut: 1024-entry LUT mapping Q10.0 (0..1023) → Q0.32 1/s.
        in_scale: multiplicative scale to convert INT32 input to FP.
        axis: softmax axis.

    Returns:
        Array of the same shape as x, dtype float32.
    """
    if exp_lut is None or recip_lut is None:
        # LUT spec is deferred to the RTL session. Fall back to the
        # oracle so unit tests are still meaningful.
        x_fp = x.astype(np.float32) * in_scale
        return softmax_fp32(x_fp, axis=axis)

    # Placeholder: when LUTs land, this will implement the actual
    # two-pass quantised math. Raising loudly rather than silently
    # returning an unquantised oracle when LUTs ARE supplied.
    raise NotImplementedError(
        "softmax_lut with explicit LUTs lands with F1-A4 RTL. Omit "
        "the LUTs to fall back to the FP32 oracle."
    )


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Properties: sums to 1 along axis, monotone in x, uniform at x=0.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 16)).astype(np.float32)
    y = softmax_fp32(x, axis=-1)
    assert np.allclose(y.sum(axis=-1), 1.0, atol=1e-6)
    # Uniform distribution for x ≡ 0.
    u = softmax_fp32(np.zeros(8, dtype=np.float32))
    assert np.allclose(u, 1.0 / 8)
    # Monotone: larger x ⇒ larger y.
    a = softmax_fp32(np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32))
    assert np.all(np.diff(a) > 0)
    print("softmax_ref self-check: PASS")
