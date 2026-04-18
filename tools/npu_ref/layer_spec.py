"""Generic layer specification shared across model traces.

Each model's analyzer (yolo_trace.py, vit_trace.py, ...) builds a list
of Layer objects.  The perf model walks these and issues matmul_cycles
calls as the common primitive.

Supported op types:
  - "conv"   : 2-D convolution.  Flattened to GEMM with M = H_out*W_out,
               K = C_in * K_h * K_w, N = C_out.
  - "linear" : dense fully-connected (Gemm).  M, K, N are given.
  - "matmul" : batched attention-style matmul; M, K, N given.
               `batch` field scales cycles by that factor (heads, batches).

Fields:
  name       — human-readable label
  op         — "conv" | "linear" | "matmul"
  m, k, n    — GEMM dimensions
  batch      — multiplier (heads × batch-size); default 1
  meta       — free-form dict for extra info (feature-map shapes, etc.)

Cycles and MACs are computed by the perf model; the Layer itself just
carries the spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Layer:
    name: str
    op: str              # "conv" | "linear" | "matmul"
    m: int
    k: int
    n: int
    batch: int = 1       # heads, spatial-positions multiplier, etc.
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def macs(self) -> int:
        return self.batch * self.m * self.n * self.k


# ---------------------------------------------------------------------------
# Helpers to build layers for common patterns
# ---------------------------------------------------------------------------
def conv_layer(name: str, c_in: int, c_out: int,
               h_out: int, w_out: int,
               k_h: int = 3, k_w: int = 3,
               meta_in=None, meta_out=None) -> Layer:
    """Flatten a Conv2D into its GEMM-equivalent Layer."""
    return Layer(
        name=name, op="conv",
        m=h_out * w_out,
        k=c_in * k_h * k_w,
        n=c_out,
        meta={"c_in": c_in, "c_out": c_out, "h_out": h_out, "w_out": w_out,
              "kernel": (k_h, k_w)},
    )


def linear_layer(name: str, in_dim: int, out_dim: int,
                 batch: int = 1) -> Layer:
    """Linear / fully-connected / Gemm layer.  M=batch, K=in_dim, N=out_dim."""
    return Layer(name=name, op="linear",
                 m=batch, k=in_dim, n=out_dim, batch=1)


def attention_layer(name: str, seq_len: int, head_dim: int,
                    num_heads: int = 1) -> List[Layer]:
    """Attention scaled dot-product: Q K^T and softmax(.)V matmuls.

    Returns a list of two Layers (one per matmul) because they have
    different dimensions:
      Q K^T: [seq, head_dim] × [head_dim, seq] → [seq, seq]      (per head)
      A V  : [seq, seq]      × [seq, head_dim] → [seq, head_dim] (per head)

    The softmax itself is ~N² per head per sequence — negligible vs the
    matmuls — and is folded into the existing compute budget at the
    perf-model level (not modelled here).
    """
    qk = Layer(name=f"{name}.QK^T", op="matmul",
               m=seq_len, k=head_dim, n=seq_len, batch=num_heads,
               meta={"kind": "attn_qk"})
    av = Layer(name=f"{name}.AV", op="matmul",
               m=seq_len, k=seq_len, n=head_dim, batch=num_heads,
               meta={"kind": "attn_av"})
    return [qk, av]


def sum_macs(layers: List[Layer]) -> int:
    return sum(L.macs for L in layers)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # A tiny model: 1 conv + 1 linear + 1 attention layer
    layers = [
        conv_layer("stem", c_in=3, c_out=16, h_out=320, w_out=320),
        linear_layer("fc", in_dim=768, out_dim=1000, batch=1),
    ] + attention_layer("attn0", seq_len=197, head_dim=64, num_heads=12)

    # Sanity: conv macs = H_out * W_out * K² * C_in * C_out
    assert layers[0].macs == 320 * 320 * 9 * 3 * 16, layers[0].macs
    # Linear: m * k * n
    assert layers[1].macs == 1 * 768 * 1000
    # Attention heads × (seq * head_dim * seq) for QK^T
    assert layers[2].macs == 12 * 197 * 64 * 197

    total = sum_macs(layers)
    print(f"Total MACs: {total:,}")
    print("layer_spec self-check: PASS")
