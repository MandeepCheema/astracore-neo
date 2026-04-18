"""LLaMA-7B per-token decode trace (Touvron et al. 2023).

Architecture:
  Embedding: 32000 → 4096 (skipped — it's a lookup, not compute)
  32 transformer blocks, each:
    RMSNorm (skip — small compute)
    Attention:  Q (4096→4096),  K (4096→1024),  V (4096→1024)     # GQA: 8:1
                QK^T and AV matmuls (on current seq context)
                Output projection (4096→4096)
    RMSNorm
    FFN (SwiGLU):  gate (4096→11008),  up (4096→11008),
                   down (11008→4096)
  RMSNorm
  LM head: Linear (4096→32000)

Per-token DECODE mode (generating one new token at a time): each block's
linear layers run on ONE token position.  Attention Q×K^T and A×V run
across the full key-value cache (seq_len long).

Per-token compute (for seq_len ≈ 1):
  Attention projections dominate: 4×4096×4096 ≈ 67M MACs per block
  FFN: 3×4096×11008 ≈ 135M MACs per block
  Attention matmuls: ~1M MACs per block (tiny at seq=1)
  × 32 blocks + LM head = ~6.5G MACs per token at seq_len=1

Per-token compute scales linearly with KV-cache size (seq_len) for the
attention matmuls but is dominated by FFN at modest seq lengths.

Published: LLaMA-7B is ~6.7B parameters; at INT8, weights are ~6.7 GB.
Memory bandwidth is the bottleneck in decode (not compute).  This trace
tells us the COMPUTE needs; memory analysis is a separate concern.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.npu_ref.layer_spec import (  # noqa: E402
    Layer, linear_layer, attention_layer, sum_macs,
)


HIDDEN     = 4096
NUM_HEADS  = 32
HEAD_DIM   = HIDDEN // NUM_HEADS      # 128
NUM_KV_HEADS = 32                      # LLaMA-7B: no GQA (same as NUM_HEADS)
KV_HIDDEN  = NUM_KV_HEADS * HEAD_DIM   # 4096 for LLaMA-7B
FFN_DIM    = 11008
NUM_BLOCKS = 32
VOCAB      = 32000


def build_llama7b_decode(seq_len: int = 1) -> List[Layer]:
    """Per-token decode: input is ONE token position; seq_len is the
    length of the KV cache the new token attends to."""
    layers: List[Layer] = []

    for i in range(NUM_BLOCKS):
        # Q projection (4096 → 4096), running on 1 token
        layers.append(linear_layer(f"blk{i}.q_proj",
                                    in_dim=HIDDEN, out_dim=HIDDEN, batch=1))
        # K, V projections (4096 → KV_HIDDEN)
        layers.append(linear_layer(f"blk{i}.k_proj",
                                    in_dim=HIDDEN, out_dim=KV_HIDDEN, batch=1))
        layers.append(linear_layer(f"blk{i}.v_proj",
                                    in_dim=HIDDEN, out_dim=KV_HIDDEN, batch=1))
        # Attention matmuls: single query attends to seq_len KV positions.
        # Q: [1 × head_dim], K^T: [head_dim × seq_len] → [1 × seq_len]
        layers.append(Layer(name=f"blk{i}.attn.QK",
                             op="matmul",
                             m=1, k=HEAD_DIM, n=seq_len,
                             batch=NUM_HEADS))
        layers.append(Layer(name=f"blk{i}.attn.AV",
                             op="matmul",
                             m=1, k=seq_len, n=HEAD_DIM,
                             batch=NUM_HEADS))
        # Output projection (4096 → 4096)
        layers.append(linear_layer(f"blk{i}.o_proj",
                                    in_dim=HIDDEN, out_dim=HIDDEN, batch=1))
        # FFN (SwiGLU): gate + up + down
        layers.append(linear_layer(f"blk{i}.ffn.gate",
                                    in_dim=HIDDEN, out_dim=FFN_DIM, batch=1))
        layers.append(linear_layer(f"blk{i}.ffn.up",
                                    in_dim=HIDDEN, out_dim=FFN_DIM, batch=1))
        layers.append(linear_layer(f"blk{i}.ffn.down",
                                    in_dim=FFN_DIM, out_dim=HIDDEN, batch=1))

    # LM head (output vocabulary projection)
    layers.append(linear_layer("lm_head",
                                in_dim=HIDDEN, out_dim=VOCAB, batch=1))

    return layers


def _fmt(n: int) -> str:
    for u in ("", "K", "M", "G", "T"):
        if n < 1000:
            return f"{n:.2f}{u}"
        n /= 1000
    return f"{n:.2f}P"


if __name__ == "__main__":
    for seq in (1, 512, 2048):
        layers = build_llama7b_decode(seq_len=seq)
        total = sum_macs(layers)
        print(f"LLaMA-7B decode @ seq_len={seq:<5}  "
              f"layers={len(layers):<4}  MACs={total:,} ({_fmt(total)})")

    layers = build_llama7b_decode(seq_len=1)
    top = sorted(layers, key=lambda L: L.macs, reverse=True)[:5]
    print("\nTop-5 layers by MAC count (seq_len=1):")
    for L in top:
        print(f"  {L.name:<28} op={L.op:<6} M={L.m:<4} K={L.k:<6} "
              f"N={L.n:<6} b={L.batch:<3} macs={L.macs:,}")
