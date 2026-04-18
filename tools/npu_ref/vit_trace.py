"""ViT-B/16 layer trace (Dosovitskiy et al. 2020, ImageNet-1k).

Architecture (from paper):
  Input: 224×224×3 RGB image
  Patch embed: Conv 16×16 stride-16, 3→768 (produces 14×14 = 196 patches)
  Add CLS token → 197 tokens × 768 hidden
  12 transformer blocks, each:
    LayerNorm → MHA (12 heads × 64 head_dim) → residual
    LayerNorm → FFN (768→3072, GELU, 3072→768) → residual
  LayerNorm → Linear(768→1000) head

Total MACs per inference at 224×224: ~17.6 GFLOPs (published).
Our model accounts for Conv, Linear, and Attention matmul only;
LayerNorm and GELU add a few % that we treat as negligible for
architecture-level sizing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Allow running as a script from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.npu_ref.layer_spec import (    # noqa: E402
    Layer, conv_layer, linear_layer, attention_layer, sum_macs,
)


# Reference dims
IMG_H = 224
IMG_W = 224
PATCH = 16
HIDDEN = 768
HEADS = 12
HEAD_DIM = HIDDEN // HEADS          # 64
FFN_DIM = 3072                      # 4× hidden
NUM_BLOCKS = 12
NUM_CLASSES = 1000


def build_vit_b16() -> List[Layer]:
    layers: List[Layer] = []

    # Patch embedding: Conv with kernel = stride = 16, 3→768
    h_out = IMG_H // PATCH          # 14
    w_out = IMG_W // PATCH          # 14
    layers.append(conv_layer(
        "patch_embed", c_in=3, c_out=HIDDEN,
        h_out=h_out, w_out=w_out, k_h=PATCH, k_w=PATCH))

    seq_len = h_out * w_out + 1     # 197 (196 patches + 1 CLS)

    for i in range(NUM_BLOCKS):
        # Q, K, V projections: Linear 768 → 768 each, over seq_len tokens
        layers.append(linear_layer(f"blk{i}.qkv",
                                    in_dim=HIDDEN, out_dim=3 * HIDDEN,
                                    batch=seq_len))
        # Attention matmuls (per head)
        layers.extend(attention_layer(f"blk{i}.attn",
                                       seq_len=seq_len,
                                       head_dim=HEAD_DIM,
                                       num_heads=HEADS))
        # Output projection
        layers.append(linear_layer(f"blk{i}.attn_out",
                                    in_dim=HIDDEN, out_dim=HIDDEN,
                                    batch=seq_len))
        # FFN
        layers.append(linear_layer(f"blk{i}.ffn.up",
                                    in_dim=HIDDEN, out_dim=FFN_DIM,
                                    batch=seq_len))
        layers.append(linear_layer(f"blk{i}.ffn.down",
                                    in_dim=FFN_DIM, out_dim=HIDDEN,
                                    batch=seq_len))

    # Classification head — single CLS token only
    layers.append(linear_layer("head",
                                in_dim=HIDDEN, out_dim=NUM_CLASSES,
                                batch=1))

    return layers


def _fmt(n: int) -> str:
    for u in ("", "K", "M", "G", "T"):
        if n < 1000:
            return f"{n:.2f}{u}"
        n /= 1000
    return f"{n:.2f}P"


if __name__ == "__main__":
    layers = build_vit_b16()
    total = sum_macs(layers)
    print(f"ViT-B/16 layer count : {len(layers)}")
    print(f"ViT-B/16 total MACs  : {total:,}  ({_fmt(total)})")
    print(f"Published reference  : ~8.8 G MACs (17.6 GFLOPs)")
    # Top-5 by MACs
    top = sorted(layers, key=lambda L: L.macs, reverse=True)[:5]
    print("\nTop-5 layers by MAC count:")
    for L in top:
        pct = 100.0 * L.macs / total
        print(f"  {L.name:<30} op={L.op:<6}  M={L.m:<5} K={L.k:<5} N={L.n:<5} "
              f"batch={L.batch:<3}  macs={L.macs:,} ({pct:.1f}%)")
