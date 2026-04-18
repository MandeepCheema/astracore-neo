"""BEVFormer-Tiny simplified layer trace (Li et al. 2022, nuScenes).

BEVFormer is a camera-only 3-D perception model that converts
multi-view camera features into a bird's-eye-view (BEV) representation
via spatial cross-attention and temporal self-attention.  The tiny
variant uses a ResNet-50 backbone + 6 encoder layers + 1 decoder layer.

This trace captures the compute-heavy operations:
  - ResNet-50 backbone on 6 camera views
  - BEV query Q/K/V projections (256-dim hidden)
  - Spatial cross-attention (BEV query ↔ camera features)
  - Temporal self-attention (current BEV ↔ previous BEV)
  - FFN layers

Ignored:
  - Deformable attention sampling offsets (data-dependent, small compute)
  - LayerNorm/Softmax (small fraction, not grid-sizing-critical)
  - Detection head (small compared to encoder)

Published compute: ~130 GFLOPs per frame on nuScenes (multi-view).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.npu_ref.layer_spec import (  # noqa: E402
    Layer, conv_layer, linear_layer, attention_layer, sum_macs,
)


# BEVFormer-Tiny dimensions (from paper + official code)
NUM_CAMS = 6
IMG_H = 480                 # tiny input resolution
IMG_W = 800
BEV_H = 50                  # BEV grid height (tiny: 50×50)
BEV_W = 50
HIDDEN = 256                # feature dim
NUM_HEADS = 8
HEAD_DIM = HIDDEN // NUM_HEADS   # 32
FFN_DIM = 512
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 1      # tiny uses 1 decoder layer
NUM_QUERIES = 900           # detection queries


def _resnet50_backbone(batch_size: int) -> List[Layer]:
    """ResNet-50 simplified into its heavy Conv layers.

    The full ResNet-50 has 50 conv layers; we roll up a few of them
    into characteristic stages.  The goal is the MAC total, not layer-
    by-layer matching.  For N cameras we scale the batch accordingly.
    """
    layers: List[Layer] = []
    # Stage 1: conv1 (7x7 stride-2) + maxpool → 1/4
    h, w = IMG_H // 2, IMG_W // 2
    layers.append(conv_layer("r50.conv1", c_in=3, c_out=64,
                              h_out=h, w_out=w, k_h=7, k_w=7))
    h, w = h // 2, w // 2
    # Stage 2: 3 bottlenecks, channels 64→256
    for i in range(3):
        layers.append(conv_layer(f"r50.s2.{i}.1x1a", 64, 64, h, w, 1, 1))
        layers.append(conv_layer(f"r50.s2.{i}.3x3",  64, 64, h, w, 3, 3))
        layers.append(conv_layer(f"r50.s2.{i}.1x1b", 64, 256, h, w, 1, 1))
    # Stage 3: 4 bottlenecks, channels 128→512, first has stride-2
    h, w = h // 2, w // 2
    for i in range(4):
        layers.append(conv_layer(f"r50.s3.{i}.1x1a", 256, 128, h, w, 1, 1))
        layers.append(conv_layer(f"r50.s3.{i}.3x3",  128, 128, h, w, 3, 3))
        layers.append(conv_layer(f"r50.s3.{i}.1x1b", 128, 512, h, w, 1, 1))
    # Stage 4: 6 bottlenecks, channels 256→1024
    h, w = h // 2, w // 2
    for i in range(6):
        layers.append(conv_layer(f"r50.s4.{i}.1x1a", 512, 256, h, w, 1, 1))
        layers.append(conv_layer(f"r50.s4.{i}.3x3",  256, 256, h, w, 3, 3))
        layers.append(conv_layer(f"r50.s4.{i}.1x1b", 256, 1024, h, w, 1, 1))
    # Stage 5: 3 bottlenecks, channels 512→2048
    h, w = h // 2, w // 2
    for i in range(3):
        layers.append(conv_layer(f"r50.s5.{i}.1x1a", 1024, 512, h, w, 1, 1))
        layers.append(conv_layer(f"r50.s5.{i}.3x3",  512, 512, h, w, 3, 3))
        layers.append(conv_layer(f"r50.s5.{i}.1x1b", 512, 2048, h, w, 1, 1))

    # Scale by batch_size (number of cameras)
    out = []
    for L in layers:
        out.append(Layer(name=f"{L.name}[cam×{batch_size}]",
                          op=L.op, m=L.m * batch_size,
                          k=L.k, n=L.n, batch=L.batch,
                          meta=L.meta))
    return out


def build_bevformer_tiny() -> List[Layer]:
    layers: List[Layer] = []

    # 1. Backbone × 6 cameras (shared weights, different inputs)
    layers.extend(_resnet50_backbone(batch_size=NUM_CAMS))

    # 2. FPN projection: 2048 → HIDDEN for each feature map level
    #    (simplified: one projection at the final feature level, per cam)
    final_h = IMG_H // 32         # 15
    final_w = IMG_W // 32         # 25
    tokens_per_cam = final_h * final_w    # 375
    total_cam_tokens = NUM_CAMS * tokens_per_cam    # 2250

    layers.append(conv_layer("fpn_proj", 2048, HIDDEN,
                              h_out=final_h, w_out=final_w, k_h=1, k_w=1,
                              meta_in=None, meta_out=None))

    num_bev = BEV_H * BEV_W       # 2500

    # 3. Encoder layers: each has spatial cross-attn + temporal self-attn + FFN
    for i in range(NUM_ENCODER_LAYERS):
        # Spatial cross-attention:
        #   Q = BEV queries (num_bev × HIDDEN)
        #   K, V = image features (total_cam_tokens × HIDDEN)
        # Deformable attention in paper samples K points per query; we
        # approximate as full softmax attention for sizing.
        layers.append(linear_layer(f"enc{i}.sa.q_proj",
                                    HIDDEN, HIDDEN, batch=num_bev))
        layers.append(linear_layer(f"enc{i}.sa.k_proj",
                                    HIDDEN, HIDDEN, batch=total_cam_tokens))
        layers.append(linear_layer(f"enc{i}.sa.v_proj",
                                    HIDDEN, HIDDEN, batch=total_cam_tokens))
        # Q × K^T: [num_bev × HEAD_DIM] × [HEAD_DIM × total_cam_tokens]
        layers.append(Layer(name=f"enc{i}.sa.qk", op="matmul",
                             m=num_bev, k=HEAD_DIM, n=total_cam_tokens,
                             batch=NUM_HEADS))
        layers.append(Layer(name=f"enc{i}.sa.av", op="matmul",
                             m=num_bev, k=total_cam_tokens, n=HEAD_DIM,
                             batch=NUM_HEADS))
        layers.append(linear_layer(f"enc{i}.sa.out",
                                    HIDDEN, HIDDEN, batch=num_bev))

        # Temporal self-attention (BEV × previous-BEV, num_bev tokens)
        layers.append(linear_layer(f"enc{i}.ta.qkv",
                                    HIDDEN, 3 * HIDDEN, batch=num_bev))
        layers.extend(attention_layer(f"enc{i}.ta.attn",
                                       seq_len=num_bev,
                                       head_dim=HEAD_DIM,
                                       num_heads=NUM_HEADS))
        layers.append(linear_layer(f"enc{i}.ta.out",
                                    HIDDEN, HIDDEN, batch=num_bev))

        # FFN
        layers.append(linear_layer(f"enc{i}.ffn.up",
                                    HIDDEN, FFN_DIM, batch=num_bev))
        layers.append(linear_layer(f"enc{i}.ffn.down",
                                    FFN_DIM, HIDDEN, batch=num_bev))

    # 4. Decoder layer (1 for tiny)
    for i in range(NUM_DECODER_LAYERS):
        # Self-attention on queries
        layers.append(linear_layer(f"dec{i}.sa.qkv",
                                    HIDDEN, 3 * HIDDEN, batch=NUM_QUERIES))
        layers.extend(attention_layer(f"dec{i}.sa.attn",
                                       seq_len=NUM_QUERIES,
                                       head_dim=HEAD_DIM,
                                       num_heads=NUM_HEADS))
        layers.append(linear_layer(f"dec{i}.sa.out",
                                    HIDDEN, HIDDEN, batch=NUM_QUERIES))
        # Cross-attention: queries attend to BEV features
        layers.append(linear_layer(f"dec{i}.ca.q_proj",
                                    HIDDEN, HIDDEN, batch=NUM_QUERIES))
        layers.append(linear_layer(f"dec{i}.ca.k_proj",
                                    HIDDEN, HIDDEN, batch=num_bev))
        layers.append(linear_layer(f"dec{i}.ca.v_proj",
                                    HIDDEN, HIDDEN, batch=num_bev))
        layers.append(Layer(name=f"dec{i}.ca.qk", op="matmul",
                             m=NUM_QUERIES, k=HEAD_DIM, n=num_bev,
                             batch=NUM_HEADS))
        layers.append(Layer(name=f"dec{i}.ca.av", op="matmul",
                             m=NUM_QUERIES, k=num_bev, n=HEAD_DIM,
                             batch=NUM_HEADS))
        layers.append(linear_layer(f"dec{i}.ca.out",
                                    HIDDEN, HIDDEN, batch=NUM_QUERIES))
        # FFN
        layers.append(linear_layer(f"dec{i}.ffn.up",
                                    HIDDEN, FFN_DIM, batch=NUM_QUERIES))
        layers.append(linear_layer(f"dec{i}.ffn.down",
                                    FFN_DIM, HIDDEN, batch=NUM_QUERIES))

    return layers


def _fmt(n: int) -> str:
    for u in ("", "K", "M", "G", "T"):
        if n < 1000:
            return f"{n:.2f}{u}"
        n /= 1000
    return f"{n:.2f}P"


if __name__ == "__main__":
    layers = build_bevformer_tiny()
    total = sum_macs(layers)
    print(f"BEVFormer-Tiny layer count : {len(layers)}")
    print(f"BEVFormer-Tiny total MACs  : {total:,}  ({_fmt(total)})")
    print(f"Published reference        : ~130 GFLOPs/frame (multi-view)")
    print(f"Expected MACs              : ~65 GMACs (FLOPs÷2)")
    top = sorted(layers, key=lambda L: L.macs, reverse=True)[:8]
    print("\nTop-8 layers by MAC count:")
    for L in top:
        pct = 100.0 * L.macs / total
        print(f"  {L.name:<34} op={L.op:<6} M={L.m:<6} K={L.k:<5} "
              f"N={L.n:<6} b={L.batch:<3} macs={L.macs:,} ({pct:.1f}%)")
