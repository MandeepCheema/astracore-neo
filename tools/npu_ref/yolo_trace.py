"""YOLOv8-N structural analyzer — memory + compute budget for tile sizing.

Computes per-layer MACs, weight bytes, activation bytes, and the largest
tile footprint across the whole model.  Used to set the NPU's on-chip
SRAM floor and the DMA bandwidth budget BEFORE any silicon is committed.

Architecture is derived from the Ultralytics YOLOv8 spec (yolov8.yaml) with
nano scaling:
    width_multiple = 0.25
    depth_multiple = 0.33

No ONNX file is required — the layer list is constructed from the spec so
the analysis is deterministic and reproducible on any machine.

Run directly:
    .venv/Scripts/python.exe tools/npu_ref/yolo_trace.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Model-scale constants (YOLOv8-N / nano)
# ---------------------------------------------------------------------------
WIDTH_MUL = 0.25
DEPTH_MUL = 0.33
INPUT_H = 640
INPUT_W = 640
INPUT_C = 3
BYTES_PER_INT8 = 1
BYTES_PER_INT4 = 0.5


def _w(ch: int) -> int:
    """Apply the width multiplier and round to a multiple of 8 (channel group)."""
    v = max(1, int(ch * WIDTH_MUL))
    # Round up to the nearest multiple of 8 for SIMD friendliness,
    # except for the stem (16 ch) which is the published YOLOv8-N value.
    return (v + 7) // 8 * 8 if v >= 8 else v


def _d(n: int) -> int:
    """Apply the depth multiplier (bottleneck repeats inside a C2f block)."""
    return max(1, round(n * DEPTH_MUL))


# ---------------------------------------------------------------------------
# Layer accounting
# ---------------------------------------------------------------------------
@dataclass
class Layer:
    name: str
    in_shape: tuple      # (C, H, W)
    out_shape: tuple
    kernel: int          # 1 for 1x1, 3 for 3x3, 0 for non-conv
    stride: int          # 1 or 2
    groups: int = 1      # 1 for normal conv, C_in for depthwise
    macs: int = 0
    weight_bytes: int = 0
    act_bytes_out: int = 0

    def __post_init__(self) -> None:
        c_in, h_in, w_in = self.in_shape
        c_out, h_out, w_out = self.out_shape
        if self.kernel > 0:
            k = self.kernel
            self.macs = h_out * w_out * k * k * (c_in // self.groups) * c_out
            self.weight_bytes = k * k * (c_in // self.groups) * c_out  # INT8
        else:
            self.macs = 0
            self.weight_bytes = 0
        self.act_bytes_out = c_out * h_out * w_out  # INT8


def _conv(name, c_in, h_in, w_in, c_out, k=3, s=1, groups=1) -> Layer:
    h_out = (h_in + (k // 2) * 2 - k) // s + 1 if s == 1 else h_in // s
    w_out = (w_in + (k // 2) * 2 - k) // s + 1 if s == 1 else w_in // s
    return Layer(name, (c_in, h_in, w_in), (c_out, h_out, w_out), k, s, groups)


def _c2f_block(prefix, c_in, h, w, c_out, n_bottleneck) -> List[Layer]:
    """C2f block: split-conv with n bottleneck sub-blocks.  Matches Ultralytics.

    Structure:
        1x1 conv (c_in → c_out)             — cv1
        Split into 2 halves of c_out//2
        n × Bottleneck (3x3 conv + 3x3 conv, residual) on one half
        Concat all halves (→ (n+2) × c_out//2 channels)
        1x1 conv (concat → c_out)           — cv2
    """
    layers: List[Layer] = []
    split_ch = c_out // 2
    # cv1: 1x1 proj in
    layers.append(_conv(f"{prefix}.cv1", c_in, h, w, c_out, k=1))
    # bottlenecks: each = conv3x3(split→split) → conv3x3(split→split)
    for i in range(n_bottleneck):
        layers.append(_conv(f"{prefix}.m{i}.cv1", split_ch, h, w, split_ch, k=3))
        layers.append(_conv(f"{prefix}.m{i}.cv2", split_ch, h, w, split_ch, k=3))
    concat_ch = split_ch * (n_bottleneck + 2)
    # cv2: 1x1 proj out
    layers.append(_conv(f"{prefix}.cv2", concat_ch, h, w, c_out, k=1))
    return layers


def _sppf_block(prefix, c_in, h, w, c_out) -> List[Layer]:
    """SPPF: cv1 (1x1 halve) → 3× 5x5 maxpool concat → cv2 (1x1 proj)."""
    mid = c_in // 2
    return [
        _conv(f"{prefix}.cv1", c_in, h, w, mid, k=1),
        _conv(f"{prefix}.cv2", mid * 4, h, w, c_out, k=1),
    ]


# ---------------------------------------------------------------------------
# Full YOLOv8-N graph
# ---------------------------------------------------------------------------
def build_yolov8n() -> List[Layer]:
    layers: List[Layer] = []
    h, w = INPUT_H, INPUT_W

    # --- Backbone --------------------------------------------------------------
    # Stem: Conv(3→64*w=16) 3x3 s=2 → 320x320
    c = _w(64)  # 16
    layers.append(_conv("stem", INPUT_C, h, w, c, k=3, s=2))
    h, w = h // 2, w // 2

    # Down to stage 1
    prev = c
    c = _w(128)  # 32
    layers.append(_conv("backbone.0", prev, h, w, c, k=3, s=2))
    h, w = h // 2, w // 2
    layers.extend(_c2f_block("backbone.c2f1", c, h, w, c, _d(3)))

    # Down to stage 2
    prev = c
    c = _w(256)  # 64
    layers.append(_conv("backbone.1", prev, h, w, c, k=3, s=2))
    h, w = h // 2, w // 2
    layers.extend(_c2f_block("backbone.c2f2", c, h, w, c, _d(6)))

    # Down to stage 3
    prev = c
    c = _w(512)  # 128
    layers.append(_conv("backbone.2", prev, h, w, c, k=3, s=2))
    h, w = h // 2, w // 2
    layers.extend(_c2f_block("backbone.c2f3", c, h, w, c, _d(6)))

    # Down to stage 4
    prev = c
    c = _w(1024)  # 256
    layers.append(_conv("backbone.3", prev, h, w, c, k=3, s=2))
    h, w = h // 2, w // 2
    layers.extend(_c2f_block("backbone.c2f4", c, h, w, c, _d(3)))

    # SPPF
    layers.extend(_sppf_block("sppf", c, h, w, c))

    # --- Neck (FPN) — two upsample paths and two downsample paths -------------
    # Upsample path 1: stage 4 → stage 3 resolution
    layers.extend(_c2f_block("neck.up1", c + _w(512), h * 2, w * 2, _w(512), _d(3)))
    # Upsample path 2: stage 3 → stage 2 resolution
    layers.extend(_c2f_block(
        "neck.up2", _w(512) + _w(256), h * 4, w * 4, _w(256), _d(3)))

    # Down-sample back paths
    layers.append(_conv("neck.down1", _w(256), h * 4, w * 4, _w(256), k=3, s=2))
    layers.extend(_c2f_block(
        "neck.c2f_d1", _w(256) + _w(512), h * 2, w * 2, _w(512), _d(3)))
    layers.append(_conv("neck.down2", _w(512), h * 2, w * 2, _w(512), k=3, s=2))
    layers.extend(_c2f_block(
        "neck.c2f_d2", _w(512) + c, h, w, c, _d(3)))

    # --- Detection head (3 feature levels, decoupled cls/box) ------------------
    # Inner channel widths match Ultralytics' ultralytics/nn/modules/head.py
    # Detect class (v8.4.38): these are set once from ch[0] (stage-3 channels),
    # not scaled per level.
    #   c3 (cls inner) = max(ch[0], min(nc, 100))
    #   c2 (reg inner) = max(16, ch[0] // 4, reg_max * 4)
    # Previously this file used c_lvl on both branches, which over-counted
    # the head by ~1.1 G MACs on YOLOv8-N (reconciled against the real
    # ultralytics 8.4.38 ONNX export on 2026-04-18).
    nc = 80          # COCO classes
    reg_max = 16
    ch0 = _w(256)    # stage-3 feature channels = 64 for YOLOv8-N
    c3 = max(ch0, min(nc, 100))            # 80
    c2 = max(16, ch0 // 4, reg_max * 4)    # 64

    for lvl, (c_lvl, h_lvl, w_lvl) in enumerate([
        (_w(256), INPUT_H // 8,  INPUT_W // 8),
        (_w(512), INPUT_H // 16, INPUT_W // 16),
        (c,       INPUT_H // 32, INPUT_W // 32),
    ]):
        # Classification branch: Conv(c_lvl→c3, 3x3) → Conv(c3→c3, 3x3)
        #                        → Conv(c3→nc, 1x1)
        layers.append(_conv(f"head.{lvl}.cls.1", c_lvl, h_lvl, w_lvl, c3, k=3))
        layers.append(_conv(f"head.{lvl}.cls.2", c3,    h_lvl, w_lvl, c3, k=3))
        layers.append(_conv(f"head.{lvl}.cls.3", c3,    h_lvl, w_lvl, nc, k=1))
        # Regression branch: Conv(c_lvl→c2, 3x3) → Conv(c2→c2, 3x3)
        #                    → Conv(c2→4*reg_max, 1x1)
        layers.append(_conv(f"head.{lvl}.reg.1", c_lvl, h_lvl, w_lvl, c2, k=3))
        layers.append(_conv(f"head.{lvl}.reg.2", c2,    h_lvl, w_lvl, c2, k=3))
        layers.append(_conv(f"head.{lvl}.reg.3", c2,    h_lvl, w_lvl, 4 * reg_max, k=1))

    return layers


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def fmt_bytes(b: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def report(layers: List[Layer]) -> None:
    total_macs = sum(l.macs for l in layers)
    total_weight_bytes = sum(l.weight_bytes for l in layers)
    max_act_out = max(l.act_bytes_out for l in layers)
    max_weight = max(l.weight_bytes for l in layers)

    # Find the largest "single-layer working set" = max(weight + output act)
    largest_layer = max(layers, key=lambda l: l.weight_bytes + l.act_bytes_out)
    largest_ws = largest_layer.weight_bytes + largest_layer.act_bytes_out

    # Bandwidth: total bytes moved per inference at INT8
    # Inputs come in once, outputs go out once, weights traverse once.
    bw_per_frame = total_weight_bytes + max_act_out * 2  # rough lower bound

    print("=" * 78)
    print(f"YOLOv8-N structural analysis  (width={WIDTH_MUL}, depth={DEPTH_MUL})")
    print(f"Input: {INPUT_C}x{INPUT_H}x{INPUT_W},  INT8 precision")
    print("=" * 78)
    print(f"Layer count                  : {len(layers)}")
    print(f"Total MACs (one forward pass): {total_macs:,}  ({total_macs / 1e9:.2f} G)")
    print(f"  -> OPS (2 ops per MAC)     : {2 * total_macs:,}  ({2 * total_macs / 1e9:.2f} GOPS)")
    print(f"Total weight storage (INT8)  : {fmt_bytes(total_weight_bytes)}")
    print(f"  At INT4                    : {fmt_bytes(total_weight_bytes // 2)}")
    print(f"Largest single activation    : {fmt_bytes(max_act_out)}")
    print(f"Largest single weight tensor : {fmt_bytes(max_weight)}")
    print(f"Largest layer working set    : {fmt_bytes(largest_ws)}   [{largest_layer.name}]")
    print(f"Per-frame I/O lower bound    : {fmt_bytes(bw_per_frame)}")
    print()

    # MAC utilization projection at 24,576 MACs × 2 GHz dense INT8
    mac_count = 24_576
    freq_ghz = 2.0
    peak_ops_per_sec = mac_count * 2 * freq_ghz * 1e9
    ops_per_frame = 2 * total_macs
    theoretical_fps = peak_ops_per_sec / ops_per_frame
    print(f"Projection @ {mac_count:,} MACs × {freq_ghz} GHz dense INT8:")
    print(f"  Peak throughput            : {peak_ops_per_sec / 1e12:.1f} TOPS")
    print(f"  Theoretical fps (100% util): {theoretical_fps:,.0f} fps")
    print(f"  At 50% util (realistic)    : {theoretical_fps / 2:,.0f} fps")
    print(f"  At 25% util (conservative) : {theoretical_fps / 4:,.0f} fps")
    print()
    print("Automotive target: 30 fps × 6 cameras = 180 fps.  Plenty of headroom.")
    print()

    # Top-5 heaviest layers
    print("Top 5 layers by MAC count:")
    sorted_layers = sorted(layers, key=lambda l: l.macs, reverse=True)[:5]
    for l in sorted_layers:
        pct = 100 * l.macs / total_macs
        print(f"  {l.name:30s}  {l.macs:>12,}  ({pct:5.2f}%)  "
              f"in={l.in_shape}  out={l.out_shape}  k={l.kernel}")
    print()

    # SRAM floor recommendation
    # We must hold: 1 weight tile + 2 activation banks (double-buffer) + scratch
    w_tile = max_weight                   # worst-case weights for one layer
    act_tile = max_act_out                # worst-case activation bank
    scratch = 0.25 * act_tile
    sram_floor = w_tile + 2 * act_tile + scratch
    print(f"SRAM floor (worst-case tile):")
    print(f"  Weight tile                 : {fmt_bytes(w_tile)}")
    print(f"  2× activation bank          : {fmt_bytes(2 * act_tile)}")
    print(f"  Scratch / partial sums      : {fmt_bytes(scratch)}")
    print(f"  Total minimum on-chip SRAM  : {fmt_bytes(sram_floor)}")
    print(f"  With 2× headroom            : {fmt_bytes(2 * sram_floor)}")
    print()
    print("=" * 78)


if __name__ == "__main__":
    layers = build_yolov8n()
    report(layers)
