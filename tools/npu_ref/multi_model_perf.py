"""Multi-model NPU performance sweep across tiers.

Produces the evidence-package table: for each model (YOLOv8-N,
ViT-B/16, BEVFormer-Tiny, LLaMA-7B decode) at each tier (demo,
starter, ultra-dense, ultra-effective), report fps / latency /
utilization / SRAM floor.

This is the document a foundry or strategic partner opens with.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.npu_ref.layer_spec import Layer                         # noqa: E402
from tools.npu_ref.perf_model import (                              # noqa: E402
    NpuConfig, matmul_cycles, conv2d_cycles,
    TIER_DEMO, TIER_STARTER, TIER_ULTRA_DENSE, TIER_ULTRA_SPARSE,
)
from tools.npu_ref.yolo_trace import build_yolov8n                  # noqa: E402
from tools.npu_ref.vit_trace import build_vit_b16                   # noqa: E402
from tools.npu_ref.bevformer_trace import build_bevformer_tiny      # noqa: E402
from tools.npu_ref.llama_trace import build_llama7b_decode          # noqa: E402


def _layer_cycles(cfg: NpuConfig, L) -> tuple[int, int, int]:
    """Return (cycles, macs, peak_sram_bytes) for one Layer on cfg.

    Duck-types over two Layer flavours: the new layer_spec.Layer (with
    op/m/k/n/batch) and the legacy yolo_trace.Layer (with
    in_shape/out_shape/kernel).
    """
    if hasattr(L, "op"):
        # new layer_spec.Layer
        if L.op == "conv":
            c_in = L.meta.get("c_in", 0)
            c_out = L.meta.get("c_out", 0)
            h_out = L.meta.get("h_out", 0)
            w_out = L.meta.get("w_out", 0)
            k_h, k_w = L.meta.get("kernel", (3, 3))
            stats = conv2d_cycles(cfg, c_in=c_in, c_out=c_out,
                                  h_out=h_out, w_out=w_out, k_h=k_h, k_w=k_w)
        else:
            stats = matmul_cycles(cfg, M=L.m, N=L.n, K=L.k)
            if L.batch > 1:
                stats.total_cycles *= L.batch
                stats.macs_issued *= L.batch
    else:
        # legacy yolo_trace.Layer
        if L.kernel == 0:
            return 0, 0, 0
        c_in, h_in, w_in = L.in_shape
        c_out, h_out, w_out = L.out_shape
        stats = conv2d_cycles(cfg, c_in=c_in, c_out=c_out,
                              h_out=h_out, w_out=w_out,
                              k_h=L.kernel, k_w=L.kernel)
    return stats.total_cycles, stats.macs_issued, stats.sram_peak_bytes


def run_model(cfg: NpuConfig, layers: List[Layer]) -> Dict:
    total_cycles = 0
    total_macs = 0
    max_sram = 0
    for L in layers:
        c, m, s = _layer_cycles(cfg, L)
        total_cycles += c
        total_macs += m
        max_sram = max(max_sram, s)

    effective_cycles = total_cycles / max(1.0, cfg.precision_mul)
    seconds = effective_cycles / cfg.clock_hz
    fps = 1.0 / seconds if seconds > 0 else float("inf")
    peak_dense = cfg.macs * 2 * cfg.clock_hz
    seconds_dense = total_cycles / cfg.clock_hz
    util = ((total_macs * 2 / seconds_dense) / peak_dense
            if seconds_dense > 0 else 0)
    return {
        "layers": len(layers),
        "macs": total_macs,
        "cycles_dense": total_cycles,
        "latency_ms": seconds * 1e3,
        "fps": fps,
        "util": util,
        "sram_peak_kb": max_sram / 1024,
        "sram_cap_kb": cfg.sram_bytes / 1024,
    }


MODELS: List[tuple[str, Callable]] = [
    ("YOLOv8-N (640×640)",         build_yolov8n),
    ("ViT-B/16 (224×224)",         build_vit_b16),
    ("BEVFormer-Tiny (6× cam)",    build_bevformer_tiny),
    ("LLaMA-7B decode (seq=512)",  lambda: build_llama7b_decode(seq_len=512)),
]


TIERS = [TIER_DEMO, TIER_STARTER, TIER_ULTRA_DENSE, TIER_ULTRA_SPARSE]


def _fmt_fps(fps: float) -> str:
    if fps == float("inf"):
        return "inf"
    if fps < 1:
        return f"{fps:.2f}"
    if fps < 100:
        return f"{fps:.1f}"
    return f"{fps:,.0f}"


if __name__ == "__main__":
    print("=" * 105)
    print("NPU multi-model performance sweep — evidence-package summary")
    print("=" * 105)
    print("Target workload: automotive perception + in-cabin NLP")
    print("  - YOLOv8-N : 180 fps (30 fps × 6 cams)")
    print("  - BEVFormer: 15 fps (one multi-camera frame at 10-20 Hz)")
    print("  - ViT-B/16 : 180 fps (general-purpose vision backbone)")
    print("  - LLaMA-7B : 10-30 tok/s for conversational UX")
    print("=" * 105)

    for name, builder in MODELS:
        layers = builder()
        print(f"\n### {name}  —  {len(layers)} layers")
        hdr = f"  {'tier':<55} {'peak TOPS':>10} {'MACs/frame':>12} {'fps':>10} {'lat_ms':>10} {'util%':>6}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for cfg in TIERS:
            r = run_model(cfg, layers)
            print(f"  {cfg.name[:55]:<55} {cfg.peak_tops:>10.2f} "
                  f"{r['macs']/1e9:>9.2f} G  {_fmt_fps(r['fps']):>10} "
                  f"{r['latency_ms']:>10.3f} {r['util']*100:>5.2f}")

    print("\n" + "=" * 105)
    print("Interpretation")
    print("=" * 105)
    print("""
- Starter tier (28nm, 4 TOPS) is sized for YOLOv8 + ViT range but not
  BEVFormer.  Production-acceptable for basic perception only.

- Ultra DENSE INT8 (98 TOPS, 24576 MACs) handles BEVFormer comfortably,
  ViT easily, YOLOv8 at >500 fps.  LLaMA-7B decode: compute is fine,
  memory bandwidth is the real constraint (not modelled here).

- Ultra EFFECTIVE (1573 TOPS with INT2+2:4+sparsity) delivers the
  headline number that justifies the ultra tier.  All four models run
  comfortably.

- MAC utilization stays low (5-25%) on single models because individual
  layers don't fully fill 24576 MACs.  Real workloads would batch or
  pipeline multiple models; that lifts utilization.

- SRAM steady-state working set is tiny (~25 KB) at ultra.  The bigger
  question is DDR bandwidth for LLaMA-7B (needs LPDDR5/HBM).  That's
  the separate analysis NOT in this sweep.
""")
