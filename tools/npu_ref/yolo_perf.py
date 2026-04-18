"""Run YOLOv8-N layer-by-layer through perf_model at multiple NPU tiers.

Produces a per-layer table (cycles, MAC util, DDR traffic) plus end-to-end
totals: latency per frame, fps, aggregate MAC utilization, peak SRAM
working set, total DDR bytes per frame.

Output is the primary evidence-package artefact for "how does YOLOv8 run
on this NPU?" — the question a foundry or investor conversation opens with.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tools.npu_ref.yolo_trace import build_yolov8n  # noqa: E402
from tools.npu_ref.perf_model import (                # noqa: E402
    NpuConfig,
    TIER_DEMO, TIER_STARTER, TIER_ULTRA_DENSE, TIER_ULTRA_SPARSE,
    conv2d_cycles,
)


def run_yolo(cfg: NpuConfig, *, verbose: bool = False):
    """Run YOLOv8-N through the perf model and return aggregate stats."""
    layers = build_yolov8n()
    total_cycles = 0
    total_macs = 0
    total_ddr = 0
    max_ws = 0

    per_layer_rows = []

    for L in layers:
        if L.kernel == 0:
            continue  # non-conv (should not occur in our trace)
        c_in, h_in, w_in = L.in_shape
        c_out, h_out, w_out = L.out_shape
        stats = conv2d_cycles(cfg,
                              c_in=c_in, c_out=c_out,
                              h_out=h_out, w_out=w_out,
                              k_h=L.kernel, k_w=L.kernel,
                              name=L.name)
        total_cycles += stats.total_cycles
        total_macs += stats.macs_issued
        total_ddr += stats.ddr_bytes
        max_ws = max(max_ws, stats.sram_peak_bytes)

        per_layer_rows.append((L.name, L.in_shape, L.out_shape, L.kernel,
                                stats.total_cycles, stats.macs_issued))

    effective_cycles = total_cycles / max(1.0, cfg.precision_mul)
    seconds = effective_cycles / cfg.clock_hz if cfg.clock_hz > 0 else 0
    fps = 1.0 / seconds if seconds > 0 else float("inf")

    # MAC utilization is INT8-equivalent (multiplier cancels in both
    # numerator and denominator).  Stays comparable across tiers.
    peak_dense = cfg.macs * 2 * cfg.clock_hz
    seconds_dense = total_cycles / cfg.clock_hz if cfg.clock_hz > 0 else 0
    util = (total_macs * 2 / seconds_dense) / peak_dense if seconds_dense > 0 else 0

    if verbose:
        print(f"\n=== YOLOv8-N on {cfg.name} ===")
        print(f"{'layer':<28} {'in':<20} {'out':<20} {'k':>2} "
              f"{'cycles':>14} {'macs':>16}")
        for r in per_layer_rows[:12]:  # show first 12 layers for brevity
            print(f"{r[0][:28]:<28} {str(r[1]):<20} {str(r[2]):<20} "
                  f"{r[3]:>2} {r[4]:>14,} {r[5]:>16,}")
        print(f"... ({len(per_layer_rows)-12} more layers)")

    return {
        "tier": cfg.name,
        "layers": len(per_layer_rows),
        "total_cycles": total_cycles,
        "total_macs": total_macs,
        "latency_ms": seconds * 1e3,
        "fps": fps,
        "mac_util": util,
        "sram_peak_kb": max_ws / 1024,
        "sram_cap_kb": cfg.sram_bytes / 1024,
        "ddr_mb_per_frame": total_ddr / (1024 * 1024),
        "peak_tops": cfg.peak_tops,
    }


def _fmt(n):
    if n == float("inf"):
        return "inf"
    return f"{n:,.2f}" if n < 10 else f"{n:,.0f}"


if __name__ == "__main__":
    print("YOLOv8-N performance analysis across NPU tiers")
    print("=" * 85)
    print("Target automotive workload: 30 fps per camera × 6 cameras = 180 fps")
    print("=" * 85)

    tiers = [TIER_DEMO, TIER_STARTER, TIER_ULTRA_DENSE, TIER_ULTRA_SPARSE]
    results = []
    for t in tiers:
        r = run_yolo(t, verbose=False)
        results.append(r)

    # Summary table
    print(f"\n{'tier':<55} {'peak':>9} {'fps':>10} {'lat_ms':>10} "
          f"{'util%':>7} {'SRAM':>7}")
    print(f"{'':55} {'TOPS':>9}")
    print("-" * 110)
    for r in results:
        tops = _fmt(r["peak_tops"])
        fps = _fmt(r["fps"])
        lat = _fmt(r["latency_ms"])
        util = _fmt(r["mac_util"] * 100)
        sram = f"{r['sram_peak_kb']:.1f}/{r['sram_cap_kb']:.0f}KB"
        print(f"{r['tier'][:55]:<55} {tops:>9} {fps:>10} {lat:>10} "
              f"{util:>7} {sram:>12}")

    print("\n" + "=" * 85)
    print("Top-5 heaviest layers (on starter tier) — where the cycles go")
    print("=" * 85)
    # Re-run starter with verbose to find the bottleneck layers
    from tools.npu_ref.yolo_trace import build_yolov8n as _b
    from tools.npu_ref.perf_model import conv2d_cycles as _c
    layers = _b()
    per = []
    for L in layers:
        if L.kernel == 0:
            continue
        c_in, h_in, w_in = L.in_shape
        c_out, h_out, w_out = L.out_shape
        s = _c(TIER_STARTER, c_in=c_in, c_out=c_out,
               h_out=h_out, w_out=w_out, k_h=L.kernel, k_w=L.kernel,
               name=L.name)
        per.append((L.name, L.in_shape, L.out_shape, L.kernel,
                    s.total_cycles, s.macs_issued))
    per.sort(key=lambda r: r[4], reverse=True)
    print(f"{'layer':<32} {'in':<20} {'out':<20} {'k':>2} "
          f"{'cycles':>14} {'macs':>16}")
    for r in per[:8]:
        print(f"{r[0][:32]:<32} {str(r[1]):<20} {str(r[2]):<20} "
              f"{r[3]:>2} {r[4]:>14,} {r[5]:>16,}")
