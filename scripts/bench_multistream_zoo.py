"""Run the multi-stream scaling measurement across the whole zoo.

Produces ``reports/multistream_zoo.{json,md}`` — a cross-model comparison
of how aggregate TOPS scales with concurrent stream count.

Usage::

    python scripts/bench_multistream_zoo.py \
        [--streams 1,2,4,8] [--duration 3] [--backend onnxruntime]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astracore import zoo
from astracore.multistream import run_multistream


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="onnxruntime")
    p.add_argument("--streams", default="1,2,4,8")
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--warmup", type=float, default=0.5)
    p.add_argument("--only", nargs="+", default=None,
                   help="limit to these zoo names")
    p.add_argument("--out", default="reports/multistream_zoo.json")
    p.add_argument("--md-out", default="reports/multistream_zoo.md")
    args = p.parse_args(argv)

    streams = tuple(int(s) for s in args.streams.split(","))

    if args.only:
        targets = [zoo.get(n) for n in args.only]
    else:
        targets = [m for m in zoo.available()]

    all_reports = []
    md_parts = [
        f"# Multi-stream zoo — {args.backend}",
        "",
        f"`streams={list(streams)}`, `duration={args.duration}s`, "
        f"`warmup={args.warmup}s`. ",
        "IPS = inferences / sec aggregate, TOPS = aggregate effective, "
        "`scale` = IPS / IPS@1-stream.",
        "",
        "| Model | GMACs | Streams | IPS | TOPS | Scale |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for m in targets:
        path = zoo.local_paths()[m.name]
        if not path.exists():
            print(f"[skip] {m.name} — not downloaded")
            continue
        print(f"[run]  {m.name} (input_shape={m.input_shape})")
        try:
            rep = run_multistream(
                path, backend=args.backend,
                n_streams_list=streams,
                duration_s=args.duration,
                warmup_s=args.warmup,
                input_shape=m.input_shape,
            )
        except Exception as exc:
            print(f"  FAIL: {exc!r}")
            continue

        all_reports.append(rep.as_dict())
        for s in rep.slices:
            md_parts.append(
                f"| {m.name} | {rep.mac_ops_per_inference/1e9:.2f} "
                f"| {s.n_streams} | {s.throughput_ips:.1f} "
                f"| {s.aggregate_tops:.3f} | {s.scaling_vs_single:.2f}× |"
            )

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump({"backend": args.backend,
                   "streams": list(streams),
                   "duration_s": args.duration,
                   "reports": all_reports}, fh, indent=2)
    with open(args.md_out, "w") as fh:
        fh.write("\n".join(md_parts) + "\n")
    print(f"\nJSON: {out}\nMarkdown: {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
