"""Quantise every zoo model that survives our loader.

For each ZooModel whose ONNX file is on disk and whose ops the
``tools.npu_ref`` loader supports:
  1. Run PTQ with N calibration samples (default 100).
  2. Emit ``data/models/zoo/int8/<name>.int8.onnx``.
  3. Measure FP32-vs-fake-INT8 SNR + cosine + max-abs error on a
     held-out probe input.
  4. Append a row to ``data/models/zoo/int8/manifest.json``.

The big ONNX binaries themselves are gitignored (they bloat the repo;
1-500 MB each). The manifest is the committed artefact — it proves
"every model in our zoo has an INT8 recipe that lands in XX dB SNR"
without the size tax.

Usage::

    python scripts/quantise_zoo.py
    python scripts/quantise_zoo.py --only yolov8n --cal-samples 50
    python scripts/quantise_zoo.py --skip bert-squad-10,gpt-2-10

Failures (loader limits, shape-inference issues) are captured in the
manifest with an ``error`` field — the sweep continues.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from astracore import zoo as zoo_mod          # noqa: E402
from astracore.quantise import quantise       # noqa: E402


OUT_DIR = REPO / "data" / "models" / "zoo" / "int8"
MANIFEST = OUT_DIR / "manifest.json"


def _short_host() -> Dict[str, Any]:
    import platform, os
    return {
        "platform": sys.platform,
        "cpu": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cal-samples", type=int, default=100)
    p.add_argument("--cal-seed", type=int, default=0)
    p.add_argument("--precision", default="int8",
                   choices=["int8", "int4", "int2"])
    p.add_argument("--granularity", default="per_channel",
                   choices=["per_channel", "per_tensor"])
    p.add_argument("--method", default="max_abs",
                   choices=["max_abs", "percentile"])
    p.add_argument("--only", default=None,
                   help="comma-separated subset of zoo model names")
    p.add_argument("--skip", default="",
                   help="comma-separated zoo model names to skip")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    only = ([x.strip() for x in args.only.split(",") if x.strip()]
            if args.only else None)
    skip = {x.strip() for x in args.skip.split(",") if x.strip()}

    paths = zoo_mod.local_paths()
    models = zoo_mod.all_models()
    if only:
        models = [m for m in models if m.name in only]

    rows: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    for m in models:
        if m.name in skip:
            continue
        src = paths.get(m.name)
        if not src or not src.exists():
            rows.append({
                "model": m.name, "family": m.family,
                "status": "skipped",
                "reason": f"source ONNX missing: {src}",
            })
            print(f"[skip] {m.name}: source missing")
            continue

        out = OUT_DIR / f"{m.name}.int8.onnx"
        print(f"\n=== {m.name} ({m.family}) ===")
        print(f"  {src} -> {out}")
        try:
            man = quantise(
                model_path=src, output_path=out,
                cal_samples=args.cal_samples,
                cal_seed=args.cal_seed,
                precision=args.precision,
                granularity=args.granularity,
                calibration_method=args.method,
            )
            rows.append({
                "model": m.name, "family": m.family, "display_name": m.display_name,
                "status": "ok",
                **asdict(man),
            })
            print(f"  [OK]  SNR={man.snr_db:.2f} dB  cosine={man.cosine:.6f}  "
                  f"wall={man.wall_s:.1f}s")
        except Exception as exc:
            rows.append({
                "model": m.name, "family": m.family,
                "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"  [FAIL] {type(exc).__name__}: {exc}")

    wall = time.perf_counter() - t_start

    # Aggregate stats for the header.
    ok = [r for r in rows if r["status"] == "ok"]
    snrs = [r["snr_db"] for r in ok]
    summary = {
        "n_models": len(rows),
        "n_ok": len(ok),
        "n_failed": sum(1 for r in rows if r["status"] == "fail"),
        "n_skipped": sum(1 for r in rows if r["status"] == "skipped"),
        "mean_snr_db": round(sum(snrs) / len(snrs), 2) if snrs else 0.0,
        "min_snr_db": round(min(snrs), 2) if snrs else 0.0,
        "max_snr_db": round(max(snrs), 2) if snrs else 0.0,
        "wall_s_total": round(wall, 2),
    }

    payload = {
        "generated_at_unix": int(time.time()),
        "host": _short_host(),
        "precision": args.precision,
        "granularity": args.granularity,
        "calibration_method": args.method,
        "calibration_samples": args.cal_samples,
        "calibration_seed": args.cal_seed,
        "summary": summary,
        "rows": rows,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"Zoo PTQ sweep complete — manifest: {MANIFEST}")
    print(f"  {summary['n_ok']}/{summary['n_models']} succeeded, "
          f"SNR range {summary['min_snr_db']}..{summary['max_snr_db']} dB "
          f"(mean {summary['mean_snr_db']})")
    print(f"  {summary['n_failed']} failed, {summary['n_skipped']} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
