"""``astracore`` command-line interface.

Subcommands (minimal for Phase A):
  * ``astracore bench --model X.onnx [--backend npu-sim]``
  * ``astracore list backends``
  * ``astracore list ops``
  * ``astracore list quantisers``
  * ``astracore version``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from astracore import __version__
from astracore.registry import list_backends, list_ops, list_quantisers


def _cmd_version(_args) -> int:
    print(__version__)
    return 0


def _cmd_list(args) -> int:
    kind = args.kind
    if kind == "backends":
        items = list_backends()
    elif kind == "ops":
        items = list_ops()
    elif kind == "quantisers":
        items = list_quantisers()
    elif kind == "eps":
        return _list_eps()
    else:
        print(f"Unknown registry: {kind}", file=sys.stderr)
        return 2
    if not items:
        print(f"(no {kind} registered)")
        return 0
    for name in items:
        print(name)
    return 0


def _list_eps() -> int:
    """Print ONNX Runtime execution providers available on this host,
    plus the short-name aliases the SDK accepts in YAML config."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed", file=sys.stderr)
        return 2
    from astracore.backends.ort import EP_ALIASES

    available = set(ort.get_available_providers())
    # Invert EP_ALIASES to full_name -> [short names]
    rev: dict = {}
    for short, full in EP_ALIASES.items():
        rev.setdefault(full, []).append(short)

    print("Available on this host (ONNX Runtime build):")
    for name in sorted(available):
        aliases = ", ".join(sorted(rev.get(name, [])))
        tag = f"  aliases: {aliases}" if aliases else ""
        print(f"  [yes] {name}{tag}")
    print("\nKnown aliases NOT available in this ORT build:")
    any_missing = False
    for full, shorts in sorted(rev.items()):
        if full in available:
            continue
        any_missing = True
        print(f"  [no]  {full:<30} (aliases: {', '.join(sorted(shorts))})")
    if not any_missing:
        print("  (none — this ORT build has every known EP)")
    return 0


def _cmd_bench(args) -> int:
    # Lazy import so `astracore version` / `astracore list` stay cheap.
    from astracore.benchmark import benchmark_model

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 2

    report = benchmark_model(
        model_path=model_path,
        backend=args.backend,
        precision=args.precision,
        sparsity=args.sparsity,
        n_iter=args.iter,
        input_shape=args.input_shape,
    )

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(report.markdown_header())
        print(report.as_markdown_row())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="astracore",
        description="AstraCore Neo — Automotive AI Inference SDK",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("version", help="print SDK version").set_defaults(func=_cmd_version)

    lst = sub.add_parser("list",
                         help="list registered ops / backends / quantisers / EPs")
    lst.add_argument("kind", choices=["ops", "backends", "quantisers", "eps"])
    lst.set_defaults(func=_cmd_list)

    bench = sub.add_parser(
        "bench",
        help="benchmark a model on a backend and print KPIs",
    )
    bench.add_argument("--model", required=True, help="path to an ONNX file")
    bench.add_argument("--backend", default="npu-sim",
                       help="target backend (default: npu-sim)")
    bench.add_argument("--precision", default="INT8",
                       choices=["INT8", "INT4", "INT2"])
    bench.add_argument("--sparsity", default="dense",
                       choices=["dense", "2:4", "4:1", "8:1"])
    bench.add_argument("--iter", type=int, default=1,
                       help="number of inference iterations to time")
    bench.add_argument("--input-shape", dest="input_shape", default=None,
                       help="override input shape, e.g. '1,3,640,640'")
    bench.add_argument("--json", action="store_true",
                       help="emit JSON instead of markdown")
    bench.set_defaults(func=_cmd_bench)

    zoo = sub.add_parser(
        "zoo",
        help="run the benchmark matrix against the curated model zoo",
    )
    zoo.add_argument("--backend", default="onnxruntime",
                     help="target backend (default: onnxruntime)")
    zoo.add_argument("--iter", type=int, default=3,
                     help="timed iterations per model (default: 3)")
    zoo.add_argument("--only", nargs="+", default=None,
                     help="restrict to these zoo model names")
    zoo.add_argument("--out", default="reports/model_zoo_matrix.json",
                     help="JSON output path")
    zoo.add_argument("--md-out", default="reports/model_zoo_matrix.md",
                     help="Markdown output path")
    zoo.set_defaults(func=_cmd_zoo)

    replay = sub.add_parser(
        "replay",
        help="replay a dataset scene through the perception pipeline",
    )
    replay.add_argument("--dataset", default="synthetic",
                        choices=["synthetic", "nuscenes"],
                        help="dataset connector to use")
    replay.add_argument("--preset", default="tiny",
                        choices=["tiny", "standard", "vlp32", "vlp64", "robotaxi"],
                        help="synthetic dataset size preset (default: tiny)")
    replay.add_argument("--dataroot", default=None,
                        help="path to dataset root (nuscenes only)")
    replay.add_argument("--version", default="v1.0-mini",
                        help="dataset version (nuscenes only)")
    replay.add_argument("--scene", default=None,
                        help="scene ID; first scene if omitted")
    replay.add_argument("--out", default="reports/replay_result.json",
                        help="JSON output path")
    replay.set_defaults(func=_cmd_replay)

    ms = sub.add_parser(
        "multistream",
        help="measure aggregate MAC utilisation vs number of concurrent streams",
    )
    ms.add_argument("--model", required=True, help="ONNX file")
    ms.add_argument("--backend", default="onnxruntime")
    ms.add_argument("--streams", default="1,2,4,8",
                    help="comma-separated list of stream counts")
    ms.add_argument("--duration", type=float, default=5.0,
                    help="seconds per stream-count slice")
    ms.add_argument("--warmup", type=float, default=1.0,
                    help="seconds of warmup before measurement")
    ms.add_argument("--input-shape", dest="input_shape", default=None)
    ms.add_argument("--out", default="reports/multistream.json")
    ms.add_argument("--md-out", default="reports/multistream.md")
    ms.set_defaults(func=_cmd_multistream)

    demo = sub.add_parser(
        "demo",
        help="run real-input inference + decoded output for a zoo model",
    )
    demo.add_argument("--model", default=None,
                      help="zoo entry name; omit to demo every downloaded model")
    demo.add_argument("--input", default=None,
                      help="input spec (image name like 'bus', a path, or omitted for the family default)")
    demo.add_argument("--backend", default="onnxruntime")
    demo.add_argument("--warmup", type=int, default=0,
                      help="run the model this many times before the timed "
                           "call (default 0). ORT compiles kernels on first "
                           "use; pass --warmup 3 for steady-state latency.")
    demo.add_argument("--out", default="reports/demo_results.json")
    demo.add_argument("--md-out", default="reports/demo_results.md")
    demo.set_defaults(func=_cmd_demo)

    mp = sub.add_parser(
        "mlperf",
        help="run MLPerf-style scenarios (SingleStream / MultiStream / Offline / Server)",
    )
    mp.add_argument("--model", required=True, help="ONNX model path")
    mp.add_argument("--backend", default="onnxruntime")
    mp.add_argument("--providers", default="",
                    help="comma-separated EP names / aliases for the backend")
    mp.add_argument("--scenario", default="single_stream",
                    choices=["single_stream", "multi_stream",
                             "offline", "server", "all"])
    mp.add_argument("--duration", type=float, default=10.0)
    mp.add_argument("--n-samples", type=int, default=256,
                    help="offline scenario only")
    mp.add_argument("--n-streams", type=int, default=4,
                    help="multi_stream + server worker count")
    mp.add_argument("--qps", type=float, default=10.0,
                    help="server scenario target QPS")
    mp.add_argument("--sla-ms", type=float, default=None,
                    help="p99 SLA in ms (server defaults to 100)")
    mp.add_argument("--out", default="reports/mlperf/",
                    help="output directory for per-scenario JSON + MD")
    mp.set_defaults(func=_cmd_mlperf)

    qt = sub.add_parser(
        "quantise",
        help="post-training-quantise an ONNX model; emit fake-quant ONNX + manifest",
    )
    qt.add_argument("--model", required=True, help="path to an FP32 ONNX file")
    qt.add_argument("--out", default=None,
                    help="output fake-quant ONNX path (default: <model>.int8.onnx)")
    qt.add_argument("--manifest", default=None,
                    help="output JSON manifest path (default: <out>.json)")
    qt.add_argument("--cal-samples", type=int, default=100,
                    help="number of calibration batches (default 100)")
    qt.add_argument("--cal-seed", type=int, default=0,
                    help="seed for the synthetic calibration set")
    qt.add_argument("--precision", default="int8",
                    choices=["int8", "int4", "int2"])
    qt.add_argument("--granularity", default="per_channel",
                    choices=["per_channel", "per_tensor"])
    qt.add_argument("--method", default="max_abs",
                    choices=["max_abs", "percentile"],
                    help="calibration strategy (default max_abs)")
    qt.add_argument("--percentile", type=float, default=99.9999,
                    help="percentile for method=percentile (default 99.9999)")
    qt.add_argument("--no-silu-fuse", action="store_true",
                    help="skip the SiLU Sigmoid+Mul fusion pass")
    qt.add_argument("--engine", default="auto",
                    choices=["internal", "ort", "auto"],
                    help="quantisation engine (default auto: try internal "
                         "tools.npu_ref pipeline, fall back to onnxruntime "
                         "quantize_static for broader op coverage)")
    qt.set_defaults(func=_cmd_quantise)

    conf = sub.add_parser(
        "conformance",
        help="run MLIR/IREE/TVM conformance suite (canonical op → NnGraph round-trip)",
    )
    conf.add_argument("--out", default="reports/compiler_ecosystem",
                      help="output directory for JSON + markdown")
    conf.add_argument("--adapter", action="append", default=None,
                      choices=["mlir-stablehlo", "tvm-relay", "jax-xla"],
                      help="restrict to specific adapters (repeatable)")
    conf.set_defaults(func=_cmd_conformance)

    cfg = sub.add_parser(
        "configure",
        help="validate, summarise, or apply an astracore YAML config",
    )
    cfg.add_argument("--validate", default=None,
                     help="path to a YAML config to validate")
    cfg.add_argument("--dump", default=None,
                     help="print the parsed config back as YAML (round-trip)")
    cfg.add_argument("--apply", default=None,
                     help="apply the YAML config: run replay + bench + (optional) "
                          "multistream for every model declared, emit a combined report")
    cfg.add_argument("--out", default=None,
                     help="output directory for --apply reports "
                          "(default: reports/apply_<slug>/)")
    cfg.add_argument("--bench-iter", type=int, default=3,
                     help="timed iterations per model for --apply (default 3)")
    cfg.add_argument("--multistream-duration", type=float, default=2.0,
                     help="seconds per multistream slice (default 2.0)")
    cfg.add_argument("--skip-replay", action="store_true",
                     help="skip the dataset replay step under --apply")
    cfg.add_argument("--skip-bench", action="store_true",
                     help="skip per-model benchmarks under --apply")
    cfg.add_argument("--skip-multistream", action="store_true",
                     help="skip multistream scaling under --apply")
    cfg.set_defaults(func=_cmd_configure)

    return p


def _cmd_conformance(args) -> int:
    from tools.frontends.iree_conformance import (
        run_conformance_suite, report_as_markdown,
    )
    from tools.frontends.tvm_byoc import describe_patterns
    import json as _json

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = run_conformance_suite(adapters=args.adapter)
    payload = report.to_dict()
    # Attach the TVM-BYOC pattern table so the conformance artefact
    # captures the full compiler-ecosystem surface in one place.
    payload["tvm_byoc_patterns"] = describe_patterns()

    (out_dir / "conformance.json").write_text(
        _json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "conformance.md").write_text(
        report_as_markdown(report), encoding="utf-8")

    print(report.summary_line())
    print(f"  JSON:     {out_dir / 'conformance.json'}")
    print(f"  Markdown: {out_dir / 'conformance.md'}")
    return 0 if report.fail_count == 0 else 1


def _cmd_mlperf(args) -> int:
    import json as _json
    from astracore.mlperf import Scenario, run_all_scenarios, run_scenario
    from astracore.registry import get_backend

    providers = [s.strip() for s in args.providers.split(",") if s.strip()]
    backend_cls = get_backend(args.backend)
    if isinstance(backend_cls, type):
        be = backend_cls(**({"providers": providers} if providers else {}))
    else:
        be = backend_cls
    be.name = getattr(be, "name", args.backend)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scenario == "all":
        reports = run_all_scenarios(
            backend=be, model_path=args.model,
            duration_s=args.duration,
            n_samples=args.n_samples,
            n_streams=args.n_streams,
            server_qps=args.qps,
        )
        for name, r in reports.items():
            (out_dir / f"{name}.json").write_text(
                _json.dumps(r.as_dict(), indent=2), encoding="utf-8")
            (out_dir / f"{name}.md").write_text(
                r.as_markdown(), encoding="utf-8")
            p99 = r.latency_stats_ms.get("p99_ms", 0)
            print(f"  [{name:<13}] {r.throughput_qps:8.2f} QPS  "
                  f"p99={p99:.2f} ms  n={r.n_queries}")
    else:
        sc = Scenario(args.scenario)
        r = run_scenario(
            backend=be, model_path=args.model,
            scenario=sc,
            duration_s=args.duration,
            n_samples=args.n_samples,
            n_streams=args.n_streams,
            server_qps=args.qps,
            sla_ms=args.sla_ms,
        )
        (out_dir / f"{sc.value}.json").write_text(
            _json.dumps(r.as_dict(), indent=2), encoding="utf-8")
        (out_dir / f"{sc.value}.md").write_text(
            r.as_markdown(), encoding="utf-8")
        print(r.as_markdown())

    print(f"\nReports: {out_dir}")
    return 0


def _cmd_quantise(args) -> int:
    from astracore.quantise import quantise, write_manifest

    model = Path(args.model)
    if not model.exists():
        print(f"model not found: {model}", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else model.with_suffix(".int8.onnx")
    manifest_path = Path(args.manifest) if args.manifest \
                    else Path(str(out) + ".json")

    print(f"Quantising {model} -> {out}")
    print(f"  cal_samples={args.cal_samples} seed={args.cal_seed} "
          f"precision={args.precision} granularity={args.granularity} "
          f"method={args.method}")
    try:
        manifest = quantise(
            model_path=model,
            output_path=out,
            cal_samples=args.cal_samples,
            cal_seed=args.cal_seed,
            precision=args.precision,
            granularity=args.granularity,
            calibration_method=args.method,
            percentile=args.percentile,
            fuse_silu_layers=not args.no_silu_fuse,
            engine=args.engine,
        )
    except Exception as exc:
        print(f"[FAIL] {exc!r}", file=sys.stderr)
        return 1

    write_manifest(manifest, manifest_path)
    print(f"  wall={manifest.wall_s:.1f}s")
    print(f"  FP32 vs fake-INT8 on probe input:")
    print(f"    SNR:     {manifest.snr_db:.2f} dB")
    print(f"    cosine:  {manifest.cosine:.6f}")
    print(f"    max|e|:  {manifest.max_abs_err:.4f}")
    size_ratio = manifest.output_bytes / max(manifest.source_bytes, 1)
    print(f"  size:    {manifest.source_bytes/1e6:.1f} MB -> "
          f"{manifest.output_bytes/1e6:.1f} MB ({size_ratio:.2f}×)")
    print(f"\nFake-quant ONNX: {out}")
    print(f"Manifest:        {manifest_path}")
    return 0


def _cmd_configure(args) -> int:
    from astracore import config as _cfg

    path = args.validate or args.dump or args.apply
    if not path:
        print("provide --validate <path>, --dump <path>, or --apply <path>",
              file=sys.stderr)
        return 2

    try:
        cfg = _cfg.load(path)
    except _cfg.ConfigError as exc:
        print(f"[INVALID] {path}:\n  {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] {exc!r}", file=sys.stderr)
        return 1

    if args.dump:
        print(_cfg.to_yaml(cfg))
        return 0
    if args.apply:
        return _run_apply(cfg, args, path)
    print(f"[VALID] {path}\n")
    print(_cfg.summary(cfg))
    return 0


def _run_apply(cfg, args, src_path) -> int:
    from astracore.apply import apply_config, render_markdown

    out_dir = Path(args.out) if args.out else None
    print(f"[APPLY] {src_path}\n")
    print(f"  backend: {cfg.backend.name}")
    print(f"  dataset: {cfg.dataset.connector}"
          + (f" preset={cfg.dataset.preset}" if cfg.dataset.preset else ""))
    print(f"  models:  {len(cfg.models)}   multistream={cfg.multistream.enabled}")
    print()

    report = apply_config(
        cfg,
        out_dir=out_dir,
        bench_iter=args.bench_iter,
        multistream_duration_s=args.multistream_duration,
        multistream_warmup_s=min(1.0, args.multistream_duration * 0.25),
        skip_replay=args.skip_replay,
        skip_bench=args.skip_bench,
        skip_multistream=args.skip_multistream,
    )

    # Terse terminal summary.
    if report.replay:
        summary = report.replay.summary
        print(f"Replay: {report.replay.scene_id} ({report.replay.n_samples} samples) "
              f"mean_ms/frame={summary.get('mean_ms_per_frame', 0):.2f}")
    for m in report.models:
        if m.bench_ok:
            print(f"  [OK]   {m.id:<24} {m.wall_ms_per_inference:7.2f} ms  "
                  f"{m.gmacs:5.2f} GMACs  {m.delivered_tops:6.3f} TOPS")
        else:
            print(f"  [FAIL] {m.id:<24} {m.bench_error}")
    if report.notes:
        print("\nNotes:")
        for n in report.notes:
            print(f"  - {n}")

    final_dir = Path(args.out) if args.out else Path("reports") / ("apply_" + _slug_for_cli(cfg.name))
    print(f"\nReport: {final_dir / 'report.json'}")
    print(f"Report: {final_dir / 'report.md'}")
    return 0


def _slug_for_cli(s: str) -> str:
    """Same slug as astracore.apply._slug, duplicated to avoid import at CLI top."""
    from astracore.apply import _slug
    return _slug(s)


def _cmd_demo(args) -> int:
    from astracore import zoo as zoo_mod
    from astracore.demo import run_demo
    import json as _json

    if args.model:
        targets = [zoo_mod.get(args.model)]
    else:
        targets = [m for m in zoo_mod.all_models()
                   if zoo_mod.local_paths()[m.name].exists()]
    if not targets:
        print("No downloaded zoo models; run scripts/fetch_model_zoo.py first.",
              file=sys.stderr)
        return 2

    rows = []
    print(f"Running {len(targets)} demo(s) on backend={args.backend}")
    for m in targets:
        path = zoo_mod.local_paths()[m.name]
        try:
            result = run_demo(m, path, input_spec=args.input,
                              backend_name=args.backend,
                              warmup=args.warmup)
            rows.append(result.as_dict())
            ok_tag = "[OK]  " if result.ok else "[FAIL]"
            print(f"  {ok_tag} {m.name:<28} {result.wall_ms:7.1f} ms   "
                  f"{result.summary[:100]}")
            if not result.ok:
                print(f"         error: {result.error}")
        except Exception as exc:
            rows.append({"model": m.name, "family": m.family,
                         "ok": False, "error": repr(exc)})
            print(f"  [FAIL] {m.name}: {exc!r}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        _json.dump({"backend": args.backend, "rows": rows}, fh, indent=2)
    with open(args.md_out, "w") as fh:
        fh.write(f"# AstraCore demo results (backend: {args.backend})\n\n")
        fh.write("Real-input inference with decoded output — proves each model "
                 "actually produces sensible predictions (not just compiles).\n\n")
        fh.write("| Model | Family | Latency (ms) | Top result |\n")
        fh.write("|---|---|---:|---|\n")
        for r in rows:
            if not r.get("ok", True):
                fh.write(f"| {r['model']} | {r.get('family', '?')} | — | "
                         f"**FAIL**: `{r.get('error', '')}` |\n")
            else:
                fh.write(f"| {r['model']} | {r['family']} | "
                         f"{r.get('wall_ms', 0):.1f} | "
                         f"{r.get('summary', '')} |\n")
    print(f"\nJSON: {args.out}\nMarkdown: {args.md_out}")
    return 0


def _cmd_multistream(args) -> int:
    from astracore.multistream import run_multistream
    import json as _json

    shape = None
    if args.input_shape:
        shape = tuple(int(s) for s in args.input_shape.split(","))

    streams = tuple(int(s) for s in args.streams.split(","))
    print(f"Measuring {args.model} on {args.backend} — "
          f"streams={streams}, duration={args.duration}s each")

    report = run_multistream(
        Path(args.model),
        backend=args.backend,
        n_streams_list=streams,
        duration_s=args.duration,
        warmup_s=args.warmup,
        input_shape=shape,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        _json.dump(report.as_dict(), fh, indent=2)
    with open(args.md_out, "w") as fh:
        fh.write(report.as_markdown() + "\n")
    print()
    print(report.as_markdown())
    print(f"\nJSON: {args.out}\nMarkdown: {args.md_out}")
    return 0


def _cmd_replay(args) -> int:
    from astracore.dataset import SyntheticDataset, replay_scene
    import json as _json

    if args.dataset == "synthetic":
        ds = SyntheticDataset(preset_name=args.preset)
    elif args.dataset == "nuscenes":
        if not args.dataroot:
            print("--dataroot required for nuscenes", file=sys.stderr)
            return 2
        try:
            from astracore.dataset import NuScenesDataset
        except ImportError as exc:
            print(f"nuscenes connector unavailable: {exc}", file=sys.stderr)
            return 2
        ds = NuScenesDataset(dataroot=args.dataroot, version=args.version)
    else:
        print(f"unknown dataset: {args.dataset}", file=sys.stderr)
        return 2

    scenes = ds.list_scenes()
    if not scenes:
        print("dataset contains no scenes", file=sys.stderr)
        return 2
    scene_id = args.scene or scenes[0]
    scene = ds.get_scene(scene_id)
    print(f"Replaying {scene_id} ({scene.name!r}, {len(scene)} samples)")
    result = replay_scene(scene)
    summary = result.summary()

    payload = {
        "dataset": ds.name,
        "scene_id": result.scene_id,
        "scene_name": result.scene_name,
        "n_samples": result.n_samples,
        "backend": result.backend,
        "wall_s_total": round(result.wall_s_total, 3),
        "summary": summary,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        _json.dump(payload, fh, indent=2)

    print(f"\nScene complete in {result.wall_s_total:.2f}s.")
    for k, v in summary.items():
        print(f"  {k:<22} {v:.2f}")
    print(f"\nJSON: {out_path}")
    return 0


def _cmd_zoo(args) -> int:
    from astracore.benchmark import benchmark_model
    from astracore import zoo as zoo_mod
    import json as _json
    import time as _time

    paths = zoo_mod.local_paths()
    if args.only:
        targets = [zoo_mod.get(n) for n in args.only]
    else:
        targets = [m for m in zoo_mod.all_models() if paths[m.name].exists()]

    if not targets:
        print("No zoo models found on disk. Run scripts/fetch_model_zoo.py first.",
              file=sys.stderr)
        return 2

    rows = []
    t_start = _time.perf_counter()
    print(f"Running {len(targets)} model(s) on backend={args.backend} "
          f"(n_iter={args.iter})")
    for m in targets:
        path = paths[m.name]
        if not path.exists():
            print(f"  [skip] {m.name}: file missing at {path}")
            continue
        try:
            # Build shape string from the zoo entry.
            shape = ",".join(str(d) for d in m.input_shape)
            rep = benchmark_model(path, backend=args.backend,
                                  precision="INT8", sparsity="dense",
                                  n_iter=args.iter,
                                  input_shape=shape, warmup=1)
            rep.model = m.name
            rep.notes = rep.notes or m.notes
            rows.append(rep.as_dict())
            print(f"  [OK]   {m.name:<28} "
                  f"{rep.wall_ms_per_inference:7.2f} ms   "
                  f"{rep.delivered_tops:6.3f} TOPS   "
                  f"{rep.mac_ops_total/1e9:6.2f} GMACs")
        except Exception as exc:
            rows.append({"model": m.name, "backend": args.backend,
                         "error": repr(exc), "family": m.family})
            print(f"  [FAIL] {m.name}: {exc!r}")

    wall = _time.perf_counter() - t_start
    payload = {
        "backend": args.backend,
        "n_models": len(rows),
        "n_iter": args.iter,
        "wall_s_total": round(wall, 3),
        "rows": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        _json.dump(payload, fh, indent=2)
    print(f"\nJSON: {out_path}")

    md_path = Path(args.md_out)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w") as fh:
        fh.write(f"# AstraCore model-zoo benchmark (backend: {args.backend})\n\n")
        fh.write(f"`n_iter={args.iter}`, wall={wall:.1f}s, "
                 f"{len(rows)} models.\n\n")
        fh.write("| Model | Latency (ms) | GMACs | Delivered TOPS | Notes |\n")
        fh.write("|---|---|---|---|---|\n")
        for r in rows:
            if "error" in r:
                fh.write(f"| {r['model']} | — | — | — | **FAIL**: `{r['error']}` |\n")
            else:
                fh.write(f"| {r['model']} | {r.get('wall_ms_per_inference', 0):.2f} "
                         f"| {r.get('mac_ops_total', 0) / 1e9:.2f} "
                         f"| {r.get('delivered_tops', 0):.3f} "
                         f"| {r.get('notes', '')} |\n")
    print(f"Markdown: {md_path}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
