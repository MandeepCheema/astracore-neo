"""Richer zoo benchmark — latency distributions, batch sweep, thread sweep.

Runs every model in the astracore zoo on the requested backend/providers
and collects:

* Warmup-vs-steady-state latency (first N runs vs median of the rest).
* Latency distribution (mean / p50 / p95 / p99 / p99.9 / max).
* Batch-size sweep for models whose first input dim is dynamic or 1.
* Thread-count sweep for ONNX Runtime (intra_op_num_threads in {1, 2, 4}).
* RSS memory delta (if ``psutil`` is installed — optional).
* Output fingerprint (SHA-256 of concatenated outputs rounded to 3 dp) —
  catches drift when swapping backends / precisions later.

Parameterised on ``--providers`` so the exact same script runs on a cloud
GPU host once CUDA / TensorRT wheels are installed. Host CPU is the
baseline ``reports/zoo_detailed/zoo_detailed.json`` committed today;
cloud runs write to a sibling directory (``reports/cloud/<tag>/...``).

Usage
-----

    python scripts/bench_zoo_detailed.py                       # defaults
    python scripts/bench_zoo_detailed.py --providers cuda,cpu  # GPU host
    python scripts/bench_zoo_detailed.py --only yolov8n --batch 1,2,4,8
    python scripts/bench_zoo_detailed.py --threads 1,2,4 --iter 30

Not promoted to ``astracore.benchmark`` yet — this is a research harness.
Once it stabilises we either promote it or fold the best bits into the
existing ``benchmark_model``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# astracore imports must come after sys.path setup when run as a script
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from astracore import zoo as zoo_mod           # noqa: E402
from astracore.backends.ort import (            # noqa: E402
    OrtBackend, _normalise_providers,
)
from astracore.benchmark import _gen_input_for  # noqa: E402


# ---------------------------------------------------------------------------
# Data classes for the JSON report
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    n_samples: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    p99p9_ms: float
    max_ms: float
    stdev_ms: float

    @staticmethod
    def from_list(ms_list: List[float]) -> "LatencyStats":
        if not ms_list:
            return LatencyStats(0, 0, 0, 0, 0, 0, 0, 0)
        arr = np.asarray(ms_list, dtype=np.float64)
        return LatencyStats(
            n_samples=len(arr),
            mean_ms=float(arr.mean()),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            p99p9_ms=float(np.percentile(arr, 99.9)),
            max_ms=float(arr.max()),
            stdev_ms=float(arr.std()) if len(arr) > 1 else 0.0,
        )


@dataclass
class ScenarioResult:
    scenario: str                          # human-readable scenario name
    backend: str
    providers_requested: List[str]
    providers_active: List[str]
    batch_size: int
    intra_op_threads: Optional[int]
    graph_opt_level: str
    warmup_ms: float                       # first-run latency
    steady: LatencyStats                   # distribution after warmup
    rss_delta_mb: Optional[float] = None
    output_fingerprint: str = ""
    notes: str = ""
    failed: bool = False
    error: str = ""


@dataclass
class ModelReport:
    model: str
    family: str
    onnx_path: str
    input_name: str
    input_shape: List[int]
    opset: int
    gmacs: float = 0.0
    scenarios: List[ScenarioResult] = field(default_factory=list)


@dataclass
class SuiteReport:
    host: Dict[str, Any]
    backend: str
    providers_requested: List[str]
    wall_s_total: float
    n_models: int
    models: List[ModelReport]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _host_info() -> Dict[str, Any]:
    info = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
        info["ort_available_providers"] = ort.get_available_providers()
    except Exception as exc:
        info["onnxruntime_error"] = repr(exc)
    try:
        info["numpy"] = np.__version__
    except Exception:
        pass
    # CPU / cores — platform module is always present.
    import os, platform
    info["cpu"] = platform.processor() or "unknown"
    info["cpu_count"] = os.cpu_count()
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:
        pass
    return info


def _current_rss_mb() -> Optional[float]:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 2
    except Exception:
        return None


def _fingerprint_outputs(outputs: Dict[str, np.ndarray]) -> str:
    """Deterministic fingerprint of an inference output.

    We round to 3 dp before hashing so tiny numerical jitter doesn't
    flip the hash, but bigger precision/algorithmic drift does.
    """
    hasher = hashlib.sha256()
    for name in sorted(outputs):
        arr = np.asarray(outputs[name])
        hasher.update(name.encode("utf-8"))
        hasher.update(str(arr.shape).encode("utf-8"))
        hasher.update(str(arr.dtype).encode("utf-8"))
        flat = arr.astype(np.float64).ravel()
        # Round AND clip — helps stabilise fingerprints when outputs
        # contain near-zero logits that flip sign run-to-run.
        rounded = np.round(flat, 3)
        hasher.update(rounded.tobytes())
    return hasher.hexdigest()[:16]


def _substitute_shape(dims, *, batch: int, seq_len: int = 8):
    """Turn ONNX dim list into concrete ints, honouring batch override."""
    out = []
    for i, d in enumerate(dims):
        if d.dim_value and d.dim_value > 0:
            out.append(int(d.dim_value))
            continue
        name = (d.dim_param or "").lower()
        if i == 0 or "batch" in name:
            out.append(batch)
        elif "seq" in name or "length" in name:
            out.append(seq_len)
        else:
            out.append(seq_len if i == 1 else 1)
    return tuple(out)


def _build_inputs(onnx_model, *, batch: int, rng):
    """Build full input dict for an ONNX model at a given batch size."""
    init_names = {t.name for t in onnx_model.graph.initializer}
    real_inputs = [inp for inp in onnx_model.graph.input
                   if inp.name not in init_names]
    inputs: Dict[str, np.ndarray] = {}
    for inp in real_inputs:
        shape = _substitute_shape(inp.type.tensor_type.shape.dim, batch=batch)
        inputs[inp.name] = _gen_input_for(inp, override_shape=shape, rng=rng)
    return inputs


def _model_supports_batch_sweep(onnx_model) -> bool:
    """True if the first input's first dim is dynamic or equal to 1."""
    init_names = {t.name for t in onnx_model.graph.initializer}
    first = next(
        (inp for inp in onnx_model.graph.input
         if inp.name not in init_names),
        None,
    )
    if first is None:
        return False
    dims = first.type.tensor_type.shape.dim
    if not dims:
        return False
    d0 = dims[0]
    if d0.dim_value == 1:
        return True
    if d0.dim_value == 0 and d0.dim_param:
        return True
    return False


# ---------------------------------------------------------------------------
# One scenario = one (providers, batch, threads, graph-opt) combo
# ---------------------------------------------------------------------------

def run_scenario(
    *, scenario_name: str,
    onnx_path: Path,
    onnx_model,                       # preloaded onnx.ModelProto
    providers: List[str],
    batch_size: int,
    intra_op_threads: Optional[int],
    graph_opt_level: str,
    warmup: int,
    n_timed: int,
) -> ScenarioResult:
    """Measure one scenario: warmup + timed runs + fingerprint + memory."""
    session_opts: Dict[str, Any] = {}
    if intra_op_threads is not None:
        session_opts["intra_op_num_threads"] = int(intra_op_threads)
    if graph_opt_level:
        try:
            import onnxruntime as ort
            lvl_map = {
                "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                "basic":   ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                "all":     ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            }
            if graph_opt_level in lvl_map:
                session_opts["graph_optimization_level"] = lvl_map[graph_opt_level]
        except Exception:
            pass

    result = ScenarioResult(
        scenario=scenario_name,
        backend="onnxruntime",
        providers_requested=list(providers),
        providers_active=[],
        batch_size=batch_size,
        intra_op_threads=intra_op_threads,
        graph_opt_level=graph_opt_level or "default",
        warmup_ms=0.0,
        steady=LatencyStats(0, 0, 0, 0, 0, 0, 0, 0),
    )

    try:
        rss_before = _current_rss_mb()
        be = OrtBackend(providers=providers or None,
                        session_options=session_opts or None)
        program = be.compile(str(onnx_path))
        result.providers_active = list(
            be.report_last().extra.get("active_providers") or []
        )

        rng = np.random.default_rng(0)
        inputs = _build_inputs(onnx_model, batch=batch_size, rng=rng)

        # Warmup — first run dominated by CUDA context init / ORT kernel
        # compile. Record it separately.
        t0 = time.perf_counter()
        out0 = be.run(program, inputs)
        warmup_ms = (time.perf_counter() - t0) * 1e3
        result.warmup_ms = warmup_ms

        # More warmup runs (unmeasured) to reach steady state.
        for _ in range(max(0, warmup - 1)):
            be.run(program, inputs)

        # Timed runs — one at a time so per-run latency is visible. Use
        # perf_counter_ns to avoid float rounding at sub-microsecond.
        ms_list: List[float] = []
        for _ in range(n_timed):
            t = time.perf_counter()
            be.run(program, inputs)
            ms_list.append((time.perf_counter() - t) * 1e3)

        result.steady = LatencyStats.from_list(ms_list)
        result.output_fingerprint = _fingerprint_outputs(out0)

        rss_after = _current_rss_mb()
        if rss_before is not None and rss_after is not None:
            result.rss_delta_mb = round(rss_after - rss_before, 1)

        # Explicit delete so a model's session doesn't leak into the
        # next scenario's RSS measurement.
        del program, be, out0
        gc.collect()
    except Exception as exc:
        result.failed = True
        result.error = f"{type(exc).__name__}: {exc}"
    return result


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_suite(
    *,
    providers: List[str],
    batch_sizes: List[int],
    thread_counts: List[Optional[int]],
    graph_opt_levels: List[str],
    warmup: int,
    n_timed: int,
    only: Optional[List[str]] = None,
) -> SuiteReport:
    import onnx

    wall_start = time.perf_counter()
    models: List[ModelReport] = []

    paths = zoo_mod.local_paths()
    targets = (
        [zoo_mod.get(n) for n in only]
        if only else
        [m for m in zoo_mod.all_models() if paths[m.name].exists()]
    )

    for m in targets:
        p = paths.get(m.name)
        if not p or not p.exists():
            continue
        print(f"\n=== {m.name} ({m.family}) ===")
        model_proto = onnx.load(str(p))
        supports_batch = _model_supports_batch_sweep(model_proto)
        effective_batches = batch_sizes if supports_batch else [1]
        if not supports_batch and batch_sizes != [1]:
            print(f"  (batch sweep skipped — first dim is static {m.input_shape[0]})")

        # GMACs — reuse OrtBackend's MAC estimator once.
        try:
            be_once = OrtBackend(providers=providers or None)
            be_once.compile(str(p))
            gmacs = round(be_once.report_last().mac_ops_total / 1e9, 3)
            del be_once
        except Exception:
            gmacs = 0.0

        mr = ModelReport(
            model=m.name,
            family=m.family,
            onnx_path=str(p),
            input_name=m.input_name,
            input_shape=list(m.input_shape),
            opset=m.opset,
            gmacs=gmacs,
        )

        # Main scenario: batch 1, default threads, default graph-opt.
        mr.scenarios.append(run_scenario(
            scenario_name="base",
            onnx_path=p, onnx_model=model_proto,
            providers=providers, batch_size=1,
            intra_op_threads=None, graph_opt_level="",
            warmup=warmup, n_timed=n_timed,
        ))
        _print_scenario(mr.scenarios[-1])

        # Thread sweep (still batch 1).
        for t in thread_counts:
            if t is None:
                continue
            s = run_scenario(
                scenario_name=f"threads={t}",
                onnx_path=p, onnx_model=model_proto,
                providers=providers, batch_size=1,
                intra_op_threads=t, graph_opt_level="",
                warmup=warmup, n_timed=n_timed,
            )
            mr.scenarios.append(s)
            _print_scenario(s)

        # Graph-opt sweep at default threads.
        for gopt in graph_opt_levels:
            if not gopt or gopt == "default":
                continue
            s = run_scenario(
                scenario_name=f"gopt={gopt}",
                onnx_path=p, onnx_model=model_proto,
                providers=providers, batch_size=1,
                intra_op_threads=None, graph_opt_level=gopt,
                warmup=warmup, n_timed=n_timed,
            )
            mr.scenarios.append(s)
            _print_scenario(s)

        # Batch sweep (for supporting models).
        for b in effective_batches:
            if b == 1:
                continue
            s = run_scenario(
                scenario_name=f"batch={b}",
                onnx_path=p, onnx_model=model_proto,
                providers=providers, batch_size=b,
                intra_op_threads=None, graph_opt_level="",
                warmup=warmup, n_timed=max(5, n_timed // 2),  # big batches cost more
            )
            mr.scenarios.append(s)
            _print_scenario(s)

        models.append(mr)

    return SuiteReport(
        host=_host_info(),
        backend="onnxruntime",
        providers_requested=list(providers),
        wall_s_total=round(time.perf_counter() - wall_start, 2),
        n_models=len(models),
        models=models,
    )


def _print_scenario(s: ScenarioResult) -> None:
    if s.failed:
        print(f"  [FAIL] {s.scenario}: {s.error}")
        return
    warm = f"warmup={s.warmup_ms:.2f}ms"
    steady = (f"p50={s.steady.p50_ms:.2f}"
              f" p99={s.steady.p99_ms:.2f}"
              f" mean={s.steady.mean_ms:.2f}"
              f" std={s.steady.stdev_ms:.2f}ms")
    rss = f" rss={s.rss_delta_mb}MB" if s.rss_delta_mb is not None else ""
    fp = f" fp={s.output_fingerprint}" if s.output_fingerprint else ""
    print(f"  [{s.scenario:<12}] {warm}  {steady}{rss}{fp}")


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def render_markdown(r: SuiteReport) -> str:
    lines: List[str] = []
    lines.append("# Detailed zoo benchmark")
    lines.append("")
    lines.append("## Host")
    lines.append("")
    lines.append(f"- Platform: {r.host.get('platform')}")
    lines.append(f"- CPU: {r.host.get('cpu')} ({r.host.get('cpu_count')} cores)")
    if "ram_gb" in r.host:
        lines.append(f"- RAM: {r.host['ram_gb']} GB")
    lines.append(f"- Python: {r.host.get('python')}")
    if "onnxruntime" in r.host:
        lines.append(f"- onnxruntime: {r.host['onnxruntime']}")
    lines.append(f"- ORT EPs available: "
                 f"{', '.join(r.host.get('ort_available_providers', []))}")
    lines.append(f"- Requested providers: "
                 f"`{', '.join(r.providers_requested) or '(default)'}`")
    lines.append(f"- Wall total: {r.wall_s_total}s over {r.n_models} models")
    lines.append("")

    # Summary table — one row per model, base scenario.
    lines.append("## Summary (base scenario: batch=1, default threads + graph-opt)")
    lines.append("")
    lines.append("| Model | Family | GMACs | warmup ms | p50 ms | p99 ms | "
                 "p99.9 ms | stdev ms | fingerprint |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for m in r.models:
        base = next((s for s in m.scenarios if s.scenario == "base"), None)
        if base is None or base.failed:
            lines.append(f"| {m.model} | {m.family} | {m.gmacs} | — | — | — | "
                         f"— | — | FAIL |")
            continue
        lines.append(
            f"| {m.model} | {m.family} | {m.gmacs} | "
            f"{base.warmup_ms:.2f} | "
            f"{base.steady.p50_ms:.2f} | {base.steady.p99_ms:.2f} | "
            f"{base.steady.p99p9_ms:.2f} | {base.steady.stdev_ms:.2f} | "
            f"`{base.output_fingerprint}` |"
        )
    lines.append("")

    # Thread sweep — shows how much headroom better threading gives.
    lines.append("## Thread-count sensitivity (p50 ms, batch=1)")
    lines.append("")
    # Collect thread counts that were actually run.
    thr_set: List[int] = []
    for m in r.models:
        for s in m.scenarios:
            if s.scenario.startswith("threads=") and not s.failed:
                t = int(s.scenario.split("=")[1])
                if t not in thr_set:
                    thr_set.append(t)
    thr_set.sort()
    if thr_set:
        header = "| Model | default | " + " | ".join(f"t={t}" for t in thr_set) + " |"
        sep = "|---|---:|" + "|".join(":---:" for _ in thr_set) + "|"
        lines.append(header)
        lines.append(sep)
        for m in r.models:
            base = next((s for s in m.scenarios if s.scenario == "base"), None)
            base_p50 = (f"{base.steady.p50_ms:.2f}"
                        if base and not base.failed else "—")
            cells = []
            for t in thr_set:
                s = next((s for s in m.scenarios
                          if s.scenario == f"threads={t}"), None)
                cells.append(f"{s.steady.p50_ms:.2f}"
                             if s and not s.failed else "—")
            lines.append(f"| {m.model} | {base_p50} | " + " | ".join(cells) + " |")
        lines.append("")

    # Batch sweep — only where supported.
    lines.append("## Batch-size scaling (p50 ms per-call, default threads)")
    lines.append("")
    batch_set: List[int] = []
    for m in r.models:
        for s in m.scenarios:
            if s.scenario.startswith("batch="):
                b = int(s.scenario.split("=")[1])
                if b not in batch_set:
                    batch_set.append(b)
    batch_set.sort()
    if batch_set:
        header = "| Model | b=1 | " + " | ".join(f"b={b}" for b in batch_set) + " |"
        sep = "|---|---:|" + "|".join(":---:" for _ in batch_set) + "|"
        lines.append(header)
        lines.append(sep)
        for m in r.models:
            base = next((s for s in m.scenarios if s.scenario == "base"), None)
            b1 = (f"{base.steady.p50_ms:.2f}"
                  if base and not base.failed else "—")
            row = [b1]
            for b in batch_set:
                s = next((s for s in m.scenarios
                          if s.scenario == f"batch={b}"), None)
                row.append(f"{s.steady.p50_ms:.2f}"
                           if s and not s.failed else "—")
            lines.append(f"| {m.model} | " + " | ".join(row) + " |")
        lines.append("")

    # Graph-opt sweep
    gopt_set: List[str] = []
    for m in r.models:
        for s in m.scenarios:
            if s.scenario.startswith("gopt="):
                g = s.scenario.split("=")[1]
                if g not in gopt_set:
                    gopt_set.append(g)
    if gopt_set:
        lines.append("## Graph-optimization level (p50 ms, batch=1)")
        lines.append("")
        header = "| Model | default | " + " | ".join(gopt_set) + " |"
        sep = "|---|---:|" + "|".join(":---:" for _ in gopt_set) + "|"
        lines.append(header)
        lines.append(sep)
        for m in r.models:
            base = next((s for s in m.scenarios if s.scenario == "base"), None)
            base_p50 = (f"{base.steady.p50_ms:.2f}"
                        if base and not base.failed else "—")
            cells = []
            for g in gopt_set:
                s = next((s for s in m.scenarios
                          if s.scenario == f"gopt={g}"), None)
                cells.append(f"{s.steady.p50_ms:.2f}"
                             if s and not s.failed else "—")
            lines.append(f"| {m.model} | {base_p50} | " + " | ".join(cells) + " |")
        lines.append("")

    # Tail-latency characterisation (max / p99 over mean ratio)
    lines.append("## Tail-latency ratios (base scenario)")
    lines.append("")
    lines.append("Ratio of p99 to mean — how bursty the backend is. "
                 "Stable runtimes are close to 1.0; jittery systems tail into 2-5×.")
    lines.append("")
    lines.append("| Model | mean ms | p99 ms | p99/mean | max/mean |")
    lines.append("|---|---:|---:|---:|---:|")
    for m in r.models:
        base = next((s for s in m.scenarios if s.scenario == "base"), None)
        if not base or base.failed or base.steady.mean_ms == 0:
            continue
        r99 = base.steady.p99_ms / base.steady.mean_ms
        rmax = base.steady.max_ms / base.steady.mean_ms
        lines.append(
            f"| {m.model} | {base.steady.mean_ms:.2f} | "
            f"{base.steady.p99_ms:.2f} | {r99:.2f}× | {rmax:.2f}× |"
        )
    lines.append("")

    # RSS memory (where we have it)
    rss_rows = [(m.model, base)
                for m in r.models
                for base in [next((s for s in m.scenarios if s.scenario == "base"),
                                  None)]
                if base and not base.failed and base.rss_delta_mb is not None]
    if rss_rows:
        lines.append("## RSS memory delta (base scenario)")
        lines.append("")
        lines.append("Delta from before-compile to after-first-run — includes "
                     "session + weights + ORT intermediate buffers.")
        lines.append("")
        lines.append("| Model | ΔRSS MB |")
        lines.append("|---|---:|")
        for name, s in rss_rows:
            lines.append(f"| {name} | {s.rss_delta_mb} |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--providers", default="",
                   help="comma-separated EP names or short aliases (cuda,cpu)")
    p.add_argument("--only", default=None,
                   help="comma-separated subset of zoo model names")
    p.add_argument("--batch", default="1",
                   help="comma-separated batch sizes (e.g. 1,2,4)")
    p.add_argument("--threads", default="",
                   help="comma-separated intra_op_num_threads values (e.g. 1,2,4)")
    p.add_argument("--gopt", default="basic,extended,all",
                   help="comma-separated graph_optimization_level values "
                        "(disable,basic,extended,all)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iter", type=int, default=20,
                   help="timed iterations after warmup")
    p.add_argument("--out", default="reports/zoo_detailed/zoo_detailed.json")
    p.add_argument("--md-out", default="reports/zoo_detailed/zoo_detailed.md")
    args = p.parse_args()

    providers = [s.strip() for s in args.providers.split(",") if s.strip()]
    batch_sizes = [int(x) for x in args.batch.split(",") if x.strip()]
    thread_counts = ([int(x) for x in args.threads.split(",") if x.strip()]
                     if args.threads else [])
    gopt_levels = [x.strip() for x in args.gopt.split(",") if x.strip()]
    only = ([x.strip() for x in args.only.split(",") if x.strip()]
            if args.only else None)

    print(f"Providers: {providers or '(default)'}")
    print(f"Batches:  {batch_sizes}")
    print(f"Threads:  {thread_counts or '(default only)'}")
    print(f"Gopt:     {gopt_levels}")
    print(f"Warmup/iter: {args.warmup}/{args.iter}")

    report = run_suite(
        providers=providers,
        batch_sizes=batch_sizes,
        thread_counts=thread_counts,
        graph_opt_levels=gopt_levels,
        warmup=args.warmup,
        n_timed=args.iter,
        only=only,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(report), indent=2, default=str))
    print(f"\nJSON: {out_path}")
    md_path = Path(args.md_out)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(report))
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
