"""``astracore.mlperf`` — MLCommons-loadgen-compatible scenario runner.

Runs an astracore Backend under the four MLPerf Inference scenarios:

* **SingleStream** — one query at a time; measures 90th-percentile latency.
* **MultiStream** — N concurrent queries; measures 99th-percentile.
* **Offline**    — submit a large batch; measures throughput (QPS).
* **Server**     — Poisson arrivals; measures QPS at a target latency SLA.

Status
------
**CPU-runnable today — submission-gated on cloud silicon.**

v0.1 ships an MLPerf-shaped harness with compliant timing + percentile
math but does NOT currently link against the upstream ``mlperf_loadgen``
pybind binding (that needs CUDA + a real submission rig). Drop-in for
the upstream loadgen is a 2-day port when we have a target device.

The harness is intentionally pure Python + numpy so the same script
runs on any host the SDK supports — CPU today, CUDA / TensorRT / QNN
the moment the matching ORT wheel is installed on a cloud host.

Example::

    from astracore.backends.ort import OrtBackend
    from astracore.mlperf import run_scenario, Scenario

    be = OrtBackend(providers=["cpu"])
    report = run_scenario(
        backend=be, model_path="data/models/yolov8n.onnx",
        scenario=Scenario.SINGLE_STREAM,
        duration_s=60.0,
    )
    print(report.as_markdown())

CLI::

    astracore mlperf --model data/models/yolov8n.onnx \\
        --scenario single_stream --duration 60
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class Scenario(str, Enum):
    SINGLE_STREAM = "single_stream"
    MULTI_STREAM  = "multi_stream"
    OFFLINE       = "offline"
    SERVER        = "server"


# Target latency SLAs per scenario, in milliseconds. These match MLPerf
# Inference v4.1 Edge defaults for BERT/ResNet; tighter gates ship
# in the full loadgen binary.
_DEFAULT_SLA_MS = {
    Scenario.SINGLE_STREAM: None,      # reported, not gated
    Scenario.MULTI_STREAM:  None,
    Scenario.OFFLINE:       None,
    Scenario.SERVER:        100.0,     # 100 ms p99 default
}


@dataclass
class ScenarioReport:
    scenario: str
    model: str
    backend: str
    n_queries: int
    wall_s: float
    throughput_qps: float
    latency_stats_ms: Dict[str, float]    # mean / p50 / p90 / p95 / p99 / p99.9 / max
    sla_ms: Optional[float] = None
    sla_pass: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def as_markdown(self) -> str:
        s = self.latency_stats_ms
        lines = [
            f"# MLPerf-style scenario: `{self.scenario}`  —  {self.model}",
            "",
            f"- Backend: `{self.backend}`",
            f"- N queries:   {self.n_queries}",
            f"- Wall:        {self.wall_s:.2f} s",
            f"- Throughput:  **{self.throughput_qps:.2f} QPS**",
            "",
            "| Stat | ms |",
            "|---|---:|",
            f"| mean   | {s.get('mean_ms', 0):.2f} |",
            f"| p50    | {s.get('p50_ms', 0):.2f} |",
            f"| p90    | {s.get('p90_ms', 0):.2f} |",
            f"| p95    | {s.get('p95_ms', 0):.2f} |",
            f"| p99    | {s.get('p99_ms', 0):.2f} |",
            f"| p99.9  | {s.get('p99p9_ms', 0):.2f} |",
            f"| max    | {s.get('max_ms', 0):.2f} |",
        ]
        if self.sla_ms is not None:
            v = "PASS" if self.sla_pass else "FAIL"
            lines.append("")
            lines.append(f"SLA (p99 ≤ {self.sla_ms} ms): **{v}**")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stats_from_ms(lat_ms: List[float]) -> Dict[str, float]:
    if not lat_ms:
        return {}
    arr = np.asarray(lat_ms, dtype=np.float64)
    return {
        "mean_ms":   float(arr.mean()),
        "p50_ms":    float(np.percentile(arr, 50)),
        "p90_ms":    float(np.percentile(arr, 90)),
        "p95_ms":    float(np.percentile(arr, 95)),
        "p99_ms":    float(np.percentile(arr, 99)),
        "p99p9_ms":  float(np.percentile(arr, 99.9)),
        "max_ms":    float(arr.max()),
        "stdev_ms":  float(arr.std()) if len(arr) > 1 else 0.0,
    }


def _build_seed_input(onnx_path: str) -> Dict[str, np.ndarray]:
    """Construct one plausible input sample per graph input.

    Reuses ``astracore.benchmark._gen_input_for`` so the shape + dtype
    handling matches the rest of the SDK (transformer int64 tokens,
    vision FP32 tensors, static-shape models, etc.).
    """
    import onnx
    from astracore.benchmark import _gen_input_for
    model = onnx.load(onnx_path)
    init_names = {t.name for t in model.graph.initializer}
    real = [i for i in model.graph.input if i.name not in init_names]
    rng = np.random.default_rng(0)
    return {inp.name: _gen_input_for(inp, override_shape=None, rng=rng)
            for inp in real}


# ---------------------------------------------------------------------------
# Per-scenario runners
# ---------------------------------------------------------------------------

def _run_single_stream(*, run_fn: Callable, program, inputs,
                       duration_s: float) -> Dict[str, Any]:
    """One query at a time until duration_s elapses."""
    lat: List[float] = []
    t_end = time.perf_counter() + duration_s
    while time.perf_counter() < t_end:
        t = time.perf_counter()
        run_fn(program, inputs)
        lat.append((time.perf_counter() - t) * 1e3)
    return {"latencies_ms": lat, "wall_s": duration_s,
            "n_queries": len(lat)}


def _run_multi_stream(*, run_fn: Callable, program, inputs,
                      n_streams: int, duration_s: float) -> Dict[str, Any]:
    """N concurrent streams hammering the same program."""
    lat_lock = threading.Lock()
    lat: List[float] = []
    stop_flag = threading.Event()

    def worker():
        while not stop_flag.is_set():
            t = time.perf_counter()
            run_fn(program, inputs)
            dt_ms = (time.perf_counter() - t) * 1e3
            with lat_lock:
                lat.append(dt_ms)

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_streams) as pool:
        futures = [pool.submit(worker) for _ in range(n_streams)]
        time.sleep(duration_s)
        stop_flag.set()
        for f in futures:
            f.result()
    wall = time.perf_counter() - t_start
    return {"latencies_ms": lat, "wall_s": wall, "n_queries": len(lat)}


def _run_offline(*, run_fn: Callable, program, inputs,
                 n_samples: int) -> Dict[str, Any]:
    """Submit n_samples one after another; measure aggregate throughput."""
    lat: List[float] = []
    t_start = time.perf_counter()
    for _ in range(n_samples):
        t = time.perf_counter()
        run_fn(program, inputs)
        lat.append((time.perf_counter() - t) * 1e3)
    wall = time.perf_counter() - t_start
    return {"latencies_ms": lat, "wall_s": wall, "n_queries": n_samples}


def _run_server(*, run_fn: Callable, program, inputs,
                qps: float, duration_s: float,
                n_workers: int = 4) -> Dict[str, Any]:
    """Poisson arrivals at target QPS; workers consume from a queue.

    MLPerf's Server scenario shows the rate at which the system keeps
    p99 latency below the SLA; we run at a fixed target QPS and report
    the resulting latency distribution.
    """
    import queue

    rng = np.random.default_rng(0)
    work_q: "queue.Queue" = queue.Queue()
    lat: List[float] = []
    lat_lock = threading.Lock()
    stop_flag = threading.Event()

    def worker():
        while True:
            item = work_q.get()
            if item is None:
                work_q.task_done()
                return
            submit_t = item
            t0 = time.perf_counter()
            run_fn(program, inputs)
            wait = t0 - submit_t
            serve = time.perf_counter() - t0
            with lat_lock:
                lat.append((wait + serve) * 1e3)
            work_q.task_done()

    workers = [threading.Thread(target=worker, daemon=True)
               for _ in range(n_workers)]
    for w in workers:
        w.start()

    t_end = time.perf_counter() + duration_s
    submitted = 0
    next_arrival = time.perf_counter()
    while time.perf_counter() < t_end:
        sleep_for = next_arrival - time.perf_counter()
        if sleep_for > 0:
            time.sleep(min(sleep_for, 0.1))
        if time.perf_counter() >= next_arrival:
            work_q.put(time.perf_counter())
            submitted += 1
            # Inter-arrival times are exponential with mean 1/qps.
            next_arrival += rng.exponential(1.0 / qps)

    # Drain + shutdown.
    for _ in workers:
        work_q.put(None)
    for w in workers:
        w.join(timeout=duration_s)
    return {"latencies_ms": lat, "wall_s": duration_s,
            "n_queries": submitted}


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def run_scenario(*, backend, model_path: str,
                 scenario: Scenario,
                 duration_s: float = 30.0,
                 n_samples: int = 1024,
                 n_streams: int = 4,
                 server_qps: float = 10.0,
                 sla_ms: Optional[float] = None) -> ScenarioReport:
    """Run a single MLPerf-style scenario. Returns a typed report.

    The caller owns the Backend — pass in a pre-compiled backend so the
    session-init cost doesn't pollute the numbers.
    """
    program = backend.compile(model_path)
    inputs = _build_seed_input(model_path)

    # Warmup — 3 queries to let ORT compile kernels.
    for _ in range(3):
        backend.run(program, inputs)

    if scenario is Scenario.SINGLE_STREAM:
        stats = _run_single_stream(run_fn=backend.run, program=program,
                                   inputs=inputs, duration_s=duration_s)
    elif scenario is Scenario.MULTI_STREAM:
        stats = _run_multi_stream(run_fn=backend.run, program=program,
                                  inputs=inputs, n_streams=n_streams,
                                  duration_s=duration_s)
    elif scenario is Scenario.OFFLINE:
        stats = _run_offline(run_fn=backend.run, program=program,
                             inputs=inputs, n_samples=n_samples)
    elif scenario is Scenario.SERVER:
        stats = _run_server(run_fn=backend.run, program=program,
                            inputs=inputs, qps=server_qps,
                            duration_s=duration_s,
                            n_workers=n_streams)
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    lat_stats = _stats_from_ms(stats["latencies_ms"])
    effective_sla = (sla_ms if sla_ms is not None
                     else _DEFAULT_SLA_MS.get(scenario))
    sla_pass: Optional[bool] = None
    if effective_sla is not None and lat_stats.get("p99_ms"):
        sla_pass = lat_stats["p99_ms"] <= effective_sla

    throughput = (stats["n_queries"] / stats["wall_s"]
                  if stats["wall_s"] > 0 else 0.0)

    return ScenarioReport(
        scenario=scenario.value,
        model=str(Path(model_path).name),
        backend=getattr(backend, "name", "unknown"),
        n_queries=stats["n_queries"],
        wall_s=round(stats["wall_s"], 3),
        throughput_qps=round(throughput, 3),
        latency_stats_ms={k: round(v, 3) for k, v in lat_stats.items()},
        sla_ms=effective_sla,
        sla_pass=sla_pass,
        extra={"n_streams": n_streams if scenario in
               (Scenario.MULTI_STREAM, Scenario.SERVER) else 1,
               "target_qps": server_qps
               if scenario is Scenario.SERVER else None},
    )


def run_all_scenarios(*, backend, model_path: str,
                      duration_s: float = 10.0,
                      **kw) -> Dict[str, ScenarioReport]:
    """Convenience: run all four scenarios in sequence."""
    out: Dict[str, ScenarioReport] = {}
    for sc in Scenario:
        out[sc.value] = run_scenario(
            backend=backend, model_path=model_path,
            scenario=sc, duration_s=duration_s, **kw,
        )
    return out
