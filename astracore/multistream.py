"""Multi-stream throughput measurement — the biggest software TOPS lever.

Single-stream inference on any silicon wastes MACs on memory stalls,
kernel-launch gaps, and serial dependencies. Running multiple
independent inference streams concurrently overlaps that idle time
across streams, pushing aggregate MAC utilisation from ~5–10 %
(single-stream) toward 70–95 % (well-saturated multi-stream).

For the spec-sheet target of ">90 % MAC utilisation", this is THE
knob — bigger lever than any individual kernel optimisation, bigger
than sparsity, bigger than lower-precision arithmetic. Worth measuring.

Automotive context: L2+/L4 ADAS typically has 4-11 cameras, each
feeding >= 1 model. A 4-camera × 2-model stack = 8 concurrent
inference streams. That's the realistic multi-stream shape.

This module:
 - Picks a model + backend, benchmarks N = {1, 2, 4, 8} concurrent
   streams for a fixed wall-clock budget.
 - Reports aggregate throughput (inferences/s), aggregate effective
   TOPS, per-stream latency, scaling factor vs single-stream.
 - Results land in a JSON/Markdown report plus programmatic return.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading
import time

import numpy as np


@dataclass
class StreamSlice:
    """One (n_streams, model) measurement."""
    n_streams: int
    n_inferences_total: int
    wall_s: float
    throughput_ips: float           # inferences / second, aggregate
    aggregate_tops: float            # effective MACs × throughput, ÷ 1e12
    mean_latency_ms: float           # mean per-inference latency
    p50_latency_ms: float
    p99_latency_ms: float
    scaling_vs_single: float         # throughput / single-stream throughput
    util_vs_single: float            # aggregate_tops / single-stream TOPS


@dataclass
class MultiStreamReport:
    model: str
    backend: str
    mac_ops_per_inference: int
    warmup_s: float
    duration_s_per_slice: float
    slices: List[StreamSlice] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def as_markdown(self) -> str:
        lines = [
            f"# Multi-stream scaling — {self.model} on {self.backend}",
            "",
            f"`{self.mac_ops_per_inference/1e9:.2f} GMACs/inference`, "
            f"`{self.duration_s_per_slice}s` per data point, "
            f"`{self.warmup_s}s` warmup.",
            "",
            "| Streams | IPS | TOPS (agg.) | Latency p50 (ms) | "
            "Latency p99 (ms) | Scale vs 1× | Util vs 1× |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for s in self.slices:
            lines.append(
                f"| {s.n_streams} | {s.throughput_ips:.1f} | "
                f"{s.aggregate_tops:.3f} | {s.p50_latency_ms:.1f} | "
                f"{s.p99_latency_ms:.1f} | "
                f"{s.scaling_vs_single:.2f}× | {s.util_vs_single:.2f}× |"
            )
        return "\n".join(lines)


def _run_stream(run_fn, program, inputs,
                stop_flag: threading.Event,
                latencies: List[float]) -> int:
    """Hammer the backend from one thread until stop_flag is set.

    ``latencies`` is a thread-local list (one per worker); we don't share
    a single list across threads to avoid relying on CPython's GIL for
    list.append atomicity.
    """
    n = 0
    while not stop_flag.is_set():
        t0 = time.perf_counter()
        run_fn(program, inputs)
        latencies.append((time.perf_counter() - t0) * 1e3)
        n += 1
    return n


def run_multistream(
    model_path: Path,
    *,
    backend: str = "onnxruntime",
    backend_options: Optional[Dict[str, Any]] = None,
    n_streams_list: Tuple[int, ...] = (1, 2, 4, 8),
    duration_s: float = 5.0,
    warmup_s: float = 1.0,
    input_shape: Optional[Tuple[int, ...]] = None,
) -> MultiStreamReport:
    """Run the multi-stream scaling measurement.

    For ``onnxruntime``, the InferenceSession is shared across worker
    threads because ORT releases the GIL during ``run()`` — this gives
    real parallelism on CPU/GPU without session-cloning overhead.
    """
    # Make built-in backends register.
    import astracore.backends  # noqa: F401
    from astracore.registry import get_backend

    # Load model + set up the backend once.
    import onnx
    model_path = Path(model_path)
    onnx_model = onnx.load(str(model_path))

    from astracore.benchmark import _gen_input_for
    rng = np.random.default_rng(0)
    init_names = {t.name for t in onnx_model.graph.initializer}
    real_inputs = [inp for inp in onnx_model.graph.input if inp.name not in init_names]
    inputs: Dict[str, np.ndarray] = {}
    for i, inp in enumerate(real_inputs):
        # Honour caller's input_shape override for the first input only.
        override = input_shape if (i == 0 and input_shape is not None) else None
        inputs[inp.name] = _gen_input_for(inp, override_shape=override, rng=rng)

    backend_cls = get_backend(backend)
    if isinstance(backend_cls, type):
        be = backend_cls(**(backend_options or {}))
    else:
        if backend_options:
            raise ValueError(
                f"backend {backend!r} is not a class; cannot pass options"
            )
        be = backend_cls
    concrete_shapes = {name: tuple(arr.shape) for name, arr in inputs.items()}
    try:
        program = be.compile(onnx_model, precision="INT8", sparsity="dense",
                             concrete_shapes=concrete_shapes)
    except TypeError:
        program = be.compile(onnx_model, precision="INT8", sparsity="dense")

    # Pull MAC count off the backend's report.
    mac_ops = be.report_last().mac_ops_total

    report = MultiStreamReport(
        model=model_path.name,
        backend=backend,
        mac_ops_per_inference=mac_ops,
        warmup_s=warmup_s,
        duration_s_per_slice=duration_s,
    )

    # Warmup once (amortised across streams — a cold first run biases slices).
    t_warm_end = time.perf_counter() + warmup_s
    while time.perf_counter() < t_warm_end:
        be.run(program, inputs)

    single_stream_tps = None
    single_stream_tops = None
    for n_streams in n_streams_list:
        stop_flag = threading.Event()
        thread_latencies: List[List[float]] = [[] for _ in range(n_streams)]

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_streams) as pool:
            futures = [
                pool.submit(_run_stream, be.run, program, inputs,
                            stop_flag, thread_latencies[i])
                for i in range(n_streams)
            ]
            # Let them run.
            time.sleep(duration_s)
            stop_flag.set()
            counts = [f.result() for f in as_completed(futures)]
        wall = time.perf_counter() - t0

        n_total = sum(counts)
        all_lat = [ms for lst in thread_latencies for ms in lst]

        throughput = n_total / wall
        agg_tops = (n_total * mac_ops / wall) / 1e12 if wall > 0 else 0.0

        if single_stream_tps is None:
            single_stream_tps = throughput
            single_stream_tops = agg_tops

        scaling = throughput / single_stream_tps if single_stream_tps else 1.0
        util = agg_tops / single_stream_tops if single_stream_tops else 1.0

        report.slices.append(StreamSlice(
            n_streams=n_streams,
            n_inferences_total=n_total,
            wall_s=wall,
            throughput_ips=throughput,
            aggregate_tops=agg_tops,
            mean_latency_ms=float(np.mean(all_lat)) if all_lat else 0.0,
            p50_latency_ms=float(np.percentile(all_lat, 50)) if all_lat else 0.0,
            p99_latency_ms=float(np.percentile(all_lat, 99)) if all_lat else 0.0,
            scaling_vs_single=scaling,
            util_vs_single=util,
        ))

    return report
