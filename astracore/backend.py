"""Backend protocol — the contract every target silicon implements.

A ``Backend`` takes a quantised ``NnGraph`` and produces:
  * compiled program (opaque to the SDK; target-specific)
  * a ``run(inputs)`` callable that executes the program
  * a ``report()`` with standardised KPIs so benchmark output is
    comparable across backends (CPU vs F1 vs Orin vs custom silicon).

Built-in backends:
  * ``"npu-sim"`` — the internal reference simulator (today's default)
  * ``"onnxruntime"`` — CPU / CUDA / TensorRT via ONNX Runtime EP
    (covers Orin, x86, generic fallback; lets OEMs run today)

Planned / via entry-points:
  * ``"f1-xrt"`` — AWS F1 Xilinx VU9P via XRT (Phase B)
  * ``"tensorrt"`` — direct TensorRT plan file (customer-specific tuning)
  * ``"snpe"`` — Qualcomm SNPE / QNN
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import time

import numpy as np


@dataclass
class BackendReport:
    """Standardised KPIs for one inference run or benchmark pass."""

    backend: str
    model: str = ""
    n_inferences: int = 0
    wall_s_total: float = 0.0
    wall_ms_per_inference: float = 0.0
    mac_ops_total: int = 0          # theoretical ops for the workload
    mac_ops_effective: int = 0       # ops actually issued (post-sparsity)
    mac_util: float = 0.0            # 0.0-1.0 fraction of peak achieved
    delivered_tops: float = 0.0      # measured ops/s ÷ 1e12
    delivered_tops_per_watt: float = 0.0
    watts_avg: Optional[float] = None  # only populated when a power source exists
    silicon_profile: str = ""        # "f1-vu9p-250MHz", "orin-275tops", etc.
    precision: str = "INT8"
    sparsity: str = "dense"
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def as_markdown_row(self) -> str:
        return (
            f"| {self.model} | {self.backend} | {self.precision} | "
            f"{self.sparsity} | {self.mac_util:.1%} | "
            f"{self.wall_ms_per_inference:.2f} | "
            f"{self.delivered_tops:.3f} | "
            f"{self.delivered_tops_per_watt:.3f} |"
        )

    @staticmethod
    def markdown_header() -> str:
        return (
            "| Model | Backend | Precision | Sparsity | MAC Util | "
            "Latency (ms) | TOPS | TOPS/W |\n"
            "|---|---|---|---|---|---|---|---|"
        )


@runtime_checkable
class Backend(Protocol):
    """Protocol every backend satisfies.

    Intentionally minimal — lets a backend be a class, a function
    returning a closure, or a thin wrapper around a third-party runtime.
    """

    name: str

    def compile(self, graph: Any, *, precision: str = "INT8",
                sparsity: str = "dense") -> Any:
        """Take a quantised ``NnGraph``; return an opaque compiled program."""
        ...

    def run(self, program: Any, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Execute a compiled program on input tensors."""
        ...

    def report_last(self) -> BackendReport:
        """Return KPIs from the most recent ``run`` / benchmark pass."""
        ...


class _BackendBase:
    """Optional convenience base class. Backends can also implement the
    Protocol directly without inheriting from here."""

    name: str = "base"
    silicon_profile: str = ""

    def __init__(self) -> None:
        self._last = BackendReport(backend=self.name,
                                   silicon_profile=self.silicon_profile)

    def compile(self, graph: Any, *, precision: str = "INT8",
                sparsity: str = "dense") -> Any:
        raise NotImplementedError

    def run(self, program: Any, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def report_last(self) -> BackendReport:
        return self._last

    def _time_run(self, program, inputs, *, mac_ops: int = 0,
                  effective_ops: Optional[int] = None,
                  precision: str = "INT8",
                  sparsity: str = "dense",
                  model_name: str = "") -> Dict[str, np.ndarray]:
        """Helper for subclasses: times ``_run_impl`` and fills report."""
        eff_ops = effective_ops if effective_ops is not None else mac_ops
        t0 = time.perf_counter()
        out = self._run_impl(program, inputs)
        wall = time.perf_counter() - t0

        delivered = (eff_ops / wall) / 1e12 if wall > 0 else 0.0
        self._last = BackendReport(
            backend=self.name,
            model=model_name,
            n_inferences=1,
            wall_s_total=wall,
            wall_ms_per_inference=wall * 1e3,
            mac_ops_total=mac_ops,
            mac_ops_effective=eff_ops,
            # mac_util is per-backend; a simulator or real-silicon backend
            # fills this in from its own counters. Leave 0 as the default.
            mac_util=0.0,
            delivered_tops=delivered,
            silicon_profile=self.silicon_profile,
            precision=precision,
            sparsity=sparsity,
        )
        return out

    def _run_impl(self, program, inputs):
        raise NotImplementedError
