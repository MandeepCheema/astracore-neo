"""Internal NPU simulator backend.

Wraps ``tools/npu_ref/nn_runtime.run_graph`` in the Backend protocol so
it shows up alongside ORT / F1 / Orin backends in benchmark output.

This is the "correctness reference" — every other backend must produce
bit-identical output to this one on the same graph + input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import time

import numpy as np

from astracore.backend import _BackendBase, BackendReport
from astracore.registry import register_backend


@dataclass
class _NpuSimProgram:
    graph: Any
    quantiser_config: Any
    total_macs: int = 0


@register_backend("npu-sim")
class NpuSimBackend(_BackendBase):
    """Runs on the reference NPU Python simulator."""

    name = "npu-sim"
    silicon_profile = "npu-ref-python"

    def compile(self, graph: Any, *, precision: str = "INT8",
                sparsity: str = "dense") -> _NpuSimProgram:
        # Graph is an NnGraph from tools.npu_ref.onnx_loader (already quantised).
        total_macs = _estimate_macs_nngraph(graph)
        self._last = BackendReport(
            backend=self.name,
            mac_ops_total=total_macs,
            silicon_profile=self.silicon_profile,
            precision=precision,
            sparsity=sparsity,
        )
        return _NpuSimProgram(graph=graph, quantiser_config=None,
                              total_macs=total_macs)

    def run(self, program: _NpuSimProgram,
            inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        from tools.npu_ref.nn_runtime import run_graph  # lazy

        t0 = time.perf_counter()
        outs = run_graph(program.graph, inputs)
        wall = time.perf_counter() - t0

        self._last.n_inferences += 1
        self._last.wall_s_total += wall
        self._last.wall_ms_per_inference = (
            self._last.wall_s_total / self._last.n_inferences * 1e3
        )
        if wall > 0:
            self._last.delivered_tops = (program.total_macs / wall) / 1e12
        self._last.mac_ops_effective = program.total_macs
        return outs


def _estimate_macs_nngraph(graph) -> int:
    """Sum MACs across Conv / MatMul nodes in an NnGraph."""
    total = 0
    for node in getattr(graph, "nodes", []):
        op = getattr(node, "op_type", None) or getattr(node, "op", None)
        if op in ("Conv", "OP_CONV", "conv"):
            attrs = getattr(node, "attrs", {}) or {}
            out_shape = attrs.get("output_shape")
            w_shape = attrs.get("weight_shape")
            if out_shape and w_shape:
                try:
                    _, M_out, H_out, W_out = out_shape
                    _, C_in, KH, KW = w_shape
                    total += M_out * C_in * KH * KW * H_out * W_out
                except Exception:
                    pass
        elif op in ("MatMul", "OP_MATMUL", "Gemm", "OP_GEMM"):
            attrs = getattr(node, "attrs", {}) or {}
            dims = attrs.get("dims") or attrs.get("mkn")
            if dims and len(dims) == 3:
                M, K, N = dims
                total += M * K * N
    return total
