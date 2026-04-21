"""ONNX Runtime backend — default fallback + multi-EP façade.

Runs a model via ``onnxruntime`` using the best available execution
provider (CPU, CUDA, TensorRT, DirectML, OpenVINO, QNN, CoreML).
Lets an OEM get a baseline number on any machine in five minutes
while the project-specific backends (F1, SNPE, custom silicon) are
still being written.

Multi-EP façade
---------------
Supply ``providers`` as:

  * a list of short names — ``["cuda", "cpu"]`` — expanded to
    ``"CUDAExecutionProvider"`` etc.
  * a list of full names — ``["CUDAExecutionProvider", "CPUExecutionProvider"]``
  * a list of ``(name, options_dict)`` tuples, honouring ORT's native
    per-EP option format — e.g. ``[("TensorrtExecutionProvider",
    {"trt_max_workspace_size": 2**30}), "CPUExecutionProvider"]``.

Requested providers that aren't available on this host are dropped
with a warning; the fallback chain always ends at
``CPUExecutionProvider`` so the backend still runs.

Reports wall-clock latency and effective TOPS (total MACs ÷ wall time).
``mac_util`` is left at 0.0 because we can't measure host CPU/GPU peak
FLOPs honestly from Python — TOPS is a count of work, not a claim of
target-silicon utilisation. Target-silicon backends (F1, NPU-sim) fill
in ``mac_util`` from hardware counters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import logging
import time

import numpy as np

from astracore.backend import _BackendBase, BackendReport
from astracore.registry import register_backend

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Short-name → full EP name table. Every short name maps to exactly one
# ORT EP; multi-option tuples stay as-is. Unknown names pass through
# verbatim, so "CUDAExecutionProvider" or a vendor-custom name also work.
# ---------------------------------------------------------------------------

EP_ALIASES: Dict[str, str] = {
    "cpu":      "CPUExecutionProvider",
    "cuda":     "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "trt":      "TensorrtExecutionProvider",
    "dml":      "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "ov":       "OpenVINOExecutionProvider",
    "qnn":      "QNNExecutionProvider",
    "coreml":   "CoreMLExecutionProvider",
    "rocm":     "ROCMExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "vitisai":  "VitisAIExecutionProvider",
    "cann":     "CANNExecutionProvider",
    "webgpu":   "WebGpuExecutionProvider",
    "xnnpack":  "XnnpackExecutionProvider",
    "nnapi":    "NnapiExecutionProvider",
    "acl":      "ACLExecutionProvider",
    "armnn":    "ArmNNExecutionProvider",
    "tidl":     "TIDLExecutionProvider",
}


def _expand_alias(name: str) -> str:
    """Accept ``cuda`` / ``CUDAExecutionProvider`` / anything. Returns the
    canonical ORT provider name; unknown strings pass through so vendor-
    custom providers still work."""
    key = name.strip().lower()
    return EP_ALIASES.get(key, name)


ProviderSpec = Union[str, Tuple[str, Dict[str, Any]]]


def _normalise_providers(
    requested: Sequence[ProviderSpec],
    available: Sequence[str],
) -> List[Union[str, Tuple[str, Dict[str, Any]]]]:
    """Expand short names, filter to available, always end at CPU.

    Parameters
    ----------
    requested : short / full names or ``(name, options)`` tuples
    available : output of ``onnxruntime.get_available_providers()``

    Returns
    -------
    A list ORT accepts directly. Missing EPs are dropped with a
    warning; CPU is appended if not already present so the session
    always has a runnable provider.
    """
    out: List[Union[str, Tuple[str, Dict[str, Any]]]] = []
    seen: set[str] = set()
    avail_set = set(available)

    for entry in requested:
        if isinstance(entry, tuple):
            raw_name, opts = entry[0], dict(entry[1] or {})
            full = _expand_alias(raw_name)
            spec: Union[str, Tuple[str, Dict[str, Any]]] = (full, opts)
        else:
            full = _expand_alias(str(entry))
            spec = full

        if full in seen:
            continue
        if full not in avail_set:
            _log.warning(
                "Execution provider %r not available on this host "
                "(this onnxruntime build ships: %s). Dropping.",
                full, ", ".join(sorted(avail_set)),
            )
            continue
        seen.add(full)
        out.append(spec)

    # Always end at CPU — ORT's guaranteed-present EP.
    if "CPUExecutionProvider" not in seen and "CPUExecutionProvider" in avail_set:
        out.append("CPUExecutionProvider")
    return out


@dataclass
class _OrtProgram:
    session: Any           # onnxruntime.InferenceSession
    input_names: List[str]
    input_shapes: Dict[str, tuple]
    total_macs: int


@register_backend("onnxruntime")
class OrtBackend(_BackendBase):
    """Runs via ``onnxruntime``'s InferenceSession.

    Parameters
    ----------
    providers : optional list of short / full / (name, opts) specs
    session_options : dict fed into ``ort.SessionOptions()`` (e.g.
        ``{"graph_optimization_level": "all", "intra_op_num_threads": 4}``)
    """

    name = "onnxruntime"
    silicon_profile = "host-cpu-or-gpu"

    def __init__(self, providers: Optional[Sequence[ProviderSpec]] = None,
                 session_options: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._providers = list(providers) if providers else None
        self._session_options = dict(session_options or {})

    def compile(self, graph: Any, *, precision: str = "INT8",
                sparsity: str = "dense",
                concrete_shapes: Optional[Dict[str, tuple]] = None) -> _OrtProgram:
        # ``graph`` here is an ONNX bytes / path / ``onnx.ModelProto``. The
        # OrtBackend skips our internal IR and feeds the original model
        # directly to onnxruntime — this is the "compare against upstream
        # ORT" baseline.
        #
        # ``concrete_shapes`` is optional — when the caller knows the
        # shapes of dynamic inputs (typical for transformers), pass them
        # so the MAC estimator can resolve MatMul output dims via
        # shape-inference.
        import onnxruntime as ort  # lazy

        if hasattr(graph, "SerializeToString"):
            onnx_bytes = graph.SerializeToString()
        elif isinstance(graph, (bytes, bytearray)):
            onnx_bytes = bytes(graph)
        elif isinstance(graph, str):
            with open(graph, "rb") as fh:
                onnx_bytes = fh.read()
        else:
            raise TypeError(
                f"OrtBackend expects an onnx.ModelProto, bytes, or path; "
                f"got {type(graph).__name__}"
            )

        available = ort.get_available_providers()
        if self._providers:
            providers = _normalise_providers(self._providers, available)
            if not providers:
                # Empty after filtering — rare but possible on a
                # non-CPU build where only an unsupported EP was asked
                # for. Fall back to whatever ORT knows about.
                providers = list(available)
        else:
            # No caller preference: use everything ORT built with. ORT
            # already routes per-node to the best available EP.
            providers = list(available)

        session_options = None
        if self._session_options:
            session_options = ort.SessionOptions()
            for k, v in self._session_options.items():
                if hasattr(session_options, k):
                    setattr(session_options, k, v)
                else:
                    _log.warning("Unknown session option %r — ignoring", k)

        session = ort.InferenceSession(
            onnx_bytes,
            sess_options=session_options,
            providers=providers,
        )
        input_names = [i.name for i in session.get_inputs()]
        input_shapes = {i.name: tuple(d if isinstance(d, int) else 1
                                      for d in i.shape)
                        for i in session.get_inputs()}
        total_macs = _estimate_macs_onnx(onnx_bytes,
                                         concrete_shapes=concrete_shapes)

        # Ask the session which EPs ORT actually used — accounts for
        # partial fallback (e.g. a TRT-unfriendly node routed to CUDA).
        try:
            active_providers = session.get_providers()
        except Exception:
            active_providers = [p if isinstance(p, str) else p[0]
                                for p in providers]

        self._last = BackendReport(
            backend=self.name,
            n_inferences=0,
            mac_ops_total=total_macs,
            silicon_profile=f"ort-{'+'.join(active_providers)}",
            precision=precision,
            sparsity=sparsity,
            notes="ORT baseline; not a target-silicon measurement",
            extra={"active_providers": active_providers},
        )
        return _OrtProgram(session=session, input_names=input_names,
                           input_shapes=input_shapes,
                           total_macs=total_macs)

    def run(self, program: _OrtProgram,
            inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        t0 = time.perf_counter()
        outs = program.session.run(None, inputs)
        wall = time.perf_counter() - t0

        out_map = {o.name: v for o, v in
                   zip(program.session.get_outputs(), outs)}

        # Fill the report. MAC util is a ratio against theoretical ops;
        # ORT doesn't tell us real peak FLOPs so we leave mac_util=0
        # (it's not comparable across hardware, which is the honest answer).
        self._last.n_inferences += 1
        self._last.wall_s_total += wall
        self._last.wall_ms_per_inference = (
            self._last.wall_s_total / self._last.n_inferences * 1e3
        )
        if wall > 0:
            self._last.delivered_tops = (program.total_macs / wall) / 1e12
        self._last.mac_ops_effective = program.total_macs
        return out_map


# ---------------------------------------------------------------------------
# Tiny ONNX MAC-estimator. Counts MACs for common ops (Conv, MatMul, Gemm).
# Undercounts non-MAC ops (softmax, layernorm, elementwise), which is the
# right thing for a TOPS-denominator.
# ---------------------------------------------------------------------------

def _estimate_macs_onnx(onnx_bytes: bytes, *,
                        concrete_shapes: Optional[Dict[str, tuple]] = None) -> int:
    try:
        import onnx  # lazy
        from onnx import shape_inference
    except Exception:
        return 0

    try:
        model = onnx.load_model_from_string(onnx_bytes)
    except Exception:
        return 0

    # If the caller knows concrete dim values for dynamic inputs (common
    # for transformers where batch / seq_len are dim_params), stamp
    # those values ONLY on dims that are still unresolved (dim_value == 0
    # and a dim_param set). Resolved dims are left alone — avoids
    # breaking CNN graphs where shape-inference already had concrete
    # spatial dims.
    if concrete_shapes:
        for inp in model.graph.input:
            shape = concrete_shapes.get(inp.name)
            if shape is None:
                continue
            tt = inp.type.tensor_type
            if not tt.HasField("shape"):
                continue
            for i, d in enumerate(tt.shape.dim):
                if i >= len(shape):
                    break
                # Only stamp if the dim is still dynamic.
                if d.dim_value == 0 and d.dim_param:
                    d.dim_value = int(shape[i])
                    d.dim_param = ""

    # Try ORT's SymbolicShapeInference first (handles transformer ops +
    # data propagation better than onnx.shape_inference). Fall back to the
    # stock shape-inference for any model where the symbolic path fails.
    try:
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
        model = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
    except Exception:
        try:
            model = shape_inference.infer_shapes(model)
        except Exception:
            pass

    # Build a name -> shape map from graph initialisers + value_info.
    shapes: Dict[str, tuple] = {}
    for init in model.graph.initializer:
        shapes[init.name] = tuple(init.dims)
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        t = vi.type.tensor_type
        if t.HasField("shape"):
            dims = tuple(d.dim_value if d.dim_value > 0 else 1
                         for d in t.shape.dim)
            shapes[vi.name] = dims

    total = 0
    for node in model.graph.node:
        if node.op_type in ("Conv",):
            w_name = node.input[1] if len(node.input) > 1 else None
            out_name = node.output[0] if node.output else None
            w_shape = shapes.get(w_name or "")
            y_shape = shapes.get(out_name or "")
            if w_shape and y_shape and len(w_shape) == 4 and len(y_shape) == 4:
                M_out, C_in, KH, KW = w_shape
                _, _, H_out, W_out = y_shape
                total += int(M_out) * int(C_in) * int(KH) * int(KW) \
                       * int(H_out) * int(W_out)
        elif node.op_type in ("MatMul", "Gemm"):
            a = shapes.get(node.input[0]) if node.input else None
            b = shapes.get(node.input[1]) if len(node.input) > 1 else None
            if a and b and len(a) >= 2 and len(b) >= 2:
                M = int(a[-2])
                K = int(a[-1])
                N = int(b[-1])
                # Transformer MatMuls are batched: (B, S, H) @ (H, H) has
                # batch = B × S leading dims. Multiply those in too.
                batch_prod = 1
                for d in a[:-2]:
                    try:
                        batch_prod *= int(d)
                    except Exception:
                        pass
                total += batch_prod * M * K * N
    return total
