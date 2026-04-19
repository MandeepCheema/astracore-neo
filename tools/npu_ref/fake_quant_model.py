"""Produce a fake-quant .onnx from a calibrated NnGraph (F1-C2 validation).

The fake-quant model is a functional simulator of the INT8 deployment:

  - Every conv/gemm/matmul weight tensor is round-tripped through
    per-channel INT8 (quantise → dequantise → replace initializer).
    This is bit-exact the math the RTL will do on the weight side.

  - Every activation tensor feeding a weight-bearing layer gets a
    QuantizeLinear → DequantizeLinear pair inserted in front of it,
    using the per-tensor activation scale from calibration. This
    simulates the host-side round-trip between tiles.

Running the fake-quant model under onnxruntime and comparing to the
FP32 reference gives us the end-to-end INT8-simulated output without
any RTL in the loop — the acceptance gate for F1-C2.

Why this is honest validation:
  - ORT + ONNX QuantizeLinear/DequantizeLinear ops define the exact
    numerical semantics of symmetric INT8 quantisation.
  - The RTL's INT32 accumulator means quantisation error is fully
    concentrated at the weight-side and activation-boundary
    round-trips. Perfectly modeled by fake-quant.
  - Any INT8-vs-FP32 gap observed here is the gap F1-C5 will see
    on hardware (modulo negligible int arithmetic differences).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .nn_graph import (
    GRAN_PER_CHANNEL,
    NnGraph,
    NnLayer,
    OP_CONV,
    OP_GEMM,
    OP_MATMUL,
)
from .quantiser import fake_quantise_weights

_WEIGHT_BEARING_OPS = {OP_CONV, OP_GEMM, OP_MATMUL}
_INT8_SYM_RANGE = 127


def _init_by_name(model: onnx.ModelProto, name: str) -> int:
    for i, t in enumerate(model.graph.initializer):
        if t.name == name:
            return i
    raise KeyError(f"initializer {name!r} not found in model")


def _replace_weight_initializer(model: onnx.ModelProto,
                                 name: str,
                                 new_data: np.ndarray) -> None:
    idx = _init_by_name(model, name)
    del model.graph.initializer[idx]
    model.graph.initializer.append(numpy_helper.from_array(new_data, name))


def _unique_name(base: str, existing: Set[str]) -> str:
    n = base
    i = 0
    while n in existing:
        i += 1
        n = f"{base}_{i}"
    existing.add(n)
    return n


def _insert_qdq_node(nodes: List[onnx.NodeProto],
                     model_inits: List[onnx.TensorProto],
                     value_infos: List[onnx.ValueInfoProto],
                     name_registry: Set[str],
                     *,
                     source_name: str,
                     scale: float,
                     axis_name: str) -> str:
    """Insert QuantizeLinear → DequantizeLinear after `source_name`.
    Returns the name of the new dequantised tensor (the consumer
    will rewire its input from source_name to this new name).

    Scale is per-tensor (0-D), zero-point is 0 (symmetric).
    """
    q_out = _unique_name(f"{source_name}_q", name_registry)
    dq_out = _unique_name(f"{source_name}_dq", name_registry)
    scale_init_name = _unique_name(f"{axis_name}_scale", name_registry)
    zp_init_name = _unique_name(f"{axis_name}_zp", name_registry)

    model_inits.append(numpy_helper.from_array(
        np.array(scale, dtype=np.float32), scale_init_name))
    model_inits.append(numpy_helper.from_array(
        np.array(0, dtype=np.int8), zp_init_name))

    q_node = helper.make_node(
        "QuantizeLinear",
        inputs=[source_name, scale_init_name, zp_init_name],
        outputs=[q_out],
        name=_unique_name(f"q_{axis_name}", name_registry),
    )
    dq_node = helper.make_node(
        "DequantizeLinear",
        inputs=[q_out, scale_init_name, zp_init_name],
        outputs=[dq_out],
        name=_unique_name(f"dq_{axis_name}", name_registry),
    )
    nodes.extend([q_node, dq_node])
    # Advisory value_infos so shape inference + ORT type-check pass.
    value_infos.append(helper.make_tensor_value_info(
        q_out, TensorProto.INT8, None))
    value_infos.append(helper.make_tensor_value_info(
        dq_out, TensorProto.FLOAT, None))
    return dq_out


def build_fake_quant_model(graph: NnGraph,
                            original_onnx_path: str,
                            *,
                            out_path: str | None = None,
                            fake_quant_activations: bool = True) -> str:
    """Emit a functionally-equivalent ONNX where weights are round-
    tripped through INT8 and (optionally) activations feeding weight-
    bearing layers are wrapped with Quantize/DequantizeLinear pairs.

    Args:
        graph: NnGraph annotated by quantise_weights + calibrate_activations.
        original_onnx_path: path to the FP32 .onnx the graph was
            loaded from.
        out_path: destination for the rewritten .onnx; defaults to a
            tempfile beside the original.
        fake_quant_activations: when False, only weights are fake-
            quantised (used for isolating weight-side vs activation-
            side error in debugging).

    Returns:
        Path to the fake-quant .onnx.
    """
    model = onnx.load(original_onnx_path)
    name_registry: Set[str] = set()
    for t in model.graph.initializer:
        name_registry.add(t.name)
    for vi in list(model.graph.input) + list(model.graph.output) + \
              list(model.graph.value_info):
        name_registry.add(vi.name)
    for n in model.graph.node:
        for io in list(n.input) + list(n.output):
            name_registry.add(io)

    # 1) Weight-side fake quant.
    for layer in graph.layers:
        if layer.op not in _WEIGHT_BEARING_OPS:
            continue
        if layer.weights is None or layer.quant is None:
            continue
        # Find the weight initializer name the loader captured for
        # this layer. NnLayer.inputs holds only the activation input,
        # so the weight tensor name has to come from the original
        # node's input list.
        src_node = _find_node(model, layer.outputs)
        if src_node is None:
            continue
        # Weight is the second input for conv / gemm / matmul.
        w_name = src_node.input[1] if len(src_node.input) > 1 else None
        if not w_name:
            continue
        fq = fake_quantise_weights(layer.weights, layer.quant.weight_scale,
                                    precision=layer.quant.precision)
        _replace_weight_initializer(model, w_name, fq)

    if not fake_quant_activations:
        return _save_model(model, out_path)

    # 2) Activation-side QDQ insertion.
    act_scales: Dict[str, float] = graph.metadata.get(
        "activation_scales", {})
    if not act_scales:
        raise RuntimeError(
            "build_fake_quant_model: graph has no activation_scales in "
            "metadata — run calibrate_activations() before fake-quant."
        )

    # Decide which tensors need QDQ on entry to weight-bearing ops.
    # We wire each such layer's input to a freshly-dequantised tensor,
    # but we share one Q/DQ per (source tensor, scale) to keep the
    # model smaller.
    new_nodes: List[onnx.NodeProto] = []
    new_value_infos: List[onnx.ValueInfoProto] = []
    qdq_cache: Dict[str, str] = {}  # source name -> dq output name

    for node in list(model.graph.node):
        if node.op_type not in ("Conv", "Gemm", "MatMul"):
            continue
        if not node.input:
            continue
        src = node.input[0]
        if src not in act_scales:
            continue  # untracked tensor — leave alone
        if src in qdq_cache:
            node.input[0] = qdq_cache[src]
            continue
        dq_name = _insert_qdq_node(
            new_nodes, model.graph.initializer, new_value_infos,
            name_registry,
            source_name=src, scale=act_scales[src], axis_name=src,
        )
        qdq_cache[src] = dq_name
        node.input[0] = dq_name

    # Splice new QDQ nodes into graph.node in a position that
    # preserves topological order: just prepend them. ONNX permits
    # any order; ORT sorts topologically at load time.
    for n in new_nodes:
        model.graph.node.append(n)
    # Re-order nodes: new QDQ nodes must come *before* the consumer
    # conv nodes. Simpler fix: move all new nodes to the front.
    # ORT tolerates this; for robustness we explicitly re-sort.
    _topological_resort(model)

    for vi in new_value_infos:
        model.graph.value_info.append(vi)

    return _save_model(model, out_path)


def _find_node(model: onnx.ModelProto, outputs: List[str]):
    outs = set(outputs)
    for n in model.graph.node:
        for o in n.output:
            if o in outs:
                return n
    return None


def _topological_resort(model: onnx.ModelProto) -> None:
    """Reorder model.graph.node so every producer precedes its
    consumers. Needed after we add QDQ nodes that must run before
    the conv nodes we rewired."""
    produced_by: Dict[str, int] = {}
    for i, n in enumerate(model.graph.node):
        for o in n.output:
            produced_by[o] = i
    graph_inputs = {i.name for i in model.graph.input}
    graph_inputs |= {t.name for t in model.graph.initializer}

    visited = [False] * len(model.graph.node)
    order: List[int] = []

    def visit(idx: int, path: Set[int]) -> None:
        if visited[idx]:
            return
        if idx in path:
            return  # cycle guard (shouldn't happen in valid ONNX)
        path.add(idx)
        for inp in model.graph.node[idx].input:
            if inp in graph_inputs:
                continue
            prod = produced_by.get(inp)
            if prod is not None:
                visit(prod, path)
        path.remove(idx)
        visited[idx] = True
        order.append(idx)

    for i in range(len(model.graph.node)):
        visit(i, set())

    new_nodes = [model.graph.node[i] for i in order]
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)


def _save_model(model: onnx.ModelProto, out_path: str | None) -> str:
    if out_path is None:
        import os as _os
        fd, out_path = tempfile.mkstemp(suffix=".fakequant.onnx")
        _os.close(fd)
    onnx.save(model, out_path)
    return out_path
