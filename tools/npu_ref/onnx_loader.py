"""ONNX → NnGraph bridge (F1-C1).

Public entry point: `load_onnx(path) -> NnGraph`.

Scope: faithfully translate the structural graph (ops + shapes + FP32
weights) into the project's NnGraph IR. No fusion, no quantisation, no
tiling — those are downstream passes (F1-C2/C3/C4).

Supported ONNX ops are those exercised by YOLOv8-N: Conv, Gemm, MatMul,
Add, Mul, Concat, Split, Resize, MaxPool, AveragePool, GlobalAveragePool,
Sigmoid, Relu, Softmax, Reshape, Transpose, Slice. Unsupported ops raise
NotImplementedError with the op name — the loader will fail loudly rather
than silently drop a layer (which would corrupt downstream analyses).

Shapes: the loader runs `onnx.shape_inference.infer_shapes` before
dispatch, then records each tensor's resolved shape. If the original
model has a symbolic batch dim we concretise it to `batch_size` (default
1) and re-infer; otherwise we leave every dim as-is.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

from .nn_graph import (
    NnGraph,
    NnLayer,
    OP_ADD,
    OP_AVGPOOL,
    OP_CONCAT,
    OP_CONV,
    OP_DIV,
    OP_GELU,
    OP_GEMM,
    OP_LAYERNORM,
    OP_MATMUL,
    OP_MAXPOOL,
    OP_MHA,
    OP_MUL,
    OP_RELU,
    OP_RESHAPE,
    OP_RESIZE,
    OP_RMSNORM,
    OP_ROTARY_EMB,
    OP_SIGMOID,
    OP_SLICE,
    OP_SOFTMAX,
    OP_SPLIT,
    OP_SUB,
    OP_TRANSPOSE,
)

Shape = Tuple[int, ...]
_Initializers = Dict[str, np.ndarray]
_ShapeMap = Dict[str, Shape]


# ---------------------------------------------------------------------------
# Attribute helpers
# ---------------------------------------------------------------------------
def _attr_map(node: onnx.NodeProto) -> Dict[str, Any]:
    """Convert a node's AttributeProto list into a plain dict."""
    out: Dict[str, Any] = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INT:
            out[a.name] = a.i
        elif a.type == onnx.AttributeProto.FLOAT:
            out[a.name] = a.f
        elif a.type == onnx.AttributeProto.STRING:
            out[a.name] = a.s.decode("utf-8")
        elif a.type == onnx.AttributeProto.INTS:
            out[a.name] = tuple(a.ints)
        elif a.type == onnx.AttributeProto.FLOATS:
            out[a.name] = tuple(a.floats)
        elif a.type == onnx.AttributeProto.STRINGS:
            out[a.name] = tuple(s.decode("utf-8") for s in a.strings)
        elif a.type == onnx.AttributeProto.TENSOR:
            out[a.name] = numpy_helper.to_array(a.t)
        else:
            # Unknown attr types are rare (GRAPH, SPARSE_TENSOR, etc.);
            # stash the raw proto so callers can inspect if needed.
            out[a.name] = a
    return out


def _init_as_fp32(arr: np.ndarray) -> np.ndarray:
    """Cast an initializer to float32. F1-C2 consumes FP32 and computes
    quant scales from it; keeping this cast at the loader means every
    downstream consumer sees a single dtype."""
    if arr.dtype == np.float32:
        return arr
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-op handlers. Each takes (node, initializers, shapes) and returns a
# fully-populated NnLayer, or raises if the node shape is malformed.
# ---------------------------------------------------------------------------
def _shape_subset(names: List[str], shapes: _ShapeMap) -> Dict[str, Shape]:
    """Gather the shapes for a list of tensor names, skipping any that
    shape inference couldn't resolve."""
    return {n: shapes[n] for n in names if n in shapes}


def _handle_conv(node, init, shapes) -> NnLayer:
    a = _attr_map(node)
    w_name = node.input[1]
    if w_name not in init:
        raise ValueError(
            f"Conv {node.name!r}: weight tensor {w_name!r} not in "
            f"initializers. Loader expects weights to be constants."
        )
    W = _init_as_fp32(init[w_name])
    # ONNX Conv weight layout: (M=C_out, C_in/groups, kH, kW)
    if W.ndim != 4:
        raise ValueError(
            f"Conv {node.name!r}: expected 4-D weight, got shape {W.shape}"
        )
    B: Optional[np.ndarray] = None
    if len(node.input) >= 3 and node.input[2]:
        b_name = node.input[2]
        if b_name not in init:
            raise ValueError(
                f"Conv {node.name!r}: bias tensor {b_name!r} not in "
                f"initializers."
            )
        B = _init_as_fp32(init[b_name])

    k_h, k_w = W.shape[2], W.shape[3]
    kernel = a.get("kernel_shape", (k_h, k_w))
    attrs = {
        "kernel": tuple(kernel),
        "stride": tuple(a.get("strides", (1, 1))),
        "pad": tuple(a.get("pads", (0, 0, 0, 0))),
        "dilation": tuple(a.get("dilations", (1, 1))),
        "groups": a.get("group", 1),
    }
    act_inputs = [node.input[0]]
    return NnLayer(
        name=node.name or f"conv_{node.output[0]}",
        op=OP_CONV,
        inputs=act_inputs,
        outputs=list(node.output),
        in_shapes=_shape_subset(act_inputs, shapes),
        out_shapes=_shape_subset(list(node.output), shapes),
        attrs=attrs,
        weights=W,
        bias=B,
    )


def _handle_gemm(node, init, shapes) -> NnLayer:
    a = _attr_map(node)
    w_name = node.input[1]
    W = _init_as_fp32(init[w_name]) if w_name in init else None
    B: Optional[np.ndarray] = None
    if len(node.input) >= 3 and node.input[2] and node.input[2] in init:
        B = _init_as_fp32(init[node.input[2]])
    act_inputs = [node.input[0]]
    attrs = {
        "alpha": a.get("alpha", 1.0),
        "beta": a.get("beta", 1.0),
        "trans_a": a.get("transA", 0),
        "trans_b": a.get("transB", 0),
    }
    return NnLayer(
        name=node.name or f"gemm_{node.output[0]}",
        op=OP_GEMM,
        inputs=act_inputs,
        outputs=list(node.output),
        in_shapes=_shape_subset(act_inputs, shapes),
        out_shapes=_shape_subset(list(node.output), shapes),
        attrs=attrs,
        weights=W,
        bias=B,
    )


def _handle_matmul(node, init, shapes) -> NnLayer:
    act_inputs = list(node.input)
    W: Optional[np.ndarray] = None
    if len(node.input) == 2 and node.input[1] in init:
        W = _init_as_fp32(init[node.input[1]])
        act_inputs = [node.input[0]]
    return NnLayer(
        name=node.name or f"matmul_{node.output[0]}",
        op=OP_MATMUL,
        inputs=act_inputs,
        outputs=list(node.output),
        in_shapes=_shape_subset(act_inputs, shapes),
        out_shapes=_shape_subset(list(node.output), shapes),
        attrs={},
        weights=W,
    )


def _handle_elementwise(op: str) -> Callable:
    def _h(node, init, shapes) -> NnLayer:
        # Element-wise ops may have one operand that is a learned constant
        # (e.g., bias folded into Add); capture it if so.
        # For non-commutative ops (Sub, Div) the original operand
        # position is preserved in attrs["const_pos"] so downstream
        # executors can apply the op in the right order.
        W: Optional[np.ndarray] = None
        const_pos: Optional[int] = None
        act_inputs: List[str] = []
        for i, name in enumerate(node.input):
            if name in init:
                W = _init_as_fp32(init[name])
                const_pos = i
            else:
                act_inputs.append(name)
        attrs: Dict[str, Any] = {}
        if const_pos is not None:
            attrs["const_pos"] = const_pos
        return NnLayer(
            name=node.name or f"{op}_{node.output[0]}",
            op=op,
            inputs=act_inputs,
            outputs=list(node.output),
            in_shapes=_shape_subset(act_inputs, shapes),
            out_shapes=_shape_subset(list(node.output), shapes),
            attrs=attrs,
            weights=W,
        )
    return _h


def _handle_unary(op: str) -> Callable:
    def _h(node, init, shapes) -> NnLayer:
        a = _attr_map(node)
        return NnLayer(
            name=node.name or f"{op}_{node.output[0]}",
            op=op,
            inputs=list(node.input),
            outputs=list(node.output),
            in_shapes=_shape_subset(list(node.input), shapes),
            out_shapes=_shape_subset(list(node.output), shapes),
            attrs=a,
        )
    return _h


def _handle_pool(op: str) -> Callable:
    def _h(node, init, shapes) -> NnLayer:
        a = _attr_map(node)
        attrs = {
            "kernel": tuple(a.get("kernel_shape", ())),
            "stride": tuple(a.get("strides", (1, 1))),
            "pad": tuple(a.get("pads", (0, 0, 0, 0))),
            "ceil_mode": a.get("ceil_mode", 0),
        }
        return NnLayer(
            name=node.name or f"{op}_{node.output[0]}",
            op=op,
            inputs=list(node.input),
            outputs=list(node.output),
            in_shapes=_shape_subset(list(node.input), shapes),
            out_shapes=_shape_subset(list(node.output), shapes),
            attrs=attrs,
        )
    return _h


def _handle_global_avgpool(node, init, shapes) -> NnLayer:
    in_shape = shapes.get(node.input[0])
    attrs = {"kernel": in_shape[2:] if in_shape else (), "stride": (1, 1),
             "pad": (0, 0, 0, 0), "ceil_mode": 0, "global": True}
    return NnLayer(
        name=node.name or f"gap_{node.output[0]}",
        op=OP_AVGPOOL,
        inputs=list(node.input),
        outputs=list(node.output),
        in_shapes=_shape_subset(list(node.input), shapes),
        out_shapes=_shape_subset(list(node.output), shapes),
        attrs=attrs,
    )


def _handle_concat(node, init, shapes) -> NnLayer:
    a = _attr_map(node)
    return NnLayer(
        name=node.name or f"concat_{node.output[0]}",
        op=OP_CONCAT,
        inputs=list(node.input),
        outputs=list(node.output),
        in_shapes=_shape_subset(list(node.input), shapes),
        out_shapes=_shape_subset(list(node.output), shapes),
        attrs={"axis": a.get("axis", 0)},
    )


def _handle_split(node, init, shapes) -> NnLayer:
    a = _attr_map(node)
    act_inputs = [node.input[0]]
    # opset 13+: split sizes are a second input (an initializer)
    split_sizes: Optional[Tuple[int, ...]] = None
    if len(node.input) >= 2 and node.input[1] and node.input[1] in init:
        split_sizes = tuple(int(x) for x in init[node.input[1]].tolist())
    elif "split" in a:
        split_sizes = tuple(a["split"])
    return NnLayer(
        name=node.name or f"split_{node.output[0]}",
        op=OP_SPLIT,
        inputs=act_inputs,
        outputs=list(node.output),
        in_shapes=_shape_subset(act_inputs, shapes),
        out_shapes=_shape_subset(list(node.output), shapes),
        attrs={"axis": a.get("axis", 0), "split": split_sizes},
    )


def _handle_resize(node, init, shapes) -> NnLayer:
    a = _attr_map(node)
    # Resize has up to 4 inputs: X, roi, scales, sizes. Weights are not
    # learnable — they're constants describing the resize. Capture
    # whichever is provided.
    scales = None
    sizes = None
    if len(node.input) >= 3 and node.input[2] and node.input[2] in init:
        scales = tuple(float(x) for x in init[node.input[2]].tolist())
    if len(node.input) >= 4 and node.input[3] and node.input[3] in init:
        sizes = tuple(int(x) for x in init[node.input[3]].tolist())
    act_inputs = [node.input[0]]
    return NnLayer(
        name=node.name or f"resize_{node.output[0]}",
        op=OP_RESIZE,
        inputs=act_inputs,
        outputs=list(node.output),
        in_shapes=_shape_subset(act_inputs, shapes),
        out_shapes=_shape_subset(list(node.output), shapes),
        attrs={"mode": a.get("mode", "nearest"), "scales": scales,
               "sizes": sizes,
               "coordinate_transformation_mode":
                   a.get("coordinate_transformation_mode", "half_pixel")},
    )


def _handle_reshape(node, init, shapes) -> NnLayer:
    act_inputs = [node.input[0]]
    target_shape: Optional[Tuple[int, ...]] = None
    if len(node.input) >= 2 and node.input[1] in init:
        target_shape = tuple(int(x) for x in init[node.input[1]].tolist())
    return NnLayer(
        name=node.name or f"reshape_{node.output[0]}",
        op=OP_RESHAPE,
        inputs=act_inputs,
        outputs=list(node.output),
        in_shapes=_shape_subset(act_inputs, shapes),
        out_shapes=_shape_subset(list(node.output), shapes),
        attrs={"shape": target_shape},
    )


def _handle_transpose(node, init, shapes) -> NnLayer:
    a = _attr_map(node)
    return NnLayer(
        name=node.name or f"transpose_{node.output[0]}",
        op=OP_TRANSPOSE,
        inputs=list(node.input),
        outputs=list(node.output),
        in_shapes=_shape_subset(list(node.input), shapes),
        out_shapes=_shape_subset(list(node.output), shapes),
        attrs={"perm": a.get("perm", ())},
    )


def _handle_slice(node, init, shapes) -> NnLayer:
    a = _attr_map(node)
    act_inputs = [node.input[0]]
    # opset 10+: starts, ends, axes, steps are inputs
    starts = ends = axes = steps = None
    if len(node.input) >= 2 and node.input[1] in init:
        starts = tuple(int(x) for x in init[node.input[1]].tolist())
    if len(node.input) >= 3 and node.input[2] in init:
        ends = tuple(int(x) for x in init[node.input[2]].tolist())
    if len(node.input) >= 4 and node.input[3] and node.input[3] in init:
        axes = tuple(int(x) for x in init[node.input[3]].tolist())
    if len(node.input) >= 5 and node.input[4] and node.input[4] in init:
        steps = tuple(int(x) for x in init[node.input[4]].tolist())
    # opset <10 fallback: starts/ends/axes are attrs
    starts = starts or a.get("starts")
    ends = ends or a.get("ends")
    axes = axes or a.get("axes")
    return NnLayer(
        name=node.name or f"slice_{node.output[0]}",
        op=OP_SLICE,
        inputs=act_inputs,
        outputs=list(node.output),
        in_shapes=_shape_subset(act_inputs, shapes),
        out_shapes=_shape_subset(list(node.output), shapes),
        attrs={"starts": starts, "ends": ends, "axes": axes, "steps": steps},
    )


# ---------------------------------------------------------------------------
# Transformer ops (F1-B1)
# ---------------------------------------------------------------------------
def _handle_gelu(node, init, shapes) -> NnLayer:
    a = _attr_map(node)
    # `approximate`: "none" (default, erf-based) or "tanh". PyTorch's
    # nn.GELU(approximate="tanh") round-trips through ONNX with this attr.
    return NnLayer(
        name=node.name or f"gelu_{node.output[0]}",
        op=OP_GELU,
        inputs=list(node.input),
        outputs=list(node.output),
        in_shapes=_shape_subset(list(node.input), shapes),
        out_shapes=_shape_subset(list(node.output), shapes),
        attrs={"approximate": a.get("approximate", "none")},
    )


def _handle_layernorm(node, init, shapes) -> NnLayer:
    """ONNX opset 17+ LayerNormalization: inputs are X, Scale, Bias (optional).
    Scale and Bias are learnable and pulled out to weights/bias fields so the
    downstream tiler can treat them like any other per-layer parameter."""
    a = _attr_map(node)
    scale = init.get(node.input[1]) if len(node.input) >= 2 and node.input[1] else None
    bias = init.get(node.input[2]) if len(node.input) >= 3 and node.input[2] else None
    act_inputs = [node.input[0]]
    return NnLayer(
        name=node.name or f"layernorm_{node.output[0]}",
        op=OP_LAYERNORM,
        inputs=act_inputs,
        outputs=list(node.output),
        in_shapes=_shape_subset(act_inputs, shapes),
        out_shapes=_shape_subset(list(node.output), shapes),
        weights=scale,
        bias=bias,
        attrs={
            "axis": a.get("axis", -1),
            "epsilon": a.get("epsilon", 1e-5),
            "stash_type": a.get("stash_type", 1),
        },
    )


def _handle_rmsnorm(node, init, shapes) -> NnLayer:
    """RMSNormalization (ONNX opset 23+ / LLaMA exports). Like LayerNorm
    but without the mean-centering step; uses only the RMS of X."""
    a = _attr_map(node)
    scale = init.get(node.input[1]) if len(node.input) >= 2 and node.input[1] else None
    act_inputs = [node.input[0]]
    return NnLayer(
        name=node.name or f"rmsnorm_{node.output[0]}",
        op=OP_RMSNORM,
        inputs=act_inputs,
        outputs=list(node.output),
        in_shapes=_shape_subset(act_inputs, shapes),
        out_shapes=_shape_subset(list(node.output), shapes),
        weights=scale,
        attrs={
            "axis": a.get("axis", -1),
            "epsilon": a.get("epsilon", 1e-6),
        },
    )


def _handle_rotary_embedding(node, init, shapes) -> NnLayer:
    """ONNX RotaryEmbedding (opset 23+ / ORT contrib). Inputs are
    (X, cos_cache, sin_cache[, position_ids]). The cos/sin caches are
    initializers captured as weights (cos) and bias (sin) — the downstream
    MHSA tile (F1-A5) will expect them in that slot."""
    a = _attr_map(node)
    act_inputs = [node.input[0]]
    cos_cache = init.get(node.input[1]) if len(node.input) >= 2 and node.input[1] else None
    sin_cache = init.get(node.input[2]) if len(node.input) >= 3 and node.input[2] else None
    # position_ids can be a graph-level input or an initializer; if
    # it's a named graph tensor, keep it as an activation input.
    pos_name = node.input[3] if len(node.input) >= 4 and node.input[3] else None
    if pos_name is not None and pos_name not in init:
        act_inputs.append(pos_name)
    return NnLayer(
        name=node.name or f"rope_{node.output[0]}",
        op=OP_ROTARY_EMB,
        inputs=act_inputs,
        outputs=list(node.output),
        in_shapes=_shape_subset(act_inputs, shapes),
        out_shapes=_shape_subset(list(node.output), shapes),
        weights=cos_cache,
        bias=sin_cache,
        attrs={
            "interleaved": bool(a.get("interleaved", 0)),
            "rotary_embedding_dim": a.get("rotary_embedding_dim", 0),
            "num_heads": a.get("num_heads", 0),
        },
    )


def _handle_mha(node, init, shapes) -> NnLayer:
    """ONNX Attention / MultiHeadAttention (ORT contrib). Captures the
    head count + scale for the downstream MHSA tile (F1-A5)."""
    a = _attr_map(node)
    return NnLayer(
        name=node.name or f"mha_{node.output[0]}",
        op=OP_MHA,
        inputs=list(node.input),
        outputs=list(node.output),
        in_shapes=_shape_subset(list(node.input), shapes),
        out_shapes=_shape_subset(list(node.output), shapes),
        attrs={
            "num_heads": a.get("num_heads", 0),
            "scale": a.get("scale", 0.0),
            "unidirectional": bool(a.get("unidirectional", 0)),
            "do_rotary": bool(a.get("do_rotary", 0)),
        },
    )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------
_HANDLERS: Dict[str, Callable] = {
    "Conv":                _handle_conv,
    "Gemm":                _handle_gemm,
    "MatMul":              _handle_matmul,
    "Add":                 _handle_elementwise(OP_ADD),
    "Sub":                 _handle_elementwise(OP_SUB),
    "Mul":                 _handle_elementwise(OP_MUL),
    "Div":                 _handle_elementwise(OP_DIV),
    "Sigmoid":             _handle_unary(OP_SIGMOID),
    "Relu":                _handle_unary(OP_RELU),
    "Softmax":             _handle_unary(OP_SOFTMAX),
    "MaxPool":             _handle_pool(OP_MAXPOOL),
    "AveragePool":         _handle_pool(OP_AVGPOOL),
    "GlobalAveragePool":   _handle_global_avgpool,
    "Concat":              _handle_concat,
    "Split":               _handle_split,
    "Resize":              _handle_resize,
    "Upsample":            _handle_resize,
    "Reshape":             _handle_reshape,
    "Transpose":           _handle_transpose,
    "Slice":               _handle_slice,
    # F1-B1 transformer ops
    "Gelu":                _handle_gelu,
    "LayerNormalization":  _handle_layernorm,
    "RMSNormalization":    _handle_rmsnorm,
    "RotaryEmbedding":     _handle_rotary_embedding,
    "Attention":           _handle_mha,
    "MultiHeadAttention":  _handle_mha,
}


# ---------------------------------------------------------------------------
# Shape collection
# ---------------------------------------------------------------------------
def _concretise_batch(model: onnx.ModelProto, batch_size: int) -> onnx.ModelProto:
    """If the model's graph inputs have a symbolic first dim, replace it
    with `batch_size`. This is needed because shape inference can't
    propagate symbolic dims through ops like Reshape."""
    changed = False
    for inp in model.graph.input:
        # Skip inputs that are also initializers (opset < 14 quirk).
        if inp.name in {i.name for i in model.graph.initializer}:
            continue
        tt = inp.type.tensor_type
        if not tt.shape.dim:
            continue
        d0 = tt.shape.dim[0]
        if d0.HasField("dim_param") or d0.dim_value == 0:
            d0.dim_param = ""
            d0.dim_value = batch_size
            changed = True
    return model


def _collect_shapes(model: onnx.ModelProto) -> _ShapeMap:
    """Walk every ValueInfoProto on the graph and flatten into a
    name → shape dict. Unknown dims (symbolic) are represented as -1."""
    shapes: _ShapeMap = {}

    def _shape_from_tt(tt: onnx.TypeProto.Tensor) -> Shape:
        dims: List[int] = []
        for d in tt.shape.dim:
            if d.HasField("dim_value") and d.dim_value > 0:
                dims.append(d.dim_value)
            else:
                dims.append(-1)
        return tuple(dims)

    for source in (model.graph.input, model.graph.output, model.graph.value_info):
        for vi in source:
            if vi.type.HasField("tensor_type"):
                shapes[vi.name] = _shape_from_tt(vi.type.tensor_type)
    # Initializers are typed constants — shape is exact.
    for init in model.graph.initializer:
        shapes[init.name] = tuple(init.dims)
    return shapes


def _collect_initializers(model: onnx.ModelProto) -> _Initializers:
    return {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}


def _io_map(vis, init_names) -> Dict[str, Shape]:
    """Build a {name: shape} dict for graph I/O, excluding tensors that
    are also initializers (older opsets list params in graph.input)."""
    out: Dict[str, Shape] = {}
    for vi in vis:
        if vi.name in init_names:
            continue
        if not vi.type.HasField("tensor_type"):
            continue
        dims: List[int] = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else -1)
        out[vi.name] = tuple(dims)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def load_onnx(path: str, *,
              batch_size: int = 1,
              strict: bool = True) -> NnGraph:
    """Parse an ONNX model into an NnGraph.

    Args:
        path: filesystem path to the .onnx file.
        batch_size: concretise the leading dimension of every dynamic
            graph input to this value so shape inference propagates.
        strict: if True, unknown ops raise NotImplementedError. If False,
            they are skipped (with a warning in metadata). Default True —
            silent drops corrupt downstream analyses.

    Returns:
        NnGraph with FP32 weights, resolved shapes, and one NnLayer per
        supported ONNX op in topological order.
    """
    model = onnx.load(path)
    onnx.checker.check_model(model)
    model = _concretise_batch(model, batch_size)
    model = shape_inference.infer_shapes(model)

    initializers = _collect_initializers(model)
    shapes = _collect_shapes(model)
    init_names = set(initializers.keys())

    layers: List[NnLayer] = []
    unsupported: List[str] = []
    for node in model.graph.node:
        handler = _HANDLERS.get(node.op_type)
        if handler is None:
            if strict:
                raise NotImplementedError(
                    f"ONNX op {node.op_type!r} (node {node.name!r}) has "
                    f"no handler. Add one to onnx_loader._HANDLERS or "
                    f"pass strict=False to skip."
                )
            unsupported.append(node.op_type)
            continue
        layers.append(handler(node, initializers, shapes))

    meta: Dict[str, Any] = {
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "ir_version": model.ir_version,
        "opset": [(o.domain or "ai.onnx", o.version) for o in model.opset_import],
        "path": path,
    }
    if unsupported:
        meta["unsupported_ops"] = unsupported

    return NnGraph(
        layers=layers,
        inputs=_io_map(model.graph.input, init_names),
        outputs=_io_map(model.graph.output, init_names),
        metadata=meta,
    )
