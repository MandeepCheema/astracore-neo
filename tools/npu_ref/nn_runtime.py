"""Layer-by-layer NnGraph execution engine (F1-C5).

Walks a quantised NnGraph layer by layer, producing a tensor env that
flows forward through the graph. For weight-bearing layers the engine
models the RTL's INT8 × INT8 → INT32 accumulator datapath exactly,
then applies host-side dequantisation, bias, and activation — the
same recipe F1-F3 will use when the driver talks to real silicon.

This is the functional companion to `tools/npu_ref/conv_compiler.py`:
  - conv_compiler emits the *instruction stream* a real NPU would run;
    proven bit-exact against the RTL at small scale by F1-C4's cocotb
    tests.
  - nn_runtime executes the *numerical behaviour* of that stream in
    fast numpy so we can validate full-model correctness (F1-C5)
    without the O(hours) cost of running every YOLOv8 layer through
    Verilator.

The engine has NO dependency on `simulate_program` for the hot path;
reference_conv2d_int8-style math is lowered to numpy matmul via
im2col for realistic speed. A small integration test (F1-C5 task #35)
separately checks that compile_conv2d + simulate_program agrees with
this numpy path on a real YOLOv8 conv shape.

Scope (v1):
  - Batch size 1, NCHW layout (YOLOv8 native).
  - INT8 symmetric quant per F1-C2 recipe (per-channel weights,
    per-tensor activations). INT4 (F1-B2) and FP8/FP16 (F1-A1) are
    later extensions.
  - Non-weight ops handled directly in FP32 numpy (the RTL defers
    shape ops and element-wise math to host software anyway).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .nn_graph import (
    NnGraph,
    NnLayer,
    OP_ADD,
    OP_AVGPOOL,
    OP_CONCAT,
    OP_CONV,
    OP_DIV,
    OP_GEMM,
    OP_MATMUL,
    OP_MAXPOOL,
    OP_MUL,
    OP_RELU,
    OP_RESHAPE,
    OP_RESIZE,
    OP_SIGMOID,
    OP_SILU,
    OP_SLICE,
    OP_SOFTMAX,
    OP_SPLIT,
    OP_SUB,
    OP_TRANSPOSE,
    QuantParams,
)

TensorEnv = Dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# Quantise helpers — mirror the RTL's effective math
# ---------------------------------------------------------------------------
_INT8_SYM_RANGE = 127


def _quantise_act_symmetric(x: np.ndarray, scale: float) -> np.ndarray:
    """FP32 → INT8 via symmetric scale. Matches the host-side quant
    step the runtime would do before shipping a tensor to the NPU."""
    if scale <= 0:
        return np.zeros_like(x, dtype=np.int8)
    q = np.round(x / scale)
    q = np.clip(q, -_INT8_SYM_RANGE, _INT8_SYM_RANGE)
    return q.astype(np.int8)


def _dequantise_conv_output(acc_i32: np.ndarray,
                             input_scale: float,
                             weight_scale: np.ndarray,
                             bias: Optional[np.ndarray]) -> np.ndarray:
    """Apply the per-channel dequantisation recipe + FP32 bias.

    acc_i32 has shape (1, C_out, H, W). weight_scale is (C_out,) for
    per-channel or () for per-tensor. The final multiplier per output
    channel is `input_scale * weight_scale[c]`.
    """
    if weight_scale.ndim == 0:
        combined = float(weight_scale) * input_scale
        out = acc_i32.astype(np.float32) * combined
    else:
        # Broadcast per C_out: acc_i32[:, c, :, :] * input_scale * w_scale[c]
        multi = (weight_scale * input_scale).astype(np.float32)
        out = acc_i32.astype(np.float32) * multi.reshape(1, -1, 1, 1)
    if bias is not None:
        out = out + bias.astype(np.float32).reshape(1, -1, 1, 1)
    return out


# ---------------------------------------------------------------------------
# Fast INT8 conv2d via im2col + numpy matmul
# ---------------------------------------------------------------------------
def _im2col(x: np.ndarray, k_h: int, k_w: int,
            stride: Tuple[int, int],
            pad: Tuple[int, int, int, int]) -> Tuple[np.ndarray, int, int]:
    """Produce the (M, C_in*k_h*k_w) column matrix plus (H_out, W_out).

    Fancy-indexed vectorised implementation — >100x faster than the
    naïve nested-loop version on YOLOv8-sized tensors. Preserves dtype.
    """
    _, C_in, H_in, W_in = x.shape
    s_h, s_w = stride
    pt, pl, pb, pr = pad
    if any(p != 0 for p in pad):
        xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)),
                     mode="constant", constant_values=0)
    else:
        xp = x
    _, _, H_pad, W_pad = xp.shape
    H_out = (H_pad - k_h) // s_h + 1
    W_out = (W_pad - k_w) // s_w + 1

    # Vectorised row/col index grids. Shape invariants:
    #   i_row = (C_in, k_h, k_w, H_out, W_out)  — source-H index
    #   j_col = same shape                       — source-W index
    # Flattening the last two dims gives (C_in*k_h*k_w, H_out*W_out).
    i0 = np.repeat(np.arange(k_h), k_w)         # (k_h*k_w,)
    i0 = np.tile(i0, C_in)                      # (C_in*k_h*k_w,)
    i1 = s_h * np.arange(H_out)                 # (H_out,)
    j0 = np.tile(np.arange(k_w), k_h)           # (k_h*k_w,)
    j0 = np.tile(j0, C_in)                      # (C_in*k_h*k_w,)
    j1 = s_w * np.arange(W_out)                 # (W_out,)
    c = np.repeat(np.arange(C_in), k_h * k_w)   # (C_in*k_h*k_w,)

    i = i0.reshape(-1, 1, 1) + i1.reshape(1, -1, 1)  # (C_in*k_h*k_w, H_out, 1)
    j = j0.reshape(-1, 1, 1) + j1.reshape(1, 1, -1)  # (C_in*k_h*k_w, 1, W_out)
    c = c.reshape(-1, 1, 1)

    # Shape: (C_in*k_h*k_w, H_out, W_out)
    gathered = xp[0, c, i, j]
    # → (M=H_out*W_out, C_in*k_h*k_w)
    cols = gathered.reshape(-1, H_out * W_out).T.copy()
    return cols.astype(x.dtype, copy=False), H_out, W_out


def _conv2d_int8_fast(x_i8: np.ndarray, w_i8: np.ndarray,
                       stride: Tuple[int, int],
                       pad: Tuple[int, int, int, int]) -> np.ndarray:
    """INT8 conv2d producing INT32 accumulator tensor, matching
    reference_conv2d_int8 byte-for-byte but running at numpy speed."""
    _, C_in, _, _ = x_i8.shape
    C_out, _, k_h, k_w = w_i8.shape
    cols, H_out, W_out = _im2col(x_i8.astype(np.int32), k_h, k_w, stride, pad)
    w_mat = w_i8.astype(np.int32).reshape(C_out, -1).T  # (K_total, C_out)
    acc = cols @ w_mat                                    # (M, C_out), int32
    # (M, C_out) → (1, C_out, H_out, W_out)
    return acc.T.reshape(1, C_out, H_out, W_out).astype(np.int32)


# ---------------------------------------------------------------------------
# Weight-bearing layer handlers
# ---------------------------------------------------------------------------
def _exec_conv(layer: NnLayer, env: TensorEnv) -> np.ndarray:
    qp: QuantParams = layer.quant
    fp_in = env[layer.inputs[0]]
    # Quantise input to INT8 using the per-tensor input_scale.
    x_i8 = _quantise_act_symmetric(fp_in, qp.input_scale)
    # Symmetric INT8 weights: the quant pass left layer.weights in
    # FP32; re-quantise on the fly here. (Pre-computing could shave
    # ~10% but bloats memory.)
    w_scale = qp.weight_scale
    if w_scale.ndim == 0:
        w_i8 = _quantise_act_symmetric(layer.weights, float(w_scale))
    else:
        # Per-channel on axis 0 (C_out).
        w_i8 = np.zeros_like(layer.weights, dtype=np.int8)
        for c in range(layer.weights.shape[0]):
            s = float(w_scale[c])
            if s <= 0:
                continue
            q = np.round(layer.weights[c] / s)
            q = np.clip(q, -_INT8_SYM_RANGE, _INT8_SYM_RANGE)
            w_i8[c] = q.astype(np.int8)
    # Conv math.
    attrs = layer.attrs
    acc = _conv2d_int8_fast(x_i8, w_i8,
                             stride=tuple(attrs.get("stride", (1, 1))),
                             pad=tuple(attrs.get("pad", (0, 0, 0, 0))))
    return _dequantise_conv_output(acc, qp.input_scale, w_scale, layer.bias)


def _exec_matmul_or_gemm(layer: NnLayer, env: TensorEnv) -> np.ndarray:
    """Minimal Gemm/MatMul for non-YOLOv8 models. yolov8n doesn't use
    these (all its compute is Conv), but the engine stays extensible."""
    qp: QuantParams = layer.quant if layer.quant is not None else None
    fp_in = env[layer.inputs[0]]
    if layer.weights is None:
        # MatMul with non-constant RHS (two activation inputs). Host
        # side can just matmul both.
        b = env[layer.inputs[1]]
        return (fp_in.astype(np.float32) @ b.astype(np.float32))
    w = layer.weights
    if layer.op == OP_GEMM and layer.attrs.get("trans_b", 0):
        w = w.T
    out = fp_in.astype(np.float32) @ w.astype(np.float32)
    if layer.bias is not None:
        out = out + layer.bias.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Non-weight op handlers
# ---------------------------------------------------------------------------
def _exec_elementwise_binary(op_func: Callable):
    def _h(layer: NnLayer, env: TensorEnv) -> np.ndarray:
        # Most element-wise ops have exactly two activation inputs or
        # one activation + one stored constant (captured into
        # layer.weights). For non-commutative ops (Sub / Div) the
        # loader recorded which position the constant occupied in the
        # original ONNX node via attrs["const_pos"]; respect it so the
        # operand order is preserved.
        if layer.weights is not None and len(layer.inputs) == 1:
            a = env[layer.inputs[0]].astype(np.float32)
            b = layer.weights.astype(np.float32)
            if layer.attrs.get("const_pos") == 0:
                return op_func(b, a)   # constant was the left operand
            return op_func(a, b)
        a = env[layer.inputs[0]].astype(np.float32)
        b = env[layer.inputs[1]].astype(np.float32)
        return op_func(a, b)
    return _h


def _exec_sigmoid(layer: NnLayer, env: TensorEnv) -> np.ndarray:
    x = env[layer.inputs[0]].astype(np.float32)
    # Clip to avoid overflow in exp — safe for INT8-range inputs.
    return (1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))).astype(np.float32)


def _exec_silu(layer: NnLayer, env: TensorEnv) -> np.ndarray:
    x = env[layer.inputs[0]].astype(np.float32)
    return (x * 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))).astype(np.float32)


def _exec_relu(layer: NnLayer, env: TensorEnv) -> np.ndarray:
    return np.maximum(env[layer.inputs[0]].astype(np.float32), 0.0)


def _exec_softmax(layer: NnLayer, env: TensorEnv) -> np.ndarray:
    x = env[layer.inputs[0]].astype(np.float32)
    axis = int(layer.attrs.get("axis", -1))
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float32)


def _exec_concat(layer: NnLayer, env: TensorEnv) -> np.ndarray:
    axis = int(layer.attrs.get("axis", 0))
    tensors = [env[name].astype(np.float32) for name in layer.inputs]
    return np.concatenate(tensors, axis=axis)


def _exec_split(layer: NnLayer, env: TensorEnv) -> Tuple[np.ndarray, ...]:
    x = env[layer.inputs[0]].astype(np.float32)
    axis = int(layer.attrs.get("axis", 0))
    split = layer.attrs.get("split")
    if split is None:
        n = len(layer.outputs)
        return np.split(x, n, axis=axis)
    # Convert cumulative split sizes into np.split indices.
    indices = []
    acc = 0
    for s in split[:-1]:
        acc += int(s)
        indices.append(acc)
    return np.split(x, indices, axis=axis)


def _exec_reshape(layer: NnLayer, env: TensorEnv) -> np.ndarray:
    x = env[layer.inputs[0]]
    shape = layer.attrs.get("shape")
    if shape is None:
        # ONNX allows dynamic shape via second input — not expected
        # for YOLOv8 export but support it.
        shape = tuple(int(v) for v in env[layer.inputs[1]].tolist())
    return x.reshape(shape)


def _exec_transpose(layer: NnLayer, env: TensorEnv) -> np.ndarray:
    x = env[layer.inputs[0]]
    perm = layer.attrs.get("perm")
    if perm:
        return np.transpose(x, perm)
    return np.transpose(x)


def _exec_slice(layer: NnLayer, env: TensorEnv) -> np.ndarray:
    x = env[layer.inputs[0]]
    starts = layer.attrs["starts"]
    ends = layer.attrs["ends"]
    axes = layer.attrs.get("axes") or tuple(range(len(starts)))
    steps = layer.attrs.get("steps") or tuple(1 for _ in starts)
    slc = [slice(None)] * x.ndim
    for s, e, a, st in zip(starts, ends, axes, steps):
        slc[a] = slice(int(s), int(e), int(st))
    return x[tuple(slc)]


def _exec_maxpool(layer: NnLayer, env: TensorEnv) -> np.ndarray:
    x = env[layer.inputs[0]].astype(np.float32)
    return _pool2d(x, layer.attrs, reducer=np.max,
                    fill=-np.inf)


def _exec_avgpool(layer: NnLayer, env: TensorEnv) -> np.ndarray:
    x = env[layer.inputs[0]].astype(np.float32)
    if layer.attrs.get("global"):
        return x.mean(axis=(2, 3), keepdims=True).astype(np.float32)
    return _pool2d(x, layer.attrs, reducer=np.mean, fill=0.0)


def _pool2d(x: np.ndarray, attrs: Dict, *,
             reducer: Callable, fill: float) -> np.ndarray:
    k_h, k_w = attrs["kernel"]
    s_h, s_w = attrs.get("stride", (1, 1))
    pad = attrs.get("pad", (0, 0, 0, 0))
    pt, pl, pb, pr = pad
    xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)),
                  mode="constant", constant_values=fill)
    N, C, H, W = xp.shape
    H_out = (H - k_h) // s_h + 1
    W_out = (W - k_w) // s_w + 1
    out = np.empty((N, C, H_out, W_out), dtype=np.float32)
    for h_out in range(H_out):
        for w_out in range(W_out):
            patch = xp[:, :, h_out * s_h:h_out * s_h + k_h,
                          w_out * s_w:w_out * s_w + k_w]
            out[:, :, h_out, w_out] = reducer(patch, axis=(2, 3))
    return out


def _exec_resize(layer: NnLayer, env: TensorEnv) -> np.ndarray:
    x = env[layer.inputs[0]].astype(np.float32)
    scales = layer.attrs.get("scales")
    sizes = layer.attrs.get("sizes")
    mode = layer.attrs.get("mode", "nearest")
    if scales is not None:
        # ONNX Resize scales are per-dim (batch, channel, H, W).
        # YOLOv8 uses nearest-mode 2x up-sample on spatial dims only.
        new_shape = tuple(int(s * d) for s, d in zip(scales, x.shape))
    elif sizes is not None:
        new_shape = tuple(int(s) for s in sizes)
    else:
        return x
    if mode != "nearest":
        raise NotImplementedError(
            f"Resize mode {mode!r} not yet supported; YOLOv8 uses 'nearest'"
        )
    # Nearest-neighbour upsample: repeat along each spatial dim.
    s_h = new_shape[2] / x.shape[2]
    s_w = new_shape[3] / x.shape[3]
    # Expect integer ratios (typical YOLO upsample = 2).
    r_h = int(round(s_h))
    r_w = int(round(s_w))
    return np.repeat(np.repeat(x, r_h, axis=2), r_w, axis=3)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------
_DISPATCH: Dict[str, Callable] = {
    OP_CONV:      _exec_conv,
    OP_GEMM:      _exec_matmul_or_gemm,
    OP_MATMUL:    _exec_matmul_or_gemm,
    OP_ADD:       _exec_elementwise_binary(np.add),
    OP_SUB:       _exec_elementwise_binary(np.subtract),
    OP_MUL:       _exec_elementwise_binary(np.multiply),
    OP_DIV:       _exec_elementwise_binary(np.divide),
    OP_SIGMOID:   _exec_sigmoid,
    OP_SILU:      _exec_silu,
    OP_RELU:      _exec_relu,
    OP_SOFTMAX:   _exec_softmax,
    OP_CONCAT:    _exec_concat,
    OP_SPLIT:     _exec_split,
    OP_RESHAPE:   _exec_reshape,
    OP_TRANSPOSE: _exec_transpose,
    OP_SLICE:     _exec_slice,
    OP_MAXPOOL:   _exec_maxpool,
    OP_AVGPOOL:   _exec_avgpool,
    OP_RESIZE:    _exec_resize,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_graph(graph: NnGraph,
              inputs: Dict[str, np.ndarray],
              *,
              progress: bool = False) -> Dict[str, np.ndarray]:
    """Execute a quantised NnGraph with FP32 inputs, returning FP32
    outputs. Weight-bearing layers route through the INT8 quant path;
    everything else is FP32 numpy.

    Args:
        graph: NnGraph that has already been processed by
            quantise_model() (F1-C2). Unquantised graphs raise because
            QuantParams is required on every conv/gemm/matmul.
        inputs: {graph_input_name: FP32 ndarray}.
        progress: when True, prints one line per layer (useful for
            large models like yolov8n where the full run takes a
            minute or two).

    Returns:
        {graph_output_name: FP32 ndarray}.
    """
    env: TensorEnv = {}
    for name, arr in inputs.items():
        if name not in graph.inputs:
            raise ValueError(
                f"input {name!r} not a graph input (known: "
                f"{sorted(graph.inputs)})"
            )
        env[name] = arr.astype(np.float32)

    # Some graphs carry Split outputs that don't cleanly fit the
    # single-output convention — handle specially.
    for i, layer in enumerate(graph.layers):
        handler = _DISPATCH.get(layer.op)
        if handler is None:
            raise NotImplementedError(
                f"nn_runtime: no handler for op {layer.op!r} "
                f"(layer {layer.name!r} index {i}). Add to _DISPATCH."
            )
        if layer.op in (OP_CONV, OP_GEMM, OP_MATMUL) and layer.weights is not None \
                and layer.quant is None:
            raise RuntimeError(
                f"layer {layer.name!r} has no QuantParams — run "
                f"quantise_model(graph, ...) before run_graph()."
            )

        result = handler(layer, env)

        # Split returns a tuple of arrays.
        if layer.op == OP_SPLIT:
            parts = result
            if len(parts) != len(layer.outputs):
                raise RuntimeError(
                    f"split {layer.name!r}: got {len(parts)} parts, "
                    f"expected {len(layer.outputs)} outputs"
                )
            for name, part in zip(layer.outputs, parts):
                env[name] = part.astype(np.float32)
        else:
            if len(layer.outputs) != 1:
                raise RuntimeError(
                    f"layer {layer.name!r} emits {len(layer.outputs)} "
                    f"outputs but handler returned a single array"
                )
            env[layer.outputs[0]] = np.asarray(result, dtype=np.float32)

        if progress:
            print(f"  [{i:3d}/{len(graph.layers)}] {layer.op:10s} "
                  f"{layer.name!r}  -> shape "
                  f"{env[layer.outputs[0]].shape if layer.outputs else '(multi)'}")

    return {name: env[name] for name in graph.outputs if name in env}
