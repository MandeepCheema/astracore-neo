"""Post-ingestion neural-net graph IR for the compiler (F1-C1 and downstream).

Place in the pipeline:

    .onnx ──[onnx_loader]──▶ NnGraph (FP32 weights, op-typed)
                             │
                             ├─[F1-C2 quantiser]──▶ NnGraph  (adds quant scales)
                             │
                             ├─[F1-C3/C4 tiler]───▶ compiler.Program
                             │                      (list of LoadWeight /
                             │                       LoadActivation / RunTile
                             │                       / ReadAO over Matmul tiles)
                             │
                             └─[perf_model]──────▶ cycle / fps estimate

This module is intentionally free of onnx / onnxruntime / torch imports so
it can be used by components (perf models, test harnesses) that don't want
the ONNX toolchain on their import path.

Why a new IR rather than extending one that already exists:
  - layer_spec.Layer — structural GEMM dims only, no weights; keep it
    untouched so ViT / LLaMA / BEVFormer traces and perf_model continue
    to work.
  - yolo_trace.Layer — legacy CNN-specific accounting, no weights.
  - compiler.Matmul   — tile-level, INT8, RTL-packed. Downstream of
    tiling and quantisation; the wrong abstraction for a whole model.

NnLayer carries the FP32 weights that survive through quantisation and
tiling; it is the layer-level stable IR the rest of the compiler stack
lowers from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Supported op identifiers. Stringly-typed deliberately (handlers switch on
# these) but listed explicitly so a typo raises at the handler dispatch
# rather than silently mis-routing.
# ---------------------------------------------------------------------------
OP_CONV       = "conv"
OP_GEMM       = "gemm"       # ONNX Gemm; flattens to a linear layer
OP_MATMUL     = "matmul"
OP_ADD        = "add"
OP_SUB        = "sub"
OP_MUL        = "mul"
OP_DIV        = "div"
OP_CONCAT     = "concat"
OP_SPLIT      = "split"
OP_RESIZE     = "resize"     # nearest/linear upsample
OP_MAXPOOL    = "maxpool"
OP_AVGPOOL    = "avgpool"
OP_SIGMOID    = "sigmoid"
OP_SILU       = "silu"       # fused Sigmoid+Mul; YOLOv8 uses this
OP_RELU       = "relu"
OP_SOFTMAX    = "softmax"
OP_RESHAPE    = "reshape"
OP_TRANSPOSE  = "transpose"
OP_SLICE      = "slice"
# Transformer ops (F1-B1). Add GELU / LayerNorm / RMSNorm / RotaryEmbedding /
# MultiHeadAttention so ViT / BERT / LLaMA / Swin exports load without
# NotImplementedError. The activation ops (GELU) lower to the existing AFU
# LUT; the normalisation + attention ops lower to the V2 activation path
# (F1-A4) + MHSA tile (F1-A5) once those land in RTL.
OP_GELU       = "gelu"
OP_LAYERNORM  = "layernorm"
OP_RMSNORM    = "rmsnorm"
OP_ROTARY_EMB = "rotary_embedding"
OP_MHA        = "multi_head_attention"

SUPPORTED_OPS = frozenset({
    OP_CONV, OP_GEMM, OP_MATMUL, OP_ADD, OP_SUB, OP_MUL, OP_DIV,
    OP_CONCAT, OP_SPLIT, OP_RESIZE, OP_MAXPOOL, OP_AVGPOOL, OP_SIGMOID,
    OP_SILU, OP_RELU, OP_SOFTMAX, OP_RESHAPE, OP_TRANSPOSE, OP_SLICE,
    OP_GELU, OP_LAYERNORM, OP_RMSNORM, OP_ROTARY_EMB, OP_MHA,
})


# ---------------------------------------------------------------------------
# Quantisation params (F1-C2). Attached to NnLayer.quant once the
# quantiser has run.
# ---------------------------------------------------------------------------
# Precision tags the downstream tiler uses to set cfg_precision_mode
# on the NPU RTL. "int8" is the F1-C5 target; "int4"/"int2" are
# future-work placeholders so the IR doesn't need a breaking change
# when they arrive.
PRECISION_INT8 = "int8"
PRECISION_INT4 = "int4"
PRECISION_INT2 = "int2"

# Granularity: per-channel only makes sense along a layer's
# output-channel axis (axis 0 for ONNX conv weights (C_out, C_in, kH, kW)
# and gemm transB=1 weights (out, in)). Per-tensor is one scalar.
GRAN_PER_TENSOR  = "per_tensor"
GRAN_PER_CHANNEL = "per_channel"


@dataclass
class QuantParams:
    """Per-layer symmetric INT8 quantisation record.

    `weight_scale` carries the scale used to quantise the learned
    weights. For conv/gemm under the standard per-channel recipe this
    is a 1-D array of length C_out. For per-tensor it's a 0-D array.
    Symmetric quant fixes `weight_zero_point = 0`; the field is kept
    for asymmetric future work.

    `input_scale` and `output_scale` are per-tensor floats derived
    from activation calibration. `input_scale` is the scale the
    *previous* layer's output uses and that this layer consumes;
    `output_scale` is the scale this layer's output will be dequantised
    with before the next layer quantises it. Keeping both explicit (vs
    inferring from the graph) lets the compiler verify the producer/
    consumer scales agree during lowering.

    The compiler reconstructs the dequantised output of a tile via
    `real = acc * weight_scale * input_scale` where `acc` is the
    INT32 accumulator value the RTL writes to AO. For per-channel
    weight_scale, this multiplication is per output column.
    """
    weight_scale: np.ndarray   # shape () for per-tensor, (C_out,) for per-channel
    weight_zero_point: int = 0
    input_scale: float = 1.0
    output_scale: float = 1.0
    input_zero_point: int = 0
    output_zero_point: int = 0
    precision: str = PRECISION_INT8
    granularity: str = GRAN_PER_CHANNEL

    def __post_init__(self) -> None:
        if self.weight_scale is None:
            raise ValueError("QuantParams.weight_scale is required")
        if not isinstance(self.weight_scale, np.ndarray):
            self.weight_scale = np.asarray(self.weight_scale, dtype=np.float32)
        if self.weight_scale.dtype != np.float32:
            self.weight_scale = self.weight_scale.astype(np.float32)
        if self.precision not in (PRECISION_INT8, PRECISION_INT4, PRECISION_INT2):
            raise ValueError(f"unknown precision {self.precision!r}")
        if self.granularity == GRAN_PER_CHANNEL and self.weight_scale.ndim != 1:
            raise ValueError(
                f"per-channel granularity requires a 1-D weight_scale, "
                f"got shape {self.weight_scale.shape}"
            )
        if self.granularity == GRAN_PER_TENSOR and self.weight_scale.ndim != 0:
            raise ValueError(
                f"per-tensor granularity requires a 0-D weight_scale, "
                f"got shape {self.weight_scale.shape}"
            )


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------
Shape = Tuple[int, ...]


@dataclass
class NnLayer:
    """One op in the post-ONNX graph.

    Tensor identity is name-based (matches ONNX convention): `inputs` and
    `outputs` hold the *activation* tensor names (learned parameters like
    weights and bias are pulled out to dedicated fields). `in_shapes`
    and `out_shapes` resolve names to shapes for any consumer that needs
    to compute sizes without walking the whole graph.

    For conv layers `attrs` holds the kernel / stride / pad / groups /
    dilation tuples and (if the original ONNX graph fused an activation
    via post-hoc peephole) a `fused_activation` entry. Shape-only ops
    (reshape, transpose, slice) store their parameters in `attrs`.
    """
    name: str
    op: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    in_shapes: Dict[str, Shape] = field(default_factory=dict)
    out_shapes: Dict[str, Shape] = field(default_factory=dict)
    attrs: Dict[str, Any] = field(default_factory=dict)
    weights: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None
    # Populated by F1-C2 quantiser. None before quantisation.
    quant: Optional[QuantParams] = None

    def __post_init__(self) -> None:
        if self.op not in SUPPORTED_OPS:
            raise ValueError(
                f"NnLayer {self.name!r}: unsupported op {self.op!r}. "
                f"Known ops: {sorted(SUPPORTED_OPS)}"
            )
        if self.weights is not None and self.weights.dtype != np.float32:
            raise TypeError(
                f"NnLayer {self.name!r}: weights must be float32, got "
                f"{self.weights.dtype}. Cast at the loader, not downstream."
            )
        if self.bias is not None and self.bias.dtype != np.float32:
            raise TypeError(
                f"NnLayer {self.name!r}: bias must be float32, got "
                f"{self.bias.dtype}."
            )

    @property
    def primary_input_shape(self) -> Optional[Shape]:
        """Shape of `inputs[0]`, or None if not known. Shortcut for the
        common single-input op case."""
        if not self.inputs:
            return None
        return self.in_shapes.get(self.inputs[0])

    @property
    def primary_output_shape(self) -> Optional[Shape]:
        if not self.outputs:
            return None
        return self.out_shapes.get(self.outputs[0])


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
@dataclass
class NnGraph:
    """A full model: ordered layer list plus external tensor descriptors.

    `layers` is topologically ordered (the order in which the loader
    emitted them; ONNX graphs are already topo-sorted by contract).
    `inputs` / `outputs` are the *graph-level* I/O tensors — i.e. the
    data the host pushes into the model and reads out at the end. They
    do NOT include layer-internal tensors or parameters.
    """
    layers: List[NnLayer] = field(default_factory=list)
    inputs: Dict[str, Shape] = field(default_factory=dict)
    outputs: Dict[str, Shape] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.layers)

    def layers_of(self, op: str) -> List[NnLayer]:
        return [L for L in self.layers if L.op == op]

    def total_weight_bytes(self) -> int:
        """Sum of weight + bias parameter bytes at FP32. For the worst-case
        memory footprint at INT8 divide by 4 (or use the quantised sizes
        after F1-C2 runs)."""
        total = 0
        for L in self.layers:
            if L.weights is not None:
                total += L.weights.nbytes
            if L.bias is not None:
                total += L.bias.nbytes
        return total


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    W = np.zeros((16, 3, 3, 3), dtype=np.float32)
    b = np.zeros((16,), dtype=np.float32)
    layer = NnLayer(
        name="stem",
        op=OP_CONV,
        inputs=["input"],
        outputs=["stem_out"],
        in_shapes={"input": (1, 3, 640, 640)},
        out_shapes={"stem_out": (1, 16, 320, 320)},
        attrs={"kernel": (3, 3), "stride": (2, 2), "pad": (1, 1, 1, 1),
               "groups": 1, "dilation": (1, 1)},
        weights=W,
        bias=b,
    )
    assert layer.primary_input_shape == (1, 3, 640, 640)
    assert layer.primary_output_shape == (1, 16, 320, 320)

    g = NnGraph(
        layers=[layer],
        inputs={"input": (1, 3, 640, 640)},
        outputs={"stem_out": (1, 16, 320, 320)},
        metadata={"producer": "self-check"},
    )
    assert len(g) == 1
    assert g.layers_of(OP_CONV) == [layer]
    assert g.total_weight_bytes() == W.nbytes + b.nbytes

    # QuantParams smoke
    qp = QuantParams(weight_scale=np.array([0.01, 0.02, 0.03, 0.04] * 4,
                                            dtype=np.float32))
    assert qp.weight_scale.shape == (16,)
    assert qp.granularity == GRAN_PER_CHANNEL
    qp_pt = QuantParams(weight_scale=np.array(0.01, dtype=np.float32),
                        granularity=GRAN_PER_TENSOR)
    assert qp_pt.weight_scale.shape == ()

    # Negative tests
    try:
        NnLayer(name="bad", op="banana")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported op should raise")

    try:
        QuantParams(weight_scale=np.array([0.1], dtype=np.float32),
                    granularity=GRAN_PER_TENSOR)
    except ValueError:
        pass
    else:
        raise AssertionError("per-tensor with 1-D scale should raise")

    try:
        NnLayer(name="bad_w", op=OP_CONV,
                weights=np.zeros((1,), dtype=np.float64))
    except TypeError:
        pass
    else:
        raise AssertionError("non-float32 weights should raise")

    print("nn_graph self-check PASS")
