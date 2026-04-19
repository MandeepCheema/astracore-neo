"""Post-training INT8 quantiser (F1-C2).

Two independent passes, both operate in-place on an NnGraph:

    quantise_weights(graph)
        — For every weight-bearing layer (conv / gemm / matmul),
          computes a per-channel symmetric scale from the FP32 weight
          tensor (axis 0 = output channel). Attaches a QuantParams
          record to NnLayer.quant.

    calibrate_activations(graph, onnx_path, calibration_inputs)
        — Runs each calibration batch through onnxruntime, captures
          every layer's output tensor, tracks running max-abs per
          tensor, and computes a per-tensor symmetric scale. Populates
          NnLayer.quant.input_scale / output_scale on weight-bearing
          layers and stashes the full per-tensor map in
          NnGraph.metadata["activation_scales"] so downstream passes
          (F1-C3/C4) can resolve scales for shape-only ops too.

Quantisation recipe (chosen to hit the <1pp accuracy target on
YOLOv8 while staying compatible with the RTL's INT32-accumulator
writeback):

    - Weights: per-channel symmetric, axis 0.
    - Activations: per-tensor symmetric.
    - Zero-points: 0 (symmetric).
    - Precision: INT8 (range [-127, 127] used symmetrically; the
      hardware's natural INT8 range is [-128, 127] but the symmetric
      recipe avoids the asymmetric endpoint to keep the scale clean).

Why per-channel weights + per-tensor activations: this is the
industry-standard recipe for INT8 post-training quant. The per-channel
scale for weights is applied *in software* at AO readback — the RTL
writes raw INT32 accumulators (no per-channel shift stage in V1 silicon),
so per-channel dequantisation is a free software multiplication on the
host. See docs/astracore_v2_npu_architecture.md §9 for the full
data-path story.

Wraps `src/inference/quantizer.py` for the core scale/clip/round math
so the calibration statistics logic lives in one place.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.inference.quantizer import (
    QuantConfig,
    QuantGranularity,
    QuantPrecision,
    Quantizer,
)

from .nn_graph import (
    GRAN_PER_CHANNEL,
    GRAN_PER_TENSOR,
    NnGraph,
    NnLayer,
    OP_CONV,
    OP_GEMM,
    OP_MATMUL,
    PRECISION_INT2,
    PRECISION_INT4,
    PRECISION_INT8,
    QuantParams,
)
from .onnx_reference import run_reference

# Ops the quantiser attaches weight QuantParams to. Other ops inherit
# their input/output scales from the activation-calibration map; they
# don't have learned parameters to quantise.
_WEIGHT_BEARING_OPS = {OP_CONV, OP_GEMM, OP_MATMUL}

# Symmetric quant ranges per precision. We use (2^(n-1) - 1) so the
# negative endpoint (e.g. -128 for INT8) is never generated — standard
# symmetric recipe that keeps the grid uniform.
_SYM_RANGES = {
    PRECISION_INT8: 127,
    PRECISION_INT4: 7,
    PRECISION_INT2: 1,
}
_INT8_SYM_RANGE = _SYM_RANGES[PRECISION_INT8]   # kept for back-compat


# Activation calibration methods.
CALIB_MAX_ABS    = "max_abs"       # conservative — scale = max(|x|) / qmax
CALIB_PERCENTILE = "percentile"    # production recipe — clip at a high percentile

# Reservoir cap per tensor for percentile calibration. 200k samples ×
# 4 bytes × ~250 tensors ≈ 200 MB peak RAM — still comfortable on any
# dev/CI box. 50k showed ~3-5 pp run-to-run variance on IoU>=0.9
# match rate because the 99.9999 percentile sits at rank 2 of a 50k
# reservoir, which is dominated by a handful of samples. 200k moves
# that to rank 20 — enough to stabilise the estimator.
_PERCENTILE_RESERVOIR_CAP = 200_000

# Default clip percentile. 99.9999 (top 1 in 1,000,000 clipped) is the
# observed sweet spot for YOLOv8-N detection accuracy on the 100-image
# COCO-128 calibration: aggressive enough to ignore true outliers but
# gentle enough to preserve the rare-but-real high-magnitude
# activations that ARE the detection signal (bbox coords in pixels
# up to ~640, high-confidence class scores). Lower values (99.99
# "TensorRT default") over-clip detection peaks and cost ~4 pp
# match rate; see scripts/compare_calibration_methods.py and
# reports/yolov8n_eval_100cal_pct*.json for the measured sweep.
_PERCENTILE = 99.9999


def _sym_range(precision: str) -> int:
    try:
        return _SYM_RANGES[precision]
    except KeyError as e:
        raise NotImplementedError(
            f"symmetric integer quant supports {list(_SYM_RANGES)}; "
            f"got {precision!r}. FP precisions (FP4/FP8) land with F1-A1."
        ) from e


# ---------------------------------------------------------------------------
# Weight quantisation
# ---------------------------------------------------------------------------
def _per_channel_symmetric_scale(w: np.ndarray,
                                  precision: str = PRECISION_INT8) -> np.ndarray:
    """scale[c] = max(|w[c, ...]|) / qmax, clamped to avoid div-by-zero.

    Axis 0 convention (ONNX): conv weights (C_out, C_in/g, kH, kW),
    gemm weights (N_out, N_in) when transB=1.

    qmax is the positive endpoint of the symmetric range for the chosen
    precision: 127 for INT8, 7 for INT4, 1 for INT2.
    """
    qmax = _sym_range(precision)
    c_out = w.shape[0]
    flat = w.reshape(c_out, -1)
    max_abs = np.abs(flat).max(axis=1)
    # Zero-range guard: an all-zero channel produces scale=1.0 so
    # downstream dequant is a no-op rather than a div-by-zero.
    scale = np.where(max_abs > 0, max_abs / qmax, 1.0)
    return scale.astype(np.float32)


def quantise_weights(graph: NnGraph, *,
                     precision: str = PRECISION_INT8,
                     granularity: str = GRAN_PER_CHANNEL) -> NnGraph:
    """Attach a QuantParams record to every weight-bearing layer.

    Idempotent: layers already carrying a QuantParams have their
    weight_scale updated in place but input_scale/output_scale are
    preserved (useful when re-quantising after a graph edit without
    losing calibration).

    Supported precisions: INT8 (F1-C2), INT4 (F1-B2), INT2 (plumbed
    but not acceptance-gated). FP4/FP8 land with F1-A1's RTL.
    """
    qmax = _sym_range(precision)  # raises NotImplementedError for FP*

    for layer in graph.layers:
        if layer.op not in _WEIGHT_BEARING_OPS:
            continue
        if layer.weights is None:
            # MatMul with a non-constant RHS — no weights to quantise.
            # Scale will come from activation calibration.
            continue
        if granularity == GRAN_PER_CHANNEL:
            scale = _per_channel_symmetric_scale(layer.weights, precision)
        else:
            abs_max = float(np.abs(layer.weights).max())
            scalar = abs_max / qmax if abs_max > 0 else 1.0
            scale = np.array(scalar, dtype=np.float32)

        if layer.quant is None:
            layer.quant = QuantParams(
                weight_scale=scale,
                precision=precision,
                granularity=granularity,
            )
        else:
            layer.quant.weight_scale = scale
            layer.quant.granularity = granularity
            layer.quant.precision = precision
    return graph


# ---------------------------------------------------------------------------
# Activation calibration
# ---------------------------------------------------------------------------
def _collect_activation_tensor_names(graph: NnGraph) -> List[str]:
    """Every named tensor we want running-max-abs for: graph inputs,
    graph outputs, and every layer's outputs. We exclude per-layer
    *inputs* because they're either a graph input or a preceding
    layer's output — already covered."""
    names: List[str] = []
    seen = set()
    for name in graph.inputs:
        if name not in seen:
            names.append(name)
            seen.add(name)
    for L in graph.layers:
        for name in L.outputs:
            if name not in seen:
                names.append(name)
                seen.add(name)
    for name in graph.outputs:
        if name not in seen:
            names.append(name)
            seen.add(name)
    return names


def _reservoir_update(reservoirs: Dict[str, np.ndarray],
                       name: str,
                       new_values: np.ndarray,
                       cap: int,
                       rng: np.random.Generator) -> None:
    """Append flattened abs-values into a tensor's reservoir, randomly
    subsampling down to `cap` when it overflows. Keeps the reservoir
    approximately representative of the full calibration distribution
    while bounding memory per tensor."""
    flat = np.abs(new_values.ravel()).astype(np.float32)
    existing = reservoirs.get(name)
    combined = flat if existing is None else np.concatenate([existing, flat])
    if combined.size > cap:
        idx = rng.choice(combined.size, cap, replace=False)
        combined = combined[idx]
    reservoirs[name] = combined


def calibrate_activations(graph: NnGraph,
                          onnx_path: str,
                          calibration_inputs: Iterable[Dict[str, np.ndarray]],
                          *,
                          precision: str = PRECISION_INT8,
                          calibration_method: str = CALIB_MAX_ABS,
                          percentile: float = _PERCENTILE,
                          reservoir_cap: int = _PERCENTILE_RESERVOIR_CAP,
                          seed: int = 0,
                          ) -> NnGraph:
    """Run each batch through ORT, fit a per-tensor symmetric scale,
    and fold it into the QuantParams of every weight-bearing layer.

    Args:
        graph: NnGraph to annotate (mutated in place).
        onnx_path: the original .onnx file to run under ORT.
        calibration_inputs: iterable of {graph_input_name: ndarray}
            batches. Must be finite — we consume it fully.
        calibration_method: `CALIB_MAX_ABS` (default) uses
            `scale = max(|x|) / qmax`. Simple and robust on well-
            behaved distributions, but sensitive to outliers on
            real-image activations (F1-C2 audit H2). `CALIB_PERCENTILE`
            uses `scale = percentile(|x|, p) / qmax`, which ignores
            the top `100 - p`% of samples — the industry-standard
            approach (TensorRT, OpenVINO) for lifting accuracy on
            tailed activation distributions.
        percentile: the clip percentile for `CALIB_PERCENTILE` mode.
            99.99 is conservative (0.01% of samples clipped); 99.9 is
            a touch more aggressive.
        reservoir_cap: max samples retained per tensor for percentile
            computation. Tradeoff: larger is more accurate but more
            RAM. Default 50k × ~250 tensors × 4B ≈ 50 MB peak.
        seed: RNG seed for the reservoir subsampling step.

    Postconditions:
        - `graph.metadata["activation_scales"]` holds a
          {tensor_name: float} map.
        - `graph.metadata["calibration_method"]` / `percentile` /
          `calibration_batches` record the provenance of the scales.
        - Every weight-bearing layer with a QuantParams already set
          has its input_scale and output_scale populated from the map.
    """
    qmax = _sym_range(precision)
    if calibration_method not in (CALIB_MAX_ABS, CALIB_PERCENTILE):
        raise ValueError(
            f"unknown calibration_method {calibration_method!r}; "
            f"expected {CALIB_MAX_ABS!r} or {CALIB_PERCENTILE!r}"
        )

    tensor_names = _collect_activation_tensor_names(graph)
    rng = np.random.default_rng(seed)

    # Max-abs path uses the shared Quantizer's running-min/max stats.
    # Percentile path uses a per-tensor reservoir of |values|.
    stats = Quantizer(
        QuantConfig(
            precision=QuantPrecision.INT8,
            granularity=QuantGranularity.PER_TENSOR,
            symmetric=True,
        )
    )
    reservoirs: Dict[str, np.ndarray] = {}

    def _accumulate(name: str, arr: np.ndarray) -> None:
        if calibration_method == CALIB_MAX_ABS:
            stats.calibrate(name, arr)
        else:
            _reservoir_update(reservoirs, name, arr, reservoir_cap, rng)

    batches_seen = 0
    for batch in calibration_inputs:
        batches_seen += 1
        intermediates = [n for n in tensor_names if n not in batch]
        run = run_reference(onnx_path, batch,
                            intermediate_names=intermediates)
        for name, arr in batch.items():
            _accumulate(name, arr)
        for name, arr in run.activations.items():
            _accumulate(name, arr)
        for name, arr in run.outputs.items():
            _accumulate(name, arr)

    if batches_seen == 0:
        raise ValueError(
            "calibrate_activations: calibration_inputs was empty — "
            "need at least one batch to fit scales"
        )

    # Compute per-tensor scales.
    activation_scales: Dict[str, float] = {}
    if calibration_method == CALIB_MAX_ABS:
        for name, s in stats.iter_stats():
            abs_max = max(abs(s.min_val), abs(s.max_val))
            activation_scales[name] = (
                abs_max / qmax if abs_max > 0 else 1.0
            )
    else:
        for name, r in reservoirs.items():
            if r.size == 0:
                activation_scales[name] = 1.0
                continue
            # Percentile of abs values — ignores the top outlier fraction
            # so the bulk of the distribution gets more quant resolution.
            clip_val = float(np.percentile(r, percentile))
            activation_scales[name] = (
                clip_val / qmax if clip_val > 0 else 1.0
            )

    graph.metadata["activation_scales"] = activation_scales
    graph.metadata["calibration_batches"] = batches_seen
    graph.metadata["calibration_method"] = calibration_method
    if calibration_method == CALIB_PERCENTILE:
        graph.metadata["calibration_percentile"] = percentile

    # Fold input/output scales into every weight-bearing layer's
    # QuantParams. If quantise_weights() hasn't run yet we'd be
    # wrong to synthesise one here — raise a pointer at the caller.
    missing_qp = []
    missing_scales: List[str] = []
    for L in graph.layers:
        if L.op not in _WEIGHT_BEARING_OPS:
            continue
        if L.weights is None:
            continue
        if L.quant is None:
            missing_qp.append(L.name)
            continue
        if L.inputs:
            in_name = L.inputs[0]
            if in_name in activation_scales:
                L.quant.input_scale = activation_scales[in_name]
            else:
                missing_scales.append(in_name)
        if L.outputs:
            out_name = L.outputs[0]
            if out_name in activation_scales:
                L.quant.output_scale = activation_scales[out_name]
            else:
                missing_scales.append(out_name)
    if missing_qp:
        raise RuntimeError(
            f"calibrate_activations: layers {missing_qp[:3]} "
            f"(and {max(0, len(missing_qp)-3)} more) have no QuantParams — "
            f"run quantise_weights() before calibrate_activations()."
        )
    if missing_scales:
        graph.metadata.setdefault("calibration_warnings", []).extend(
            f"tensor {n!r} not seen during calibration — scale defaulted to 1.0"
            for n in missing_scales
        )
    return graph


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------
def quantise_model(graph: NnGraph,
                   onnx_path: str,
                   calibration_inputs: Iterable[Dict[str, np.ndarray]],
                   *,
                   precision: str = PRECISION_INT8,
                   granularity: str = GRAN_PER_CHANNEL,
                   calibration_method: str = CALIB_MAX_ABS,
                   percentile: float = _PERCENTILE,
                   ) -> NnGraph:
    """Weight quant + activation calibration in one call.

    Each call re-runs activation calibration; any input_scale /
    output_scale values set by a previous call are replaced. Re-run
    with a fresh graph or re-load from ONNX if you need to compare
    two calibration sets on the same weights.

    `calibration_method=CALIB_PERCENTILE` applies 99.99%-ile clipping
    to activation scales — the industry-standard fix for tailed real-
    image activation distributions (see F1-C2 audit H2).
    """
    quantise_weights(graph, precision=precision, granularity=granularity)
    calibrate_activations(graph, onnx_path, calibration_inputs,
                          precision=precision,
                          calibration_method=calibration_method,
                          percentile=percentile)
    return graph


def make_seeded_calibration_set(input_name: str,
                                 shape: Sequence[int],
                                 n_batches: int = 20,
                                 *,
                                 seed: int = 0,
                                 low: float = 0.0,
                                 high: float = 1.0,
                                 ) -> List[Dict[str, np.ndarray]]:
    """Deterministic bootstrap calibration set for F1-C2 v1.

    Real image-based calibration arrives with F1-T1. Until then this
    produces uniform-noise batches so the scale machinery can be
    exercised end-to-end. The range [0, 1] matches YOLOv8's standard
    pre-norm input domain.
    """
    rng = np.random.default_rng(seed)
    batches = []
    for i in range(n_batches):
        x = rng.uniform(low, high, size=tuple(shape)).astype(np.float32)
        batches.append({input_name: x})
    return batches


# ---------------------------------------------------------------------------
# Weight dequantiser (useful for downstream compiler lowering + for the
# fake-quant validation harness).
# ---------------------------------------------------------------------------
def fake_quantise_weights(w: np.ndarray, scale: np.ndarray,
                           precision: str = PRECISION_INT8) -> np.ndarray:
    """Round-trip weights through symmetric int-N and back to FP32. For
    per-channel scales axis 0 is the channel axis.

    `precision` selects the symmetric range (INT8=±127, INT4=±7, INT2=±1).
    Default is INT8 for back-compat with the F1-C2 call sites.
    """
    qmax = _sym_range(precision)
    if scale.ndim == 0:
        s = float(scale)
        if s == 0:
            return np.zeros_like(w, dtype=np.float32)
        q = np.clip(np.round(w / s), -qmax, qmax)
        return (q * s).astype(np.float32)
    # Per-channel on axis 0 — vectorised via broadcast. Zero-scale
    # channels are forced to zero output (same semantics as the
    # previous Python-loop implementation).
    s_bcast = scale.reshape(scale.shape[0], *([1] * (w.ndim - 1)))
    safe_s = np.where(s_bcast == 0, 1.0, s_bcast)
    q = np.clip(np.round(w / safe_s), -qmax, qmax)
    out = np.where(s_bcast == 0, 0.0, q * s_bcast)
    return out.astype(np.float32)


def fake_quantise_activation(a: np.ndarray, scale: float,
                              precision: str = PRECISION_INT8) -> np.ndarray:
    """Round-trip activations through per-tensor symmetric int-N and back
    to FP32. `precision` selects the range (INT8=±127, INT4=±7, INT2=±1)."""
    qmax = _sym_range(precision)
    if scale == 0:
        return np.zeros_like(a, dtype=np.float32)
    q = np.clip(np.round(a / scale), -qmax, qmax)
    return (q * scale).astype(np.float32)
