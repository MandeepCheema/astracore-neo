"""``astracore quantise`` — one-command PTQ pipeline.

Wraps the existing ``tools.npu_ref`` quantiser + fake-quant ONNX emitter
into a single function + CLI subcommand so customers get a quantised
artefact in seconds instead of writing their own calibration loop.

Inputs:
    - FP32 ONNX file
    - number of calibration samples (synthetic unless caller supplies real ones)
    - precision / granularity / calibration method
Outputs:
    - fake-quant ONNX (runs on plain ORT, weights round-tripped through
      INT8, activations wrapped in QuantizeLinear/DequantizeLinear)
    - JSON manifest with FP32-vs-INT8 SNR, cosine, max-abs error on
      a held-out probe input

This is a *fake-quant* pipeline — the output runs on FP32 backends but
produces INT8-equivalent numerics. Customers then re-export to their
target silicon's INT8 format (TensorRT plan, SNPE DLC, etc).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Lazy imports — defer heavy deps until the CLI actually runs.
# ---------------------------------------------------------------------------

def _import_deps():
    from tools.npu_ref.onnx_loader import load_onnx
    from tools.npu_ref.fusion import fuse_silu
    from tools.npu_ref.quantiser import (
        quantise_model, make_seeded_calibration_set,
    )
    from tools.npu_ref.fake_quant_model import build_fake_quant_model
    from tools.npu_ref.onnx_reference import run_reference
    return (load_onnx, fuse_silu, quantise_model,
            make_seeded_calibration_set, build_fake_quant_model,
            run_reference)


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------

@dataclass
class QuantiseManifest:
    source_onnx: str
    output_onnx: str
    source_sha256: str
    output_sha256: str
    source_bytes: int
    output_bytes: int
    precision: str
    granularity: str
    calibration_method: str
    percentile: float
    calibration_samples: int
    calibration_seed: int
    input_name: str
    input_shape: List[int]
    snr_db: float
    cosine: float
    max_abs_err: float
    output_tensor: str
    output_shape: List[int]
    wall_s: float
    engine: str = "internal"      # "internal" (tools.npu_ref) or "ort"
    drift_error: str = ""         # non-empty when drift measurement failed

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _probe_input(model_path: Path, seed: int = 999) -> Dict[str, np.ndarray]:
    """Seeded probe tensor dict for FP32-vs-INT8 drift measurement.

    Distinct seed from the calibration set — we want to measure drift
    on data the quantiser did NOT see. Builds one tensor per real
    (non-initialiser) input, with dtype + shape honoured. Multi-input
    transformers (BERT-Squad, GPT-2) produce the same 4- or 1-element
    feed dict they expect at runtime.
    """
    import onnx
    model_proto = onnx.load(str(model_path))
    init_names = {t.name for t in model_proto.graph.initializer}
    real_inputs = [i for i in model_proto.graph.input
                   if i.name not in init_names]
    rng = np.random.default_rng(seed)
    _ONNX_DTYPE = {
        1: np.float32, 2: np.uint8, 3: np.int8, 5: np.int16,
        6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16,
        11: np.float64,
    }
    feed: Dict[str, np.ndarray] = {}
    for inp in real_inputs:
        dtype = _ONNX_DTYPE.get(inp.type.tensor_type.elem_type, np.float32)
        dims = inp.type.tensor_type.shape.dim
        shape = tuple((d.dim_value if d.dim_value and d.dim_value > 0 else 1)
                      for d in dims) or (1,)
        if np.issubdtype(dtype, np.integer):
            feed[inp.name] = rng.integers(
                low=0, high=30_000, size=shape, dtype=dtype,
            )
        elif dtype == np.bool_:
            feed[inp.name] = rng.integers(0, 2, size=shape).astype(np.bool_)
        else:
            feed[inp.name] = rng.uniform(
                0.0, 1.0, size=shape,
            ).astype(dtype, copy=False)
    return feed


def _first_real_input(onnx_model) -> Tuple[str, Tuple[int, ...]]:
    """Return (name, static shape) of the first non-initialiser input."""
    inits = {t.name for t in onnx_model.graph.initializer}
    real = [inp for inp in onnx_model.graph.input if inp.name not in inits]
    if not real:
        raise ValueError("no non-initialiser inputs found")
    inp = real[0]
    dims = inp.type.tensor_type.shape.dim
    shape: List[int] = []
    for i, d in enumerate(dims):
        if d.dim_value and d.dim_value > 0:
            shape.append(int(d.dim_value))
        elif i == 0:
            shape.append(1)
        else:
            shape.append(1)
    return inp.name, tuple(shape)


def _measure_drift(fp32_onnx: str, fq_onnx: str,
                   probe: Dict[str, np.ndarray],
                   output_tensor: Optional[str] = None,
                   ) -> Tuple[float, float, float, str, List[int]]:
    """FP32 vs fake-quant SNR / cosine / max-abs on one probe input."""
    from tools.npu_ref.onnx_reference import run_reference
    fp32 = run_reference(fp32_onnx, probe)
    fq   = run_reference(fq_onnx, probe)

    if output_tensor is None:
        output_tensor = next(iter(fp32.outputs.keys()))
    a = np.asarray(fp32.outputs[output_tensor], dtype=np.float64)
    b = np.asarray(fq.outputs[output_tensor], dtype=np.float64)
    diff = a - b
    sig = float(np.linalg.norm(a))
    nse = float(np.linalg.norm(diff))
    snr = 20 * np.log10(sig / max(nse, 1e-10)) if sig > 0 else 0.0
    cos = float(
        (a.ravel() @ b.ravel())
        / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-10)
    )
    max_abs = float(np.max(np.abs(diff)))
    return snr, cos, max_abs, output_tensor, list(a.shape)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _quantise_internal(
    model_path: Path, output_path: Path, *,
    cal_samples: int, cal_seed: int,
    precision: str, granularity: str,
    calibration_method: str, percentile: float,
    fuse_silu_layers: bool,
) -> Tuple[str, Tuple[int, ...]]:
    """Our tools.npu_ref pipeline. Strict loader — fails on unsupported ops.

    Returns (input_name, input_shape) for drift measurement.
    """
    (load_onnx, fuse_silu, quantise_model,
     make_seeded_calibration_set, build_fake_quant_model,
     _run_reference) = _import_deps()

    g = load_onnx(str(model_path))
    if fuse_silu_layers:
        try:
            fuse_silu(g)
        except Exception:
            pass

    import onnx
    model_proto = onnx.load(str(model_path))
    input_name, input_shape = _first_real_input(model_proto)

    cal = make_seeded_calibration_set(
        input_name, input_shape, n_batches=cal_samples, seed=cal_seed,
    )
    quantise_model(
        g, str(model_path), cal,
        precision=precision, granularity=granularity,
        calibration_method=calibration_method, percentile=percentile,
    )
    build_fake_quant_model(g, str(model_path), out_path=str(output_path))
    return input_name, input_shape


def _quantise_ort(
    model_path: Path, output_path: Path, *,
    cal_samples: int, cal_seed: int,
    granularity: str,
) -> Tuple[str, Tuple[int, ...]]:
    """Fallback using ``onnxruntime.quantization.quantize_static``.

    Produces standard QDQ-format INT8 ONNX that every major runtime
    (TensorRT, SNPE, OpenVINO, CoreML, DirectML) accepts. Broader op
    coverage than our internal loader.

    Opset < 11 models are first version-upgraded to opset 13 because
    ``DequantizeLinear`` only supports per-channel axis from opset 13
    onward — attempting per-channel quantisation on opset 7 SqueezeNet
    otherwise produces an invalid graph at load time.
    """
    from onnxruntime.quantization import (
        quantize_static, QuantType, QuantFormat, CalibrationDataReader,
    )
    import onnx
    model_proto = onnx.load(str(model_path))
    input_name, input_shape = _first_real_input(model_proto)

    # Determine opset of the default (empty-domain) graph.
    opset = 11
    for oi in model_proto.opset_import:
        if oi.domain in ("", "ai.onnx"):
            opset = int(oi.version)
            break

    # Version-upgrade old models so per-channel quantisation is valid.
    input_for_quant = str(model_path)
    upgrade_tmp: Optional[Path] = None
    if opset < 13:
        from onnx import version_converter
        try:
            upgraded = version_converter.convert_version(model_proto, 13)
            upgrade_tmp = output_path.with_suffix(".opset13.onnx")
            onnx.save(upgraded, str(upgrade_tmp))
            input_for_quant = str(upgrade_tmp)
        except Exception:
            # Version-converter can't handle every graph; fall back to
            # per-tensor quantisation on the original.
            pass

    # Build per-input dtype + shape map for multi-input / int-token models
    # (BERT-Squad = 4 inputs; GPT-2 = 1 input but int64). Non-primary
    # inputs get zeros of the right dtype so the calibration reader
    # feeds a valid forward pass.
    init_names = {t.name for t in model_proto.graph.initializer}
    real_inputs = [i for i in model_proto.graph.input
                   if i.name not in init_names]

    _ONNX_DTYPE = {
        1: np.float32, 2: np.uint8, 3: np.int8, 5: np.int16,
        6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16,
        11: np.float64,
    }

    def _concrete_shape(inp) -> Tuple[int, ...]:
        dims = inp.type.tensor_type.shape.dim
        out = []
        for i, d in enumerate(dims):
            if d.dim_value and d.dim_value > 0:
                out.append(int(d.dim_value))
            else:
                out.append(1)
        return tuple(out) if out else (1,)

    input_dtypes: Dict[str, type] = {}
    input_shapes: Dict[str, Tuple[int, ...]] = {}
    for inp in real_inputs:
        dtype = _ONNX_DTYPE.get(inp.type.tensor_type.elem_type, np.float32)
        input_dtypes[inp.name] = dtype
        input_shapes[inp.name] = _concrete_shape(inp)

    rng = np.random.default_rng(cal_seed)

    def _random_feed() -> Dict[str, np.ndarray]:
        feed: Dict[str, np.ndarray] = {}
        for name, shp in input_shapes.items():
            dtype = input_dtypes[name]
            if np.issubdtype(dtype, np.integer):
                # Conservative token range — BERT vocab 30522, GPT-2 50257;
                # 0..30000 is valid for both.
                feed[name] = rng.integers(
                    low=0, high=30_000, size=shp, dtype=dtype,
                )
            elif dtype == np.bool_:
                feed[name] = rng.integers(0, 2, size=shp).astype(np.bool_)
            else:
                feed[name] = rng.uniform(
                    0.0, 1.0, size=shp,
                ).astype(dtype, copy=False)
        return feed

    class _Reader(CalibrationDataReader):
        def __init__(self, n):
            self._batches = [_random_feed() for _ in range(n)]
            self._it = iter(self._batches)

        def get_next(self):
            return next(self._it, None)

        def rewind(self):
            self._it = iter(self._batches)

    # On pre-opset-13 models that we couldn't upgrade, force per-tensor.
    per_channel = (granularity == "per_channel"
                   and not (opset < 13 and upgrade_tmp is None))
    quantize_static(
        model_input=input_for_quant,
        model_output=str(output_path),
        calibration_data_reader=_Reader(cal_samples),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
    )
    if upgrade_tmp is not None and upgrade_tmp.exists():
        upgrade_tmp.unlink()
    return input_name, input_shape


def quantise(
    *,
    model_path: Path,
    output_path: Optional[Path] = None,
    cal_samples: int = 100,
    cal_seed: int = 0,
    precision: str = "int8",
    granularity: str = "per_channel",
    calibration_method: str = "max_abs",
    percentile: float = 99.9999,
    fuse_silu_layers: bool = True,
    engine: str = "auto",
) -> QuantiseManifest:
    """Post-training-quantise an ONNX model end-to-end.

    ``engine``:
      * ``"internal"`` — our ``tools.npu_ref`` pipeline. Strict loader;
        optimised for the F1 target. Fails on Dropout/BatchNorm/Clip/
        Identity/Shape — ops our loader doesn't implement.
      * ``"ort"`` — ``onnxruntime.quantization.quantize_static``. Broad
        op coverage; emits standard QDQ format. Lower SNR on average
        but runs on every runtime.
      * ``"auto"`` (default) — try internal first; fall back to ORT on
        loader failure.

    Returns the manifest; caller is responsible for persisting it.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"source ONNX not found: {model_path}")
    if output_path is None:
        output_path = model_path.with_suffix(".int8.onnx")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    chosen_engine = engine
    input_name = ""
    input_shape: Tuple[int, ...] = ()

    def _try_internal():
        return _quantise_internal(
            model_path, output_path,
            cal_samples=cal_samples, cal_seed=cal_seed,
            precision=precision, granularity=granularity,
            calibration_method=calibration_method, percentile=percentile,
            fuse_silu_layers=fuse_silu_layers,
        )

    def _try_ort():
        if precision != "int8":
            raise ValueError(f"ORT engine only supports INT8; got {precision}")
        return _quantise_ort(
            model_path, output_path,
            cal_samples=cal_samples, cal_seed=cal_seed,
            granularity=granularity,
        )

    if engine == "internal":
        input_name, input_shape = _try_internal()
        chosen_engine = "internal"
    elif engine == "ort":
        input_name, input_shape = _try_ort()
        chosen_engine = "ort"
    elif engine == "auto":
        try:
            input_name, input_shape = _try_internal()
            chosen_engine = "internal"
        except Exception as exc:
            # Loader / quantiser failure on the internal path — try ORT.
            input_name, input_shape = _try_ort()
            chosen_engine = "ort"
    else:
        raise ValueError(f"unknown engine {engine!r}; use internal|ort|auto")

    # Measure FP32-vs-INT8 drift on a held-out probe input. For some
    # models (transformers with int64 token-id inputs) ORT's quantize_
    # static rewrites the input type, breaking our direct drift probe —
    # in that case we still have a valid quantised artefact, we just
    # record drift as unmeasurable rather than failing the whole run.
    snr, cos, max_abs, out_tensor, out_shape = 0.0, 0.0, 0.0, "", []
    drift_error = ""
    try:
        probe = _probe_input(model_path)
        snr, cos, max_abs, out_tensor, out_shape = _measure_drift(
            str(model_path), str(output_path), probe,
        )
    except Exception as exc:
        drift_error = f"{type(exc).__name__}: {exc}"

    wall = time.perf_counter() - t0

    return QuantiseManifest(
        source_onnx=str(model_path),
        output_onnx=str(output_path),
        source_sha256=_file_sha256(model_path)[:16],
        output_sha256=_file_sha256(output_path)[:16],
        source_bytes=model_path.stat().st_size,
        output_bytes=output_path.stat().st_size,
        precision=precision,
        granularity=granularity,
        calibration_method=calibration_method,
        percentile=percentile,
        calibration_samples=cal_samples,
        calibration_seed=cal_seed,
        input_name=input_name,
        input_shape=list(input_shape),
        snr_db=round(snr, 2),
        cosine=round(cos, 6),
        max_abs_err=round(max_abs, 4),
        output_tensor=out_tensor,
        output_shape=out_shape,
        wall_s=round(wall, 2),
        engine=chosen_engine,
        drift_error=drift_error,
    )


def write_manifest(manifest: QuantiseManifest, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.as_dict(), indent=2), encoding="utf-8",
    )
