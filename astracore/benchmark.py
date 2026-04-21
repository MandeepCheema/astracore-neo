"""Model-zoo benchmark harness.

``astracore bench --model X.onnx --backend <name>`` loads a model,
runs ``n_iter`` inferences on the chosen backend with random input,
and reports standardised KPIs via ``BackendReport``.

Usable from code too:

    from astracore.benchmark import benchmark_model
    report = benchmark_model(model_path="resnet50.onnx", backend="onnxruntime")
    print(report.as_markdown_row())
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import time

import numpy as np

from astracore.backend import BackendReport
from astracore.registry import get_backend

# Make sure the built-in backends register before anyone calls get_backend().
import astracore.backends  # noqa: F401


def _parse_shape(spec: Optional[str]) -> Optional[Tuple[int, ...]]:
    if not spec:
        return None
    return tuple(int(s) for s in spec.split(","))


def _build_backend(name: str, options: Optional[Dict[str, Any]] = None):
    cls = get_backend(name)
    if isinstance(cls, type):
        return cls(**(options or {}))
    # Non-class factory (rare) — can't accept options.
    if options:
        raise ValueError(
            f"backend {name!r} is not a class; cannot pass options"
        )
    return cls


def _load_model_as_onnx(path: Path):
    import onnx
    return onnx.load(str(path))


# Seq-length for dynamic transformer dims. 256 matches BERT-Squad, 8 is a
# short GPT-2 prompt — realistic for latency measurement without wasting time
# on long context.
_SEQ_LEN_FALLBACK = 8


def _substitute_shape(dims, *, batch: int = 1, seq_len: int = _SEQ_LEN_FALLBACK) -> Tuple[int, ...]:
    """Turn ONNX dim_values + dim_params into concrete ints.

    Rules:
     * static value > 0 kept as-is
     * dim_param containing 'batch' or index 0 → batch
     * dim_param containing 'seq' / 'length' → seq_len
     * anything else dynamic → 1
    """
    out = []
    for i, d in enumerate(dims):
        if d.dim_value and d.dim_value > 0:
            out.append(d.dim_value)
            continue
        name = (d.dim_param or "").lower()
        if i == 0 or "batch" in name:
            out.append(batch)
        elif "seq" in name or "length" in name or "sequence" in name:
            out.append(seq_len)
        else:
            # Common transformer default for the second dim is sequence
            # length; fall back to seq_len for index 1, otherwise 1.
            out.append(seq_len if i == 1 else 1)
    return tuple(out)


def _default_input_shape(model) -> Tuple[int, ...]:
    """Read shape of the first input; substitute concrete values for
    dynamic dimensions using common transformer conventions."""
    first = model.graph.input[0]
    return _substitute_shape(first.type.tensor_type.shape.dim)


_ONNX_DTYPE = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    5: np.int16,
    6: np.int32,
    7: np.int64,
    9: np.bool_,
    10: np.float16,
    11: np.float64,
    16: np.float32,   # bfloat16 — represent as float32 for host memory
}


# Conservative value ranges for well-known transformer input names.
# HF Optimum exports (and the ONNX model zoo BERT-Squad / GPT-2 entries)
# use these names canonically, so a substring match picks the right
# range for each. Order matters — the first substring match wins, so
# more specific names must come before their `_mask` / `_ids` suffixes.
#
# Why the ranges differ:
#   * ``attention_mask`` / ``*_mask`` feed softmax-masking paths and are
#     binary by construction.
#   * ``token_type_ids`` / ``segment_ids`` feed an embedding table whose
#     vocabulary is ``type_vocab_size`` (2 for every stock BERT variant).
#     Random values in the 30k range crash ORT's Gather bounds check on
#     recent opsets (observed on BERT-large-squad, opset 17) and produce
#     garbage on older opsets (bert-squad-10, opset 10).
#   * ``position_ids`` indexes the positional embedding table sized
#     ``max_position_embeddings`` — 512 is the conservative default
#     across BERT / DistilBERT / RoBERTa; GPT-2 uses 1024.
#   * ``input_ids`` uses the full token vocabulary; 30k covers BERT's
#     30522 and stays safely under GPT-2's 50257.
_INT_INPUT_NAME_RANGES: Tuple[Tuple[str, Tuple[int, int]], ...] = (
    # (substring of input name, (low_inclusive, high_exclusive))
    ("token_type_ids", (0, 2)),     # BERT family, type_vocab_size=2
    ("segment_ids",    (0, 2)),     # bert-squad-10 / older exports
    ("_type_ids",      (0, 2)),     # decoder_token_type_ids, ...
    ("attention_mask", (0, 2)),
    ("input_mask",     (0, 2)),     # bert-squad-10 naming
    ("pad_mask",       (0, 2)),
    ("_mask",          (0, 2)),     # catch-all: causal_mask, etc.
    ("position_ids",   (0, 512)),
    ("input_ids",      (0, 30_000)),
    ("decoder_input_ids", (0, 30_000)),
)

_INT_DEFAULT_RANGE: Tuple[int, int] = (0, 30_000)


def _int_range_for_input(name: str) -> Tuple[int, int]:
    """Pick a safe [low, high) integer range for a graph input by name.

    Falls back to the default token-vocab range when the name doesn't
    match a known transformer convention. The default is intentionally
    wide — it's right for ``input_ids``-shaped inputs and doesn't crash
    anything whose valid range is at least 30k.
    """
    low_name = name.lower()
    for sub, rng in _INT_INPUT_NAME_RANGES:
        if sub in low_name:
            return rng
    return _INT_DEFAULT_RANGE


def _gen_input_for(onnx_input, *, override_shape: Optional[Tuple[int, ...]] = None,
                   rng: np.random.Generator) -> np.ndarray:
    """Produce a plausible tensor for one ONNX graph input."""
    tt = onnx_input.type.tensor_type
    dtype = _ONNX_DTYPE.get(tt.elem_type, np.float32)
    shape = override_shape or _substitute_shape(tt.shape.dim)

    # Defensive: reject shapes with non-positive dims. Happens when a
    # caller passes a bad override or a model has weird dim metadata.
    if any(int(d) <= 0 for d in shape):
        raise ValueError(
            f"Input {onnx_input.name!r} resolved to invalid shape "
            f"{shape}; all dims must be positive."
        )

    if np.issubdtype(dtype, np.integer):
        low, high = _int_range_for_input(onnx_input.name)
        return rng.integers(low=low, high=high, size=shape, dtype=dtype)
    if dtype is np.bool_:
        return rng.integers(0, 2, size=shape).astype(np.bool_)
    return rng.standard_normal(shape).astype(dtype, copy=False)


def benchmark_model(
    model_path: Path,
    *,
    backend: str = "onnxruntime",
    backend_options: Optional[Dict[str, Any]] = None,
    precision: str = "INT8",
    sparsity: str = "dense",
    n_iter: int = 1,
    input_shape: Optional[str] = None,
    warmup: int = 1,
) -> BackendReport:
    """Benchmark a model on a named backend.

    Returns a :class:`BackendReport` populated with latency + TOPS. The
    caller converts to JSON / markdown / dashboard-row as needed.
    """
    model_path = Path(model_path)
    model = _load_model_as_onnx(model_path)

    override = _parse_shape(input_shape)
    # Multi-input models (BERT-Squad has 4, GPT-2 has 1 but int64). Populate
    # every graph input with a plausible tensor of the right dtype/shape.
    # For the first input we honour the --input-shape override if given.
    rng = np.random.default_rng(0)
    # Old ONNX opsets listed weight initialisers in graph.input too. Only
    # populate entries that are NOT in the initialiser set (i.e. the real
    # runtime inputs).
    init_names = {t.name for t in model.graph.initializer}
    real_inputs = [inp for inp in model.graph.input if inp.name not in init_names]
    inputs: Dict[str, np.ndarray] = {}
    for i, inp in enumerate(real_inputs):
        shape_override = override if (i == 0 and override) else None
        inputs[inp.name] = _gen_input_for(inp, override_shape=shape_override, rng=rng)

    be = _build_backend(backend, backend_options)
    # If the backend accepts concrete_shapes (OrtBackend does), feed the
    # input shapes we just resolved. This lets the MAC estimator walk
    # transformer graphs cleanly.
    concrete_shapes = {name: tuple(arr.shape) for name, arr in inputs.items()}
    try:
        program = be.compile(model, precision=precision, sparsity=sparsity,
                             concrete_shapes=concrete_shapes)
    except TypeError:
        program = be.compile(model, precision=precision, sparsity=sparsity)

    # Warmup — avoids first-run-is-slow artefacts.
    for _ in range(max(0, warmup)):
        be.run(program, inputs)

    # Timed runs.
    be._last.n_inferences = 0
    be._last.wall_s_total = 0.0
    for _ in range(n_iter):
        be.run(program, inputs)

    report = be.report_last()
    report.model = model_path.name
    return report


def benchmark_matrix(model_paths, *, backends=("onnxruntime",),
                     precisions=("INT8",), sparsities=("dense",),
                     n_iter: int = 1):
    """Yield (model, backend, precision, sparsity) -> BackendReport for all combos."""
    for mp in model_paths:
        for be in backends:
            for pr in precisions:
                for sp in sparsities:
                    try:
                        yield benchmark_model(
                            Path(mp), backend=be, precision=pr,
                            sparsity=sp, n_iter=n_iter,
                        )
                    except Exception as exc:  # pragma: no cover
                        err = BackendReport(
                            backend=be, model=str(Path(mp).name),
                            precision=pr, sparsity=sp,
                            notes=f"failed: {exc!r}",
                        )
                        yield err
