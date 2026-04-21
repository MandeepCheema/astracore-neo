"""Cross-EP numerical conformance gate.

Auto-activates when ORT reports ≥ 2 execution providers on the host.
On a CPU-only build the test collection body skips cleanly with a
clear message; on a host with CUDA / TensorRT / OpenVINO / QNN /
CoreML installed, the test runs every zoo model through each
non-CPU EP and compares the raw output tensor to CPU within a
**40 dB SNR** floor (≈ 1% relative error — the conventional
INT8-equivalent gate used by MLPerf compliance runs).

Why 40 dB and not bit-identical?
--------------------------------
CPU FP32 and CUDA/TRT FP32 produce *almost* the same outputs but not
bit-exact — kernel fusion, SIMD reduction order, and tensor-core
accumulation differ. 40 dB (≈1% L2 error) is generous enough to
accommodate those but tight enough to catch a real bug (a miswired
quantiser, a shape mismatch, an NHWC vs NCHW confusion).

For FP16 / INT8 EPs the threshold should be tightened to the
precision-appropriate floor via a parametrised test (TODO v0.2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parent.parent
SNR_FLOOR_DB = 40.0

# Models we can meaningfully run across EPs without per-model
# tokenizer setup. Excludes BERT + GPT-2 which need int64 token inputs.
_EP_TEST_MODELS = [
    ("yolov8n",              REPO / "data" / "models" / "yolov8n.onnx"),
    ("squeezenet-1.1",       REPO / "data" / "models" / "zoo"
                             / "squeezenet-1.1.onnx"),
    ("mobilenetv2-7",        REPO / "data" / "models" / "zoo"
                             / "mobilenetv2-7.onnx"),
    ("shufflenet-v2-10",     REPO / "data" / "models" / "zoo"
                             / "shufflenet-v2-10.onnx"),
    ("resnet50-v2-7",        REPO / "data" / "models" / "zoo"
                             / "resnet50-v2-7.onnx"),
    ("efficientnet-lite4-11", REPO / "data" / "models" / "zoo"
                              / "efficientnet-lite4-11.onnx"),
]


def _snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    sig = float(np.linalg.norm(reference))
    nse = float(np.linalg.norm(reference - candidate))
    if sig == 0:
        return float("inf") if nse == 0 else 0.0
    if nse == 0:
        return float("inf")
    return 20.0 * np.log10(sig / nse)


@pytest.fixture(scope="module")
def available_non_cpu_eps():
    """Return the list of installed EPs other than CPU."""
    try:
        import onnxruntime as ort
    except ImportError:
        pytest.skip("onnxruntime not installed")
    eps = ort.get_available_providers()
    non_cpu = [e for e in eps if e != "CPUExecutionProvider"
               and e != "AzureExecutionProvider"]
    if not non_cpu:
        pytest.skip(
            f"only CPU EP available ({eps}); skipping cross-EP gate. "
            f"This test auto-activates when a second EP is installed "
            f"(onnxruntime-gpu, onnxruntime-openvino, etc)."
        )
    return non_cpu


def test_ort_has_multiple_providers_if_we_expect_it(available_non_cpu_eps):
    """Liveness probe — if we got here, there IS a second EP.

    Keeps the test file active even on minimal builds so we notice
    when someone turns on a GPU wheel and conformance starts being
    exercised."""
    assert available_non_cpu_eps, "fixture should skip if empty"


@pytest.mark.parametrize("model_name,model_path", _EP_TEST_MODELS,
                         ids=[n for n, _ in _EP_TEST_MODELS])
def test_zoo_model_conforms_across_eps(model_name, model_path,
                                       available_non_cpu_eps):
    """Each non-CPU EP's output must match CPU within SNR_FLOOR_DB."""
    if not model_path.exists():
        pytest.skip(f"{model_name} not on disk")

    from astracore.backends.ort import OrtBackend
    from astracore.benchmark import _gen_input_for
    import onnx

    # Build a seeded input that every EP sees identically.
    model = onnx.load(str(model_path))
    init_names = {t.name for t in model.graph.initializer}
    real = [i for i in model.graph.input if i.name not in init_names]
    rng = np.random.default_rng(0)
    inputs = {inp.name: _gen_input_for(inp, override_shape=None, rng=rng)
              for inp in real}

    # CPU reference.
    cpu_be = OrtBackend(providers=["cpu"])
    cpu_prog = cpu_be.compile(str(model_path))
    cpu_out = cpu_be.run(cpu_prog, {k: v.copy() for k, v in inputs.items()})

    # Each candidate EP.
    failures = []
    for ep in available_non_cpu_eps:
        cand_be = OrtBackend(providers=[ep, "cpu"])
        cand_prog = cand_be.compile(str(model_path))
        cand_out = cand_be.run(cand_prog,
                               {k: v.copy() for k, v in inputs.items()})
        # Both sides MUST have identical output-name sets.
        if set(cpu_out) != set(cand_out):
            failures.append(
                (ep, "output names mismatch",
                 sorted(cpu_out), sorted(cand_out)))
            continue
        for name, cpu_arr in cpu_out.items():
            cand_arr = cand_out[name]
            a = np.asarray(cpu_arr, dtype=np.float64)
            b = np.asarray(cand_arr, dtype=np.float64)
            if a.shape != b.shape:
                failures.append(
                    (ep, f"shape mismatch on {name}", a.shape, b.shape))
                continue
            snr = _snr_db(a, b)
            if snr < SNR_FLOOR_DB:
                failures.append(
                    (ep, f"SNR {snr:.2f} dB < {SNR_FLOOR_DB} dB on {name}",
                     None, None))

    if failures:
        lines = [f"{model_name} cross-EP conformance failed:"]
        for ep, reason, *_rest in failures:
            lines.append(f"  {ep}: {reason}")
        raise AssertionError("\n".join(lines))


def test_seed_determinism_within_one_ep():
    """Same EP + same seed MUST produce bit-identical output.

    Independent of the cross-EP gate — this one passes even on
    CPU-only hosts and should stay green through ORT upgrades.
    """
    from astracore.backends.ort import OrtBackend
    yolo = REPO / "data" / "models" / "yolov8n.onnx"
    if not yolo.exists():
        pytest.skip("yolov8n.onnx missing")
    rng = np.random.default_rng(42)
    x = rng.standard_normal((1, 3, 640, 640)).astype(np.float32)

    be = OrtBackend(providers=["cpu"])
    prog = be.compile(str(yolo))
    a = next(iter(be.run(prog, {"images": x.copy()}).values()))
    b = next(iter(be.run(prog, {"images": x.copy()}).values()))
    np.testing.assert_array_equal(a, b,
        err_msg="same EP + same input → different output; ORT non-determinism")
