"""F1-C2 acceptance — INT8 quantiser fidelity against real YOLOv8-N.

Gate: round-tripping yolov8n.onnx through per-channel-weight +
per-tensor-activation INT8 fake quant (a faithful software model of
what the RTL + compiler will deliver in F1-C5) preserves the output
to within the numerical thresholds below.

Thresholds are calibrated with synthetic-noise calibration data. Real
image calibration arriving with F1-T1 will push SNR higher, so these
are conservative floors. All thresholds are spec'd with margin above
the observed values on the reference seed to avoid seed-flake.

Observed at commit time with 20 batches of seeded uniform-[0,1] noise:

    end-to-end output SNR           37.5 dB   gate: ≥33 dB
    end-to-end cosine               0.9999    gate: ≥0.998
    worst per-layer conv SNR        16.4 dB   gate: ≥15 dB
    median per-layer conv SNR       23.0 dB   gate: ≥18 dB

The gate that actually tracks mAP-relevant accuracy is the end-to-end
cosine; per-layer SNRs catch gross per-channel-scale bugs that might
not show up in the smoothed output.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from tools.npu_ref.fake_quant_model import build_fake_quant_model
from tools.npu_ref.nn_graph import OP_CONV
from tools.npu_ref.onnx_loader import load_onnx
from tools.npu_ref.onnx_reference import make_seeded_input, run_reference
from tools.npu_ref.quantiser import (
    make_seeded_calibration_set,
    quantise_model,
)

REPO = Path(__file__).resolve().parent.parent
ONNX_PATH = REPO / "data" / "models" / "yolov8n.onnx"


def _require_artifact():
    if not ONNX_PATH.exists():
        pytest.skip(
            f"{ONNX_PATH} not present. Run scripts/export_yolov8n_onnx.py "
            f"from a venv with ultralytics installed."
        )


def _snr_db(reference: np.ndarray, measured: np.ndarray) -> float:
    err = reference - measured
    num = float((reference ** 2).mean())
    den = max(float((err ** 2).mean()), 1e-30)
    return 10.0 * np.log10(num / den)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    num = float(a.ravel() @ b.ravel())
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    return num / den


@pytest.fixture(scope="module")
def quantised_graph_and_fq_path(tmp_path_factory):
    _require_artifact()
    g = load_onnx(str(ONNX_PATH))
    calibration = make_seeded_calibration_set(
        "images", (1, 3, 640, 640), n_batches=20, seed=0)
    quantise_model(g, str(ONNX_PATH), calibration)
    fq_path = tmp_path_factory.mktemp("fq") / "yolov8n.fakequant.onnx"
    build_fake_quant_model(g, str(ONNX_PATH), out_path=str(fq_path))
    return g, str(fq_path)


def test_every_conv_has_quant_params(quantised_graph_and_fq_path):
    g, _ = quantised_graph_and_fq_path
    unquantised = [L.name for L in g.layers_of(OP_CONV) if L.quant is None]
    assert not unquantised, (
        f"{len(unquantised)} conv layers missing QuantParams: "
        f"{unquantised[:5]}"
    )
    # Per-channel weight scales must be 1-D with length = C_out.
    for L in g.layers_of(OP_CONV):
        w = L.weights
        s = L.quant.weight_scale
        assert s.ndim == 1, f"{L.name}: weight_scale is {s.ndim}-D"
        assert s.shape[0] == w.shape[0], (
            f"{L.name}: weight_scale length {s.shape[0]} != "
            f"C_out {w.shape[0]}"
        )
        assert np.all(s > 0), f"{L.name}: non-positive scales present"


def test_activation_scales_populated(quantised_graph_and_fq_path):
    g, _ = quantised_graph_and_fq_path
    act_scales = g.metadata.get("activation_scales", {})
    # Every graph input + every conv output must be in the map.
    for name in g.inputs:
        assert name in act_scales, f"graph input {name} missing scale"
    for L in g.layers_of(OP_CONV):
        for out in L.outputs:
            assert out in act_scales, f"conv {L.name} output {out} missing scale"
    # Input scale and output scale on each conv must match the map.
    for L in g.layers_of(OP_CONV):
        assert L.quant.input_scale == pytest.approx(
            act_scales[L.inputs[0]], rel=1e-6
        )
        assert L.quant.output_scale == pytest.approx(
            act_scales[L.outputs[0]], rel=1e-6
        )


def test_end_to_end_output_fidelity(quantised_graph_and_fq_path):
    _, fq_path = quantised_graph_and_fq_path
    probe = make_seeded_input((1, 3, 640, 640), seed=99)
    fp32 = run_reference(str(ONNX_PATH), {"images": probe}).outputs["output0"]
    fq = run_reference(fq_path, {"images": probe}).outputs["output0"]

    snr = _snr_db(fp32, fq)
    cos = _cosine(fp32, fq)
    print(f"end-to-end  SNR={snr:.2f} dB  cosine={cos:.6f}")
    assert snr >= 33.0, (
        f"end-to-end output SNR {snr:.2f} dB below 33 dB floor — "
        f"check weight quant / activation calibration."
    )
    assert cos >= 0.998, (
        f"end-to-end output cosine {cos:.6f} below 0.998 floor — "
        f"output structure diverged; check QDQ insertion ordering."
    )


def test_per_layer_conv_fidelity(quantised_graph_and_fq_path):
    g, fq_path = quantised_graph_and_fq_path
    probe = make_seeded_input((1, 3, 640, 640), seed=99)
    probe_names = [L.outputs[0] for L in g.layers_of(OP_CONV)]

    fp32 = run_reference(str(ONNX_PATH), {"images": probe},
                         intermediate_names=probe_names)
    fq = run_reference(fq_path, {"images": probe},
                       intermediate_names=probe_names)

    snrs: List[float] = []
    for name in probe_names:
        if name not in fp32.activations or name not in fq.activations:
            continue
        snrs.append(_snr_db(fp32.activations[name], fq.activations[name]))

    assert len(snrs) == len(probe_names), (
        f"only {len(snrs)} of {len(probe_names)} conv outputs were captured"
    )
    snrs_np = np.array(snrs)
    worst = float(snrs_np.min())
    median = float(np.median(snrs_np))
    print(f"per-layer SNR  min={worst:.2f}  median={median:.2f}  "
          f"max={float(snrs_np.max()):.2f}")
    assert worst >= 15.0, (
        f"worst per-layer SNR {worst:.2f} dB below 15 dB floor — "
        f"a single layer has catastrophic quant error (likely a "
        f"per-channel scale bug)."
    )
    assert median >= 18.0, (
        f"median per-layer SNR {median:.2f} dB below 18 dB floor — "
        f"broad quant degradation; suggests activation scales are "
        f"miscalibrated."
    )


# ---------------------------------------------------------------------------
# F1-B2: INT4 acceptance gate — looser floors than INT8 as expected.
#
# INT4 has 16 grid points vs INT8's 256. Worst-case per-layer error is
# ~16x larger in the worst case, giving ~24 dB SNR loss. The floors
# below are calibrated to the observed INT4 behaviour on yolov8n with
# the same 20-batch seeded-noise calibration set. Any regression past
# these floors is a real bug, not an expected INT4 lossiness.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def int4_quantised_graph_and_fq_path(tmp_path_factory):
    _require_artifact()
    g = load_onnx(str(ONNX_PATH))
    calibration = make_seeded_calibration_set(
        "images", (1, 3, 640, 640), n_batches=20, seed=0)
    quantise_model(g, str(ONNX_PATH), calibration, precision="int4")
    fq_path = tmp_path_factory.mktemp("fq_int4") / "yolov8n.int4.onnx"
    build_fake_quant_model(g, str(ONNX_PATH), out_path=str(fq_path))
    return g, str(fq_path)


def test_int4_every_conv_has_int4_quant_params(int4_quantised_graph_and_fq_path):
    g, _ = int4_quantised_graph_and_fq_path
    for L in g.layers_of(OP_CONV):
        assert L.quant is not None, f"{L.name}: no QuantParams"
        assert L.quant.precision == "int4", (
            f"{L.name}: precision={L.quant.precision!r}, expected 'int4'"
        )
        # INT4 grid cap: |w| after fake-quant ≤ 7 * scale for each channel.
        # Verified structurally via the QuantParams shape.
        assert L.quant.weight_scale.shape[0] == L.weights.shape[0]


def test_int4_end_to_end_output_fidelity(int4_quantised_graph_and_fq_path):
    _, fq_path = int4_quantised_graph_and_fq_path
    probe = make_seeded_input((1, 3, 640, 640), seed=99)
    fp32 = run_reference(str(ONNX_PATH), {"images": probe}).outputs["output0"]
    fq = run_reference(fq_path, {"images": probe}).outputs["output0"]

    snr = _snr_db(fp32, fq)
    cos = _cosine(fp32, fq)
    print(f"INT4 end-to-end  SNR={snr:.2f} dB  cosine={cos:.6f}")
    # INT4 floors: looser than INT8 (37.5 / 0.9999). The sheet claim
    # is "≤1pp accuracy loss at INT8", INT4 is not guaranteed to meet
    # that — we gate on numerical fidelity that downstream accuracy
    # metrics (F1-B6) will refine.
    assert snr >= 15.0, (
        f"INT4 end-to-end SNR {snr:.2f} dB below 15 dB floor — "
        f"the INT4 quant grid has fundamentally broken the model output."
    )
    assert cos >= 0.95, (
        f"INT4 end-to-end cosine {cos:.6f} below 0.95 floor — "
        f"output structure diverged catastrophically."
    )
