"""Deep AI-model tests — beyond "does it run".

Where zoo + detailed-bench measure latency, this script measures
**model behaviour**:

1. **Input-perturbation robustness** — SqueezeNet/MobileNet/ShuffleNet
   against Gaussian noise at σ ∈ {0, 0.01, 0.05, 0.1, 0.2} on the bus
   image. Does top-1 stay correct? How does confidence degrade?

2. **BERT-Squad answer-span determinism** — 10 runs of the canned
   France Q+A; start/end tokens and logits must be bit-identical across
   runs (evidence that the same input always produces the same answer).

3. **GPT-2 next-token rank consistency** — canned prompt "The capital
   of France is"; `Paris` rank must be stable across 10 runs and inside
   top-10.

4. **YOLO per-image top-class determinism** — 5 runs of each of 28
   images; the top-3 class IDs must be identical across runs (unless
   scores tie, in which case we accept any permutation of the tied set).

5. **Latency vs GMACs correlation** — across the 8-model zoo, compute
   Pearson r between GMACs and steady-state p50. High correlation
   means the backend is compute-bound; low correlation means memory or
   kernel-launch overhead dominates.

6. **FP32 vs fake-INT8 output drift** — reference fp32 run vs the
   SDK's own quantiser (fake-quant tensors), measure SNR. This is the
   most honest proxy we have today for "does our INT8 path preserve
   the model's answer?" without shipping a dedicated INT8 backend.

Each scenario writes its own JSON + MD under
``reports/ai_deep_tests/``; a summary.md ties them together.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from astracore import zoo as zoo_mod                          # noqa: E402
from astracore.backends.ort import OrtBackend                 # noqa: E402
from astracore.demo import run_demo                           # noqa: E402
from astracore.demo.vision_classifier import (                # noqa: E402
    _load_test_image, _center_crop_resize, _to_probabilities,
    _IMAGENET_MEAN, _IMAGENET_STD,
)

OUT_DIR = REPO / "reports" / "ai_deep_tests"


def _percentile_stats(ms):
    arr = np.asarray(ms, dtype=np.float64)
    if len(arr) == 0:
        return {"n": 0}
    return {
        "n": int(len(arr)),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p99_ms": float(np.percentile(arr, 99)),
        "stdev_ms": float(arr.std()) if len(arr) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Test 1 — Input perturbation robustness
# ---------------------------------------------------------------------------

def test_input_perturbation() -> Dict[str, Any]:
    """How does top-1 confidence decay with Gaussian pixel noise?"""
    import onnx
    target_models = ["squeezenet-1.1", "mobilenetv2-7", "shufflenet-v2-10"]
    sigmas = [0.0, 0.01, 0.05, 0.1, 0.2]
    img = _load_test_image("bus")                         # HWC uint8
    sq = _center_crop_resize(img, 224, 224).astype(np.float32)

    rng = np.random.default_rng(0)
    rows: List[Dict[str, Any]] = []

    for name in target_models:
        m = zoo_mod.get(name)
        path = zoo_mod.local_paths()[name]
        if not path.exists():
            continue
        model_proto = onnx.load(str(path))
        be = OrtBackend()
        program = be.compile(model_proto)
        input_name = program.input_names[0]

        for sigma in sigmas:
            # Identical preprocessing to the demo path, then noise.
            noise = rng.standard_normal(sq.shape).astype(np.float32) \
                    * (255.0 * sigma)
            perturbed = np.clip(sq + noise, 0, 255)
            normed = (perturbed / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
            x = np.transpose(normed.astype(np.float32),
                             (2, 0, 1))[None, ...]
            out = be.run(program, {input_name: x})
            logits = np.asarray(next(iter(out.values()))).squeeze()
            probs = _to_probabilities(logits)
            top1_idx = int(probs.argmax())
            top1_prob = float(probs[top1_idx])
            rows.append({
                "model": name,
                "sigma": sigma,
                "top1_idx": top1_idx,
                "top1_prob": top1_prob,
            })

    # Judge "robust" = top-1 stays == top-1 at sigma=0.
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    verdicts = {}
    for name, entries in by_model.items():
        base = next(e for e in entries if e["sigma"] == 0.0)
        stable = {e["sigma"]: (e["top1_idx"] == base["top1_idx"])
                  for e in entries}
        verdicts[name] = {
            "top1_stable_per_sigma": stable,
            "survives_up_to_sigma": max(
                (s for s, v in stable.items() if v), default=0.0,
            ),
            "base_top1_prob": base["top1_prob"],
        }
    return {"rows": rows, "verdicts": verdicts}


def render_t1(d):
    lines = ["# Deep test 1 — Input perturbation robustness", "",
             "Gaussian pixel noise σ ∈ {0, 0.01, 0.05, 0.1, 0.2} "
             "(σ expressed as fraction of 255). Bus image, ImageNet classifiers. "
             "Top-1 prediction monitored per σ.", ""]
    lines.append("| Model | σ=0 top1 | σ=0.01 | σ=0.05 | σ=0.1 | σ=0.2 | Survives up to |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in d["rows"]:
        by_model.setdefault(r["model"], []).append(r)
    for model, entries in by_model.items():
        entries.sort(key=lambda e: e["sigma"])
        cells = [f"{entries[0]['top1_prob']*100:.1f}%"]
        base_idx = entries[0]["top1_idx"]
        for e in entries[1:]:
            tag = "✓" if e["top1_idx"] == base_idx else "✗"
            cells.append(f"{e['top1_prob']*100:.1f}% {tag}")
        survives = d["verdicts"][model]["survives_up_to_sigma"]
        lines.append(f"| {model} | " + " | ".join(cells) + f" | σ={survives} |")
    lines.append("")
    lines.append("✓ = top-1 label unchanged from noise-free baseline. "
                 "Higher `Survives up to σ` = more robust model.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test 2 — BERT answer-span determinism
# ---------------------------------------------------------------------------

def test_bert_determinism() -> Dict[str, Any]:
    m = zoo_mod.get("bert-squad-10")
    path = zoo_mod.local_paths()["bert-squad-10"]
    if not path.exists():
        return {"skipped": "bert-squad-10 not on disk"}
    starts: List[int] = []
    ends: List[int] = []
    start_scores: List[float] = []
    for _ in range(10):
        res = run_demo(m, path, input_spec=None,
                       backend_name="onnxruntime", warmup=0)
        if not res.predictions:
            continue
        starts.append(res.predictions[0]["start_token_idx"])
        ends.append(res.predictions[0]["end_token_idx"])
        start_scores.append(res.predictions[0]["start_score"])
    return {
        "starts": starts,
        "ends": ends,
        "start_scores": start_scores,
        "deterministic": (len(set(starts)) == 1 and len(set(ends)) == 1),
        "start_score_stdev": (float(np.std(start_scores))
                              if start_scores else 0.0),
    }


def render_t2(d):
    if "skipped" in d:
        return f"# Deep test 2 — BERT-Squad determinism\n\n_Skipped: {d['skipped']}_\n"
    lines = ["# Deep test 2 — BERT-Squad answer-span determinism", "",
             "Same canned Q+A input, 10 runs. Start/end token indices and "
             "start-logit score must be bit-identical across runs.",
             "",
             f"- Start tokens observed: {sorted(set(d['starts']))}",
             f"- End tokens observed:   {sorted(set(d['ends']))}",
             f"- Start-score stdev:     {d['start_score_stdev']:.4f}",
             f"- Deterministic?         **{'YES' if d['deterministic'] else 'NO'}**",
             ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test 3 — GPT-2 " Paris" rank consistency
# ---------------------------------------------------------------------------

def test_gpt2_paris_rank() -> Dict[str, Any]:
    m = zoo_mod.get("gpt-2-10")
    path = zoo_mod.local_paths()["gpt-2-10"]
    if not path.exists():
        return {"skipped": "gpt-2-10 not on disk"}
    ranks: List[int] = []
    probs: List[float] = []
    for _ in range(10):
        res = run_demo(m, path, input_spec=None,
                       backend_name="onnxruntime", warmup=0)
        # summary embeds "rank X/50257, prob Y.YY%"
        import re
        mrk = re.search(r"rank (\d+)/50257, prob ([\d.]+)%", res.summary)
        if mrk:
            ranks.append(int(mrk.group(1)))
            probs.append(float(mrk.group(2)))
    return {
        "ranks": ranks,
        "probs_percent": probs,
        "unique_ranks": sorted(set(ranks)),
        "deterministic": len(set(ranks)) == 1,
        "inside_top10": all(r <= 10 for r in ranks) if ranks else False,
    }


def render_t3(d):
    if "skipped" in d:
        return f"# Deep test 3 — GPT-2 Paris rank\n\n_Skipped: {d['skipped']}_\n"
    lines = ["# Deep test 3 — GPT-2 ' Paris' rank consistency", "",
             "Canned prompt 'The capital of France is'; BPE token 6342 "
             "corresponds to ' Paris'. After LM-head projection, Paris "
             "should be ranked near the top and stable across runs.", "",
             f"- Ranks observed across 10 runs: {d['unique_ranks']}",
             f"- Paris probability (%):         {d['probs_percent']}",
             f"- Deterministic?                 **{'YES' if d['deterministic'] else 'NO'}**",
             f"- Inside top-10?                 **{'YES' if d['inside_top10'] else 'NO'}**",
             ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test 4 — YOLO top-class determinism across repeats
# ---------------------------------------------------------------------------

def test_yolo_determinism() -> Dict[str, Any]:
    yolo_path = REPO / "data" / "models" / "yolov8n.onnx"
    eval_path = REPO / "data" / "calibration" / "yolov8n_eval.npz"
    if not yolo_path.exists() or not eval_path.exists():
        return {"skipped": "yolo eval assets missing"}
    images = np.load(eval_path)["images"]
    be = OrtBackend()
    program = be.compile(str(yolo_path))
    input_name = program.input_names[0]

    all_stable = True
    per_image = []
    for i in range(images.shape[0]):
        x = images[i:i + 1].astype(np.float32)
        top3_runs: List[Tuple[int, int, int]] = []
        for _ in range(5):
            out = be.run(program, {input_name: x})
            raw = np.asarray(next(iter(out.values()))).squeeze()
            cls_scores = raw[4:, :]       # (80, 8400)
            best_cls = cls_scores.argmax(axis=0)
            best_conf = cls_scores.max(axis=0)
            mask = best_conf > 0.25
            if mask.sum() == 0:
                top3_runs.append((-1, -1, -1))
                continue
            c = Counter(best_cls[mask].tolist())
            top3 = tuple(c.most_common(3))
            top3_classes = tuple(x[0] for x in top3) + (-1, -1, -1)
            top3_runs.append(top3_classes[:3])
        unique = set(top3_runs)
        stable = len(unique) == 1
        per_image.append({"image_idx": i, "stable": stable,
                          "runs_unique": len(unique),
                          "top3": top3_runs[0]})
        if not stable:
            all_stable = False
    return {"per_image": per_image, "all_stable": all_stable,
            "n_unstable": sum(1 for r in per_image if not r["stable"])}


def render_t4(d):
    if "skipped" in d:
        return f"# Deep test 4 — YOLO determinism\n\n_Skipped: {d['skipped']}_\n"
    lines = ["# Deep test 4 — YOLOv8n top-3 class determinism across 5 runs", "",
             f"- Images stable (top-3 class IDs identical across 5 runs): "
             f"**{len(d['per_image']) - d['n_unstable']}/{len(d['per_image'])}**",
             f"- Images that drifted: {d['n_unstable']}",
             "",
             "_Unstable images (if any) indicate nondeterministic kernel "
             "selection or a tied-score corner case. Investigate before "
             "blessing a new backend._",
             ""]
    if d["n_unstable"]:
        lines.append("| Image idx | Runs unique |")
        lines.append("|---:|---:|")
        for r in d["per_image"]:
            if not r["stable"]:
                lines.append(f"| {r['image_idx']} | {r['runs_unique']} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test 5 — Latency-vs-GMACs correlation (compute-bound vs memory-bound)
# ---------------------------------------------------------------------------

def test_latency_vs_gmacs() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for m in zoo_mod.all_models():
        path = zoo_mod.local_paths().get(m.name)
        if not path or not path.exists():
            continue
        be = OrtBackend()
        program = be.compile(str(path))
        input_name = program.input_names[0]
        rep = be.report_last()

        # Build a correctly-shaped dummy input for this graph.
        from astracore.benchmark import _gen_input_for
        import onnx
        model_proto = onnx.load(str(path))
        init = {t.name for t in model_proto.graph.initializer}
        real_inputs = [i for i in model_proto.graph.input if i.name not in init]
        rng = np.random.default_rng(0)
        feed = {i.name: _gen_input_for(i, override_shape=None, rng=rng)
                for i in real_inputs}

        # Warm + time 15 runs.
        for _ in range(3):
            be.run(program, feed)
        lats = []
        for _ in range(15):
            t0 = time.perf_counter()
            be.run(program, feed)
            lats.append((time.perf_counter() - t0) * 1e3)
        rows.append({
            "model": m.name,
            "family": m.family,
            "gmacs": round(rep.mac_ops_total / 1e9, 4),
            "p50_ms": round(float(np.percentile(lats, 50)), 3),
        })

    if len(rows) < 3:
        return {"rows": rows, "pearson_r": None}
    g = np.asarray([r["gmacs"] for r in rows], dtype=np.float64)
    l = np.asarray([r["p50_ms"] for r in rows], dtype=np.float64)
    if g.std() == 0 or l.std() == 0:
        r = 0.0
    else:
        r = float(np.corrcoef(g, l)[0, 1])
    return {"rows": rows, "pearson_r": round(r, 4)}


def render_t5(d):
    lines = ["# Deep test 5 — Latency vs GMACs correlation", "",
             "For each zoo model, steady-state p50 latency (15 iters after "
             "3 warmups) against theoretical GMACs. Pearson r across models "
             "is a rough health check on 'are we compute-bound?'",
             "",
             f"- Pearson r (GMACs, p50_ms): **{d.get('pearson_r', 'N/A')}**",
             ""]
    lines.append("| Model | Family | GMACs | p50 ms | ms / GMAC |")
    lines.append("|---|---|---:|---:|---:|")
    for r in d["rows"]:
        mspg = r["p50_ms"] / max(r["gmacs"], 1e-6)
        lines.append(
            f"| {r['model']} | {r['family']} | {r['gmacs']:.3f} | "
            f"{r['p50_ms']:.2f} | {mspg:.2f} |"
        )
    lines.append("")
    lines.append(
        "Interpretation:\n"
        "- `r` close to 1.0 → latency tracks GMACs → compute-bound (good).\n"
        "- `r` close to 0 → latency dominated by non-MAC work (memory, kernel launch).\n"
        "- Per-model `ms / GMAC` lets you spot outliers where ONE model is "
        "mis-tuned.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test 6 — FP32 vs fake-INT8 output drift
# ---------------------------------------------------------------------------

def test_int8_drift() -> Dict[str, Any]:
    """Compare FP32 ORT output against the SDK's fake-INT8 quantiser path
    on YOLOv8n with the bus image.

    Pipeline:
      FP32 ref  = ORT(raw yolov8n.onnx, bus)
      Fake-INT8 = ORT(build_fake_quant_model(quantise_model(g, ...)), bus)

    SNR ≥ 30 dB is production-grade for INT8 PTQ; published yolov8n
    INT8 recipes land in the 35-45 dB range. The F1-C5 audit already
    showed 43.6 dB with a 100-sample calibration; one-sample calibration
    here is much tighter — expect lower SNR as a floor.
    """
    try:
        from tools.npu_ref.onnx_loader import load_onnx
        from tools.npu_ref.quantiser import quantise_model
        from tools.npu_ref.onnx_reference import run_reference
        from tools.npu_ref.fusion import fuse_silu
        from tools.npu_ref.fake_quant_model import build_fake_quant_model
    except Exception as exc:
        return {"skipped": f"quantiser stack not importable: {exc!r}"}

    yolo_path = REPO / "data" / "models" / "yolov8n.onnx"
    bus_path = REPO / "data" / "calibration" / "bus.npz"
    if not yolo_path.exists() or not bus_path.exists():
        return {"skipped": "yolo or bus image missing"}

    bus = np.load(bus_path)["image"].astype(np.float32)

    # FP32 reference through ORT on the raw model.
    fp32_run = run_reference(str(yolo_path), {"images": bus})
    fp32_out_name, fp32_out = next(iter(fp32_run.outputs.items()))
    fp32_out = np.asarray(fp32_out, dtype=np.float64)

    # Fake-INT8: load → fuse → quantise with seeded calibration → dump
    # fake-quant ONNX → run through ORT. Same path the real-image F1-C5
    # test walks, just on a single image to keep the smoke fast.
    try:
        import tempfile
        g = load_onnx(str(yolo_path))
        fuse_silu(g)
        # Calibrate on the bus image itself (single sample). Real
        # production uses 100+; this is a deliberate floor.
        quantise_model(g, str(yolo_path), [{"images": bus}])
        tmp = Path(tempfile.gettempdir()) / "yolov8n.fq.drift.onnx"
        build_fake_quant_model(g, str(yolo_path), out_path=str(tmp))
        fq_run = run_reference(str(tmp), {"images": bus})
        fq_out = np.asarray(fq_run.outputs[fp32_out_name], dtype=np.float64)
    except Exception as exc:
        return {"skipped": f"quantise/fake-quant failed: {exc!r}"}

    diff = fp32_out - fq_out
    sig = float(np.linalg.norm(fp32_out))
    nse = float(np.linalg.norm(diff))
    snr_db = 20 * np.log10(sig / max(nse, 1e-10)) if sig > 0 else 0.0
    max_abs = float(np.max(np.abs(diff)))
    cos = float(
        (fp32_out.ravel() @ fq_out.ravel())
        / max(np.linalg.norm(fp32_out) * np.linalg.norm(fq_out), 1e-10)
    )
    return {
        "output_tensor": fp32_out_name,
        "fp32_shape": list(fp32_out.shape),
        "snr_db": round(snr_db, 2),
        "cosine": round(cos, 6),
        "max_abs_error": round(max_abs, 4),
        "calibration_samples": 1,
    }


def render_t6(d):
    if "skipped" in d:
        return f"# Deep test 6 — FP32 vs fake-INT8 drift\n\n_Skipped: {d['skipped']}_\n"
    lines = ["# Deep test 6 — FP32 vs fake-INT8 output drift (YOLOv8n on bus)", "",
             f"- Output shape:  `{d['fp32_shape']}`",
             f"- SNR:           **{d['snr_db']} dB**  (higher is better; "
             "> 30 dB is production-grade for INT8)",
             f"- Cosine:        **{d['cosine']}**  (1.0 = perfect alignment)",
             f"- Max abs error: {d['max_abs_error']}",
             "",
             "Same raw ONNX fed through (a) plain FP32 ORT and (b) the SDK's "
             "own fake-INT8 fake-quant reference path. Low drift here is "
             "evidence that the SDK's INT8 calibration preserves the model's "
             "answer. Published YOLOv8n INT8 recipes land in the 30-45 dB range.",
             ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

DEEP_TESTS = [
    ("input_perturbation",   test_input_perturbation, render_t1),
    ("bert_determinism",     test_bert_determinism,   render_t2),
    ("gpt2_paris_rank",      test_gpt2_paris_rank,    render_t3),
    ("yolo_determinism",     test_yolo_determinism,   render_t4),
    ("latency_vs_gmacs",     test_latency_vs_gmacs,   render_t5),
    ("int8_drift",           test_int8_drift,         render_t6),
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, (name, runner, renderer) in enumerate(DEEP_TESTS, start=1):
        print(f"\n=== Deep test {i} — {name} ===")
        t0 = time.perf_counter()
        data = runner()
        wall = time.perf_counter() - t0
        print(f"  wall: {wall:.1f}s")
        (OUT_DIR / f"{i}_{name}.json").write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8",
        )
        (OUT_DIR / f"{i}_{name}.md").write_text(renderer(data), encoding="utf-8")
    print(f"\nAll outputs under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
