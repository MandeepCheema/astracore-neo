"""Quantise a FP32 ONNX via Qualcomm AI Hub's *native* quantize_job
(instead of ORT's ``quantize_static``). Then compile+profile the
AI-Hub-optimised INT8.

Why
---
Our §6b leaderboard showed that bring-your-own ORT-QDQ INT8 is a mixed
result on AI Hub: SqueezeNet 1.27× faster but MobileNet 3.33× slower
because many QDQ insertions don't fuse into QNN INT8 kernels.
AI Hub's own quantize_job produces compiler-optimal INT8 — should fix.

Flow
----
1. Prep FP32 ONNX for AI Hub (shared pipeline).
2. Upload + submit_quantize_job with random calibration data.
3. Take the quantised TargetModel and submit_compile_and_profile_jobs
   on the target device.
4. Save result to ``reports/qualcomm_aihub/<model>.aihub_native_int8_<device>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "reports" / "qualcomm_aihub"
PREP = REPO / "scripts" / "prep_onnx_for_ai_hub.py"


def _prep(src: Path) -> Path:
    dst = src.with_name(f"{src.stem}.aihub.onnx")
    r = subprocess.run(
        [sys.executable, str(PREP), "--in", str(src),
         "--out", str(dst), "-q"],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"prep failed: {r.stderr}")
    return dst


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="FP32 ONNX source")
    p.add_argument("--device", default="QCS8550 (Proxy)")
    p.add_argument("--cal-samples", type=int, default=8,
                   help="number of calibration samples (AI Hub quantize_job)")
    p.add_argument("--cal-seed", type=int, default=0)
    args = p.parse_args()

    src = Path(args.model)
    if not src.exists():
        print(f"ERROR: model not found: {src}", file=sys.stderr)
        return 2

    import qai_hub as hub
    import onnx

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Prep FP32.
    prepped = _prep(src)
    m = onnx.load(str(prepped))
    init_names = {t.name for t in m.graph.initializer}
    real = [i for i in m.graph.input if i.name not in init_names]
    # Build calibration dict: one sample per real input, random per its
    # declared shape/dtype. For transformers we stay away — pure CNNs
    # tolerate random uniform FP32 inputs fine for QNN calibration.
    _TYPE = {
        1: np.float32, 2: np.uint8, 3: np.int8, 5: np.int16,
        6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16,
        11: np.float64,
    }
    rng = np.random.default_rng(args.cal_seed)
    cal: dict = {inp.name: [] for inp in real}
    for _ in range(args.cal_samples):
        for inp in real:
            dtype = _TYPE.get(inp.type.tensor_type.elem_type, np.float32)
            shape = tuple((d.dim_value if d.dim_value else 1)
                          for d in inp.type.tensor_type.shape.dim) or (1,)
            if np.issubdtype(dtype, np.integer):
                arr = rng.integers(0, 30000, size=shape, dtype=dtype)
            elif dtype == np.bool_:
                arr = rng.integers(0, 2, size=shape).astype(np.bool_)
            else:
                arr = rng.uniform(0, 1, size=shape).astype(dtype, copy=False)
            cal[inp.name].append(arr)

    stem = src.stem
    device = hub.Device(args.device)

    print(f"Stage 1: quantize_job (AI Hub native INT8)")
    t0 = time.perf_counter()
    qjob = hub.submit_quantize_job(
        model=str(prepped),
        calibration_data=cal,
        name=f"astracore-native-int8-{stem}",
    )
    print(f"  quantize job: {qjob.job_id}")
    print(f"  URL: https://app.aihub.qualcomm.com/jobs/{qjob.job_id}")
    target_model = qjob.get_target_model()   # blocks
    if target_model is None:
        st = qjob.get_status()
        print(f"  QUANTIZE FAILED: {getattr(st, 'message', st)}")
        return 1
    print(f"  quantize wall: {time.perf_counter() - t0:.1f}s")

    # WORKAROUND for qai-hub SDK bug: quantize_job output uses external
    # tensor data; downstream compile_job rejects it with "stored
    # externally and should not have data field.float_data".
    # Fix: download → inline external tensors → re-upload as fresh Model.
    print(f"\nStage 1.5: download + inline external data (SDK-bug workaround)")
    import tempfile
    tmp = Path(tempfile.gettempdir()) / f"{stem}_quantized.onnx"
    actual = target_model.download(str(tmp))   # SDK appends ext; returns path
    dl_path = Path(actual) if actual else tmp
    print(f"  downloaded: {dl_path}  exists={dl_path.exists()}  "
          f"({dl_path.stat().st_size/1e6:.1f} MB)"
          if dl_path.exists() else f"  download path: {dl_path} MISSING")
    if not dl_path.exists():
        raise FileNotFoundError(f"target_model.download returned {actual!r}; "
                                f"nothing at {tmp}")
    mp = onnx.load(str(dl_path), load_external_data=True)
    # Strip external-data metadata so saving writes inline.
    for init in mp.graph.initializer:
        if init.HasField("raw_data") or init.float_data or init.int64_data:
            init.data_location = onnx.TensorProto.DEFAULT
            # Remove any external_data entries
            del init.external_data[:]
    inlined = tmp.with_name(f"{tmp.stem}_inline.onnx")
    onnx.save(mp, str(inlined))
    print(f"  inlined: {inlined}  ({inlined.stat().st_size/1e6:.1f} MB)")
    # Re-upload as a fresh Model handle
    uploaded = hub.upload_model(str(inlined), name=f"{stem}_int8_inline")
    print(f"  uploaded as: {uploaded.model_id}")

    print(f"\nStage 2: compile quantized model to QNN for {args.device}")
    cjob = hub.submit_compile_job(
        model=uploaded,
        device=device,
        name=f"astracore-native-int8-{stem}-compile",
    )
    print(f"  compile: {cjob.job_id}")
    compiled = cjob.get_target_model()
    if compiled is None:
        cst = cjob.get_status()
        print(f"  COMPILE FAILED: {getattr(cst, 'message', cst)}")
        return 1

    print(f"\nStage 3: profile on {args.device}")
    pjob = hub.submit_profile_job(
        model=compiled,
        device=device,
        name=f"astracore-native-int8-{stem}-profile",
    )
    print(f"  profile: {pjob.job_id}")
    print(f"  profile URL: https://app.aihub.qualcomm.com/jobs/{pjob.job_id}")

    # Block until profile completes.
    try:
        profile = pjob.download_profile()
    except Exception as exc:
        print(f"  profile download failed: {exc}")
        profile = {}

    wall = time.perf_counter() - t0

    summary = {
        "model": str(src),
        "device": args.device,
        "pipeline": "aihub_native_int8",
        "quantize_job": qjob.job_id,
        "compile_job": cjob.job_id,
        "profile_job": pjob.job_id,
        "job_url": f"https://app.aihub.qualcomm.com/jobs/{pjob.job_id}",
        "wall_s": round(wall, 1),
        "generated_at": int(time.time()),
    }
    exec_sum = profile.get("execution_summary") if isinstance(profile, dict) else {}
    exec_us = (exec_sum or {}).get("estimated_inference_time") or \
              (exec_sum or {}).get("execution_time_us")
    if exec_us:
        summary["inference_ms"] = round(float(exec_us) / 1000, 3)
    peak = (exec_sum or {}).get("estimated_inference_peak_memory") or \
           (exec_sum or {}).get("peak_memory_bytes")
    if peak:
        summary["peak_memory_mb"] = round(float(peak) / 1024 / 1024, 2)
    if "inference_ms" not in summary:
        st = pjob.get_status()
        summary["status"] = "profile_failed"
        summary["failure_message"] = getattr(st, "message", "")[:400]

    dev_slug = args.device.replace(" ", "_")
    dst = OUT_DIR / f"{stem}.aihub_native_int8_{dev_slug}.json"
    dst.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResult: {dst}")
    if "inference_ms" in summary:
        print(f"  {summary['inference_ms']} ms/inf on {args.device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
