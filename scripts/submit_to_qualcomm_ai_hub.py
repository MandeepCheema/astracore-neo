"""Qualcomm AI Hub submission scaffold.

Sends an ONNX model to https://aihub.qualcomm.com and records the
measured latency + power numbers in ``reports/qualcomm_aihub/``.

Status
------
**Scaffold only — no real upload today.** Qualcomm AI Hub requires an
authenticated account and an API token (`qai-hub configure --api_token
<TOKEN>`). The script is ready to run the moment credentials are set;
without them it prints the exact prep steps and exits cleanly.

Why this lives in the repo now
------------------------------
Qualcomm AI Hub is one of the few paths to **measured silicon numbers
on real Snapdragon DSP + NPU without any cloud spend** — it's a REST
service, not a GPU instance. Having the submission script ready means
the only gate is the account signup; the moment that's done, a
15-minute run produces numbers we can publish on the leaderboard
alongside the host-CPU rows.

Usage (once credentials are set)
--------------------------------

    pip install qai-hub
    qai-hub configure --api_token <token>
    python scripts/submit_to_qualcomm_ai_hub.py \\
        --model data/models/yolov8n.onnx \\
        --device 'Snapdragon 8 Gen 3'

Output: ``reports/qualcomm_aihub/<model>_<device>.json`` with
inference latency, peak memory, and (for supported devices) power.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "reports" / "qualcomm_aihub"

# Devices AI Hub currently exposes for public benchmarking. Keep this
# list short and in-sync with https://app.aihub.qualcomm.com/docs/hub/api_examples.html
# (last sync: 2026-04-21).
# Device names must match AI Hub's own catalogue exactly — get the
# live list with `qai-hub list-devices`. Common picks for our story:
#   - QCS8550 (Proxy)          automotive / IoT class Snapdragon
#   - Samsung Galaxy S24       Snapdragon 8 Gen 3 (current flagship)
#   - Samsung Galaxy S23       Snapdragon 8 Gen 2 (prior gen)
#   - Snapdragon X Elite CRD   laptop / PC silicon
KNOWN_DEVICES = [
    "QCS8550 (Proxy)",
    "Samsung Galaxy S24",
    "Samsung Galaxy S23",
    "Snapdragon X Elite CRD",
]


def _require_qai_hub() -> "object | None":
    try:
        import qai_hub  # noqa: F401
        return __import__("qai_hub")
    except ImportError:
        print(
            "ERROR: qai-hub not installed.\n"
            "  pip install qai-hub\n"
            "  qai-hub configure --api_token <your-token>\n"
            "  (get a token by signing up at https://aihub.qualcomm.com)",
            file=sys.stderr,
        )
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True,
                   help="path to the ONNX model to upload")
    p.add_argument("--device", default="Snapdragon 8 Gen 3",
                   help=f"target device (known: {', '.join(KNOWN_DEVICES)})")
    p.add_argument("--inputs", default=None,
                   help="optional path to a .npy/.npz with a single sample "
                        "input — if omitted AI Hub auto-generates random")
    p.add_argument("--dry-run", action="store_true",
                   help="validate arguments + print what would be submitted "
                        "without actually uploading")
    p.add_argument("--compile-options", default="",
                   help="raw string passed to AI Hub's compile_options "
                        "(e.g. '--truncate_64bit_io' for int64-token "
                        "transformer models like BERT / GPT-2)")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 2
    if args.device not in KNOWN_DEVICES:
        print(f"WARNING: device '{args.device}' not in our known-good list — "
              f"AI Hub will accept unknown names but this is fragile",
              file=sys.stderr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = (out_dir /
                   f"{model_path.stem}_{args.device.replace(' ', '_')}.json")

    if args.dry_run:
        print(f"DRY RUN — would upload:")
        print(f"  model:   {model_path} ({model_path.stat().st_size/1e6:.1f} MB)")
        print(f"  device:  {args.device}")
        print(f"  inputs:  {args.inputs or '(auto-generated)'}")
        print(f"  result:  {result_path}")
        return 0

    qai_hub = _require_qai_hub()
    if qai_hub is None:
        return 2

    print(f"Submitting {model_path} to AI Hub for device '{args.device}'...")
    t0 = time.perf_counter()

    # AI Hub needs an explicit ``input_specs`` so it can tell the real
    # graph inputs apart from initializers (which some ONNX exporters
    # also list in ``graph.input`` per the pre-IR-4 convention). We
    # read the non-initialiser inputs from the ONNX file and pass them
    # as {name: (shape_tuple, dtype_string)}.
    import onnx
    _ONNX_TO_NP_DTYPE = {
        1: "float32", 2: "uint8", 3: "int8", 5: "int16",
        6: "int32", 7: "int64", 9: "bool", 10: "float16",
        11: "float64",
    }
    mproto = onnx.load(str(model_path))
    init_names = {t.name for t in mproto.graph.initializer}
    input_specs = {}
    for inp in mproto.graph.input:
        if inp.name in init_names:
            continue
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            v = int(d.dim_value) if d.dim_value > 0 else 1
            dims.append(v)
        dtype = _ONNX_TO_NP_DTYPE.get(
            inp.type.tensor_type.elem_type, "float32")
        input_specs[inp.name] = (tuple(dims), dtype)
    print(f"  input_specs: {input_specs}")

    # Auto-detect int64 inputs (transformer token IDs) and inject
    # --truncate_64bit_io so the QNN compiler falls back to int32.
    compile_options = args.compile_options
    has_int64_input = any(dtype == "int64" for _, dtype in input_specs.values())
    if has_int64_input and "--truncate_64bit_io" not in compile_options:
        compile_options = (compile_options + " --truncate_64bit_io").strip()
        print(f"  auto-added: --truncate_64bit_io (int64 inputs detected)")
    if compile_options:
        print(f"  compile_options: {compile_options}")

    # Use compile+profile — most raw ONNX models need a compile pass
    # to target Snapdragon's QNN / tflite format before profiling.
    # See https://app.aihub.qualcomm.com/docs/hub/api_examples.html
    compile_job, profile_job = qai_hub.submit_compile_and_profile_jobs(
        model=str(model_path),
        device=qai_hub.Device(args.device),
        name=f"astracore-{model_path.stem}-{args.device}",
        input_specs=input_specs,
        compile_options=compile_options,
    )
    print(f"  compile job: {compile_job.job_id}")
    if profile_job is None:
        print("  WARNING: compile produced no profile job; check job log on AI Hub")
        return 1
    print(f"  profile job: {profile_job.job_id}")
    print(f"  URL: https://app.aihub.qualcomm.com/jobs/{profile_job.job_id}")
    print("  waiting for results (typical 3-10 minutes)...")

    # Block until complete.
    profile_result = profile_job.download_profile()

    wall = time.perf_counter() - t0

    # profile_result is a dict-shaped object; the relevant keys live under
    # profile_result["execution_summary"]. We pull the headline numbers
    # and record everything raw alongside.
    exec_sum = {}
    try:
        exec_sum = profile_result.get("execution_summary", {})
    except AttributeError:
        exec_sum = getattr(profile_result, "execution_summary", {}) or {}

    summary = {
        "model":        str(model_path),
        "device":       args.device,
        "compile_job":  compile_job.job_id,
        "profile_job":  profile_job.job_id,
        "job_url":      f"https://app.aihub.qualcomm.com/jobs/{profile_job.job_id}",
        "wall_s":       round(wall, 1),
        "generated_at": int(time.time()),
    }
    # Headline metrics — keys vary slightly across targets; pull what's there.
    exec_us = exec_sum.get("estimated_inference_time") or \
              exec_sum.get("execution_time_us")
    if exec_us:
        summary["inference_ms"] = round(float(exec_us) / 1000, 3)
    else:
        # Profile job failed or returned no execution summary (most often
        # QNN-delegate op-unsupported on the target device).
        try:
            status = profile_job.get_status()
            summary["status"] = "profile_failed"
            summary["failure_message"] = getattr(status, "message", "")[:400]
        except Exception:
            summary["status"] = "no_profile_data"
    peak = exec_sum.get("estimated_inference_peak_memory") or \
           exec_sum.get("peak_memory_bytes")
    if peak:
        summary["peak_memory_mb"] = round(float(peak) / 1024 / 1024, 2)
    # Optional fields.
    for k in ("compile_memory_bytes", "compile_runtime_us",
              "first_load_time_us"):
        if k in exec_sum:
            summary[k] = exec_sum[k]

    # Raw dump alongside the summary so we never lose metadata AI Hub
    # adds in future versions.
    raw_path = out_dir / f"{model_path.stem}_{args.device.replace(' ', '_')}_raw.json"
    try:
        raw_path.write_text(json.dumps(profile_result, indent=2, default=str),
                            encoding="utf-8")
        summary["raw_profile"] = str(raw_path)
    except Exception:
        pass

    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResult: {result_path}")
    print(f"  {summary.get('inference_ms', '?')} ms/inf on {args.device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
