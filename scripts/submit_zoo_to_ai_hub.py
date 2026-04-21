"""Batch-submit the zoo's vision CNN models to Qualcomm AI Hub.

For each model: prep for AI Hub (opset + IR + input conventions) then
submit compile+profile. Writes per-model+device JSON under
``reports/qualcomm_aihub/`` and is idempotent (skips any combination
that already has a result file).

Usage::

    python scripts/submit_zoo_to_ai_hub.py \\
        --devices "QCS8550 (Proxy)" "Samsung Galaxy S24"
    python scripts/submit_zoo_to_ai_hub.py --only squeezenet-1.1
    python scripts/submit_zoo_to_ai_hub.py --refresh   # ignore existing
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "reports" / "qualcomm_aihub"
PREP = REPO / "scripts" / "prep_onnx_for_ai_hub.py"
SUBMIT = REPO / "scripts" / "submit_to_qualcomm_ai_hub.py"

# Full zoo — vision CNNs + transformers. AI Hub latency measurement
# is shape/dtype-driven; random token IDs give the same timing as
# real ones, so tokenizer setup is optional for perf numbers.
CNN_ZOO = [
    ("yolov8n",              REPO / "data" / "models" / "yolov8n.onnx"),
    ("squeezenet-1.1",       REPO / "data" / "models" / "zoo" / "squeezenet-1.1.onnx"),
    ("mobilenetv2-7",        REPO / "data" / "models" / "zoo" / "mobilenetv2-7.onnx"),
    ("resnet50-v2-7",        REPO / "data" / "models" / "zoo" / "resnet50-v2-7.onnx"),
    ("shufflenet-v2-10",     REPO / "data" / "models" / "zoo" / "shufflenet-v2-10.onnx"),
    ("efficientnet-lite4-11",REPO / "data" / "models" / "zoo" / "efficientnet-lite4-11.onnx"),
    ("bert-squad-10",        REPO / "data" / "models" / "zoo" / "bert-squad-10.onnx"),
    ("gpt-2-10",             REPO / "data" / "models" / "zoo" / "gpt-2-10.onnx"),
]

# INT8 fake-quant variants — produced by scripts/quantise_zoo.py.
# Emit QDQ-format ONNX that AI Hub can compile to QNN INT8 kernels.
INT8_ZOO = [
    ("yolov8n",              REPO / "data" / "models" / "zoo" / "int8" / "yolov8n.int8.onnx"),
    ("squeezenet-1.1",       REPO / "data" / "models" / "zoo" / "int8" / "squeezenet-1.1.int8.onnx"),
    ("mobilenetv2-7",        REPO / "data" / "models" / "zoo" / "int8" / "mobilenetv2-7.int8.onnx"),
    ("resnet50-v2-7",        REPO / "data" / "models" / "zoo" / "int8" / "resnet50-v2-7.int8.onnx"),
    ("shufflenet-v2-10",     REPO / "data" / "models" / "zoo" / "int8" / "shufflenet-v2-10.int8.onnx"),
]


def _result_path(model_name: str, device: str, *, int8: bool = False) -> Path:
    suffix = ".int8.aihub" if int8 else ".aihub"
    slug = f"{model_name}{suffix}_{device.replace(' ', '_')}.json"
    return OUT_DIR / slug


def _prep(model_name: str, src: Path) -> Path:
    """Produce a cleaned ``.aihub.onnx`` next to the source."""
    dst = src.with_name(f"{src.stem}.aihub.onnx")
    r = subprocess.run(
        [sys.executable, str(PREP),
         "--in", str(src), "--out", str(dst), "-q"],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"prep failed: {r.stderr}")
    return dst


def _submit(prepped: Path, device: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    # Force UTF-8 on the subprocess pipes too — qai-hub prints unicode
    # progress glyphs that cp1252 can't decode on Windows default.
    return subprocess.run(
        [sys.executable, "-X", "utf8", str(SUBMIT),
         "--model", str(prepped),
         "--device", device,
         "--out-dir", str(OUT_DIR)],
        capture_output=True, text=True, timeout=1200, env=env,
        encoding="utf-8", errors="replace",
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--devices", nargs="+",
                   default=["QCS8550 (Proxy)"],
                   help="AI Hub device names (quoted)")
    p.add_argument("--only", nargs="+", default=None,
                   help="subset of zoo model names")
    p.add_argument("--refresh", action="store_true",
                   help="re-submit even if a result file already exists")
    p.add_argument("--int8", action="store_true",
                   help="submit the INT8 fake-quant variants (results "
                        "land under <model>.int8.aihub_<device>.json)")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    zoo = INT8_ZOO if args.int8 else CNN_ZOO
    targets = [(n, p) for n, p in zoo
               if (args.only is None or n in args.only)
               and p.exists()]
    if not targets:
        print("no zoo models matched; check --only and disk state",
              file=sys.stderr)
        return 2

    rows = []
    t_start = time.perf_counter()
    for name, src in targets:
        # INT8 fake-quant files need the same IR-v7 + init-removal
        # prep as FP32 ones (verified not to touch QDQ nodes).
        # `_prep` drops ``.aihub.onnx`` next to the source, but for
        # INT8 we want ``.int8.aihub.onnx`` under the same dir.
        try:
            print(f"\n=== prep {name}{' (INT8)' if args.int8 else ''} ===")
            prepped = _prep(name, src)
        except Exception as exc:
            print(f"  PREP FAIL: {exc}")
            rows.append({"model": name, "status": "prep_fail",
                         "error": str(exc)})
            continue

        for device in args.devices:
            rp = _result_path(name, device, int8=args.int8)
            if rp.exists() and not args.refresh:
                print(f"  [skip cached] {name} / {device}  -> {rp.name}")
                continue
            print(f"  submit: {name} / {device}")
            r = _submit(prepped, device)
            stdout = r.stdout or ""
            # Strip non-ASCII so parent shells with cp1252 stdout (Windows
            # PowerShell, default Python print) don't crash on qai-hub's
            # unicode progress glyphs.
            tail = "\n".join(stdout.splitlines()[-6:]).encode(
                "ascii", "replace").decode("ascii")
            if r.returncode == 0 and "ms/inf on" in stdout:
                # Relocate the inner script's default-named output to
                # our idempotent path.
                inner = OUT_DIR / (
                    f"{prepped.stem}_{device.replace(' ', '_')}.json")
                if inner.exists():
                    inner.replace(rp)
                print(f"    OK -> {rp.name}")
                rows.append({"model": name, "device": device,
                             "status": "ok",
                             "result_file": str(rp)})
            else:
                print(f"    FAIL\n{tail}")
                rows.append({"model": name, "device": device,
                             "status": "fail",
                             "tail": tail})

    wall = time.perf_counter() - t_start
    summary = OUT_DIR / "batch_summary.json"
    summary.write_text(json.dumps({
        "wall_s": round(wall, 1),
        "devices": args.devices,
        "rows": rows,
    }, indent=2), encoding="utf-8")

    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_fail = sum(1 for r in rows if r.get("status") == "fail")
    print("\n=============================================")
    print(f"Wall: {wall:.1f}s   OK: {n_ok}   FAIL: {n_fail}")
    print(f"Summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
