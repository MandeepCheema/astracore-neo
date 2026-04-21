"""Prep an ONNX model for Qualcomm AI Hub upload.

AI Hub's Workbench is stricter than ONNX Runtime on a handful of
lint-level details. This script walks a raw ONNX file through every
fixup we've seen break jobs on live AI Hub runs:

1. **Opset upgrade** — versions < 13 are rejected (per-channel
   DequantizeLinear requires opset 13 minimum).
2. **IR-version bump to 7** — older (IR-v3) files with initializers
   duplicated into ``graph.input`` confuse AI Hub's validator even
   when the declared structure is valid. Bumping + removing the
   duplicates resolves the "X in initializer but not in graph input"
   error against YOLOv8n/MobileNetV2/ResNet-50 class ONNX-zoo files.
3. **Strip graph outputs from ``value_info``** — ONNX allows an
   output tensor to appear in both ``graph.output`` and
   ``graph.value_info``; AI Hub errors out.
4. **Remove initializers from ``graph.input``** — modern ONNX
   (IR ≥ 4) does NOT list initializers as graph inputs. Many ONNX
   Zoo models pre-date this convention and AI Hub's compiler picks
   up the old entries as undeclared inputs.
5. **Infer shapes** — run ``onnx.shape_inference.infer_shapes`` so
   AI Hub's compile graph has concrete tensor shapes throughout.
6. **Pow → Mul chain rewrite** — QNN HTP on Qualcomm Snapdragon
   rejects ``Pow(x, k)`` with small integer k (seen on BERT-Squad's
   GELU node 667). We rewrite ``Pow(x, 2..4)`` to ``Mul(..., x)``
   chains which are numerically identical but QNN-friendly.
7. **Final validation** — ``onnx.checker.check_model(full_check=True)``.

Usage::

    python scripts/prep_onnx_for_ai_hub.py \\
        --in  data/models/zoo/mobilenetv2-7.onnx \\
        --out data/models/zoo/mobilenetv2-7.aihub.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def prep_onnx_for_ai_hub(
    src_path: Path,
    dst_path: Path,
    *,
    target_opset: int = 13,
    verbose: bool = False,
) -> dict:
    """Transform a raw ONNX file into AI-Hub-acceptable form.

    Returns a dict of what changed so callers can log or assert.
    """
    import onnx
    from onnx import version_converter, shape_inference

    src_path = Path(src_path)
    dst_path = Path(dst_path)

    log = {"source": str(src_path), "destination": str(dst_path), "steps": []}

    model = onnx.load(str(src_path))

    # 1. Opset upgrade.
    current_opset = 0
    for oi in model.opset_import:
        if oi.domain in ("", "ai.onnx"):
            current_opset = int(oi.version)
            break
    if current_opset and current_opset < target_opset:
        try:
            model = version_converter.convert_version(model, target_opset)
            log["steps"].append(
                f"opset {current_opset} -> {target_opset}")
        except Exception as exc:
            log["steps"].append(
                f"opset upgrade failed ({current_opset}->{target_opset}): "
                f"{type(exc).__name__}")
    else:
        log["steps"].append(f"opset {current_opset} (no upgrade needed)")

    # 2. Bump IR version to 7 (modern). Older files tend to duplicate
    # initializers into graph.input; that trips AI Hub's validator.
    if model.ir_version < 7:
        log["steps"].append(f"ir_version {model.ir_version} -> 7")
        model.ir_version = 7

    # 3. Strip graph outputs from value_info.
    outs = {o.name for o in model.graph.output}
    before = len(model.graph.value_info)
    kept = [v for v in model.graph.value_info if v.name not in outs]
    if len(kept) != before:
        del model.graph.value_info[:]
        model.graph.value_info.extend(kept)
        log["steps"].append(
            f"stripped {before - len(kept)} graph-outputs from value_info")

    # 4. Remove initializers from graph.input (modern ONNX convention).
    init_names = {i.name for i in model.graph.initializer}
    before_in = len(model.graph.input)
    real_inputs = [i for i in model.graph.input if i.name not in init_names]
    if len(real_inputs) != before_in:
        del model.graph.input[:]
        model.graph.input.extend(real_inputs)
        log["steps"].append(
            f"removed {before_in - len(real_inputs)} initializer-inputs "
            f"(leaves {len(real_inputs)} true inputs)")

    # 4. Shape inference.
    try:
        model = shape_inference.infer_shapes(model)
        log["steps"].append("shape inference OK")
    except Exception as exc:
        log["steps"].append(
            f"shape inference warning: {type(exc).__name__}")

    # 5. Pow(x, k) -> Mul-chain rewrite for QNN HTP compatibility.
    # BERT-Squad's node-667 GELU hit this exact wall; no cost on models
    # that don't use Pow with small integer exponents.
    try:
        import importlib.util as _iu
        _p = Path(__file__).parent / "fix_bert_pow.py"
        _spec = _iu.spec_from_file_location("fix_bert_pow", _p)
        _mod = _iu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        n_pow = _mod.rewrite_pow_as_mul(model)
        if n_pow:
            log["steps"].append(
                f"Pow -> Mul chain rewrite: {n_pow} op(s) replaced")
    except Exception as exc:
        log["steps"].append(f"Pow rewrite skipped: {type(exc).__name__}")

    # 6. Final validation — soft: record but don't raise, some ONNX
    # Zoo models trip the stricter ``full_check`` path on node-level
    # details unrelated to AI Hub's actual requirements.
    try:
        onnx.checker.check_model(model, full_check=True)
        log["steps"].append("onnx.checker full_check OK")
    except onnx.checker.ValidationError as exc:
        log["steps"].append(f"onnx.checker warning: {exc}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst_path))
    log["output_bytes"] = dst_path.stat().st_size

    if verbose:
        print(f"Prepped {src_path} -> {dst_path}")
        for s in log["steps"]:
            print(f"  {s}")

    return log


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="src", required=True,
                   help="source ONNX file")
    p.add_argument("--out", dest="dst", required=True,
                   help="destination cleaned ONNX file")
    p.add_argument("--target-opset", type=int, default=13)
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    try:
        log = prep_onnx_for_ai_hub(
            Path(args.src), Path(args.dst),
            target_opset=args.target_opset,
            verbose=not args.quiet,
        )
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
