"""Magnitude-based structured N:M pruning of yolov8n.onnx — feasibility
probe before committing RTL effort to 8:1 sparsity.

Applies N:M pruning along each conv's flattened reduction dimension
(K_total = C_in × k_h × k_w), row by row per output channel. This
matches how our NPU's sparsity engine (F1-A3) sees the weight stream:
groups of M consecutive K-elements with N nonzeros each, per output
channel.

No fine-tuning. This is the *pessimistic* baseline: any accuracy it
produces is a lower bound — QAT / fine-tuning can recover 2-5 pp.
If magnitude-only 1:8 costs ≤10 pp, the QAT track is worth pursuing.
If it costs ≥20 pp, the model isn't structurally compatible and we
fall back to 2:4 / 4:1.

Output: one pruned .onnx per (N, M) pattern, alongside the original.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

REPO = Path(__file__).resolve().parent.parent
SRC_ONNX = REPO / "data" / "models" / "yolov8n.onnx"
OUT_DIR = REPO / "data" / "models" / "pruned"

# (N nonzeros, M block size). So (2, 4) is NVIDIA's 2:4 (50 % density).
# (1, 8) is the spec sheet's "8:1 sparsity" (12.5 % density, 8× multiplier).
PATTERNS: List[Tuple[int, int]] = [(2, 4), (2, 8), (1, 8)]


def _nm_prune_row(row: np.ndarray, n: int, m: int) -> np.ndarray:
    """Keep top-n values by magnitude in each length-m block along a
    1-D row; zero the other m-n. Tail block (if K % m ≠ 0) is treated
    leniently: keep top min(n, len) of whatever fits."""
    out = np.zeros_like(row)
    for start in range(0, row.size, m):
        block = row[start:start + m]
        keep = min(n, block.size)
        if keep == block.size:
            out[start:start + block.size] = block
            continue
        idx = np.argpartition(np.abs(block), -keep)[-keep:]
        out[start + idx] = block[idx]
    return out


def _nm_prune_weight(w: np.ndarray, n: int, m: int) -> np.ndarray:
    """w: (C_out, C_in, kH, kW). Reshape to (C_out, K_total) where
    K_total = C_in * kH * kW, prune per-row, reshape back.
    Matches our compiler's im2col K ordering (C_in × dH × dW flattened
    row-major)."""
    assert w.ndim == 4
    C_out, C_in, kH, kW = w.shape
    flat = w.reshape(C_out, C_in * kH * kW)
    pruned = np.empty_like(flat)
    for c in range(C_out):
        pruned[c] = _nm_prune_row(flat[c], n, m)
    return pruned.reshape(C_out, C_in, kH, kW)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sparsity_ratio(w: np.ndarray) -> float:
    return 1.0 - float((w != 0).sum()) / w.size


def prune_model(src_path: Path, n: int, m: int,
                 out_path: Path) -> dict:
    """Apply N:M pruning to every Conv weight initialiser in the
    model. Returns a dict of per-layer sparsity stats for reporting."""
    model = onnx.load(str(src_path))
    # Map initializer name -> node that consumes it as weight (= input[1])
    # so we prune ONLY conv/gemm/matmul weights, not bias or constants.
    weight_names = set()
    for node in model.graph.node:
        if node.op_type in ("Conv", "Gemm", "MatMul"):
            if len(node.input) >= 2:
                weight_names.add(node.input[1])

    stats = {"layers": [], "total_weights": 0, "total_pruned": 0}
    new_inits = []
    for t in model.graph.initializer:
        if t.name not in weight_names:
            new_inits.append(t)
            continue
        arr = numpy_helper.to_array(t).astype(np.float32)
        if arr.ndim != 4:
            # Gemm/MatMul 2-D weights: pattern is along axis 1 (input dim).
            # Reshape to (out, 1, 1, in) so our 4-D pruner applies.
            if arr.ndim == 2:
                C_out, C_in = arr.shape
                w4 = arr.reshape(C_out, C_in, 1, 1)
                pruned4 = _nm_prune_weight(w4, n, m)
                pruned = pruned4.reshape(C_out, C_in)
            else:
                new_inits.append(t)
                continue
        else:
            pruned = _nm_prune_weight(arr, n, m)

        ratio = _sparsity_ratio(pruned)
        stats["layers"].append({
            "name": t.name, "shape": list(arr.shape),
            "sparsity": ratio,
        })
        stats["total_weights"] += arr.size
        stats["total_pruned"] += int((pruned == 0).sum())
        new_inits.append(numpy_helper.from_array(pruned, t.name))

    # Replace initialisers.
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_inits)

    onnx.save(model, str(out_path))
    stats["overall_sparsity"] = (
        stats["total_pruned"] / stats["total_weights"]
        if stats["total_weights"] else 0.0
    )
    stats["n"] = n
    stats["m"] = m
    stats["expected_density"] = n / m
    stats["sha256"] = _sha256(out_path)
    stats["bytes"] = out_path.stat().st_size
    return stats


def main() -> int:
    if not SRC_ONNX.exists():
        print(f"ERROR: {SRC_ONNX} missing. Run scripts/export_yolov8n_onnx.py.",
              file=sys.stderr)
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_stats = {"source_onnx": str(SRC_ONNX.relative_to(REPO)),
                 "source_sha256": _sha256(SRC_ONNX),
                 "variants": []}
    for n, m in PATTERNS:
        out_path = OUT_DIR / f"yolov8n_pruned_{n}of{m}.onnx"
        print(f"pruning {n}:{m} -> {out_path.relative_to(REPO)}")
        stats = prune_model(SRC_ONNX, n, m, out_path)
        print(f"  overall sparsity: {stats['overall_sparsity']:.3f} "
              f"(expected {1 - stats['expected_density']:.3f})")
        print(f"  output: {stats['bytes']:,} bytes "
              f"sha256={stats['sha256'][:12]}")
        all_stats["variants"].append({
            "n": n, "m": m,
            "path": str(out_path.relative_to(REPO).as_posix()),
            "overall_sparsity": stats["overall_sparsity"],
            "sha256": stats["sha256"],
            "bytes": stats["bytes"],
            "per_layer": stats["layers"],
        })

    manifest = OUT_DIR / "pruning_manifest.json"
    manifest.write_text(json.dumps(all_stats, indent=2) + "\n")
    print(f"wrote manifest: {manifest.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
