"""TVM front-end adapter (F1-B5).

Converts a TVM Relay IRModule to an NnGraph by going through ONNX.
TVM itself has `relay.frontend.from_onnx` (the forward direction), so
for the forward (TVM → our IR) path the clean bridge is:

    TVM Relay  ──[tvm.relay.save_onnx]──▶  .onnx  ──[F1-C1 loader]──▶  NnGraph

TVM doesn't have a first-class Relay → ONNX export, but
`tvm.relay.frontend.from_onnx` + round-tripping through the model
zoo's ONNX export is the documented workflow. We provide a helper
that takes a raw ONNX string (what TVM produces via its TIR → ONNX
lowering path, or what the user already has on disk) and hands it
to the existing loader.

In practice most TVM users already start from ONNX; this adapter's
value is unifying the call signature with the other B5 adapters.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Optional

from tools.npu_ref.nn_graph import NnGraph
from tools.npu_ref.onnx_loader import load_onnx


def load_tvm_from_onnx_bytes(onnx_bytes: bytes,
                              *,
                              batch_size: int = 1) -> NnGraph:
    """Load from a serialised ONNX byte string (typically what
    `tvm.relay.frontend.from_onnx` was itself given, or what
    `onnx.save_model` produced from a TVM round-trip).

    Args:
        onnx_bytes: the ONNX model serialised to bytes.
        batch_size: concretised batch dim.

    Returns:
        NnGraph.
    """
    fd, path = tempfile.mkstemp(suffix=".onnx", prefix="tvm_fe_")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(onnx_bytes)
        return load_onnx(path, batch_size=batch_size)
    finally:
        if os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass


def load_tvm_onnx_path(onnx_path: str, *, batch_size: int = 1) -> NnGraph:
    """Pass-through helper: if the user already has a TVM-originated
    .onnx on disk, just use the F1-C1 loader directly. Provided here
    so every B5 adapter exposes the same entry-point pattern."""
    return load_onnx(onnx_path, batch_size=batch_size)
