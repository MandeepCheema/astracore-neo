"""XLA / JAX front-end adapter (F1-B5).

JAX/XLA models reach ONNX via jax2tf → tflite → onnx2tf (or more
directly via the jax-to-onnx community bridge). This adapter accepts
the resulting ONNX.

Call pattern:

    JAX/XLA  ──[jax_to_onnx]──▶  .onnx  ──[F1-C1 loader]──▶  NnGraph

The JAX→ONNX conversion itself is outside this adapter's scope; the
ecosystem tooling (jaxlib, jax_to_onnx, or StableHLO via the MLIR
adapter) handles that. This file exists so every B5 front-end has
the same entry-point shape.
"""

from __future__ import annotations

import os
import tempfile

from tools.npu_ref.nn_graph import NnGraph
from tools.npu_ref.onnx_loader import load_onnx


def load_xla_from_onnx_bytes(onnx_bytes: bytes,
                              *,
                              batch_size: int = 1) -> NnGraph:
    """Load from JAX/XLA → ONNX bytes."""
    fd, path = tempfile.mkstemp(suffix=".onnx", prefix="xla_fe_")
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


def load_xla_from_onnx_path(onnx_path: str,
                             *,
                             batch_size: int = 1) -> NnGraph:
    """Pass-through for a JAX/XLA originated .onnx that's on disk."""
    return load_onnx(onnx_path, batch_size=batch_size)
