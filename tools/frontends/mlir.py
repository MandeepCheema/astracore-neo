"""MLIR / StableHLO front-end adapter (F1-B5).

Covers the MLIR ecosystem path:

    torch-mlir / stablehlo / IREE  ──▶  ONNX  ──[F1-C1 loader]──▶  NnGraph

The conversion from StableHLO / torch-mlir to ONNX is provided by the
respective projects (torch_mlir.compile(..., output_type="onnx"), or
stablehlo_to_onnx from the stablehlo tooling). This adapter takes the
resulting ONNX and feeds it through the project's loader.

If the user has a `.mlir` source file and prefers to run the
stablehlo-to-onnx converter themselves, they can use
`load_mlir_from_onnx_path` after producing the .onnx manually.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Optional

from tools.npu_ref.nn_graph import NnGraph
from tools.npu_ref.onnx_loader import load_onnx


def load_stablehlo_from_onnx_bytes(onnx_bytes: bytes,
                                    *,
                                    batch_size: int = 1) -> NnGraph:
    """Load from StableHLO → ONNX bytes (produced by the stablehlo
    tooling or torch_mlir.compile with output_type='onnx')."""
    fd, path = tempfile.mkstemp(suffix=".onnx", prefix="mlir_fe_")
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


def load_mlir_from_onnx_path(onnx_path: str,
                              *,
                              batch_size: int = 1) -> NnGraph:
    """Pass-through for a StableHLO / torch-mlir originated .onnx
    that's already on disk."""
    return load_onnx(onnx_path, batch_size=batch_size)
