"""NNEF (Khronos Neural Network Exchange Format) front-end adapter (F1-B5).

NNEF is converted to ONNX via Khronos's `nnef-tools` package, which
exposes `nnef_tools.convert` with source/target format selection.

If nnef-tools is installed, `load_nnef(path)` below runs the
conversion in-process. Otherwise the caller is expected to run the
conversion manually and hand us the resulting .onnx path via
`load_nnef_from_onnx_path`.

The NNEF spec predates most transformer ops; expect NNEF sources to
cover CNN-class models (ResNet, MobileNet, etc.) and lean on the
existing F1-C1 handler table.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Optional

from tools.npu_ref.nn_graph import NnGraph
from tools.npu_ref.onnx_loader import load_onnx


def load_nnef_from_onnx_bytes(onnx_bytes: bytes,
                               *,
                               batch_size: int = 1) -> NnGraph:
    """Load from NNEF → ONNX bytes produced by nnef-tools."""
    fd, path = tempfile.mkstemp(suffix=".onnx", prefix="nnef_fe_")
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


def load_nnef_from_onnx_path(onnx_path: str,
                              *,
                              batch_size: int = 1) -> NnGraph:
    """Pass-through for an nnef-tools produced .onnx on disk."""
    return load_onnx(onnx_path, batch_size=batch_size)


def load_nnef(nnef_path: str,
              *,
              batch_size: int = 1,
              onnx_cache_path: Optional[str] = None) -> NnGraph:
    """Full in-process conversion: requires the `nnef-tools` package.

    Args:
        nnef_path: path to an .nnef file or an NNEF graph directory.
        batch_size: concretised batch dim.
        onnx_cache_path: if provided, the intermediate .onnx is written
            here and kept. Default: tempfile, deleted after loading.

    Raises:
        ImportError: if nnef-tools is not installed.
    """
    try:
        from nnef_tools.convert import convert  # type: ignore
    except ImportError as e:
        raise ImportError(
            "NNEF in-process conversion requires the nnef-tools "
            "package. Either `pip install nnef-tools` or run the "
            "converter manually and use load_nnef_from_onnx_path()."
        ) from e

    owned_tempfile = onnx_cache_path is None
    if owned_tempfile:
        fd, onnx_cache_path = tempfile.mkstemp(suffix=".onnx", prefix="nnef_fe_")
        os.close(fd)
    try:
        convert(
            input_format="nnef",
            output_format="onnx",
            input_model=nnef_path,
            output_model=onnx_cache_path,
        )
        return load_onnx(onnx_cache_path, batch_size=batch_size)
    finally:
        if owned_tempfile and onnx_cache_path and os.path.exists(onnx_cache_path):
            try:
                os.unlink(onnx_cache_path)
            except OSError:
                pass
