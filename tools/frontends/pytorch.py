"""PyTorch front-end adapter (F1-B4).

Converts a `torch.nn.Module` to an NnGraph via ONNX export + F1-C1 loader.
Torch is not a hard dependency of this module: the import is deferred
into `load_pytorch` so the rest of the compiler stack runs without it.

Usage
-----

    import torch
    from torchvision.models import resnet50
    from tools.frontends.pytorch import load_pytorch

    model = resnet50(weights=None).eval()
    example = torch.randn(1, 3, 224, 224)
    graph = load_pytorch(model, example, input_name="images")

The returned NnGraph is the same IR `onnx_loader.load_onnx` produces.
Downstream passes (F1-C2 quantiser, F1-C3/C4 tiler) consume it
unchanged.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from tools.npu_ref.nn_graph import NnGraph
from tools.npu_ref.onnx_loader import load_onnx


def load_pytorch(module: Any,
                 example_input: Any,
                 *,
                 input_name: str = "x",
                 output_name: str = "y",
                 opset: int = 17,
                 dynamic_axes: Optional[Mapping[str, Mapping[int, str]]] = None,
                 do_constant_folding: bool = True,
                 training_mode: bool = False,
                 onnx_path: Optional[str] = None,
                 batch_size: int = 1) -> NnGraph:
    """Export `module` to ONNX via `torch.onnx.export` and load it.

    Args:
        module: a `torch.nn.Module`. Will be called with `example_input`
            during export.
        example_input: tensor or tuple of tensors matching the module's
            forward signature. Used by torch.onnx.export to trace shapes.
        input_name: name to give the graph's first input. Defaults to
            "x" for generality; use "images" for vision models that
            match YOLOv8's convention.
        output_name: name for the graph output.
        opset: ONNX opset version. 17 is the F1-C1 baseline; bump to
            20 if you need Gelu-as-single-op from PyTorch 2.3+.
        dynamic_axes: dict in torch.onnx.export format for dynamic
            batch/sequence dims. If None, shapes are fully concretised.
        do_constant_folding: run torch's constant folding pass pre-export.
        training_mode: False uses EVAL export (drops dropout etc.); True
            preserves training-mode ops (needed for the F1-B6 on-chip
            training path, which is not in the current plan).
        onnx_path: if given, write the intermediate .onnx there so the
            caller can inspect it. If None, a temp file is used and
            deleted on success.
        batch_size: concretised batch dimension (see onnx_loader).

    Returns:
        NnGraph from `load_onnx` applied to the exported model.

    Raises:
        ImportError: if torch isn't installed (install in .venv-export).
    """
    try:
        import torch  # noqa: F401
        import torch.onnx
    except ImportError as e:
        raise ImportError(
            "PyTorch front-end requires torch. Install via "
            ".venv-export or pip install torch."
        ) from e

    # torch.onnx.export happily overwrites existing files; route to
    # a fresh tempfile unless the caller asked for a specific path.
    owned_tempfile = onnx_path is None
    if owned_tempfile:
        fd, onnx_path = tempfile.mkstemp(suffix=".onnx", prefix="pytorch_fe_")
        os.close(fd)

    try:
        torch.onnx.export(
            module,
            example_input,
            onnx_path,
            input_names=[input_name],
            output_names=[output_name],
            opset_version=opset,
            do_constant_folding=do_constant_folding,
            training=(torch.onnx.TrainingMode.TRAINING if training_mode
                      else torch.onnx.TrainingMode.EVAL),
            dynamic_axes=dict(dynamic_axes) if dynamic_axes else None,
        )
        return load_onnx(onnx_path, batch_size=batch_size)
    finally:
        if owned_tempfile and onnx_path and os.path.exists(onnx_path):
            try:
                os.unlink(onnx_path)
            except OSError:
                # Best-effort cleanup; tempfile will be GC'd by OS.
                pass


def export_pytorch_to_onnx(module: Any,
                           example_input: Any,
                           out_path: str,
                           *,
                           input_name: str = "x",
                           output_name: str = "y",
                           opset: int = 17) -> str:
    """Lower-level helper that just runs torch.onnx.export, without
    going through the NnGraph loader. Useful for scripts that want
    to keep a persistent .onnx on disk for later manual inspection.
    """
    try:
        import torch.onnx
    except ImportError as e:
        raise ImportError(
            "PyTorch front-end requires torch. Install via "
            ".venv-export or pip install torch."
        ) from e
    torch.onnx.export(
        module, example_input, out_path,
        input_names=[input_name], output_names=[output_name],
        opset_version=opset,
    )
    return out_path
