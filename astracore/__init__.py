"""AstraCore Neo — Automotive AI Inference SDK.

Public API surface. Anything not re-exported here is internal and may
change without notice between releases.

Core concepts
-------------
``Graph``
    Internal representation of a neural network (the ``NnGraph`` from
    ``tools.npu_ref``). Produced by loaders (ONNX, PyTorch, NNEF).

``Quantiser``
    Converts a float ``Graph`` into an INT8 / INT4 / INT2 graph plus a
    scale/zero-point table. Default is the production PTQ recipe
    (per-channel weights + percentile-99.9999 activations). Override via
    the ``astracore.quantisers`` entry-point.

``Backend``
    Emits executable code or instructions from a quantised ``Graph``.
    The built-in backend is the internal NPU simulator. External
    backends (TensorRT, SNPE, OpenVINO, custom silicon) plug in via the
    ``astracore.backends`` entry-point.

``Runtime``
    Executes a compiled program on a target backend and returns output
    tensors.

Plugin API
----------
Three decorator-based registries let downstream packages extend the SDK
without forking:

    @astracore.register_op("MyCustomOp")
    def handle_my_custom_op(node, graph): ...

    @astracore.register_quantiser("percentile_98")
    class Percentile98(Quantiser): ...

    @astracore.register_backend("tensorrt")
    class TensorRTBackend(Backend): ...

Packaged plugins declare the same names via setuptools entry-points in
their own ``pyproject.toml``; ``astracore`` discovers them at import time.
"""

from __future__ import annotations

from astracore._version import __version__
from astracore.registry import (
    register_op,
    register_quantiser,
    register_backend,
    get_op,
    get_quantiser,
    get_backend,
    list_ops,
    list_quantisers,
    list_backends,
)
from astracore.backend import Backend, BackendReport
from astracore.quantiser import Quantiser

# Import built-in backends so they appear in the registry without the
# user having to import ``astracore.backends`` explicitly. External
# backends still register via ``astracore.backends`` entry-points.
from astracore import backends as _builtin_backends  # noqa: F401

__all__ = [
    "__version__",
    "register_op",
    "register_quantiser",
    "register_backend",
    "get_op",
    "get_quantiser",
    "get_backend",
    "list_ops",
    "list_quantisers",
    "list_backends",
    "Backend",
    "BackendReport",
    "Quantiser",
]
