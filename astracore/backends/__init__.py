"""Built-in backends shipped with the SDK.

Importing this sub-package registers the built-ins (``npu-sim``,
``onnxruntime``) with the global backend registry. Keep imports at the
bottom so the registry module is fully initialised first.
"""

from astracore.backends.npu_sim import NpuSimBackend
from astracore.backends.ort import OrtBackend

__all__ = ["NpuSimBackend", "OrtBackend"]
