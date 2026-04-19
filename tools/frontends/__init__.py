"""Front-end adapters that bring external model formats into the
project's NnGraph IR via the existing ONNX loader (F1-C1).

Each adapter converts its source format to a (possibly temporary)
.onnx file and then delegates to `tools.npu_ref.onnx_loader.load_onnx`.
This keeps all op-handler logic in one place.

Adapters land as part of F1-B4 (PyTorch) and F1-B5 (TVM/MLIR/XLA/NNEF).
"""
