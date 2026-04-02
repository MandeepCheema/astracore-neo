"""AstraCore Neo Inference — Exception hierarchy."""


class InferenceError(Exception):
    """Base for all inference subsystem errors."""


class CompilerError(InferenceError):
    """Raised on graph compilation failure."""


class QuantizationError(InferenceError):
    """Raised on quantization/calibration failure."""


class RuntimeError(InferenceError):
    """Raised on inference runtime failure."""


class TilingError(InferenceError):
    """Raised when auto-tiling cannot fit a tensor."""


class FusionError(InferenceError):
    """Raised when operator fusion produces an invalid graph."""
