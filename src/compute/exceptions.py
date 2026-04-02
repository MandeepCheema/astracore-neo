"""AstraCore Neo Compute — Exception hierarchy."""


class ComputeError(Exception):
    """Base for all compute subsystem errors."""


class MACError(ComputeError):
    """Raised on invalid MAC array operation."""


class SparsityError(ComputeError):
    """Raised on invalid sparsity configuration or pruning failure."""


class TransformerError(ComputeError):
    """Raised on invalid transformer engine operation."""


class PrecisionError(ComputeError):
    """Raised when data does not match declared precision mode."""
