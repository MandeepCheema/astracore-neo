"""
AstraCore Neo Compute Engine.

Public API::

    from compute import MACArray, PrecisionMode
    from compute import SparsityEngine, SparsityPattern
    from compute import TransformerEngine, TransformerBlock
    from compute import MultiHeadSelfAttention, FeedForward
    from compute.transformer import fused_softmax, fused_layer_norm, fused_gelu
    from compute import ComputeError, MACError, SparsityError, TransformerError
"""

from .mac_array import MACArray, MACCore, PrecisionMode, NUM_CORES, MACS_PER_CORE, TOTAL_MACS
from .sparsity import SparsityEngine, SparsityPattern
from .transformer import (
    TransformerEngine, TransformerBlock,
    MultiHeadSelfAttention, FeedForward,
    fused_softmax, fused_layer_norm, fused_gelu, rotary_position_embedding,
)
from .exceptions import ComputeError, MACError, SparsityError, TransformerError, PrecisionError

__all__ = [
    "MACArray", "MACCore", "PrecisionMode", "NUM_CORES", "MACS_PER_CORE", "TOTAL_MACS",
    "SparsityEngine", "SparsityPattern",
    "TransformerEngine", "TransformerBlock",
    "MultiHeadSelfAttention", "FeedForward",
    "fused_softmax", "fused_layer_norm", "fused_gelu", "rotary_position_embedding",
    "ComputeError", "MACError", "SparsityError", "TransformerError", "PrecisionError",
]
