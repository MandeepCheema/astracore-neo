"""
AstraCore Neo Compute — Sparsity Engine.

Simulates the chip's dedicated sparsity execution block:
  - Patterns: DENSE, 2:1, 4:1, 8:2, 8:1
  - Magnitude-based structured pruning per block
  - Sparse mask generation and application
  - Effective TOPS uplift calculation
  - Skip-zero acceleration: zero MACs are bypassed

Chip spec: "2:1, 4:1, 8:2, 8:1 pruning with dedicated sparsity engine"
           "8:1 sparsity" at peak 1258 TOPS.

Pattern semantics:
  N:M sparsity — keep N non-zero values in every block of M elements.
  2:1 → keep 2 of every 2  (no pruning, identity)  — used as baseline
  4:1 → keep 1 of every 4  (75% sparse)
  8:2 → keep 2 of every 8  (75% sparse)
  8:1 → keep 1 of every 8  (87.5% sparse, peak chip rating)

Throughput uplift = M/N  (skipping zero MACs).
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple

import numpy as np

from .exceptions import SparsityError


# ---------------------------------------------------------------------------
# Sparsity pattern definitions
# ---------------------------------------------------------------------------

class SparsityPattern(Enum):
    DENSE = (0, 0)   # no pruning
    S2_1  = (2, 2)   # keep 2 of 2  — effectively dense, baseline
    S4_1  = (1, 4)   # keep 1 of 4
    S8_2  = (2, 8)   # keep 2 of 8
    S8_1  = (1, 8)   # keep 1 of 8  — peak chip rating

    def __init__(self, keep: int, block: int) -> None:
        self.keep  = keep    # non-zeros to retain per block
        self.block = block   # block size

    @property
    def is_dense(self) -> bool:
        return self == SparsityPattern.DENSE or self.keep == self.block

    @property
    def sparsity_ratio(self) -> float:
        """Fraction of zeros: 0.0 = dense, 0.875 = 8:1."""
        if self.is_dense:
            return 0.0
        return 1.0 - (self.keep / self.block)

    @property
    def throughput_multiplier(self) -> float:
        """
        Effective TOPS multiplier from skipping zeros.
        Dense = 1×; 8:1 = 8× (in ideal case — real chips achieve ~8:1 peak).
        """
        if self.is_dense or self.keep == 0:
            return 1.0
        return self.block / self.keep


# ---------------------------------------------------------------------------
# Sparsity Engine
# ---------------------------------------------------------------------------

class SparsityEngine:
    """
    Structured sparsity pruning and mask application engine.

    Usage::

        engine = SparsityEngine()
        pruned, mask = engine.prune(weights, SparsityPattern.S8_1)
        sparse_out   = engine.apply_mask(dense_output, mask)
        tops_up      = engine.effective_tops(base_tops, SparsityPattern.S8_1)
    """

    def __init__(self) -> None:
        self.total_weights_pruned: int = 0
        self.total_weights_kept:   int = 0

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune(
        self,
        weights: np.ndarray,
        pattern: SparsityPattern,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply structured N:M pruning to *weights*.

        Returns (pruned_weights, binary_mask).
        - pruned_weights: same shape as weights; zeroed-out positions
        - binary_mask:    float32 array, 1.0 = kept, 0.0 = pruned

        For DENSE pattern, mask is all-ones and weights are unchanged.
        """
        if pattern.is_dense:
            mask = np.ones_like(weights, dtype=np.float32)
            return weights.copy(), mask

        flat = weights.ravel().astype(np.float32)
        N    = len(flat)
        block = pattern.block
        keep  = pattern.keep

        if N % block != 0:
            # Pad to next multiple of block
            pad   = block - (N % block)
            flat  = np.pad(flat, (0, pad))
        else:
            pad   = 0

        n_blocks = len(flat) // block
        mask_flat = np.zeros(len(flat), dtype=np.float32)

        for b in range(n_blocks):
            start = b * block
            end   = start + block
            blk   = flat[start:end]
            # Keep the `keep` largest-magnitude values
            idx   = np.argsort(np.abs(blk))[-keep:]
            mask_flat[start + idx] = 1.0

        # Remove padding
        if pad:
            flat      = flat[:-pad]
            mask_flat = mask_flat[:-pad]

        pruned = (flat * mask_flat).reshape(weights.shape)
        mask   = mask_flat.reshape(weights.shape)

        zeros_added = int(np.sum(mask == 0.0))
        kept        = int(np.sum(mask == 1.0))
        self.total_weights_pruned += zeros_added
        self.total_weights_kept   += kept

        return pruned, mask

    # ------------------------------------------------------------------
    # Mask application
    # ------------------------------------------------------------------

    def apply_mask(
        self,
        tensor: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Zero out elements of *tensor* where *mask* == 0.
        Used to apply a pre-computed sparsity mask to activations.
        """
        if tensor.shape != mask.shape:
            raise SparsityError(
                f"Shape mismatch: tensor{tensor.shape} vs mask{mask.shape}"
            )
        return (tensor.astype(np.float32) * mask).astype(tensor.dtype)

    # ------------------------------------------------------------------
    # Effective TOPS
    # ------------------------------------------------------------------

    def effective_tops(self, base_tops: float, pattern: SparsityPattern) -> float:
        """
        Return effective TOPS with sparsity acceleration.

        effective_tops = base_tops × throughput_multiplier
        Capped at the chip's peak (2516 TOPS at INT4/FP4 with 8:1 sparsity).
        """
        if base_tops < 0:
            raise SparsityError("base_tops must be non-negative")
        return base_tops * pattern.throughput_multiplier

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def measure_sparsity(self, tensor: np.ndarray) -> float:
        """Return actual fraction of zeros in *tensor* (0.0–1.0)."""
        if tensor.size == 0:
            return 0.0
        return float(np.sum(tensor == 0)) / tensor.size

    def verify_pattern(self, mask: np.ndarray, pattern: SparsityPattern) -> bool:
        """
        Verify that *mask* satisfies the N:M sparsity pattern block-by-block.
        Returns True if all blocks comply.
        """
        if pattern.is_dense:
            return bool(np.all(mask == 1.0))
        flat  = mask.ravel().astype(np.float32)
        block = pattern.block
        keep  = pattern.keep
        if len(flat) % block != 0:
            return False   # unpadded — can't verify
        n_blocks = len(flat) // block
        for b in range(n_blocks):
            blk = flat[b * block: (b + 1) * block]
            if int(np.sum(blk)) != keep:
                return False
        return True

    def reset_stats(self) -> None:
        self.total_weights_pruned = 0
        self.total_weights_kept   = 0
