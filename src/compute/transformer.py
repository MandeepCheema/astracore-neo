"""
AstraCore Neo Compute — Transformer Engine.

Simulates the chip's dedicated transformer acceleration block:
  - 8× Multi-Head Self-Attention (MHSA) heads, run in parallel
  - Rotary Position Embedding (RoPE)
  - Fused softmax (numerically stable)
  - Fused layer norm (pre-norm / post-norm)
  - Fused GeLU activation
  - Dynamic sparsity in attention (top-k sparse attention)
  - Full transformer block: LN → MHSA → LN → FFN

Chip spec: "8xMHSA, dynamic sparsity, rotary PE, fused softmax, layer norm, GeLU"

All ops use numpy; precision is controlled by the calling MACArray.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .exceptions import TransformerError


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_ATTENTION_HEADS = 8   # chip has 8 parallel MHSA units


# ---------------------------------------------------------------------------
# Fused primitives
# ---------------------------------------------------------------------------

def fused_softmax(x: np.ndarray, dim: int = -1) -> np.ndarray:
    """Numerically stable softmax (max-subtraction trick)."""
    x_max = np.max(x, axis=dim, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=dim, keepdims=True)


def fused_layer_norm(
    x: np.ndarray,
    gamma: Optional[np.ndarray] = None,
    beta: Optional[np.ndarray] = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """Layer normalisation over the last dimension."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var  = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    if gamma is not None:
        x_norm = x_norm * gamma
    if beta is not None:
        x_norm = x_norm + beta
    return x_norm


def fused_gelu(x: np.ndarray) -> np.ndarray:
    """GeLU activation: x × Φ(x) approximated via tanh formula."""
    return x * 0.5 * (1.0 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def rotary_position_embedding(
    x: np.ndarray,
    seq_len: int,
    head_dim: int,
) -> np.ndarray:
    """
    Apply Rotary Position Embedding (RoPE) to query or key tensor.

    x: (batch, seq_len, num_heads, head_dim)
    Returns tensor of the same shape with RoPE applied.
    """
    if x.shape[-1] != head_dim:
        raise TransformerError(
            f"RoPE head_dim mismatch: tensor last dim {x.shape[-1]} vs {head_dim}"
        )
    half = head_dim // 2
    theta = np.array([
        1.0 / (10000 ** (2 * i / head_dim)) for i in range(half)
    ], dtype=np.float32)
    positions = np.arange(seq_len, dtype=np.float32)
    angles = np.outer(positions, theta)  # (seq_len, half)
    cos = np.cos(angles)
    sin = np.sin(angles)

    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = np.concatenate([
        x1 * cos[None, :, None, :] - x2 * sin[None, :, None, :],
        x1 * sin[None, :, None, :] + x2 * cos[None, :, None, :],
    ], axis=-1)
    return rotated.astype(x.dtype)


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention:
    """
    Single MHSA block — the chip runs 8 of these in parallel.

    Each instance handles one head (or a subset of heads if num_heads > 1
    and split across chip units).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = NUM_ATTENTION_HEADS,
        dropout: float = 0.0,
        use_rope: bool = True,
        sparse_top_k: Optional[int] = None,
    ) -> None:
        if embed_dim % num_heads != 0:
            raise TransformerError(
                f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
            )
        self.embed_dim   = embed_dim
        self.num_heads   = num_heads
        self.head_dim    = embed_dim // num_heads
        self.scale       = math.sqrt(self.head_dim)
        self.use_rope    = use_rope
        self.sparse_top_k = sparse_top_k  # None = dense attention

        # Learnable projections (initialised to identity-like random)
        rng = np.random.default_rng(seed=0)
        self.W_q = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * 0.02
        self.W_k = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * 0.02
        self.W_v = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * 0.02
        self.W_o = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * 0.02

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.

        x:    (batch, seq_len, embed_dim)
        mask: (batch, seq_len, seq_len) optional causal/padding mask
        Returns: (output, attention_weights)
          output:  (batch, seq_len, embed_dim)
          attn:    (batch, num_heads, seq_len, seq_len)
        """
        B, T, D = x.shape
        if D != self.embed_dim:
            raise TransformerError(
                f"Input embed_dim {D} != expected {self.embed_dim}"
            )

        # Linear projections: (B, T, D)
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape to (B, T, num_heads, head_dim) → (B, num_heads, T, head_dim)
        Q = Q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Rotary PE on Q and K
        if self.use_rope:
            Q_rope = Q.transpose(0, 2, 1, 3)  # (B, T, H, head_dim)
            K_rope = K.transpose(0, 2, 1, 3)
            Q_rope = rotary_position_embedding(Q_rope, T, self.head_dim)
            K_rope = rotary_position_embedding(K_rope, T, self.head_dim)
            Q = Q_rope.transpose(0, 2, 1, 3)  # (B, H, T, head_dim)
            K = K_rope.transpose(0, 2, 1, 3)

        # Scaled dot-product attention: (B, H, T, T)
        scores = (Q @ K.transpose(0, 1, 3, 2)) / self.scale

        # Optional mask
        if mask is not None:
            scores = scores + mask[:, None, :, :]  # broadcast over heads

        # Dynamic sparsity: top-k sparse attention
        if self.sparse_top_k is not None and self.sparse_top_k < T:
            threshold = np.sort(scores, axis=-1)[
                :, :, :, -self.sparse_top_k
            ][..., None]
            scores = np.where(scores >= threshold, scores, -1e9)

        attn = fused_softmax(scores, dim=-1)   # (B, H, T, T)

        # Context: (B, H, T, head_dim)
        context = attn @ V
        # Reshape: (B, T, D)
        context = context.transpose(0, 2, 1, 3).reshape(B, T, D)
        output  = context @ self.W_o

        return output, attn


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward:
    """Position-wise FFN: Linear → GeLU → Linear (4× hidden expansion)."""

    def __init__(self, embed_dim: int, expansion: int = 4) -> None:
        self.embed_dim = embed_dim
        hidden = embed_dim * expansion
        rng = np.random.default_rng(seed=1)
        self.W1 = rng.standard_normal((embed_dim, hidden)).astype(np.float32) * 0.02
        self.W2 = rng.standard_normal((hidden, embed_dim)).astype(np.float32) * 0.02
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.b2 = np.zeros(embed_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = fused_gelu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2


# ---------------------------------------------------------------------------
# Full Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock:
    """
    One transformer decoder/encoder block:
      LN → MHSA → residual → LN → FFN → residual

    Uses pre-norm (LN before sub-layer) per modern practice (GPT-style).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = NUM_ATTENTION_HEADS,
        use_rope: bool = True,
        sparse_top_k: Optional[int] = None,
    ) -> None:
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, use_rope=use_rope,
                                            sparse_top_k=sparse_top_k)
        self.ffn   = FeedForward(embed_dim)
        self.norm1_gamma = np.ones(embed_dim,  dtype=np.float32)
        self.norm1_beta  = np.zeros(embed_dim, dtype=np.float32)
        self.norm2_gamma = np.ones(embed_dim,  dtype=np.float32)
        self.norm2_beta  = np.zeros(embed_dim, dtype=np.float32)

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        x: (batch, seq_len, embed_dim)
        Returns: (output, attention_weights)
        """
        # MHSA sub-layer
        x_norm, attn = self.attn.forward(
            fused_layer_norm(x, self.norm1_gamma, self.norm1_beta), mask
        )
        x = x + x_norm

        # FFN sub-layer
        x = x + self.ffn.forward(
            fused_layer_norm(x, self.norm2_gamma, self.norm2_beta)
        )
        return x, attn


# ---------------------------------------------------------------------------
# Transformer Engine — top-level controller
# ---------------------------------------------------------------------------

class TransformerEngine:
    """
    Manages the chip's 8 parallel MHSA units and transformer block execution.

    Usage::

        engine = TransformerEngine()
        block  = engine.build_block(embed_dim=512)
        out, attn = engine.run_block(block, x)
    """

    def __init__(self, dev=None) -> None:
        self._dev = dev
        self.num_heads = NUM_ATTENTION_HEADS
        self.blocks_run: int = 0
        self.total_tokens_processed: int = 0

    def build_block(
        self,
        embed_dim: int,
        use_rope: bool = True,
        sparse_top_k: Optional[int] = None,
    ) -> TransformerBlock:
        """Instantiate a TransformerBlock configured for this engine."""
        if embed_dim % self.num_heads != 0:
            raise TransformerError(
                f"embed_dim {embed_dim} must be divisible by num_heads {self.num_heads}"
            )
        return TransformerBlock(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            use_rope=use_rope,
            sparse_top_k=sparse_top_k,
        )

    def run_block(
        self,
        block: TransformerBlock,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run one transformer block forward pass.
        x: (batch, seq_len, embed_dim)
        Returns (output, attention_weights).
        """
        if x.ndim != 3:
            raise TransformerError("Input must be 3-D: (batch, seq_len, embed_dim)")
        out, attn = block.forward(x, mask)
        self.blocks_run += 1
        self.total_tokens_processed += x.shape[0] * x.shape[1]
        return out, attn

    # Expose fused primitives for use by other modules
    @staticmethod
    def softmax(x: np.ndarray, dim: int = -1) -> np.ndarray:
        return fused_softmax(x, dim)

    @staticmethod
    def layer_norm(x, gamma=None, beta=None, eps=1e-5) -> np.ndarray:
        return fused_layer_norm(x, gamma, beta, eps)

    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        return fused_gelu(x)

    @staticmethod
    def rope(x: np.ndarray, seq_len: int, head_dim: int) -> np.ndarray:
        return rotary_position_embedding(x, seq_len, head_dim)

    def reset_stats(self) -> None:
        self.blocks_run = 0
        self.total_tokens_processed = 0
