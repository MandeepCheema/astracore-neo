"""
AstraCore Neo Compute — MAC Array.

Simulates the chip's 24,576-MAC compute array:
  - 48 cores × 512 MAC units per core
  - 5-stage pipeline: FETCH → DECODE → EXECUTE → ACCUMULATE → WRITEBACK
  - Precision modes: INT4, INT8, FP8, FP16, BF16, FP32
  - INT4 delivers 2× throughput vs INT8 (two ops per MAC cycle)
  - Utilisation tracking and TOPS estimation
  - HAL integration: MAC_CTRL (0x0030), MAC_STATUS (0x0034), IRQ_MAC_DONE

Chip spec: 24,576 MAC units, 5-stage pipeline, 2.5–3.2 GHz,
           >90% MAC utilisation for CNNs/ViTs/LLMs via AI-driven compiler.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from .exceptions import MACError, PrecisionError

# IRQ bit for MAC done (matches hal/interrupts.py)
IRQ_MAC_DONE = 0

# ---------------------------------------------------------------------------
# Architecture constants
# ---------------------------------------------------------------------------
NUM_CORES       = 48
MACS_PER_CORE   = 512
TOTAL_MACS      = NUM_CORES * MACS_PER_CORE   # 24,576
PIPELINE_STAGES = 5
CLK_DEFAULT_GHZ = 3.2


# ---------------------------------------------------------------------------
# Precision modes
# ---------------------------------------------------------------------------

class PrecisionMode(Enum):
    INT4  = "int4"    # 4-bit integer  — 2× throughput (two ops/MAC)
    INT8  = "int8"    # 8-bit integer  — 1× throughput (baseline)
    FP8   = "fp8"     # 8-bit float    — 1× throughput
    FP16  = "fp16"    # 16-bit float   — 0.5× throughput
    BF16  = "bf16"    # bfloat16       — 0.5× throughput
    FP32  = "fp32"    # 32-bit float   — 0.25× throughput

# Throughput multiplier relative to INT8
_THROUGHPUT_MUL = {
    PrecisionMode.INT4:  2.0,
    PrecisionMode.INT8:  1.0,
    PrecisionMode.FP8:   1.0,
    PrecisionMode.FP16:  0.5,
    PrecisionMode.BF16:  0.5,
    PrecisionMode.FP32:  0.25,
}

# numpy dtype to use for each precision in simulation
_NP_DTYPE = {
    PrecisionMode.INT4:  np.int8,     # stored as int8, ops counted at 2×
    PrecisionMode.INT8:  np.int8,
    PrecisionMode.FP8:   np.float16,  # approximate: FP8 simulated via FP16
    PrecisionMode.FP16:  np.float16,
    PrecisionMode.BF16:  np.float32,  # numpy has no bfloat16; use float32
    PrecisionMode.FP32:  np.float32,
}


# ---------------------------------------------------------------------------
# Pipeline stage tracker
# ---------------------------------------------------------------------------

class PipelineStage(Enum):
    FETCH       = 0
    DECODE      = 1
    EXECUTE     = 2
    ACCUMULATE  = 3
    WRITEBACK   = 4


# ---------------------------------------------------------------------------
# Single MAC core
# ---------------------------------------------------------------------------

class MACCore:
    """One of 48 compute cores, each housing 512 MAC units."""

    def __init__(self, core_id: int) -> None:
        self.core_id   = core_id
        self._enabled  = True
        self.ops_done  = 0       # total MAC operations executed
        self.stage     = PipelineStage.FETCH

    def enable(self)  -> None: self._enabled = True
    def disable(self) -> None: self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def matmul_slice(
        self,
        A: np.ndarray,
        B: np.ndarray,
        precision: PrecisionMode,
    ) -> np.ndarray:
        """
        Execute a matrix slice multiply on this core's 512 MACs.
        A: (M, K)  B: (K, N)  → (M, N)
        Advances pipeline through all 5 stages.
        """
        if not self._enabled:
            raise MACError(f"Core {self.core_id} is disabled")
        dtype = _NP_DTYPE[precision]
        a = A.astype(dtype)
        b = B.astype(dtype)
        for stage in PipelineStage:
            self.stage = stage
        result = a @ b
        macs = A.shape[0] * A.shape[1] * B.shape[1]
        mul = _THROUGHPUT_MUL[precision]
        self.ops_done += int(macs * mul)
        self.stage = PipelineStage.FETCH   # reset to idle
        return result.astype(np.float32)


# ---------------------------------------------------------------------------
# MAC Array controller
# ---------------------------------------------------------------------------

class MACArray:
    """
    24,576-MAC array controller.

    Distributes matmul/conv work across 48 cores, tracks utilisation,
    estimates TOPS, and integrates with the HAL register file and IRQ.

    Usage::

        array = MACArray(dev)
        C = array.matmul(A, B, PrecisionMode.INT8)
        print(array.utilisation_pct)
        print(array.tops_achieved)
    """

    def __init__(self, dev=None) -> None:
        self._dev   = dev
        self._cores = [MACCore(i) for i in range(NUM_CORES)]
        self._clock_ghz = CLK_DEFAULT_GHZ
        self._precision = PrecisionMode.INT8
        self.total_ops: int = 0           # lifetime MAC operations
        self.total_calls: int = 0         # number of compute calls
        self._last_utilisation: float = 0.0
        self._last_tops: float = 0.0
        self._sync_hal()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_precision(self, mode: PrecisionMode) -> None:
        self._precision = mode

    def set_clock(self, ghz: float) -> None:
        if not (0.5 <= ghz <= 3.2):
            raise MACError(f"Clock {ghz} GHz out of range [0.5–3.2]")
        self._clock_ghz = ghz

    def enable_core(self, core_id: int) -> None:
        self._validate_core(core_id)
        self._cores[core_id].enable()

    def disable_core(self, core_id: int) -> None:
        self._validate_core(core_id)
        self._cores[core_id].disable()

    @property
    def active_cores(self) -> int:
        return sum(1 for c in self._cores if c.enabled)

    @property
    def active_macs(self) -> int:
        return self.active_cores * MACS_PER_CORE

    # ------------------------------------------------------------------
    # Compute operations
    # ------------------------------------------------------------------

    def matmul(
        self,
        A: np.ndarray,
        B: np.ndarray,
        precision: Optional[PrecisionMode] = None,
    ) -> np.ndarray:
        """
        Distributed matrix multiply: C = A @ B.

        A: (M, K)  B: (K, N)  → C: (M, N)

        Rows of A are distributed across enabled cores. If M < active_cores,
        only that many cores are used; the rest sit idle (utilisation < 100%).
        """
        mode = precision or self._precision
        if A.ndim != 2 or B.ndim != 2:
            raise MACError("matmul requires 2-D arrays")
        if A.shape[1] != B.shape[0]:
            raise MACError(
                f"Shape mismatch: A{A.shape} @ B{B.shape} — inner dims must match"
            )

        M, K = A.shape
        _, N = B.shape
        enabled = [c for c in self._cores if c.enabled]
        if not enabled:
            raise MACError("All cores disabled — cannot compute")

        # Distribute rows across cores
        rows_per_core = max(1, math.ceil(M / len(enabled)))
        result_parts: List[np.ndarray] = []
        cores_used = 0

        for i, core in enumerate(enabled):
            row_start = i * rows_per_core
            if row_start >= M:
                break
            row_end = min(row_start + rows_per_core, M)
            slice_A = A[row_start:row_end, :]
            result_parts.append(core.matmul_slice(slice_A, B, mode))
            cores_used += 1

        result = np.vstack(result_parts)
        mac_ops = M * K * N
        self.total_ops  += mac_ops
        self.total_calls += 1
        self._update_stats(mac_ops, cores_used, len(enabled))
        self._fire_done()
        return result

    def conv2d(
        self,
        inp: np.ndarray,
        weight: np.ndarray,
        stride: int = 1,
        padding: int = 0,
        precision: Optional[PrecisionMode] = None,
    ) -> np.ndarray:
        """
        2D convolution via im2col → matmul.

        inp:    (C_in, H, W)
        weight: (C_out, C_in, kH, kW)
        → out:  (C_out, H_out, W_out)
        """
        mode = precision or self._precision
        if inp.ndim != 3 or weight.ndim != 4:
            raise MACError("conv2d expects inp(C,H,W) and weight(Cout,Cin,kH,kW)")
        C_in, H, W = inp.shape
        C_out, _, kH, kW = weight.shape
        if weight.shape[1] != C_in:
            raise MACError("Channel mismatch between input and weight")

        H_out = (H + 2 * padding - kH) // stride + 1
        W_out = (W + 2 * padding - kW) // stride + 1

        # im2col
        col = self._im2col(inp, kH, kW, stride, padding)  # (C_in*kH*kW, H_out*W_out)
        W_mat = weight.reshape(C_out, -1)                  # (C_out, C_in*kH*kW)

        out_flat = self.matmul(W_mat, col, mode)           # (C_out, H_out*W_out)
        return out_flat.reshape(C_out, H_out, W_out)

    def elementwise_mul(
        self,
        A: np.ndarray,
        B: np.ndarray,
        precision: Optional[PrecisionMode] = None,
    ) -> np.ndarray:
        """Element-wise multiply (used in sparsity masking, gating)."""
        mode = precision or self._precision
        if A.shape != B.shape:
            raise MACError(f"Shape mismatch: {A.shape} vs {B.shape}")
        dtype = _NP_DTYPE[mode]
        result = A.astype(dtype) * B.astype(dtype)
        self.total_ops += A.size
        self.total_calls += 1
        self._fire_done()
        return result.astype(np.float32)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def utilisation_pct(self) -> float:
        """MAC utilisation of the last operation (0–100%)."""
        return self._last_utilisation

    @property
    def tops_achieved(self) -> float:
        """Estimated TOPS of the last operation at current clock."""
        return self._last_tops

    def peak_tops(self, precision: PrecisionMode = PrecisionMode.INT8) -> float:
        """Theoretical peak TOPS at current clock and precision."""
        mul = _THROUGHPUT_MUL[precision]
        # TOPS = MACs × 2 (mul+add) × clock × multiplier / 1e12
        return (self.active_macs * 2 * self._clock_ghz * 1e9 * mul) / 1e12

    def reset_stats(self) -> None:
        self.total_ops   = 0
        self.total_calls = 0
        for c in self._cores:
            c.ops_done = 0
        self._last_utilisation = 0.0
        self._last_tops = 0.0

    def reset(self) -> None:
        self.reset_stats()
        for c in self._cores:
            c.enable()
        self._sync_hal()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_stats(self, mac_ops: int, cores_used: int, cores_available: int) -> None:
        self._last_utilisation = min(100.0, (cores_used / cores_available) * 100.0)
        # Estimate TOPS: ops × 2 (mul+add) / clock_period / 1e12
        mul = _THROUGHPUT_MUL[self._precision]
        effective_ops = mac_ops * 2 * mul
        self._last_tops = (effective_ops * self._clock_ghz * 1e9) / 1e12
        self._sync_hal()

    def _sync_hal(self) -> None:
        if self._dev is None:
            return
        # MAC_STATUS: bits[15:8] = utilisation%, bits[7:0] = active_cores
        util_int = int(self._last_utilisation)
        status = (util_int << 8) | (self.active_cores & 0xFF)
        self._dev.regs._hw_write(0x0034, status)

    def _fire_done(self) -> None:
        if self._dev is not None:
            self._dev.irq.fire(IRQ_MAC_DONE)

    def _validate_core(self, core_id: int) -> None:
        if not (0 <= core_id < NUM_CORES):
            raise MACError(f"core_id {core_id} out of range [0–{NUM_CORES - 1}]")

    @staticmethod
    def _im2col(
        inp: np.ndarray,
        kH: int, kW: int,
        stride: int, padding: int,
    ) -> np.ndarray:
        C, H, W = inp.shape
        if padding:
            inp = np.pad(inp, ((0, 0), (padding, padding), (padding, padding)))
        H_out = (H + 2 * padding - kH) // stride + 1
        W_out = (W + 2 * padding - kW) // stride + 1
        col = np.zeros((C * kH * kW, H_out * W_out), dtype=inp.dtype)
        for h in range(H_out):
            for w in range(W_out):
                patch = inp[:, h*stride:h*stride+kH, w*stride:w*stride+kW]
                col[:, h * W_out + w] = patch.ravel()
        return col
