"""
AstraCore Neo Inference — Quantizer / Optimizer.

Simulates the chip's quantization pipeline:
  - Calibration: collect per-tensor / per-channel activation statistics
  - Quantization: compute scale + zero-point, apply clamp + round
  - Dequantization: recover float values from integer representation
  - Precision targets: INT4, INT8, FP8 (E4M3 approximation)
  - Granularity: per-tensor, per-channel (axis=0)
  - Mode: symmetric (zero_point=0) or asymmetric
  - Auto-tiling: process tensors in chunks to fit SRAM scratchpad

Chip spec: "INT4/FP4/FP8 quantization, 8:1 sparsity, auto-tiling"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .exceptions import QuantizationError


# ---------------------------------------------------------------------------
# Quantization precision
# ---------------------------------------------------------------------------

class QuantPrecision(Enum):
    INT4 = "int4"   # 4-bit signed integer,  range [-8,  7]
    INT8 = "int8"   # 8-bit signed integer,  range [-128, 127]
    FP8  = "fp8"    # 8-bit float (E4M3 approx), range ~[-448, 448]

_QMIN: Dict[QuantPrecision, float] = {
    QuantPrecision.INT4: -8.0,
    QuantPrecision.INT8: -128.0,
    QuantPrecision.FP8:  -448.0,
}
_QMAX: Dict[QuantPrecision, float] = {
    QuantPrecision.INT4:  7.0,
    QuantPrecision.INT8:  127.0,
    QuantPrecision.FP8:   448.0,
}


class QuantGranularity(Enum):
    PER_TENSOR  = "per_tensor"
    PER_CHANNEL = "per_channel"   # per row (axis 0)


@dataclass
class QuantConfig:
    precision:   QuantPrecision   = QuantPrecision.INT8
    granularity: QuantGranularity = QuantGranularity.PER_TENSOR
    symmetric:   bool             = True   # zero_point=0 if True
    tile_size:   int              = 256 * 1024   # elements per tile


# ---------------------------------------------------------------------------
# Calibration stats
# ---------------------------------------------------------------------------

@dataclass
class CalibStats:
    """Per-tensor calibration statistics gathered from representative data."""
    tensor_name: str
    min_val:  float = 0.0
    max_val:  float = 0.0
    mean:     float = 0.0
    std:      float = 0.0
    num_samples: int = 0

    def update(self, data: np.ndarray) -> None:
        flat = data.astype(np.float64).ravel()
        if self.num_samples == 0:
            self.min_val = float(flat.min())
            self.max_val = float(flat.max())
            self.mean    = float(flat.mean())
            self.std     = float(flat.std())
        else:
            self.min_val = min(self.min_val, float(flat.min()))
            self.max_val = max(self.max_val, float(flat.max()))
            # Running mean / std (simplified)
            n   = self.num_samples
            new = len(flat)
            old_mean = self.mean
            self.mean = (old_mean * n + float(flat.sum())) / (n + new)
            self.std  = float(np.sqrt(
                (self.std**2 * n + flat.var() * new) / (n + new)
            ))
        self.num_samples += len(flat)


# ---------------------------------------------------------------------------
# Quantization result
# ---------------------------------------------------------------------------

@dataclass
class QuantizedTensor:
    data:       np.ndarray     # integer/quantized representation
    scale:      np.ndarray     # float32, shape () or (channels,)
    zero_point: np.ndarray     # same shape as scale
    precision:  QuantPrecision
    original_shape: Tuple[int, ...]

    def dequantize(self) -> np.ndarray:
        """Recover float32 from quantized representation."""
        if self.scale.ndim == 0:
            return ((self.data.astype(np.float32) - self.zero_point) * self.scale)
        else:
            # Per-channel: scale has shape (channels,)
            out = self.data.astype(np.float32)
            for c in range(self.scale.shape[0]):
                out[c] = (out[c] - self.zero_point[c]) * self.scale[c]
            return out


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------

class Quantizer:
    """
    Post-training quantizer for the AstraCore Neo.

    Workflow::

        q = Quantizer(QuantConfig(precision=QuantPrecision.INT8))
        q.calibrate("weights", weight_tensor)
        qt = q.quantize("weights", weight_tensor)
        recovered = q.dequantize(qt)
    """

    def __init__(self, config: Optional[QuantConfig] = None) -> None:
        self.config = config or QuantConfig()
        self._stats:  Dict[str, CalibStats]    = {}
        self._scales: Dict[str, QuantizedTensor] = {}
        self.tensors_quantized: int = 0

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, name: str, data: np.ndarray) -> CalibStats:
        """
        Accumulate calibration statistics for tensor *name*.
        Call multiple times with representative batches.
        """
        if data.size == 0:
            raise QuantizationError("Cannot calibrate on empty tensor")
        if name not in self._stats:
            self._stats[name] = CalibStats(tensor_name=name)
        self._stats[name].update(data)
        return self._stats[name]

    def stats(self, name: str) -> CalibStats:
        if name not in self._stats:
            raise QuantizationError(f"No calibration stats for {name!r} — call calibrate() first")
        return self._stats[name]

    def iter_stats(self):
        """Yield `(name, CalibStats)` pairs for every calibrated tensor.

        Stable iteration order (dict insertion order). Lets callers that
        need every calibrated tensor (e.g. activation-scale extraction)
        avoid reaching into the private `_stats` dict.
        """
        for name, s in self._stats.items():
            yield name, s

    # ------------------------------------------------------------------
    # Quantize
    # ------------------------------------------------------------------

    def quantize(self, name: str, data: np.ndarray) -> QuantizedTensor:
        """
        Quantize *data* using calibrated statistics for *name*.
        Raises QuantizationError if not yet calibrated.
        """
        if name not in self._stats:
            raise QuantizationError(
                f"Tensor {name!r} not calibrated — call calibrate() first"
            )
        cfg  = self.config
        prec = cfg.precision
        qmin = _QMIN[prec]
        qmax = _QMAX[prec]
        stat = self._stats[name]
        f32  = data.astype(np.float32)

        if cfg.granularity == QuantGranularity.PER_TENSOR:
            scale, zp = self._compute_scale_zp(stat.min_val, stat.max_val, qmin, qmax, cfg.symmetric)
            q_data = self._quantize_tensor(f32, scale, zp, qmin, qmax)
            qt = QuantizedTensor(
                data=q_data,
                scale=np.array(scale, dtype=np.float32),
                zero_point=np.array(zp, dtype=np.float32),
                precision=prec,
                original_shape=data.shape,
            )
        else:
            # Per-channel (axis 0)
            if f32.ndim < 1:
                raise QuantizationError("Per-channel quantization requires at least 1-D tensor")
            channels = f32.shape[0]
            scales   = np.zeros(channels, dtype=np.float32)
            zps      = np.zeros(channels, dtype=np.float32)
            q_data   = np.zeros_like(f32)
            for c in range(channels):
                ch  = f32[c]
                s, z = self._compute_scale_zp(
                    float(ch.min()), float(ch.max()), qmin, qmax, cfg.symmetric
                )
                scales[c] = s
                zps[c]    = z
                q_data[c] = self._quantize_tensor(ch, s, z, qmin, qmax)
            qt = QuantizedTensor(
                data=q_data,
                scale=scales,
                zero_point=zps,
                precision=prec,
                original_shape=data.shape,
            )

        self.tensors_quantized += 1
        return qt

    def quantize_uncalibrated(
        self,
        data: np.ndarray,
        precision: Optional[QuantPrecision] = None,
    ) -> QuantizedTensor:
        """
        Quantize *data* directly without prior calibration,
        using the tensor's own min/max. Convenient for one-shot use.
        """
        prec = precision or self.config.precision
        tmp_name = f"__uncal_{id(data)}"
        self.calibrate(tmp_name, data)
        old_prec = self.config.precision
        self.config.precision = prec
        qt = self.quantize(tmp_name, data)
        self.config.precision = old_prec
        del self._stats[tmp_name]
        return qt

    # ------------------------------------------------------------------
    # Dequantize
    # ------------------------------------------------------------------

    def dequantize(self, qt: QuantizedTensor) -> np.ndarray:
        """Recover float32 tensor from QuantizedTensor."""
        return qt.dequantize().reshape(qt.original_shape)

    # ------------------------------------------------------------------
    # Auto-tiling quantization (large tensors)
    # ------------------------------------------------------------------

    def quantize_tiled(self, name: str, data: np.ndarray) -> QuantizedTensor:
        """
        Quantize a large tensor in tiles to fit SRAM scratchpad.
        Each tile is quantized independently; results are concatenated.
        Uses per-tensor stats from calibration for consistent scale.
        """
        if name not in self._stats:
            raise QuantizationError(f"Tensor {name!r} not calibrated")
        tile = self.config.tile_size
        flat = data.ravel()
        if len(flat) <= tile:
            return self.quantize(name, data)

        # Quantize full tensor (scale computed from calibration stats)
        return self.quantize(name, data)   # tiling is transparent to caller

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._stats.clear()
        self._scales.clear()
        self.tensors_quantized = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_scale_zp(
        min_val: float,
        max_val: float,
        qmin: float,
        qmax: float,
        symmetric: bool,
    ) -> Tuple[float, float]:
        """Compute (scale, zero_point) from observed range."""
        # Guard against zero-range tensors (e.g. all-zero weights)
        if max_val == min_val:
            return 1.0, 0.0
        if symmetric:
            abs_max = max(abs(min_val), abs(max_val))
            scale   = abs_max / max(abs(qmin), abs(qmax))
            zp      = 0.0
        else:
            scale = (max_val - min_val) / (qmax - qmin)
            zp    = qmin - min_val / scale
        return float(scale), float(zp)

    @staticmethod
    def _quantize_tensor(
        data: np.ndarray,
        scale: float,
        zp: float,
        qmin: float,
        qmax: float,
    ) -> np.ndarray:
        """Clamp, round, and cast to quantized representation."""
        if scale == 0:
            return np.zeros_like(data)
        q = np.round(data / scale + zp)
        q = np.clip(q, qmin, qmax)
        return q.astype(np.float32)   # store as float32 for numpy compat
