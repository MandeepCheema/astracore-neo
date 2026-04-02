"""
AstraCore Neo — Camera subsystem simulation.

Models the MIPI CSI-2 camera interface, ISP-Pro pipeline:
  - Up to 8K HDR frame capture (simulated as numpy arrays)
  - ISP stages: demosaic, white-balance, noise-reduction, tone-map, gamma
  - AI denoising (simplified statistical model)
  - Frame metadata tracking (exposure, gain, temperature)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

from .exceptions import CameraError, FrameError


# ---------------------------------------------------------------------------
# Enums & configuration
# ---------------------------------------------------------------------------

class BayerPattern(Enum):
    RGGB = "RGGB"
    BGGR = "BGGR"
    GRBG = "GRBG"
    GBRG = "GBRG"


class ISPStage(Enum):
    RAW        = auto()
    DEMOSAIC   = auto()
    WHITE_BAL  = auto()
    DENOISE    = auto()
    TONE_MAP   = auto()
    GAMMA      = auto()
    OUTPUT     = auto()


class PixelFormat(Enum):
    RAW10 = "RAW10"   # 10-bit Bayer raw
    RAW12 = "RAW12"   # 12-bit Bayer raw
    YUV420 = "YUV420"
    RGB888 = "RGB888"
    HDR    = "HDR"    # 32-bit float HDR


@dataclass
class CameraConfig:
    """MIPI CSI-2 + ISP configuration."""
    width: int = 3840             # pixels
    height: int = 2160            # pixels (4K default)
    fps: float = 30.0
    bayer: BayerPattern = BayerPattern.RGGB
    pixel_format: PixelFormat = PixelFormat.RGB888
    hdr_enabled: bool = False
    lanes: int = 4                # MIPI CSI-2 lanes (1–4)
    bit_depth: int = 10
    ai_denoise: bool = True
    gamma: float = 2.2


@dataclass
class FrameMetadata:
    frame_id: int
    timestamp_us: float
    exposure_us: float
    analog_gain: float
    digital_gain: float
    color_temp_k: float
    isp_stages_applied: list[str] = field(default_factory=list)


@dataclass
class Frame:
    """A captured and ISP-processed camera frame."""
    data: np.ndarray              # H×W×C (float32, [0,1])
    metadata: FrameMetadata
    width: int
    height: int
    channels: int
    pixel_format: PixelFormat

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype


# ---------------------------------------------------------------------------
# ISP pipeline
# ---------------------------------------------------------------------------

class ISPPipeline:
    """
    Simulated Image Signal Processing pipeline.

    Stages (in order):
      1. Demosaic   — reconstruct RGB from Bayer mosaic (bilinear sim)
      2. White balance — scale R/G/B channels by gains
      3. AI Denoise — Gaussian smoothing as a proxy for learned denoising
      4. Tone map   — Reinhard global tone mapping (for HDR)
      5. Gamma      — power-law gamma correction
    """

    def __init__(self, config: CameraConfig) -> None:
        self._cfg = config

    def demosaic(self, raw: np.ndarray) -> np.ndarray:
        """
        Bayer → RGB demosaic (nearest-neighbour simulation).
        Input: H×W single-channel uint16 array.
        Output: H×W×3 float32 array in [0, 1].
        """
        if raw.ndim != 2:
            raise FrameError(f"demosaic expects 2-D raw array, got shape {raw.shape}")
        h, w = raw.shape
        scale = float((1 << self._cfg.bit_depth) - 1)
        norm = raw.astype(np.float32) / scale

        # Tile 2×2 RGGB pattern across the whole frame
        r = np.zeros((h, w), np.float32)
        g = np.zeros((h, w), np.float32)
        b = np.zeros((h, w), np.float32)

        pat = self._cfg.bayer
        if pat == BayerPattern.RGGB:
            r[0::2, 0::2] = norm[0::2, 0::2]
            g[0::2, 1::2] = norm[0::2, 1::2]
            g[1::2, 0::2] = norm[1::2, 0::2]
            b[1::2, 1::2] = norm[1::2, 1::2]
        elif pat == BayerPattern.BGGR:
            b[0::2, 0::2] = norm[0::2, 0::2]
            g[0::2, 1::2] = norm[0::2, 1::2]
            g[1::2, 0::2] = norm[1::2, 0::2]
            r[1::2, 1::2] = norm[1::2, 1::2]
        elif pat == BayerPattern.GRBG:
            g[0::2, 0::2] = norm[0::2, 0::2]
            r[0::2, 1::2] = norm[0::2, 1::2]
            b[1::2, 0::2] = norm[1::2, 0::2]
            g[1::2, 1::2] = norm[1::2, 1::2]
        else:  # GBRG
            g[0::2, 0::2] = norm[0::2, 0::2]
            b[0::2, 1::2] = norm[0::2, 1::2]
            r[1::2, 0::2] = norm[1::2, 0::2]
            g[1::2, 1::2] = norm[1::2, 1::2]

        return np.stack([r, g, b], axis=-1)

    def white_balance(
        self,
        img: np.ndarray,
        r_gain: float = 1.6,
        g_gain: float = 1.0,
        b_gain: float = 1.8,
    ) -> np.ndarray:
        """Apply per-channel white balance gains, clip to [0, 1]."""
        gains = np.array([r_gain, g_gain, b_gain], dtype=np.float32)
        return np.clip(img * gains, 0.0, 1.0)

    def ai_denoise(self, img: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        Lightweight denoise proxy.
        Uses a 3×3 mean filter weighted by 'strength'.
        strength=0 → passthrough, strength=1 → full blur.
        """
        if not self._cfg.ai_denoise:
            return img
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength == 0.0:
            return img
        # Manual 3×3 average
        kernel_size = 3
        pad = kernel_size // 2
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        blurred = np.zeros_like(img)
        count = 0
        for dy in range(kernel_size):
            for dx in range(kernel_size):
                blurred += padded[dy:dy + img.shape[0], dx:dx + img.shape[1], :]
                count += 1
        blurred /= count
        return (1.0 - strength) * img + strength * blurred

    def tone_map(self, img: np.ndarray) -> np.ndarray:
        """Reinhard global tone mapping: L' = L / (1 + L)."""
        if not self._cfg.hdr_enabled:
            return img
        return img / (1.0 + img)

    def gamma_correct(self, img: np.ndarray) -> np.ndarray:
        """Apply gamma correction: out = in^(1/gamma)."""
        gamma = self._cfg.gamma
        if gamma <= 0:
            raise FrameError(f"Invalid gamma value: {gamma}")
        return np.clip(img, 0.0, None) ** (1.0 / gamma)

    def run(self, raw: np.ndarray, meta: FrameMetadata) -> np.ndarray:
        """Run the full ISP pipeline on a raw Bayer frame."""
        img = self.demosaic(raw)
        meta.isp_stages_applied.append(ISPStage.DEMOSAIC.name)

        img = self.white_balance(img)
        meta.isp_stages_applied.append(ISPStage.WHITE_BAL.name)

        if self._cfg.ai_denoise:
            img = self.ai_denoise(img)
            meta.isp_stages_applied.append(ISPStage.DENOISE.name)

        if self._cfg.hdr_enabled:
            img = self.tone_map(img)
            meta.isp_stages_applied.append(ISPStage.TONE_MAP.name)

        img = self.gamma_correct(img)
        meta.isp_stages_applied.append(ISPStage.GAMMA.name)

        return np.clip(img, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Camera sensor
# ---------------------------------------------------------------------------

class CameraSensor:
    """
    Simulated MIPI CSI-2 camera sensor with ISP-Pro.

    Usage::

        cam = CameraSensor(CameraConfig(width=1920, height=1080))
        cam.power_on()
        frame = cam.capture()
        cam.power_off()
    """

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        self._cfg = config or CameraConfig()
        self._powered = False
        self._frame_counter = 0
        self._isp = ISPPipeline(self._cfg)
        self._exposure_us: float = 10_000.0   # 10 ms default
        self._analog_gain: float = 1.0
        self._digital_gain: float = 1.0
        self._color_temp_k: float = 6500.0    # daylight

    # ------------------------------------------------------------------
    # Power control
    # ------------------------------------------------------------------

    def power_on(self) -> None:
        if self._powered:
            raise CameraError("Camera already powered on")
        self._powered = True

    def power_off(self) -> None:
        if not self._powered:
            raise CameraError("Camera already powered off")
        self._powered = False

    @property
    def is_powered(self) -> bool:
        return self._powered

    # ------------------------------------------------------------------
    # Exposure / gain control
    # ------------------------------------------------------------------

    def set_exposure(self, exposure_us: float) -> None:
        if exposure_us <= 0:
            raise CameraError(f"Exposure must be > 0, got {exposure_us}")
        self._exposure_us = exposure_us

    def set_gain(self, analog: float = 1.0, digital: float = 1.0) -> None:
        if analog < 1.0 or analog > 64.0:
            raise CameraError(f"Analog gain out of range [1, 64]: {analog}")
        if digital < 1.0 or digital > 16.0:
            raise CameraError(f"Digital gain out of range [1, 16]: {digital}")
        self._analog_gain = analog
        self._digital_gain = digital

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def _generate_raw(self, rng: np.random.Generator) -> np.ndarray:
        """
        Generate a synthetic Bayer raw frame.
        Simulates exposure and gain effects.
        """
        h, w = self._cfg.height, self._cfg.width
        max_val = (1 << self._cfg.bit_depth) - 1

        # Normalised exposure factor (0..1)
        exp_factor = min(1.0, self._exposure_us / 33_333.0)  # 33 ms = full bright
        brightness = exp_factor * self._analog_gain * self._digital_gain
        brightness = min(brightness, 1.0)

        base = rng.integers(
            low=int(0.05 * max_val),
            high=int(max(0.06, brightness) * max_val),
            size=(h, w),
            dtype=np.uint16,
        )
        # Add small Gaussian noise
        noise = rng.normal(0, max_val * 0.01, size=(h, w)).astype(np.int32)
        raw = np.clip(base.astype(np.int32) + noise, 0, max_val).astype(np.uint16)
        return raw

    def capture(self, seed: Optional[int] = None) -> Frame:
        """
        Capture one frame through the full ISP pipeline.

        Args:
            seed: optional RNG seed for reproducible synthetic frames.

        Returns:
            Frame with ISP-processed RGB data and metadata.
        """
        if not self._powered:
            raise CameraError("Cannot capture: camera is powered off")

        self._frame_counter += 1
        rng = np.random.default_rng(seed if seed is not None else self._frame_counter)

        meta = FrameMetadata(
            frame_id=self._frame_counter,
            timestamp_us=time.monotonic() * 1e6,
            exposure_us=self._exposure_us,
            analog_gain=self._analog_gain,
            digital_gain=self._digital_gain,
            color_temp_k=self._color_temp_k,
        )

        raw = self._generate_raw(rng)
        processed = self._isp.run(raw, meta)

        return Frame(
            data=processed,
            metadata=meta,
            width=self._cfg.width,
            height=self._cfg.height,
            channels=3,
            pixel_format=PixelFormat.RGB888,
        )

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def config(self) -> CameraConfig:
        return self._cfg

    @property
    def frames_captured(self) -> int:
        return self._frame_counter

    def __repr__(self) -> str:
        return (
            f"CameraSensor({self._cfg.width}×{self._cfg.height}@{self._cfg.fps}fps, "
            f"powered={'ON' if self._powered else 'OFF'})"
        )
