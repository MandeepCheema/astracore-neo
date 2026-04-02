"""
AstraCore Neo — Radar subsystem simulation.

Models a 4D mmWave radar (range, azimuth, elevation, Doppler velocity):
  - FMCW waveform simulation (range-Doppler processing)
  - CFAR detection (Cell-Averaging)
  - RadarDetection objects with range, azimuth, velocity, RCS
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .exceptions import RadarError


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RadarConfig:
    """FMCW radar configuration."""
    max_range_m: float = 250.0
    range_resolution_m: float = 0.15
    max_velocity_mps: float = 72.0        # ±72 m/s
    velocity_resolution_mps: float = 0.1
    azimuth_fov_deg: float = 120.0
    elevation_fov_deg: float = 20.0
    azimuth_resolution_deg: float = 1.0
    num_chirps: int = 128                  # chirps per frame
    num_samples: int = 512                 # ADC samples per chirp
    carrier_freq_ghz: float = 77.0
    bandwidth_ghz: float = 4.0
    noise_figure_db: float = 12.0
    # CFAR parameters
    cfar_guard_cells: int = 2
    cfar_training_cells: int = 8
    cfar_threshold_factor: float = 2.5    # linear scale above noise floor


# ---------------------------------------------------------------------------
# Detection output
# ---------------------------------------------------------------------------

@dataclass
class RadarDetection:
    """Single radar detection (target)."""
    range_m: float
    azimuth_deg: float
    elevation_deg: float
    velocity_mps: float          # positive = moving away, negative = approaching
    rcs_dbsm: float              # radar cross-section in dBsm
    snr_db: float
    detection_id: int


# ---------------------------------------------------------------------------
# Range-Doppler processing
# ---------------------------------------------------------------------------

class RangeDopplerProcessor:
    """
    Simulates FMCW range-Doppler processing.

    Pipeline:
      1. Generate synthetic ADC data cube (chirps × samples)
      2. 2D FFT → range-Doppler map
      3. CFAR detection in range dimension
    """

    def __init__(self, config: RadarConfig) -> None:
        self._cfg = config

    def _generate_adc_cube(
        self,
        targets: list[dict],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate a complex ADC cube (num_chirps × num_samples) for given targets.

        Each target dict: {'range_m', 'velocity_mps', 'rcs_linear'}
        """
        cfg = self._cfg
        c = 3e8  # speed of light
        # Range bin spacing
        range_bin_width = cfg.max_range_m / cfg.num_samples
        vel_bin_width = cfg.max_velocity_mps / cfg.num_chirps

        cube = np.zeros((cfg.num_chirps, cfg.num_samples), dtype=np.complex64)

        for tgt in targets:
            r = tgt.get("range_m", 50.0)
            v = tgt.get("velocity_mps", 0.0)
            rcs = tgt.get("rcs_linear", 1.0)

            range_bin = int(np.clip(r / range_bin_width, 0, cfg.num_samples - 1))
            # Doppler frequency proportional to velocity
            lam = c / (cfg.carrier_freq_ghz * 1e9)
            doppler_bin = int(
                np.clip(
                    v / vel_bin_width + cfg.num_chirps // 2,
                    0,
                    cfg.num_chirps - 1,
                )
            )

            amplitude = np.sqrt(rcs) * 10.0
            for chirp in range(cfg.num_chirps):
                phase = 2 * np.pi * (
                    range_bin * chirp / cfg.num_samples
                    + doppler_bin * chirp / cfg.num_chirps
                )
                cube[chirp, range_bin] += amplitude * np.exp(1j * phase)

        # Add thermal noise
        noise_power = 10 ** (cfg.noise_figure_db / 10) * 1e-3
        noise = rng.normal(0, np.sqrt(noise_power / 2), cube.shape).astype(np.float32) + \
                1j * rng.normal(0, np.sqrt(noise_power / 2), cube.shape).astype(np.float32)
        cube += noise.astype(np.complex64)

        return cube

    def range_doppler_map(self, adc_cube: np.ndarray) -> np.ndarray:
        """
        Compute 2D range-Doppler map via 2D FFT.

        Input: (num_chirps × num_samples) complex.
        Output: (num_chirps × num_samples) magnitude (float32).
        """
        rd = np.fft.fft2(adc_cube)
        rd = np.fft.fftshift(rd, axes=0)   # centre Doppler at zero
        return np.abs(rd).astype(np.float32)

    def cfar_detect(self, rd_map: np.ndarray) -> list[tuple[int, int]]:
        """
        1-D CA-CFAR in range dimension for each Doppler bin.

        Returns list of (doppler_bin, range_bin) detection indices.
        """
        cfg = self._cfg
        G = cfg.cfar_guard_cells
        T = cfg.cfar_training_cells
        alpha = cfg.cfar_threshold_factor

        detections: list[tuple[int, int]] = []
        n_doppler, n_range = rd_map.shape

        for d in range(n_doppler):
            row = rd_map[d]
            for r in range(T + G, n_range - T - G):
                left = row[r - T - G: r - G]
                right = row[r + G + 1: r + G + T + 1]
                noise_est = (left.sum() + right.sum()) / (2 * T)
                threshold = alpha * noise_est
                if row[r] > threshold:
                    detections.append((d, r))

        return detections


# ---------------------------------------------------------------------------
# Radar sensor
# ---------------------------------------------------------------------------

class RadarSensor:
    """
    Simulated 4D mmWave radar sensor.

    Usage::

        radar = RadarSensor()
        radar.power_on()
        detections = radar.scan()
        radar.power_off()
    """

    def __init__(self, config: Optional[RadarConfig] = None) -> None:
        self._cfg = config or RadarConfig()
        self._powered = False
        self._frame_counter = 0
        self._processor = RangeDopplerProcessor(self._cfg)

    def power_on(self) -> None:
        if self._powered:
            raise RadarError("Radar already powered on")
        self._powered = True

    def power_off(self) -> None:
        if not self._powered:
            raise RadarError("Radar already powered off")
        self._powered = False

    @property
    def is_powered(self) -> bool:
        return self._powered

    def scan(
        self,
        targets: Optional[list[dict]] = None,
        seed: Optional[int] = None,
    ) -> list[RadarDetection]:
        """
        Perform one radar scan.

        Args:
            targets: list of dicts with keys {'range_m', 'velocity_mps', 'rcs_linear'}.
                     If None, uses default synthetic scenario.
            seed: RNG seed.

        Returns:
            List of RadarDetection objects.
        """
        if not self._powered:
            raise RadarError("Cannot scan: radar is powered off")

        self._frame_counter += 1
        cfg = self._cfg
        rng = np.random.default_rng(seed if seed is not None else self._frame_counter)

        if targets is None:
            # Default: 3 synthetic targets
            targets = [
                {"range_m": 40.0,  "velocity_mps": -15.0, "rcs_linear": 10.0},
                {"range_m": 80.0,  "velocity_mps": 5.0,   "rcs_linear": 5.0},
                {"range_m": 150.0, "velocity_mps": 0.0,   "rcs_linear": 2.0},
            ]

        adc_cube = self._processor._generate_adc_cube(targets, rng)
        rd_map = self._processor.range_doppler_map(adc_cube)
        raw_detections = self._processor.cfar_detect(rd_map)

        range_bin_width = cfg.max_range_m / cfg.num_samples
        vel_bin_width = cfg.max_velocity_mps / cfg.num_chirps

        results: list[RadarDetection] = []
        for det_id, (d_bin, r_bin) in enumerate(raw_detections):
            r_m = r_bin * range_bin_width
            v_mps = (d_bin - cfg.num_chirps // 2) * vel_bin_width
            snr = float(rd_map[d_bin, r_bin]) / (1e-9 + float(rd_map.mean()))
            snr_db = float(10 * np.log10(max(snr, 1e-9)))
            rcs_db = float(10 * np.log10(max(1e-9, snr * (r_m ** 4) * 1e-6)))

            # Assign random azimuth/elevation within FOV
            az = float(rng.uniform(-cfg.azimuth_fov_deg / 2, cfg.azimuth_fov_deg / 2))
            el = float(rng.uniform(-cfg.elevation_fov_deg / 2, cfg.elevation_fov_deg / 2))

            results.append(RadarDetection(
                range_m=float(r_m),
                azimuth_deg=az,
                elevation_deg=el,
                velocity_mps=float(v_mps),
                rcs_dbsm=rcs_db,
                snr_db=snr_db,
                detection_id=det_id,
            ))

        return results

    @property
    def config(self) -> RadarConfig:
        return self._cfg

    @property
    def frames_captured(self) -> int:
        return self._frame_counter

    def __repr__(self) -> str:
        return (
            f"RadarSensor(77GHz, {self._cfg.max_range_m}m, "
            f"powered={'ON' if self._powered else 'OFF'})"
        )
