"""Dataset abstractions — the OEM-facing contract.

Every concrete connector produces the same typed ``Sample`` objects so
the perception pipeline is dataset-agnostic.

Design notes
------------
* ``Dataset`` is a collection of ``Scene``s. Think of nuScenes' 850-scene
  split; each scene is ~20 s of multi-sensor data.
* ``Scene`` is an ordered list of ``Sample``s, each at a single
  timestamp with all sensors' data.
* ``Sample`` holds typed frames (``CameraFrame``, ``LidarFrame``,
  ``RadarFrame``), plus optional pose (ego IMU), GNSS, CAN, and ground-
  truth annotations.
* All spatial coordinates are in the ego-vehicle frame: +x forward,
  +y left, +z up. Connectors convert dataset-specific conventions on
  ingest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterator, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Sensor kind enum
# ---------------------------------------------------------------------------

class SensorKind(Enum):
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    IMU = "imu"
    GNSS = "gnss"
    CAN = "can"
    # Added 2026-04-19 to close the "what sensors do you handle" gap.
    ULTRASONIC = "ultrasonic"       # parking / low-speed obstacle
    MICROPHONE = "microphone"       # in-cabin ASR, horn / siren detection
    THERMAL = "thermal"             # thermal / IR camera (LWIR band)
    EVENT = "event"                 # event camera (Prophesee / DVS)
    DEPTH = "depth"                 # ToF / structured-light depth sensor


# ---------------------------------------------------------------------------
# Sensor-frame dataclasses. Mirror the types the perception modules
# expect so a conversion is a one-line cast, not a parser.
# ---------------------------------------------------------------------------

@dataclass
class CameraFrame:
    """One camera image at one timestamp.

    ``data`` shape: H × W × 3 (RGB uint8), or H × W × 1 for mono.
    """
    sensor_id: str                        # "CAM_FRONT", "CAM_LEFT", etc.
    timestamp_us: int                     # epoch microseconds
    data: np.ndarray                      # H, W, C
    intrinsics: Optional[np.ndarray] = None   # 3×3 K matrix
    extrinsics: Optional[np.ndarray] = None   # 4×4 T_sensor_ego


@dataclass
class LidarFrame:
    """One lidar sweep. Point cloud in ego-vehicle frame.

    ``points`` is (N, 4) = [x, y, z, intensity]. (N, 5) if velocity known.
    """
    sensor_id: str
    timestamp_us: int
    points: np.ndarray


@dataclass
class RadarFrame:
    """One radar sweep in ego-vehicle frame.

    ``detections`` is (N, 6) = [x, y, z, vx, vy, rcs]. Connectors produce
    either detections or the raw ADC cube; both optional.
    """
    sensor_id: str
    timestamp_us: int
    detections: np.ndarray
    adc_cube: Optional[np.ndarray] = None   # (chirps, samples) complex64


@dataclass
class ImuSample:
    timestamp_us: int
    accel_mps2: Tuple[float, float, float]
    gyro_radps: Tuple[float, float, float]


@dataclass
class GnssSample:
    timestamp_us: int
    lat_deg: float
    lon_deg: float
    alt_m: float
    heading_deg: float


@dataclass
class CanMessage:
    timestamp_us: int
    dbc_name: str            # "WheelSpeed_FL", "SteeringAngle", ...
    value: float
    unit: str                # "rpm", "deg", "m/s", etc.


# ---------------------------------------------------------------------------
# Additional sensor-frame dataclasses (ultrasonic, audio, thermal, event,
# ToF depth). All added 2026-04-19 to complete the automotive sensor set.
# ---------------------------------------------------------------------------

@dataclass
class UltrasonicSample:
    """Single range reading from a parking / low-speed US sensor.

    Typical rig: 4-12 sensors around the bumpers, each sampling at
    20-50 Hz with ~0.2-3.0 m effective range.
    """
    sensor_id: str                        # e.g. "US_FRONT_LEFT"
    timestamp_us: int
    distance_m: float                     # primary echo distance (-1 = no return)
    position: str = ""                    # "front-center", "rear-left", ...
    snr_db: Optional[float] = None
    pulse_width_us: Optional[float] = None


@dataclass
class MicrophoneFrame:
    """PCM audio frame. Used for in-cabin ASR, horn detection, sirens.

    ``data`` shape: (samples,) mono or (samples, channels) multi-mic.
    """
    sensor_id: str                        # e.g. "MIC_CABIN_ARRAY"
    timestamp_us: int
    data: np.ndarray                      # int16 or float32
    sample_rate_hz: int = 16_000
    channel_layout: str = "mono"          # "mono", "stereo", "4ch-array", ...


@dataclass
class ThermalFrame:
    """Long-wave IR (thermal) camera frame.

    Separate from CameraFrame because thermal pipelines don't use
    Bayer/WB/gamma — they have their own NUC, AGC, and false-color
    mapping. ``data`` is raw thermal counts or temperature-calibrated
    kelvin values depending on the sensor.
    """
    sensor_id: str                        # e.g. "THERMAL_FRONT"
    timestamp_us: int
    data: np.ndarray                      # (H, W) uint16 raw counts, or float32 K
    units: str = "raw_counts"             # "raw_counts" | "kelvin" | "celsius"
    temperature_range_k: Optional[Tuple[float, float]] = None
    extrinsics: Optional[np.ndarray] = None


@dataclass
class EventFrame:
    """Event-camera (DVS / Prophesee) integration window.

    Events are sparse (x, y, t, polarity) rather than dense pixels. The
    ``events`` array is (N, 4) with columns [x, y, t_us_offset, polarity].
    Polarity is +1 (brightness increase) or -1 (decrease).
    """
    sensor_id: str                        # e.g. "EVT_FRONT"
    timestamp_us_start: int
    timestamp_us_end: int
    events: np.ndarray                    # (N, 4) int32
    resolution: Tuple[int, int] = (640, 480)   # (W, H)
    extrinsics: Optional[np.ndarray] = None


@dataclass
class DepthFrame:
    """Time-of-Flight or structured-light depth map.

    Dense depth (metres) per pixel. ``confidence`` optional — many ToF
    sensors emit it as a second channel.
    """
    sensor_id: str                        # e.g. "TOF_CABIN"
    timestamp_us: int
    depth_m: np.ndarray                   # (H, W) float32
    confidence: Optional[np.ndarray] = None  # (H, W) float32 in [0, 1]
    intrinsics: Optional[np.ndarray] = None
    extrinsics: Optional[np.ndarray] = None


@dataclass
class GroundTruthObject:
    """Ground-truth annotation — used to score predictions."""
    timestamp_us: int
    track_id: str
    object_class: str        # "car", "truck", "pedestrian", ...
    x_m: float
    y_m: float
    z_m: float
    length_m: float
    width_m: float
    height_m: float
    yaw_rad: float
    velocity_mps: Tuple[float, float, float] = (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Sample — one timestamp, all sensors.
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """One timestamp slice of a scene. Any subset of sensors may be present."""
    sample_id: str
    timestamp_us: int
    cameras: Dict[str, CameraFrame] = field(default_factory=dict)
    lidars: Dict[str, LidarFrame] = field(default_factory=dict)
    radars: Dict[str, RadarFrame] = field(default_factory=dict)
    imu: Optional[ImuSample] = None
    gnss: Optional[GnssSample] = None
    can: List[CanMessage] = field(default_factory=list)
    ground_truth: List[GroundTruthObject] = field(default_factory=list)
    # Extended sensor set (added 2026-04-19).
    ultrasonics: Dict[str, UltrasonicSample] = field(default_factory=dict)
    microphones: Dict[str, MicrophoneFrame] = field(default_factory=dict)
    thermals: Dict[str, ThermalFrame] = field(default_factory=dict)
    events: Dict[str, EventFrame] = field(default_factory=dict)
    depths: Dict[str, DepthFrame] = field(default_factory=dict)

    def has(self, kind: SensorKind) -> bool:
        return bool(self.sensors(kind))

    def sensors(self, kind: SensorKind) -> Dict[str, object]:
        if kind is SensorKind.CAMERA:
            return dict(self.cameras)
        if kind is SensorKind.LIDAR:
            return dict(self.lidars)
        if kind is SensorKind.RADAR:
            return dict(self.radars)
        if kind is SensorKind.IMU:
            return {"imu": self.imu} if self.imu else {}
        if kind is SensorKind.GNSS:
            return {"gnss": self.gnss} if self.gnss else {}
        if kind is SensorKind.CAN:
            return {"can": self.can} if self.can else {}
        if kind is SensorKind.ULTRASONIC:
            return dict(self.ultrasonics)
        if kind is SensorKind.MICROPHONE:
            return dict(self.microphones)
        if kind is SensorKind.THERMAL:
            return dict(self.thermals)
        if kind is SensorKind.EVENT:
            return dict(self.events)
        if kind is SensorKind.DEPTH:
            return dict(self.depths)
        return {}


# ---------------------------------------------------------------------------
# Scene + Dataset — the iterable layers.
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    """An ordered list of samples (e.g. 20 s of driving @ 2 Hz = 40 samples)."""
    scene_id: str
    name: str
    description: str = ""
    samples: Sequence[Sample] = field(default_factory=list)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)


@runtime_checkable
class Dataset(Protocol):
    """Every dataset connector implements this protocol."""

    name: str                                   # "nuscenes-mini", "kitti-raw", ...

    def list_scenes(self) -> List[str]:
        """Return scene IDs available in this dataset."""
        ...

    def get_scene(self, scene_id: str) -> Scene:
        """Load a scene by ID. Raises KeyError if not present."""
        ...

    def available_sensors(self) -> List[SensorKind]:
        """Which sensor kinds this dataset actually provides."""
        ...
