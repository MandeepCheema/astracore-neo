"""Deterministic synthetic dataset — no downloads, always works.

Purpose: give tests and CI a reproducible Sample / Scene / Dataset
without any filesystem dependency. Also useful as a reference for
connector authors — every field is populated with a plausible value.

The synthesis is seedable, so the same ``seed`` always produces the
exact same scene.

Presets
-------
Use the :data:`PRESETS` dict (or :func:`preset`) to match the scale an
OEM's hardware target ingests:

* ``"tiny"`` — fast CI default (2 scenes × 10 samples, 512 lidar points)
* ``"standard"`` — realistic ADAS demo (3 scenes × 20 samples, 4 k lidar)
* ``"vlp32"`` — Velodyne VLP-32 scale (5 scenes × 80 samples, 32 k lidar,
  1920×1080 camera) — stresses the downstream clusterer
* ``"vlp64"`` — VLP-64 / HDL-64 / premium lidar (5 scenes × 100 samples,
  130 k lidar, 4K camera)
* ``"robotaxi"`` — maximum (7 scenes × 150 samples, 130 k lidar, 4K,
  8 cameras, 6 radars) for long-running soak tests
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from astracore.dataset.base import (
    CameraFrame,
    CanMessage,
    Dataset,
    DepthFrame,
    EventFrame,
    GnssSample,
    GroundTruthObject,
    ImuSample,
    LidarFrame,
    MicrophoneFrame,
    RadarFrame,
    Sample,
    Scene,
    SensorKind,
    ThermalFrame,
    UltrasonicSample,
)


_DEFAULT_CLASSES = ("car", "truck", "pedestrian", "cyclist")


# ---------------------------------------------------------------------------
# Presets — named configurations that match common hardware scales.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SyntheticPreset:
    n_scenes: int
    samples_per_scene: int
    dt_us: int
    image_shape: Tuple[int, int]
    n_lidar_points: int
    n_radar_detections: int
    n_cameras: int = 1
    n_radars: int = 1
    # Extended sensor rig (added 2026-04-19). Default 0 = not included.
    n_ultrasonics: int = 0
    n_microphones: int = 0
    n_thermals: int = 0
    n_events: int = 0
    n_depths: int = 0
    audio_samples: int = 1024
    event_rate_per_frame: int = 512
    depth_shape: Tuple[int, int] = (120, 160)
    thermal_shape: Tuple[int, int] = (240, 320)


PRESETS: Dict[str, SyntheticPreset] = {
    "tiny": SyntheticPreset(
        n_scenes=2, samples_per_scene=10, dt_us=500_000,
        image_shape=(120, 160), n_lidar_points=512,
        n_radar_detections=8, n_cameras=1, n_radars=1,
    ),
    "standard": SyntheticPreset(
        n_scenes=3, samples_per_scene=20, dt_us=500_000,
        image_shape=(480, 640), n_lidar_points=4_096,
        n_radar_detections=32, n_cameras=1, n_radars=1,
    ),
    "vlp32": SyntheticPreset(
        n_scenes=5, samples_per_scene=80, dt_us=100_000,   # 10 Hz
        image_shape=(1080, 1920), n_lidar_points=32_000,
        n_radar_detections=128, n_cameras=4, n_radars=4,
    ),
    "vlp64": SyntheticPreset(
        n_scenes=5, samples_per_scene=100, dt_us=100_000,
        image_shape=(2160, 3840), n_lidar_points=130_000,
        n_radar_detections=256, n_cameras=4, n_radars=4,
    ),
    "robotaxi": SyntheticPreset(
        n_scenes=7, samples_per_scene=150, dt_us=100_000,
        image_shape=(2160, 3840), n_lidar_points=130_000,
        n_radar_detections=512, n_cameras=8, n_radars=6,
        # Full robotaxi rig: 12 ultrasonics (6 front, 6 rear), 4-mic cabin
        # array, 2 thermal cameras, 2 event cameras, 1 in-cabin ToF.
        n_ultrasonics=12, n_microphones=1, n_thermals=2,
        n_events=2, n_depths=1,
    ),
    # Dedicated preset for exercising the extended sensor set without
    # the full robotaxi resolution blow-up.
    "extended-sensors": SyntheticPreset(
        n_scenes=2, samples_per_scene=10, dt_us=500_000,
        image_shape=(240, 320), n_lidar_points=1_024,
        n_radar_detections=16, n_cameras=2, n_radars=2,
        n_ultrasonics=8, n_microphones=1, n_thermals=1,
        n_events=1, n_depths=1,
    ),
}


def preset(name: str = "standard") -> SyntheticPreset:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; "
                       f"known: {sorted(PRESETS)}")
    return PRESETS[name]


class SyntheticDataset:
    """Fake dataset that yields realistic-shaped sensor tensors.

    Defaults: 3 scenes × 20 samples @ 2 Hz. One camera (640×480), one
    lidar sweep (4096 points), one radar (32 detections), IMU, GNSS,
    CAN speed/steer messages, plus 2–4 ground-truth objects per sample.
    """

    name = "synthetic"

    # Sentinel so we can distinguish "caller passed None" from "caller
    # didn't pass anything at all". Lets a preset supply defaults that
    # explicit kwargs still override.
    _UNSET = object()

    def __init__(self,
                 n_scenes=_UNSET,
                 samples_per_scene=_UNSET,
                 dt_us=_UNSET,
                 image_shape=_UNSET,
                 n_lidar_points=_UNSET,
                 n_radar_detections=_UNSET,
                 n_cameras=_UNSET,
                 n_radars=_UNSET,
                 n_ultrasonics=_UNSET,
                 n_microphones=_UNSET,
                 n_thermals=_UNSET,
                 n_events=_UNSET,
                 n_depths=_UNSET,
                 audio_samples=_UNSET,
                 event_rate_per_frame=_UNSET,
                 depth_shape=_UNSET,
                 thermal_shape=_UNSET,
                 seed: int = 0,
                 preset_name: Optional[str] = None):

        base = preset(preset_name) if preset_name else None

        def _pick(val, base_attr: str, default):
            if val is not self._UNSET:
                return val
            if base is not None:
                return getattr(base, base_attr)
            return default

        n_scenes         = _pick(n_scenes,          "n_scenes",          3)
        samples_per_scene = _pick(samples_per_scene, "samples_per_scene", 20)
        dt_us            = _pick(dt_us,             "dt_us",             500_000)
        image_shape      = _pick(image_shape,       "image_shape",       (480, 640))
        n_lidar_points   = _pick(n_lidar_points,    "n_lidar_points",    4096)
        n_radar_detections = _pick(n_radar_detections, "n_radar_detections", 32)
        n_cameras        = _pick(n_cameras,         "n_cameras",         1)
        n_radars         = _pick(n_radars,          "n_radars",          1)
        n_ultrasonics    = _pick(n_ultrasonics,     "n_ultrasonics",     0)
        n_microphones    = _pick(n_microphones,     "n_microphones",     0)
        n_thermals       = _pick(n_thermals,        "n_thermals",        0)
        n_events         = _pick(n_events,          "n_events",          0)
        n_depths         = _pick(n_depths,          "n_depths",          0)
        audio_samples    = _pick(audio_samples,     "audio_samples",     1024)
        event_rate_per_frame = _pick(event_rate_per_frame,
                                     "event_rate_per_frame", 512)
        depth_shape      = _pick(depth_shape,       "depth_shape",       (120, 160))
        thermal_shape    = _pick(thermal_shape,     "thermal_shape",     (240, 320))

        self._n_scenes = n_scenes
        self._samples_per_scene = samples_per_scene
        self._dt_us = dt_us
        self._image_shape = image_shape
        self._n_lidar = n_lidar_points
        self._n_radar = n_radar_detections
        self._n_cameras = n_cameras
        self._n_radars = n_radars
        self._n_ultrasonics = n_ultrasonics
        self._n_microphones = n_microphones
        self._n_thermals = n_thermals
        self._n_events = n_events
        self._n_depths = n_depths
        self._audio_samples = audio_samples
        self._event_rate = event_rate_per_frame
        self._depth_shape = depth_shape
        self._thermal_shape = thermal_shape
        self._seed = seed
        self._preset_name = preset_name or "custom"

    @property
    def config(self) -> Dict:
        return {
            "preset": self._preset_name,
            "n_scenes": self._n_scenes,
            "samples_per_scene": self._samples_per_scene,
            "dt_us": self._dt_us,
            "image_shape": self._image_shape,
            "n_lidar_points": self._n_lidar,
            "n_radar_detections": self._n_radar,
            "n_cameras": self._n_cameras,
            "n_radars": self._n_radars,
            "n_ultrasonics": self._n_ultrasonics,
            "n_microphones": self._n_microphones,
            "n_thermals": self._n_thermals,
            "n_events": self._n_events,
            "n_depths": self._n_depths,
        }

    # ---- Dataset protocol ----------------------------------------------

    def list_scenes(self) -> List[str]:
        return [f"synthetic-scene-{i:03d}" for i in range(self._n_scenes)]

    def available_sensors(self) -> List[SensorKind]:
        kinds = [
            SensorKind.CAMERA, SensorKind.LIDAR, SensorKind.RADAR,
            SensorKind.IMU, SensorKind.GNSS, SensorKind.CAN,
        ]
        if self._n_ultrasonics > 0: kinds.append(SensorKind.ULTRASONIC)
        if self._n_microphones > 0: kinds.append(SensorKind.MICROPHONE)
        if self._n_thermals > 0:    kinds.append(SensorKind.THERMAL)
        if self._n_events > 0:      kinds.append(SensorKind.EVENT)
        if self._n_depths > 0:      kinds.append(SensorKind.DEPTH)
        return kinds

    def get_scene(self, scene_id: str) -> Scene:
        if scene_id not in self.list_scenes():
            raise KeyError(f"unknown scene {scene_id!r}")
        idx = int(scene_id.rsplit("-", 1)[1])
        rng = np.random.default_rng(self._seed * 1_000_003 + idx)
        samples: List[Sample] = []
        t0 = 1_700_000_000 * 1_000_000 + idx * 60 * 1_000_000
        for k in range(self._samples_per_scene):
            ts = t0 + k * self._dt_us
            samples.append(self._make_sample(idx, k, ts, rng))
        return Scene(
            scene_id=scene_id,
            name=f"Synthetic drive #{idx}",
            description=(f"Deterministic synthetic scene (seed={self._seed}, "
                         f"idx={idx}); {self._samples_per_scene} samples "
                         f"@ {1e6/self._dt_us:.1f} Hz"),
            samples=samples,
        )

    # ---- Sample factory -------------------------------------------------

    # Default multi-camera rig: front / left / right / rear, expandable
    # to 8 cameras (adds 4 diagonal viewpoints) for robotaxi-class stacks.
    _CAMERA_NAMES = (
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK",
        "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_WIDE", "CAM_REAR_WIDE",
    )
    _RADAR_NAMES = (
        "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
        "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT", "RADAR_REAR",
    )
    _US_POSITIONS = (
        "front-left-corner",  "front-left",  "front-center",
        "front-right",        "front-right-corner",
        "rear-left-corner",   "rear-left",   "rear-center",
        "rear-right",         "rear-right-corner",
        "side-left",          "side-right",
    )
    _THERMAL_NAMES = ("THERMAL_FRONT", "THERMAL_REAR")
    _EVENT_NAMES = ("EVT_FRONT", "EVT_REAR")
    _MIC_NAMES = ("MIC_CABIN_ARRAY", "MIC_DRIVER", "MIC_PASSENGER")
    _DEPTH_NAMES = ("TOF_CABIN", "TOF_DRIVER_MONITOR")

    def _make_sample(self, scene_idx: int, sample_idx: int,
                     ts: int, rng: np.random.Generator) -> Sample:
        H, W = self._image_shape

        # One CameraFrame per configured camera. Share intrinsics; offset
        # extrinsics symbolically so downstream code doesn't dedupe.
        cameras: Dict[str, CameraFrame] = {}
        n_cam = min(self._n_cameras, len(self._CAMERA_NAMES))
        for cam_i in range(n_cam):
            name = self._CAMERA_NAMES[cam_i]
            ext = np.eye(4, dtype=np.float64)
            # Offset by 0.5m per sensor to give them distinct poses.
            ext[0, 3] = 0.5 * cam_i
            cameras[name] = CameraFrame(
                sensor_id=name,
                timestamp_us=ts,
                data=rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8),
                intrinsics=np.array([[800, 0, W / 2],
                                     [0, 800, H / 2],
                                     [0, 0, 1]], dtype=np.float64),
                extrinsics=ext,
            )

        # Lidar: background noise + ``n_gt`` dense object clusters so that
        # the downstream clusterer has something structured to find.
        n_bg = max(self._n_lidar // 2, 1024)
        bg_xyz = rng.uniform(low=[-20, -10, -2],
                             high=[80, 10, 3],
                             size=(n_bg, 3)).astype(np.float32)
        bg_int = rng.uniform(0.0, 0.5, size=(n_bg, 1)).astype(np.float32)

        n_obj = int(rng.integers(3, 5))
        obj_points: list = []
        for _ in range(n_obj):
            cx = float(rng.uniform(10, 60))
            cy = float(rng.uniform(-4, 4))
            cz = 0.5
            pts = rng.normal(loc=[cx, cy, cz], scale=[0.6, 0.6, 0.3],
                             size=(256, 3)).astype(np.float32)
            obj_points.append(pts)
        obj_xyz = (np.concatenate(obj_points, axis=0)
                   if obj_points else np.empty((0, 3), dtype=np.float32))
        obj_int = np.full((len(obj_xyz), 1), 0.9, dtype=np.float32)

        xyz = np.concatenate([bg_xyz, obj_xyz], axis=0)
        intensity = np.concatenate([bg_int, obj_int], axis=0)
        lidar = LidarFrame(
            sensor_id="LIDAR_TOP",
            timestamp_us=ts,
            points=np.concatenate([xyz, intensity], axis=1),
        )

        # Radars: one RadarFrame per configured radar sensor.
        radars: Dict[str, RadarFrame] = {}
        n_rad = min(self._n_radars, len(self._RADAR_NAMES))
        for rad_i in range(n_rad):
            rd = rng.uniform(low=[-10, -10, -1, -30, -5, 0.1],
                             high=[80, 10, 2, 30, 5, 30.0],
                             size=(self._n_radar, 6)).astype(np.float32)
            radars[self._RADAR_NAMES[rad_i]] = RadarFrame(
                sensor_id=self._RADAR_NAMES[rad_i],
                timestamp_us=ts,
                detections=rd,
            )

        # IMU @ 100 Hz — just sample at this sample boundary for simplicity.
        imu = ImuSample(
            timestamp_us=ts,
            accel_mps2=(float(rng.normal(0, 0.3)),
                        float(rng.normal(0, 0.3)),
                        float(9.81 + rng.normal(0, 0.05))),
            gyro_radps=(float(rng.normal(0, 0.01)),
                        float(rng.normal(0, 0.01)),
                        float(rng.normal(0, 0.05))),
        )

        # GNSS — ego near a canonical lat/lon.
        gnss = GnssSample(
            timestamp_us=ts,
            lat_deg=12.97 + 0.0001 * sample_idx,
            lon_deg=77.59 + 0.0001 * sample_idx,
            alt_m=920.0,
            heading_deg=float((sample_idx * 3) % 360),
        )

        # CAN — speed + steering.
        speed = 50.0 + float(rng.normal(0, 2))
        steer = float(rng.normal(0, 5))
        can = [
            CanMessage(timestamp_us=ts, dbc_name="VehicleSpeed",
                       value=speed, unit="kph"),
            CanMessage(timestamp_us=ts, dbc_name="SteeringAngle",
                       value=steer, unit="deg"),
        ]

        # 2–4 ground-truth objects.
        n_gt = int(rng.integers(2, 5))
        gts: List[GroundTruthObject] = []
        for i in range(n_gt):
            x = float(rng.uniform(5, 60))
            y = float(rng.uniform(-5, 5))
            gts.append(GroundTruthObject(
                timestamp_us=ts,
                track_id=f"s{scene_idx:02d}_t{i}",
                object_class=_DEFAULT_CLASSES[i % len(_DEFAULT_CLASSES)],
                x_m=x, y_m=y, z_m=0.5,
                length_m=4.5, width_m=1.8, height_m=1.5,
                yaw_rad=0.0,
                velocity_mps=(float(rng.normal(12, 3)), 0.0, 0.0),
            ))

        # Extended sensors -------------------------------------------------
        ultrasonics = {}
        for us_i in range(min(self._n_ultrasonics, len(self._US_POSITIONS))):
            pos = self._US_POSITIONS[us_i]
            ultrasonics[f"US_{pos.upper().replace('-', '_')}"] = UltrasonicSample(
                sensor_id=f"US_{pos.upper().replace('-', '_')}",
                timestamp_us=ts,
                distance_m=float(rng.uniform(0.2, 3.0)),
                position=pos,
                snr_db=float(rng.uniform(10, 30)),
                pulse_width_us=float(rng.uniform(150, 250)),
            )

        microphones = {}
        for m_i in range(min(self._n_microphones, len(self._MIC_NAMES))):
            name = self._MIC_NAMES[m_i]
            # Audio as int16 PCM — typical in-cabin format
            audio = rng.integers(-2000, 2000, size=self._audio_samples,
                                 dtype=np.int16)
            microphones[name] = MicrophoneFrame(
                sensor_id=name, timestamp_us=ts,
                data=audio, sample_rate_hz=16_000, channel_layout="mono",
            )

        thermals = {}
        for t_i in range(min(self._n_thermals, len(self._THERMAL_NAMES))):
            name = self._THERMAL_NAMES[t_i]
            # Thermal imagers: uint16 raw counts, typical range 0..65535
            th = rng.integers(20_000, 40_000,
                              size=self._thermal_shape, dtype=np.uint16)
            thermals[name] = ThermalFrame(
                sensor_id=name, timestamp_us=ts,
                data=th, units="raw_counts",
                temperature_range_k=(280.0, 320.0),
            )

        event_frames = {}
        for e_i in range(min(self._n_events, len(self._EVENT_NAMES))):
            name = self._EVENT_NAMES[e_i]
            W, H = 640, 480
            # (N, 4) int32: [x, y, t_us_offset, polarity]
            n = self._event_rate
            evts = np.stack([
                rng.integers(0, W, size=n, dtype=np.int32),
                rng.integers(0, H, size=n, dtype=np.int32),
                rng.integers(0, int(self._dt_us), size=n, dtype=np.int32),
                rng.choice(np.array([-1, 1], dtype=np.int32), size=n),
            ], axis=1)
            event_frames[name] = EventFrame(
                sensor_id=name,
                timestamp_us_start=ts,
                timestamp_us_end=ts + self._dt_us,
                events=evts, resolution=(W, H),
            )

        depths = {}
        for d_i in range(min(self._n_depths, len(self._DEPTH_NAMES))):
            name = self._DEPTH_NAMES[d_i]
            dh, dw = self._depth_shape
            depth = rng.uniform(0.2, 2.0,
                                size=(dh, dw)).astype(np.float32)
            conf = rng.uniform(0.5, 1.0, size=(dh, dw)).astype(np.float32)
            depths[name] = DepthFrame(
                sensor_id=name, timestamp_us=ts,
                depth_m=depth, confidence=conf,
            )

        return Sample(
            sample_id=f"synthetic-s{scene_idx:03d}-n{sample_idx:03d}",
            timestamp_us=ts,
            cameras=cameras,
            lidars={lidar.sensor_id: lidar},
            radars=radars,
            imu=imu,
            gnss=gnss,
            can=can,
            ground_truth=gts,
            ultrasonics=ultrasonics,
            microphones=microphones,
            thermals=thermals,
            events=event_frames,
            depths=depths,
        )
