"""
AstraCore Neo — Sensor Fusion Engine.

Fuses camera, lidar, and radar detections into a unified object list:
  - Extrinsic calibration (camera←→lidar←→radar transforms)
  - Camera-Lidar projection (Lidar points onto image plane)
  - Camera-Radar association by angular proximity
  - FusedObject with position, velocity, class, confidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

from .camera import Frame
from .exceptions import CalibrationError, FusionError
from .lidar import LidarCluster, PointCloud
from .radar import RadarDetection


# ---------------------------------------------------------------------------
# Object class
# ---------------------------------------------------------------------------

class ObjectClass(Enum):
    UNKNOWN    = auto()
    VEHICLE    = auto()
    PEDESTRIAN = auto()
    CYCLIST    = auto()
    ANIMAL     = auto()
    OBSTACLE   = auto()


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass
class ExtrinsicCalib:
    """
    Rigid-body transform from sensor frame → ego-vehicle frame.

    T_vehicle_sensor  =  [R | t]  (4×4 homogeneous)
    """
    rotation: np.ndarray      # (3, 3) rotation matrix
    translation: np.ndarray   # (3,) translation vector

    def __post_init__(self) -> None:
        r = np.asarray(self.rotation, dtype=np.float64)
        t = np.asarray(self.translation, dtype=np.float64)
        if r.shape != (3, 3):
            raise CalibrationError(f"Rotation must be (3,3), got {r.shape}")
        if t.shape != (3,):
            raise CalibrationError(f"Translation must be (3,), got {t.shape}")
        self.rotation = r
        self.translation = t

    @classmethod
    def identity(cls) -> "ExtrinsicCalib":
        return cls(rotation=np.eye(3), translation=np.zeros(3))

    def transform(self, points: np.ndarray) -> np.ndarray:
        """
        Apply extrinsic transform to (N, 3) points.
        Returns (N, 3) points in target frame.
        """
        pts = np.asarray(points, dtype=np.float64)
        return (self.rotation @ pts.T).T + self.translation

    def homogeneous(self) -> np.ndarray:
        """Return (4, 4) homogeneous transformation matrix."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T


@dataclass
class IntrinsicCalib:
    """Pinhole camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        """(3, 3) camera intrinsic matrix."""
        return np.array([
            [self.fx,    0.0, self.cx],
            [   0.0, self.fy, self.cy],
            [   0.0,    0.0,    1.0],
        ], dtype=np.float64)


# ---------------------------------------------------------------------------
# FusedObject
# ---------------------------------------------------------------------------

@dataclass
class FusedObject:
    """
    A single tracked object fused from multiple sensor modalities.
    """
    object_id: int
    position_m: np.ndarray         # (3,) x,y,z in ego frame
    velocity_mps: np.ndarray       # (3,) vx,vy,vz
    object_class: ObjectClass
    confidence: float              # 0–1
    bbox_3d_min: Optional[np.ndarray] = None   # (3,) from lidar
    bbox_3d_max: Optional[np.ndarray] = None   # (3,)
    range_m: Optional[float] = None
    azimuth_deg: Optional[float] = None
    sources: list[str] = field(default_factory=list)  # e.g. ["camera", "lidar", "radar"]

    @property
    def speed_mps(self) -> float:
        return float(np.linalg.norm(self.velocity_mps))


# ---------------------------------------------------------------------------
# Sensor Fusion Engine
# ---------------------------------------------------------------------------

class SensorFusionEngine:
    """
    Fuses camera frames, lidar clusters, and radar detections.

    Calibration:
      - ``lidar_to_camera``: ExtrinsicCalib mapping lidar→camera frame
      - ``radar_to_camera``: ExtrinsicCalib mapping radar→camera frame
      - ``camera_intrinsic``: IntrinsicCalib for 2D projection

    Fusion strategy (simplified track-before-detect):
      1. Project lidar clusters into ego frame.
      2. For each lidar cluster, find closest radar detection by range.
      3. Assign object class (VEHICLE / PEDESTRIAN) based on cluster volume.
      4. Merge radar velocity into FusedObject.
      5. Camera frame is stored as context (full vision pipeline out of scope here).
    """

    def __init__(
        self,
        lidar_to_ego: Optional[ExtrinsicCalib] = None,
        radar_to_ego: Optional[ExtrinsicCalib] = None,
        camera_intrinsic: Optional[IntrinsicCalib] = None,
    ) -> None:
        self._lidar_to_ego = lidar_to_ego or ExtrinsicCalib.identity()
        self._radar_to_ego = radar_to_ego or ExtrinsicCalib.identity()
        self._camera_K = camera_intrinsic
        self._object_counter = 0

    # ------------------------------------------------------------------
    # Calibration updates
    # ------------------------------------------------------------------

    def set_lidar_extrinsic(self, calib: ExtrinsicCalib) -> None:
        self._lidar_to_ego = calib

    def set_radar_extrinsic(self, calib: ExtrinsicCalib) -> None:
        self._radar_to_ego = calib

    def set_camera_intrinsic(self, calib: IntrinsicCalib) -> None:
        self._camera_K = calib

    # ------------------------------------------------------------------
    # Camera-Lidar projection
    # ------------------------------------------------------------------

    def project_lidar_to_image(
        self,
        cloud: PointCloud,
        lidar_to_camera: ExtrinsicCalib,
        intrinsic: IntrinsicCalib,
    ) -> np.ndarray:
        """
        Project 3-D lidar points into 2-D image coordinates.

        Returns (N, 2) array of (u, v) pixel coordinates.
        Points behind the camera (z ≤ 0) are filtered out.
        """
        xyz = cloud.xyz()
        cam_pts = lidar_to_camera.transform(xyz)   # (N, 3) in camera frame

        # Keep only points in front of camera
        front = cam_pts[:, 2] > 0
        cam_pts = cam_pts[front]
        if len(cam_pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        K = intrinsic.K
        uv_h = (K @ cam_pts.T).T     # (N, 3)
        u = uv_h[:, 0] / uv_h[:, 2]
        v = uv_h[:, 1] / uv_h[:, 2]

        # Clip to image bounds
        in_bounds = (
            (u >= 0) & (u < intrinsic.width) &
            (v >= 0) & (v < intrinsic.height)
        )
        return np.stack([u[in_bounds], v[in_bounds]], axis=1).astype(np.float32)

    # ------------------------------------------------------------------
    # Classification helper
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_cluster(cluster: LidarCluster) -> tuple[ObjectClass, float]:
        """Heuristic object classification from bounding-box dimensions."""
        dims = cluster.dimensions  # L, W, H
        L, W, H = float(dims[0]), float(dims[1]), float(dims[2])
        volume = L * W * H

        if volume < 0.5:
            return ObjectClass.PEDESTRIAN, 0.60
        elif volume < 5.0 and H < 2.0:
            return ObjectClass.CYCLIST, 0.55
        elif volume < 50.0:
            return ObjectClass.VEHICLE, 0.75
        else:
            return ObjectClass.OBSTACLE, 0.50

    # ------------------------------------------------------------------
    # Main fusion entry point
    # ------------------------------------------------------------------

    def fuse(
        self,
        lidar_clusters: list[LidarCluster],
        radar_detections: list[RadarDetection],
        camera_frame: Optional[Frame] = None,
    ) -> list[FusedObject]:
        """
        Fuse lidar clusters with radar detections.

        For each lidar cluster:
          - Transform centroid to ego frame.
          - Find nearest radar detection by range (within 10 m tolerance).
          - Assign velocity from radar if matched; else zero.
          - Classify by bounding-box dimensions.

        Returns list of FusedObject.
        """
        fused: list[FusedObject] = []

        # Pre-compute radar detections in ego frame
        radar_ego: list[tuple[np.ndarray, RadarDetection]] = []
        for rd in radar_detections:
            # Radar gives (range, azimuth, elevation) → Cartesian in radar frame
            az_rad = np.deg2rad(rd.azimuth_deg)
            el_rad = np.deg2rad(rd.elevation_deg)
            pos_radar = np.array([
                rd.range_m * np.cos(el_rad) * np.cos(az_rad),
                rd.range_m * np.cos(el_rad) * np.sin(az_rad),
                rd.range_m * np.sin(el_rad),
            ])
            pos_ego = self._radar_to_ego.transform(pos_radar.reshape(1, 3))[0]
            radar_ego.append((pos_ego, rd))

        matched_radar: set[int] = set()

        for cluster in lidar_clusters:
            # Transform lidar centroid to ego frame
            pos_ego = self._lidar_to_ego.transform(cluster.centroid.reshape(1, 3))[0]
            bbox_min = self._lidar_to_ego.transform(cluster.bbox_min.reshape(1, 3))[0]
            bbox_max = self._lidar_to_ego.transform(cluster.bbox_max.reshape(1, 3))[0]

            # Find closest radar detection within 10 m
            velocity = np.zeros(3, dtype=np.float32)
            best_dist = 10.0
            best_idx = -1
            for i, (r_pos, rd) in enumerate(radar_ego):
                if i in matched_radar:
                    continue
                dist = float(np.linalg.norm(pos_ego - r_pos))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            sources = ["lidar"]
            if best_idx >= 0:
                matched_radar.add(best_idx)
                rd_matched = radar_ego[best_idx][1]
                # Radial velocity along x-axis in ego frame (simplification)
                velocity[0] = rd_matched.velocity_mps
                sources.append("radar")

            if camera_frame is not None:
                sources.append("camera")

            obj_class, conf = self._classify_cluster(cluster)
            self._object_counter += 1

            fused.append(FusedObject(
                object_id=self._object_counter,
                position_m=pos_ego.astype(np.float32),
                velocity_mps=velocity,
                object_class=obj_class,
                confidence=conf,
                bbox_3d_min=bbox_min.astype(np.float32),
                bbox_3d_max=bbox_max.astype(np.float32),
                range_m=float(np.linalg.norm(pos_ego)),
                azimuth_deg=float(np.rad2deg(np.arctan2(pos_ego[1], pos_ego[0]))),
                sources=sources,
            ))

        return fused

    def reset_counters(self) -> None:
        """Reset internal object ID counter."""
        self._object_counter = 0

    def __repr__(self) -> str:
        return (
            f"SensorFusionEngine("
            f"lidar_offset={self._lidar_to_ego.translation}, "
            f"radar_offset={self._radar_to_ego.translation})"
        )
