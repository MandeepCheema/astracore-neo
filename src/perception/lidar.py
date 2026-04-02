"""
AstraCore Neo — Lidar subsystem simulation.

Models a 4D solid-state Lidar (x, y, z, intensity + radial velocity):
  - PointCloud data structure
  - Range filtering, ground removal
  - Voxel grid downsampling
  - DBSCAN-style cluster extraction (pure numpy, no sklearn)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .exceptions import LidarError


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LidarConfig:
    """Lidar sensor configuration."""
    channels: int = 128          # vertical scan lines
    h_fov_deg: float = 120.0     # horizontal field of view
    v_fov_deg: float = 25.0      # vertical field of view
    max_range_m: float = 200.0   # maximum detection range
    min_range_m: float = 0.5
    range_resolution_m: float = 0.02
    angular_resolution_deg: float = 0.1
    rpm: float = 20.0            # rotation speed (Hz for spinning; scan rate for solid-state)
    has_velocity: bool = True    # 4D: includes radial velocity


@dataclass
class PointCloud:
    """
    4D point cloud: x, y, z (metres), intensity (0–1), velocity (m/s).

    All arrays have shape (N,).
    """
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    intensity: np.ndarray
    velocity: Optional[np.ndarray] = None    # radial velocity, m/s
    timestamp_us: float = 0.0
    frame_id: int = 0

    def __post_init__(self) -> None:
        n = len(self.x)
        for arr, name in [(self.y, "y"), (self.z, "z"), (self.intensity, "intensity")]:
            if len(arr) != n:
                raise LidarError(f"PointCloud array '{name}' length {len(arr)} != x length {n}")
        if self.velocity is not None and len(self.velocity) != n:
            raise LidarError(f"PointCloud velocity length {len(self.velocity)} != x length {n}")

    @property
    def num_points(self) -> int:
        return len(self.x)

    def xyz(self) -> np.ndarray:
        """Return (N, 3) array of [x, y, z]."""
        return np.stack([self.x, self.y, self.z], axis=1)

    def xyziv(self) -> np.ndarray:
        """Return (N, 5) array of [x, y, z, intensity, velocity] (velocity 0 if absent)."""
        v = self.velocity if self.velocity is not None else np.zeros(self.num_points, np.float32)
        return np.stack([self.x, self.y, self.z, self.intensity, v], axis=1)


@dataclass
class VoxelGrid:
    """Downsampled voxel grid representation."""
    voxels: np.ndarray          # (N_vox, 3) — voxel centroids
    voxel_size: float
    occupancy: np.ndarray       # (N_vox,) — point count per voxel
    intensity_mean: np.ndarray  # (N_vox,) — mean intensity per voxel


@dataclass
class LidarCluster:
    """A detected object cluster from lidar point cloud."""
    cluster_id: int
    points: np.ndarray          # (N, 3) xyz
    centroid: np.ndarray        # (3,) xyz
    bbox_min: np.ndarray        # (3,) axis-aligned bounding box min
    bbox_max: np.ndarray        # (3,) axis-aligned bounding box max
    point_count: int

    @property
    def dimensions(self) -> np.ndarray:
        """(3,) L×W×H of bounding box."""
        return self.bbox_max - self.bbox_min


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------

def filter_range(cloud: PointCloud, min_r: float, max_r: float) -> PointCloud:
    """Keep only points within [min_r, max_r] metres from origin."""
    r = np.sqrt(cloud.x**2 + cloud.y**2 + cloud.z**2)
    mask = (r >= min_r) & (r <= max_r)
    v = cloud.velocity[mask] if cloud.velocity is not None else None
    return PointCloud(
        x=cloud.x[mask], y=cloud.y[mask], z=cloud.z[mask],
        intensity=cloud.intensity[mask], velocity=v,
        timestamp_us=cloud.timestamp_us, frame_id=cloud.frame_id,
    )


def remove_ground(cloud: PointCloud, height_threshold_m: float = -1.5) -> PointCloud:
    """Remove points below height_threshold_m (simple height filter)."""
    mask = cloud.z > height_threshold_m
    v = cloud.velocity[mask] if cloud.velocity is not None else None
    return PointCloud(
        x=cloud.x[mask], y=cloud.y[mask], z=cloud.z[mask],
        intensity=cloud.intensity[mask], velocity=v,
        timestamp_us=cloud.timestamp_us, frame_id=cloud.frame_id,
    )


def voxelize(cloud: PointCloud, voxel_size: float = 0.2) -> VoxelGrid:
    """
    Voxel-grid downsampling.

    Each point is assigned to a voxel; voxel centroid = mean of member points.
    """
    if voxel_size <= 0:
        raise LidarError(f"voxel_size must be > 0, got {voxel_size}")
    if cloud.num_points == 0:
        return VoxelGrid(
            voxels=np.zeros((0, 3), np.float32),
            voxel_size=voxel_size,
            occupancy=np.zeros(0, np.int32),
            intensity_mean=np.zeros(0, np.float32),
        )

    xyz = cloud.xyz().astype(np.float32)
    # Compute voxel indices
    min_xyz = xyz.min(axis=0)
    indices = np.floor((xyz - min_xyz) / voxel_size).astype(np.int32)

    # Encode 3-D index into a single integer for grouping
    max_idx = indices.max(axis=0) + 1
    flat = (indices[:, 0] * max_idx[1] * max_idx[2]
            + indices[:, 1] * max_idx[2]
            + indices[:, 2])

    unique_flat, inverse = np.unique(flat, return_inverse=True)
    n_vox = len(unique_flat)

    centroids = np.zeros((n_vox, 3), np.float32)
    occupancy = np.zeros(n_vox, np.int32)
    intensity_sum = np.zeros(n_vox, np.float32)

    for i in range(len(xyz)):
        v = inverse[i]
        centroids[v] += xyz[i]
        occupancy[v] += 1
        intensity_sum[v] += cloud.intensity[i]

    counts = occupancy[:, np.newaxis].astype(np.float32)
    centroids /= counts
    intensity_mean = intensity_sum / occupancy.astype(np.float32)

    return VoxelGrid(
        voxels=centroids,
        voxel_size=voxel_size,
        occupancy=occupancy,
        intensity_mean=intensity_mean,
    )


def cluster_points(
    cloud: PointCloud,
    eps: float = 1.0,
    min_points: int = 5,
) -> list[LidarCluster]:
    """
    Simple grid-based clustering (DBSCAN-like, pure numpy).

    Groups nearby points using eps-radius neighbour search.
    Returns list of LidarCluster objects.
    """
    if cloud.num_points == 0:
        return []

    xyz = cloud.xyz().astype(np.float32)
    n = len(xyz)
    labels = np.full(n, -1, dtype=np.int32)   # -1 = unvisited
    cluster_id = 0

    def region_query(idx: int) -> np.ndarray:
        diffs = xyz - xyz[idx]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        return np.where(dists <= eps)[0]

    for i in range(n):
        if labels[i] != -1:
            continue
        neighbours = region_query(i)
        if len(neighbours) < min_points:
            labels[i] = 0   # noise
            continue
        cluster_id += 1
        labels[i] = cluster_id
        seed_set = list(neighbours)
        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if labels[q] == 0:
                labels[q] = cluster_id
            if labels[q] == -1:
                labels[q] = cluster_id
                q_neighbours = region_query(q)
                if len(q_neighbours) >= min_points:
                    seed_set.extend(q_neighbours.tolist())
            j += 1

    clusters: list[LidarCluster] = []
    for cid in range(1, cluster_id + 1):
        mask = labels == cid
        pts = xyz[mask]
        centroid = pts.mean(axis=0)
        clusters.append(LidarCluster(
            cluster_id=cid,
            points=pts,
            centroid=centroid,
            bbox_min=pts.min(axis=0),
            bbox_max=pts.max(axis=0),
            point_count=int(mask.sum()),
        ))
    return clusters


# ---------------------------------------------------------------------------
# Lidar sensor
# ---------------------------------------------------------------------------

class LidarSensor:
    """
    Simulated 4D solid-state lidar sensor.

    Usage::

        lidar = LidarSensor()
        lidar.power_on()
        cloud = lidar.scan()
        lidar.power_off()
    """

    def __init__(self, config: Optional[LidarConfig] = None) -> None:
        self._cfg = config or LidarConfig()
        self._powered = False
        self._frame_counter = 0

    def power_on(self) -> None:
        if self._powered:
            raise LidarError("Lidar already powered on")
        self._powered = True

    def power_off(self) -> None:
        if not self._powered:
            raise LidarError("Lidar already powered off")
        self._powered = False

    @property
    def is_powered(self) -> bool:
        return self._powered

    def scan(self, num_points: Optional[int] = None, seed: Optional[int] = None) -> PointCloud:
        """
        Perform one lidar scan, returning a synthetic point cloud.

        Args:
            num_points: override number of points (default: channels × ~360/resolution).
            seed: RNG seed for reproducibility.
        """
        if not self._powered:
            raise LidarError("Cannot scan: lidar is powered off")

        self._frame_counter += 1
        cfg = self._cfg
        if num_points is None:
            pts_per_channel = int(cfg.h_fov_deg / cfg.angular_resolution_deg)
            num_points = cfg.channels * pts_per_channel

        rng = np.random.default_rng(seed if seed is not None else self._frame_counter)

        # Spherical coordinates → Cartesian
        azimuth = rng.uniform(
            -cfg.h_fov_deg / 2, cfg.h_fov_deg / 2, num_points
        ) * np.pi / 180.0
        elevation = rng.uniform(
            -cfg.v_fov_deg / 2, cfg.v_fov_deg / 2, num_points
        ) * np.pi / 180.0
        r = rng.uniform(cfg.min_range_m, cfg.max_range_m, num_points)

        x = (r * np.cos(elevation) * np.cos(azimuth)).astype(np.float32)
        y = (r * np.cos(elevation) * np.sin(azimuth)).astype(np.float32)
        z = (r * np.sin(elevation)).astype(np.float32)
        intensity = rng.random(num_points).astype(np.float32)

        velocity = None
        if cfg.has_velocity:
            velocity = rng.uniform(-30.0, 30.0, num_points).astype(np.float32)

        return PointCloud(
            x=x, y=y, z=z, intensity=intensity, velocity=velocity,
            timestamp_us=float(self._frame_counter) * 1e6 / cfg.rpm,
            frame_id=self._frame_counter,
        )

    @property
    def config(self) -> LidarConfig:
        return self._cfg

    @property
    def scans_captured(self) -> int:
        return self._frame_counter

    def __repr__(self) -> str:
        return (
            f"LidarSensor({self._cfg.channels}ch, "
            f"{self._cfg.max_range_m}m, "
            f"powered={'ON' if self._powered else 'OFF'})"
        )
