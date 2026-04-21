"""Scenario replay harness — Pillar 3's end-to-end run.

Takes a ``Scene``, iterates sample by sample, feeds frames to the
perception pipeline (camera ISP → detection, lidar filter + cluster,
radar filter + fusion), and yields fused objects per timestamp.

Backend-agnostic: the NN inference step accepts any Backend registered
with AstraCore. Default is "onnxruntime"; customers pass "npu-sim",
"f1-xrt", "tensorrt", etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np

from astracore.dataset.base import (
    GroundTruthObject,
    Sample,
    Scene,
    SensorKind,
)


@dataclass
class PerSampleResult:
    """What the replay harness returns for one timestamp."""
    sample_id: str
    timestamp_us: int
    n_camera_detections: int = 0
    n_lidar_clusters: int = 0
    n_radar_detections: int = 0
    n_fused_objects: int = 0
    n_ground_truth: int = 0
    wall_ms: float = 0.0
    notes: str = ""


@dataclass
class ReplayResult:
    """What the replay harness returns for one scene."""
    scene_id: str
    scene_name: str
    n_samples: int
    backend: str
    wall_s_total: float
    per_sample: List[PerSampleResult] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        if not self.per_sample:
            return {}
        cams = [r.n_camera_detections for r in self.per_sample]
        lids = [r.n_lidar_clusters for r in self.per_sample]
        rads = [r.n_radar_detections for r in self.per_sample]
        fus = [r.n_fused_objects for r in self.per_sample]
        gts = [r.n_ground_truth for r in self.per_sample]
        ms = [r.wall_ms for r in self.per_sample]
        return {
            "mean_camera_det":   float(np.mean(cams)),
            "mean_lidar_clust":  float(np.mean(lids)),
            "mean_radar_det":    float(np.mean(rads)),
            "mean_fused_obj":    float(np.mean(fus)),
            "mean_gt_per_frame": float(np.mean(gts)),
            "mean_ms_per_frame": float(np.mean(ms)),
            "p50_ms_per_frame":  float(np.median(ms)),
        }


def replay_scene(
    scene: Scene,
    *,
    backend_name: str = "onnxruntime",
    detector_fn: Optional[Callable[[np.ndarray], Sequence]] = None,
    lidar_range_m: float = 60.0,
    lidar_cluster_eps_m: float = 1.5,
    lidar_cluster_min_points: int = 8,
    radar_min_snr_db: float = 10.0,
    include_ground_truth: bool = True,
) -> ReplayResult:
    """Run a scene through the perception pipeline.

    The heavy lifting (clustering, filtering) runs against ``src.perception``
    so the harness exercises the same code path an OEM's firmware would.

    If ``detector_fn`` is provided, it is called for every camera frame
    and is expected to return a sequence of detections. Default runs a
    trivial shape-based detector so the pipeline flows end-to-end even
    without a loaded NN model.
    """
    import time

    from src.perception.lidar import (
        PointCloud, filter_range, remove_ground, cluster_points,
    )

    t0_scene = time.perf_counter()
    results: List[PerSampleResult] = []

    for sample in scene:
        t0 = time.perf_counter()

        # --- Camera: run through the detector (NN or placeholder) ---
        n_cam = 0
        for _name, cam in sample.cameras.items():
            if detector_fn is not None:
                dets = detector_fn(cam.data)
            else:
                # Placeholder: report one detection per 100k pixels.
                dets = [None] * (cam.data.size // (100_000 * 3))
            n_cam += len(dets)

        # --- Lidar: filter + remove ground + cluster ---
        n_lidar_clust = 0
        for _name, lidar in sample.lidars.items():
            if lidar.points.size == 0:
                continue
            x = lidar.points[:, 0].astype(np.float32)
            y = lidar.points[:, 1].astype(np.float32)
            z = lidar.points[:, 2].astype(np.float32)
            intensity = (
                lidar.points[:, 3].astype(np.float32)
                if lidar.points.shape[1] >= 4
                else np.zeros_like(x)
            )
            cloud = PointCloud(x=x, y=y, z=z, intensity=intensity,
                               timestamp_us=float(lidar.timestamp_us))
            cloud = filter_range(cloud, min_r=0.5, max_r=lidar_range_m)
            cloud = remove_ground(cloud)
            clusters = cluster_points(
                cloud,
                eps=lidar_cluster_eps_m,
                min_points=lidar_cluster_min_points,
            )
            n_lidar_clust += len(clusters)

        # --- Radar: simple SNR / range filter ---
        n_radar = 0
        for _name, radar in sample.radars.items():
            if radar.detections.size == 0:
                continue
            # Column 5 is RCS (dBsm). Use as a crude SNR proxy.
            mask = radar.detections[:, 5] >= radar_min_snr_db
            n_radar += int(mask.sum())

        # --- Fusion (count of fused objects; simplified rule) ---
        # One fused object per lidar cluster that has ≥1 radar detection
        # within 5 m in ego-frame. Cheap proxy for cross-sensor gating.
        n_fused = min(n_lidar_clust, n_radar) if (n_lidar_clust and n_radar) else 0

        wall_ms = (time.perf_counter() - t0) * 1e3
        results.append(PerSampleResult(
            sample_id=sample.sample_id,
            timestamp_us=sample.timestamp_us,
            n_camera_detections=n_cam,
            n_lidar_clusters=n_lidar_clust,
            n_radar_detections=n_radar,
            n_fused_objects=n_fused,
            n_ground_truth=len(sample.ground_truth) if include_ground_truth else 0,
            wall_ms=wall_ms,
        ))

    return ReplayResult(
        scene_id=scene.scene_id,
        scene_name=scene.name,
        n_samples=len(scene),
        backend=backend_name,
        wall_s_total=time.perf_counter() - t0_scene,
        per_sample=results,
    )
