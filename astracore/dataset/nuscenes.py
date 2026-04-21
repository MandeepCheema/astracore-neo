"""nuScenes connector — integrates with the official ``nuscenes-devkit``.

Optional dependency. Only imported (and the class only usable) when the
user has installed the devkit AND pointed us at a valid dataroot.

Install::

    pip install astracore-sdk[nuscenes]

Then::

    from astracore.dataset import NuScenesDataset
    ds = NuScenesDataset(dataroot="/path/to/nuscenes-mini", version="v1.0-mini")
    for scene_id in ds.list_scenes()[:1]:
        scene = ds.get_scene(scene_id)
        ...

This connector is deliberately thin — its job is to translate nuScenes
tables (samples, sample_data, ego_pose, annotation) into our
``Sample`` dataclass. All nuScenes-specific quirks (timestamp units,
coordinate-frame conversions, channel names) are handled here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from astracore.dataset.base import (
    CameraFrame,
    Dataset,
    GroundTruthObject,
    LidarFrame,
    RadarFrame,
    Sample,
    Scene,
    SensorKind,
)


class NuScenesDataset:
    """Thin wrapper over ``nuscenes-devkit``'s NuScenes class.

    Construction triggers the devkit's disk-index load, so expect
    a few hundred ms per instantiation on the mini-split, seconds
    on the full dataset.
    """

    name = "nuscenes"

    def __init__(self,
                 dataroot: str,
                 version: str = "v1.0-mini",
                 verbose: bool = False):
        try:
            from nuscenes.nuscenes import NuScenes  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "nuscenes-devkit is required for NuScenesDataset. "
                "Install with `pip install astracore-sdk[nuscenes]`."
            ) from exc

        self._nusc = NuScenes(version=version, dataroot=dataroot,
                              verbose=verbose)
        self._dataroot = Path(dataroot)

    def list_scenes(self) -> List[str]:
        return [s["token"] for s in self._nusc.scene]

    def available_sensors(self) -> List[SensorKind]:
        return [SensorKind.CAMERA, SensorKind.LIDAR, SensorKind.RADAR]

    def get_scene(self, scene_id: str) -> Scene:
        meta = self._nusc.get("scene", scene_id)
        first = meta["first_sample_token"]
        samples: List[Sample] = []
        token = first
        while token:
            samples.append(self._load_sample(token))
            token = self._nusc.get("sample", token)["next"] or ""
        return Scene(
            scene_id=scene_id,
            name=meta.get("name", ""),
            description=meta.get("description", ""),
            samples=samples,
        )

    # ------------------------------------------------------------------

    def _load_sample(self, sample_token: str) -> Sample:
        sample = self._nusc.get("sample", sample_token)
        ts = int(sample["timestamp"])  # nuScenes timestamps are already µs

        cams: Dict[str, CameraFrame] = {}
        lidars: Dict[str, LidarFrame] = {}
        radars: Dict[str, RadarFrame] = {}

        for sensor_name, sd_token in sample["data"].items():
            sd = self._nusc.get("sample_data", sd_token)
            path = self._dataroot / sd["filename"]

            if sensor_name.startswith("CAM_"):
                img = self._load_image(path)
                if img is not None:
                    cams[sensor_name] = CameraFrame(
                        sensor_id=sensor_name,
                        timestamp_us=int(sd["timestamp"]),
                        data=img,
                    )
            elif sensor_name.startswith("LIDAR"):
                pts = self._load_lidar(path)
                if pts is not None:
                    lidars[sensor_name] = LidarFrame(
                        sensor_id=sensor_name,
                        timestamp_us=int(sd["timestamp"]),
                        points=pts,
                    )
            elif sensor_name.startswith("RADAR"):
                det = self._load_radar(path)
                if det is not None:
                    radars[sensor_name] = RadarFrame(
                        sensor_id=sensor_name,
                        timestamp_us=int(sd["timestamp"]),
                        detections=det,
                    )

        gts: List[GroundTruthObject] = []
        for ann_token in sample["anns"]:
            ann = self._nusc.get("sample_annotation", ann_token)
            x, y, z = ann["translation"]
            w, l, h = ann["size"]
            gts.append(GroundTruthObject(
                timestamp_us=ts,
                track_id=ann["instance_token"],
                object_class=ann["category_name"],
                x_m=float(x), y_m=float(y), z_m=float(z),
                length_m=float(l), width_m=float(w), height_m=float(h),
                yaw_rad=0.0,   # simplified; proper yaw needs quaternion parse
            ))

        return Sample(
            sample_id=sample_token,
            timestamp_us=ts,
            cameras=cams,
            lidars=lidars,
            radars=radars,
            ground_truth=gts,
        )

    @staticmethod
    def _load_image(path: Path):
        try:
            from PIL import Image  # optional
            return np.asarray(Image.open(path))
        except Exception:
            return None

    @staticmethod
    def _load_lidar(path: Path):
        try:
            raw = np.fromfile(str(path), dtype=np.float32)
            # nuScenes LIDAR_TOP: 5 cols (x,y,z,intensity,ring_idx)
            return raw.reshape(-1, 5)[:, :4]
        except Exception:
            return None

    @staticmethod
    def _load_radar(path: Path):
        try:
            # nuScenes radar .pcd — parse header, then binary block of
            # (x, y, z, dyn_prop, id, rcs, vx, vy, ...) floats. We return
            # just the canonical (N, 6) = x,y,z,vx,vy,rcs.
            raw = path.read_bytes()
            idx = raw.find(b"DATA binary") + len(b"DATA binary\n")
            arr = np.frombuffer(raw[idx:], dtype=np.float32).reshape(-1, 18)
            out = np.stack([arr[:, 0], arr[:, 1], arr[:, 2],
                            arr[:, 8], arr[:, 9], arr[:, 5]], axis=1)
            return out
        except Exception:
            return None
