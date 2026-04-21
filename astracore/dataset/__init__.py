"""Dataset connectors — sensor I/O integration (Pillar 3).

External-facing contract that normalises third-party datasets
(nuScenes, KITTI, Waymo, ROS bags, OEM-custom formats) into a single
internal format the AstraCore perception + fusion modules consume.

Public API::

    from astracore.dataset import Dataset, Scene, Sample, SensorKind
    from astracore.dataset import SyntheticDataset, NuScenesDataset
    from astracore.dataset import replay_scene

OEMs register their own connector via the ``astracore.datasets``
entry-point (see ``pyproject.toml``), or in-process via
``@register_dataset("my-format")``.
"""

from astracore.dataset.base import (
    Dataset,
    Scene,
    Sample,
    SensorKind,
    CameraFrame,
    LidarFrame,
    RadarFrame,
    ImuSample,
    GnssSample,
    CanMessage,
    GroundTruthObject,
    UltrasonicSample,
    MicrophoneFrame,
    ThermalFrame,
    EventFrame,
    DepthFrame,
)
from astracore.dataset.synthetic import SyntheticDataset, PRESETS, preset
from astracore.dataset.replay import replay_scene, ReplayResult

__all__ = [
    "Dataset",
    "Scene",
    "Sample",
    "SensorKind",
    "CameraFrame",
    "LidarFrame",
    "RadarFrame",
    "ImuSample",
    "GnssSample",
    "CanMessage",
    "GroundTruthObject",
    "UltrasonicSample",
    "MicrophoneFrame",
    "ThermalFrame",
    "EventFrame",
    "DepthFrame",
    "SyntheticDataset",
    "PRESETS",
    "preset",
    "replay_scene",
    "ReplayResult",
]

# Optional: nuScenes support, only import if devkit is available.
try:
    from astracore.dataset.nuscenes import NuScenesDataset  # noqa: F401
    __all__.append("NuScenesDataset")
except Exception:
    # nuscenes-devkit not installed — connector still discoverable by name
    # but will raise on instantiation. Intentional lazy failure.
    pass
