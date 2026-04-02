"""
AstraCore Neo Perception Subsystem.

Public API::

    from perception import CameraSensor, CameraConfig, Frame
    from perception import LidarSensor, LidarConfig, PointCloud, VoxelGrid, LidarCluster
    from perception import RadarSensor, RadarConfig, RadarDetection
    from perception import SensorFusionEngine, FusedObject, ObjectClass
    from perception import ExtrinsicCalib, IntrinsicCalib
    from perception import PerceptionError, CameraError, LidarError, RadarError, FusionError
"""

from .camera import (
    CameraSensor, CameraConfig, Frame, FrameMetadata,
    ISPPipeline, ISPStage, BayerPattern, PixelFormat,
)
from .lidar import (
    LidarSensor, LidarConfig, PointCloud, VoxelGrid, LidarCluster,
    filter_range, remove_ground, voxelize, cluster_points,
)
from .radar import (
    RadarSensor, RadarConfig, RadarDetection, RangeDopplerProcessor,
)
from .fusion import (
    SensorFusionEngine, FusedObject, ObjectClass,
    ExtrinsicCalib, IntrinsicCalib,
)
from .exceptions import (
    PerceptionError, CameraError, LidarError, RadarError,
    FusionError, CalibrationError, FrameError,
)

__all__ = [
    # Camera
    "CameraSensor", "CameraConfig", "Frame", "FrameMetadata",
    "ISPPipeline", "ISPStage", "BayerPattern", "PixelFormat",
    # Lidar
    "LidarSensor", "LidarConfig", "PointCloud", "VoxelGrid", "LidarCluster",
    "filter_range", "remove_ground", "voxelize", "cluster_points",
    # Radar
    "RadarSensor", "RadarConfig", "RadarDetection", "RangeDopplerProcessor",
    # Fusion
    "SensorFusionEngine", "FusedObject", "ObjectClass",
    "ExtrinsicCalib", "IntrinsicCalib",
    # Exceptions
    "PerceptionError", "CameraError", "LidarError", "RadarError",
    "FusionError", "CalibrationError", "FrameError",
]
