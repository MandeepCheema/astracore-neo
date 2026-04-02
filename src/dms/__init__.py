"""
AstraCore Neo DMS (Driver Monitoring System) Subsystem.

Public API::

    from dms import GazeTracker, EyeState, GazeReading
    from dms import HeadPoseTracker, HeadPose, AttentionZone
    from dms import DMSAnalyzer, DMSMonitor, DriverState, AlertLevel, DMSAlert
    from dms import DMSBaseError, GazeError, HeadPoseError, DMSAnalyzerError
"""

from .gaze import (
    GazeTracker, EyeState, GazeReading,
)
from .head_pose import (
    HeadPoseTracker, HeadPose, AttentionZone,
)
from .dms_analyzer import (
    DMSAnalyzer, DMSMonitor, DriverState, AlertLevel, DMSAlert,
)
from .exceptions import (
    DMSBaseError, GazeError, HeadPoseError, DMSAnalyzerError,
)

__all__ = [
    # Gaze
    "GazeTracker", "EyeState", "GazeReading",
    # Head pose
    "HeadPoseTracker", "HeadPose", "AttentionZone",
    # Analyzer / monitor
    "DMSAnalyzer", "DMSMonitor", "DriverState", "AlertLevel", "DMSAlert",
    # Exceptions
    "DMSBaseError", "GazeError", "HeadPoseError", "DMSAnalyzerError",
]
