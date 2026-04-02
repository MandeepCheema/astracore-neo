"""
AstraCore Neo DMS exceptions.
"""


class DMSBaseError(Exception):
    """Base exception for all DMS subsystem errors."""


class GazeError(DMSBaseError):
    """Invalid gaze/EAR input or GazeTracker mis-use."""


class HeadPoseError(DMSBaseError):
    """Invalid head pose input or HeadPoseTracker mis-use."""


class DMSAnalyzerError(DMSBaseError):
    """DMS analysis logic error."""
