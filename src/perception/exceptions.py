"""Perception module exceptions."""


class PerceptionError(Exception):
    """Base exception for all perception errors."""


class CameraError(PerceptionError):
    """Camera sensor or ISP error."""


class LidarError(PerceptionError):
    """Lidar sensor or point-cloud processing error."""


class RadarError(PerceptionError):
    """Radar sensor or signal processing error."""


class FusionError(PerceptionError):
    """Sensor fusion error."""


class CalibrationError(PerceptionError):
    """Extrinsic/intrinsic calibration error."""


class FrameError(PerceptionError):
    """Frame capture or processing error."""
