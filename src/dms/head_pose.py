"""
AstraCore Neo — Head Pose Tracker simulation.

Models the head-pose estimation pipeline of a DMS camera:
  - Accepts per-frame yaw / pitch / roll angles
  - Classifies frames as in or out of the forward-looking AttentionZone
  - Computes distraction ratio over a rolling window
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .exceptions import HeadPoseError


@dataclass
class HeadPose:
    """
    Estimated head orientation.

    Angles in degrees:
      yaw   — rotation around vertical axis;   +right, -left
      pitch — rotation around lateral axis;    +up,    -down
      roll  — rotation around forward axis;    +clockwise
    """
    yaw: float
    pitch: float
    roll: float
    timestamp_us: float = field(default=0.0)


class AttentionZone:
    """
    Defines the forward-looking angular region.

    A HeadPose is *in zone* when |yaw| <= max_yaw AND |pitch| <= max_pitch
    AND |roll| <= max_roll.
    """

    def __init__(
        self,
        max_yaw_deg: float = 30.0,
        max_pitch_deg: float = 20.0,
        max_roll_deg: float = 20.0,
    ) -> None:
        if max_yaw_deg <= 0 or max_pitch_deg <= 0 or max_roll_deg <= 0:
            raise HeadPoseError(
                f"AttentionZone thresholds must be > 0; "
                f"got yaw={max_yaw_deg}, pitch={max_pitch_deg}, roll={max_roll_deg}"
            )
        self.max_yaw = max_yaw_deg
        self.max_pitch = max_pitch_deg
        self.max_roll = max_roll_deg

    def in_zone(self, pose: HeadPose) -> bool:
        """Return True if the pose is within the forward-looking zone."""
        return (
            abs(pose.yaw) <= self.max_yaw
            and abs(pose.pitch) <= self.max_pitch
            and abs(pose.roll) <= self.max_roll
        )

    def __repr__(self) -> str:
        return (
            f"AttentionZone(yaw±{self.max_yaw}°, "
            f"pitch±{self.max_pitch}°, roll±{self.max_roll}°)"
        )


class HeadPoseTracker:
    """
    Tracks head orientation and computes distraction ratio.

    Usage::

        tracker = HeadPoseTracker()
        pose = tracker.update(yaw=5.0, pitch=-3.0)
        assert tracker.distraction_ratio == 0.0
    """

    def __init__(
        self,
        zone: Optional[AttentionZone] = None,
        window: int = 15,
    ) -> None:
        if window < 1:
            raise HeadPoseError(f"window must be >= 1, got {window}")
        self._zone = zone or AttentionZone()
        self._history: deque[bool] = deque(maxlen=window)   # True = in zone
        self._total_frames: int = 0
        self._last_pose: Optional[HeadPose] = None

    def update(self, yaw: float, pitch: float, roll: float = 0.0) -> HeadPose:
        """Process one pose estimate and update the distraction window."""
        pose = HeadPose(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            timestamp_us=time.monotonic() * 1e6,
        )
        self._history.append(self._zone.in_zone(pose))
        self._last_pose = pose
        self._total_frames += 1
        return pose

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def distraction_ratio(self) -> float:
        """Fraction of window frames where head was OUTSIDE the attention zone (0–1)."""
        if not self._history:
            return 0.0
        outside = sum(1 for v in self._history if not v)
        return outside / len(self._history)

    @property
    def last_pose(self) -> Optional[HeadPose]:
        return self._last_pose

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def attention_zone(self) -> AttentionZone:
        return self._zone

    def reset(self) -> None:
        """Reset all tracking state."""
        self._history.clear()
        self._total_frames = 0
        self._last_pose = None

    def __repr__(self) -> str:
        return (
            f"HeadPoseTracker(distraction={self.distraction_ratio:.2f}, "
            f"zone={self._zone})"
        )
