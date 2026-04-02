"""
AstraCore Neo — DMS Analyzer and Monitor.

Integrates GazeTracker and HeadPoseTracker to produce:
  - DriverState (ALERT / DROWSY / DISTRACTED / MICROSLEEP / EMERGENCY)
  - AlertLevel (NONE / INFO / WARNING / CRITICAL / EMERGENCY)
  - DMSAlert dataclass with full diagnostic payload

DMSMonitor is the top-level coordinator that ties gaze + head pose processing
together into a single `process_frame()` call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from .gaze import GazeTracker, EyeState
from .head_pose import HeadPoseTracker, AttentionZone
from .exceptions import DMSAnalyzerError


# ---------------------------------------------------------------------------
# Public enums and dataclasses
# ---------------------------------------------------------------------------

class DriverState(Enum):
    ALERT       = auto()   # fully attentive
    DROWSY      = auto()   # elevated PERCLOS
    DISTRACTED  = auto()   # head outside attention zone
    MICROSLEEP  = auto()   # high PERCLOS, severe drowsiness
    EMERGENCY   = auto()   # critical PERCLOS, driver may be incapacitated


class AlertLevel(Enum):
    NONE      = auto()
    INFO      = auto()
    WARNING   = auto()
    CRITICAL  = auto()
    EMERGENCY = auto()


@dataclass
class DMSAlert:
    """Result of a single DMS evaluation frame."""
    level: AlertLevel
    state: DriverState
    perclos: float
    distraction_ratio: float
    message: str
    timestamp_us: float


# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------

_PERCLOS_DROWSY      = 0.15   # 15% closed → DROWSY / WARNING
_PERCLOS_MICROSLEEP  = 0.50   # 50% closed → MICROSLEEP / CRITICAL
_PERCLOS_EMERGENCY   = 0.80   # 80% closed → EMERGENCY

_DISTRACT_WARNING    = 0.30   # 30% out-of-zone → DISTRACTED / WARNING
_DISTRACT_CRITICAL   = 0.60   # 60% out-of-zone → DISTRACTED / CRITICAL


# ---------------------------------------------------------------------------
# DMSAnalyzer
# ---------------------------------------------------------------------------

class DMSAnalyzer:
    """
    Evaluates driver state from gaze and head pose trackers.

    Takes live references to a GazeTracker and a HeadPoseTracker; each
    `evaluate()` call snapshots their current rolling-window metrics and
    classifies the driver state.

    Usage::

        gaze = GazeTracker()
        head = HeadPoseTracker()
        analyzer = DMSAnalyzer(gaze, head)
        for _ in range(30):
            gaze.update(0.10, 0.10)   # eyes closed
            head.update(0.0, 0.0)
        alert = analyzer.evaluate()
        assert alert.state == DriverState.MICROSLEEP
    """

    def __init__(self, gaze: GazeTracker, head: HeadPoseTracker) -> None:
        self._gaze = gaze
        self._head = head

    def evaluate(self) -> DMSAlert:
        """Snapshot current tracker metrics and return a DMSAlert."""
        perclos = self._gaze.perclos
        dr = self._head.distraction_ratio
        state, level, message = self._classify(perclos, dr)
        return DMSAlert(
            level=level,
            state=state,
            perclos=perclos,
            distraction_ratio=dr,
            message=message,
            timestamp_us=time.monotonic() * 1e6,
        )

    # ------------------------------------------------------------------
    # Internal classification
    # ------------------------------------------------------------------

    def _classify(
        self, perclos: float, dr: float
    ) -> tuple[DriverState, AlertLevel, str]:
        # --- Drowsiness classification ---
        if perclos >= _PERCLOS_EMERGENCY:
            drowse_state = DriverState.EMERGENCY
            drowse_level = AlertLevel.EMERGENCY
            drowse_msg = f"EMERGENCY: PERCLOS={perclos:.0%} — driver may be incapacitated"
        elif perclos >= _PERCLOS_MICROSLEEP:
            drowse_state = DriverState.MICROSLEEP
            drowse_level = AlertLevel.CRITICAL
            drowse_msg = f"Microsleep detected: PERCLOS={perclos:.0%}"
        elif perclos >= _PERCLOS_DROWSY:
            drowse_state = DriverState.DROWSY
            drowse_level = AlertLevel.WARNING
            drowse_msg = f"Driver drowsy: PERCLOS={perclos:.0%}"
        else:
            drowse_state = DriverState.ALERT
            drowse_level = AlertLevel.NONE
            drowse_msg = ""

        # --- Distraction classification ---
        if dr >= _DISTRACT_CRITICAL:
            dist_state = DriverState.DISTRACTED
            dist_level = AlertLevel.CRITICAL
            dist_msg = f"Severe distraction: {dr:.0%} frames off-road"
        elif dr >= _DISTRACT_WARNING:
            dist_state = DriverState.DISTRACTED
            dist_level = AlertLevel.WARNING
            dist_msg = f"Driver distracted: {dr:.0%} frames off-road"
        else:
            dist_state = DriverState.ALERT
            dist_level = AlertLevel.NONE
            dist_msg = ""

        # --- Highest severity wins ---
        if drowse_level.value >= dist_level.value:
            state = drowse_state
            level = drowse_level
            message = drowse_msg
        else:
            state = dist_state
            level = dist_level
            message = dist_msg

        if level == AlertLevel.NONE:
            message = "Driver alert"

        return state, level, message


# ---------------------------------------------------------------------------
# DMSMonitor — top-level coordinator
# ---------------------------------------------------------------------------

class DMSMonitor:
    """
    Top-level Driver Monitoring System coordinator.

    A single `process_frame()` call feeds EAR and head-pose data into the
    tracker pipeline and returns a DMSAlert.  All history is retained for
    post-hoc analysis.

    Usage::

        monitor = DMSMonitor()
        alert = monitor.process_frame(
            left_ear=0.35, right_ear=0.33,
            head_yaw=5.0, head_pitch=-2.0,
        )
        assert alert.state == DriverState.ALERT
    """

    def __init__(
        self,
        perclos_window: int = 30,
        head_window: int = 15,
        ear_closed_threshold: float = 0.20,
        ear_partial_threshold: float = 0.30,
        max_yaw_deg: float = 30.0,
        max_pitch_deg: float = 20.0,
        max_roll_deg: float = 20.0,
    ) -> None:
        self._gaze = GazeTracker(
            perclos_window=perclos_window,
            ear_closed_threshold=ear_closed_threshold,
            ear_partial_threshold=ear_partial_threshold,
        )
        zone = AttentionZone(
            max_yaw_deg=max_yaw_deg,
            max_pitch_deg=max_pitch_deg,
            max_roll_deg=max_roll_deg,
        )
        self._head = HeadPoseTracker(zone=zone, window=head_window)
        self._analyzer = DMSAnalyzer(self._gaze, self._head)
        self._alert_history: list[DMSAlert] = []

    def process_frame(
        self,
        left_ear: float,
        right_ear: float,
        head_yaw: float,
        head_pitch: float,
        head_roll: float = 0.0,
    ) -> DMSAlert:
        """Process one frame of sensor data and return the current alert."""
        self._gaze.update(left_ear, right_ear)
        self._head.update(head_yaw, head_pitch, head_roll)
        alert = self._analyzer.evaluate()
        self._alert_history.append(alert)
        return alert

    def latest_alert(self) -> Optional[DMSAlert]:
        """Return the most recent alert, or None if no frames have been processed."""
        return self._alert_history[-1] if self._alert_history else None

    def alert_history(self) -> list[DMSAlert]:
        """Return a snapshot of all alerts generated so far."""
        return list(self._alert_history)

    def any_active_alert(self) -> bool:
        """True if the latest alert level is above NONE."""
        alert = self.latest_alert()
        return alert is not None and alert.level != AlertLevel.NONE

    def reset(self) -> None:
        """Reset all state: gaze tracker, head tracker, and alert history."""
        self._gaze.reset()
        self._head.reset()
        self._alert_history.clear()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def gaze_tracker(self) -> GazeTracker:
        return self._gaze

    @property
    def head_pose_tracker(self) -> HeadPoseTracker:
        return self._head

    @property
    def analyzer(self) -> DMSAnalyzer:
        return self._analyzer

    def __repr__(self) -> str:
        alert = self.latest_alert()
        level = alert.level.name if alert else "—"
        return f"DMSMonitor(frames={len(self._alert_history)}, latest_level={level})"
