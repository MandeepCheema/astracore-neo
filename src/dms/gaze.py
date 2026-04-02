"""
AstraCore Neo — Gaze / Eye Tracker simulation.

Models the eye-state pipeline of a DMS camera:
  - Accepts per-frame Eye Aspect Ratio (EAR) for left and right eyes
  - Classifies eye state: OPEN / PARTIAL / CLOSED
  - Computes PERCLOS (percentage of eye closure) over a rolling window
  - Counts blink events (CLOSED → OPEN transition)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

from .exceptions import GazeError


class EyeState(Enum):
    OPEN    = auto()   # avg EAR >= partial threshold
    PARTIAL = auto()   # closed threshold <= avg EAR < partial threshold
    CLOSED  = auto()   # avg EAR < closed threshold


@dataclass
class GazeReading:
    """Output of a single GazeTracker frame update."""
    left_ear: float
    right_ear: float
    avg_ear: float
    eye_state: EyeState
    gaze_yaw: float       # degrees; +right
    gaze_pitch: float     # degrees; +up
    timestamp_us: float


class GazeTracker:
    """
    Tracks Eye Aspect Ratio and computes PERCLOS.

    EAR is a normalised value in [0, 1] where 0 = fully closed, 1 = fully open.
    PERCLOS = fraction of rolling-window frames where the average EAR is below
    the closed threshold.

    Usage::

        tracker = GazeTracker(perclos_window=30)
        reading = tracker.update(left_ear=0.35, right_ear=0.33)
        assert reading.eye_state == EyeState.OPEN
        assert tracker.perclos == 0.0
    """

    def __init__(
        self,
        perclos_window: int = 30,
        ear_closed_threshold: float = 0.20,
        ear_partial_threshold: float = 0.30,
    ) -> None:
        if perclos_window < 1:
            raise GazeError(f"perclos_window must be >= 1, got {perclos_window}")
        if not (0.0 < ear_closed_threshold < ear_partial_threshold <= 1.0):
            raise GazeError(
                "EAR thresholds must satisfy 0 < ear_closed < ear_partial <= 1.0; "
                f"got closed={ear_closed_threshold}, partial={ear_partial_threshold}"
            )
        self._ear_closed = ear_closed_threshold
        self._ear_partial = ear_partial_threshold
        self._window: deque[bool] = deque(maxlen=perclos_window)   # True = frame was CLOSED
        self._blink_count: int = 0
        self._last_state: EyeState = EyeState.OPEN
        self._total_frames: int = 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        left_ear: float,
        right_ear: float,
        gaze_yaw: float = 0.0,
        gaze_pitch: float = 0.0,
    ) -> GazeReading:
        """
        Process one camera frame.

        Parameters
        ----------
        left_ear, right_ear : float
            Eye Aspect Ratio for each eye, in [0, 1].
        gaze_yaw, gaze_pitch : float
            Estimated gaze direction in degrees (informational; not used in
            PERCLOS/blink logic).
        """
        if not (0.0 <= left_ear <= 1.0):
            raise GazeError(f"left_ear must be in [0, 1], got {left_ear}")
        if not (0.0 <= right_ear <= 1.0):
            raise GazeError(f"right_ear must be in [0, 1], got {right_ear}")

        avg = (left_ear + right_ear) / 2.0

        if avg < self._ear_closed:
            state = EyeState.CLOSED
        elif avg < self._ear_partial:
            state = EyeState.PARTIAL
        else:
            state = EyeState.OPEN

        # Blink = transition from CLOSED back to OPEN
        if self._last_state == EyeState.CLOSED and state == EyeState.OPEN:
            self._blink_count += 1

        # PERCLOS only counts fully-closed frames
        self._window.append(state == EyeState.CLOSED)
        self._last_state = state
        self._total_frames += 1

        return GazeReading(
            left_ear=left_ear,
            right_ear=right_ear,
            avg_ear=avg,
            eye_state=state,
            gaze_yaw=gaze_yaw,
            gaze_pitch=gaze_pitch,
            timestamp_us=time.monotonic() * 1e6,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def perclos(self) -> float:
        """Fraction of window frames where eyes were CLOSED (0–1)."""
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    @property
    def blink_count(self) -> int:
        """Total blink events detected (CLOSED→OPEN transitions)."""
        return self._blink_count

    @property
    def last_eye_state(self) -> EyeState:
        return self._last_state

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def window_size(self) -> int:
        return self._window.maxlen  # type: ignore[return-value]

    def reset(self) -> None:
        """Reset all tracking state."""
        self._window.clear()
        self._blink_count = 0
        self._last_state = EyeState.OPEN
        self._total_frames = 0

    def __repr__(self) -> str:
        return (
            f"GazeTracker(perclos={self.perclos:.2f}, "
            f"blinks={self._blink_count}, "
            f"state={self._last_state.name})"
        )
