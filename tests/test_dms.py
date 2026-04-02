"""
AstraCore Neo — DMS (Driver Monitoring System) testbench.

Coverage:
  - GazeTracker: EAR thresholds, PERCLOS, blink counting, reset, errors
  - HeadPoseTracker: AttentionZone in/out, distraction ratio, reset, errors
  - DMSAnalyzer: all state/level classifications
  - DMSMonitor: end-to-end frame processing, history, reset
"""

import math
import sys
import os

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from dms import (
    GazeTracker, EyeState, GazeReading,
    HeadPoseTracker, HeadPose, AttentionZone,
    DMSAnalyzer, DMSMonitor, DriverState, AlertLevel, DMSAlert,
    GazeError, HeadPoseError, DMSAnalyzerError,
)


# ===========================================================================
# GazeTracker
# ===========================================================================

class TestEyeStateClassification:

    def _tracker(self):
        return GazeTracker(perclos_window=10, ear_closed_threshold=0.20, ear_partial_threshold=0.30)

    def test_open_state(self):
        t = self._tracker()
        r = t.update(0.40, 0.38)
        assert r.eye_state == EyeState.OPEN

    def test_open_at_partial_boundary(self):
        t = self._tracker()
        r = t.update(0.30, 0.30)   # avg = 0.30 == partial threshold → OPEN (>= partial)
        assert r.eye_state == EyeState.OPEN

    def test_partial_state(self):
        t = self._tracker()
        r = t.update(0.25, 0.25)
        assert r.eye_state == EyeState.PARTIAL

    def test_partial_just_above_closed(self):
        t = self._tracker()
        r = t.update(0.21, 0.21)   # avg = 0.21 — above closed=0.20, below partial=0.30
        assert r.eye_state == EyeState.PARTIAL

    def test_closed_state(self):
        t = self._tracker()
        r = t.update(0.10, 0.10)
        assert r.eye_state == EyeState.CLOSED

    def test_closed_at_boundary(self):
        t = self._tracker()
        r = t.update(0.15, 0.15)   # avg = 0.15 < 0.20 → CLOSED
        assert r.eye_state == EyeState.CLOSED

    def test_avg_ear_is_mean_of_left_right(self):
        t = self._tracker()
        r = t.update(0.40, 0.20)
        assert abs(r.avg_ear - 0.30) < 1e-9

    def test_asymmetric_ears_closed(self):
        # avg < closed threshold even if one eye is open
        t = self._tracker()
        r = t.update(0.30, 0.05)   # avg = 0.175 → CLOSED
        assert r.eye_state == EyeState.CLOSED

    def test_reading_fields_populated(self):
        t = self._tracker()
        r = t.update(0.35, 0.33, gaze_yaw=5.0, gaze_pitch=-2.0)
        assert r.left_ear == 0.35
        assert r.right_ear == 0.33
        assert abs(r.avg_ear - 0.34) < 1e-9
        assert r.gaze_yaw == 5.0
        assert r.gaze_pitch == -2.0
        assert r.timestamp_us > 0


class TestGazePERCLOS:

    def test_perclos_zero_all_open(self):
        t = GazeTracker(perclos_window=10)
        for _ in range(10):
            t.update(0.40, 0.40)
        assert t.perclos == 0.0

    def test_perclos_one_all_closed(self):
        t = GazeTracker(perclos_window=10)
        for _ in range(10):
            t.update(0.10, 0.10)
        assert t.perclos == 1.0

    def test_perclos_half(self):
        t = GazeTracker(perclos_window=10)
        for _ in range(5):
            t.update(0.10, 0.10)   # closed
        for _ in range(5):
            t.update(0.40, 0.40)   # open
        assert abs(t.perclos - 0.5) < 1e-9

    def test_perclos_partial_does_not_count(self):
        # PARTIAL frames count as open in PERCLOS
        t = GazeTracker(perclos_window=10)
        for _ in range(10):
            t.update(0.25, 0.25)   # PARTIAL
        assert t.perclos == 0.0

    def test_perclos_window_rolls(self):
        t = GazeTracker(perclos_window=4)
        # Fill with closed frames
        for _ in range(4):
            t.update(0.10, 0.10)
        assert t.perclos == 1.0
        # Push 4 open frames — should evict all closed ones
        for _ in range(4):
            t.update(0.40, 0.40)
        assert t.perclos == 0.0

    def test_perclos_before_any_frames(self):
        t = GazeTracker(perclos_window=10)
        assert t.perclos == 0.0

    def test_perclos_partial_window(self):
        # Only 3 frames in a window of 10 — both closed
        t = GazeTracker(perclos_window=10)
        t.update(0.10, 0.10)
        t.update(0.40, 0.40)
        t.update(0.10, 0.10)
        # 2 of 3 closed
        assert abs(t.perclos - 2/3) < 1e-9


class TestGazeBlinks:

    def test_blink_count_zero_initial(self):
        t = GazeTracker()
        assert t.blink_count == 0

    def test_single_blink(self):
        t = GazeTracker()
        t.update(0.40, 0.40)   # OPEN
        t.update(0.10, 0.10)   # CLOSED
        t.update(0.40, 0.40)   # OPEN  ← blink counted here
        assert t.blink_count == 1

    def test_multiple_blinks(self):
        t = GazeTracker()
        for _ in range(3):
            t.update(0.40, 0.40)
            t.update(0.10, 0.10)
            t.update(0.40, 0.40)
        assert t.blink_count == 3

    def test_no_blink_if_no_reopen(self):
        t = GazeTracker()
        t.update(0.40, 0.40)
        t.update(0.10, 0.10)
        t.update(0.10, 0.10)
        assert t.blink_count == 0   # eyes still closed

    def test_partial_to_open_not_counted_as_blink(self):
        # Only CLOSED→OPEN transitions count
        t = GazeTracker()
        t.update(0.40, 0.40)
        t.update(0.25, 0.25)   # PARTIAL
        t.update(0.40, 0.40)
        assert t.blink_count == 0


class TestGazeTrackerMisc:

    def test_total_frames_counter(self):
        t = GazeTracker()
        for i in range(7):
            t.update(0.40, 0.40)
        assert t.total_frames == 7

    def test_last_eye_state(self):
        t = GazeTracker()
        t.update(0.10, 0.10)
        assert t.last_eye_state == EyeState.CLOSED

    def test_window_size_property(self):
        t = GazeTracker(perclos_window=20)
        assert t.window_size == 20

    def test_reset_clears_state(self):
        t = GazeTracker(perclos_window=5)
        for _ in range(5):
            t.update(0.10, 0.10)
        assert t.perclos == 1.0
        t.reset()
        assert t.perclos == 0.0
        assert t.blink_count == 0
        assert t.total_frames == 0
        assert t.last_eye_state == EyeState.OPEN

    def test_invalid_ear_left_negative(self):
        t = GazeTracker()
        with pytest.raises(GazeError):
            t.update(-0.1, 0.4)

    def test_invalid_ear_right_too_large(self):
        t = GazeTracker()
        with pytest.raises(GazeError):
            t.update(0.4, 1.1)

    def test_invalid_perclos_window(self):
        with pytest.raises(GazeError):
            GazeTracker(perclos_window=0)

    def test_invalid_ear_thresholds(self):
        with pytest.raises(GazeError):
            GazeTracker(ear_closed_threshold=0.35, ear_partial_threshold=0.20)


# ===========================================================================
# HeadPoseTracker
# ===========================================================================

class TestAttentionZone:

    def _zone(self):
        return AttentionZone(max_yaw_deg=30.0, max_pitch_deg=20.0, max_roll_deg=20.0)

    def test_forward_gaze_in_zone(self):
        z = self._zone()
        assert z.in_zone(HeadPose(0.0, 0.0, 0.0)) is True

    def test_exactly_at_yaw_boundary_in_zone(self):
        z = self._zone()
        assert z.in_zone(HeadPose(30.0, 0.0, 0.0)) is True

    def test_just_outside_yaw(self):
        z = self._zone()
        assert z.in_zone(HeadPose(30.1, 0.0, 0.0)) is False

    def test_negative_yaw_in_zone(self):
        z = self._zone()
        assert z.in_zone(HeadPose(-25.0, 0.0, 0.0)) is True

    def test_pitch_out_of_zone(self):
        z = self._zone()
        assert z.in_zone(HeadPose(0.0, 21.0, 0.0)) is False

    def test_roll_out_of_zone(self):
        z = self._zone()
        assert z.in_zone(HeadPose(0.0, 0.0, -25.0)) is False

    def test_all_axes_in_zone(self):
        z = self._zone()
        assert z.in_zone(HeadPose(20.0, -10.0, 15.0)) is True

    def test_custom_zone_thresholds(self):
        z = AttentionZone(max_yaw_deg=10.0, max_pitch_deg=10.0, max_roll_deg=10.0)
        assert z.in_zone(HeadPose(10.0, 0.0, 0.0)) is True
        assert z.in_zone(HeadPose(10.1, 0.0, 0.0)) is False

    def test_invalid_zero_threshold(self):
        with pytest.raises(HeadPoseError):
            AttentionZone(max_yaw_deg=0.0)


class TestHeadPoseTracker:

    def _tracker(self):
        return HeadPoseTracker(window=10)

    def test_initial_distraction_ratio_zero(self):
        t = self._tracker()
        assert t.distraction_ratio == 0.0

    def test_distraction_ratio_zero_all_in_zone(self):
        t = self._tracker()
        for _ in range(10):
            t.update(0.0, 0.0)
        assert t.distraction_ratio == 0.0

    def test_distraction_ratio_one_all_out(self):
        t = self._tracker()
        for _ in range(10):
            t.update(90.0, 0.0)   # way outside yaw limit
        assert t.distraction_ratio == 1.0

    def test_distraction_ratio_half(self):
        t = self._tracker()
        for _ in range(5):
            t.update(0.0, 0.0)    # in zone
        for _ in range(5):
            t.update(90.0, 0.0)   # out
        assert abs(t.distraction_ratio - 0.5) < 1e-9

    def test_distraction_window_rolls(self):
        t = HeadPoseTracker(window=4)
        for _ in range(4):
            t.update(90.0, 0.0)   # out
        assert t.distraction_ratio == 1.0
        for _ in range(4):
            t.update(0.0, 0.0)    # in zone
        assert t.distraction_ratio == 0.0

    def test_last_pose_updated(self):
        t = self._tracker()
        t.update(15.0, -5.0, 3.0)
        pose = t.last_pose
        assert pose is not None
        assert pose.yaw == 15.0
        assert pose.pitch == -5.0
        assert pose.roll == 3.0

    def test_total_frames_counter(self):
        t = self._tracker()
        for _ in range(6):
            t.update(0.0, 0.0)
        assert t.total_frames == 6

    def test_reset_clears_all(self):
        t = self._tracker()
        for _ in range(10):
            t.update(90.0, 0.0)
        t.reset()
        assert t.distraction_ratio == 0.0
        assert t.total_frames == 0
        assert t.last_pose is None

    def test_invalid_window_size(self):
        with pytest.raises(HeadPoseError):
            HeadPoseTracker(window=0)

    def test_default_attention_zone_applied(self):
        t = HeadPoseTracker(window=5)
        t.update(0.0, 0.0)   # forward — in zone
        assert t.distraction_ratio == 0.0
        t.update(90.0, 0.0)  # side look — out
        assert t.distraction_ratio > 0.0


# ===========================================================================
# DMSAnalyzer
# ===========================================================================

class TestDMSAnalyzerAlert:
    """Driver is alert — all metrics nominal."""

    def _setup(self, n_frames=10, ear=0.40, yaw=0.0, pitch=0.0):
        gaze = GazeTracker(perclos_window=n_frames)
        head = HeadPoseTracker(window=n_frames)
        for _ in range(n_frames):
            gaze.update(ear, ear)
            head.update(yaw, pitch)
        return DMSAnalyzer(gaze, head)

    def test_state_alert(self):
        ana = self._setup()
        alert = ana.evaluate()
        assert alert.state == DriverState.ALERT

    def test_level_none(self):
        ana = self._setup()
        assert ana.evaluate().level == AlertLevel.NONE

    def test_message_nominal(self):
        ana = self._setup()
        assert "alert" in ana.evaluate().message.lower()

    def test_perclos_in_alert(self):
        ana = self._setup()
        alert = ana.evaluate()
        assert alert.perclos == 0.0

    def test_distraction_in_alert(self):
        ana = self._setup()
        assert ana.evaluate().distraction_ratio == 0.0


class TestDMSAnalyzerDrowsy:
    """PERCLOS just above 15% → DROWSY / WARNING."""

    def _drowsy_analyzer(self, n=20, closed_frac=0.20):
        gaze = GazeTracker(perclos_window=n)
        head = HeadPoseTracker(window=n)
        n_closed = int(n * closed_frac)
        for i in range(n):
            ear = 0.10 if i < n_closed else 0.40
            gaze.update(ear, ear)
            head.update(0.0, 0.0)
        return DMSAnalyzer(gaze, head)

    def test_state_drowsy(self):
        ana = self._drowsy_analyzer(closed_frac=0.20)
        assert ana.evaluate().state == DriverState.DROWSY

    def test_level_warning(self):
        ana = self._drowsy_analyzer(closed_frac=0.20)
        assert ana.evaluate().level == AlertLevel.WARNING

    def test_message_contains_perclos(self):
        ana = self._drowsy_analyzer(closed_frac=0.20)
        assert "perclos" in ana.evaluate().message.lower() or "drowsy" in ana.evaluate().message.lower()


class TestDMSAnalyzerMicrosleep:
    """PERCLOS >= 50% → MICROSLEEP / CRITICAL."""

    def _microsleep_analyzer(self):
        gaze = GazeTracker(perclos_window=10)
        head = HeadPoseTracker(window=10)
        for _ in range(6):   # 60% closed
            gaze.update(0.10, 0.10)
            head.update(0.0, 0.0)
        for _ in range(4):
            gaze.update(0.40, 0.40)
            head.update(0.0, 0.0)
        return DMSAnalyzer(gaze, head)

    def test_state_microsleep(self):
        assert self._microsleep_analyzer().evaluate().state == DriverState.MICROSLEEP

    def test_level_critical(self):
        assert self._microsleep_analyzer().evaluate().level == AlertLevel.CRITICAL


class TestDMSAnalyzerEmergency:
    """PERCLOS >= 80% → EMERGENCY."""

    def _emergency_analyzer(self):
        gaze = GazeTracker(perclos_window=10)
        head = HeadPoseTracker(window=10)
        for _ in range(9):   # 90% closed
            gaze.update(0.10, 0.10)
            head.update(0.0, 0.0)
        gaze.update(0.40, 0.40)
        head.update(0.0, 0.0)
        return DMSAnalyzer(gaze, head)

    def test_state_emergency(self):
        assert self._emergency_analyzer().evaluate().state == DriverState.EMERGENCY

    def test_level_emergency(self):
        assert self._emergency_analyzer().evaluate().level == AlertLevel.EMERGENCY


class TestDMSAnalyzerDistracted:
    """Head outside zone → DISTRACTED."""

    def _distracted_analyzer(self, out_frac=0.40):
        n = 10
        gaze = GazeTracker(perclos_window=n)
        head = HeadPoseTracker(window=n)
        n_out = int(n * out_frac)
        for i in range(n):
            gaze.update(0.40, 0.40)
            yaw = 90.0 if i < n_out else 0.0
            head.update(yaw, 0.0)
        return DMSAnalyzer(gaze, head)

    def test_state_distracted_warning(self):
        ana = self._distracted_analyzer(out_frac=0.40)
        assert ana.evaluate().state == DriverState.DISTRACTED
        assert ana.evaluate().level == AlertLevel.WARNING

    def test_state_distracted_critical(self):
        ana = self._distracted_analyzer(out_frac=0.70)
        assert ana.evaluate().state == DriverState.DISTRACTED
        assert ana.evaluate().level == AlertLevel.CRITICAL

    def test_message_contains_distraction_info(self):
        ana = self._distracted_analyzer(out_frac=0.40)
        msg = ana.evaluate().message.lower()
        assert "distract" in msg or "off-road" in msg


class TestDMSAnalyzerCombined:
    """Drowsy + distracted — higher severity wins."""

    def test_emergency_beats_distraction(self):
        # 90% closed (EMERGENCY) + 40% distraction (WARNING)
        n = 10
        gaze = GazeTracker(perclos_window=n)
        head = HeadPoseTracker(window=n)
        for i in range(n):
            ear = 0.10 if i < 9 else 0.40
            yaw = 90.0 if i < 4 else 0.0
            gaze.update(ear, ear)
            head.update(yaw, 0.0)
        ana = DMSAnalyzer(gaze, head)
        assert ana.evaluate().state == DriverState.EMERGENCY
        assert ana.evaluate().level == AlertLevel.EMERGENCY

    def test_critical_distraction_beats_drowsy(self):
        # 20% closed (DROWSY/WARNING) + 70% distraction (CRITICAL)
        n = 10
        gaze = GazeTracker(perclos_window=n)
        head = HeadPoseTracker(window=n)
        for i in range(n):
            ear = 0.10 if i < 2 else 0.40
            yaw = 90.0 if i < 7 else 0.0
            gaze.update(ear, ear)
            head.update(yaw, 0.0)
        ana = DMSAnalyzer(gaze, head)
        alert = ana.evaluate()
        assert alert.level == AlertLevel.CRITICAL


# ===========================================================================
# DMSMonitor
# ===========================================================================

class TestDMSMonitorBasic:

    def test_initial_no_alert(self):
        m = DMSMonitor()
        assert m.latest_alert() is None

    def test_any_active_alert_false_before_frames(self):
        m = DMSMonitor()
        assert m.any_active_alert() is False

    def test_process_frame_returns_alert(self):
        m = DMSMonitor()
        alert = m.process_frame(0.40, 0.40, 0.0, 0.0)
        assert isinstance(alert, DMSAlert)

    def test_alert_history_grows(self):
        m = DMSMonitor(perclos_window=5, head_window=5)
        for _ in range(5):
            m.process_frame(0.40, 0.40, 0.0, 0.0)
        assert len(m.alert_history()) == 5

    def test_latest_alert_matches_last_process(self):
        m = DMSMonitor()
        alert = m.process_frame(0.40, 0.40, 0.0, 0.0)
        assert m.latest_alert() is alert

    def test_reset_clears_history(self):
        m = DMSMonitor()
        for _ in range(5):
            m.process_frame(0.40, 0.40, 0.0, 0.0)
        m.reset()
        assert m.latest_alert() is None
        assert len(m.alert_history()) == 0


class TestDMSMonitorDrowsyScenario:

    def test_drowsy_after_closed_frames(self):
        m = DMSMonitor(perclos_window=10, head_window=10)
        # 3 closed then 7 open — PERCLOS = 30% → DROWSY
        for _ in range(3):
            m.process_frame(0.10, 0.10, 0.0, 0.0)
        for _ in range(7):
            m.process_frame(0.40, 0.40, 0.0, 0.0)
        alert = m.latest_alert()
        assert alert.state == DriverState.DROWSY
        assert alert.level == AlertLevel.WARNING

    def test_any_active_alert_true_when_drowsy(self):
        m = DMSMonitor(perclos_window=10, head_window=10)
        for _ in range(3):
            m.process_frame(0.10, 0.10, 0.0, 0.0)
        for _ in range(7):
            m.process_frame(0.40, 0.40, 0.0, 0.0)
        assert m.any_active_alert() is True


class TestDMSMonitorDistractedScenario:

    def test_distracted_warning_after_side_look(self):
        m = DMSMonitor(perclos_window=10, head_window=10)
        # 4 frames looking 90° right, 6 forward → 40% out → WARNING
        for _ in range(4):
            m.process_frame(0.40, 0.40, 90.0, 0.0)
        for _ in range(6):
            m.process_frame(0.40, 0.40, 0.0, 0.0)
        alert = m.latest_alert()
        assert alert.state == DriverState.DISTRACTED
        assert alert.level == AlertLevel.WARNING

    def test_alert_after_reset_is_fresh(self):
        m = DMSMonitor(perclos_window=5, head_window=5)
        for _ in range(5):
            m.process_frame(0.10, 0.10, 0.0, 0.0)
        m.reset()
        alert = m.process_frame(0.40, 0.40, 0.0, 0.0)
        assert alert.state == DriverState.ALERT

    def test_history_snapshot_is_copy(self):
        m = DMSMonitor()
        m.process_frame(0.40, 0.40, 0.0, 0.0)
        history = m.alert_history()
        history.clear()
        assert len(m.alert_history()) == 1


class TestDMSMonitorEmergencyScenario:

    def test_emergency_after_prolonged_closure(self):
        m = DMSMonitor(perclos_window=10, head_window=10)
        for _ in range(9):
            m.process_frame(0.10, 0.10, 0.0, 0.0)
        m.process_frame(0.40, 0.40, 0.0, 0.0)
        alert = m.latest_alert()
        assert alert.state == DriverState.EMERGENCY
        assert alert.level == AlertLevel.EMERGENCY

    def test_accessors(self):
        from dms import GazeTracker, HeadPoseTracker, DMSAnalyzer
        m = DMSMonitor()
        assert isinstance(m.gaze_tracker, GazeTracker)
        assert isinstance(m.head_pose_tracker, HeadPoseTracker)
        assert isinstance(m.analyzer, DMSAnalyzer)
