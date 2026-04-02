# Module 9 — DMS (Driver Monitoring System)

## Overview

The DMS subsystem models the on-chip real-time driver monitoring pipeline for ISO 26262 ASIL-D compliance in L2+–L4 vehicles. It processes per-frame eye and head-pose measurements from an in-cabin camera and classifies the driver's attentiveness state.

Three vertically stacked components:

| Layer | Class | Input | Output |
|-------|-------|-------|--------|
| Sensor | `GazeTracker` | Per-eye EAR values | `EyeState`, PERCLOS |
| Sensor | `HeadPoseTracker` | Yaw/pitch/roll angles | Distraction ratio |
| Decision | `DMSAnalyzer` | Tracker snapshots | `DMSAlert` |
| Coordinator | `DMSMonitor` | Raw frame values | `DMSAlert` + history |

---

## GazeTracker

### Eye Aspect Ratio (EAR)

EAR is a normalised eye-openness score in [0, 1]. Values are computed upstream by a facial landmark detector (not modelled here) and passed in as `left_ear` and `right_ear`. The tracker uses the mean (`avg_ear`) for classification.

### EyeState Classification

| avg_ear | State |
|---------|-------|
| >= `ear_partial_threshold` (default 0.30) | `OPEN` |
| >= `ear_closed_threshold` (default 0.20) | `PARTIAL` |
| < `ear_closed_threshold` | `CLOSED` |

`PARTIAL` frames are intentionally treated as open for PERCLOS — only fully-`CLOSED` frames count.

### PERCLOS

Percentage of Eye Closure — fraction of rolling-window frames classified as `CLOSED`:

```
PERCLOS = (closed frames in window) / (total frames in window)
```

Window size is configurable (default 30 frames).

### Blink Detection

A blink event is a `CLOSED → OPEN` transition. `PARTIAL → OPEN` does not count. This intentionally avoids double-counting frames where eyes re-open through the partial zone.

### Public API

```python
from dms import GazeTracker, EyeState

tracker = GazeTracker(perclos_window=30)
reading = tracker.update(left_ear=0.35, right_ear=0.33)
assert reading.eye_state == EyeState.OPEN
assert tracker.perclos == 0.0
print(tracker.blink_count)
```

---

## HeadPoseTracker

### AttentionZone

Defines the forward-looking angular window. A `HeadPose` is *in zone* when:

```
|yaw| <= max_yaw_deg  AND  |pitch| <= max_pitch_deg  AND  |roll| <= max_roll_deg
```

Defaults: yaw ±30°, pitch ±20°, roll ±20°.

### Distraction Ratio

Fraction of rolling-window frames where the head pose was **outside** the attention zone:

```
distraction_ratio = (out-of-zone frames) / (frames in window)
```

Window size is configurable (default 15 frames).

### Public API

```python
from dms import HeadPoseTracker, AttentionZone

zone = AttentionZone(max_yaw_deg=25.0)
tracker = HeadPoseTracker(zone=zone, window=15)
tracker.update(yaw=20.0, pitch=-5.0)
assert tracker.distraction_ratio == 0.0
```

---

## DMSAnalyzer

### DriverState and AlertLevel

| Condition | DriverState | AlertLevel |
|-----------|-------------|------------|
| PERCLOS >= 80% | EMERGENCY | EMERGENCY |
| PERCLOS >= 50% | MICROSLEEP | CRITICAL |
| PERCLOS >= 15% | DROWSY | WARNING |
| Distraction >= 60% | DISTRACTED | CRITICAL |
| Distraction >= 30% | DISTRACTED | WARNING |
| All nominal | ALERT | NONE |

**Severity tie-breaking:** when both drowsiness and distraction are active, the classification with the higher `AlertLevel` value wins. EMERGENCY always beats CRITICAL; CRITICAL beats WARNING; same-level conditions use the drowsiness classification.

### DMSAlert Fields

```python
@dataclass
class DMSAlert:
    level: AlertLevel
    state: DriverState
    perclos: float           # snapshot of GazeTracker.perclos
    distraction_ratio: float # snapshot of HeadPoseTracker.distraction_ratio
    message: str             # human-readable description
    timestamp_us: float
```

### Public API

```python
from dms import GazeTracker, HeadPoseTracker, DMSAnalyzer, DriverState

gaze = GazeTracker(perclos_window=10)
head = HeadPoseTracker(window=10)
for _ in range(6):
    gaze.update(0.10, 0.10)   # eyes closed
    head.update(0.0, 0.0)
for _ in range(4):
    gaze.update(0.40, 0.40)
    head.update(0.0, 0.0)

analyzer = DMSAnalyzer(gaze, head)
alert = analyzer.evaluate()
assert alert.state == DriverState.MICROSLEEP
```

---

## DMSMonitor

Top-level coordinator. `process_frame()` feeds one frame through the entire pipeline in a single call.

```python
from dms import DMSMonitor, DriverState

monitor = DMSMonitor(perclos_window=30, head_window=15)

# Nominal driving
for _ in range(20):
    alert = monitor.process_frame(
        left_ear=0.38, right_ear=0.36,
        head_yaw=5.0, head_pitch=-2.0,
    )

assert alert.state == DriverState.ALERT
assert not monitor.any_active_alert()
```

### Key Methods

| Method | Description |
|--------|-------------|
| `process_frame(left_ear, right_ear, head_yaw, head_pitch, head_roll=0)` | Process one frame; returns `DMSAlert` |
| `latest_alert()` | Most recent `DMSAlert` or `None` |
| `alert_history()` | Snapshot list of all alerts |
| `any_active_alert()` | True when latest level > NONE |
| `reset()` | Clear all tracker and history state |

---

## Exception Hierarchy

```
DMSBaseError
├── GazeError        — invalid EAR value or config
├── HeadPoseError    — invalid AttentionZone config or window size
└── DMSAnalyzerError — reserved for future classifier errors
```

---

## Test Coverage (78/78)

| Category | Tests |
|----------|-------|
| EyeState classification (OPEN/PARTIAL/CLOSED) | 9 |
| PERCLOS calculation and rolling window | 7 |
| Blink counting | 5 |
| GazeTracker misc (reset, counters, errors) | 8 |
| AttentionZone in/out of zone | 9 |
| HeadPoseTracker distraction ratio + reset | 10 |
| DMSAnalyzer — ALERT state | 5 |
| DMSAnalyzer — DROWSY | 3 |
| DMSAnalyzer — MICROSLEEP | 2 |
| DMSAnalyzer — EMERGENCY | 2 |
| DMSAnalyzer — DISTRACTED (warning + critical) | 3 |
| DMSAnalyzer — combined scenarios | 2 |
| DMSMonitor — basic operation | 6 |
| DMSMonitor — drowsy scenario | 2 |
| DMSMonitor — distracted scenario | 3 |
| DMSMonitor — emergency scenario | 2 |
| **Total** | **78** |

---

## RTL Notes (for future Verilog implementation)

- **GazeTracker** → Two parallel comparators (closed threshold, partial threshold) driving a 2-bit state register. PERCLOS counter is a saturating `closed_frames` register divided by `window_size`; rolling window is a 1-bit shift register of depth `perclos_window`.
- **HeadPoseTracker** → Three absolute-value comparators against zone thresholds, AND-reduced to a single `in_zone` bit per frame. Rolling window is a 1-bit shift register; distraction ratio is `out_count / window` computed combinationally.
- **DMSAnalyzer** → Pure combinational priority encoder: PERCLOS thresholds and distraction ratio thresholds drive a 3-bit `DriverState` and `AlertLevel` output. No sequential logic needed beyond what the trackers supply.
- **DMSMonitor** → Thin FSM wrapper; in hardware this is the register-read path that assembles the alert word for the safety manager interrupt line.
