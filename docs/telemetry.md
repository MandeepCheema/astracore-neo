# Module 8 — Telemetry

## Overview

The Telemetry subsystem provides real-time observability for the AstraCore Neo chip: structured logging, per-zone thermal management, and predictive fault detection. It sits above HAL and Safety (its declared dependencies) and feeds diagnostic data to any higher-level subsystem.

Three orthogonal components:

| Component | Class | Purpose |
|-----------|-------|---------|
| Logger | `TelemetryLogger` | Structured ring-buffer log with per-level counters |
| Thermal | `ThermalMonitor` / `ThermalZone` | Multi-zone thermal state machine with throttling |
| Fault predictor | `FaultPredictor` / `MetricTracker` | Rolling-window anomaly detection and risk scoring |

---

## TelemetryLogger

### Design

A fixed-capacity ring buffer (Python `deque(maxlen=N)`) that stores `LogEntry` dataclasses. When the buffer is full, the oldest entry is silently dropped and the `dropped` counter is incremented.

```
LogLevel (IntEnum): DEBUG=0, INFO=1, WARNING=2, ERROR=3, CRITICAL=4
```

Because `LogLevel` is an `IntEnum`, threshold comparison is a simple integer comparison — no special-casing required.

### Key Behaviours

- `log(level, message, source)` — records entry only if `level >= min_level`; if at capacity the deque auto-evicts oldest and `dropped` increments
- `filter_by_level(level)` — returns all entries at or above the given level
- `filter_by_source(source)` — returns all entries from a named source component
- `latest(n)` — the n most-recent entries (tail of deque)
- `counts` — dict of `{LogLevel: int}` across all entries currently in the buffer (not historical)
- `clear()` — empties buffer; counters reset but `dropped` accumulates across clears

### Public API

```python
from telemetry import TelemetryLogger, LogLevel, LogEntry

logger = TelemetryLogger(capacity=100, min_level=LogLevel.INFO)
logger.log(LogLevel.WARNING, "Voltage spike on rail 1V2", source="pmic")
entries = logger.filter_by_level(LogLevel.WARNING)
```

---

## ThermalMonitor / ThermalZone

### Design

Each `ThermalZone` runs an independent 5-state machine against a `ThermalZoneConfig` threshold set. `ThermalMonitor` is a named-zone registry that routes readings and provides aggregate queries.

### State Machine

```
NOMINAL   temp < warning_c
WARNING   warning_c ≤ temp < throttle_c
THROTTLED throttle_c ≤ temp < critical_c   ← clock reduced by throttle_pct
CRITICAL  critical_c ≤ temp < shutdown_c
SHUTDOWN  temp ≥ shutdown_c                ← ThermalShutdownError raised
```

Default thresholds: warning=75 °C, throttle=85 °C, critical=95 °C, shutdown=105 °C.

### Clock Throttling

When `THROTTLED`, `effective_clock_pct()` returns `100 - throttle_pct` (default 50%). In `CRITICAL` or `SHUTDOWN` states it returns 0 %.

### Slope Tracking

`temperature_slope()` runs linear regression over the 16-sample rolling history window. Returns °C per sample — positive means rising.

### Shutdown Behaviour

`ThermalZone.update()` raises `ThermalShutdownError` before returning a reading whenever `temp >= shutdown_c`. The caller (ThermalMonitor) propagates this exception; the chip supervisor is expected to catch it and initiate emergency shutdown.

### Public API

```python
from telemetry import ThermalMonitor, ThermalZoneConfig, ThermalState

monitor = ThermalMonitor()
monitor.add_zone("npu", ThermalZoneConfig(shutdown_c=110.0))
reading = monitor.update_zone("npu", 88.0)
assert reading.state == ThermalState.THROTTLED
assert monitor.any_throttled()
print(monitor.summary())
```

---

## FaultPredictor / MetricTracker

### Design

`MetricTracker` maintains a fixed-length rolling window of float samples for one metric and produces a `FaultPrediction` on every `push()`. `FaultPredictor` manages a named dictionary of trackers and provides cross-metric risk aggregation.

### Risk Levels

```
NONE     nominal
LOW      above warning threshold (< 30% of warning→critical band)
MEDIUM   above warning (30–70%) or spike detected
HIGH     above warning (> 70%) or trend will reach critical within window
CRITICAL value >= critical_threshold
```

### Detection Rules (applied in priority order)

1. **Threshold breach** — direct comparison to `warning_threshold` and `critical_threshold`
2. **Spike detection** — `z_score = (value - mean) / std > spike_std_multiplier` (default 3×) with ≥4 window samples. Escalates to MEDIUM if currently below HIGH.
3. **Rising trend** — linear slope projected: `samples_to_critical = (critical - value) / slope`. If 0 < eta < `window_size` with ≥ `window_size/2` samples, escalates to HIGH.

### Rolling Statistics

`_stats()` returns `(mean, std_dev)` using population variance. `_slope()` uses the same least-squares linear regression as `ThermalZone.temperature_slope()`.

### Public API

```python
from telemetry import FaultPredictor, MetricConfig, FaultRisk

predictor = FaultPredictor()
predictor.add_metric(MetricConfig("ecc_rate", warning_threshold=5.0, critical_threshold=20.0))
prediction = predictor.push("ecc_rate", 8.0)
assert prediction.risk == FaultRisk.LOW
assert not predictor.any_high_risk()
```

---

## Exception Hierarchy

```
TelemetryBaseError
├── LoggerError        — logger mis-use (e.g. bad level)
├── ThermalError       — zone not found, duplicate registration
│   └── ThermalShutdownError  — shutdown threshold exceeded
└── FaultPredictorError — metric not found, duplicate registration
```

---

## Test Coverage (76/76)

| Category | Tests |
|----------|-------|
| Logger — basic log/filter/level | 18 |
| Logger — ring buffer overflow, drop counter | 8 |
| Logger — source filter, latest, counts | 10 |
| Thermal — state transitions | 12 |
| Thermal — throttle, clock pct, slope | 8 |
| Thermal — ThermalMonitor multi-zone | 10 |
| FaultPredictor — threshold risk levels | 6 |
| FaultPredictor — spike detection | 4 |
| FaultPredictor — trend escalation | 4 |
| FaultPredictor — multi-metric, highest_risk | 6 |
| **Total** | **76** |

---

## RTL Notes (for future Verilog implementation)

- **Logger** → FIFO with configurable depth; level filter is a 3-bit comparator. `dropped` is a saturating counter register.
- **ThermalZone** → Combinational comparator tree driving a 3-bit state register. Slope computation needs a small fixed-point accumulator; can be done in a background task at lower frequency.
- **FaultPredictor** → Rolling window is a shift register bank. Mean/std require a fixed-point accumulator and square-root unit; slope needs a least-squares engine. All can share a single DSP slice per metric if time-multiplexed.
- All three components are stateless between clock edges except for their respective shift registers — no blocking memory accesses.
