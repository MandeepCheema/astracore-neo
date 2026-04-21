"""End-to-end custom sensor fusion example — UltrasonicProximityAlarm.

Scope
-----
Takes an AstraCore :class:`Sample` (which already carries ultrasonic +
lidar + CAN) and fuses them into a discrete proximity-alarm state:
``OFF``, ``CAUTION``, ``WARNING``, or ``CRITICAL``.

Why fuse three sensors?
-----------------------
* Ultrasonics see close (~0.15-3 m) obstacles the camera misses in
  blind spots — curbs, low bollards, walls. But they false-fire on
  plastic bags, heavy rain, and sensor crosstalk. A single US reading
  alone is not safe enough to trigger emergency brake.
* Lidar has the range and accuracy (±2 cm) to cross-check: if both
  agree within 0.5 m on an obstacle, confidence goes way up. If only
  one reports something, it's a CAUTION, not a WARNING.
* CAN gives us vehicle speed. Proximity thresholds must scale with
  speed — at 30 km/h you need 10× more stopping distance than at
  3 km/h. Hard-coded thresholds would either false-fire at parking
  speeds or miss real hazards at road speed.

This is exactly the kind of fusion ADAS Tier-1 suppliers write every
day. We ship it as a reference to show the SDK plumbing supports
plugins like this without touching ``astracore`` source.

Run directly::

    python examples/ultrasonic_proximity_alarm.py

The demo scans a synthetic robotaxi-style scene, prints per-sample
alarm states, and summarises the distribution at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from astracore.dataset import (
    CanMessage,
    LidarFrame,
    Sample,
    Scene,
    SensorKind,
    SyntheticDataset,
    UltrasonicSample,
)


# ---------------------------------------------------------------------------
# Alarm state machine
# ---------------------------------------------------------------------------

class AlarmLevel(Enum):
    OFF = 0           # nothing within range
    CAUTION = 1       # a single sensor reports something nearby
    WARNING = 2       # two sensors agree on an obstacle inside warning band
    CRITICAL = 3      # critical-band hit; brake assist should engage


@dataclass
class AlarmThresholds:
    """Ranges at which each alarm band engages, as a function of speed.

    All values are in metres. Thresholds are ``base + reaction * speed``:
    the car needs `reaction_s * speed_mps` metres to stop without panic,
    plus a base margin for the sensor's own uncertainty.
    """
    base_critical_m: float = 0.30          # sensor-close-to-bumper band
    base_warning_m:  float = 0.60
    base_caution_m:  float = 1.00
    reaction_s:      float = 0.6           # driver / AEB latency budget

    def for_speed(self, speed_mps: float) -> Tuple[float, float, float]:
        """Return ``(critical_m, warning_m, caution_m)`` at this speed.

        * CRITICAL is a bumper-kiss band — does NOT scale with speed.
          It means "even at v=0, we are unacceptably close."
        * WARNING scales with speed: the driver / AEB needs this
          distance to brake at the current velocity.
        * CAUTION is the earliest warning band; scales more aggressively.
        """
        # Speed can be negative when reversing; take magnitude for the
        # threshold (both directions need a stopping distance).
        v = max(0.0, abs(float(speed_mps)))
        crit = self.base_critical_m
        warn = self.base_warning_m  + self.reaction_s * v * 0.5
        caut = self.base_caution_m  + self.reaction_s * v * 1.0
        return crit, warn, caut


@dataclass
class AlarmDecision:
    """Output of one fusion step."""
    level: AlarmLevel
    min_us_distance_m: float = float("inf")
    closest_us_sensor: str = ""
    min_lidar_distance_m: float = float("inf")
    lidar_confirmed: bool = False
    vehicle_speed_kph: float = 0.0
    thresholds_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    reason: str = ""

    def as_dict(self) -> Dict:
        d = asdict(self)
        d["level"] = self.level.name
        return d


# ---------------------------------------------------------------------------
# Fusion rule
# ---------------------------------------------------------------------------

# Zone in front of the vehicle where we consider obstacles relevant.
# +x is forward, +y is left. 4 m wide × 4 m forward is a
# plausible bumper-forward search zone for parking speeds.
_FORWARD_X_M = 4.0
_HALF_WIDTH_Y_M = 2.0
_Z_BAND_M = 1.5             # ignore ground + sky


def _forward_zone_mask(xyz: np.ndarray) -> np.ndarray:
    """Keep points in front of + close to the bumper."""
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    return (
        (x > -_FORWARD_X_M * 0.25)   # allow slight overhang behind bumper
        & (x < _FORWARD_X_M)
        & (np.abs(y) < _HALF_WIDTH_Y_M)
        & (np.abs(z) < _Z_BAND_M)
    )


def _min_lidar_distance(lidar: Optional[LidarFrame]) -> float:
    if lidar is None or lidar.points.size == 0:
        return float("inf")
    xyz = lidar.points[:, :3].astype(np.float32)
    mask = _forward_zone_mask(xyz)
    if not mask.any():
        return float("inf")
    # Horizontal (x-y) distance from ego. z filtered already.
    d = np.sqrt(xyz[mask, 0] ** 2 + xyz[mask, 1] ** 2)
    return float(d.min())


def _min_us_reading(
    ultrasonics: Dict[str, UltrasonicSample],
    *,
    position_prefixes: Tuple[str, ...] = ("front",),
    sensor_max_range_m: float = 3.0,
    max_range_epsilon: float = 0.02,
) -> Tuple[float, str]:
    """Return (min_distance_m, sensor_name) for US sensors on the
    specified positions.

    Readings interpreted as *no echo received*:

    * ``distance_m < 0``  — canonical OEM sentinel for "no return".
    * ``distance_m >= sensor_max_range_m - max_range_epsilon`` — reading
      is at or beyond the sensor's range limit, which physically means
      "no obstacle detected in range", not "obstacle at the max-range
      boundary". Without this check, a sensor reporting 3.0 m when its
      effective range is 3.0 m would fire the caution alarm at highway
      speeds (where the caution threshold is ≥ 3 m). Bug found
      2026-04-20 via real-world scenario sweep.

    Returns ``(inf, "")`` when no sensor has a valid in-range reading.
    """
    best_d = float("inf")
    best_name = ""
    max_range_threshold = sensor_max_range_m - max_range_epsilon
    for name, us in ultrasonics.items():
        if us.distance_m is None or us.distance_m < 0:
            continue
        if us.distance_m >= max_range_threshold:
            continue
        if position_prefixes and not any(
            us.position.startswith(p) for p in position_prefixes
        ):
            continue
        if us.distance_m < best_d:
            best_d = float(us.distance_m)
            best_name = name
    return best_d, best_name


def _vehicle_speed_mps(can: List[CanMessage]) -> float:
    """Extract current speed from CAN messages.

    Accepts ``VehicleSpeed`` in ``kph`` or ``m/s`` (or ``mps``). Returns
    0.0 if absent — safer to assume stopped than to pick a large number.
    """
    for m in can:
        if m.dbc_name.lower().endswith("vehiclespeed") or m.dbc_name.lower() == "vehiclespeed":
            if m.unit.lower() in {"kph", "km/h", "kmph"}:
                return float(m.value) * (1000.0 / 3600.0)
            if m.unit.lower() in {"m/s", "mps"}:
                return float(m.value)
            return float(m.value) * (1000.0 / 3600.0)   # default: kph
    return 0.0


class UltrasonicProximityAlarm:
    """Fuses ultrasonic + lidar + CAN into a proximity-alarm decision.

    Usage::

        alarm = UltrasonicProximityAlarm()
        decision = alarm.decide(sample)
        if decision.level is AlarmLevel.CRITICAL:
            fire_brake_assist()

    The fusion:
      1. Take the minimum US distance across all forward sensors.
      2. Take the minimum lidar-point distance inside the forward zone.
      3. Look up vehicle speed from CAN (defaults to 0 km/h).
      4. Compute speed-adaptive critical / warning / caution thresholds.
      5. Decide alarm level:
         * CRITICAL if US < critical (hardware-close regardless of speed
           uncertainty) OR both US and lidar < warning AND speed > 1 m/s
         * WARNING if both sensors agree within ``agree_tol_m`` and
           their common reading < warning
         * CAUTION if either US or lidar reports something < caution
         * OFF otherwise.
    """

    def __init__(self,
                 thresholds: Optional[AlarmThresholds] = None,
                 agree_tol_m: float = 0.5,
                 sensor_max_range_m: float = 3.0):
        self.thresholds = thresholds or AlarmThresholds()
        self.agree_tol_m = agree_tol_m
        # Forwarded into ``_min_us_reading`` so readings at the sensor's
        # physical limit are treated as no-echo, not obstacle.
        self.sensor_max_range_m = float(sensor_max_range_m)

    # ---- Fusion ---------------------------------------------------------

    def decide(self, sample: Sample) -> AlarmDecision:
        lidar = next(iter(sample.lidars.values()), None)
        lidar_d = _min_lidar_distance(lidar)
        us_d, us_name = _min_us_reading(
            sample.ultrasonics,
            sensor_max_range_m=self.sensor_max_range_m,
        )
        v_mps = _vehicle_speed_mps(sample.can)
        crit, warn, caut = self.thresholds.for_speed(v_mps)

        lidar_confirms = bool(
            np.isfinite(us_d)
            and np.isfinite(lidar_d)
            and abs(us_d - lidar_d) <= self.agree_tol_m
        )

        level = AlarmLevel.OFF
        reason = "no obstacle in range"

        # CRITICAL — either US says very close, or both agree close + moving.
        if us_d < crit:
            level = AlarmLevel.CRITICAL
            reason = (f"US {us_name} reports {us_d:.2f}m < "
                      f"critical={crit:.2f}m")
        elif lidar_confirms and us_d < warn and v_mps > 1.0:
            level = AlarmLevel.CRITICAL
            reason = (f"US+lidar agree ({us_d:.2f}m / {lidar_d:.2f}m) "
                      f"inside warning={warn:.2f}m while moving "
                      f"{v_mps * 3.6:.1f} kph")
        elif lidar_confirms and us_d < warn:
            level = AlarmLevel.WARNING
            reason = (f"US+lidar agree on obstacle at "
                      f"~{min(us_d, lidar_d):.2f}m, "
                      f"warning={warn:.2f}m")
        elif us_d < caut or lidar_d < caut:
            level = AlarmLevel.CAUTION
            single = "US" if us_d < lidar_d else "lidar"
            reason = (f"{single} reports {min(us_d, lidar_d):.2f}m < "
                      f"caution={caut:.2f}m; other sensor did not confirm")

        return AlarmDecision(
            level=level,
            min_us_distance_m=us_d,
            closest_us_sensor=us_name,
            min_lidar_distance_m=lidar_d,
            lidar_confirmed=lidar_confirms,
            vehicle_speed_kph=v_mps * 3.6,
            thresholds_m=(crit, warn, caut),
            reason=reason,
        )

    def decide_scene(self, scene: Scene) -> List[AlarmDecision]:
        return [self.decide(s) for s in scene]


# ---------------------------------------------------------------------------
# Validation helper — run across a scene and report histogram
# ---------------------------------------------------------------------------

@dataclass
class AlarmScenarioReport:
    scene_id: str
    n_samples: int
    histogram: Dict[str, int] = field(default_factory=dict)
    min_us_observed_m: float = float("inf")
    min_lidar_observed_m: float = float("inf")
    first_critical_sample: Optional[str] = None
    decisions: List[AlarmDecision] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "scene_id": self.scene_id,
            "n_samples": self.n_samples,
            "histogram": self.histogram,
            "min_us_observed_m": self.min_us_observed_m,
            "min_lidar_observed_m": self.min_lidar_observed_m,
            "first_critical_sample": self.first_critical_sample,
            "decisions": [d.as_dict() for d in self.decisions],
        }


def run_scenario(scene: Scene,
                 alarm: Optional[UltrasonicProximityAlarm] = None,
                 ) -> AlarmScenarioReport:
    alarm = alarm or UltrasonicProximityAlarm()
    out = AlarmScenarioReport(
        scene_id=scene.scene_id,
        n_samples=len(scene),
        histogram={lv.name: 0 for lv in AlarmLevel},
    )
    for sample in scene:
        d = alarm.decide(sample)
        out.decisions.append(d)
        out.histogram[d.level.name] = out.histogram.get(d.level.name, 0) + 1
        if d.min_us_distance_m < out.min_us_observed_m:
            out.min_us_observed_m = d.min_us_distance_m
        if d.min_lidar_distance_m < out.min_lidar_observed_m:
            out.min_lidar_observed_m = d.min_lidar_distance_m
        if (d.level is AlarmLevel.CRITICAL
                and out.first_critical_sample is None):
            out.first_critical_sample = sample.sample_id
    return out


# ---------------------------------------------------------------------------
# Demo entry-point
# ---------------------------------------------------------------------------

def _inject_close_obstacle(scene: Scene, *,
                           at_sample: int,
                           x_m: float = 0.25,
                           us_position: str = "front-center",
                           lidar_only: bool = False,
                           us_only: bool = False) -> None:
    """Synthesise a close-in obstacle at one sample.

    In-place mutates: (optionally) adds a 200-point dense lidar cluster
    at ``x_m`` and/or overrides one front US sensor's reading to ``x_m``.
    Two flags let the demo simulate single-sensor confusion (e.g. US
    false-positive without lidar confirmation).
    """
    if at_sample >= len(scene):
        return
    s: Sample = scene.samples[at_sample]

    if not us_only and s.lidars:
        lidar = next(iter(s.lidars.values()))
        extra = np.tile(
            np.array([[x_m, 0.0, 0.6, 0.9]], dtype=np.float32),
            (200, 1),
        )
        extra[:, :3] += np.random.default_rng(0).normal(
            scale=[0.05, 0.15, 0.1], size=(200, 3)
        ).astype(np.float32)
        lidar.points = np.concatenate([lidar.points, extra], axis=0)

    if not lidar_only:
        for name, us in s.ultrasonics.items():
            if us.position == us_position:
                us.distance_m = float(x_m)
                break


def _simulate_parking_crawl(scene: Scene,
                            speeds_kph: Optional[List[float]] = None) -> None:
    """Make a synthetic scene behave like a parking manoeuvre.

    * Overwrites CAN ``VehicleSpeed`` per sample to a slow crawl.
    * Clears every US sensor's distance to 3.0 m (max range, "no echo").
    * Drops lidar points inside the proximity zone so baseline samples
      show OFF.

    Caller then injects synthetic obstacles via :func:`_inject_close_obstacle`
    to drive WARNING / CRITICAL decisions.
    """
    speeds_kph = speeds_kph or [5.0] * len(scene)
    for i, sample in enumerate(scene):
        v = float(speeds_kph[min(i, len(speeds_kph) - 1)])
        found = False
        for m in sample.can:
            if m.dbc_name.lower().endswith("vehiclespeed"):
                m.value = v
                m.unit = "kph"
                found = True
                break
        if not found:
            sample.can.append(CanMessage(
                timestamp_us=sample.timestamp_us,
                dbc_name="VehicleSpeed",
                value=v,
                unit="kph",
            ))
        # Use the canonical no-echo sentinel so the alarm correctly
        # interprets a clear baseline as "no obstacle", not "obstacle at
        # sensor's max range". Readings == max_range are also filtered by
        # the alarm, but -1 is the explicit signal.
        for us in sample.ultrasonics.values():
            us.distance_m = -1.0
        for lidar in sample.lidars.values():
            xyz = lidar.points[:, :3]
            mask = _forward_zone_mask(xyz)
            lidar.points = lidar.points[~mask]


def build_parking_scenario(n_samples: int = 10) -> Scene:
    """Canonical parking scenario used in the demo + regression test.

    Starts with a clean scene (v=5->0 kph ramp, US all clear, no lidar
    obstacles in proximity zone), then injects four events:

    * sample 2: US-only echo at 0.9 m — should be CAUTION (single-sensor)
    * sample 4: US + lidar agree at 0.8 m — should be WARNING
    * sample 6: obstacle appears at 0.35 m  — should be CRITICAL
      (inside the fixed critical band, speed-independent)
    * sample 8: lidar-only echo at 0.9 m — should be CAUTION
    """
    ds = SyntheticDataset(preset_name="extended-sensors", n_ultrasonics=12)
    scene = ds.get_scene(ds.list_scenes()[0])
    scene.samples = list(scene.samples[:n_samples])

    speeds = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5][:n_samples]
    _simulate_parking_crawl(scene, speeds_kph=speeds)

    _inject_close_obstacle(scene, at_sample=2, x_m=0.9,
                           us_position="front-center", us_only=True)
    _inject_close_obstacle(scene, at_sample=4, x_m=0.8,
                           us_position="front-center")
    _inject_close_obstacle(scene, at_sample=6, x_m=0.20,
                           us_position="front-center")
    _inject_close_obstacle(scene, at_sample=8, x_m=0.9,
                           us_position="front-center", lidar_only=True)

    return scene


def main() -> int:
    print("=" * 72)
    print("Custom sensor example — UltrasonicProximityAlarm (US + lidar + CAN)")
    print("=" * 72)

    print("\nScenario: 10-sample parking crawl decelerating 5 -> 0.5 kph")
    print("Events: sample 2 US-only, sample 4 both agree, sample 6 critical,")
    print("        sample 8 lidar-only")
    scene = build_parking_scenario()

    alarm = UltrasonicProximityAlarm()
    print("\nPer-sample alarm decisions:")
    print("-" * 72)
    for sample in scene:
        d = alarm.decide(sample)
        tag = d.level.name
        # Keep the OFF rows visible so the contrast is clear.
        print(f"  {sample.sample_id}  {tag:<8}  "
              f"us={d.min_us_distance_m:5.2f}m  "
              f"lidar={d.min_lidar_distance_m:5.2f}m  "
              f"v={d.vehicle_speed_kph:4.1f}kph")
        if d.level is not AlarmLevel.OFF:
            print(f"      -> {d.reason}")

    report = run_scenario(scene, alarm)
    print("\nHistogram:")
    for level, n in report.histogram.items():
        bar = "#" * n
        print(f"  {level:<8} {n:>3}  {bar}")
    print(f"\n  min US observed:    {report.min_us_observed_m:.2f} m")
    print(f"  min lidar observed: {report.min_lidar_observed_m:.2f} m")
    print(f"  first CRITICAL at:  {report.first_critical_sample}")

    print("\n" + "=" * 72)
    print("Fused three sensor kinds (ultrasonic + lidar + CAN) with zero edits")
    print("to astracore source. The fusion rule is pluggable — swap")
    print("thresholds or cross-sensor logic for the OEM's own tuning.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
