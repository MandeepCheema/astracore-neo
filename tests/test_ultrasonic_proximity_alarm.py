"""Regression tests for examples/ultrasonic_proximity_alarm.py.

Pins the canonical parking-crawl scenario so the US + lidar + CAN
fusion example stays reproducible. If someone tweaks the fusion rule,
they have to consciously update the expected histogram.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


EXAMPLE = (Path(__file__).resolve().parent.parent
           / "examples" / "ultrasonic_proximity_alarm.py")


@pytest.fixture(scope="module")
def mod():
    import sys
    spec = importlib.util.spec_from_file_location("ultrasonic_proximity_alarm",
                                                  EXAMPLE)
    m = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can find the module via
    # sys.modules[cls.__module__] (required on Python 3.12+).
    sys.modules["ultrasonic_proximity_alarm"] = m
    spec.loader.exec_module(m)
    return m


def test_alarm_levels_enum(mod):
    lvls = {lv.name for lv in mod.AlarmLevel}
    assert lvls == {"OFF", "CAUTION", "WARNING", "CRITICAL"}


def test_thresholds_scale_with_speed_except_critical(mod):
    t = mod.AlarmThresholds()
    crit_stationary, warn_stationary, caut_stationary = t.for_speed(0.0)
    crit_fast, warn_fast, caut_fast = t.for_speed(10.0)      # 36 kph
    # Critical band does NOT grow with speed (hard bumper-kiss band).
    assert crit_stationary == crit_fast == t.base_critical_m
    # Warning + caution do scale.
    assert warn_fast > warn_stationary
    assert caut_fast > caut_stationary


def test_min_us_filters_negative_and_non_front(mod):
    from astracore.dataset import UltrasonicSample
    samples = {
        "us_no_echo":  UltrasonicSample("us_no_echo", 0, -1.0, "front-center"),
        "us_rear":     UltrasonicSample("us_rear", 0, 0.3, "rear-center"),
        "us_front_close": UltrasonicSample("us_front_close", 0, 0.9,
                                           "front-left"),
        "us_front_far":   UltrasonicSample("us_front_far", 0, 2.5,
                                           "front-right"),
    }
    d, name = mod._min_us_reading(samples)
    assert name == "us_front_close"
    assert d == pytest.approx(0.9)


def test_vehicle_speed_extracted_from_can(mod):
    from astracore.dataset import CanMessage
    can = [CanMessage(0, "VehicleSpeed", 36.0, "kph"),
           CanMessage(0, "SteeringAngle", 5.0, "deg")]
    assert mod._vehicle_speed_mps(can) == pytest.approx(10.0, rel=1e-6)

    # m/s unit also accepted
    can2 = [CanMessage(0, "VehicleSpeed", 7.5, "m/s")]
    assert mod._vehicle_speed_mps(can2) == pytest.approx(7.5)

    # Missing speed -> 0, not a panic value
    assert mod._vehicle_speed_mps([]) == 0.0


def test_canonical_parking_scenario_histogram(mod):
    """All four alarm bands must fire on the canonical scenario."""
    scene = mod.build_parking_scenario()
    report = mod.run_scenario(scene)
    # 10 samples, 4 injected events + 6 background OFF
    assert report.n_samples == 10
    assert report.histogram["CRITICAL"] == 1
    assert report.histogram["WARNING"] == 1
    assert report.histogram["CAUTION"] == 2
    assert report.histogram["OFF"] == 6


def test_critical_fires_regardless_of_speed(mod):
    """A 0.20 m obstacle in front must be CRITICAL even stopped."""
    from astracore.dataset import CanMessage, Sample, UltrasonicSample
    # Build a minimal sample with the US front-center at 0.20 m, v=0.
    sample = Sample(
        sample_id="t0", timestamp_us=0,
        cameras={}, lidars={}, radars={},
        can=[CanMessage(0, "VehicleSpeed", 0.0, "kph")],
        ultrasonics={"us_fc": UltrasonicSample(
            "us_fc", 0, 0.20, "front-center", snr_db=20.0,
        )},
    )
    d = mod.UltrasonicProximityAlarm().decide(sample)
    assert d.level.name == "CRITICAL"
    assert d.vehicle_speed_kph == pytest.approx(0.0)


def test_lidar_only_obstacle_degrades_to_caution(mod):
    from astracore.dataset import CanMessage, LidarFrame, Sample
    lidar_pts = np.zeros((16, 4), dtype=np.float32)
    lidar_pts[:, 0] = 0.9           # 0.9 m in front
    lidar_pts[:, 2] = 0.5           # above ground
    lidar_pts[:, 3] = 0.8           # intensity
    sample = Sample(
        sample_id="t1", timestamp_us=0,
        cameras={}, lidars={"l": LidarFrame("l", 0, lidar_pts)},
        radars={},
        can=[CanMessage(0, "VehicleSpeed", 3.0, "kph")],
        ultrasonics={},
    )
    d = mod.UltrasonicProximityAlarm().decide(sample)
    # With no US reading, we never get a two-sensor WARNING, so the
    # lidar-only echo in caution band produces CAUTION only.
    assert d.level.name == "CAUTION"
    assert not bool(d.lidar_confirmed)


def test_us_at_max_range_is_treated_as_no_echo(mod):
    """Regression for the 2026-04-20 highway-clear-road bug.

    A US sensor reporting ``distance_m = sensor_max_range_m`` means
    "no echo received", not "obstacle at the range limit". The alarm
    must not fire CAUTION just because the sensor maxed out.
    """
    from astracore.dataset import CanMessage, Sample, UltrasonicSample
    # Highway speed, 12 US sensors all at their 3.0 m max range.
    uss = {
        f"us_{i}": UltrasonicSample(
            f"us_{i}", 0, 3.0, "front-center", snr_db=20.0,
        )
        for i in range(12)
    }
    sample = Sample(
        sample_id="h0", timestamp_us=0,
        cameras={}, lidars={}, radars={},
        can=[CanMessage(0, "VehicleSpeed", 100.0, "kph")],
        ultrasonics=uss,
    )
    d = mod.UltrasonicProximityAlarm().decide(sample)
    # With no echo and no lidar, everything is clear.
    assert d.level.name == "OFF", (
        f"max-range US reading was interpreted as an obstacle at "
        f"{d.min_us_distance_m} m; alarm fired {d.level.name}"
    )


def test_parking_scenario_highway_cruise_is_all_OFF(mod):
    """Tightened gate after the max-range fix.

    10 samples at 100 kph, no obstacles. After the no-echo fix, the
    expected histogram is 10/0/0/0, not 10 CAUTIONs.
    """
    from astracore.dataset import SyntheticDataset
    ds = SyntheticDataset(preset_name="extended-sensors", n_ultrasonics=12)
    scene = ds.get_scene(ds.list_scenes()[0])
    scene.samples = list(scene.samples[:10])
    mod._simulate_parking_crawl(scene, speeds_kph=[100.0] * 10)
    rep = mod.run_scenario(scene)
    assert rep.histogram["OFF"] == 10
    assert rep.histogram["CAUTION"] == 0
    assert rep.histogram["WARNING"] == 0
    assert rep.histogram["CRITICAL"] == 0


def test_us_only_obstacle_degrades_to_caution(mod):
    from astracore.dataset import CanMessage, LidarFrame, Sample, UltrasonicSample
    # Lidar has no points in proximity zone -> distance = inf
    lidar_pts = np.full((8, 4), 50.0, dtype=np.float32)
    sample = Sample(
        sample_id="t2", timestamp_us=0,
        cameras={}, lidars={"l": LidarFrame("l", 0, lidar_pts)},
        radars={},
        can=[CanMessage(0, "VehicleSpeed", 3.0, "kph")],
        ultrasonics={"us_fc": UltrasonicSample(
            "us_fc", 0, 0.9, "front-center", snr_db=20.0,
        )},
    )
    d = mod.UltrasonicProximityAlarm().decide(sample)
    assert d.level.name == "CAUTION"
    assert d.closest_us_sensor == "us_fc"


def test_script_runs_cleanly():
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        capture_output=True, text=True,
        timeout=30,
        cwd=EXAMPLE.parent.parent,
    )
    assert r.returncode == 0, r.stderr
    assert "CRITICAL" in r.stdout
    assert "WARNING" in r.stdout
    assert "CAUTION" in r.stdout
