"""Tests for the 5 new sensor types + YAML config layer (2026-04-19)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astracore.dataset import (
    DepthFrame,
    EventFrame,
    MicrophoneFrame,
    Sample,
    SensorKind,
    SyntheticDataset,
    ThermalFrame,
    UltrasonicSample,
)


# ---------------------------------------------------------------------------
# New sensor dataclasses
# ---------------------------------------------------------------------------

def test_sensorkind_covers_all_11_kinds():
    names = {k.name for k in SensorKind}
    expected = {"CAMERA", "LIDAR", "RADAR", "IMU", "GNSS", "CAN",
                "ULTRASONIC", "MICROPHONE", "THERMAL", "EVENT", "DEPTH"}
    assert names == expected


def test_extended_sensors_preset_generates_all_kinds():
    ds = SyntheticDataset(preset_name="extended-sensors")
    sample = next(iter(ds.get_scene(ds.list_scenes()[0])))
    assert sample.ultrasonics and sample.microphones
    assert sample.thermals and sample.events and sample.depths


def test_robotaxi_preset_has_12_ultrasonics_and_full_rig():
    ds = SyntheticDataset(preset_name="extended-sensors",
                          n_ultrasonics=12)     # emulate robotaxi without the 4k cameras
    sample = next(iter(ds.get_scene(ds.list_scenes()[0])))
    assert len(sample.ultrasonics) == 12
    names = set(sample.ultrasonics)
    # Must contain distinct positions around the vehicle
    assert len({s.position for s in sample.ultrasonics.values()}) == 12


def test_ultrasonic_sample_shape():
    ds = SyntheticDataset(preset_name="extended-sensors")
    sample = next(iter(ds.get_scene(ds.list_scenes()[0])))
    us = next(iter(sample.ultrasonics.values()))
    assert isinstance(us, UltrasonicSample)
    assert 0.0 < us.distance_m <= 5.0
    assert us.position
    assert us.snr_db is not None


def test_microphone_frame_shape_and_dtype():
    ds = SyntheticDataset(preset_name="extended-sensors", audio_samples=2048)
    sample = next(iter(ds.get_scene(ds.list_scenes()[0])))
    mic = next(iter(sample.microphones.values()))
    assert isinstance(mic, MicrophoneFrame)
    assert mic.data.shape == (2048,)
    assert mic.data.dtype == np.int16
    assert mic.sample_rate_hz == 16_000


def test_thermal_frame_shape_and_units():
    ds = SyntheticDataset(preset_name="extended-sensors",
                          thermal_shape=(180, 320))
    sample = next(iter(ds.get_scene(ds.list_scenes()[0])))
    th = next(iter(sample.thermals.values()))
    assert isinstance(th, ThermalFrame)
    assert th.data.shape == (180, 320)
    assert th.units in {"raw_counts", "kelvin", "celsius"}


def test_event_frame_shape_and_polarity():
    ds = SyntheticDataset(preset_name="extended-sensors",
                          event_rate_per_frame=1024)
    sample = next(iter(ds.get_scene(ds.list_scenes()[0])))
    evt = next(iter(sample.events.values()))
    assert isinstance(evt, EventFrame)
    assert evt.events.shape == (1024, 4)
    # Polarity column is -1 or +1
    polarity = evt.events[:, 3]
    assert set(polarity.tolist()).issubset({-1, 1})
    assert evt.timestamp_us_end > evt.timestamp_us_start


def test_depth_frame_has_confidence():
    ds = SyntheticDataset(preset_name="extended-sensors",
                          depth_shape=(100, 120))
    sample = next(iter(ds.get_scene(ds.list_scenes()[0])))
    dep = next(iter(sample.depths.values()))
    assert isinstance(dep, DepthFrame)
    assert dep.depth_m.shape == (100, 120)
    assert dep.confidence is not None
    assert dep.confidence.min() >= 0 and dep.confidence.max() <= 1


def test_sample_sensors_helper_returns_new_kinds():
    ds = SyntheticDataset(preset_name="extended-sensors")
    sample = next(iter(ds.get_scene(ds.list_scenes()[0])))
    assert sample.sensors(SensorKind.ULTRASONIC)
    assert sample.sensors(SensorKind.MICROPHONE)
    assert sample.sensors(SensorKind.THERMAL)
    assert sample.sensors(SensorKind.EVENT)
    assert sample.sensors(SensorKind.DEPTH)


# ---------------------------------------------------------------------------
# YAML config layer
# ---------------------------------------------------------------------------

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def test_yaml_config_loads_valid_tier1_adas():
    from astracore import config
    cfg = config.load(EXAMPLES_DIR / "tier1_adas.yaml")
    assert cfg.version == config.SCHEMA_VERSION
    assert len(cfg.sensors.cameras) == 4
    assert len(cfg.sensors.ultrasonics) == 12
    assert len(cfg.sensors.radars) == 6
    assert cfg.sensors.gnss is not None
    assert cfg.sensors.imu is not None
    assert len(cfg.models) == 3


def test_yaml_config_sparsity_2colon4_survives_sexagesimal_trap():
    """The Tier-1 YAML has unquoted `sparsity: "2:4"` — must parse cleanly."""
    from astracore import config
    cfg = config.load(EXAMPLES_DIR / "tier1_adas.yaml")
    sparsities = {m.id: m.sparsity for m in cfg.models}
    assert sparsities["side_perception"] == "2:4"


def test_yaml_config_rejects_unknown_schema_version(tmp_path):
    from astracore import config
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 99\nname: test\n")
    with pytest.raises(config.ConfigError, match="schema version"):
        config.load(bad)


def test_yaml_config_rejects_duplicate_sensor_names(tmp_path):
    from astracore import config
    bad = tmp_path / "dup.yaml"
    bad.write_text("""
version: 1
sensors:
  cameras:
    - name: CAM_FRONT
    - name: CAM_FRONT    # duplicate!
""")
    with pytest.raises(config.ConfigError, match="duplicate sensor names"):
        config.load(bad)


def test_yaml_config_rejects_dangling_input_sensor(tmp_path):
    from astracore import config
    bad = tmp_path / "dangle.yaml"
    bad.write_text("""
version: 1
sensors:
  cameras:
    - name: CAM_FRONT
models:
  - id: m1
    path: /some/path.onnx
    input_sensor: CAM_BACK       # doesn't exist in sensors
""")
    with pytest.raises(config.ConfigError, match="not in sensors"):
        config.load(bad)


def test_yaml_config_rejects_duplicate_model_ids(tmp_path):
    from astracore import config
    bad = tmp_path / "dupmodel.yaml"
    bad.write_text("""
version: 1
models:
  - { id: m1, path: /a.onnx }
  - { id: m1, path: /b.onnx }
""")
    with pytest.raises(config.ConfigError, match="duplicate model ids"):
        config.load(bad)


def test_yaml_config_rejects_invalid_precision(tmp_path):
    from astracore import config
    bad = tmp_path / "badprec.yaml"
    bad.write_text("""
version: 1
models:
  - { id: m1, path: /a.onnx, precision: MYCUSTOM }
""")
    with pytest.raises(config.ConfigError, match="precision"):
        config.load(bad)


def test_yaml_config_round_trip_preserves_content(tmp_path):
    """`load` + `to_yaml` should be idempotent for a non-trivial config."""
    from astracore import config
    cfg = config.load(EXAMPLES_DIR / "tier1_adas.yaml")
    dumped = config.to_yaml(cfg)
    # Write to tmp and re-load
    tmp = tmp_path / "roundtrip.yaml"
    tmp.write_text(dumped)
    cfg2 = config.load(tmp)
    assert len(cfg2.sensors.cameras) == len(cfg.sensors.cameras)
    assert len(cfg2.sensors.ultrasonics) == len(cfg.sensors.ultrasonics)
    assert len(cfg2.models) == len(cfg.models)


def test_config_cli_validate_works():
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "-m", "astracore.cli", "configure",
         "--validate", str(EXAMPLES_DIR / "tier1_adas.yaml")],
        cwd=EXAMPLES_DIR.parent, capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0
    assert "[VALID]" in r.stdout
    assert "ultrasonics" in r.stdout
    assert "Tier-1" in r.stdout


def test_config_cli_returns_1_on_bad_file(tmp_path):
    import subprocess, sys
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1\nmodels:\n  - { id: m1, path: /x.onnx, precision: FOO }\n")
    r = subprocess.run(
        [sys.executable, "-m", "astracore.cli", "configure",
         "--validate", str(bad)],
        cwd=EXAMPLES_DIR.parent, capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 1
    assert "[INVALID]" in r.stderr
