"""Pillar 3 proof — OEM can feed sensor data through the stack.

Covers:
 - Dataset / Scene / Sample contract (dataclass shapes)
 - SyntheticDataset determinism + coverage (all sensor kinds present)
 - Replay harness runs a scene end-to-end through the perception
   pipeline without exceptions
 - nuScenes connector is discoverable (import gated) but fails
   gracefully when devkit is absent
"""

from __future__ import annotations

import numpy as np
import pytest

from astracore.dataset import (
    CameraFrame,
    Dataset,
    GroundTruthObject,
    LidarFrame,
    RadarFrame,
    Sample,
    Scene,
    SensorKind,
    SyntheticDataset,
    replay_scene,
)
from astracore.dataset.replay import PerSampleResult, ReplayResult
from astracore.dataset import PRESETS, preset


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

def test_synthetic_dataset_implements_protocol():
    ds = SyntheticDataset(n_scenes=2, samples_per_scene=5)
    assert isinstance(ds, Dataset)
    assert ds.name == "synthetic"
    scenes = ds.list_scenes()
    assert len(scenes) == 2
    assert all(s.startswith("synthetic-scene-") for s in scenes)


def test_synthetic_dataset_is_deterministic():
    a = SyntheticDataset(seed=42, n_scenes=1, samples_per_scene=3).get_scene(
        "synthetic-scene-000"
    )
    b = SyntheticDataset(seed=42, n_scenes=1, samples_per_scene=3).get_scene(
        "synthetic-scene-000"
    )
    assert a.name == b.name
    assert len(a) == len(b)
    # Same seed ⇒ same camera pixel values
    for sa, sb in zip(a, b):
        for key in sa.cameras:
            np.testing.assert_array_equal(
                sa.cameras[key].data, sb.cameras[key].data,
            )


def test_synthetic_scene_has_all_sensor_kinds():
    ds = SyntheticDataset(n_scenes=1, samples_per_scene=1)
    scene = ds.get_scene("synthetic-scene-000")
    sample = next(iter(scene))
    for kind in (SensorKind.CAMERA, SensorKind.LIDAR, SensorKind.RADAR,
                 SensorKind.IMU, SensorKind.GNSS, SensorKind.CAN):
        assert sample.has(kind), f"missing {kind}"


def test_sample_sensors_helper_matches_fields():
    ds = SyntheticDataset(n_scenes=1, samples_per_scene=1)
    sample = next(iter(ds.get_scene("synthetic-scene-000")))
    assert sample.sensors(SensorKind.CAMERA) == sample.cameras
    assert sample.sensors(SensorKind.LIDAR) == sample.lidars


def test_camera_frame_shape_matches_request():
    ds = SyntheticDataset(n_scenes=1, samples_per_scene=1,
                          image_shape=(240, 320))
    sample = next(iter(ds.get_scene("synthetic-scene-000")))
    cam = next(iter(sample.cameras.values()))
    assert cam.data.shape == (240, 320, 3)
    assert cam.data.dtype == np.uint8


def test_ground_truth_objects_populated():
    ds = SyntheticDataset(n_scenes=1, samples_per_scene=3, seed=1)
    total = 0
    for sample in ds.get_scene("synthetic-scene-000"):
        total += len(sample.ground_truth)
        for gt in sample.ground_truth:
            assert isinstance(gt, GroundTruthObject)
            assert gt.length_m > 0 and gt.width_m > 0 and gt.height_m > 0
    assert total > 0


def test_scene_unknown_id_raises():
    ds = SyntheticDataset(n_scenes=1)
    with pytest.raises(KeyError):
        ds.get_scene("does-not-exist")


# ---------------------------------------------------------------------------
# Replay harness tests
# ---------------------------------------------------------------------------

def test_replay_scene_runs_end_to_end():
    ds = SyntheticDataset(n_scenes=1, samples_per_scene=3,
                          image_shape=(120, 160),
                          n_lidar_points=512,
                          n_radar_detections=8,
                          seed=7)
    scene = ds.get_scene("synthetic-scene-000")
    result = replay_scene(scene)
    assert isinstance(result, ReplayResult)
    assert result.n_samples == 3
    assert len(result.per_sample) == 3
    assert result.wall_s_total > 0
    for r in result.per_sample:
        assert isinstance(r, PerSampleResult)
        assert r.wall_ms > 0


def test_replay_summary_aggregates_cleanly():
    ds = SyntheticDataset(n_scenes=1, samples_per_scene=4,
                          image_shape=(120, 160),
                          n_lidar_points=512,
                          n_radar_detections=8,
                          seed=11)
    scene = ds.get_scene("synthetic-scene-000")
    result = replay_scene(scene)
    summary = result.summary()
    assert "mean_camera_det" in summary
    assert "mean_fused_obj" in summary
    assert "mean_ms_per_frame" in summary


def test_replay_lidar_detects_synthetic_clusters():
    """Synthetic data has 3–4 dense clusters; replay should find some."""
    ds = SyntheticDataset(n_scenes=1, samples_per_scene=2,
                          image_shape=(120, 160),
                          n_lidar_points=512,
                          seed=21)
    scene = ds.get_scene("synthetic-scene-000")
    result = replay_scene(scene, lidar_cluster_eps_m=1.5,
                          lidar_cluster_min_points=8)
    total_clusters = sum(r.n_lidar_clusters for r in result.per_sample)
    assert total_clusters >= 1, "clustering found zero synthetic clusters"


def test_replay_detector_fn_is_invoked_per_camera_frame():
    ds = SyntheticDataset(n_scenes=1, samples_per_scene=3,
                          image_shape=(120, 160),
                          n_lidar_points=256,
                          seed=0)
    scene = ds.get_scene("synthetic-scene-000")

    call_count = {"n": 0}

    def fake_detector(rgb: np.ndarray):
        call_count["n"] += 1
        assert rgb.ndim == 3
        # Pretend we detected two boxes every frame.
        return [(0, 0, 10, 10), (20, 20, 40, 40)]

    result = replay_scene(scene, detector_fn=fake_detector)
    # One camera per sample × 3 samples.
    assert call_count["n"] == 3
    assert sum(r.n_camera_detections for r in result.per_sample) == 6


# ---------------------------------------------------------------------------
# nuScenes connector discoverability (graceful if devkit absent)
# ---------------------------------------------------------------------------

def test_presets_cover_hardware_scales():
    """Tiny → standard → vlp32 → vlp64 → robotaxi are all defined."""
    for name in ("tiny", "standard", "vlp32", "vlp64", "robotaxi"):
        p = preset(name)
        assert p.n_scenes > 0
        assert p.samples_per_scene > 0
        assert p.n_lidar_points > 0
        assert p.image_shape[0] > 0 and p.image_shape[1] > 0


def test_presets_strictly_increase_in_scale():
    tiny = preset("tiny")
    std = preset("standard")
    vlp32 = preset("vlp32")
    vlp64 = preset("vlp64")
    assert tiny.n_lidar_points < std.n_lidar_points < vlp32.n_lidar_points \
           < vlp64.n_lidar_points
    assert (tiny.image_shape[0] * tiny.image_shape[1]
            <= std.image_shape[0] * std.image_shape[1]
            <= vlp32.image_shape[0] * vlp32.image_shape[1])


def test_multi_camera_multi_radar_preset():
    """VLP-32 has 4 cameras and 4 radars; sample must carry all of them."""
    ds = SyntheticDataset(preset_name="vlp32", seed=3)
    sample = next(iter(ds.get_scene(ds.list_scenes()[0])))
    assert len(sample.cameras) == 4
    assert len(sample.radars) == 4
    # The lidar is always a single top-mounted sensor in our preset set.
    assert len(sample.lidars) == 1


def test_preset_config_roundtrip():
    ds = SyntheticDataset(preset_name="standard")
    cfg = ds.config
    assert cfg["preset"] == "standard"
    assert cfg["n_lidar_points"] == preset("standard").n_lidar_points


def test_preset_unknown_name_raises():
    with pytest.raises(KeyError):
        preset("nonexistent-preset")


def test_nuscenes_module_importable_or_gracefully_skipped():
    try:
        from astracore.dataset import NuScenesDataset
    except Exception:
        pytest.skip("nuscenes-devkit not installed — connector intentionally omitted")
    # Don't instantiate; that requires disk data.
    assert NuScenesDataset.name == "nuscenes"
