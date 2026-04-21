"""Regression test for `examples/tier1_custom_dataset.py`.

Pins the customer-integration story: if a refactor breaks this, the
integration guide's claim ("plug in your own sensor format + your own
safety rule without editing our source") is wrong.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "tier1_custom_dataset.py"


@pytest.fixture(scope="module")
def example_output() -> str:
    """Run the example exactly as a customer would, capture output."""
    if not EXAMPLE.exists():
        pytest.skip(f"example not found: {EXAMPLE}")
    out = subprocess.check_output(
        [sys.executable, str(EXAMPLE)],
        cwd=EXAMPLE.parent.parent,
        text=True, timeout=120,
    )
    return out


def test_example_runs_to_completion(example_output):
    assert "Integration succeeded" in example_output
    assert "without a single edit" in example_output


def test_example_shows_all_four_sensor_kinds(example_output):
    for expected in ("CAMERA", "LIDAR", "RADAR", "CAN"):
        assert expected in example_output, f"missing sensor: {expected}"


def test_example_exercises_all_4_cameras(example_output):
    for cam in ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK"):
        assert cam in example_output


def test_example_exercises_both_radars(example_output):
    for rad in ("RADAR_FRONT", "RADAR_REAR"):
        assert rad in example_output


def test_example_oem_safety_rule_fires(example_output):
    """The OEM's custom plausibility rule must reject the pedestrian
    placed at 0.3m. If it doesn't fire, the integration story is hollow."""
    assert "REJECT" in example_output
    assert "pedestrian" in example_output
    assert "Rule fired 5/5 times" in example_output


def test_example_runs_oem_model_benchmark(example_output):
    """Proves backend + compiler + quantiser path works on OEM's model."""
    assert "ms/inference" in example_output
    assert "GMACs" in example_output


def test_dataset_is_api_importable_from_examples():
    """The example imports only from public astracore API."""
    text = EXAMPLE.read_text()
    # Must not import from tools.npu_ref or src.* (those are internal)
    lines = text.splitlines()
    for line in lines:
        if line.startswith("from tools.") or line.startswith("from src."):
            raise AssertionError(
                f"example uses internal import: {line!r} — "
                f"customer-facing code must only use astracore.*"
            )


def _load_example():
    """Import the example module reliably (dataclasses in it need the
    module to be registered in sys.modules before execution)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("tier1_example", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tier1_example"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_my_fleet_dataset_satisfies_protocol():
    """Direct-import the example's Dataset class and check it passes
    isinstance vs the Dataset Protocol (not via subprocess)."""
    mod = _load_example()

    from astracore.dataset import Dataset
    ds = mod.MyFleetDataset(n_samples=1)
    assert isinstance(ds, Dataset), (
        "MyFleetDataset doesn't satisfy the Dataset Protocol — integration "
        "guide promise is broken"
    )
    assert ds.list_scenes() == ["trip-20260419-0800"]
    scene = ds.get_scene(ds.list_scenes()[0])
    assert len(scene) == 1
    sample = next(iter(scene))
    assert len(sample.cameras) == 4
    assert len(sample.lidars) == 1
    assert len(sample.radars) == 2
    assert len(sample.can) == 2


def test_oem_safety_rule_is_pure_function():
    """Safety rule must be importable + callable without side effects."""
    mod = _load_example()

    from astracore.dataset import GroundTruthObject

    # Rejected case: within threshold
    close = GroundTruthObject(
        timestamp_us=0, track_id="t1", object_class="pedestrian",
        x_m=0.3, y_m=0, z_m=0.5, length_m=0.5, width_m=0.5,
        height_m=1.7, yaw_rad=0,
    )
    v = mod.my_oem_plausibility_rule(close)
    assert v.accepted is False
    assert "0.30" in v.reason

    # Accepted case: far away
    far = GroundTruthObject(
        timestamp_us=0, track_id="t2", object_class="pedestrian",
        x_m=5.0, y_m=0, z_m=0.5, length_m=0.5, width_m=0.5,
        height_m=1.7, yaw_rad=0,
    )
    v = mod.my_oem_plausibility_rule(far)
    assert v.accepted is True

    # Non-pedestrian never triggered
    car = GroundTruthObject(
        timestamp_us=0, track_id="t3", object_class="car",
        x_m=0.3, y_m=0, z_m=0.5, length_m=4.5, width_m=1.8,
        height_m=1.5, yaw_rad=0,
    )
    v = mod.my_oem_plausibility_rule(car)
    assert v.accepted is True
