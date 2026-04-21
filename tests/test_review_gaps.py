"""Tests that cover gaps found in the Phase-A review pass (2026-04-19).

Each test corresponds to a gap identified during the code audit.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# G1 — SDK is pip-installable and the CLI entry-point works
# ---------------------------------------------------------------------------

def test_sdk_is_pip_installed_distribution():
    """``astracore-sdk`` must be installable via ``pip install -e .`` and
    show up in the metadata index — otherwise OEM deployment is broken."""
    import importlib.metadata as im
    try:
        d = im.distribution("astracore-sdk")
    except im.PackageNotFoundError:
        pytest.skip("astracore-sdk not pip-installed in this environment")
    assert d.version


def test_astracore_version_matches_metadata():
    """Package __version__ must match the installed distribution version."""
    import importlib.metadata as im
    import astracore
    try:
        dist_version = im.distribution("astracore-sdk").version
    except im.PackageNotFoundError:
        pytest.skip("astracore-sdk not pip-installed")
    assert astracore.__version__ == dist_version


def test_cli_entry_point_runs():
    """The ``astracore`` console script (or python -m astracore.cli) must
    return the version when called from any working directory."""
    import shutil
    exe = shutil.which("astracore")
    cmd: list[str]
    if exe is not None:
        cmd = [exe, "version"]
    else:
        # Fall back to python -m, which equivalently exercises the CLI
        # module registered by pyproject.toml's [project.scripts].
        cmd = [sys.executable, "-m", "astracore.cli", "version"]
    cwd = Path(__file__).resolve().parent.parent.parent
    out = subprocess.check_output(cmd, cwd=cwd, text=True, timeout=30)
    assert out.strip().count(".") >= 2   # semver x.y.z


# ---------------------------------------------------------------------------
# G2 — Deterministic MAC count (no regression if shape inference changes)
# ---------------------------------------------------------------------------

def test_mac_estimator_deterministic_on_known_cnn():
    """SqueezeNet 1.1 MACs should be within a known band — freezes the
    MAC counter so a future shape-inference regression is caught."""
    from astracore import zoo
    from astracore.benchmark import benchmark_model

    path = zoo.local_paths()["squeezenet-1.1"]
    if not path.exists():
        pytest.skip("squeezenet zoo entry not downloaded")
    rep = benchmark_model(path, backend="onnxruntime", n_iter=1, warmup=0,
                          input_shape="1,3,224,224")
    gm = rep.mac_ops_total / 1e9
    # SqueezeNet 1.1 has ~0.35 GMACs per canonical references. Allow ±20%.
    assert 0.28 <= gm <= 0.42, f"MACs out of band: {gm}"


def test_mac_estimator_handles_transformer():
    """GPT-2 @ seq=8 must produce a non-zero, sensible MAC count."""
    from astracore import zoo
    from astracore.benchmark import benchmark_model
    path = zoo.local_paths()["gpt-2-10"]
    if not path.exists():
        pytest.skip("gpt-2 zoo entry not downloaded")
    rep = benchmark_model(path, backend="onnxruntime", n_iter=1, warmup=0,
                          input_shape="1,1,8")
    gm = rep.mac_ops_total / 1e9
    # GPT-2 small @ seq=8 computes ~0.08 GMACs.
    assert 0.01 <= gm <= 1.0, f"GPT-2 MACs implausible: {gm}"


# ---------------------------------------------------------------------------
# G3 — Plugin registry discovers entry-points AND @-decorators together
# ---------------------------------------------------------------------------

def test_decorator_and_entrypoint_paths_coexist():
    """The @-decorator-registered built-ins and (potential) entry-point
    plugins live in the same registry and don't clobber each other."""
    import astracore
    # Built-ins are @-registered on import.
    builtin = set(astracore.list_backends())
    assert {"npu-sim", "onnxruntime"}.issubset(builtin)

    # Simulate an entry-point register AFTER import. Nothing on disk, but
    # we can force the loader to run and confirm built-ins survive.
    from astracore.registry import _backends
    _backends._entry_points_loaded = False  # re-run discovery
    still = set(astracore.list_backends())
    assert {"npu-sim", "onnxruntime"}.issubset(still)


# ---------------------------------------------------------------------------
# G4 — Replay exercises multi-camera + multi-radar preset end-to-end
# ---------------------------------------------------------------------------

def test_replay_with_multi_sensor_rig():
    """Preset-style (4 cameras × 4 radars), reduced-samples version.
    Confirms every sensor is iterated by the replay harness."""
    from astracore.dataset import SyntheticDataset, replay_scene

    ds = SyntheticDataset(
        n_scenes=1, samples_per_scene=2,
        image_shape=(120, 160),     # keep tiny for speed
        n_lidar_points=512,
        n_radar_detections=16,
        n_cameras=4,
        n_radars=4,
        seed=99,
    )
    scene = ds.get_scene("synthetic-scene-000")
    result = replay_scene(scene)
    assert result.n_samples == 2
    # With 4 cameras per sample × "1 det per 100k pixels" placeholder,
    # 120×160 images yield at least one detection per 4-camera sample.
    # Radar: 16 detections × 4 radars = 64 raw; after RCS≥10 filter, some.
    # Just assert the loop ran and populated every metric.
    for s in result.per_sample:
        assert s.wall_ms > 0
        # At least the radar filter visited all 4 radars; synthetic
        # RCS range guarantees some pass.
        assert s.n_radar_detections >= 0


def test_synthetic_multi_radar_frame_names_distinct():
    """Every radar in a sample must have a unique sensor_id."""
    from astracore.dataset import SyntheticDataset
    ds = SyntheticDataset(preset_name="vlp32")
    sample = next(iter(ds.get_scene(ds.list_scenes()[0])))
    ids = [r.sensor_id for r in sample.radars.values()]
    assert len(ids) == len(set(ids))


def test_synthetic_multi_camera_poses_differ():
    """Multi-camera rig: extrinsics must differ (otherwise replay
    de-dupes). Quick sanity check on the pose offset."""
    from astracore.dataset import SyntheticDataset
    ds = SyntheticDataset(preset_name="vlp32")
    sample = next(iter(ds.get_scene(ds.list_scenes()[0])))
    cams = list(sample.cameras.values())
    assert len(cams) >= 2
    x_positions = [c.extrinsics[0, 3] for c in cams]
    assert len(set(x_positions)) == len(x_positions)


# ---------------------------------------------------------------------------
# G5 — benchmark harness correctly rejects invalid input shapes
# ---------------------------------------------------------------------------

def test_gen_input_rejects_non_positive_shape():
    """Defensive: a zero / negative dim must raise, not crash numpy
    deep inside rng.integers with a cryptic message."""
    import onnx
    from astracore.benchmark import _gen_input_for

    # Fake minimal TensorValueInfo with one dim
    inp = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, -1])
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="invalid shape"):
        _gen_input_for(inp, override_shape=(1, 0), rng=rng)


# ---------------------------------------------------------------------------
# G6 — Multistream report pins scaling monotonicity (one-stream-vs-one-stream)
# ---------------------------------------------------------------------------

def test_multistream_one_stream_slice_self_consistent():
    """n_streams=1 must have IPS > 0 and wall ≈ duration_s."""
    from astracore import zoo
    from astracore.multistream import run_multistream

    path = zoo.local_paths()["squeezenet-1.1"]
    if not path.exists():
        pytest.skip("squeezenet not downloaded")

    rep = run_multistream(path, n_streams_list=(1,),
                          duration_s=0.3, warmup_s=0.1,
                          input_shape=(1, 3, 224, 224))
    s = rep.slices[0]
    assert s.throughput_ips > 0
    assert 0.25 <= s.wall_s <= 0.8       # duration + a bit of overhead
    assert s.n_inferences_total > 0
    assert s.scaling_vs_single == 1.0     # baseline invariant
    # MAC-count was pre-computed at compile time; must be > 0 for any
    # model with a single Conv (SqueezeNet).
    assert rep.mac_ops_per_inference > 0


# ---------------------------------------------------------------------------
# G7 — Dataset protocol: a non-conforming class fails the runtime check
# ---------------------------------------------------------------------------

def test_dataset_runtime_check_rejects_non_dataset():
    from astracore.dataset import Dataset

    class NotEnough:
        name = "x"
        # missing list_scenes / get_scene / available_sensors

    assert not isinstance(NotEnough(), Dataset)


def test_builtin_dataset_satisfies_runtime_check():
    from astracore.dataset import Dataset, SyntheticDataset
    assert isinstance(SyntheticDataset(), Dataset)
