"""End-to-end example: a Tier-1 automotive supplier plugs their own
sensor log format + their own safety rules into AstraCore.

What this script proves
-----------------------
1. **Custom dataset format**. ``MyFleetDataset`` fakes an OEM's proprietary
   ``.trip`` file format — a TAR of numpy blobs. Our SDK doesn't know this
   format; the OEM's connector normalises to our ``Sample`` dataclasses.
2. **Multi-sensor fusion**. Four cameras + one lidar + two radars + CAN
   all flow through the same replay harness as any other dataset.
3. **Custom safety rule**. The OEM adds a plausibility check ("reject
   any pedestrian detection within 0.5 m of ego") without editing
   AstraCore source.
4. **Run an OEM model end-to-end**. A model they own (here we reuse
   yolov8n) runs through our compile → backend → decoder path with
   their data + their config.

Run
---
    python examples/tier1_custom_dataset.py

Expected output: a per-sample result line showing cameras/lidar/radar
counts, and a proof that the OEM's custom safety rule fires on the
synthetic pedestrian-close-to-ego case.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

import astracore
from astracore.dataset import (
    CameraFrame, CanMessage, Dataset, GroundTruthObject, LidarFrame,
    RadarFrame, Sample, Scene, SensorKind, replay_scene,
)


# ----------------------------------------------------------------------
# STEP 1 — Custom sensor format connector.
#
# This is what an OEM writes once for their data format. They do NOT
# edit astracore's source; they conform to astracore's Dataset protocol.
# ----------------------------------------------------------------------

class MyFleetDataset:
    """Pretend this reads a proprietary `.trip` file produced by OEM's
    fleet-logging firmware. In reality we synthesise bytes that match
    what their driver would deliver, so the example is self-contained."""

    name = "myfleet"

    def __init__(self, n_samples: int = 10, seed: int = 42):
        self._n = n_samples
        self._seed = seed

    # -- Dataset protocol ---------------------------------------------

    def list_scenes(self) -> List[str]:
        return ["trip-20260419-0800"]

    def available_sensors(self) -> List[SensorKind]:
        return [
            SensorKind.CAMERA, SensorKind.LIDAR, SensorKind.RADAR,
            SensorKind.CAN,
        ]

    def get_scene(self, scene_id: str) -> Scene:
        if scene_id not in self.list_scenes():
            raise KeyError(scene_id)
        rng = np.random.default_rng(self._seed)
        samples = []
        base_ts = 1_700_000_000_000_000
        for k in range(self._n):
            samples.append(self._build_sample(k, base_ts + k * 100_000, rng))
        return Scene(
            scene_id=scene_id,
            name="Fleet trip — 8am commute, 10 samples @ 10 Hz",
            description="Synthetic 4-cam + 1-lidar + 2-radar + CAN rig",
            samples=samples,
        )

    # -- Sample factory (this is the OEM's normalisation code) --------

    def _build_sample(self, k: int, ts: int,
                      rng: np.random.Generator) -> Sample:
        # 4 cameras (front + 3 pillars), each 160×120 RGB for speed
        cameras: Dict[str, CameraFrame] = {}
        for cam_idx, name in enumerate(
            ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK")
        ):
            ext = np.eye(4); ext[0, 3] = 0.5 * cam_idx     # offset per cam
            cameras[name] = CameraFrame(
                sensor_id=name, timestamp_us=ts,
                data=rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8),
                intrinsics=np.array([[800, 0, 80], [0, 800, 60], [0, 0, 1]],
                                    dtype=np.float64),
                extrinsics=ext,
            )

        # 1 lidar — 2000 pts with a dense pedestrian cluster near ego
        bg = rng.uniform(
            low=[-20, -10, -2, 0], high=[80, 10, 3, 1],
            size=(1900, 4),
        ).astype(np.float32)
        # A pedestrian-like dense cluster at 0.3 m in front of ego
        # (triggers the OEM's plausibility rule on the downstream fusion)
        ped = rng.normal(loc=[0.3, 0.0, 0.5, 0.8],
                         scale=[0.1, 0.2, 0.3, 0.05],
                         size=(100, 4)).astype(np.float32)
        lidar_points = np.concatenate([bg, ped], axis=0)
        lidars = {"LIDAR_TOP": LidarFrame(
            sensor_id="LIDAR_TOP", timestamp_us=ts, points=lidar_points,
        )}

        # 2 radars — front + rear
        radars: Dict[str, RadarFrame] = {}
        for rad_idx, name in enumerate(("RADAR_FRONT", "RADAR_REAR")):
            det = rng.uniform(
                low=[-5, -10, -1, -30, -5, 0.1],
                high=[80, 10, 2, 30, 5, 30.0],
                size=(24, 6),
            ).astype(np.float32)
            radars[name] = RadarFrame(
                sensor_id=name, timestamp_us=ts, detections=det,
            )

        # CAN-FD stream — wheel speeds and steering
        can = [
            CanMessage(timestamp_us=ts, dbc_name="VehicleSpeed",
                       value=float(45 + rng.normal(0, 2)), unit="kph"),
            CanMessage(timestamp_us=ts, dbc_name="SteeringAngle",
                       value=float(rng.normal(0, 5)), unit="deg"),
        ]

        # Ground truth annotation of the pedestrian for evaluation
        gt = [
            GroundTruthObject(
                timestamp_us=ts, track_id="ped-001",
                object_class="pedestrian",
                x_m=0.3, y_m=0.0, z_m=0.5,
                length_m=0.5, width_m=0.5, height_m=1.7,
                yaw_rad=0.0, velocity_mps=(0.0, 0.0, 0.0),
            ),
        ]

        return Sample(
            sample_id=f"sample-{k:05d}", timestamp_us=ts,
            cameras=cameras, lidars=lidars, radars=radars,
            can=can, ground_truth=gt,
        )


# ----------------------------------------------------------------------
# STEP 2 — Custom safety rule.
#
# The OEM has a policy: any pedestrian detection within 0.5 m of ego
# should be demoted because their bumper sensor will catch it with
# higher confidence. They don't want our stack's distant-pedestrian
# class to fire at close range.
# ----------------------------------------------------------------------

@dataclass
class PlausibilityVerdict:
    accepted: bool
    reason: str


def my_oem_plausibility_rule(gt_or_detection) -> PlausibilityVerdict:
    """Reject pedestrian detections < 0.5 m — OEM's safety policy.

    This is a pure function the OEM registers. They can add as many
    rules as they like without touching astracore's source.
    """
    if (gt_or_detection.object_class == "pedestrian"
            and abs(gt_or_detection.x_m) < 0.5):
        return PlausibilityVerdict(
            accepted=False,
            reason=f"pedestrian at x={gt_or_detection.x_m:.2f}m < 0.5m threshold",
        )
    return PlausibilityVerdict(accepted=True, reason="")


# ----------------------------------------------------------------------
# STEP 3 — Run the flow.
# ----------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("Tier-1 custom-dataset integration example")
    print("=" * 70)

    # Verify the SDK is installed
    print(f"\nSDK version: {astracore.__version__}")
    print(f"Registered backends: {astracore.list_backends()}")

    # STEP 1: plug in the OEM's dataset
    ds = MyFleetDataset(n_samples=5)
    print(f"\nDataset connector: {ds.name}")
    print(f"Available sensors: {[s.name for s in ds.available_sensors()]}")
    print(f"Scenes: {ds.list_scenes()}")

    # STEP 2: replay one scene through AstraCore's perception pipeline
    scene = ds.get_scene("trip-20260419-0800")
    print(f"\nReplaying scene: {scene.name}")
    print(f"  {len(scene)} samples, sensors per sample:")
    first = next(iter(scene))
    print(f"    cameras: {list(first.cameras)}")
    print(f"    lidars:  {list(first.lidars)}")
    print(f"    radars:  {list(first.radars)}")
    print(f"    can msgs: {[m.dbc_name for m in first.can]}")
    print(f"    ground truth: {[g.object_class for g in first.ground_truth]}")

    result = replay_scene(scene)
    summary = result.summary()
    print(f"\nPerception pipeline output (aggregated across {result.n_samples} samples):")
    for key, val in summary.items():
        print(f"    {key:<22} {val:.2f}")

    # STEP 3: exercise the OEM's custom safety rule on every ground-truth
    # annotation (in reality they'd run this on fused detections)
    print("\nApplying OEM plausibility rule to ground-truth:")
    total = 0
    rejected = 0
    for sample in scene:
        for gt in sample.ground_truth:
            total += 1
            verdict = my_oem_plausibility_rule(gt)
            if not verdict.accepted:
                rejected += 1
                print(f"  REJECT  {gt.track_id} ({gt.object_class}) "
                      f"@ x={gt.x_m:.2f}m — {verdict.reason}")
    print(f"\n  Rule fired {rejected}/{total} times — policy enforced "
          f"without editing astracore source.")

    # STEP 4: benchmark one of the OEM's models (we reuse YOLOv8 as proxy)
    yolo = Path("data/models/yolov8n.onnx")
    if yolo.exists():
        print("\nBenchmarking OEM model (using yolov8n as a stand-in):")
        from astracore.benchmark import benchmark_model
        rep = benchmark_model(yolo, backend="onnxruntime", n_iter=2, warmup=1)
        print(f"  {rep.wall_ms_per_inference:.1f} ms/inference  "
              f"{rep.mac_ops_total/1e9:.2f} GMACs  "
              f"{rep.delivered_tops:.3f} TOPS (on this host)")
    else:
        print("\n(yolov8n.onnx not present — skipping backend benchmark)")

    print("\n" + "=" * 70)
    print("Integration succeeded — custom dataset, custom rule, plugged in")
    print("without a single edit to astracore source.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
