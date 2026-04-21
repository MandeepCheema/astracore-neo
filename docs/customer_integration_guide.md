# AstraCore SDK — Customer Integration Guide

**Audience.** Tier-1 automotive supplier integrating AstraCore's software on their own silicon + their own sensor rig + their own target-vehicle ECU.

**Contract summary.** AstraCore provides the **logical layer** (preprocessing, model execution, fusion, safety algorithms). The customer provides the **physical layer** (sensor drivers, target silicon, vehicle integration). The SDK's plugin surface exists to make the boundary clean.

---

## 1. The three-way contract

```
┌────────────────────────────────────────────────────────────────────┐
│  YOUR SILICON (Qualcomm Ride / Nvidia Orin / Ambarella / custom)   │
│  • physical sensor interfaces (MIPI CSI-2, CAN-FD, Ethernet, PCIe) │
│  • memory / DMA / compute                                          │
│  • OS + drivers                                                    │
└───────────────────────────┬────────────────────────────────────────┘
                            │  raw sensor bytes in, decisions out
┌───────────────────────────▼────────────────────────────────────────┐
│  YOUR SHIMS (you write these — ~500-2000 lines per integration)    │
│  • Dataset connector for your sensor format                        │
│  • Backend for your target silicon (if not using ours)             │
│  • CAN DBC parser for your vehicle                                 │
│  • Plausibility rules for your OEM's safety policies               │
└───────────────────────────┬────────────────────────────────────────┘
                            │  astracore public API
┌───────────────────────────▼────────────────────────────────────────┐
│  ASTRACORE SDK                                                     │
│  • astracore.dataset   — sensor-frame abstractions                 │
│  • astracore.benchmark — model loading + compile + run             │
│  • astracore.backend   — target silicon abstraction                │
│  • astracore.quantiser — INT8/INT4/INT2 calibration                │
│  • astracore.demo      — decoded output per model family           │
│  • astracore.multistream — MAC-util at 4-8 concurrent streams      │
│  • src/perception      — camera ISP, lidar clustering, radar DSP   │
│  • src/dms             — gaze + head-pose + attention tracking     │
│  • src/safety          — TMR voter, ECC, safe-state FSM patterns   │
└────────────────────────────────────────────────────────────────────┘
```

Everything above the SDK line is the customer's. Everything below is ours.

---

## 2. Sensor capability matrix (honest status)

Legend: ✅ first-class | 🟡 partial | ❌ not supported

| Sensor | Status | What we do | What you provide | Gaps |
|---|---|---|---|---|
| **Camera** (single) | ✅ | ISP pipeline (`src/perception/camera.py`): Bayer → RGB, white balance, gamma, tone. Per-frame API via `CameraFrame` dataclass | Driver that delivers YUV420/NV12/RGB888 frames + intrinsics + extrinsics | No stereo disparity; no rolling-shutter correction |
| **Camera** (multi-cam rig, up to 8) | ✅ | Multi-cam sample (`Sample.cameras: Dict[str, CameraFrame]`). Synthetic preset goes 1/4/8 cameras (`PRESETS["robotaxi"]`) | Extrinsic calibration per camera | No cross-cam feature matching / SfM |
| **Lidar** (point cloud) | ✅ | Range filter + ground removal + DBSCAN clustering (`src/perception/lidar.py`). Up to 130k points per sweep in `vlp64` preset | Driver that delivers `(N, 4)` [x, y, z, intensity] in ego frame | O(N²) DBSCAN is fine at 10k pts, slow at 100k+ |
| **Radar** (detections) | ✅ | Per-detection filtering, Doppler velocity, fusion gating | `(N, 6)` detections [x, y, z, vx, vy, rcs_dBsm] | We don't do CFAR on raw IQ — assume your radar outputs detections |
| **Radar** (raw IQ / ADC cubes) | 🟡 | `RadarFrame.adc_cube` field + range-Doppler processor skeleton (`src/perception/radar.py`) | `(chirps, samples)` complex64 ADC tensor | FFT chain is reference-only; swap in your DSP |
| **IMU** (6-DoF) | ✅ | `ImuSample` dataclass: accel [m/s²] + gyro [rad/s] | Driver at 100-1000 Hz | No filter (Kalman/complementary) — you bring your own |
| **GNSS** | ✅ | `GnssSample`: lat / lon / alt / heading | NMEA-parsed or raw | No RTK/PPK corrections; no ionosphere modelling |
| **CAN-FD** | ✅ | `CanMessage` stream, `canfd_controller` RTL reference | DBC-parsed messages (you parse DBC, we consume semantic signals) | No DBC parser included; use python-can |
| **V2X** | 🟡 | `src/connectivity/v2x.py` Python skeleton (BSM / SPaT / MAP / PSM message types) | Cellular modem + J2735 decoder | No real J2735/DSRC stack — bring your own |
| **Ethernet** | ✅ | Transport layer (`src/connectivity/ethernet.py`) | Driver | Not a sensor pipeline — just data transport |
| **PCIe** | ✅ | Host↔accelerator link reference | — | — |
| **Ultrasonic** (parking) | ✅ | `UltrasonicSample` (distance + SNR + pulse width); 12 positions across bumpers | Driver delivering distance-per-sensor at 20-50 Hz | No multi-echo parsing yet — first echo only |
| **Microphone / audio** | ✅ | `MicrophoneFrame` (PCM int16/float32, mono/multi-channel) | Audio stream at 16/48 kHz | No ASR model shipped; bring your own (wav2vec/whisper ONNX) |
| **Thermal / IR camera** | ✅ | `ThermalFrame` (raw counts / kelvin / celsius, LWIR/MWIR/NIR) separate from CameraFrame — thermal pipelines have their own NUC/AGC | Driver delivering uint16 raw or calibrated temperatures | No NUC calibration in code — assumes sensor outputs corrected |
| **Event camera** (Prophesee / DVS) | ✅ | `EventFrame` (N, 4) [x, y, t_us, polarity] sparse event lists | Integration window (~1-10 ms) | No event-to-frame reconstruction; feed direct to custom backend |
| **Time-of-Flight depth** | ✅ | `DepthFrame` (H, W) float32 meters + optional confidence | Depth map with optional confidence | No IMU-coupled motion correction |

---

## 3. What the customer CAN do today

- ✅ **Plug in their own sensor data format** by subclassing `astracore.dataset.Dataset` — the `Scene`/`Sample` contract is dataset-agnostic
- ✅ **Run their own ONNX models** through our compiler/quantiser/runtime via `astracore bench --model their_model.onnx`
- ✅ **Register custom ONNX ops** for architectures we haven't seen — `@astracore.register_op("MyCustomLayer")`
- ✅ **Plug in a custom backend** for their target silicon — `@astracore.register_backend("their-silicon")`
- ✅ **Override quantiser calibration** — `@astracore.register_quantiser("their-method")`
- ✅ **Replay scenarios** through the full perception + fusion pipeline — `astracore replay`
- ✅ **Measure aggregate MAC utilisation** under concurrent streams — `astracore multistream`
- ✅ **Verify decoded output quality** on their workload — `astracore demo --model X --input Y`
- ✅ **Use the reference sensor-fusion + DMS + safety Python** as executable specs to port to their firmware
- ✅ **Extend the synthetic dataset** with their own presets for HIL / regression testing

## 4. What the customer CANNOT do today (honest)

- ❌ **Deploy C++ firmware** — our runtime is Python today. C++ port is Phase B (4-6 weeks).
- ❌ **Use arbitrary non-ONNX models** — PyTorch / TVM / MLIR / XLA / NNEF all go through an ONNX hop. If their model can't export to ONNX, they need to.
- ✅ **Configure the whole stack via one YAML file** — see `examples/tier1_adas.yaml`. Validate with `astracore configure --validate your_config.yaml`. Schema covers sensors, models, backend, safety policies, multi-stream, dataset source.
- ❌ **Use BERT / GPT-2 / LLaMA with English prompts end-to-end** — no tokeniser ships with the SDK. Install `transformers` and wire it in, or use our canned-token demos.
- ❌ **Quantise-then-tune with QAT for 8:1 sparsity** — this is Phase C. Today: magnitude pruning at high sparsity ratios collapses accuracy (`reports/pruning_accuracy.json`).
- ❌ **Run on AWS F1 FPGA without C++ runtime** — see Phase B.
- ❌ **Get ISO 26262 ASIL-D certification artefacts** — we have RTL primitives (TMR, ECC, safe-state) but no FMEDA + formal certification package.
- ❌ **Run on bare metal without a Python runtime** — OEMs deploying on a microcontroller without Python need to wait for C++ or port manually.

---

## 5. Configuration surface

Two paths, depending on the customer's style:

- **YAML (declarative)** — `examples/tier1_adas.yaml` is a realistic 4-camera + 1-lidar + 6-radar + 12-ultrasonic + mic + thermal + event + ToF + dual-CAN + GNSS + IMU config. Validate with:

  ```bash
  astracore configure --validate examples/tier1_adas.yaml
  astracore configure --dump     examples/tier1_adas.yaml   # round-trip
  ```

  Schema lives in `astracore/config.py` — includes cross-validation (duplicate sensor names, dangling `input_sensor` references, unknown precision/sparsity, schema version).

- **Programmatic (subclass + register)** — for plugins that go beyond what YAML can express (custom ops, custom backends, custom quantisers, custom Dataset connectors). Described below.

### 5.1 Dataset connector (per customer data format)

```python
from astracore.dataset import Dataset, Scene, Sample, CameraFrame, LidarFrame, RadarFrame

class MyOEMDataset:
    """Custom connector for <OEM>'s internal log format."""
    name = "my-oem-format"

    def __init__(self, path_to_logs: str):
        self._path = path_to_logs
        # ... index the logs, build scene table

    def list_scenes(self):
        return [...]   # your scene IDs

    def available_sensors(self):
        from astracore.dataset import SensorKind
        return [SensorKind.CAMERA, SensorKind.LIDAR, SensorKind.CAN]

    def get_scene(self, scene_id: str) -> Scene:
        # Read your binary, normalize to our CameraFrame / LidarFrame / ...
        samples = []
        for timestamp, payload in self._read_scene(scene_id):
            samples.append(Sample(
                sample_id=f"{scene_id}-{timestamp}",
                timestamp_us=timestamp,
                cameras={"CAM_FRONT": CameraFrame(...your normalization...)},
                lidars={"LIDAR_TOP": LidarFrame(...)},
                # ...
            ))
        return Scene(scene_id=scene_id, name=..., samples=samples)
```

### 5.2 Backend (per target silicon)

```python
from astracore import register_backend, BackendReport
import numpy as np

@register_backend("qualcomm-sa8650")
class QualcommBackend:
    name = "qualcomm-sa8650"
    silicon_profile = "qualcomm-sa8650-200tops"

    def compile(self, onnx_model, *, precision="INT8", sparsity="dense"):
        # Use SNPE / QNN SDK to compile ONNX → HTP-native format
        import snpe
        dlc = snpe.compile(onnx_model, precision)
        return {"dlc": dlc, "hw_runtime": snpe.create_runtime(dlc)}

    def run(self, program, inputs):
        # Execute on Hexagon HTP DSP
        return program["hw_runtime"].infer(inputs)

    def report_last(self):
        # Fill in MAC util from HW counters
        return self._last
```

### 5.3 Safety rules (plausibility checker, TMR policies)

```python
from src.perception.fusion import PlausibilityRule  # reference layout

class MyOEMRule(PlausibilityRule):
    """Our fleet's specific ADAS policy: reject detections of pedestrians
    within 0.5m of ego because our bumper sensor should catch those."""

    def evaluate(self, detection, context) -> bool:
        if detection.class_name == "pedestrian" and detection.distance_m < 0.5:
            return False       # mark implausible — demote in fusion
        return True
```

---

## 6. Integration checklist (what a customer needs to do)

| # | Step | Effort | Checkpoint |
|---|---|---|---|
| 1 | `pip install astracore-sdk` | 5 min | `astracore version` returns `0.1.0` |
| 2 | Subclass `Dataset` for your sensor logs | 1-2 days | `astracore replay --dataset your-module --scene X` produces tracks |
| 3 | Register backend for your silicon (skip if using ours for now) | 1-2 weeks | `astracore bench --backend your-silicon --model your_model.onnx` produces TOPS numbers |
| 4 | Port your ONNX models through the quantiser | 2-3 days per model | INT8 accuracy within 2% of FP32 on your eval set |
| 5 | Wire DMS / fusion / safety Python reference into your firmware | 2-4 weeks | HIL scenario replay shows correct outputs |
| 6 | Expand to multi-stream (per-camera parallel pipeline) | 1-2 weeks | `astracore multistream` aggregate TOPS scales 3-6× at 4-8 streams |
| 7 | QAT + sparsity for performance envelope (requires Phase C deliverable from us) | — | ≤ 2% mAP drop at 2:4 / 4:1 sparsity on your models |

Total: ~6-10 weeks for first integration once Phase B (C++ runtime) ships.

---

## 7. End-to-end example — Tier-1 with 4-camera + lidar + radar + CAN

See `examples/tier1_custom_dataset.py` for a runnable version of this.

```python
# 1. Your sensor format — here we fake a "proprietary .trip file"
from pathlib import Path
from astracore.dataset import Dataset, Scene, Sample, CameraFrame, LidarFrame, RadarFrame

class MyFleetDataset:
    name = "myfleet"
    def __init__(self, data_dir): self._dir = Path(data_dir)
    def list_scenes(self): return ["trip-20260419-0800"]
    def available_sensors(self):
        from astracore.dataset import SensorKind
        return [SensorKind.CAMERA, SensorKind.LIDAR, SensorKind.RADAR, SensorKind.CAN]
    def get_scene(self, scene_id):
        # ... read your .trip files, normalise to our Sample dataclasses
        samples = [...]
        return Scene(scene_id=scene_id, name="8am drive", samples=samples)

# 2. Replay through our fusion pipeline
from astracore.dataset import replay_scene
ds = MyFleetDataset("/data/myfleet")
result = replay_scene(ds.get_scene("trip-20260419-0800"))
print(result.summary())

# 3. Run the customer's model through our compiler
from astracore.benchmark import benchmark_model
rep = benchmark_model("/models/my_detector.onnx", backend="qualcomm-sa8650")
print(rep.as_markdown_row())

# 4. Measure multi-stream scaling for their 4-camera rig
from astracore.multistream import run_multistream
scaling = run_multistream("/models/my_detector.onnx", n_streams_list=(1, 2, 4))
```

---

## 8. Where the RTL fits (or doesn't)

If the customer **uses their own silicon**, they don't need our RTL. They need:
- Our **compiler + quantiser + runtime** (software)
- Our **sensor fusion + DMS + safety Python reference** (to port to their firmware)
- Our **backend plugin** for their silicon (they write it, or we partner)

If the customer **wants a reference for their own chip's NPU**, they can look at `rtl/npu_pe/` + `rtl/npu_systolic_array/` — but those are internal reference designs, not IP licences.

If the customer **wants ASIL-D sensor-fusion IP**, the 20 fusion RTL modules under `rtl/` could be licensed separately — see business discussion in `docs/software_first_execution_plan.md`.

---

## 9. Open gaps the customer would flag in a DD review

| Gap | Severity | Path to close |
|---|---|---|
| No C++ runtime | HIGH | Phase B (4-6 weeks) |
| No YAML config (everything programmatic) | MED | 1-2 weeks new WP |
| No tokeniser for BERT/GPT text demos | MED | Ship a tiny BPE + WordPiece; or declare transformers dependency |
| Ultrasonic / audio / event-camera not supported | MED | Custom frame subclass; OEM writes |
| No ASIL-D certification package | HIGH | Phase B with Cadence Jasper + formal proofs |
| No QAT pipeline (8:1 sparsity claim unbacked) | HIGH | Phase C (4-5 weeks) |
| No direct customer silicon measured on real hardware | HIGH | Phase B customer-silicon backends (TensorRT / SNPE) |

All seven are in the known backlog — no silent gaps here.
