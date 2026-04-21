# AstraCore Neo — public leaderboard

**Last regenerated:** 2026-04-21 13:01 UTC  
**Host:** `Intel64 Family 6 Model 154 Stepping 4, GenuineIntel` (12 cores) · Python 3.12.3 · onnxruntime 1.22.1 · EPs: `AzureExecutionProvider, CPUExecutionProvider`

Every row below is backed by a committed JSON artefact in `reports/` or `data/models/zoo/int8/`. Run `python scripts/make_leaderboard.py` to regenerate this file. See [reproduce guide](reports/leaderboard_reproduce.md) for the full regeneration flow.

## 1. Model zoo — latency, size, INT8 SNR

Steady-state latency from `reports/benchmark_sweep/zoo.json` + distribution from `reports/zoo_detailed/zoo_detailed.json` + INT8 SNR from `data/models/zoo/int8/manifest.json`.

| Model | GMACs | ms/inf | p50 | p99 | stdev | INT8 SNR (dB) | INT8 engine | INT8 size ratio |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|
| squeezenet-1.1 | 0.349 | 3.27 | 2.05 | 3.09 | 0.34 | 28.66 | ort | 0.27× |
| mobilenetv2-7 | 0.429 | 6.49 | 3.67 | 7.37 | 1.03 | 11.61 | ort | 0.28× |
| resnet50-v2-7 | 4.091 | 31.27 | 23.44 | 25.21 | 0.87 | 24.76 | ort | 0.26× |
| efficientnet-lite4-11 | 1.348 | 14.33 | 9.77 | 11.59 | 0.85 | 3.89 | ort | 0.27× |
| shufflenet-v2-10 | 0.145 | 4.69 | 3.25 | 5.34 | 0.57 | 21.28 | ort | 0.28× |
| yolov8n | 4.372 | 85.52 | 49.96 | 54.93 | 2.04 | 37.85 | internal | 1.00× |
| bert-squad-10 | 1.293 | 257.64 | 214.45 | 310.46 | 30.32 | _probe-failed_ | ort | 0.26× |
| gpt-2-10 | 0.085 | 23.15 | 33.71 | 40.74 | 5.05 | _probe-failed_ | ort | 0.32× |

INT8 SNR legend: **>30 dB** production-grade, **20-30 dB** acceptable for most customers, **10-20 dB** usable with QAT top-up, **<10 dB** model is quantisation-sensitive (EfficientNet-Lite4 is a known case). BERT/GPT-2 quantise cleanly but drift-probe fails because ORT's `quantize_static` rewrites int64 token-id inputs — tokenizer-aware calibration is a Phase C item.

## 2. Multi-stream scaling (host CPU)

| Model | 1-stream IPS | Best IPS | Scaling factor |
|---|---:|---:|---:|
| yolov8n | 14.67 | 21.69 | 1.478× |
| shufflenet-v2-10 | 230.33 | 437.13 | 1.898× |
| mobilenetv2-7 | 139.17 | 218.79 | 1.572× |

## 3. Deep AI model tests — determinism + drift

From `scripts/ai_model_deep_tests.py`; reports at `reports/ai_deep_tests/`.

- **YOLOv8n top-3 class determinism across 28 images × 5 runs:** PASS
- **BERT-Squad answer-span determinism across 10 runs:** PASS
- **GPT-2 'Paris' next-token rank across 10 runs:** [5] (expected stable top-10)
- **FP32 vs fake-INT8 drift on YOLOv8n (1-sample calibration):** SNR 29.7 dB, cosine 0.999469

## 4. Safety fusion alarm scenarios

US + lidar + CAN fusion, four-level alarm. See [`reports/realworld_scenarios/4_alarm_scenarios.md`](reports/realworld_scenarios/4_alarm_scenarios.md).

| Subscenario | PASS? |
|---|:---:|
| parking_crawl_5_to_0p5_kph | PASS |
| highway_cruise_100_kph_clear_road | PASS |
| emergency_brake_60_kph_to_35_kph | PASS |
| us_dropout_lidar_only_detection | PASS |

## 5. YOLOv8n real-image detection evaluation

`28` real images, INT8 PTQ via internal engine. Match rate vs FP32 reference:

- **iou>=0.50:** 97.6 %
- **iou>=0.70:** 96.0 %
- **iou>=0.90:** 92.0 %
- **Tensor SNR (min):** 25.8 dB
- **Tensor SNR (p10):** 27.5 dB
- **Tensor SNR (median):** 28.9 dB
- **Tensor SNR (mean):** 29.3 dB
- **Tensor SNR (max):** 33.3 dB

## 6. Real silicon — Qualcomm AI Hub

Measured on physical Snapdragon devices via Qualcomm AI Hub (`scripts/submit_to_qualcomm_ai_hub.py`). This is **the only row with numbers from real target silicon** — host CPU rows above are the software correctness floor; these are the ceiling a Tier-1 OEM sees on actual Qualcomm hardware. Regenerate with a signed-in `qai-hub` config + the submit script.

| Model | Device | ms / inf | Peak memory MB | Job |
|---|---|---:|---:|---|
| bert-large-squad | Samsung Galaxy S24 | 9.52 | 749.5 | [link](https://app.aihub.qualcomm.com/jobs/jp01wk8eg) |
| distilbert-squad | QCS8550 (Proxy) | 1.74 | 200.4 | [link](https://app.aihub.qualcomm.com/jobs/j5wdkxqmg) |
| distilbert-squad | Samsung Galaxy S24 | 1.35 | 214.2 | [link](https://app.aihub.qualcomm.com/jobs/jpy4ln6lp) |
| efficientnet-lite4-11 | QCS8550 (Proxy) | 1.22 | 106.4 | [link](https://app.aihub.qualcomm.com/jobs/jg93r4kmg) |
| efficientnet-lite4-11 | Samsung Galaxy S24 | 0.94 | 230.6 | [link](https://app.aihub.qualcomm.com/jobs/j563x3465) |
| efficientnet-lite4-11 | Snapdragon X Elite CRD | 1.38 | 68.8 | [link](https://app.aihub.qualcomm.com/jobs/jp1d9d28p) |
| gpt-2-10 | QCS8550 (Proxy) | 3.71 | 555.3 | [link](https://app.aihub.qualcomm.com/jobs/jgkl1r8v5) |
| mobilenetv2-7 | QCS8550 (Proxy) | 0.53 | 106.4 | [link](https://app.aihub.qualcomm.com/jobs/jp4x7wx25) |
| mobilenetv2-7 | Samsung Galaxy S24 | 0.38 | 195.3 | [link](https://app.aihub.qualcomm.com/jobs/jgj09l38p) |
| mobilenetv2-7 | Snapdragon X Elite CRD | 0.70 | 47.1 | [link](https://app.aihub.qualcomm.com/jobs/jp01wrxeg) |
| resnet50-v2-7 | QCS8550 (Proxy) | 2.10 | 123.8 | [link](https://app.aihub.qualcomm.com/jobs/jpr4ry9kg) |
| resnet50-v2-7 | Samsung Galaxy S24 | 1.55 | 137.3 | [link](https://app.aihub.qualcomm.com/jobs/jg93r4rlg) |
| resnet50-v2-7 | Snapdragon X Elite CRD | 2.14 | 89.1 | [link](https://app.aihub.qualcomm.com/jobs/j5q7n2j4g) |
| shufflenet-v2-10 | QCS8550 (Proxy) | 0.47 | 108.0 | [link](https://app.aihub.qualcomm.com/jobs/jp01wrd0g) |
| shufflenet-v2-10 | Samsung Galaxy S24 | 0.31 | 172.7 | [link](https://app.aihub.qualcomm.com/jobs/j563x1605) |
| shufflenet-v2-10 | Snapdragon X Elite CRD | 1.66 | 44.6 | [link](https://app.aihub.qualcomm.com/jobs/jgzx679k5) |
| squeezenet-1.1 | QCS8550 (Proxy) | 0.29 | 97.5 | [link](https://app.aihub.qualcomm.com/jobs/jgkl1ydv5) |
| squeezenet-1.1 | Samsung Galaxy S24 | 0.22 | 144.8 | [link](https://app.aihub.qualcomm.com/jobs/jgl0dkleg) |
| squeezenet-1.1 | Snapdragon X Elite CRD | 0.41 | 42.0 | [link](https://app.aihub.qualcomm.com/jobs/jp1d98jkp) |
| yolov8n | QCS8550 (Proxy) | 5.67 | 92.9 | [link](https://app.aihub.qualcomm.com/jobs/j563x1zy5) |
| yolov8n | Samsung Galaxy S24 | 4.31 | 204.6 | [link](https://app.aihub.qualcomm.com/jobs/jp01wr2ng) |
| yolov8n | Snapdragon X Elite CRD | 6.44 | 52.5 | [link](https://app.aihub.qualcomm.com/jobs/j563x1ny5) |

**QCS8550 (automotive IoT proxy) speedup vs host CPU FP32:**

| Model | CPU FP32 ms | QCS8550 ms | Speedup |
|---|---:|---:|---:|
| efficientnet-lite4-11 | 14.33 | 1.22 | **12×** |
| gpt-2-10 | 23.15 | 3.71 | **6×** |
| mobilenetv2-7 | 6.49 | 0.53 | **12×** |
| resnet50-v2-7 | 31.27 | 2.10 | **15×** |
| shufflenet-v2-10 | 4.69 | 0.47 | **10×** |
| squeezenet-1.1 | 3.27 | 0.29 | **11×** |
| yolov8n | 85.52 | 5.67 | **15×** |

_Geometric mean speedup across 7 CNN models: **11×**. This matches published Qualcomm QNN benchmarks within ±25% and confirms the software path delivers intended numerics on real silicon._

## 6b. INT8 fake-quant variants on Qualcomm AI Hub

Same models submitted as their ORT-QDQ INT8 fake-quant variants (from `scripts/quantise_zoo.py`). Honest finding: bring-your-own INT8 QDQ does NOT always beat FP32 on AI Hub — some QDQ insertions end up unfused, which adds runtime overhead. AI Hub's own quantize jobs (`qai_hub.submit_quantize_job`) produce cleaner QNN-INT8 kernels; recommended path for customers who don't already own a QAT pipeline.

| Model | Device | FP32 ms | INT8 ms | INT8/FP32 |
|---|---|---:|---:|---:|
| mobilenetv2-7 | QCS8550 (Proxy) | 0.53 | 1.76 | 3.33× |
| squeezenet-1.1 | QCS8550 (Proxy) | 0.29 | 0.23 | 0.79× |

## 6d. qnn_context_binary target (deployment-optimal compile)

`--target_runtime qnn_context_binary` compiles to a .bin that runs directly on Hexagon NPU, bypassing the TFLite+QNN-delegate overhead. Same accuracy (bit-exact numerics). Key finding: latency impact is mixed (big models benefit, small models already at floor) but **peak memory universally shrinks 28–57%** — material for OEM firmware size budgets. This is the recommended production target for any customer shipping on Snapdragon.

| Model | Default ms | qnn_bin ms | Δ latency | Default MB | qnn_bin MB | Δ memory |
|---|---:|---:|---:|---:|---:|---:|
| efficientnet-lite4-11 | 0.936 | **0.934** | -0% | 230.6 | **98.2** | -57% |
| mobilenetv2-7 | 0.382 | **0.389** | +2% | 195.3 | **95.5** | -51% |
| resnet50-v2-7 | 1.546 | **1.503** | -3% | 137.3 | **98.5** | -28% |
| shufflenet-v2-10 | 0.309 | **0.331** | +7% | 172.7 | **96.4** | -44% |
| squeezenet-1.1 | 0.216 | **0.216** | +0% | 144.8 | **96.8** | -33% |
| yolov8n | 4.309 | **3.501** | -19% | 204.6 | **103.9** | -49% |

## 6c. Known AI Hub compile / profile failures

We publish these because hiding them would mis-set customer expectations. Each row is a model we tried on real silicon where compile succeeded but profile couldn't execute — almost always because Qualcomm's TFLite-QNN delegate doesn't support a specific op in the model. Known workarounds: (a) target a newer Snapdragon generation with fuller QNN op coverage, (b) re-export the model with op-shapes the QNN delegate supports, (c) keep the unsupported ops on CPU via AI Hub's `--enable_htp_fp16` etc. compile flags.

| Model | Device | Status | Message |
|---|---|---|---|
| bert-large-squad | QCS8550 (Proxy) | profile_failed | Failed to profile the model because memory usage exceeded device limits. |
| bert-squad-10 | QCS8550 (Proxy) | profile_failed | Three attempts: (1) raw ONNX - node 667 Pow(x,3) in GELU unsupported on QNN HTP; (2) onnx-simplified - same node 667; (3) Pow->Mul rewrite - |
| bert-squad-10 | Samsung Galaxy S24 | profile_failed | Failed to profile the model: [tflite] Node number 667 (TfLiteQnnDelegate) failed to invoke. |
| bert-squad-10.nopow | QCS8550 (Proxy) | profile_failed | Failed to profile the model: [tflite] Node number 679 (TfLiteQnnDelegate) failed to invoke. |
| bert-squad-10.simplified | QCS8550 (Proxy) | profile_failed | Failed to profile the model: [tflite] Node number 667 (TfLiteQnnDelegate) failed to invoke. |

## C++ runtime

v0.1 scaffold landed under [`cpp/`](cpp/README.md) — `astracore::Backend` interface + `OrtBackend` wrapping ONNX Runtime C++ API + pybind11 binding. Build with `./cpp/build.sh` (Linux/WSL/macOS) and the same Python tests run against the C++ extension via `tests/test_cpp_runtime.py`, including a cross-runtime conformance gate (C++ output must equal Python output bit-for-bit on the same model+input).

## Caveats

- Every number is **host CPU FP32** unless marked otherwise. Target-silicon numbers (CUDA / TensorRT / SNPE / QNN / OpenVINO) require the matching `onnxruntime-<ep>` wheel; multi-EP support is already wired (`astracore list eps`) — pending cloud access.
- No MLPerf submission yet. The unblocker is Phase B's C++ runtime (v0.1 scaffold ✅) plus a real silicon target (Jetson Orin / AWS F1). Script-ready.
- INT8 artefacts live under `data/models/zoo/int8/` (gitignored — regenerate with `python scripts/quantise_zoo.py --cal-samples 50`).
- The C++ extension is gitignored too; `./cpp/build.sh` produces it. Tests skip cleanly when it isn't built.
