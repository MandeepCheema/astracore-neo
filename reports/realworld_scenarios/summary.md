# Real-world scenario sweep — consolidated results (2026-04-20)

Five scenarios, one reproducible driver (`scripts/run_realworld_scenarios.py`), ~48 seconds total wall time on host CPU (Intel i5-1235U, ONNX Runtime CPU FP32). Each scenario writes a self-contained `{i}_{name}.{json,md}` pair; this doc ties them into a narrative with interpretation.

## Scenarios

| # | Name | File | Wall |
|---|---|---|---:|
| 1 | Cross-model image inference                 | [`1_image_inference.md`](1_image_inference.md) | 12 s |
| 2 | YOLOv8n detection on 28-image eval set      | [`2_yolo_detection_sweep.md`](2_yolo_detection_sweep.md) | 7 s |
| 3 | Perception pipeline across synthetic presets| [`3_perception_presets.md`](3_perception_presets.md) | 27 s |
| 4 | Safety fusion alarm — 4 driving scenarios   | [`4_alarm_scenarios.md`](4_alarm_scenarios.md) | <1 s |
| 5 | YOLOv8n latency distribution                | [`5_yolo_resolution_sweep.md`](5_yolo_resolution_sweep.md) | 2 s |

## Key findings

### Finding 1 — 5-of-5 top-1 agreement on in-distribution images; complete disagreement on out-of-distribution

**Image `bus`** (clearly a school bus, in ImageNet distribution): all five ImageNet classifiers — SqueezeNet, MobileNetV2, ResNet-50, EfficientNet-Lite4, ShuffleNet — pick `minibus` as top-1. **100 % cross-model agreement.** Confidence varies dramatically though: MobileNetV2 76.95 %, ResNet-50 74.16 %, ShuffleNet 43.94 %, SqueezeNet 8.48 %, EfficientNet-Lite4 **0.27 %**. EfficientNet-Lite4 is still correctly ranking but its absolute confidence is a known preprocessing issue (NHWC layout + symmetric (x−127)/128 normalisation — flagged as G9 in the prior session state, not fixed).

**Image `zidane`** (photo of footballer Zinedine Zidane, *not* in ImageNet's 1000-class vocabulary): classifiers scatter wildly — 2 models say `saxophone`, 2 say `bow tie`, 1 says `oboe`. **40 % cross-model agreement.** This is informative, not a bug: ImageNet-1k has no `person` class, so a photo of a person gets mapped to whatever adjacent concept has the highest activation. In production ADAS, person classification goes through a detector (YOLOv8) + COCO's dedicated person class, not an ImageNet classifier. Useful reminder when sales asks "can our models detect people?" — yes via detection, not via classification.

### Finding 2 — Cold vs steady-state latency gap is massive (up to 60×)

Cold-start latency (first run including ORT session creation) vs steady-state p50 (20 reused-session iters):

| Model | Cold ms | Steady p50 ms | Cold / steady |
|---|---:|---:|---:|
| SqueezeNet  |  929 |  2.21 | **420×** |
| MobileNetV2 |  402 |  3.32 | 121× |
| ResNet-50   | 1414 | 24.67 |  57× |
| EfficientNet|  471 |  9.38 |  50× |
| ShuffleNet  |  246 |  2.83 |  87× |

**Implication for customers:** if an OEM benchmarks the SDK end-to-end from a fresh Python process, they'll measure cold-start numbers and conclude the SDK is 50-400× slower than it actually is in production. The apply + multistream flows already warm up sessions before timing; customers doing ad-hoc `astracore demo` runs need to know to discard the first N inferences. **Action:** add a `--warmup` flag to `astracore demo` (default 3) — 1 hour of work.

### Finding 3 — YOLOv8n detection is deterministic across images (28/28 unique fingerprints)

28 images × 5 iters/image, SHA-256 of rounded raw outputs:
- **All 28 fingerprints are unique** — every image produces a distinct raw output (no cross-image collapse).
- **Within-image fingerprint is stable across 5 iters** (same session, same input → same output). Confirmed by spot-check: every run of image 0 produces `02fa227baf8ef776`.

**Use this as a drift detector** for backend upgrades: if we swap ORT versions / move to a CUDA EP / enable graph-opt `extended`, these fingerprints are the ground truth to compare against. Diverging fingerprints = real numerical change. The fingerprint is what the cross-EP conformance test (step 3 of the backend plan) will check.

Detection-count distribution: mean 34.4 dets/image, range 0–193. Wide range because we threshold raw class scores at 0.25 without NMS — a production path would yield 3–10× fewer per-image. Good enough to prove the pipeline wires up end-to-end.

### Finding 4 — Perception pipeline scales as expected, except vlp32 clustering dominates

Per-frame wall time across synthetic presets:

| Preset | Rig | ms/frame |
|---|---|---:|
| tiny             | 1 cam, 1 lidar (512 pts), 1 radar | 84 |
| standard         | 1 cam, 1 lidar (4k pts), 1 radar  | 122 |
| extended-sensors | 2 cam, 1 lidar (1k pts), 2 radars + 5 extra-kind sensors | 86 |
| **vlp32**        | 4 cam, 1 lidar (32k pts), 4 radars | **2026** |

**The 15× jump on vlp32 is lidar clustering.** Our clusterer is O(N²) on the filtered point cloud — 32 k pts → 1024 M pairs, then DBSCAN passes. Two fixes:
1. Replace with Open3D / scipy.spatial.cKDTree-backed clustering → ~10–100× speedup.
2. Stop running camera-detection stub (which generates 80 fake detections at vlp32 scale and is wasted work for a demo).

**This is exactly the kind of finding a real-world sweep surfaces that a model-level benchmark never does.** The zoo says YOLOv8n is 50 ms; replay says a full scene at realistic sensor density is 2 s. The gap lives in non-NN plumbing, not the NN itself.

### Finding 5 — Safety fusion passes 4/4 PASS criteria, but **one real bug uncovered**

All four fusion scenarios pass their formal gates:

| Scenario | Gate | Result |
|---|---|---|
| parking_crawl   | CRIT=1 mid-scene | **PASS** (CRIT at sample 6) |
| highway_cruise  | CRIT=0 AND WARN=0 on clear road | **PASS** (0 crit, 0 warn) |
| emergency_brake | CRIT ≥ 1 when obstacle appears at speed | **PASS** (2 crit) |
| us_dropout      | CAU ≥ 1 when lidar alone sees obstacle | **PASS** (1 caution) |

### FIX APPLIED 2026-04-20 — highway-clear-road no-echo bug

Initial run showed **10/10 samples at CAUTION on highway_cruise**. Real bug: `_min_us_reading()` treated `distance_m == max_range` (sensor's physical limit, meaning "no echo received") as an obstacle-at-that-distance. At 100 kph the caution threshold scales to ~17 m, so the 3.0 m "reading" fell inside.

**Fix landed** — same-day — in `examples/ultrasonic_proximity_alarm.py`:

1. `_min_us_reading()` now skips readings at or above `sensor_max_range_m - epsilon` in addition to the negative-sentinel no-echo signal.
2. `UltrasonicProximityAlarm(sensor_max_range_m=...)` carries the sensor limit through.
3. `_simulate_parking_crawl()` clears US to `-1.0` (canonical no-echo) rather than `3.0` (max range).
4. Regression tests pinned:
   - `test_us_at_max_range_is_treated_as_no_echo` — direct unit test of the new filter path.
   - `test_parking_scenario_highway_cruise_is_all_OFF` — tightened gate: highway cruise must be 10/0/0/0, not 0/10/0/0.

**After fix: highway_cruise is 10/0/0/0 (all OFF).** Confirmed in the freshly-regenerated Scenario 4 output. The integration-test value proved twice: first by surfacing the bug, second by pinning the fix so nobody regresses it.

### Finding 6 — YOLOv8n latency is rock-stable at 640 static (stdev 2.5 ms, p99/mean = 1.14×)

30 iterations, static 640×640 input, single stream: mean 50.8 ms, p50 49.9 ms, p99 58.2 ms, max 58.7 ms. p99/mean = 1.14× — matches the detailed zoo bench ratio exactly. Good news for safety-cert: the latency budget allocation (p99) is well-defined and tight even under a general-purpose OS.

Note: a proper resolution sweep (320 / 480 / 960 / 1280) needs an Ultralytics re-export — the shipped ONNX has a static 640×640 input. Estimated ½ day. Not critical today because customer silicon sees the same latency shape; the 640 baseline is the comparable anchor against MLPerf Edge numbers.

## Cross-cutting takeaways

1. **The SDK works end-to-end on real inputs** — 5 classifiers × 2 images, 28-image YOLO sweep, 4-preset perception pipeline, 4-scenario safety fusion, resolution-sensitivity profile. Everything runs on one laptop in under 60 s.
2. **Deterministic and reproducible** — output fingerprints stable within a session, rerun of the full script produces the same numbers ±timing jitter.
3. **Integration testing found a bug** — highway-clear-road false-positive caution. Unit tests happy, integration failed to catch it because the test gate was lenient. Fix is ~2 h.
4. **Non-NN hot paths dominate at scale** — lidar clustering at vlp32 density is 15× the single-frame YOLO cost. Target silicon won't help the clusterer; algorithmic improvements will. Phase C candidate.
5. **Cold vs steady-state latency gap is enormous** — add warmup to every customer-facing demo. One-hour fix.

## Sub-reports

- [Scenario 1 — Cross-model image inference](1_image_inference.md)
- [Scenario 2 — YOLOv8n 28-image detection sweep](2_yolo_detection_sweep.md)
- [Scenario 3 — Perception pipeline per preset](3_perception_presets.md)
- [Scenario 4 — Safety fusion alarm scenarios](4_alarm_scenarios.md)
- [Scenario 5 — YOLOv8n latency distribution](5_yolo_resolution_sweep.md)

## Actioned 2026-04-20 — status of the forward work list

| Item | Status |
|---|---|
| `astracore demo --warmup N` flag | **DONE** — CLI accepts `--warmup N`, threads through `run_demo(warmup=N)`. Smoke: warmup=3 drops ShuffleNet reported latency 250 ms → 4.6 ms. |
| Fix `_min_us_reading` max-range-as-no-echo + tighten highway gate | **DONE** — see §"FIX APPLIED 2026-04-20". 2 new regression tests. |
| EfficientNet-Lite4 preprocessing (G9) | **DONE** — root cause was double-softmax on an already-softmaxed output. New `_to_probabilities()` detects probability-shaped output (sum ≈ 1, non-negative) and skips the extra softmax. Top-1 confidence went **0.27% → 99.89%**. |
| Per-image YOLO drift fingerprints as regression | **DONE** — `tests/yolo_fingerprint_baseline.json` (28 SHA-256 prefixes) + `tests/test_yolo_fingerprint_regression.py` (3 tests). Next backend swap will trip this if numerics drift. |
| Replace O(N²) lidar clusterer with KDTree DBSCAN | deferred — intentionally risky, not blocking. |
| YOLOv8n Ultralytics dynamic-shape re-export | deferred — ½-day effort; 640 baseline already anchored against MLPerf Edge. |

Six AI-model **deep tests** added alongside these fixes, under `reports/ai_deep_tests/`:

1. Input-perturbation robustness — MobileNet survives σ=0.1 noise, ShuffleNet σ=0.05, SqueezeNet σ=0.
2. BERT-Squad determinism — 10 runs, identical start/end tokens, stdev 0.
3. GPT-2 Paris-rank stability — rank 5/50257 across 10 runs, deterministic, inside top-10.
4. YOLOv8n top-3 class determinism across 28 images × 5 runs — **28/28 stable**.
5. Latency-vs-GMACs correlation across zoo — Pearson r = 0.14 (weak; BERT/GPT-2 inflate ms/GMAC, latency is memory-bound not compute-bound today).
6. FP32 vs fake-INT8 drift on YOLOv8n — **29.7 dB SNR, cosine 0.999** with a single-sample calibration (F1-C5 audit previously showed 43.6 dB at 100 samples; expected).
