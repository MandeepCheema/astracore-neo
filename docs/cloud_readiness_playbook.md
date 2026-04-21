# Cloud readiness playbook — what to run the moment cloud access lands

**Purpose.** We have built the multi-EP façade (Step 1 of the backend plan). The
SDK now accepts `providers: [cuda, cpu]` / `[tensorrt, cuda, cpu]` /
`[qnn, cpu]` / `[openvino, cpu]` in YAML, falls back gracefully when an EP
isn't available on the host, and reports which EPs ORT actually used. The
one thing missing is hosts with those EPs installed. This doc captures,
while the Python layer is still warm, exactly what to run when any of the
cloud accounts below goes live. Everything listed uses code already in this
repo — no new engineering is in the critical path.

> **Preconditions:** `pip install -e .` in a fresh venv on the target host,
> then verify with `astracore list eps` — the EP you care about must print
> under "Available on this host".

## Targets (in order of ease-of-access)

| Rank | Target | Provider | Cost | Auth-friction | Why first |
|---|---|---|---|---|---|
| 1 | **AWS `g5.xlarge`** (NVIDIA A10G, 24 GB) | CUDA / TensorRT | ~$1/hr | IAM + EC2 key | Biggest EP coverage matrix in one VM. Both CUDA *and* TensorRT ship in one ORT build. |
| 2 | **Intel Developer Cloud** (Xeon Sapphire Rapids + Arc GPU) | OpenVINO | free | account sign-up | OpenVINO covers Intel CPU + iGPU + Meteor Lake NPU; zero hardware spend. |
| 3 | **Qualcomm AI Hub** (`aihub.qualcomm.com`) | QNN / SNPE | free | account | Real Snapdragon DSP / NPU benchmarking, remote upload → measured numbers back. Not ORT-EP — dedicated backend work (step 5b) — but the host API returns latency + power so we can table it alongside. |
| 4 | **Seeed JetsonCloud** / **Vastai Jetson rentals** | TensorRT | ~$0.5/hr | card | Real Jetson Orin AGX / Nano. Only way to get MLPerf Edge-comparable numbers without owning a board. |
| 5 | **GCP `n1-standard-4` + T4** | CUDA | ~$0.35/hr | billing | Cheapest CUDA in cloud. Good for CI / nightly sweeps where A10G is overkill. |
| 6 | **AWS `f1.2xlarge`** (Xilinx VU9P) | custom XRT backend (Phase B) | ~$1.65/hr | already set up | Our existing F1 roadmap target. Not ORT — separate backend. |
| 7 | **Azure `ND A100 v4`** | CUDA / TensorRT | ~$27/hr | committed spend | Only if MLPerf Performance submission demands A100. |

## Per-target command recipe

All commands assume `cd astracore-neo && source .venv/bin/activate` (Linux) or
the Windows equivalent. They are copy-pasteable.

### 1. NVIDIA GPU (AWS g5 / GCP T4 / Azure NC) — CUDA + TensorRT

```bash
# ORT build that actually has CUDA + TensorRT EPs:
pip install onnxruntime-gpu==1.18.1

# Verify the right EPs showed up:
astracore list eps
# Expect: CUDAExecutionProvider [yes], TensorrtExecutionProvider [yes]

# Baseline: the full zoo on CUDA with CPU fallback for any unsupported op.
astracore zoo --iter 10 --backend onnxruntime \
    --out reports/cloud/g5_cuda/zoo.json \
    --md-out reports/cloud/g5_cuda/zoo.md
# NOTE: as of this writing, the CLI `zoo` subcommand does not expose
# --backend-options. Add `--providers cuda,cpu` wiring (<50 lines) when
# running this — or drive the bench from Python directly, e.g.
#
#   from astracore.benchmark import benchmark_model
#   benchmark_model("data/models/yolov8n.onnx", backend="onnxruntime",
#                   backend_options={"providers": ["cuda", "cpu"]},
#                   n_iter=10, warmup=3)

# Multi-stream scaling on GPU. Expect different shape vs CPU — GPUs
# saturate faster, so the 8× / 1× ratio is typically 1.2-1.8× and
# plateaus by 4 streams.
astracore multistream --model data/models/yolov8n.onnx \
    --streams 1,2,4,8,16 --duration 10.0 \
    --out reports/cloud/g5_cuda/multistream_yolo.json \
    --md-out reports/cloud/g5_cuda/multistream_yolo.md

# Apply Tier-1 YAML on CUDA. Edit the YAML's backend.options.providers
# to [cuda, cpu] first (or set up a cuda-specific YAML).
astracore configure --apply examples/tier1_adas_cuda.yaml \
    --bench-iter 10 --multistream-duration 5.0 \
    --out reports/cloud/g5_cuda/apply

# Next: TensorRT EP. Higher effort (EP pre-compiles engine, takes a
# minute per model). Expect 2-5× speedup over raw CUDA for CNNs.
#   providers: [(tensorrt, {trt_fp16_enable: true,
#                           trt_max_workspace_size: 4294967296,
#                           trt_engine_cache_enable: true,
#                           trt_engine_cache_path: "./trt_cache"}),
#               cuda, cpu]
```

**Expected sanity ranges on g5 / A10G (INT8 / FP16 path):**

| Model | CPU (ours, FP32) | g5 CUDA FP16 expected | g5 TRT INT8 expected |
|---|---:|---:|---:|
| squeezenet-1.1   | 3.3 ms | 0.3-0.6 ms | 0.1-0.2 ms |
| mobilenetv2-7    | 6.5 ms | 0.5-1.0 ms | 0.2-0.4 ms |
| resnet50-v2-7    | 31 ms  | 1.5-3.0 ms | 0.5-1.0 ms |
| yolov8n          | 86 ms  | 3-5 ms     | 1.5-2.5 ms |
| bert-squad-10    | 258 ms | 15-30 ms   | 5-10 ms |

If measurements fall outside 2× of expected range on either side, something is misconfigured (almost certainly: model fell back to CPU for some op, or TRT engine compile didn't cache).

### 2. Intel OpenVINO (Intel Developer Cloud)

```bash
pip install onnxruntime-openvino
astracore list eps   # OpenVINOExecutionProvider [yes]

# OpenVINO device selection via per-EP options:
#   providers: [(openvino, {device_type: "GPU_FP16"}), cpu]
#   providers: [(openvino, {device_type: "CPU_FP32"}), cpu]
#   providers: [(openvino, {device_type: "AUTO"}),     cpu]
# Meteor Lake NPU shows up as device_type "NPU".
```

### 3. Qualcomm AI Hub — QNN / SNPE ✅ **LIVE**

AI Hub is a REST service — not ORT — and it already works today:

```bash
pip install qai-hub
# signup at https://aihub.qualcomm.com → Settings → copy API token
qai-hub configure --api_token <TOKEN>
qai-hub list-devices   # verify auth

# Submit any zoo model (handles prep automatically):
python scripts/prep_onnx_for_ai_hub.py \
    --in  data/models/zoo/squeezenet-1.1.onnx \
    --out data/models/zoo/squeezenet-1.1.aihub.onnx
python scripts/submit_to_qualcomm_ai_hub.py \
    --model data/models/zoo/squeezenet-1.1.aihub.onnx \
    --device "QCS8550 (Proxy)"

# OR batch the full vision zoo:
python scripts/submit_zoo_to_ai_hub.py --devices "QCS8550 (Proxy)"
```

**First-run fixups we've landed** (documented in `prep_onnx_for_ai_hub.py`):
1. Opset upgrade to 13 (ONNX Zoo ships opset 7 for MobileNet, SqueezeNet, etc.)
2. IR version bump to 7
3. Strip outputs from `value_info` (ONNX spec-violating duplicate entries)
4. Remove initializers from `graph.input` (modern convention)
5. Use `input_specs` at submit time — disambiguates real inputs from initializer-entries

Without these the compile job fails with "X in initializer but not in graph input".

### 4. AWS F1 — VU9P Xilinx FPGA

Already in the Phase B roadmap (F1-F1/F2/F3). Not an ORT EP; needs our own XRT backend. When F1 hardware is live:

```bash
# Assume the F1 bitstream is in ./afi, and the F1 backend package is installed.
astracore bench --model data/models/yolov8n.onnx \
    --backend f1-xrt --iter 100
```

## Backend conformance gate — MUST ADD before claiming portability

Once there are ≥2 EPs available, we need a test that proves "same ONNX in → same numerical output out, within ±1 % SNR" across EPs. **Without this, `providers: [X, cpu]` just means "ran on some silicon" — not "gave the right answer."**

Scope: `tests/test_ep_conformance.py`:
1. For every model in the zoo whose file is on disk:
2. Run it on every EP that `ort.get_available_providers()` lists.
3. Compare outputs pairwise against the CPU baseline via SNR (20·log10(‖ref‖ / ‖ref − cand‖)).
4. Fail the test if SNR < 40 dB for any EP-vs-CPU pair (generous threshold allows FP16 cast loss; tighten to 60 dB for FP32 EPs).

Scaffold (~80 lines) is estimated at ½ day after the first GPU host is live.

## What we can test today (CPU-only) that doesn't need cloud

1. **Richer zoo bench** (see `scripts/bench_zoo_detailed.py`) — latency distribution, batch sweep, thread-count sweep, memory footprint, output fingerprint. Landed 2026-04-20, produces `reports/zoo_detailed/`.
2. **Session-options sweep** — `intra_op_num_threads` ∈ {1, 2, 4, host}, `graph_optimization_level` ∈ {basic, extended, all}. Captured in the detailed harness.
3. **Warmup characterisation** — first-run vs steady-state latency; surfaces the ORT kernel-compile cost.

These are already on the CPU host and reproduce every run. They establish the baseline the cloud runs measure against.

## Reference — one-shot "when cloud lands" script

A 5-minute smoke on a new cloud host:

```bash
astracore version
astracore list eps                                     # verify EPs
.venv/bin/python -m pytest -m "not integration" -q     # ≥1150 tests pass
astracore zoo --iter 5 --out /tmp/zoo.json             # baseline
python scripts/bench_zoo_detailed.py \
    --providers cuda,cpu \
    --out reports/cloud/$(hostname)/zoo_detailed.json  # the new scenarios
astracore configure --apply examples/tier1_adas.yaml --bench-iter 5
```

If all five steps print green, the multi-EP façade is working correctly
on that host. The numbers are then directly comparable to the CPU baseline
under `reports/benchmark_sweep/` and `reports/zoo_detailed/`.

## Things explicitly NOT to do on a first cloud run

1. Don't jump straight to TensorRT INT8. The engine compile is slow and
   the quantisation cache path needs to be set correctly; if we skip CUDA
   baseline first we have nothing to compare against when TRT misbehaves.
2. Don't submit to MLPerf on the first run. MLPerf needs a pinned config,
   a power harness, a loadgen integration. None of that is in the repo
   today. It's a separate ~2-week effort.
3. Don't parallel-run cloud + F1 + Qualcomm sweeps simultaneously. Each
   target has its own gotchas; run them sequentially so failures are
   attributable.
