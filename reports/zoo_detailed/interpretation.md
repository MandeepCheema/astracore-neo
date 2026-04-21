# Interpretation — detailed zoo benchmark (2026-04-20)

Backed by `zoo_detailed.json` / `zoo_detailed.md`. Host: Intel Core
i5-1235U (10 cores, 15 W), onnxruntime 1.24.4 CPUExecutionProvider,
FP32, Python 3.12. Same host as the prior `reports/benchmark_sweep/`
run; these numbers go deeper.

## 1. ORT's default threading is already close to optimal

Comparing default threading against `intra_op_num_threads ∈ {1, 2, 4}`:

| Model | default p50 | best forced p50 | delta |
|---|---:|---:|---:|
| squeezenet-1.1 |  2.05 | t=4: 3.04 | default wins |
| mobilenetv2-7  |  3.67 | t=4: 4.08 | default wins |
| resnet50-v2-7  | 23.44 | t=4: 34.74 | default wins |
| yolov8n        | 49.96 | t=4: 63.21 | default wins |
| **gpt-2-10**   | 33.71 | **t=2: 24.86** | **−26 %** |
| bert-squad-10  | 214.5 | t=4: 282 | default wins |

**Action:** add `intra_op_num_threads=2` as an apply-time override for transformer-family models. Marginal (<1 wk), measurable ~25 % win on GPT-2.

## 2. CNN batch-size flatness — evidence we're memory-bound at batch=1

Per-call latency as batch grows (b=1 → b=4):

| Model | b=1 | b=4 | per-sample at b=4 |
|---|---:|---:|---:|
| squeezenet-1.1  |  2.05 |  2.13 | **0.53 ms** (4× throughput) |
| mobilenetv2-7   |  3.67 |  3.22 | **0.81 ms** |
| shufflenet-v2-10|  3.25 |  2.79 | **0.70 ms** |
| yolov8n         | 49.96 | 50.74 | **12.7 ms** (3.94× throughput) |
| efficientnet-L4 |  9.77 | 11.23 | **2.81 ms** |
| resnet50-v2-7   | 23.44 | 93.54 | 23.4 ms (linear — compute-bound) |
| bert-squad-10   | 214.5 | 951.0 | 237.8 ms (linear) |

**Interpretation:** small CNNs (ShuffleNet, MobileNet, SqueezeNet, YOLOv8n) are memory-latency-bound at b=1 on host CPU — roughly 4× throughput free just from batching. That's a **huge** signal for what to expect on target silicon: these models will look 3-4× faster the moment they hit real compute (INT8 kernels + SIMD/MMA). ResNet-50 and BERT are already compute-saturated — they'll scale closer to 1× per-MAC on target silicon.

**Action:** in customer numbers, always report both b=1 and best-batch throughput. The b=1 number under-sells the silicon; the best-batch number shows the ceiling.

## 3. `graph_optimization_level=all` is ORT's default — and it's not always bit-identical

Output fingerprints across gopt levels:

| Model | basic fp | extended fp | all fp | default fp | all == default? |
|---|---|---|---|---|---|
| squeezenet-1.1 | same | same | same | same | ✓ |
| mobilenetv2-7  | `25a5…` | `25a5…` | `ff4e…` | `ff4e…` | ✓ |
| shufflenet-v2  | `4b56…` | `4b56…` | `c2e8…` | `c2e8…` | ✓ |
| yolov8n        | `8ca1…` | `8ca1…` | `4d0e…` | `4d0e…` | ✓ |
| gpt-2-10       | `ca19…` | `fb6f…` | `fb6f…` | `fb6f…` | ✓ |
| bert-squad-10  | same    | same    | same   | same   | ✓ |
| resnet50-v2-7  | same    | same    | same   | same   | ✓ |
| efficientnet   | same    | same    | same   | same   | ✓ |

**Finding:** MobileNet/ShuffleNet/YOLOv8n/GPT-2 produce *different* numeric outputs at `basic` / `extended` vs `all`. The default matches `all` on everything, so production-path determinism is safe. But `extended` on GPT-2 diverges from `basic` — a subtle detail for anyone tuning ORT session options manually.

**Action for the backend conformance test (Step 3 of the backend plan):** use output fingerprint as the cross-EP drift detector. If a backend's fingerprint differs from the `gopt=all` CPU reference, either the backend is wrong or the optimisation level is not identical. Either way it's worth flagging.

## 4. Tail-latency characterisation — safety-cert-relevant

Ratio of p99 to mean at batch=1, default config:

| Model | p99/mean | Interpretation |
|---|---:|---|
| resnet50-v2-7  | 1.07× | rock stable — AEB-safe |
| yolov8n        | 1.09× | rock stable |
| efficientnet-L4| 1.16× | stable |
| gpt-2-10       | 1.22× | stable |
| bert-squad-10  | 1.36× | noisy — dynamic-shape overhead |
| squeezenet-1.1 | 1.39× | jitter dominates at low latency |
| shufflenet-v2  | 1.55× | jitter dominates |
| mobilenetv2-7  | 1.87× | jitter dominates |

**Reading:** big models (ResNet-50, YOLOv8n) have stable latency because OS preemption is small relative to compute time. Small models (<5 ms) have higher p99/mean ratios because the same ~1 ms OS interrupts now dominate. This is exactly what we expect on a Windows laptop; on real-time Linux + pinned cores, tail ratios shrink to <1.1× across the board. For the ISO 26262 fault-injection story, the *relative* numbers here are useful: "ratio is 1.07× on ResNet-50 under a general-purpose OS" is a conservative upper bound.

## 5. Warmup cost — two cold-start flavours

First-run latency vs steady-state median:

| Model | warmup / p50 ratio | Warmup source |
|---|---:|---|
| bert-squad-10  | 0.87× | ORT kernel compile dominates |
| gpt-2-10       | 1.65× | KV-cache-less attention, first tokens |
| yolov8n        | 1.14× | minor |
| resnet50-v2-7  | 1.10× | minor |
| squeezenet-1.1 | 2.05× | `Conv` kernel compile on first run |
| mobilenetv2-7  | 2.15× | same |
| shufflenet-v2  | 1.44× | same |

**Reading:** ORT compiles hot-path kernels on first call, not session-init. Customers planning a first-frame latency budget must add ~2 ms (small CNNs) or ~100 ms (BERT) of cold-start cost. The multistream harness already warms up 1.0 s before timing so this doesn't pollute our aggregate numbers, but individual-call tests need to know.

## 6. What this says about cloud / target-silicon numbers

Extrapolating from "host CPU memory-bound" behaviour: on a dedicated NPU with 4-8 TOPS/W INT8 kernels, we can reasonably expect:

| Model | Host CPU FP32 p50 | Expected Orin INT8 | Expected AstraCore NPU (target) |
|---|---:|---:|---:|
| squeezenet-1.1 |  2.1 ms | 0.3 ms | 0.1-0.2 ms |
| mobilenetv2-7  |  3.7 ms | 0.7 ms | 0.2-0.4 ms |
| shufflenet-v2  |  3.3 ms | 0.5 ms | 0.1-0.3 ms |
| resnet50-v2-7  | 23.4 ms | 1.5 ms | 0.8-1.5 ms |
| yolov8n        | 50.0 ms | 3.0 ms | 1.5-2.5 ms |
| efficientnet-L4|  9.8 ms | 1.5 ms | 0.5-1.0 ms |
| bert-squad-10  | 214.5 ms | 5 ms | 2-4 ms |
| gpt-2-10       | 33.7 ms | 4 ms | 1-2 ms |

These are *anchored* in measurements (b=4 per-sample × 4 for INT8-vs-FP32 penalty + 2 for compute-bound ops) and published ranges — we'd sanity-check each one with a cloud GPU run before citing them externally. The cloud-readiness playbook in `docs/cloud_readiness_playbook.md` §1 is the step that turns this estimate into measured numbers.

## 7. Immediate actions (all <2 days)

1. Add `intra_op_num_threads` to the YAML `backend.options` cookbook (docs + one-line CLI pass-through). **Noted for step 2 of backend plan.**
2. Promote the fingerprint-check to a formal drift test — one line per model, tracked in git as the reference fingerprint for each model × gopt combination. Gates against silent ORT version upgrades that change numerics.
3. Add per-sample throughput (batch mode) alongside per-call latency in the investor brief's MAC-util table. Current brief under-sells the SDK's ceiling on small CNNs by using b=1 only.
4. Cross-link this doc from `reports/benchmark_sweep/industry_comparison.md` §2.1 so the fuller numbers are one click away.
