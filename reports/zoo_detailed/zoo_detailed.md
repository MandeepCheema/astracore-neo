# Detailed zoo benchmark

## Host

- Platform: win32
- CPU: Intel64 Family 6 Model 154 Stepping 4, GenuineIntel (12 cores)
- Python: 3.12.3
- onnxruntime: 1.24.4
- ORT EPs available: AzureExecutionProvider, CPUExecutionProvider
- Requested providers: `(default)`
- Wall total: 150.18s over 8 models

## Summary (base scenario: batch=1, default threads + graph-opt)

| Model | Family | GMACs | warmup ms | p50 ms | p99 ms | p99.9 ms | stdev ms | fingerprint |
|---|---|---:|---:|---:|---:|---:|---:|---|
| squeezenet-1.1 | vision-classification | 0.349 | 4.20 | 2.05 | 3.09 | 3.14 | 0.34 | `b2b2616eb8983529` |
| mobilenetv2-7 | vision-classification | 0.429 | 7.89 | 3.67 | 7.37 | 7.98 | 1.03 | `ff4ec4c029d2fb39` |
| resnet50-v2-7 | vision-classification | 4.091 | 25.86 | 23.44 | 25.21 | 25.22 | 0.87 | `feee425fbda37992` |
| efficientnet-lite4-11 | vision-classification | 1.348 | 11.93 | 9.77 | 11.59 | 11.71 | 0.85 | `a9e75ec8e543bff7` |
| shufflenet-v2-10 | vision-classification | 0.145 | 4.68 | 3.25 | 5.34 | 5.61 | 0.57 | `c2e83c17f8325d97` |
| yolov8n | vision-detection | 4.372 | 56.88 | 49.96 | 54.93 | 55.19 | 2.04 | `4d0e67ca350be102` |
| bert-squad-10 | nlp-encoder-transformer | 1.293 | 185.66 | 214.45 | 310.46 | 314.35 | 30.32 | `ad626defe99d70f3` |
| gpt-2-10 | nlp-decoder-transformer | 0.085 | 55.60 | 33.71 | 40.74 | 40.76 | 5.05 | `fb6f6da263ea651e` |

## Thread-count sensitivity (p50 ms, batch=1)

| Model | default | t=1 | t=2 | t=4 |
|---|---:|:---:|:---:|:---:|
| squeezenet-1.1 | 2.05 | 6.47 | 3.78 | 3.04 |
| mobilenetv2-7 | 3.67 | 9.05 | 4.87 | 4.08 |
| resnet50-v2-7 | 23.44 | 73.09 | 39.35 | 34.74 |
| efficientnet-lite4-11 | 9.77 | 29.65 | 16.05 | 13.60 |
| shufflenet-v2-10 | 3.25 | 4.67 | 2.90 | 2.78 |
| yolov8n | 49.96 | 102.50 | 65.67 | 63.21 |
| bert-squad-10 | 214.45 | 478.79 | 306.98 | 282.04 |
| gpt-2-10 | 33.71 | 35.21 | 24.86 | 28.01 |

## Batch-size scaling (p50 ms per-call, default threads)

| Model | b=1 | b=2 | b=4 |
|---|---:|:---:|:---:|
| squeezenet-1.1 | 2.05 | 2.17 | 2.13 |
| mobilenetv2-7 | 3.67 | 3.26 | 3.22 |
| resnet50-v2-7 | 23.44 | 46.61 | 93.54 |
| efficientnet-lite4-11 | 9.77 | 11.37 | 11.23 |
| shufflenet-v2-10 | 3.25 | 2.94 | 2.79 |
| yolov8n | 49.96 | 50.40 | 50.74 |
| bert-squad-10 | 214.45 | 450.48 | 951.02 |
| gpt-2-10 | 33.71 | 32.82 | 42.10 |

## Graph-optimization level (p50 ms, batch=1)

| Model | default | basic | extended | all |
|---|---:|:---:|:---:|:---:|
| squeezenet-1.1 | 2.05 | 4.36 | 3.92 | 2.05 |
| mobilenetv2-7 | 3.67 | 8.22 | 7.64 | 3.21 |
| resnet50-v2-7 | 23.44 | 43.07 | 41.46 | 24.85 |
| efficientnet-lite4-11 | 9.77 | 22.82 | 22.92 | 11.05 |
| shufflenet-v2-10 | 3.25 | 4.29 | 3.73 | 2.87 |
| yolov8n | 49.96 | 64.79 | 61.31 | 50.09 |
| bert-squad-10 | 214.45 | 214.18 | 225.39 | 227.64 |
| gpt-2-10 | 33.71 | 28.74 | 26.74 | 30.36 |

## Tail-latency ratios (base scenario)

Ratio of p99 to mean — how bursty the backend is. Stable runtimes are close to 1.0; jittery systems tail into 2-5×.

| Model | mean ms | p99 ms | p99/mean | max/mean |
|---|---:|---:|---:|---:|
| squeezenet-1.1 | 2.23 | 3.09 | 1.39× | 1.41× |
| mobilenetv2-7 | 3.94 | 7.37 | 1.87× | 2.04× |
| resnet50-v2-7 | 23.60 | 25.21 | 1.07× | 1.07× |
| efficientnet-lite4-11 | 10.01 | 11.59 | 1.16× | 1.17× |
| shufflenet-v2-10 | 3.44 | 5.34 | 1.55× | 1.64× |
| yolov8n | 50.17 | 54.93 | 1.09× | 1.10× |
| bert-squad-10 | 229.05 | 310.46 | 1.36× | 1.37× |
| gpt-2-10 | 33.42 | 40.74 | 1.22× | 1.22× |
