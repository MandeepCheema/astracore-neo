# Multi-stream scaling — shufflenet-v2-10.onnx on onnxruntime

`0.14 GMACs/inference`, `2.0s` per data point, `1.0s` warmup.

| Streams | IPS | TOPS (agg.) | Latency p50 (ms) | Latency p99 (ms) | Scale vs 1× | Util vs 1× |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 230.3 | 0.033 | 4.0 | 8.8 | 1.00× | 1.00× |
| 2 | 337.4 | 0.049 | 5.6 | 11.8 | 1.46× | 1.46× |
| 4 | 402.7 | 0.058 | 6.4 | 41.2 | 1.75× | 1.75× |
| 8 | 437.1 | 0.063 | 9.2 | 70.5 | 1.90× | 1.90× |
