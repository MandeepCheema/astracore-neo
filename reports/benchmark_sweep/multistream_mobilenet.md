# Multi-stream scaling — mobilenetv2-7.onnx on onnxruntime

`0.43 GMACs/inference`, `2.0s` per data point, `1.0s` warmup.

| Streams | IPS | TOPS (agg.) | Latency p50 (ms) | Latency p99 (ms) | Scale vs 1× | Util vs 1× |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 139.2 | 0.060 | 5.7 | 25.2 | 1.00× | 1.00× |
| 2 | 205.1 | 0.088 | 9.2 | 19.3 | 1.47× | 1.47× |
| 4 | 218.6 | 0.094 | 11.3 | 63.8 | 1.57× | 1.57× |
| 8 | 218.8 | 0.094 | 32.6 | 94.0 | 1.57× | 1.57× |
