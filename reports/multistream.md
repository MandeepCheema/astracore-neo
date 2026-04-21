# Multi-stream scaling — yolov8n.onnx on onnxruntime

`4.37 GMACs/inference`, `1.0s` per data point, `0.3s` warmup.

| Streams | IPS | TOPS (agg.) | Latency p50 (ms) | Latency p99 (ms) | Scale vs 1× | Util vs 1× |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23.6 | 0.103 | 40.1 | 54.7 | 1.00× | 1.00× |
| 2 | 28.4 | 0.124 | 69.1 | 77.2 | 1.20× | 1.20× |
