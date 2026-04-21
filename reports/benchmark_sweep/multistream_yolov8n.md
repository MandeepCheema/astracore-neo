# Multi-stream scaling — yolov8n.onnx on onnxruntime

`4.37 GMACs/inference`, `3.0s` per data point, `1.0s` warmup.

| Streams | IPS | TOPS (agg.) | Latency p50 (ms) | Latency p99 (ms) | Scale vs 1× | Util vs 1× |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.7 | 0.064 | 63.3 | 109.2 | 1.00× | 1.00× |
| 2 | 18.5 | 0.081 | 106.2 | 141.3 | 1.26× | 1.26× |
| 4 | 19.8 | 0.087 | 194.5 | 336.8 | 1.35× | 1.35× |
| 8 | 21.7 | 0.095 | 348.5 | 530.3 | 1.48× | 1.48× |
