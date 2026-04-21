# AstraCore demo results (backend: onnxruntime)

Real-input inference with decoded output — proves each model actually produces sensible predictions (not just compiles).

| Model | Family | Latency (ms) | Top result |
|---|---|---:|---|
| efficientnet-lite4-11 | vision-classification | 11.3 | top-1: minibus (99.89%)  |  top-5: minibus, police van, trolleybus, minivan, passenger car |
