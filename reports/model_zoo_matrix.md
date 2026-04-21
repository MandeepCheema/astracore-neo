# AstraCore model-zoo benchmark (backend: onnxruntime)

`n_iter=2`, wall=8.1s, 8 models.

| Model | Latency (ms) | GMACs | Delivered TOPS | Notes |
|---|---|---|---|---|
| squeezenet-1.1 | 2.35 | 0.35 | 0.139 | ORT baseline; not a target-silicon measurement |
| mobilenetv2-7 | 3.41 | 0.43 | 0.133 | ORT baseline; not a target-silicon measurement |
| resnet50-v2-7 | 23.44 | 4.09 | 0.198 | ORT baseline; not a target-silicon measurement |
| efficientnet-lite4-11 | 10.22 | 1.35 | 0.155 | ORT baseline; not a target-silicon measurement |
| shufflenet-v2-10 | 2.60 | 0.14 | 0.063 | ORT baseline; not a target-silicon measurement |
| yolov8n | 41.42 | 4.37 | 0.103 | ORT baseline; not a target-silicon measurement |
| bert-squad-10 | 169.19 | 1.29 | 0.007 | ORT baseline; not a target-silicon measurement |
| gpt-2-10 | 18.89 | 0.08 | 0.005 | ORT baseline; not a target-silicon measurement |
