# AstraCore model-zoo benchmark (backend: onnxruntime)

`n_iter=3`, wall=12.5s, 8 models.

| Model | Latency (ms) | GMACs | Delivered TOPS | Notes |
|---|---|---|---|---|
| squeezenet-1.1 | 3.27 | 0.35 | 0.117 | ORT baseline; not a target-silicon measurement |
| mobilenetv2-7 | 6.49 | 0.43 | 0.043 | ORT baseline; not a target-silicon measurement |
| resnet50-v2-7 | 31.27 | 4.09 | 0.127 | ORT baseline; not a target-silicon measurement |
| efficientnet-lite4-11 | 14.33 | 1.35 | 0.104 | ORT baseline; not a target-silicon measurement |
| shufflenet-v2-10 | 4.69 | 0.14 | 0.051 | ORT baseline; not a target-silicon measurement |
| yolov8n | 85.52 | 4.37 | 0.053 | ORT baseline; not a target-silicon measurement |
| bert-squad-10 | 257.65 | 1.29 | 0.006 | ORT baseline; not a target-silicon measurement |
| gpt-2-10 | 23.15 | 0.08 | 0.004 | ORT baseline; not a target-silicon measurement |
