# Deep test 5 — Latency vs GMACs correlation

For each zoo model, steady-state p50 latency (15 iters after 3 warmups) against theoretical GMACs. Pearson r across models is a rough health check on 'are we compute-bound?'

- Pearson r (GMACs, p50_ms): **0.15**

| Model | Family | GMACs | p50 ms | ms / GMAC |
|---|---|---:|---:|---:|
| squeezenet-1.1 | vision-classification | 0.349 | 2.65 | 7.60 |
| mobilenetv2-7 | vision-classification | 0.429 | 4.39 | 10.22 |
| resnet50-v2-7 | vision-classification | 4.091 | 30.77 | 7.52 |
| efficientnet-lite4-11 | vision-classification | 1.348 | 12.83 | 9.52 |
| shufflenet-v2-10 | vision-classification | 0.145 | 3.65 | 25.20 |
| yolov8n | vision-detection | 4.372 | 50.45 | 11.54 |
| bert-squad-10 | nlp-encoder-transformer | 1.293 | 208.45 | 161.23 |
| gpt-2-10 | nlp-decoder-transformer | 0.085 | 24.92 | 293.20 |

Interpretation:
- `r` close to 1.0 → latency tracks GMACs → compute-bound (good).
- `r` close to 0 → latency dominated by non-MAC work (memory, kernel launch).
- Per-model `ms / GMAC` lets you spot outliers where ONE model is mis-tuned.