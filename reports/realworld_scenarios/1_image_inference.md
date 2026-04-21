# Scenario 1 — Image inference cross-model sweep

Five ImageNet classifiers × two COCO demo images × 20 iterations each on ONNX Runtime CPU FP32. Same host as the zoo baseline.

## Image: `bus`

| Model | Top-1 | Prob | Top-5 | Cold ms | Steady p50 | Steady p99 |
|---|---|---:|---|---:|---:|---:|
| squeezenet-1.1 | minibus | 8.48% | minibus, waste container, shopping cart, stretcher, police van | 928.8 | 2.21 | 3.29 |
| mobilenetv2-7 | minibus | 76.95% | minibus, police van, trolleybus, tram, amphibious vehicle | 402.2 | 3.32 | 3.73 |
| resnet50-v2-7 | minibus | 74.16% | minibus, police van, trolleybus, minivan, tram | 1413.7 | 24.67 | 30.17 |
| efficientnet-lite4-11 | minibus | 0.27% | minibus, police van, trolleybus, minivan, passenger car | 470.8 | 9.38 | 12.06 |
| shufflenet-v2-10 | minibus | 43.94% | minibus, jeep, police van, recreational vehicle, minivan | 245.6 | 2.83 | 3.82 |

**Cross-model agreement:** 5/5 models pick `minibus` (100%).

## Image: `zidane`

| Model | Top-1 | Prob | Top-5 | Cold ms | Steady p50 | Steady p99 |
|---|---|---:|---|---:|---:|---:|
| squeezenet-1.1 | saxophone | 13.43% | saxophone, oboe, bassoon, flute, stage | 118.3 | 2.39 | 5.17 |
| mobilenetv2-7 | bow tie | 37.53% | bow tie, flute, stage, banjo, saxophone | 409.9 | 4.08 | 5.51 |
| resnet50-v2-7 | bow tie | 50.08% | bow tie, suit, bridegroom, mobile phone, television | 1361.3 | 23.49 | 31.25 |
| efficientnet-lite4-11 | oboe | 0.13% | oboe, flute, cello, suit, violin | 524.6 | 9.84 | 12.05 |
| shufflenet-v2-10 | saxophone | 68.46% | saxophone, trombone, oboe, cornet, stage | 193.6 | 3.03 | 4.66 |

**Cross-model agreement:** 2/5 models pick `saxophone` (40%).

*Cold ms* = full demo path (session create + preprocess + inference + decode).  *Steady* = 20 iters of reused session, 3 warmups.