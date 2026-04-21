# Scenario 2 — YOLOv8n detection on 28-image eval set

28 images × 5 iters each on ONNX Runtime CPU FP32. Detection threshold 0.25 on raw class scores.

## Aggregate

- Detections per image: mean=34.4, min=0, max=193
- Latency p50: 48.38 ms (averaged across images)
- Worst-case p99: 59.58 ms
- Unique output fingerprints: 28 (every image should produce a distinct one)

## Per-image (first 10 of 28)

| Image | N-det | Top class(id,n) | p50 ms | p99 ms | Fingerprint |
|---:|---:|---|---:|---:|---|
| 0 | 10 | (4,10) | 41.85 | 45.52 | `02fa227baf8ef776` |
| 1 | 89 | (0,77), (4,12) | 40.56 | 42.32 | `4ac61b0103d15036` |
| 2 | 76 | (0,62), (35,11), (34,3) | 39.07 | 40.57 | `ac777061c13d8499` |
| 3 | 23 | (71,13), (58,10) | 40.22 | 47.28 | `fd3068f6c959703d` |
| 4 | 15 | (79,15) | 40.43 | 45.44 | `5c00299764392c61` |
| 5 | 193 | (56,176), (0,17) | 40.89 | 44.50 | `932da9dc0ae2275e` |
| 6 | 11 | (0,11) | 39.71 | 41.55 | `6c05d484794aedd6` |
| 7 | 29 | (0,19), (28,10) | 40.08 | 40.39 | `54f44c4e2165fd51` |
| 8 | 10 | (15,10) | 45.77 | 52.38 | `a0b425ac8fdf0c98` |
| 9 | 1 | (16,1) | 53.31 | 56.89 | `c8d4f64eb28bd635` |

_Full 28 rows in the sibling JSON._