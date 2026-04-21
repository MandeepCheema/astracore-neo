# Scenario 5 — YOLOv8n static-640 latency distribution

30 iterations after 3 warmups, single stream, default ORT CPU.

- Mean: 50.82 ms
- p50: 49.94 ms
- p95: 56.57 ms
- p99: 58.16 ms
- Max: 58.69 ms
- Stdev: 2.54 ms

_YOLOv8n ONNX has static 640×640 input. Full resolution sweep requires re-export; captured only the static-640 baseline. Extending would add ~½ day of Ultralytics export work._