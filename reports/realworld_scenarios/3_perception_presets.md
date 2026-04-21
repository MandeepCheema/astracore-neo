# Scenario 3 — Perception pipeline across synthetic presets

Feeds each preset's first scene through the SDK's replay pipeline (camera detector stub + lidar filter + cluster + radar SNR filter + cross-sensor fusion). Scene lengths clipped so every preset completes in seconds.

| Preset | N samples | Wall s | cam det | lidar clust | radar det | fused | ms/frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny | 10 | 0.85 | 0.0 | 2.9 | 5.7 | 2.9 | 83.6 |
| standard | 15 | 1.83 | 3.0 | 13.2 | 20.3 | 13.2 | 121.8 |
| extended-sensors | 10 | 0.86 | 0.0 | 3.3 | 21.2 | 3.3 | 86.2 |
| vlp32 | 10 | 20.26 | 80.0 | 1.0 | 341.0 | 1.0 | 2026.3 |