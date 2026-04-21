# AstraCore --apply report — Tier-1 reference ADAS rig

Four perception cameras + one roof-mounted lidar + six radars + twelve ultrasonics + in-cabin mic array + 1 thermal + 1 event camera + 1 ToF depth for driver monitoring. Target silicon: Qualcomm SA8650P.

Backend: `onnxruntime`  •  Dataset: `synthetic` / `robotaxi`  •  Wall: 24.4s

## Sensors

| Kind | Count |
|---|---:|
| cameras | 4 |
| lidars | 1 |
| radars | 6 |
| ultrasonics | 12 |
| microphones | 1 |
| thermals | 1 |
| events | 1 |
| depths | 1 |
| can | 2 |
| gnss | 1 |
| imu | 1 |

## Replay

Scene `synthetic-scene-000` ('Synthetic drive #0'), 10 samples, 1.02s wall.

| Metric | Mean |
|---|---:|
| mean_camera_det | 0.00 |
| mean_lidar_clust | 3.30 |
| mean_radar_det | 21.20 |
| mean_fused_obj | 3.30 |
| mean_gt_per_frame | 3.20 |
| mean_ms_per_frame | 102.47 |
| p50_ms_per_frame | 102.86 |

## Models

| id | family | precision | sparsity | ms / inf | GMACs | TOPS | notes |
|---|---|---|---|---:|---:|---:|---|
| front_perception | vision-detection | INT8 | dense | 52.52 | 4.37 | 0.081 | input=CAM_FRONT |
| side_perception | vision-detection | INT8 | 2:4 | 58.09 | 4.37 | 0.076 | input=CAM_FRONT_LEFT |
| cabin_dms | vision-detection | INT8 | dense | 51.40 | 4.37 | 0.086 | input=TOF_DRIVER_MONITOR |

## Multi-stream scaling (streams_per_model=4)

### front_perception

| Streams | IPS | TOPS (agg) | p50 ms | p99 ms | scale |
|---:|---:|---:|---:|---:|---:|
| 1 | 17.7 | 0.077 | 55.71 | 65.28 | 1.00× |
| 2 | 21.1 | 0.092 | 92.92 | 113.91 | 1.20× |
| 4 | 21.0 | 0.092 | 184.43 | 256.33 | 1.19× |

### side_perception

| Streams | IPS | TOPS (agg) | p50 ms | p99 ms | scale |
|---:|---:|---:|---:|---:|---:|
| 1 | 16.2 | 0.071 | 59.73 | 92.28 | 1.00× |
| 2 | 18.8 | 0.082 | 101.88 | 160.84 | 1.16× |
| 4 | 19.9 | 0.087 | 183.15 | 293.48 | 1.23× |

### cabin_dms

| Streams | IPS | TOPS (agg) | p50 ms | p99 ms | scale |
|---:|---:|---:|---:|---:|---:|
| 1 | 17.0 | 0.074 | 57.79 | 73.54 | 1.00× |
| 2 | 17.8 | 0.078 | 104.62 | 162.64 | 1.04× |
| 4 | 21.4 | 0.094 | 177.79 | 282.35 | 1.26× |

## Safety policies (declared)

| Type | Value | Description |
|---|---|---|
| min_pedestrian_distance_m | `0.5` | Reject pedestrian detections < 0.5m â€” bumper sensor handles these |
| max_fused_object_velocity_m_per_s | `55` | Motorway cap â€” anything faster is a radar ghost |
| require_lidar_confirmation_for_emergency_brake | `True` | ASIL-D policy â€” camera alone must not trigger AEB |

## Notes

- replay preset downsized from 'robotaxi' to 'extended-sensors' to keep --apply tractable (use `astracore replay --preset robotaxi` for the full rig)
