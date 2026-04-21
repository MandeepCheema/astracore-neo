# AstraCore --apply report — Tier-1 reference ADAS rig

Four perception cameras + one roof-mounted lidar + six radars + twelve ultrasonics + in-cabin mic array + 1 thermal + 1 event camera + 1 ToF depth for driver monitoring. Target silicon: Qualcomm SA8650P.

Backend: `onnxruntime`  •  Dataset: `synthetic` / `robotaxi`  •  Wall: 4911.7s

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

Scene `synthetic-scene-000` ('Synthetic drive #0'), 150 samples, 4775.70s wall.

| Metric | Mean |
|---|---:|
| mean_camera_det | 656.00 |
| mean_lidar_clust | 1.00 |
| mean_radar_det | 2057.78 |
| mean_fused_obj | 1.00 |
| mean_gt_per_frame | 3.10 |
| mean_ms_per_frame | 31837.82 |
| p50_ms_per_frame | 29040.03 |

## Models

| id | family | precision | sparsity | ms / inf | GMACs | TOPS | notes |
|---|---|---|---|---:|---:|---:|---|
| front_perception | vision-detection | INT8 | dense | 48.30 | 4.37 | 0.091 | input=CAM_FRONT |
| side_perception | vision-detection | INT8 | 2:4 | 50.39 | 4.37 | 0.090 | input=CAM_FRONT_LEFT |
| cabin_dms | vision-detection | INT8 | dense | 50.24 | 4.37 | 0.089 | input=TOF_DRIVER_MONITOR |

## Multi-stream scaling (streams_per_model=4)

### front_perception

| Streams | IPS | TOPS (agg) | p50 ms | p99 ms | scale |
|---:|---:|---:|---:|---:|---:|
| 1 | 19.7 | 0.086 | 51.09 | 52.83 | 1.00× |
| 2 | 21.1 | 0.092 | 88.90 | 153.59 | 1.07× |
| 4 | 24.2 | 0.106 | 161.25 | 222.00 | 1.23× |

### side_perception

| Streams | IPS | TOPS (agg) | p50 ms | p99 ms | scale |
|---:|---:|---:|---:|---:|---:|
| 1 | 19.3 | 0.084 | 51.04 | 58.00 | 1.00× |
| 2 | 23.4 | 0.102 | 84.61 | 97.95 | 1.22× |
| 4 | 26.1 | 0.114 | 148.10 | 204.12 | 1.35× |

### cabin_dms

| Streams | IPS | TOPS (agg) | p50 ms | p99 ms | scale |
|---:|---:|---:|---:|---:|---:|
| 1 | 19.6 | 0.086 | 51.15 | 55.02 | 1.00× |
| 2 | 20.1 | 0.088 | 92.61 | 140.04 | 1.03× |
| 4 | 24.5 | 0.107 | 158.34 | 268.92 | 1.25× |

## Safety policies (declared)

| Type | Value | Description |
|---|---|---|
| min_pedestrian_distance_m | `0.5` | Reject pedestrian detections < 0.5m â€” bumper sensor handles these |
| max_fused_object_velocity_m_per_s | `55` | Motorway cap â€” anything faster is a radar ghost |
| require_lidar_confirmation_for_emergency_brake | `True` | ASIL-D policy â€” camera alone must not trigger AEB |
