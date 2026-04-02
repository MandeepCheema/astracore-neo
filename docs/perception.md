# Module 5 — Perception

**Status:** DONE | **Tests:** 83/83 (100%) | **Date:** 2026-04-02

## Overview

The Perception module simulates the complete sensor stack for an autonomous vehicle — camera, lidar, and radar — and fuses their outputs into a unified list of tracked objects in the ego-vehicle frame.

## Sub-modules

### camera.py
- **CameraSensor** — MIPI CSI-2 camera with power control and frame capture
- **ISPPipeline** — Image Signal Processing chain: demosaic → white balance → AI denoise → tone map → gamma
- **CameraConfig** — Configures resolution (up to 8K), FPS, Bayer pattern, HDR, bit depth, gamma
- **Frame / FrameMetadata** — Output structure with processed RGB data (float32, [0,1]) and capture metadata
- **BayerPattern** — Supports RGGB, BGGR, GRBG, GBRG
- **PixelFormat** — RAW10, RAW12, YUV420, RGB888, HDR

### lidar.py
- **LidarSensor** — 4D solid-state lidar (x, y, z + radial velocity) with power control and scan
- **LidarConfig** — 128 channels, 120° H-FOV, 200m max range, configurable resolution
- **PointCloud** — Core data structure (N arrays: x, y, z, intensity, velocity)
- **VoxelGrid** — Downsampled voxel representation with occupancy and intensity mean
- **LidarCluster** — Detected object cluster with centroid, bounding box, point count
- **filter_range()** — Keep points within [min_r, max_r] metres
- **remove_ground()** — Height-threshold ground removal
- **voxelize()** — Voxel-grid downsampling (pure numpy)
- **cluster_points()** — DBSCAN-style clustering (pure numpy, no sklearn)

### radar.py
- **RadarSensor** — 4D mmWave FMCW radar (77 GHz) with power control and scan
- **RadarConfig** — 250m max range, ±72 m/s velocity, 120° azimuth FOV, 128 chirps
- **RadarDetection** — Single target: range, azimuth, elevation, velocity, RCS, SNR
- **RangeDopplerProcessor** — 2D FFT range-Doppler map + CA-CFAR detection

### fusion.py
- **SensorFusionEngine** — Fuses lidar clusters + radar detections → FusedObject list
- **FusedObject** — Unified tracked object: position (ego frame), velocity, class, confidence, sources
- **ObjectClass** — UNKNOWN, VEHICLE, PEDESTRIAN, CYCLIST, ANIMAL, OBSTACLE
- **ExtrinsicCalib** — Rigid-body sensor-to-ego transform (rotation + translation)
- **IntrinsicCalib** — Pinhole camera K matrix (fx, fy, cx, cy)
- Fusion strategy: lidar clusters → ego frame → nearest radar match (within 10m) → heuristic classification by bounding-box volume

## Dependencies
- Depends on: HAL (chip timing model), Memory (frame buffer transfers)
- Required by: DMS (module 9), Models (module 11)

## Key Design Notes
- All arrays are float32 for consistency with downstream inference pipeline
- Synthetic data generation uses seeded numpy RNG for reproducible tests
- DBSCAN clustering is pure numpy (no sklearn) to match chip's constraint of no external library calls
- Fusion uses simplified track-before-detect; full Kalman tracking is deferred to models module
