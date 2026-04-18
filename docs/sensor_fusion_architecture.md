# AstraCore Neo — Sensor Fusion Architecture
# Rev 1.0 | 2026-04-15
# Target: L2+–L4 ADAS | ISO 26262 ASIL-D

---

## Sensor Suite

| Sensor | Count | Output | Rate | Bandwidth | Interface |
|--------|-------|--------|------|-----------|-----------|
| Driver Monitor Camera (DMC) | 1 | 1080p RGB/IR | 30 fps | ~250 Mbps | MIPI CSI-2 4-lane |
| Exterior Cameras | 4–6 | 1080p–4K | 30–60 fps | ~1 Gbps × N | MIPI CSI-2 4-lane × N |
| Radar (77 GHz FMCW) | 2–4 | Range+velocity+azimuth point cloud | 20–50 Hz | ~10 Mbps | SPI / CAN-FD / SGMII |
| LiDAR (solid-state or mech) | 1–2 | 3D point cloud (100K–1M pts/frame) | 10–20 Hz | ~100 Mbps | 100BASE-T1 Automotive Ethernet |
| Ultrasonic | 8–12 | Distance 0.2–5m | 25 Hz | <1 Mbps | UART/PWM (LIN-compatible) |

**Total aggregate bandwidth:** 3–6 Gbps raw sensor data per second.
This cannot all be processed on-chip — AstraCore Neo acts as a **fusion co-processor**, not a raw sensor processor.

---

## Architecture Philosophy

AstraCore Neo is NOT a camera ISP or radar DSP. The correct split is:

```
[Sensor]  →  [Sensor ECU / ISP]  →  [Detection output]  →  [AstraCore Neo Fusion]
Camera         ISP + CNN (or GPU)     Bounding boxes
Radar          Radar MCU + CFAR       Object clusters  
LiDAR          LiDAR MCU + cluster    Point clusters   
Ultrasonic     Analog frontend        Distance values  
DMS Camera     On-chip (existing)     eye_state, pose  
```

AstraCore Neo receives **pre-processed detections** from upstream sensor ECUs,
not raw pixels or ADC samples. This is standard in production ADAS architectures
(e.g., Mobileye + sensor hub pattern).

**Exception — DMS camera:** AstraCore Neo processes DMS video directly on-chip
using existing gaze_tracker + head_pose_tracker modules.

---

## Sensor Data Profiles

### 1. Driver Monitoring Camera (DMC) — On-Chip Processing
**Interface:** MIPI CSI-2 4-lane D-PHY (up to 4.5 Gbps)
**Data:** 1080p monochrome IR, 30 fps, face crop region ~320×240 pixels
**Processing on AstraCore Neo:**
- Face detection (existing: gaze_tracker + head_pose_tracker)
- PERCLOS calculation (eye closure ratio)
- Head pose (yaw/pitch/roll)
- Output → `dms_fusion` module

**RTL needed:**
- `mipi_csi2_rx.v` — MIPI CSI-2 D-PHY receiver, deserializer, line buffer
- Face crop + resize pipeline → feed into gaze/head_pose modules
- Or: receive pre-cropped face region from ISP (simpler)

---

### 2. Exterior Cameras — Off-Chip CNN, On-Chip Fusion
**Interface (to AstraCore Neo):** AXI4-Stream or AXI4-Lite register feed
**What AstraCore Neo receives:** Detection list from external CNN (GPU or vision MCU)

```
struct camera_detection_t {
    uint16_t  class_id;       // pedestrian, vehicle, cyclist, sign...
    uint16_t  confidence;     // 0–1000 (fixed point 0.001)
    uint16_t  bbox_x, bbox_y; // bounding box top-left (pixels)
    uint16_t  bbox_w, bbox_h; // bounding box width/height
    uint32_t  timestamp_us;   // frame capture time
    uint8_t   camera_id;      // 0=front, 1=rear, 2=left, 3=right
}
```

**Max detections:** 32 per frame per camera → 192 detections/frame at 60 fps = ~11K detections/sec
**RTL needed:** AXI4-Stream detection receiver, detection FIFO (SRAM-backed, 256-entry)

---

### 3. Radar (77 GHz FMCW) — Range + Velocity + Angle
**Interface:** SPI (short-range), SGMII (long-range), or CAN-FD (low bandwidth)
**What AstraCore Neo receives:** Radar object list after CFAR + clustering in radar MCU

```
struct radar_object_t {
    int16_t   range_cm;        // 0–30000 cm (0–300m)
    int16_t   velocity_cms;    // -5000–+5000 cm/s (radial velocity)
    int16_t   azimuth_mdeg;    // -6000–+6000 millidegs (±60°)
    int16_t   elevation_mdeg;  // -2000–+2000 millidegs (±20°)
    uint16_t  rcs_dbsm;        // radar cross section (object size proxy)
    uint8_t   confidence;      // 0–100
    uint32_t  timestamp_us;
}
```

**Max objects:** 64 per radar frame at 20 Hz = 1280 objects/sec
**Key advantage over camera:** works in rain, fog, night; provides direct velocity (Doppler)
**RTL needed:** `radar_interface.v` — SPI slave or AXI4-Lite register array for radar MCU writes

---

### 4. LiDAR — 3D Point Cloud
**Interface:** 100BASE-T1 Automotive Ethernet (IEEE 802.3bw) → Ethernet controller
**What AstraCore Neo receives:** Clustered object list after on-LiDAR or LiDAR-MCU processing

```
struct lidar_object_t {
    int32_t   x_mm, y_mm, z_mm;  // 3D centroid in vehicle frame
    uint16_t  length_mm;          // bounding box dimensions
    uint16_t  width_mm;
    uint16_t  height_mm;
    uint8_t   class_id;           // car, truck, pedestrian, unknown
    uint8_t   confidence;
    uint32_t  timestamp_us;
    uint16_t  point_count;        // number of points in cluster
}
```

**Max objects:** 128 per LiDAR scan at 10 Hz = 1280 objects/sec
**Key advantage:** precise 3D geometry, height information (camera lacks depth, radar lacks height)
**Interface:** AstraCore Neo already has `ethernet_controller.v` — extend to handle LiDAR UDP packets

---

### 5. Ultrasonic — Proximity / Parking
**Interface:** UART (LIN-compatible) or GPIO PWM capture
**Data:** 8–12 distance measurements per scan at 25 Hz

```
struct ultrasonic_t {
    uint16_t  distance_mm[12];  // per-sensor distance, 0=no echo
    uint8_t   sensor_health;    // bitmask: 1=OK, 0=fault
    uint32_t  timestamp_us;
}
```

**Range:** 0.2–5m — useful for low-speed maneuvers, parking, blind spot close-range
**RTL needed:** `ultrasonic_interface.v` — simple UART RX + distance register array

---

## Sensor Fusion Pipeline — RTL Architecture

```
                      ┌─────────────────────────────────────────────┐
                      │           AstraCore Neo Fusion Engine         │
                      │                                               │
  DMS Camera ─────────┤→ gaze_tracker → ┐                            │
                      │                  ├→ dms_fusion ──────────────┤→ driver_state
  Head Pose  ─────────┤→ head_pose_tracker┘                          │
                      │                                               │
  Radar MCU ──SPI────►┤→ radar_interface → radar_object_fifo ──────►┤→             │
                      │                                               │  object_    │
  Camera CNN ──AXI───►┤→ cam_det_receiver → cam_object_fifo ───────►┤  tracker ──►┤→ fused_
                      │                                               │  (SRAM     │   object_
  LiDAR MCU ─ETH─────►┤→ ethernet_controller → lidar_object_fifo ──►┤  backed)   │   list[]
                      │                                               │             │
  Ultrasonic──UART───►┤→ ultrasonic_interface → proximity_regs ─────┤→             │
                      │                                               │
                      │  ┌─────────────────────────────┐             │
                      │  │   Temporal Sync Engine       │             │
                      │  │  timestamp alignment + gate  │             │
                      │  └─────────────────────────────┘             │
                      └─────────────────────────────────────────────┘
```

---

## New RTL Modules Required

### A. `dms_fusion.v`
**Inputs:** gaze_tracker outputs + head_pose_tracker outputs
**Outputs:** `driver_attention_level[2:0]`, `dms_confidence[7:0]`, `dms_alert`

```
Level encoding:
  3'b000 = ATTENTIVE   — in zone, eyes open, PERCLOS < 20%
  3'b001 = DROWSY      — PERCLOS 20–50% or blink rate high
  3'b010 = DISTRACTED  — head out of zone >3s
  3'b100 = CRITICAL    — PERCLOS >50% or eyes closed >2s
  3'b111 = SENSOR_FAIL — DMS camera fault
```

**Fusion logic:**
- Weighted vote: PERCLOS (40%) + blink_count (20%) + head_pose (40%)
- IIR temporal smoother: `level_filtered = 0.7*prev + 0.3*new` (prevents single-frame alerts)
- ASIL-D: if DMS camera fails (valid stuck low) → output SENSOR_FAIL, assert `dms_alert`

---

### B. `sensor_sync.v`
**Purpose:** All sensors run on different clocks. Fusion requires temporal alignment.

**Approach — timestamp-based gating:**
- Each sensor object carries `timestamp_us` (microsecond GPS/PTP time)
- `sensor_sync` maintains a 100 μs fusion window
- Gates all FIFOs: releases detections within the current window simultaneously
- Drops stale detections (>200ms old = sensor fault)

**RTL:** 4 comparison windows, 4 valid flags, synchronous release on window close

---

### C. `object_tracker.v` (SRAM-backed)
**Purpose:** Maintain a track table of active objects across frames.

**Track table:** 128 entries × 64 bytes = 8 KB SRAM
```
struct track_t {
    uint16_t  track_id;
    uint8_t   sensor_mask;    // bitmask: which sensors see this object
    uint8_t   age_frames;     // frames since last update
    uint8_t   confidence;     // fused confidence 0–100
    uint8_t   class_id;       // pedestrian, vehicle, cyclist...
    int32_t   x_mm, y_mm;    // position in vehicle frame (Kalman estimate)
    int16_t   vx_mms, vy_mms; // velocity estimate
    uint16_t  width_mm, length_mm; // dimensions
}
```

**Association logic (hardware):**
- Euclidean distance matching: for each new detection, find nearest track
- If distance < threshold AND class matches: update track
- If no match: create new track
- If track unseen for N frames: mark as dead

**Kalman update:** Simplified constant-velocity predictor in hardware
- Predict: `x_pred = x + vx * dt`
- Update: `x_est = x_pred + K * (z - x_pred)` where K is fixed gain

---

### D. `radar_interface.v`
**Interface:** SPI slave (up to 20 MHz), 64-entry object FIFO
**Receives:** Radar object list packets from radar MCU
**Outputs:** AXI4-Stream of `radar_object_t` structs

---

### E. `lidar_interface.v`
**Interface:** Extends existing `ethernet_controller.v`
**Adds:** UDP packet parser for LiDAR vendor protocol (Velodyne/Ouster/Innoviz)
**Outputs:** AXI4-Stream of `lidar_object_t` structs

---

### F. `ultrasonic_interface.v`
**Interface:** UART 9600–115200 baud, 12-channel
**Outputs:** 12× `distance_mm[15:0]` registers + health bitmask
**Simple:** just UART RX + register file

---

### G. `cam_detection_receiver.v`
**Interface:** AXI4-Lite register writes from external CNN processor
**FIFO:** 256 × `camera_detection_t` (SRAM-backed)
**Outputs:** AXI4-Stream to object_tracker

---

## ASIL-D Fusion Requirements

### Redundancy Rules
| Detection | Required Sources | ASIL Level |
|-----------|-----------------|------------|
| Vehicle ahead (collision) | Camera AND Radar | ASIL-D |
| Pedestrian (collision) | Camera AND (Radar OR LiDAR) | ASIL-D |
| Lane departure | Camera only | ASIL-B |
| Driver drowsy | DMS AND (head pose OR blink) | ASIL-D |
| Proximity (<0.5m) | Ultrasonic AND Camera | ASIL-B |

### Sensor Availability Monitoring
Each sensor has a watchdog — if no valid data in N milliseconds:
- Camera: 200ms timeout → mark camera_fault
- Radar: 100ms timeout → mark radar_fault  
- LiDAR: 500ms timeout → mark lidar_fault
- Ultrasonic: 200ms timeout → mark us_fault

If ASIL-D-required sensor is faulted:
- Degrade to safe mode (lower speed limit, alert driver)
- Assert IRQ to safety controller
- Log fault in telemetry engine

---

## Integration with Existing AstraCore Neo Modules

| Existing Module | Role in Fusion |
|-----------------|----------------|
| gaze_tracker | DMS → feeds dms_fusion |
| head_pose_tracker | DMS → feeds dms_fusion |
| ethernet_controller | LiDAR data ingress (UDP extension) |
| canfd_controller | Radar interface (low-BW option) + safety message output |
| fault_predictor | Sensor health monitoring |
| thermal_zone | Sensor ECU temperature monitoring |
| inference_runtime | CNN inference for DMS on-chip feature extraction |
| mac_array | INT8 multiply for any on-chip CNN forward pass |
| tmr_voter | Triple-redundant vote on final driver_attention_level |
| ecc_secded | Track table SRAM error correction |

---

## Chip I/O Requirements Added

| Signal Group | Count | Direction | Notes |
|-------------|-------|-----------|-------|
| MIPI CSI-2 D-PHY (DMS) | 4 data + 1 clk | IN | 4-lane |
| MIPI CSI-2 D-PHY (ext cam) | 4 × N lanes | IN | Per camera, or off-chip ISP |
| Radar SPI | 4 (MISO/MOSI/CS/CLK) | IN | Per radar unit |
| 100BASE-T1 Ethernet | 2 (TX+/TX-) | BIDIR | LiDAR |
| UART (ultrasonic) | 1 RX | IN | LIN-compatible |
| Driver state output | 3 | OUT | `driver_attention_level` |
| Sensor fault IRQ | 5 | OUT | Per sensor modality |

**Impact on pin count:** Adds ~30–50 pins. The current chip (astracore_top) uses ~65 pins.
Total with full sensor suite: **~110–120 pins** — still feasible for a QFN or BGA package.

---

## Complete Module List — All Four Layers

The original 7-module plan covered only the sensor ingestion layer.
A production L2+–L4 system requires all four layers below.

### Layer 1 — Sensor Interface (hardware entry points)

| Module | Interface | Purpose | Status |
|--------|-----------|---------|--------|
| `dms_fusion.v` | Internal (gaze + pose signals) | Combines gaze_tracker + head_pose_tracker → driver_attention_level | ✅ DONE — 14/14 sim |
| `mipi_csi2_rx.v` | MIPI CSI-2 D-PHY 4-lane | DMS camera hardware entry point — feeds raw frames to gaze/head_pose modules | ⏳ PENDING |
| `imu_interface.v` | SPI 4-wire, up to 4 MHz | 6-DOF IMU (accel + gyro) → ego-motion for Kalman; ASIL-D redundancy | ⏳ PENDING |
| `gnss_interface.v` | UART (NMEA) + GPIO PPS | Absolute time source for all sensor timestamps; position fix | ⏳ PENDING |
| `ptp_clock_sync.v` | Extends ethernet_controller | Distributes GNSS time to sensor ECUs via IEEE 1588 / gPTP (802.1AS) | ⏳ PENDING |
| `can_odometry_decoder.v` | Extends canfd_controller | Wheel speed ×4, steering angle, yaw rate → velocity prior for Kalman | ⏳ PENDING |
| `radar_interface.v` | SPI slave, up to 20 MHz | Radar MCU object list → 64-entry radar_object_t FIFO | ⏳ PENDING |
| `ultrasonic_interface.v` | UART 9600–115200 baud | 12-channel proximity → distance_mm[] registers | ⏳ PENDING |
| `cam_detection_receiver.v` | AXI4-Lite reg writes | External CNN detections → 256-entry camera_detection_t FIFO | ⏳ PENDING |
| `lidar_interface.v` | Extends ethernet_controller | LiDAR UDP packet parser → lidar_object_t FIFO | ⏳ PENDING |

### Layer 2 — Fusion Processing

| Module | Purpose | Key Dependency | Status |
|--------|---------|----------------|--------|
| `sensor_sync.v` | Timestamp alignment, 100 μs fusion window, stale detection gating | gnss_interface / ptp_clock_sync for timestamp reference | ⏳ PENDING |
| `coord_transform.v` | Rotate/translate all detections from sensor frame → vehicle body frame using calibration offsets | Calibration registers (AXI4-Lite) | ⏳ PENDING |
| `ego_motion_estimator.v` | Fuse IMU + wheel odometry → ego velocity, yaw rate, acceleration | imu_interface + can_odometry_decoder | ⏳ PENDING |
| `object_tracker.v` | 128-entry SRAM-backed track table, Kalman predict+update, multi-sensor association | coord_transform + ego_motion_estimator + all sensor FIFOs | ⏳ PENDING |
| `lane_fusion.v` | Fuse camera lane detections + HD map geometry → lane estimate for LKA | cam_detection_receiver + HD map SRAM | ⏳ PENDING |
| `plausibility_checker.v` | Cross-sensor consistency validation (ASIL-D): flags detections unsupported by required redundant sensors | object_tracker output + ASIL-D redundancy rules | ⏳ PENDING |

### Layer 3 — Decision / Output

| Module | Output | ASIL | Status |
|--------|--------|------|--------|
| `ttc_calculator.v` | TTC per object → WARNING / PREPARE / BRAKE flags | ASIL-D | ⏳ PENDING |
| `aeb_controller.v` | Brake command → CAN-FD to brake ECU; ≤100ms end-to-end | ASIL-D | ⏳ PENDING |
| `ldw_lka_controller.v` | Lane departure alert + steering correction request | ASIL-B | ⏳ PENDING |
| `safe_state_controller.v` | ASIL-D fault → driver alert escalation → minimal risk condition (MRC) | ASIL-D | ⏳ PENDING |

### Layer 4 — Infrastructure (from Phase 4A/4B/4C gaps analysis)

AXI4-Full interconnect, DMA engine, LPDDR4 controller, SRAM macros, interrupt controller,
PLL, RISC-V CPU, hardware task scheduler, telemetry aggregator, security (HSM + SecOC +
bus firewall), watchdog timer, JTAG debug TAP, I/O pad ring.

---

## Critical Dependencies (build-order constraints)

```
gnss_interface ──────────────────────────────► sensor_sync
ptp_clock_sync (extends ethernet_controller) ─► sensor_sync

imu_interface ──────────────┐
can_odometry_decoder ───────┴──► ego_motion_estimator ──────────────────────► object_tracker
                                                                                    ▲
mipi_csi2_rx → gaze_tracker → dms_fusion                                          │
mipi_csi2_rx → head_pose_tracker ↗                                                │
radar_interface ──────────────────► coord_transform ──► sensor_sync (gated) ─────┤
cam_detection_receiver ───────────► coord_transform ──► sensor_sync (gated) ─────┤
lidar_interface ──────────────────► coord_transform ──► sensor_sync (gated) ─────┤
ultrasonic_interface ─────────────────────────────────► sensor_sync (gated) ─────┘

object_tracker ──► plausibility_checker ──► ttc_calculator ──► aeb_controller ──► CAN-FD
object_tracker ──► lane_fusion ──────────────────────────────► ldw_lka_controller ► CAN-FD
plausibility_checker (fault) ────────────────────────────────► safe_state_controller
```

---

## Revised Build Plan

**Step 1 — Already done:**
- `dms_fusion.v` ✅ RTL + 14/14 cocotb tests

**Step 2 — Sensor interfaces (no inter-module dependencies, can build in parallel):**
1. `sensor_sync.v` — timestamp alignment engine (~200 lines)
2. `mipi_csi2_rx.v` — MIPI D-PHY receiver + line buffer (~300 lines)
3. `imu_interface.v` — SPI slave, 6-DOF data registers (~120 lines)
4. `gnss_interface.v` — UART RX + PPS timestamper (~150 lines)
5. `ptp_clock_sync.v` — IEEE 1588 clock servo (~250 lines)
6. `can_odometry_decoder.v` — CAN-FD frame decoder (~120 lines)
7. `radar_interface.v` — SPI slave + FIFO (~200 lines)
8. `ultrasonic_interface.v` — UART RX + register array (~100 lines)
9. `cam_detection_receiver.v` — AXI4-Lite → FIFO (~150 lines)
10. `lidar_interface.v` — UDP parser extension (~200 lines)

**Step 3 — Fusion processing (depends on Step 2):**
11. `coord_transform.v` — calibrated frame rotation + translation (~200 lines)
12. `ego_motion_estimator.v` — IMU + odometry fusion (~250 lines)
13. `object_tracker.v` — track table + Kalman (most complex, ~500 lines)
14. `lane_fusion.v` — camera lanes + map geometry (~200 lines)
15. `plausibility_checker.v` — ASIL-D cross-sensor validation (~150 lines)

**Step 4 — Decision layer (depends on Step 3):**
16. `ttc_calculator.v` — time-to-collision (~150 lines)
17. `aeb_controller.v` — AEB decision + CAN-FD output (~200 lines)
18. `ldw_lka_controller.v` — lane keeping assist (~150 lines)
19. `safe_state_controller.v` — fault → MRC sequencer (~200 lines)

**Step 5 — Integration:**
- Add all modules to astracore_top.v
- Extend AXI4-Lite register map
- Add all sensor fault IRQs to interrupt controller
- Wire tmr_voter to ASIL-D output paths (aeb, safe_state)

**Step 6 — Verification:**
- cocotb: inject synthetic data from all sensor modalities
- ASIL-D redundancy: disable each sensor, verify safe-mode response
- Formal: dms_fusion never ATTENTIVE when eye_state=CLOSED
- Formal: aeb_controller always responds within 100ms of TTC < threshold
- Fault injection: plausibility_checker catches single-sensor spoofing
