# head_pose_tracker — RTL Module

**File:** `rtl/head_pose_tracker/head_pose_tracker.v`
**Purpose:** DMS head-pose attention zone classifier with distraction counter

## What it does
Takes per-frame head orientation (yaw/pitch/roll) as signed 8-bit angles and:
1. Checks if the head is within the safe attention zone
2. Tracks how many of the last 15 frames were out-of-zone (distracted)

## Parameters
| Parameter | Default | Meaning |
|-----------|---------|---------|
| YAW_THRESH | 7'd30 | Max yaw: ±30° |
| PITCH_THRESH | 7'd20 | Max pitch: ±20° |
| ROLL_THRESH | 7'd20 | Max roll: ±20° |
| WINDOW_SIZE | 15 | Rolling window depth for distraction count |

## Attention zone
`in_zone = |yaw| <= 30° AND |pitch| <= 20° AND |roll| <= 20°`

## Ports
| Port | Dir | Width | Description |
|------|-----|-------|-------------|
| clk | in | 1 | System clock |
| rst_n | in | 1 | Active-low synchronous reset |
| valid | in | 1 | Pulse: new pose data ready |
| yaw | in | 8 | Signed yaw angle (-128 to +127°) |
| pitch | in | 8 | Signed pitch angle (-128 to +127°) |
| roll | in | 8 | Signed roll angle (-128 to +127°) |
| in_zone | out | 1 | 1 = head in attention zone this frame |
| distracted_count | out | 4 | Out-of-zone frames in rolling window |

## Key implementation details
- Absolute value computed as: `abs = angle[7] ? (-angle[6:0]) : angle[6:0]` (uses 7-bit magnitude)
- Zone window shift register: 1=in-zone, 0=distracted. Distracted count = WINDOW_SIZE - popcount
- Resets to in-zone (all 1s) so distracted_count starts at 0
- `in_zone` output is registered — reflects previous frame when next `valid` fires

## AXI4-Lite connection (in astracore_top)
- Write: `HEAD_POSE` register (0x2C) — `[7:0]=yaw`, `[15:8]=pitch`, `[23:16]=roll`
- Read: `HEAD_ST` register (0xA4) — `[0]=in_zone`, `[5:1]=distracted_count`
