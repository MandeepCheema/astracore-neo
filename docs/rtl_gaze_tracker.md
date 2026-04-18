# gaze_tracker — RTL Module

**File:** `rtl/gaze_tracker/gaze_tracker.v`
**Purpose:** Eye-state classification and PERCLOS drowsiness engine

## What it does
Takes left and right Eye Aspect Ratio (EAR) values each frame and:
1. Averages them: `avg = (left_ear + right_ear) / 2`
2. Classifies eye state (OPEN / PARTIAL / CLOSED)
3. Tracks PERCLOS — how many of the last 30 frames were CLOSED
4. Counts blinks (CLOSED → OPEN transitions)

## Parameters
| Parameter | Default | Meaning |
|-----------|---------|---------|
| WINDOW_SIZE | 30 | PERCLOS rolling window depth |
| EAR_CLOSED_THRESH | 51 | avg < 51 → CLOSED (= 0.20 × 255) |
| EAR_PARTIAL_THRESH | 76 | avg < 76 → PARTIAL (= 0.30 × 255) |

## Ports
| Port | Dir | Width | Description |
|------|-----|-------|-------------|
| clk | in | 1 | System clock |
| rst_n | in | 1 | Active-low synchronous reset |
| valid | in | 1 | Pulse: new EAR values ready |
| left_ear | in | 8 | Left eye EAR (0–255 unsigned) |
| right_ear | in | 8 | Right eye EAR (0–255 unsigned) |
| eye_state | out | 2 | 00=OPEN, 01=PARTIAL, 10=CLOSED |
| avg_ear_out | out | 8 | Registered average EAR |
| perclos_num | out | 6 | Count of CLOSED frames in window |
| blink_count | out | 16 | Total blinks since reset |

## Key implementation details
- EAR average is computed combinationally as `ear_sum[8:1]` (1-bit right shift = divide by 2)
- PERCLOS uses a shift register of depth 30; popcount = number of CLOSED frames
- Blink detected by comparing registered previous state with current combinational state (no extra pipeline stage needed)
- `eye_state` output is registered — reflects PREVIOUS frame's classification when valid fires

## AXI4-Lite connection (in astracore_top)
- Write: `GAZE` register (0x04) — `[7:0]=left_ear`, `[15:8]=right_ear`
- Read: `GAZE_ST` register (0x80) — `[1:0]=eye_state`, `[7:2]=perclos_num`, `[23:8]=blink_count`
