# thermal_zone — RTL Module

**File:** `rtl/thermal_zone/thermal_zone.v`
**Purpose:** On-chip temperature monitor and thermal state machine

## What it does
Takes an 8-bit temperature reading (degrees Celsius, 0–255) and outputs a 5-state thermal zone classification. Automatically signals throttle and shutdown.

## Parameters
| Parameter | Default | Meaning |
|-----------|---------|---------|
| WARN_THRESH | 75 | 75°C — enter WARNING |
| THROTTLE_THRESH | 85 | 85°C — enter THROTTLED |
| CRITICAL_THRESH | 95 | 95°C — enter CRITICAL |
| SHUTDOWN_THRESH | 105 | 105°C — enter SHUTDOWN |

## State encoding
| Value | State | Meaning |
|-------|-------|---------|
| 3'd0 | NOMINAL | Normal operation |
| 3'd1 | WARNING | Getting warm — monitor |
| 3'd2 | THROTTLED | Clock reduced |
| 3'd3 | CRITICAL | Prepare for shutdown |
| 3'd4 | SHUTDOWN | Must shut down |

## Ports
| Port | Dir | Width | Description |
|------|-----|-------|-------------|
| clk | in | 1 | System clock |
| rst_n | in | 1 | Active-low synchronous reset |
| valid | in | 1 | Pulse: new temperature reading ready |
| temp_in | in | 8 | Temperature in °C (unsigned) |
| state | out | 3 | Current thermal state (see above) |
| throttle_en | out | 1 | High when state == THROTTLED |
| shutdown_req | out | 1 | High when state == SHUTDOWN |

## Key implementation details
- Next-state is purely combinational (priority decoder on temp_in vs thresholds)
- State register updates only on `valid` pulse
- `throttle_en` and `shutdown_req` are combinational outputs derived from `state`
- No hysteresis — transitions happen immediately when threshold is crossed

## AXI4-Lite connection (in astracore_top)
- Write: `THERMAL` register (0x08) — `[7:0]=temp_in`
- Read: `THERM_ST` register (0x84) — `[2:0]=state`, `[3]=throttle_en`, `[4]=shutdown_req`
- LED: `led[0]` = `throttle_en`
