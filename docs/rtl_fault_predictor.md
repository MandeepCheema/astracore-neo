# fault_predictor — RTL Module

**File:** `rtl/fault_predictor/fault_predictor.v`
**Purpose:** Rolling-window fault risk classifier with spike detection

## What it does
Tracks a 16-sample rolling window of a health metric. Classifies risk level based on:
1. How far the value exceeds the warning threshold
2. Whether a spike is detected (value > rolling mean + offset)

## Parameters
| Parameter | Default | Meaning |
|-----------|---------|---------|
| WARN_THRESH | 16'd50 | Warning starts here |
| CRITICAL_THRESH | 16'd100 | Critical starts here |
| WINDOW_SIZE | 16 | Rolling window depth (must be power of 2) |
| SPIKE_OFFSET | 16'd30 | Spike = value > mean + 30 |

## Risk levels
| Value | Level | Condition |
|-------|-------|-----------|
| 3'd0 | NONE | value < WARN_THRESH |
| 3'd1 | LOW | above warn, 0–30% into warn–critical range |
| 3'd2 | MEDIUM | above warn, 30–70% — OR spike detected |
| 3'd3 | HIGH | above warn, 70–100% of range |
| 3'd4 | CRITICAL | value >= CRITICAL_THRESH |

## Ports
| Port | Dir | Width | Description |
|------|-----|-------|-------------|
| clk | in | 1 | System clock |
| rst_n | in | 1 | Active-low synchronous reset |
| valid | in | 1 | Pulse: new metric value ready |
| value | in | 16 | Unsigned metric value |
| risk | out | 3 | Risk level (0–4) |
| alarm | out | 1 | High when risk >= 3 (HIGH or CRITICAL) |
| rolling_mean | out | 16 | Running mean of last WINDOW_SIZE samples |

## Key implementation details
- Window sum maintained incrementally: `sum = sum - window[oldest] + value` (O(1) per cycle)
- Mean = `window_sum >> log2(WINDOW_SIZE)` — only works correctly for power-of-2 window sizes
- Spike detection only enabled after 4+ samples (`fill_count >= 4`) — matches Python model behaviour
- `alarm` is a combinational output: `risk >= 3'd3`
- **Timing risk note:** The range multiply (`range * 7 / 10`, `range * 3 / 10`) in the combinational path is the most complex logic in the design — main candidate for pipelining if 50 MHz timing fails in OpenLane

## AXI4-Lite connection (in astracore_top)
- Write: `FAULT` register (0x28) — `[15:0]=value`
- Read: `FAULT_ST` register (0xA0) — `[2:0]=risk`, `[3]=alarm`, `[19:4]=rolling_mean`
- LED: `led[1]` = `alarm`
