# tmr_voter — RTL Module

**File:** `rtl/tmr_voter/tmr_voter.v`
**Purpose:** Triple Modular Redundancy majority voter for ASIL-D safety paths

## What it does
Takes three identical 32-bit results (from three independent compute lanes) and outputs the majority-voted correct result. Flags which lane (if any) produced the wrong answer.

## Vote rules
| Condition | voted | agreement | fault flags | vote_count |
|-----------|-------|-----------|-------------|------------|
| A == B == C | A | 1 | none | 3 |
| A == B != C | A | 1 | fault_c | 2 |
| A == C != B | A | 1 | fault_b | 2 |
| B == C != A | B | 1 | fault_a | 2 |
| A != B != C | A (undefined) | 0 | triple_fault | 0 |

## Ports
| Port | Dir | Width | Description |
|------|-----|-------|-------------|
| clk | in | 1 | System clock |
| rst_n | in | 1 | Active-low synchronous reset |
| valid | in | 1 | Pulse: lane values are ready |
| lane_a | in | 32 | Result from lane A |
| lane_b | in | 32 | Result from lane B |
| lane_c | in | 32 | Result from lane C |
| voted | out | 32 | Majority-voted output |
| agreement | out | 1 | At least 2 lanes agreed |
| fault_a | out | 1 | Lane A was the minority |
| fault_b | out | 1 | Lane B was the minority |
| fault_c | out | 1 | Lane C was the minority |
| triple_fault | out | 1 | All three lanes disagree |
| vote_count | out | 2 | Number of lanes that agreed (2 or 3) |

## Key implementation details
- All vote logic is purely combinational (3 equality comparisons: ab_eq, ac_eq, bc_eq)
- A==B or A==C: A wins (covers all-agree case too)
- B==C only: B wins (A was the outlier)
- Triple fault: defaults to A output (value is undefined/unreliable — caller must handle)
- `vote_count` is 0 in triple_fault case (not 1)

## AXI4-Lite connection (in astracore_top)
- Write: `TMR_A/B/C` registers (0x1C/0x20/0x24) — lane_a/b/c inputs
- Read: `TMR_RES` (0x98) — voted result; `TMR_ST` (0x9C) — agreement + fault flags + vote_count
