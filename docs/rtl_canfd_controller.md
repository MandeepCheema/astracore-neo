# canfd_controller — RTL Module

**File:** `rtl/canfd_controller/canfd_controller.v`
**Purpose:** ISO 11898-1 CAN-FD error counter and bus-state FSM

## What it does
Maintains the two CAN-FD error counters (TEC and REC) and tracks bus state according to the CAN specification. Supports bus-off recovery.

## Error counters
| Counter | Width | Increment | Decrement |
|---------|-------|-----------|-----------|
| TEC (Transmit Error Counter) | 9-bit | +8 on `tx_error` | -1 on `tx_success` (floor 0) |
| REC (Receive Error Counter) | 8-bit | +1 on `rx_error` | — (saturates at 255) |

## Bus states
| Value | State | Condition |
|-------|-------|-----------|
| 2'b00 | ERROR_ACTIVE | TEC < 128 AND REC < 128 |
| 2'b01 | ERROR_PASSIVE | TEC >= 128 OR REC >= 128 (but TEC < 256) |
| 2'b10 | BUS_OFF | TEC >= 256 |

## Ports
| Port | Dir | Width | Description |
|------|-----|-------|-------------|
| clk | in | 1 | System clock |
| rst_n | in | 1 | Active-low synchronous reset |
| tx_success | in | 1 | TX frame sent successfully — decrement TEC |
| tx_error | in | 1 | TX error — increment TEC by 8 |
| rx_error | in | 1 | RX error — increment REC by 1 |
| bus_off_recovery | in | 1 | Initiate recovery (only effective in BUS_OFF) |
| tec | out | 9 | Transmit Error Counter |
| rec | out | 8 | Receive Error Counter |
| bus_state | out | 2 | Current bus state |

## Key implementation details
- All next-state logic is combinational; registers update every clock cycle (no `valid` gating)
- Bus-off recovery resets both TEC and REC to 0 and returns to ERROR_ACTIVE immediately
- TEC/REC adjustments are blocked while in BUS_OFF (except recovery)
- Input strobes (`tx_success`, `tx_error`, `rx_error`, `bus_off_recovery`) auto-clear next cycle in the top-level (wreg_canfd[3:0] clears each cycle)

## AXI4-Lite connection (in astracore_top)
- Write: `CANFD` register (0x0C) — `[0]=tx_success`, `[1]=tx_error`, `[2]=rx_error`, `[3]=bus_off_recovery`
- Read: `CANFD_ST` register (0x88) — `[8:0]=tec`, `[17:9]=rec`, `[19:18]=bus_state`
