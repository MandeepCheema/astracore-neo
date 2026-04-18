# mac_array — RTL Module

**File:** `rtl/mac_array/mac_array.v`
**Purpose:** Signed INT8 multiply-accumulate unit (core AI compute leaf)

## What it does
Computes `result += a * b` each valid cycle. Represents one leaf MAC unit of the chip's 24,576-MAC array. The `clear` input resets the accumulator at the start of a new dot-product (matrix tile boundary).

## Operation
| clear | Operation |
|-------|-----------|
| 0 | `result = result + sign_extend(a * b)` |
| 1 | `result = sign_extend(a * b)` (clear then accumulate) |

## Precision
- Inputs: signed 8-bit (INT8), range -128 to +127
- Product: signed 16-bit (automatic sign extension in Verilog)
- Accumulator: signed 32-bit (no saturation — wraps on overflow, matching `np.int32`)

## Ports
| Port | Dir | Width | Description |
|------|-----|-------|-------------|
| clk | in | 1 | System clock |
| rst_n | in | 1 | Active-low synchronous reset |
| valid | in | 1 | Pulse: a and b inputs are ready |
| clear | in | 1 | Clear accumulator before adding |
| a | in | 8 | Signed 8-bit input (activation or weight) |
| b | in | 8 | Signed 8-bit input (weight or activation) |
| result | out | 32 | Signed 32-bit accumulated result |
| ready | out | 1 | Pulsed one cycle after valid — result is stable |

## Key implementation details
- `product = a * b` is combinational (Verilog signed multiply, 16-bit result)
- Product is sign-extended to 32 bits before accumulation: `{{16{product[15]}}, product}`
- `ready` is simply the registered `valid` — result is always ready exactly one cycle later
- No overflow/saturation detection — matches Python numpy int32 wrapping behaviour

## AXI4-Lite connection (in astracore_top)
- Write: `MAC` register (0x44) — `[0]=valid`, `[1]=clear`, `[9:2]=a`, `[17:10]=b`
- Read: `MAC_RES` register (0xC0) — result latched in top-level on `ready` pulse
- LED: `led[2]` = `inf_busy` (not mac directly, but inference drives MAC)
