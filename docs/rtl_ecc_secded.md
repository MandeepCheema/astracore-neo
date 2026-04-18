# ecc_secded — RTL Module

**File:** `rtl/ecc_secded/ecc_secded.v`
**Purpose:** Hamming(72,64) SECDED — Single Error Correct, Double Error Detect

## What it does
Protects 64-bit memory words with 8 parity bits. Two modes:
- **Encode (mode=0):** Given 64-bit data, compute 8 parity bits
- **Decode (mode=1):** Given 64-bit data + 8 received parity bits, detect and correct errors

## Hamming scheme
- 7 Hamming parity bits (h[6:0]): each covers a subset of data bits defined by bit position
- 1 overall parity bit (p7): XOR of all 64 data bits + all 7 Hamming bits → ensures even parity across the full 72-bit codeword
- `parity[7:0] = {p7, h[6:0]}`

## Error classification (decode mode)
| Condition | Classification |
|-----------|---------------|
| syndrome == 0 AND overall parity == 0 | No error |
| overall parity == 1 | Single-bit error (correctable) |
| syndrome != 0 AND overall parity == 0 | Double-bit error (detected, uncorrectable) |

## Ports
| Port | Dir | Width | Description |
|------|-----|-------|-------------|
| clk | in | 1 | System clock |
| rst_n | in | 1 | Active-low synchronous reset |
| valid | in | 1 | Pulse: inputs are ready |
| mode | in | 1 | 0=encode, 1=decode |
| data_in | in | 64 | Data word |
| parity_in | in | 8 | Received parity (decode only) |
| data_out | out | 64 | Pass-through (encode) or corrected data (decode) |
| parity_out | out | 8 | Computed parity (encode) or recomputed syndrome (decode) |
| single_err | out | 1 | Single-bit error detected and corrected |
| double_err | out | 1 | Double-bit error detected (uncorrectable) |
| corrected | out | 1 | Alias for single_err |
| err_pos | out | 7 | 1-indexed error bit position (0 = parity bit flipped) |

## Key implementation details
- Hamming bits computed combinationally over nested for-loops (synthesises to XOR trees)
- Correction: syndrome value is directly the 1-indexed position of the bad data bit
- Error position 0 or >64 means a parity bit (not a data bit) was flipped
- 32-bit AXI interface uses two registers (ECC_LO / ECC_HI) to pass the 64-bit data_in

## AXI4-Lite connection (in astracore_top)
- Write: `ECC_LO` (0x10), `ECC_HI` (0x14) = data_in[63:0]; `ECC_CTRL` (0x18) `[0]=mode`, `[15:8]=parity_in`
- Read: `ECC_ST` (0x8C) — flags + err_pos + parity_out; `ECC_DLO/DHI` (0x90/0x94) — data_out
