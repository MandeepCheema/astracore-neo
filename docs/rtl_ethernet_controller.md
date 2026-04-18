# ethernet_controller — RTL Module

**File:** `rtl/ethernet_controller/ethernet_controller.v`
**Purpose:** Ethernet II frame receiver, validator, and classifier

## What it does
Receives an Ethernet frame byte-by-byte from the PHY layer, extracts key header fields, validates frame length, and classifies the frame type and MAC address type.

## Frame validation
- Valid frame length: 64–1518 bytes (Ethernet II standard)
- `frame_ok` pulses on final byte if length is valid
- `frame_err` pulses on final byte if length is out of range

## Frame type encoding
| Value | Type | EtherType |
|-------|------|-----------|
| 2'd0 | DATA (unknown) | anything else |
| 2'd1 | IPv4 | 0x0800 |
| 2'd2 | ARP | 0x0806 |
| 2'd3 | IPv6 | 0x86DD |

## MAC type encoding
| Value | Type | Condition |
|-------|------|-----------|
| 2'd0 | UNICAST | LSB of dst_mac byte 0 == 0 |
| 2'd1 | MULTICAST | LSB of dst_mac byte 0 == 1 |
| 2'd2 | BROADCAST | dst_mac == FF:FF:FF:FF:FF:FF |

## Ports
| Port | Dir | Width | Description |
|------|-----|-------|-------------|
| clk | in | 1 | System clock |
| rst_n | in | 1 | Active-low synchronous reset |
| rx_valid | in | 1 | High when rx_byte is valid |
| rx_byte | in | 8 | Incoming byte from PHY |
| rx_last | in | 1 | Assert with last byte of frame |
| frame_ok | out | 1 | Pulsed: valid frame received |
| frame_err | out | 1 | Pulsed: invalid frame length |
| ethertype | out | 16 | EtherType field (valid after frame_ok/err) |
| frame_type | out | 2 | Frame classification |
| mac_type | out | 2 | MAC address type |
| byte_count | out | 11 | Total bytes in frame |

## Key implementation details
- Bytes 0–5: destination MAC captured into `dst_mac_reg`
- Bytes 6–11: source MAC (counted but not stored)
- Byte 12: EtherType high byte stored in `et_hi`
- Byte 13: EtherType low byte → `ethertype = {et_hi, rx_byte}`
- `frame_type` decoded from the `ethertype` register (not `et_hi`+byte simultaneously)
- `rx_count` resets to 0 after `rx_last` — ready for next frame immediately
- `frame_ok` and `frame_err` are default-0, only asserted for one cycle on `rx_last`

## AXI4-Lite connection (in astracore_top)
- Write: `ETH` register (0x40) — `[0]=rx_valid`, `[1]=rx_last`, `[9:2]=rx_byte`
- Read: `ETH_ST` (0xB8) — frame_ok/err, frame_type, mac_type, byte_count; `ETYPE` (0xBC) — ethertype
- LED: `led[3]` = frame_ok (pulse-stretched 24-bit counter for visibility)
