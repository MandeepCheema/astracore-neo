# pcie_controller — RTL Module

**File:** `rtl/pcie_controller/pcie_controller.v`
**Purpose:** PCIe link state machine and TLP header assembler

## What it does
Manages the PCIe link training sequence and builds 3-DWORD (96-bit) TLP headers for memory read, memory write, and completion-with-data transactions.

## Link states
| Value | State | Meaning |
|-------|-------|---------|
| 3'd0 | DETECT | Initial — receiver detection |
| 3'd1 | POLLING | Bit-lock and symbol lock |
| 3'd2 | CONFIG | Lane/link width negotiation |
| 3'd3 | L0 | Active — data transfer enabled |
| 3'd4 | L1 | Low-power ASPM |
| 3'd5 | L2 | Powered down |

Link advances DETECT→POLLING→CONFIG→L0 on each `link_up` pulse. `link_down` returns to DETECT immediately.

## TLP types
| tlp_type | Type | fmt | tcode |
|----------|------|-----|-------|
| 2'd0 | MEM_READ | 2'b00 | 5'h00 |
| 2'd1 | MEM_WRITE | 2'b10 | 5'h00 |
| 2'd2 | COMPLETION_DATA | 2'b10 | 5'h0A |

## TLP header layout (96 bits)
| DWORD | Bits | Contents |
|-------|------|---------|
| DW0 | [31:0] | fmt, type, reserved fields, length_dw |
| DW1 | [63:32] | req_id, tag, last_BE=0xF, first_BE=0xF |
| DW2 | [95:64] | addr[31:2] + 2'b00 (DWORD-aligned) |

Assembly takes 3 clock cycles (one DWORD per cycle).

## Ports
| Port | Dir | Width | Description |
|------|-----|-------|-------------|
| clk | in | 1 | System clock |
| rst_n | in | 1 | Active-low synchronous reset |
| link_up | in | 1 | Advance link training state |
| link_down | in | 1 | Force link to DETECT |
| tlp_start | in | 1 | Begin TLP assembly (only in L0) |
| tlp_type | in | 2 | TLP type selector |
| req_id | in | 16 | Requester ID (bus:dev:fn) |
| tag | in | 8 | Transaction tag |
| addr | in | 32 | Target address (DWORD-aligned) |
| length_dw | in | 10 | Transfer length in DWORDs |
| link_state | out | 3 | Current link state |
| busy | out | 1 | TLP assembly in progress |
| tlp_done | out | 1 | Pulsed one cycle when TLP complete |
| tlp_hdr | out | 96 | Assembled TLP header |

## Key implementation details
- `tlp_start` is ignored unless `link_state == L0`
- `tlp_done` pulses for exactly one cycle when assembly finishes
- `busy` is high during all 3 assembly cycles
- Lower 2 bits of addr are forced to 0 (DWORD alignment)

## AXI4-Lite connection (in astracore_top)
- Write: `PCIE_CTRL` (0x30), `PCIE_REQID` (0x34), `PCIE_ADDR` (0x38), `PCIE_LEN` (0x3C)
- Read: `PCIE_ST` (0xA8), `PCIE_H0/H1/H2` (0xAC/0xB0/0xB4)
