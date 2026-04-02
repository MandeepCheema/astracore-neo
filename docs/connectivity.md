# Module 10 — Connectivity

## Overview

The Connectivity subsystem models the four on-chip communication interfaces of the AstraCore Neo chip. Each interface is an independent controller; `ConnectivityManager` ties them together under a single initialisation and status API.

| Interface | Class | Standard | Role |
|-----------|-------|----------|------|
| CAN-FD | `CANFDController` | ISO 11898-1 | Vehicle ECU bus (ADAS ↔ gateway) |
| Ethernet | `EthernetController` | 100/1000BASE-T1 | High-bandwidth sensor/backbone bus |
| PCIe | `PCIeController` | PCIe Gen3 | SoC ↔ accelerator high-speed link |
| V2X | `V2XController` | DSRC / C-V2X | Vehicle-to-Everything radio |

---

## CAN-FD Controller

### Frame Format

```python
CANFrame(can_id=0x100, id_format=CANIDFormat.STANDARD, data=b"hello!!!")
CANFrame(can_id=0x1FFFFFFF, id_format=CANIDFormat.EXTENDED, data=b"ext", is_fd=True)
```

- **STANDARD**: 11-bit CAN ID (0–0x7FF)
- **EXTENDED**: 29-bit CAN ID (0–0x1FFFFFFF)
- **Classic CAN**: `is_fd=False` — max 8 bytes
- **CAN-FD**: `is_fd=True` — max 64 bytes; DLC 9-15 map to 12/16/20/24/32/48/64 bytes

### TX Priority Queue

Frames are inserted in CAN ID priority order (lower ID = higher priority). `transmit_next()` pops the highest-priority frame.

### Error Counters and Bus State

| Event | Counter | Change |
|-------|---------|--------|
| Successful TX | TEC | -1 (floor 0) |
| TX error | TEC | +8 |
| RX error | REC | +1 |

| Condition | Bus State |
|-----------|-----------|
| TEC < 128 AND REC < 128 | ERROR_ACTIVE |
| TEC >= 128 OR REC >= 128 | ERROR_PASSIVE |
| TEC >= 256 | BUS_OFF |

`bus_off_recovery()` resets TEC=0, REC=0 and returns to ERROR_ACTIVE (simulates 128×11 recessive bit sequence).

---

## Ethernet Controller

### Frame Validation

`EthernetFrame.__post_init__` validates:
- `dest_mac` and `src_mac` must be exactly 6 bytes
- `ethertype` must be in [0, 0xFFFF]
- `payload` must be ≤ 1500 bytes

### MAC Filtering

`receive()` accepts frames when `dest_mac` matches:
1. This controller's MAC (unicast)
2. `BROADCAST_MAC` (FF:FF:FF:FF:FF:FF)
3. Any multicast address (LSB of first byte = 1)

Frames addressed to another unicast MAC are silently dropped (returns `False`).

### Link State

TX requires `link_state == UP`. Initial state is DOWN. Call `link_up()` to enable transmission.

---

## PCIe Controller

### Link Training

```python
ep = PCIeController(device_id=0x1000, num_lanes=4)
ep.train_link()   # DETECT → POLLING → CONFIG → L0
assert ep.link_state == PCIeLinkState.L0
```

Low-power state: `enter_l1()` / `exit_l1()` (both require L0 as prerequisite/current state respectively).

### BAR Memory Regions

```python
ep.add_bar(bar_index=0, base_addr=0x10000000, size=4096)
ep.mmio_write(0x10000000, b"\xDE\xAD\xBE\xEF")
data = ep.mmio_read(0x10000000, 4)
```

- BAR size must be a power of 2
- Base address must be size-aligned
- Up to 6 BARs (BAR0–BAR5)
- All MMIO operations require L0 link state

### TLP Processing

`receive_tlp()` dispatches inbound TLPs:
- `MEM_WRITE` → writes to the matching BAR; returns `None`
- `MEM_READ` → reads from matching BAR; returns `COMPLETION_DATA` TLP with the data
- Unknown BAR address → increments `error_count`; returns error `COMPLETION`

---

## V2X Controller

### DSRC Channels

Valid channels: 172, 174, 176, 178 (CCH), 180, 182, 184. CCH=178 is the mandatory Control Channel for BSM messages.

### Message Types (SAE J2735)

| Type | Enum | Purpose |
|------|------|---------|
| Basic Safety Message | `BSM` | Position, speed, heading, brake status |
| Signal Phase & Timing | `SPaT` | Traffic light phase info |
| Map Data | `MAP` | Intersection geometry |
| Personal Safety Msg | `PSM` | Pedestrian/cyclist presence |

### RX Filtering

`receive()` rejects messages where:
1. `msg.channel != self.channel` (tuned channel mismatch)
2. `msg.rssi_dbm < rssi_threshold_dbm` (default −90 dBm) — weak signal

Rejected-by-RSSI count is tracked in `filtered_rssi`.

---

## ConnectivityManager

One-stop initialisation. Each `init_*()` raises `ConnectivityBaseError` if called more than once.

```python
mgr = ConnectivityManager()
can = mgr.init_canfd(node_id=1)
eth = mgr.init_ethernet(mac_address=bytes.fromhex("aabbccddeeff"))
pcie = mgr.init_pcie(device_id=0x1000, num_lanes=4)
v2x = mgr.init_v2x(node_id=0xCAFE_BABE)

print(mgr.link_status())
# {'canfd': {'bus_state': 'ERROR_ACTIVE', 'tec': 0, 'rec': 0},
#  'ethernet': {'link': 'DOWN', 'tx': 0, 'rx': 0}, ...}
```

`any_link_up()` returns True when at least one interface is in an active state (CAN not BUS_OFF, Ethernet UP, PCIe L0, or V2X initialised).

---

## Exception Hierarchy

```
ConnectivityBaseError
├── CANError      — CAN-FD frame/controller errors
├── EthernetError — Ethernet frame/controller errors
├── PCIeError     — PCIe link/BAR/TLP errors
└── V2XError      — V2X channel/message errors
```

---

## Test Coverage (75/75)

| Category | Tests |
|----------|-------|
| CANFrame validation (DLC, ID format, overflow) | 7 |
| CAN-FD TX/RX (send, priority, FIFO, available) | 8 |
| CAN-FD bus state (escalation, recovery, block) | 8 |
| EthernetFrame validation | 4 |
| Ethernet link state and TX | 4 |
| Ethernet RX (unicast, broadcast, filter, overflow, multicast) | 7 |
| PCIe link training and power states | 5 |
| PCIe BAR management and MMIO | 7 |
| PCIe TLP send/receive/completion | 3 |
| V2X broadcast, channel filter, RSSI filter, queue | 11 |
| ConnectivityManager init, double-init, status, any_link_up | 11 |
| **Total** | **75** |

---

## RTL Notes

- **CAN-FD** → Arbitration logic uses a priority encoder over the TX FIFO sorted by CAN ID. TEC/REC are 9-bit saturating counters with threshold comparators driving the 2-bit bus-state register.
- **Ethernet** → Standard 1000BASE-T1 MAC; the RX address filter is a 48-bit comparator plus multicast LSB check. The RX FIFO depth corresponds to the `rx_buffer_size` parameter.
- **PCIe** → Link training is a standard LTSSM. BAR decode is a set of address range comparators. TLP processing is a small state machine — write path goes directly to BAR SRAM; read path triggers a completion generator.
- **V2X** → DSRC radio sits off-chip; the controller here is the MAC layer — a channel register, a TX descriptor queue, and an RX FIFO with two filter comparators (channel number, RSSI threshold register).
