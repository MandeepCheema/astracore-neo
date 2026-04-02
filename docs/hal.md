# HAL — Hardware Abstraction Layer

**Module:** `src/hal/`  
**Depends on:** nothing (foundation layer)  
**Status:** DONE  
**Test log:** `logs/test_hal.log`  
**Test result:** 79/79 passed ✓

---

## Purpose

The HAL is the single boundary between simulation code and all hardware concepts. Every other module accesses chip state exclusively through HAL. This enforces the ASIL-D requirement that safety-critical hardware access is centralized, auditable, and testable in isolation.

In simulation, the HAL runs as an in-process mock. When ported to the real A2 Neo silicon, only `hal/` needs to change — all modules above it remain identical.

---

## Files

| File | Description |
|------|-------------|
| `hal/device.py` | `AstraCoreDevice` — top-level chip handle, power/clock state machine |
| `hal/registers.py` | `RegisterFile` — memory-mapped register simulation |
| `hal/interrupts.py` | `InterruptController` — 32-bit IRQ controller |
| `hal/exceptions.py` | Exception hierarchy |
| `hal/__init__.py` | Public API exports |

---

## Power State Machine

```
        power_on()
OFF ─────────────► RESET ──► IDLE ◄────────────── exit_low_power()
 ▲                            │  │
 │         power_off()        │  │ start()
 └────────────────────────────┘  ▼
                              ACTIVE
                                 │
                    enter_low_power() (from IDLE or ACTIVE)
                                 ▼
                            LOW_POWER
                         (256 MACs @ 500 MHz)

Any state ──► hard reset() ──► IDLE   (except OFF)
```

---

## Register Map

| Address | Name | Access | Reset Value | Description |
|---------|------|--------|-------------|-------------|
| 0x0000 | CHIP_ID | R/O | 0xA24EE001 | Silicon identity |
| 0x0004 | CHIP_REV | R/O | 0x00000013 | Revision 1.3 |
| 0x0008 | CTRL | R/W | 0x00000000 | Global control |
| 0x000C | STATUS | R/O (HW) | 0x00000001 | Global status (bit0=ready, bit1=active, bit2=low-power) |
| 0x0010 | CLK_CTRL | R/W | 0x00000320 | Clock freq in MHz (2500–3200) |
| 0x0014 | CLK_STATUS | R/O (HW) | 0x00000320 | Actual clock (mirrors CLK_CTRL after PLL lock) |
| 0x0018 | PWR_CTRL | R/W | 0x000000FF | Power domain enables (8 domains) |
| 0x001C | PWR_STATUS | R/O (HW) | 0x000000FF | Power domain status |
| 0x0020 | RESET_CTRL | W/O | — | Write 1 to reset domain |
| 0x0024 | INT_ENABLE | R/W | 0x00000000 | Interrupt enable mask |
| 0x0028 | INT_STATUS | R/O (HW) | 0x00000000 | Pending interrupts |
| 0x002C | INT_CLEAR | W/O | — | Write 1 to clear interrupt bit |
| 0x0030 | MAC_CTRL | R/W | 0x00000001 | MAC array config |
| 0x0034 | MAC_STATUS | R/O (HW) | 0x00000000 | MAC utilization |
| 0x0038 | MEM_CTRL | R/W | 0x0000FFFF | SRAM bank enables (16 banks) |
| 0x003C | MEM_STATUS | R/O (HW) | 0x00000000 | ECC errors, bank faults |
| 0x0040 | THERMAL_STATUS | R/O (HW) | 0x00000019 | Temperature in °C |
| 0x0044 | THERMAL_THRESH | R/W | 0x0000007D | Throttle threshold (125°C) |
| 0x0048 | SECURITY_STATUS | R/O (HW) | 0x00000000 | Secure boot / TEE flags |
| 0x004C | OTA_CTRL | R/W | 0x00000000 | OTA control |

> **R/O (HW):** read-only to software; written internally by the device simulation via `_hw_write()`.

---

## Interrupt Map

| Bit | Name | Trigger |
|-----|------|---------|
| 0 | MAC_DONE | Compute array job complete |
| 1 | DMA_DONE | DMA transfer complete |
| 2 | MEM_ECC_ERR | Uncorrectable ECC error |
| 3 | THERMAL_WARN | Temperature above threshold |
| 4 | THERMAL_CRIT | Critical thermal shutdown |
| 5 | SAFETY_TMR | Safety watchdog expired |
| 6 | V2X_RX | V2X frame received |
| 7 | OTA_READY | OTA package validated |
| 8 | DMS_ALERT | Driver monitoring alert |
| 9 | SECURE_BOOT_OK | Secure boot chain completed |
| 10–31 | — | Reserved |

---

## API Reference

### `AstraCoreDevice`

```python
dev = AstraCoreDevice(name="astracore-neo-0")
```

| Method | Description |
|--------|-------------|
| `power_on()` | OFF → IDLE. Initializes registers and IRQ controller. |
| `power_off()` | Any → OFF. |
| `reset()` | Any (except OFF) → IDLE. Full register + IRQ reset. |
| `start()` | IDLE → ACTIVE. |
| `stop()` | ACTIVE → IDLE. |
| `enter_low_power()` | IDLE/ACTIVE → LOW_POWER. Drops clock to 500 MHz. |
| `exit_low_power()` | LOW_POWER → IDLE. Restores clock to 3.2 GHz. |
| `set_clock_ghz(f)` | DVFS: set clock 2.5–3.2 GHz (or 0.5 in low-power). |
| `clock_ghz` | Current clock frequency (float). |
| `state` | Current `PowerState` enum value. |
| `chip_id` | Read CHIP_ID register. |
| `chip_rev` | Read CHIP_REV register. |
| `uptime_seconds` | Seconds since `power_on()`. |
| `dev.regs` | `RegisterFile` instance. |
| `dev.irq` | `InterruptController` instance. |

### `RegisterFile`

```python
reg = RegisterFile()
reg.write(0x0008, 0xDEAD_BEEF)
val = reg.read(0x0008)
reg.write_field(0x0008, 7, 0, 0xFF)      # write bits [7:0]
val = reg.read_field(0x0008, 31, 16)     # read bits [31:16]
reg.named_write("CTRL", 0x1)
val = reg.named_read("STATUS")
reg.reset()
d = reg.dump()                            # {name: value}
```

### `InterruptController`

```python
irq = InterruptController()
irq.enable(IRQ_MAC_DONE)
irq.register_handler(IRQ_MAC_DONE, lambda n: print(f"IRQ {n} fired"))
irq.fire(IRQ_MAC_DONE)          # dispatches handler synchronously
irq.is_pending(IRQ_MAC_DONE)   # → True
irq.clear(IRQ_MAC_DONE)
irq.disable(IRQ_MAC_DONE)
irq.reset()                     # clears all state
```

---

## Design Decisions

1. **`_hw_write()` vs external `write()`** — Status registers (STATUS, CLK_STATUS, etc.) are R/O to software but need to be updated by the simulation to reflect hardware state. Rather than removing the R/O protection, a separate `_hw_write()` path maintains the distinction: external code hitting a R/O register gets an exception; the device simulation can update it freely.

2. **Synchronous interrupt dispatch** — Handlers are called inline on `fire()` rather than queued. This keeps the simulation deterministic and avoids threading complexity. Higher-level modules can add their own queuing if needed.

3. **Clock encoded as MHz integer** — CLK_CTRL stores the clock as an integer MHz value (e.g. 3200 for 3.2 GHz) matching what a real register word would hold. The float `clock_ghz` property is a convenience derived from this.

4. **No threading in HAL** — HAL is single-threaded. Concurrency is a concern for higher modules (DMS always-on thread, telemetry background loop) — they will coordinate through the IRQ controller.

---

## Test Coverage Summary

| Test Class | Tests | Result |
|------------|-------|--------|
| TestDeviceLifecycle | 12 | ✓ All pass |
| TestPowerStateTransitions | 16 | ✓ All pass |
| TestClockDvfs | 10 | ✓ All pass |
| TestRegisterBasic | 9 | ✓ All pass |
| TestRegisterBitfields | 8 | ✓ All pass |
| TestRegisterNamed | 5 | ✓ All pass |
| TestInterruptEnableDisable | 7 | ✓ All pass |
| TestInterruptFireClear | 5 | ✓ All pass |
| TestInterruptHandlers | 6 | ✓ All pass |
| TestInterruptReset | 3 | ✓ All pass |
| **Total** | **79** | **79/79 ✓** |

---

## Next Module

→ **Module 2: Memory** (`src/memory/`) — SRAM banks, DMA engine, neural compression.  
Depends on: `hal/`
