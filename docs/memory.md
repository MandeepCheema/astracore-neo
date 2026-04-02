# Memory — SRAM, DMA, Neural Compression

**Module:** `src/memory/`  
**Depends on:** `hal/`  
**Status:** DONE  
**Test log:** `logs/test_memory.log`  
**Test result:** 73/73 passed ✓

---

## Purpose

Models the AstraCore Neo's on-chip memory subsystem:

- **128 MB SRAM** — 16 banks × 8 MB, ECC, dual-port
- **DMA Engine** — 8-channel, prefetch-aware, cache-coherent, 2D strided
- **Neural Compressor** — INT8/INT4 delta+RLE encoding, 3–5× ratio on weight tensors

All components integrate with `AstraCoreDevice` to mirror state into HAL registers and fire interrupts.

---

## Files

| File | Description |
|------|-------------|
| `memory/sram.py` | `SRAMController`, `SRAMBank` — 128 MB SRAM simulation |
| `memory/dma.py` | `DMAEngine`, `DMADescriptor`, `DMAChannel` — transfer engine |
| `memory/compression.py` | `NeuralCompressor` — INT8/INT4 neural-aware codec |
| `memory/exceptions.py` | `MemoryError`, `EccError`, `BankError`, `DmaError`, `CompressionError` |
| `memory/__init__.py` | Public API exports |

---

## SRAM Architecture

```
SRAMController (128 MB flat address space)
│
├── Bank 0   [0x0000_0000 – 0x007F_FFFF]  8 MB
├── Bank 1   [0x0080_0000 – 0x00FF_FFFF]  8 MB
├── Bank 2   [0x0100_0000 – 0x017F_FFFF]  8 MB
├── ...
└── Bank 15  [0x0780_0000 – 0x07FF_FFFF]  8 MB

Address decode:
  bank_id = addr >> 23          (bits [27:23])
  offset  = addr &  0x7FFFFF    (bits [22:0])
```

### ECC Model

| Error type | Behaviour |
|-----------|-----------|
| Single-bit (SECDED correctable) | Transparently corrected on read; `ecc_corrections` counter incremented |
| Double-bit (SECDED detectable) | Raises `EccError(addr, bank)` — must be handled by safety layer |

Fault injection API for ASIL-D testing:
```python
sram.bank(0).inject_single_bit_error(offset=0x100, bit=3)
sram.bank(0).inject_double_bit_error(offset=0x200)
```

### Dual-Port Access

```python
result = sram.dual_port_transfer(
    read_addr=0x000000,  read_len=64,
    write_addr=0x800000, write_data=payload,
)
# Cross-bank: true parallel (no serialisation)
# Same-bank:  write-first, then read (serialised)
```

---

## DMA Engine

### Descriptor fields

| Field | Type | Description |
|-------|------|-------------|
| `src_addr` | int | Source flat byte address |
| `dst_addr` | int | Destination flat byte address |
| `length` | int | Bytes per row |
| `rows` | int | Number of rows (1 = contiguous) |
| `src_stride` | int | Source row stride in bytes (0 = contiguous) |
| `dst_stride` | int | Destination row stride in bytes |
| `flags` | TransferFlags | `NONE`, `PREFETCH`, `INVALIDATE` |
| `channel_id` | int | DMA channel 0–7 |

### 2D Strided Transfer (tensor extraction)

```python
# Copy 4 rows × 8 bytes from a 16-byte-stride source
desc = DMADescriptor(
    src_addr=0, dst_addr=0x800000,
    length=8, rows=4, src_stride=16, dst_stride=8,
)
dma.submit(desc)
dma.execute_all()  # → 32 bytes transferred
```

### Cache Coherency

```
PREFETCH flag  → marks destination cache lines as valid after write
INVALIDATE flag → invalidates destination cache lines after write
Manual:         dma.invalidate_cache(addr, length)
```

### IRQ integration

Every completed transfer fires `IRQ_DMA_DONE` (bit 1) on the device interrupt controller. Register a handler:

```python
dev.irq.enable(IRQ_DMA_DONE)
dev.irq.register_handler(IRQ_DMA_DONE, lambda n: on_dma_done())
```

### Channel state machine

```
IDLE → (submit + execute) → BUSY → IDLE
                                 ↘ FAULT  (on transfer error)
FAULT → reset_channel(id) → IDLE
```

---

## Neural Compression

### Algorithm

```
encode(data, INT8):
  1. Delta encode   — store byte-to-byte differences
  2. RLE            — collapse runs of identical deltas
  3. Prepend header (magic + mode + original_length)

encode(data, INT4):
  1. Clamp to lower nibble  (b & 0x0F)
  2. Delta encode (mod 16)
  3. Nibble RLE             — (count[4b] | value[4b]) per byte
  4. Prepend header
```

### API

```python
nc = NeuralCompressor()

# Compress
compressed = nc.encode(weights, CompressionMode.INT8)
print(nc.last_ratio)    # e.g. 3.1

# Decompress
original = nc.decode(compressed)

# Stats
print(nc.overall_ratio)
nc.reset_stats()
```

### Typical Ratios

| Data pattern | INT8 ratio | INT4 ratio |
|-------------|-----------|-----------|
| All-zeros (e.g. zero-padded layer) | ~200× | ~400× |
| Uniform value | ~100× | ~200× |
| Smooth gradients (typical weights) | 2–4× | 3–6× |
| Random noise | ~0.9× (expansion) | ~0.9× |

> INT4 is only lossless for data already in 0–15 range. Applying it to arbitrary bytes clamps the upper nibble — use the quantiser (Module 4) before INT4 compression.

---

## HAL Register Integration

| Register | Address | Effect |
|----------|---------|--------|
| `MEM_CTRL` | 0x0038 | Mirror of bank enable mask (16 bits); updated on every `enable_bank()` / `disable_bank()` / `reset()` call |

---

## API Reference

### SRAMController

```python
ctrl = SRAMController(dev=None)
ctrl.read(addr, length) → bytes
ctrl.write(addr, data)
ctrl.read_word(addr) → int
ctrl.write_word(addr, value)
ctrl.dual_port_transfer(read_addr, read_len, write_addr, write_data) → bytes
ctrl.enable_bank(bank_id)
ctrl.disable_bank(bank_id)
ctrl.bank_enabled(bank_id) → bool
ctrl.enabled_bank_mask() → int
ctrl.bank(bank_id) → SRAMBank
ctrl.total_ecc_corrections() → int
ctrl.total_ecc_detections() → int
ctrl.reset()
```

### DMAEngine

```python
engine = DMAEngine(sram=None, dev=None)
engine.submit(desc)
engine.submit_many(descs)
engine.execute_all() → int          # returns total bytes transferred
engine.execute_channel(id) → int
engine.channel(id) → DMAChannel
engine.is_idle(id) → bool
engine.pending_count() → int
engine.cache_is_valid(addr) → bool
engine.invalidate_cache(addr, length)
engine.reset()
engine.reset_channel(id)
```

### NeuralCompressor

```python
nc = NeuralCompressor()
nc.encode(data, mode) → bytes
nc.decode(compressed) → bytes
nc.last_ratio           # float — ratio of last encode
nc.overall_ratio        # float — cumulative ratio
nc.total_bytes_in       # int
nc.total_bytes_out      # int
nc.reset_stats()
```

---

## Design Decisions

1. **ECC via injection, not Hamming codes** — Real SECDED is a hardware function. Simulation models it as an address-tagged fault dict so tests can exercise correction/detection paths deterministically without a full bit-manipulation encoder.

2. **DMA is synchronous in simulation** — `execute_all()` runs transfers immediately. Real DMA is async; the IRQ mechanism preserves that interface so upper modules written against it will work when the engine is made async.

3. **INT4 compressor clamps, not truncates** — `b & 0x0F` preserves the lower nibble; the quantiser (Module 4) is responsible for ensuring values are already in 0–15 range before INT4 compression is applied.

4. **`_hw_write` not needed here** — MEM_CTRL (0x0038) is R/W by software, so `SRAMController` uses the normal `regs.write()` path.

---

## Test Coverage Summary

| Test Class | Tests | Result |
|------------|-------|--------|
| TestSRAMBankBasics | 8 | ✓ All pass |
| TestSRAMReadWrite | 12 | ✓ All pass |
| TestSRAMEcc | 6 | ✓ All pass |
| TestSRAMDualPort | 3 | ✓ All pass |
| TestSRAMHalSync | 3 | ✓ All pass |
| TestDMASubmission | 5 | ✓ All pass |
| TestDMAExecution | 8 | ✓ All pass |
| TestDMAIrq | 2 | ✓ All pass |
| TestDMACache | 3 | ✓ All pass |
| TestDMAReset | 4 | ✓ All pass |
| TestCompressionInt8 | 7 | ✓ All pass |
| TestCompressionInt4 | 5 | ✓ All pass |
| TestCompressionErrors | 4 | ✓ All pass |
| TestCompressionStats | 3 | ✓ All pass |
| **Total** | **73** | **73/73 ✓** |

---

## Next Module

→ **Module 3: Compute** (`src/compute/`) — MAC array (24,576 units), sparsity engine, transformer engine.  
Depends on: `hal/`, `memory/`
