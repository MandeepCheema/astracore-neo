"""
AstraCore Neo Memory — DMA Engine.

Simulates the chip's prefetch-aware, cache-coherent DMA controller:
  - 8 independent DMA channels
  - Descriptor-based transfers (src, dst, size, stride, flags)
  - 2D strided transfers for tensor data (row × stride)
  - Prefetch queue per channel
  - Cache-line invalidation model (64-byte lines)
  - Transfer completion fires IRQ_DMA_DONE via device interrupt controller
  - Overlapping transfers on different channels are independent
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional
from .exceptions import DmaError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CHANNELS       = 8
CACHE_LINE_BYTES   = 64
MAX_PREFETCH_DEPTH = 4    # descriptors queued ahead per channel

# IRQ bit for DMA done (matches hal/interrupts.py)
IRQ_DMA_DONE = 1


# ---------------------------------------------------------------------------
# Transfer descriptor
# ---------------------------------------------------------------------------

class TransferFlags(Enum):
    NONE        = 0
    PREFETCH    = 1   # hint: prefetch next cache lines
    INVALIDATE  = 2   # invalidate cache lines after write
    SYNC        = 4   # wait for completion before returning


@dataclass
class DMADescriptor:
    """Single DMA transfer descriptor."""
    src_addr:   int             # source flat byte address
    dst_addr:   int             # destination flat byte address
    length:     int             # number of bytes to transfer
    src_stride: int = 0         # 0 = contiguous; >0 = 2D row stride
    dst_stride: int = 0
    rows:       int = 1         # number of rows (for 2D transfers)
    flags:      TransferFlags = TransferFlags.NONE
    channel_id: int = 0


# ---------------------------------------------------------------------------
# Channel state
# ---------------------------------------------------------------------------

class ChannelState(Enum):
    IDLE    = auto()
    BUSY    = auto()
    FAULT   = auto()


@dataclass
class DMAChannel:
    channel_id: int
    state: ChannelState = ChannelState.IDLE
    bytes_transferred: int = 0
    transfers_completed: int = 0
    last_descriptor: Optional[DMADescriptor] = None
    prefetch_queue: List[DMADescriptor] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cache model (simple invalidation tracker)
# ---------------------------------------------------------------------------

class _CacheModel:
    """
    Minimal cache coherency model.

    Tracks which cache lines are 'valid' (loaded) vs 'invalid'.
    A write with INVALIDATE flag marks the affected lines invalid.
    A subsequent read from an invalid line triggers a simulated reload.
    """

    def __init__(self) -> None:
        self._valid: set = set()

    def load(self, addr: int, length: int) -> None:
        for line in self._lines(addr, length):
            self._valid.add(line)

    def invalidate(self, addr: int, length: int) -> None:
        for line in self._lines(addr, length):
            self._valid.discard(line)

    def is_valid(self, addr: int) -> bool:
        return (addr // CACHE_LINE_BYTES) in self._valid

    def _lines(self, addr: int, length: int):
        start = addr // CACHE_LINE_BYTES
        end   = (addr + length - 1) // CACHE_LINE_BYTES
        return range(start, end + 1)

    def reset(self) -> None:
        self._valid.clear()


# ---------------------------------------------------------------------------
# DMA Engine
# ---------------------------------------------------------------------------

class DMAEngine:
    """
    Prefetch-aware, cache-coherent DMA engine.

    Usage::

        dma = DMAEngine(sram_ctrl, dev)
        desc = DMADescriptor(src_addr=0x000000, dst_addr=0x800000, length=4096)
        dma.submit(desc)
        dma.execute_all()
    """

    def __init__(self, sram=None, dev=None) -> None:
        """
        *sram* — SRAMController (optional; if None, transfers are no-ops).
        *dev*  — AstraCoreDevice (optional; used to fire IRQ_DMA_DONE).
        """
        self._sram = sram
        self._dev  = dev
        self._channels: Dict[int, DMAChannel] = {
            i: DMAChannel(channel_id=i) for i in range(NUM_CHANNELS)
        }
        self._cache = _CacheModel()
        self._pending: List[DMADescriptor] = []   # global submission queue
        self.total_bytes_transferred: int = 0

    # ------------------------------------------------------------------
    # Submit descriptors
    # ------------------------------------------------------------------

    def submit(self, desc: DMADescriptor) -> None:
        """Enqueue a transfer descriptor for the specified channel."""
        if not (0 <= desc.channel_id < NUM_CHANNELS):
            raise DmaError(f"Invalid channel_id {desc.channel_id}")
        if desc.length <= 0:
            raise DmaError(f"DMA length must be > 0, got {desc.length}")
        if desc.rows < 1:
            raise DmaError(f"DMA rows must be >= 1, got {desc.rows}")
        ch = self._channels[desc.channel_id]
        if ch.state == ChannelState.FAULT:
            raise DmaError(f"Channel {desc.channel_id} is in FAULT state — reset required")
        ch.prefetch_queue.append(desc)
        self._pending.append(desc)

    def submit_many(self, descriptors: List[DMADescriptor]) -> None:
        for d in descriptors:
            self.submit(d)

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute_all(self) -> int:
        """
        Execute all pending descriptors in submission order.
        Returns total bytes transferred in this call.
        """
        transferred = 0
        while self._pending:
            desc = self._pending.pop(0)
            transferred += self._execute_one(desc)
        return transferred

    def execute_channel(self, channel_id: int) -> int:
        """Execute all pending descriptors for a specific channel."""
        if not (0 <= channel_id < NUM_CHANNELS):
            raise DmaError(f"Invalid channel_id {channel_id}")
        channel_descs = [d for d in self._pending if d.channel_id == channel_id]
        for d in channel_descs:
            self._pending.remove(d)
        transferred = 0
        for d in channel_descs:
            transferred += self._execute_one(d)
        return transferred

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def channel(self, channel_id: int) -> DMAChannel:
        if not (0 <= channel_id < NUM_CHANNELS):
            raise DmaError(f"Invalid channel_id {channel_id}")
        return self._channels[channel_id]

    def is_idle(self, channel_id: int) -> bool:
        return self._channels[channel_id].state == ChannelState.IDLE

    def pending_count(self) -> int:
        return len(self._pending)

    # ------------------------------------------------------------------
    # Cache access
    # ------------------------------------------------------------------

    def cache_is_valid(self, addr: int) -> bool:
        return self._cache.is_valid(addr)

    def invalidate_cache(self, addr: int, length: int) -> None:
        self._cache.invalidate(addr, length)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        for ch in self._channels.values():
            ch.state = ChannelState.IDLE
            ch.bytes_transferred = 0
            ch.transfers_completed = 0
            ch.last_descriptor = None
            ch.prefetch_queue.clear()
        self._pending.clear()
        self._cache.reset()
        self.total_bytes_transferred = 0

    def reset_channel(self, channel_id: int) -> None:
        ch = self._channels[channel_id]
        ch.state = ChannelState.IDLE
        ch.prefetch_queue.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute_one(self, desc: DMADescriptor) -> int:
        ch = self._channels[desc.channel_id]
        ch.state = ChannelState.BUSY
        ch.last_descriptor = desc

        try:
            transferred = self._do_transfer(desc)
        except Exception as e:
            ch.state = ChannelState.FAULT
            raise DmaError(f"DMA fault on channel {desc.channel_id}: {e}") from e

        ch.state = ChannelState.IDLE
        ch.bytes_transferred += transferred
        ch.transfers_completed += 1
        # Remove from prefetch queue
        if desc in ch.prefetch_queue:
            ch.prefetch_queue.remove(desc)
        self.total_bytes_transferred += transferred

        # Fire IRQ if device attached
        if self._dev is not None:
            self._dev.irq.fire(IRQ_DMA_DONE)

        return transferred

    def _do_transfer(self, desc: DMADescriptor) -> int:
        """Perform the actual memory copy via SRAM controller."""
        if self._sram is None:
            # No SRAM attached — simulate transfer as a no-op but track bytes
            return desc.length * desc.rows

        transferred = 0
        if desc.rows == 1 or desc.src_stride == 0:
            # Contiguous transfer
            data = self._sram.read(desc.src_addr, desc.length)
            # Prefetch: mark source lines valid
            if desc.flags == TransferFlags.PREFETCH:
                self._cache.load(desc.src_addr, desc.length)
            self._sram.write(desc.dst_addr, data)
            # Invalidate destination cache lines if requested
            if desc.flags == TransferFlags.INVALIDATE:
                self._cache.invalidate(desc.dst_addr, desc.length)
            else:
                self._cache.load(desc.dst_addr, desc.length)
            transferred = desc.length
        else:
            # 2D strided transfer (e.g. tensor row extraction)
            for row in range(desc.rows):
                src = desc.src_addr + row * desc.src_stride
                dst = desc.dst_addr + row * (desc.dst_stride or desc.length)
                data = self._sram.read(src, desc.length)
                self._sram.write(dst, data)
                transferred += desc.length

        return transferred
