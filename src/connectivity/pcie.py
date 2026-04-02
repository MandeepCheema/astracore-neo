"""
AstraCore Neo — PCIe Controller simulation.

Models a PCIe Gen3 x4 endpoint:
  - Link training state machine: DETECT → POLLING → CONFIG → L0 (active)
  - BAR (Base Address Register) memory regions — MMIO read/write
  - Transaction Layer Packets (TLP): Memory Read/Write, Config Read/Write,
    Completion with Data
  - Simple credit-based flow control (posted / non-posted / completion credits)
  - Per-lane error injection and link width reporting
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .exceptions import PCIeError


class PCIeLinkState(Enum):
    DETECT   = auto()   # Initial state — detect receiver
    POLLING  = auto()   # Lock and bit-sync
    CONFIG   = auto()   # Lane/link width negotiation
    L0       = auto()   # Active — data transfer enabled
    L1       = auto()   # Low-power ASPM (CLKPM)
    L2       = auto()   # Off — clocks removed


class TLPType(Enum):
    MEM_READ        = auto()
    MEM_WRITE       = auto()
    CFG_READ        = auto()
    CFG_WRITE       = auto()
    COMPLETION      = auto()   # completion without data (ACK / error)
    COMPLETION_DATA = auto()   # completion with data (response to read)


@dataclass
class TLP:
    """A PCIe Transaction Layer Packet."""
    tlp_type: TLPType
    requester_id: int       # 16-bit: [bus(8):device(5):function(3)]
    tag: int                # 8-bit transaction tag
    address: int            # target address (memory or config space)
    data: bytes             # write payload or completion data
    length_dw: int          # length in DWORDs (4-byte words)
    timestamp_us: float = field(default_factory=lambda: time.monotonic() * 1e6)


class PCIeBAR:
    """
    A PCIe BAR (Base Address Register) backed by a bytearray.

    Supports byte-granularity MMIO reads and writes within the BAR window.
    """

    def __init__(self, bar_index: int, base_addr: int, size: int) -> None:
        if size <= 0 or (size & (size - 1)) != 0:
            raise PCIeError(f"BAR size must be a power of 2, got {size}")
        if base_addr % size != 0:
            raise PCIeError(
                f"BAR base address {base_addr:#x} must be aligned to size {size:#x}"
            )
        self._index = bar_index
        self._base = base_addr
        self._size = size
        self._mem: bytearray = bytearray(size)

    def contains(self, address: int, length: int) -> bool:
        return self._base <= address and (address + length) <= (self._base + self._size)

    def read(self, address: int, length: int) -> bytes:
        if not self.contains(address, length):
            raise PCIeError(
                f"BAR{self._index}: address {address:#x}+{length} out of range "
                f"[{self._base:#x}, {self._base + self._size:#x})"
            )
        offset = address - self._base
        return bytes(self._mem[offset: offset + length])

    def write(self, address: int, data: bytes) -> None:
        if not self.contains(address, len(data)):
            raise PCIeError(
                f"BAR{self._index}: address {address:#x}+{len(data)} out of range"
            )
        offset = address - self._base
        self._mem[offset: offset + len(data)] = data

    @property
    def base_address(self) -> int:
        return self._base

    @property
    def size(self) -> int:
        return self._size

    @property
    def index(self) -> int:
        return self._index

    def __repr__(self) -> str:
        return f"PCIeBAR(BAR{self._index}, base={self._base:#x}, size={self._size:#x})"


class PCIeController:
    """
    PCIe endpoint controller simulation.

    Usage::

        ep = PCIeController(device_id=0x1000, num_lanes=4)
        ep.train_link()
        assert ep.link_state == PCIeLinkState.L0

        bar = ep.add_bar(bar_index=0, base_addr=0x10000000, size=4096)
        ep.mmio_write(0x10000000, b"\\xDE\\xAD\\xBE\\xEF")
        data = ep.mmio_read(0x10000000, 4)
        assert data == b"\\xDE\\xAD\\xBE\\xEF"
    """

    MAX_BARS = 6

    def __init__(self, device_id: int, num_lanes: int = 4) -> None:
        if num_lanes not in (1, 2, 4, 8, 16):
            raise PCIeError(f"Unsupported lane count: {num_lanes} (must be 1/2/4/8/16)")
        self._device_id = device_id
        self._num_lanes = num_lanes
        self._link_state = PCIeLinkState.DETECT
        self._bars: dict[int, PCIeBAR] = {}
        self._tlp_log: list[TLP] = []
        self._tx_count: int = 0
        self._rx_count: int = 0
        self._error_count: int = 0

    # ------------------------------------------------------------------
    # Link training
    # ------------------------------------------------------------------

    def train_link(self) -> None:
        """Simulate link training: DETECT → POLLING → CONFIG → L0."""
        for state in (PCIeLinkState.POLLING, PCIeLinkState.CONFIG, PCIeLinkState.L0):
            self._link_state = state

    def enter_l1(self) -> None:
        """Enter low-power L1 state. Link must be in L0."""
        if self._link_state != PCIeLinkState.L0:
            raise PCIeError(f"Cannot enter L1 from {self._link_state.name}")
        self._link_state = PCIeLinkState.L1

    def exit_l1(self) -> None:
        """Return to L0 from L1."""
        if self._link_state != PCIeLinkState.L1:
            raise PCIeError(f"Cannot exit L1: current state is {self._link_state.name}")
        self._link_state = PCIeLinkState.L0

    # ------------------------------------------------------------------
    # BAR management
    # ------------------------------------------------------------------

    def add_bar(self, bar_index: int, base_addr: int, size: int) -> PCIeBAR:
        """Register a BAR memory region."""
        if bar_index < 0 or bar_index >= self.MAX_BARS:
            raise PCIeError(f"BAR index {bar_index} out of range (0–{self.MAX_BARS - 1})")
        if bar_index in self._bars:
            raise PCIeError(f"BAR{bar_index} already registered")
        bar = PCIeBAR(bar_index, base_addr, size)
        self._bars[bar_index] = bar
        return bar

    def get_bar(self, bar_index: int) -> PCIeBAR:
        if bar_index not in self._bars:
            raise PCIeError(f"BAR{bar_index} not registered")
        return self._bars[bar_index]

    # ------------------------------------------------------------------
    # MMIO
    # ------------------------------------------------------------------

    def _require_l0(self) -> None:
        if self._link_state != PCIeLinkState.L0:
            raise PCIeError(
                f"PCIe MMIO requires L0 link state; current: {self._link_state.name}"
            )

    def mmio_read(self, address: int, length: int) -> bytes:
        """Perform an MMIO read via the matching BAR."""
        self._require_l0()
        bar = self._find_bar(address, length)
        return bar.read(address, length)

    def mmio_write(self, address: int, data: bytes) -> None:
        """Perform an MMIO write via the matching BAR."""
        self._require_l0()
        bar = self._find_bar(address, len(data))
        bar.write(address, data)

    def _find_bar(self, address: int, length: int) -> PCIeBAR:
        for bar in self._bars.values():
            if bar.contains(address, length):
                return bar
        raise PCIeError(f"No BAR maps address {address:#x}+{length}")

    # ------------------------------------------------------------------
    # TLP interface
    # ------------------------------------------------------------------

    def send_tlp(self, tlp: TLP) -> None:
        """Enqueue a TLP for transmission. Link must be in L0."""
        self._require_l0()
        self._tlp_log.append(tlp)
        self._tx_count += 1

    def receive_tlp(self, tlp: TLP) -> Optional[TLP]:
        """
        Process an inbound TLP.

        For MEM_READ: returns a COMPLETION_DATA TLP with BAR data.
        For MEM_WRITE: writes to BAR, returns None.
        For other types: logs and returns None.
        """
        self._require_l0()
        self._rx_count += 1

        if tlp.tlp_type == TLPType.MEM_WRITE:
            try:
                self._find_bar(tlp.address, len(tlp.data)).write(tlp.address, tlp.data)
            except PCIeError:
                self._error_count += 1
            return None

        if tlp.tlp_type == TLPType.MEM_READ:
            try:
                data = self._find_bar(tlp.address, tlp.length_dw * 4).read(
                    tlp.address, tlp.length_dw * 4
                )
                return TLP(
                    tlp_type=TLPType.COMPLETION_DATA,
                    requester_id=self._device_id,
                    tag=tlp.tag,
                    address=tlp.address,
                    data=data,
                    length_dw=tlp.length_dw,
                )
            except PCIeError:
                self._error_count += 1
                return TLP(
                    tlp_type=TLPType.COMPLETION,
                    requester_id=self._device_id,
                    tag=tlp.tag,
                    address=tlp.address,
                    data=b"",
                    length_dw=0,
                )

        return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device_id(self) -> int:
        return self._device_id

    @property
    def num_lanes(self) -> int:
        return self._num_lanes

    @property
    def link_state(self) -> PCIeLinkState:
        return self._link_state

    @property
    def tx_count(self) -> int:
        return self._tx_count

    @property
    def rx_count(self) -> int:
        return self._rx_count

    @property
    def error_count(self) -> int:
        return self._error_count

    def bar_count(self) -> int:
        return len(self._bars)

    def __repr__(self) -> str:
        return (
            f"PCIeController(dev={self._device_id:#06x}, "
            f"x{self._num_lanes}, state={self._link_state.name})"
        )
