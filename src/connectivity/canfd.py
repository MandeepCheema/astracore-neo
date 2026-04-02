"""
AstraCore Neo — CAN-FD Controller simulation.

Models an ISO 11898-1 CAN-FD controller:
  - Standard (11-bit) and Extended (29-bit) frame IDs
  - Classic CAN (≤ 8 bytes) and CAN-FD (≤ 64 bytes) data frames
  - Priority-ordered TX queue (lower CAN ID = higher priority)
  - FIFO receive queue with configurable depth
  - Error counters (TEC / REC) driving bus state transitions:
      ERROR_ACTIVE (TEC/REC < 128)  →  ERROR_PASSIVE (≥ 128)  →  BUS_OFF (TEC > 255)
  - BUS_OFF recovery simulation
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .exceptions import CANError


# DLC → payload byte length mapping (CAN-FD extended DLC 9-15)
_DLC_TO_LEN: dict[int, int] = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
    9: 12, 10: 16, 11: 20, 12: 24, 13: 32, 14: 48, 15: 64,
}
_LEN_TO_DLC: dict[int, int] = {v: k for k, v in _DLC_TO_LEN.items()}
# For lengths not in the table, map to the next valid CAN-FD size
_VALID_FD_LENS = sorted(set(_DLC_TO_LEN.values()))


def _length_to_dlc(n: int) -> int:
    """Return the smallest valid DLC that accommodates n bytes."""
    for length in _VALID_FD_LENS:
        if n <= length:
            return _LEN_TO_DLC[length]
    raise CANError(f"Payload too large: {n} bytes (max 64)")


class CANIDFormat(Enum):
    STANDARD = auto()   # 11-bit identifier (0–0x7FF)
    EXTENDED = auto()   # 29-bit identifier (0–0x1FFFFFFF)


class CANBusState(Enum):
    ERROR_ACTIVE  = auto()   # TEC and REC < 128 — normal operation
    ERROR_PASSIVE = auto()   # TEC or REC >= 128 — reduced error signalling
    BUS_OFF       = auto()   # TEC > 255 — node disconnected


@dataclass
class CANFrame:
    """A single CAN or CAN-FD data frame."""
    can_id: int
    id_format: CANIDFormat
    data: bytes
    is_fd: bool = False           # True = CAN-FD frame (BRS and ESI bits set)
    dlc: int = field(init=False)
    timestamp_us: float = field(default_factory=lambda: time.monotonic() * 1e6)

    def __post_init__(self) -> None:
        max_id = 0x1FFFFFFF if self.id_format == CANIDFormat.EXTENDED else 0x7FF
        if not (0 <= self.can_id <= max_id):
            raise CANError(
                f"CAN ID {self.can_id:#x} out of range for {self.id_format.name} format"
            )
        max_len = 64 if self.is_fd else 8
        if len(self.data) > max_len:
            raise CANError(
                f"Payload {len(self.data)} bytes exceeds max {max_len} for "
                f"{'CAN-FD' if self.is_fd else 'classic CAN'}"
            )
        self.dlc = _length_to_dlc(len(self.data))


class CANFDController:
    """
    CAN-FD controller simulation.

    Usage::

        ctrl = CANFDController(node_id=1)
        frame = CANFrame(can_id=0x100, id_format=CANIDFormat.STANDARD, data=b"hello!!!")
        ctrl.send(frame)
        sent = ctrl.transmit_next()
        assert sent.can_id == 0x100
    """

    _TEC_INCREMENT_TX_ERROR  = 8
    _TEC_DECREMENT_SUCCESS   = 1
    _REC_INCREMENT_ERROR     = 1
    _BUS_OFF_THRESHOLD       = 256
    _ERROR_PASSIVE_THRESHOLD = 128

    def __init__(self, node_id: int, rx_queue_size: int = 32) -> None:
        self._node_id = node_id
        self._tec: int = 0
        self._rec: int = 0
        self._bus_state = CANBusState.ERROR_ACTIVE
        self._tx_queue: list[CANFrame] = []
        self._rx_queue: deque[CANFrame] = deque(maxlen=rx_queue_size)
        self._tx_count: int = 0
        self._rx_count: int = 0
        self._tx_error_count: int = 0
        self._rx_error_count: int = 0
        self._dropped: int = 0

    # ------------------------------------------------------------------
    # Transmit path
    # ------------------------------------------------------------------

    def send(self, frame: CANFrame) -> None:
        """Enqueue a frame for transmission (priority = lower CAN ID)."""
        if self._bus_state == CANBusState.BUS_OFF:
            raise CANError("Cannot transmit: CAN bus is in BUS_OFF state")
        # Insert maintaining priority order (lower ID first)
        self._tx_queue.append(frame)
        self._tx_queue.sort(key=lambda f: f.can_id)
        self._tx_count += 1
        # Successful TX decrements TEC (floor at 0)
        self._tec = max(0, self._tec - self._TEC_DECREMENT_SUCCESS)
        self._update_bus_state()

    def transmit_next(self) -> Optional[CANFrame]:
        """Pop and return the highest-priority frame from the TX queue."""
        if not self._tx_queue:
            return None
        if self._bus_state == CANBusState.BUS_OFF:
            raise CANError("Cannot transmit: CAN bus is in BUS_OFF state")
        return self._tx_queue.pop(0)

    def tx_pending(self) -> int:
        """Number of frames waiting in the TX queue."""
        return len(self._tx_queue)

    # ------------------------------------------------------------------
    # Receive path
    # ------------------------------------------------------------------

    def receive(self, frame: CANFrame) -> None:
        """Inject a frame into the RX queue (simulates bus reception)."""
        if len(self._rx_queue) == self._rx_queue.maxlen:
            self._dropped += 1
        self._rx_queue.append(frame)
        self._rx_count += 1

    def read(self) -> Optional[CANFrame]:
        """Read the oldest frame from the RX queue (FIFO)."""
        return self._rx_queue.popleft() if self._rx_queue else None

    def rx_available(self) -> int:
        return len(self._rx_queue)

    # ------------------------------------------------------------------
    # Error injection (for testing / fault simulation)
    # ------------------------------------------------------------------

    def inject_tx_error(self) -> None:
        """Simulate a transmit error — increments TEC by 8."""
        self._tec += self._TEC_INCREMENT_TX_ERROR
        self._tx_error_count += 1
        self._update_bus_state()

    def inject_rx_error(self) -> None:
        """Simulate a receive error — increments REC by 1."""
        self._rec += self._REC_INCREMENT_ERROR
        self._rx_error_count += 1
        self._update_bus_state()

    # ------------------------------------------------------------------
    # Bus-off recovery
    # ------------------------------------------------------------------

    def bus_off_recovery(self) -> None:
        """
        Perform BUS_OFF recovery (simulates 128 × 11 consecutive recessive bits).
        Resets TEC and REC to 0 and returns the node to ERROR_ACTIVE.
        """
        if self._bus_state != CANBusState.BUS_OFF:
            raise CANError("bus_off_recovery called but bus is not in BUS_OFF state")
        self._tec = 0
        self._rec = 0
        self._bus_state = CANBusState.ERROR_ACTIVE

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _update_bus_state(self) -> None:
        if self._tec >= self._BUS_OFF_THRESHOLD:
            self._bus_state = CANBusState.BUS_OFF
        elif self._tec >= self._ERROR_PASSIVE_THRESHOLD or self._rec >= self._ERROR_PASSIVE_THRESHOLD:
            self._bus_state = CANBusState.ERROR_PASSIVE
        else:
            self._bus_state = CANBusState.ERROR_ACTIVE

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def node_id(self) -> int:
        return self._node_id

    @property
    def bus_state(self) -> CANBusState:
        return self._bus_state

    @property
    def tec(self) -> int:
        return self._tec

    @property
    def rec(self) -> int:
        return self._rec

    @property
    def tx_count(self) -> int:
        return self._tx_count

    @property
    def rx_count(self) -> int:
        return self._rx_count

    @property
    def dropped(self) -> int:
        return self._dropped

    def __repr__(self) -> str:
        return (
            f"CANFDController(node={self._node_id}, "
            f"state={self._bus_state.name}, "
            f"TEC={self._tec}, REC={self._rec})"
        )
