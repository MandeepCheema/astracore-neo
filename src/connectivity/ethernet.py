"""
AstraCore Neo — Ethernet Controller simulation.

Models an automotive Ethernet MAC (100BASE-T1 / 1000BASE-T1):
  - MAC address filtering (unicast and broadcast)
  - Frame validation (payload size 0-1500 bytes, 6-byte MAC addresses)
  - TX path: link-state check, frame dispatch
  - RX path: address filter, FIFO buffer with overflow drop counter
  - Link state: DOWN → UP
  - Per-controller statistics
"""

from __future__ import annotations

import struct
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .exceptions import EthernetError


BROADCAST_MAC: bytes = bytes([0xFF] * 6)
MAX_PAYLOAD = 1500
MIN_PAYLOAD = 0   # jumbo/zero-payload frames accepted in simulation


class EthernetLinkState(Enum):
    DOWN = auto()
    UP   = auto()


@dataclass
class EthernetFrame:
    """
    An Ethernet II frame.

    Fields
    ------
    dest_mac  : 6-byte destination MAC
    src_mac   : 6-byte source MAC
    ethertype : 2-byte EtherType (e.g. 0x0800=IPv4, 0x0806=ARP, 0x8892=PROFINET)
    payload   : variable-length data (0–1500 bytes)
    """
    dest_mac: bytes
    src_mac: bytes
    ethertype: int
    payload: bytes
    timestamp_us: float = field(default_factory=lambda: time.monotonic() * 1e6)

    def __post_init__(self) -> None:
        if len(self.dest_mac) != 6:
            raise EthernetError(f"dest_mac must be 6 bytes, got {len(self.dest_mac)}")
        if len(self.src_mac) != 6:
            raise EthernetError(f"src_mac must be 6 bytes, got {len(self.src_mac)}")
        if not (0 <= self.ethertype <= 0xFFFF):
            raise EthernetError(f"ethertype out of range: {self.ethertype:#x}")
        if len(self.payload) > MAX_PAYLOAD:
            raise EthernetError(
                f"Payload {len(self.payload)} bytes exceeds maximum {MAX_PAYLOAD}"
            )


class EthernetController:
    """
    Ethernet MAC controller simulation.

    Usage::

        mac = bytes.fromhex("aabb ccddeeff".replace(" ", ""))
        ctrl = EthernetController(mac_address=mac)
        ctrl.link_up()
        frame = EthernetFrame(
            dest_mac=BROADCAST_MAC, src_mac=mac,
            ethertype=0x0806, payload=b"\\x00" * 28
        )
        ctrl.send(frame)
        assert ctrl.tx_count == 1
    """

    def __init__(self, mac_address: bytes, rx_buffer_size: int = 64) -> None:
        if len(mac_address) != 6:
            raise EthernetError(f"mac_address must be 6 bytes, got {len(mac_address)}")
        self._mac = mac_address
        self._link_state = EthernetLinkState.DOWN
        self._rx_buffer: deque[EthernetFrame] = deque(maxlen=rx_buffer_size)
        self._tx_count: int = 0
        self._rx_count: int = 0
        self._tx_errors: int = 0
        self._rx_errors: int = 0
        self._dropped: int = 0

    # ------------------------------------------------------------------
    # Link management
    # ------------------------------------------------------------------

    def link_up(self) -> None:
        self._link_state = EthernetLinkState.UP

    def link_down(self) -> None:
        self._link_state = EthernetLinkState.DOWN

    # ------------------------------------------------------------------
    # TX path
    # ------------------------------------------------------------------

    def send(self, frame: EthernetFrame) -> None:
        """Transmit a frame. Raises EthernetError if link is DOWN."""
        if self._link_state == EthernetLinkState.DOWN:
            raise EthernetError("Cannot send: Ethernet link is DOWN")
        self._tx_count += 1

    # ------------------------------------------------------------------
    # RX path
    # ------------------------------------------------------------------

    def receive(self, frame: EthernetFrame) -> bool:
        """
        Simulate frame reception from the network.

        Returns True if the frame was accepted (dest MAC matches this
        controller's MAC or is broadcast/multicast).  Returns False if
        filtered out.  Dropped frames (buffer full) are not counted as
        rejected — they are counted in `dropped`.
        """
        if not self._accepts(frame.dest_mac):
            return False

        if len(self._rx_buffer) == self._rx_buffer.maxlen:
            self._dropped += 1
            # Still count toward rx_count (frame arrived; we just dropped it)
        else:
            self._rx_buffer.append(frame)
        self._rx_count += 1
        return True

    def read(self) -> Optional[EthernetFrame]:
        """Read the oldest frame from the RX buffer."""
        return self._rx_buffer.popleft() if self._rx_buffer else None

    def rx_available(self) -> int:
        return len(self._rx_buffer)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _accepts(self, dest_mac: bytes) -> bool:
        """True if dest_mac is this node's MAC, broadcast, or multicast."""
        if dest_mac == self._mac:
            return True
        if dest_mac == BROADCAST_MAC:
            return True
        # Multicast: LSB of first byte set
        if dest_mac[0] & 0x01:
            return True
        return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mac_address(self) -> bytes:
        return self._mac

    @property
    def link_state(self) -> EthernetLinkState:
        return self._link_state

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
        mac_str = ":".join(f"{b:02x}" for b in self._mac)
        return (
            f"EthernetController(mac={mac_str}, "
            f"link={self._link_state.name}, "
            f"tx={self._tx_count}, rx={self._rx_count})"
        )
