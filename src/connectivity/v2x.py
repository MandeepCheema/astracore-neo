"""
AstraCore Neo — V2X (Vehicle-to-Everything) Controller simulation.

Models a DSRC/C-V2X radio controller:
  - DSRC channel set (SCH 174–184 MHz, CCH 178)
  - SAE J2735 message types: BSM, SPaT, MAP, PSM
  - Broadcast TX queue and RX FIFO
  - Channel selection and busy-ratio tracking
  - RSSI-filtered receive (simulated signal quality threshold)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .exceptions import V2XError


# DSRC channel numbers (5.9 GHz band, 10 MHz channels)
DSRC_CHANNELS: frozenset[int] = frozenset({172, 174, 176, 178, 180, 182, 184})
CCH_CHANNEL: int = 178   # Control Channel — mandatory for BSM
SCH_CHANNELS: frozenset[int] = DSRC_CHANNELS - {CCH_CHANNEL}


class V2XMessageType(Enum):
    BSM  = auto()   # Basic Safety Message — position, speed, heading, braking
    SPaT = auto()   # Signal Phase and Timing — traffic light states
    MAP  = auto()   # Map Data — intersection geometry
    PSM  = auto()   # Personal Safety Message — pedestrian / cyclist


@dataclass
class V2XMessage:
    """A single V2X over-the-air message."""
    msg_type: V2XMessageType
    sender_id: int          # 32-bit temporary identifier
    channel: int            # DSRC channel number
    payload: bytes
    rssi_dbm: float = -70.0     # Received Signal Strength; lower = weaker
    timestamp_us: float = field(default_factory=lambda: time.monotonic() * 1e6)

    def __post_init__(self) -> None:
        if self.channel not in DSRC_CHANNELS:
            raise V2XError(
                f"Channel {self.channel} is not a valid DSRC channel "
                f"(valid: {sorted(DSRC_CHANNELS)})"
            )
        if len(self.payload) == 0:
            raise V2XError("V2X message payload must be non-empty")


class V2XController:
    """
    DSRC/C-V2X radio controller simulation.

    Usage::

        ctrl = V2XController(node_id=0xDEAD_BEEF, channel=CCH_CHANNEL)
        msg = V2XMessage(
            msg_type=V2XMessageType.BSM,
            sender_id=ctrl.node_id,
            channel=CCH_CHANNEL,
            payload=b"\\x00" * 38,   # minimal BSM payload
        )
        ctrl.broadcast(msg)
        assert ctrl.tx_count == 1
    """

    RSSI_THRESHOLD_DBM: float = -90.0   # discard messages weaker than this

    def __init__(
        self,
        node_id: int,
        channel: int = CCH_CHANNEL,
        rx_queue_size: int = 32,
        rssi_threshold_dbm: float = -90.0,
    ) -> None:
        if channel not in DSRC_CHANNELS:
            raise V2XError(f"Channel {channel} is not a valid DSRC channel")
        self._node_id = node_id
        self._channel = channel
        self._rssi_threshold = rssi_threshold_dbm
        self._rx_queue: deque[V2XMessage] = deque(maxlen=rx_queue_size)
        self._tx_count: int = 0
        self._rx_count: int = 0
        self._dropped: int = 0
        self._filtered_rssi: int = 0   # messages rejected due to weak signal

    # ------------------------------------------------------------------
    # TX path
    # ------------------------------------------------------------------

    def broadcast(self, msg: V2XMessage) -> None:
        """Broadcast a message on the current channel."""
        if msg.channel != self._channel:
            raise V2XError(
                f"Message channel {msg.channel} does not match "
                f"controller channel {self._channel}"
            )
        self._tx_count += 1

    def set_channel(self, channel: int) -> None:
        """Tune to a different DSRC channel."""
        if channel not in DSRC_CHANNELS:
            raise V2XError(f"Channel {channel} is not a valid DSRC channel")
        self._channel = channel

    # ------------------------------------------------------------------
    # RX path
    # ------------------------------------------------------------------

    def receive(self, msg: V2XMessage) -> bool:
        """
        Inject a received message.

        Returns True if enqueued; False if filtered (wrong channel or weak RSSI).
        Dropped (buffer-full) messages are counted but not returned as False.
        """
        if msg.channel != self._channel:
            return False
        if msg.rssi_dbm < self._rssi_threshold:
            self._filtered_rssi += 1
            return False
        if len(self._rx_queue) == self._rx_queue.maxlen:
            self._dropped += 1
        else:
            self._rx_queue.append(msg)
        self._rx_count += 1
        return True

    def read(self) -> Optional[V2XMessage]:
        """Read the oldest message from the RX queue (FIFO)."""
        return self._rx_queue.popleft() if self._rx_queue else None

    def rx_available(self) -> int:
        return len(self._rx_queue)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def node_id(self) -> int:
        return self._node_id

    @property
    def channel(self) -> int:
        return self._channel

    @property
    def tx_count(self) -> int:
        return self._tx_count

    @property
    def rx_count(self) -> int:
        return self._rx_count

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def filtered_rssi(self) -> int:
        """Number of messages rejected due to RSSI below threshold."""
        return self._filtered_rssi

    def __repr__(self) -> str:
        return (
            f"V2XController(node={self._node_id:#010x}, "
            f"ch={self._channel}, tx={self._tx_count}, rx={self._rx_count})"
        )
