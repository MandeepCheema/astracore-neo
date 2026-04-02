"""
AstraCore Neo — Connectivity Manager.

Top-level coordinator that initialises and holds references to all four
on-chip communication controllers: CAN-FD, Ethernet, PCIe, and V2X.

Provides aggregate link-status queries and ensures each controller is
initialised at most once.
"""

from __future__ import annotations

from typing import Optional

from .canfd import CANFDController, CANBusState
from .ethernet import EthernetController, EthernetLinkState
from .pcie import PCIeController, PCIeLinkState
from .v2x import V2XController, CCH_CHANNEL
from .exceptions import ConnectivityBaseError


class ConnectivityManager:
    """
    Connectivity subsystem coordinator.

    Usage::

        mgr = ConnectivityManager()
        can = mgr.init_canfd(node_id=1)
        eth = mgr.init_ethernet(mac_address=bytes.fromhex("aabbccddeeff"))
        pcie = mgr.init_pcie(device_id=0x1000, num_lanes=4)
        v2x = mgr.init_v2x(node_id=0xCAFE_BABE)

        print(mgr.link_status())
    """

    def __init__(self) -> None:
        self._canfd: Optional[CANFDController] = None
        self._ethernet: Optional[EthernetController] = None
        self._pcie: Optional[PCIeController] = None
        self._v2x: Optional[V2XController] = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_canfd(self, node_id: int, rx_queue_size: int = 32) -> CANFDController:
        if self._canfd is not None:
            raise ConnectivityBaseError("CAN-FD controller already initialised")
        self._canfd = CANFDController(node_id=node_id, rx_queue_size=rx_queue_size)
        return self._canfd

    def init_ethernet(
        self, mac_address: bytes, rx_buffer_size: int = 64
    ) -> EthernetController:
        if self._ethernet is not None:
            raise ConnectivityBaseError("Ethernet controller already initialised")
        self._ethernet = EthernetController(
            mac_address=mac_address, rx_buffer_size=rx_buffer_size
        )
        return self._ethernet

    def init_pcie(
        self, device_id: int, num_lanes: int = 4
    ) -> PCIeController:
        if self._pcie is not None:
            raise ConnectivityBaseError("PCIe controller already initialised")
        self._pcie = PCIeController(device_id=device_id, num_lanes=num_lanes)
        return self._pcie

    def init_v2x(
        self,
        node_id: int,
        channel: int = CCH_CHANNEL,
        rx_queue_size: int = 32,
    ) -> V2XController:
        if self._v2x is not None:
            raise ConnectivityBaseError("V2X controller already initialised")
        self._v2x = V2XController(
            node_id=node_id, channel=channel, rx_queue_size=rx_queue_size
        )
        return self._v2x

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def canfd(self) -> Optional[CANFDController]:
        return self._canfd

    @property
    def ethernet(self) -> Optional[EthernetController]:
        return self._ethernet

    @property
    def pcie(self) -> Optional[PCIeController]:
        return self._pcie

    @property
    def v2x(self) -> Optional[V2XController]:
        return self._v2x

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def link_status(self) -> dict:
        """Return a dict summarising the status of each active interface."""
        status: dict = {}
        if self._canfd:
            status["canfd"] = {
                "bus_state": self._canfd.bus_state.name,
                "tec": self._canfd.tec,
                "rec": self._canfd.rec,
            }
        if self._ethernet:
            status["ethernet"] = {
                "link": self._ethernet.link_state.name,
                "tx": self._ethernet.tx_count,
                "rx": self._ethernet.rx_count,
            }
        if self._pcie:
            status["pcie"] = {
                "link": self._pcie.link_state.name,
                "lanes": self._pcie.num_lanes,
                "tx_tlp": self._pcie.tx_count,
            }
        if self._v2x:
            status["v2x"] = {
                "channel": self._v2x.channel,
                "tx": self._v2x.tx_count,
                "rx": self._v2x.rx_count,
            }
        return status

    def any_link_up(self) -> bool:
        """True if at least one interface is in an active/connected state."""
        if self._canfd and self._canfd.bus_state != CANBusState.BUS_OFF:
            return True
        if self._ethernet and self._ethernet.link_state == EthernetLinkState.UP:
            return True
        if self._pcie and self._pcie.link_state == PCIeLinkState.L0:
            return True
        if self._v2x:
            return True
        return False

    def __repr__(self) -> str:
        active = []
        if self._canfd:
            active.append("CAN-FD")
        if self._ethernet:
            active.append("Ethernet")
        if self._pcie:
            active.append("PCIe")
        if self._v2x:
            active.append("V2X")
        return f"ConnectivityManager(active={active})"
