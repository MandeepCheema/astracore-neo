"""
AstraCore Neo Connectivity Subsystem.

Public API::

    from connectivity import CANFDController, CANFrame, CANIDFormat, CANBusState
    from connectivity import EthernetController, EthernetFrame, EthernetLinkState, BROADCAST_MAC
    from connectivity import PCIeController, PCIeBAR, PCIeLinkState, TLP, TLPType
    from connectivity import V2XController, V2XMessage, V2XMessageType, CCH_CHANNEL, DSRC_CHANNELS
    from connectivity import ConnectivityManager
    from connectivity import ConnectivityBaseError, CANError, EthernetError, PCIeError, V2XError
"""

from .canfd import (
    CANFDController, CANFrame, CANIDFormat, CANBusState,
)
from .ethernet import (
    EthernetController, EthernetFrame, EthernetLinkState, BROADCAST_MAC,
)
from .pcie import (
    PCIeController, PCIeBAR, PCIeLinkState, TLP, TLPType,
)
from .v2x import (
    V2XController, V2XMessage, V2XMessageType, CCH_CHANNEL, DSRC_CHANNELS,
)
from .connectivity_manager import ConnectivityManager
from .exceptions import (
    ConnectivityBaseError, CANError, EthernetError, PCIeError, V2XError,
)

__all__ = [
    # CAN-FD
    "CANFDController", "CANFrame", "CANIDFormat", "CANBusState",
    # Ethernet
    "EthernetController", "EthernetFrame", "EthernetLinkState", "BROADCAST_MAC",
    # PCIe
    "PCIeController", "PCIeBAR", "PCIeLinkState", "TLP", "TLPType",
    # V2X
    "V2XController", "V2XMessage", "V2XMessageType", "CCH_CHANNEL", "DSRC_CHANNELS",
    # Manager
    "ConnectivityManager",
    # Exceptions
    "ConnectivityBaseError", "CANError", "EthernetError", "PCIeError", "V2XError",
]
