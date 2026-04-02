"""
AstraCore Neo Connectivity exceptions.
"""


class ConnectivityBaseError(Exception):
    """Base exception for all connectivity subsystem errors."""


class CANError(ConnectivityBaseError):
    """CAN-FD controller error."""


class EthernetError(ConnectivityBaseError):
    """Ethernet controller error."""


class PCIeError(ConnectivityBaseError):
    """PCIe controller error."""


class V2XError(ConnectivityBaseError):
    """V2X controller error."""
