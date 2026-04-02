"""
AstraCore Neo Memory System.

Public API::

    from memory import SRAMController, DMAEngine, NeuralCompressor
    from memory import MemoryError, EccError, DmaError, BankError
"""

from .sram import SRAMController, SRAMBank
from .dma import DMAEngine, DMADescriptor, DMAChannel
from .compression import NeuralCompressor, CompressionMode
from .exceptions import MemoryError, EccError, DmaError, BankError

__all__ = [
    "SRAMController",
    "SRAMBank",
    "DMAEngine",
    "DMADescriptor",
    "DMAChannel",
    "NeuralCompressor",
    "CompressionMode",
    "MemoryError",
    "EccError",
    "DmaError",
    "BankError",
]
