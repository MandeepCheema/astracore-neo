"""AstraCore Neo Memory — Exception hierarchy."""


class MemoryError(Exception):
    """Base exception for all memory subsystem errors."""


class BankError(MemoryError):
    """Raised on invalid bank access (disabled, out-of-range)."""


class EccError(MemoryError):
    """
    Raised on uncorrectable ECC error (double-bit flip).

    Single-bit errors are silently corrected; double-bit errors raise this.
    """
    def __init__(self, addr: int, bank: int) -> None:
        super().__init__(f"Uncorrectable ECC error at addr=0x{addr:08X} bank={bank}")
        self.addr = addr
        self.bank = bank


class DmaError(MemoryError):
    """Raised on DMA transfer fault (invalid address, channel busy, etc.)."""


class CompressionError(MemoryError):
    """Raised on compression/decompression failure."""
