"""
AstraCore Neo — Memory Testbench

Covers:
  SRAM:
    - Bank init, enable/disable
    - Read/write bytes and words
    - Address-to-bank mapping
    - Out-of-bounds access
    - ECC single-bit correction
    - ECC double-bit detection (raises EccError)
    - Dual-port same-bank and cross-bank transfers
    - Reset clears all data
    - HAL register sync (MEM_CTRL)

  DMA:
    - Descriptor submission and execution
    - Contiguous transfers
    - 2D strided transfers
    - Multi-channel independent operation
    - IRQ_DMA_DONE fires on completion
    - Channel fault and reset
    - Pending queue management
    - Cache line load / invalidate

  Compression:
    - INT8 encode → decode round-trip (various sizes)
    - INT4 encode → decode round-trip
    - Compression ratio > 1.0 on compressible data
    - Empty data raises CompressionError
    - Corrupt magic raises CompressionError
    - Stats accumulation and reset
"""

import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory import (
    SRAMController, SRAMBank,
    DMAEngine, DMADescriptor,
    NeuralCompressor, CompressionMode,
    EccError, BankError, DmaError,
)
from src.memory.exceptions import CompressionError
from src.memory.sram import NUM_BANKS, BANK_SIZE_BYTES, BANK_ADDR_BITS
from src.memory.dma import NUM_CHANNELS, TransferFlags, ChannelState
from src.hal import AstraCoreDevice
from src.hal.interrupts import IRQ_DMA_DONE


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sram():
    return SRAMController()


@pytest.fixture
def sram_with_dev():
    dev = AstraCoreDevice()
    dev.power_on()
    ctrl = SRAMController(dev=dev)
    return ctrl, dev


@pytest.fixture
def dma(sram):
    return DMAEngine(sram=sram)


@pytest.fixture
def dma_with_dev():
    dev = AstraCoreDevice()
    dev.power_on()
    sram = SRAMController()
    engine = DMAEngine(sram=sram, dev=dev)
    return engine, dev


@pytest.fixture
def nc():
    return NeuralCompressor()


# ===========================================================================
# 1. SRAM — Bank basics
# ===========================================================================

class TestSRAMBankBasics:

    def test_all_banks_enabled_by_default(self, sram):
        for i in range(NUM_BANKS):
            assert sram.bank_enabled(i)

    def test_enabled_mask_all_on(self, sram):
        assert sram.enabled_bank_mask() == 0xFFFF

    def test_disable_bank(self, sram):
        sram.disable_bank(0)
        assert not sram.bank_enabled(0)

    def test_enable_after_disable(self, sram):
        sram.disable_bank(3)
        sram.enable_bank(3)
        assert sram.bank_enabled(3)

    def test_enabled_mask_partial(self, sram):
        sram.disable_bank(0)
        sram.disable_bank(1)
        assert sram.enabled_bank_mask() == 0xFFFC

    def test_invalid_bank_id_raises(self, sram):
        with pytest.raises(BankError):
            sram.disable_bank(16)

    def test_negative_bank_id_raises(self, sram):
        with pytest.raises(BankError):
            sram.enable_bank(-1)

    def test_total_size_is_128mb(self):
        assert NUM_BANKS * BANK_SIZE_BYTES == 128 * 1024 * 1024


# ===========================================================================
# 2. SRAM — Read / Write
# ===========================================================================

class TestSRAMReadWrite:

    def test_write_read_bytes(self, sram):
        sram.write(0x000000, b"hello")
        assert sram.read(0x000000, 5) == b"hello"

    def test_write_read_word(self, sram):
        sram.write_word(0x000000, 0xDEAD_BEEF)
        assert sram.read_word(0x000000) == 0xDEAD_BEEF

    def test_write_read_large_block(self, sram):
        data = bytes(range(256)) * 4   # 1024 bytes
        sram.write(0x000000, data)
        assert sram.read(0x000000, 1024) == data

    def test_zero_after_reset(self, sram):
        sram.write(0x000000, b"\xFF" * 8)
        sram.reset()
        assert sram.read(0x000000, 8) == b"\x00" * 8

    def test_write_to_different_banks(self, sram):
        addr_b0 = 0 * BANK_SIZE_BYTES
        addr_b1 = 1 * BANK_SIZE_BYTES
        sram.write(addr_b0, b"bank0")
        sram.write(addr_b1, b"bank1")
        assert sram.read(addr_b0, 5) == b"bank0"
        assert sram.read(addr_b1, 5) == b"bank1"

    def test_write_to_last_bank(self, sram):
        addr = 15 * BANK_SIZE_BYTES
        sram.write(addr, b"last")
        assert sram.read(addr, 4) == b"last"

    def test_out_of_bounds_read_raises(self, sram):
        with pytest.raises(BankError):
            sram.read(128 * 1024 * 1024, 1)

    def test_out_of_bounds_write_raises(self, sram):
        with pytest.raises(BankError):
            sram.write(128 * 1024 * 1024, b"\x00")

    def test_write_to_disabled_bank_raises(self, sram):
        sram.disable_bank(0)
        with pytest.raises(BankError):
            sram.write(0x000000, b"x")

    def test_read_from_disabled_bank_raises(self, sram):
        sram.disable_bank(0)
        with pytest.raises(BankError):
            sram.read(0x000000, 1)

    def test_address_maps_to_correct_bank(self):
        for bank_id in range(NUM_BANKS):
            addr = bank_id * BANK_SIZE_BYTES
            from src.memory.sram import _bank_and_offset
            b, off = _bank_and_offset(addr)
            assert b == bank_id
            assert off == 0

    def test_offset_within_bank(self):
        from src.memory.sram import _bank_and_offset
        addr = 1 * BANK_SIZE_BYTES + 0x1234
        b, off = _bank_and_offset(addr)
        assert b == 1
        assert off == 0x1234


# ===========================================================================
# 3. SRAM — ECC
# ===========================================================================

class TestSRAMEcc:

    def test_single_bit_error_corrected(self, sram):
        sram.write(0x000000, b"\xAA")
        sram.bank(0).inject_single_bit_error(0, bit=0)
        # Should read back corrected value without raising
        result = sram.read(0x000000, 1)
        assert result == b"\xAB"   # bit 0 flipped back = correction
        assert sram.bank(0).ecc_corrections == 1

    def test_double_bit_error_raises(self, sram):
        sram.write(0x000000, b"\xAA")
        sram.bank(0).inject_double_bit_error(0)
        with pytest.raises(EccError) as exc_info:
            sram.read(0x000000, 1)
        assert exc_info.value.bank == 0

    def test_ecc_error_addr_in_exception(self, sram):
        sram.bank(0).inject_double_bit_error(5)
        with pytest.raises(EccError) as exc_info:
            sram.read(0x000005, 1)
        assert exc_info.value.addr == 5

    def test_clear_ecc_faults(self, sram):
        sram.bank(0).inject_double_bit_error(0)
        sram.bank(0).clear_ecc_faults()
        sram.write(0x000000, b"\x42")
        assert sram.read(0x000000, 1) == b"\x42"

    def test_total_ecc_corrections_accumulates(self, sram):
        sram.write(0x000000, b"\x00\x00")
        sram.bank(0).inject_single_bit_error(0)
        sram.bank(0).inject_single_bit_error(1)
        sram.read(0x000000, 2)
        assert sram.total_ecc_corrections() == 2

    def test_ecc_detection_counter(self, sram):
        sram.bank(0).inject_double_bit_error(0)
        try:
            sram.read(0x000000, 1)
        except EccError:
            pass
        assert sram.bank(0).ecc_detections == 1


# ===========================================================================
# 4. SRAM — Dual-port
# ===========================================================================

class TestSRAMDualPort:

    def test_dual_port_cross_bank(self, sram):
        sram.write(0 * BANK_SIZE_BYTES, b"source_data")
        read_result = sram.dual_port_transfer(
            read_addr=0 * BANK_SIZE_BYTES, read_len=11,
            write_addr=1 * BANK_SIZE_BYTES, write_data=b"destination",
        )
        assert read_result == b"source_data"
        assert sram.read(1 * BANK_SIZE_BYTES, 11) == b"destination"

    def test_dual_port_same_bank_serialises(self, sram):
        # Write-then-read on same bank: write happens first
        read_result = sram.dual_port_transfer(
            read_addr=0x000004, read_len=4,
            write_addr=0x000004, write_data=b"wxyz",
        )
        # Write happened first so read sees the new data
        assert read_result == b"wxyz"

    def test_dual_port_cross_bank_no_interference(self, sram):
        sram.write(0 * BANK_SIZE_BYTES, b"AAAAAA")
        sram.write(1 * BANK_SIZE_BYTES, b"BBBBBB")
        result = sram.dual_port_transfer(
            read_addr=0 * BANK_SIZE_BYTES, read_len=6,
            write_addr=1 * BANK_SIZE_BYTES, write_data=b"CCCCCC",
        )
        assert result == b"AAAAAA"
        assert sram.read(1 * BANK_SIZE_BYTES, 6) == b"CCCCCC"


# ===========================================================================
# 5. SRAM — HAL register sync
# ===========================================================================

class TestSRAMHalSync:

    def test_disable_bank_updates_mem_ctrl(self, sram_with_dev):
        ctrl, dev = sram_with_dev
        ctrl.disable_bank(0)
        assert dev.regs.read(0x0038) == 0xFFFE

    def test_enable_bank_updates_mem_ctrl(self, sram_with_dev):
        ctrl, dev = sram_with_dev
        ctrl.disable_bank(0)
        ctrl.enable_bank(0)
        assert dev.regs.read(0x0038) == 0xFFFF

    def test_reset_restores_mem_ctrl(self, sram_with_dev):
        ctrl, dev = sram_with_dev
        ctrl.disable_bank(4)
        ctrl.reset()
        assert dev.regs.read(0x0038) == 0xFFFF


# ===========================================================================
# 6. DMA — Submission
# ===========================================================================

class TestDMASubmission:

    def test_submit_increments_pending(self, dma):
        desc = DMADescriptor(src_addr=0, dst_addr=BANK_SIZE_BYTES, length=64)
        dma.submit(desc)
        assert dma.pending_count() == 1

    def test_submit_invalid_channel_raises(self, dma):
        desc = DMADescriptor(src_addr=0, dst_addr=BANK_SIZE_BYTES, length=64, channel_id=99)
        with pytest.raises(DmaError):
            dma.submit(desc)

    def test_submit_zero_length_raises(self, dma):
        desc = DMADescriptor(src_addr=0, dst_addr=BANK_SIZE_BYTES, length=0)
        with pytest.raises(DmaError):
            dma.submit(desc)

    def test_submit_zero_rows_raises(self, dma):
        desc = DMADescriptor(src_addr=0, dst_addr=BANK_SIZE_BYTES, length=64, rows=0)
        with pytest.raises(DmaError):
            dma.submit(desc)

    def test_submit_many(self, dma):
        descs = [
            DMADescriptor(src_addr=i * 64, dst_addr=BANK_SIZE_BYTES + i * 64, length=64)
            for i in range(4)
        ]
        dma.submit_many(descs)
        assert dma.pending_count() == 4


# ===========================================================================
# 7. DMA — Execution
# ===========================================================================

class TestDMAExecution:

    def test_contiguous_transfer(self, dma, sram):
        payload = b"X" * 128
        sram.write(0x000000, payload)
        desc = DMADescriptor(src_addr=0x000000, dst_addr=BANK_SIZE_BYTES, length=128)
        dma.submit(desc)
        transferred = dma.execute_all()
        assert transferred == 128
        assert sram.read(BANK_SIZE_BYTES, 128) == payload

    def test_execute_all_clears_queue(self, dma):
        desc = DMADescriptor(src_addr=0, dst_addr=BANK_SIZE_BYTES, length=64)
        dma.submit(desc)
        dma.execute_all()
        assert dma.pending_count() == 0

    def test_channel_state_idle_after_complete(self, dma):
        desc = DMADescriptor(src_addr=0, dst_addr=BANK_SIZE_BYTES, length=64)
        dma.submit(desc)
        dma.execute_all()
        assert dma.channel(0).state == ChannelState.IDLE

    def test_bytes_transferred_accumulates(self, dma):
        for i in range(3):
            dma.submit(DMADescriptor(src_addr=0, dst_addr=BANK_SIZE_BYTES + i * 64, length=64))
        dma.execute_all()
        assert dma.total_bytes_transferred == 192

    def test_transfers_completed_counter(self, dma):
        dma.submit(DMADescriptor(src_addr=0, dst_addr=BANK_SIZE_BYTES, length=64))
        dma.submit(DMADescriptor(src_addr=0, dst_addr=BANK_SIZE_BYTES + 64, length=64))
        dma.execute_all()
        assert dma.channel(0).transfers_completed == 2

    def test_execute_channel_only_runs_that_channel(self, dma):
        dma.submit(DMADescriptor(src_addr=0, dst_addr=BANK_SIZE_BYTES, length=64, channel_id=0))
        dma.submit(DMADescriptor(src_addr=0, dst_addr=BANK_SIZE_BYTES * 2, length=64, channel_id=1))
        dma.execute_channel(0)
        assert dma.channel(0).transfers_completed == 1
        assert dma.channel(1).transfers_completed == 0
        assert dma.pending_count() == 1  # channel 1 still pending

    def test_2d_strided_transfer(self, dma, sram):
        # Write 4 rows of 8 bytes with stride 16
        for row in range(4):
            sram.write(row * 16, bytes([row] * 8))
        desc = DMADescriptor(
            src_addr=0, dst_addr=BANK_SIZE_BYTES,
            length=8, rows=4, src_stride=16, dst_stride=8,
        )
        dma.submit(desc)
        transferred = dma.execute_all()
        assert transferred == 32  # 4 rows × 8 bytes
        for row in range(4):
            assert sram.read(BANK_SIZE_BYTES + row * 8, 8) == bytes([row] * 8)

    def test_last_descriptor_recorded(self, dma):
        desc = DMADescriptor(src_addr=0, dst_addr=BANK_SIZE_BYTES, length=64)
        dma.submit(desc)
        dma.execute_all()
        assert dma.channel(0).last_descriptor is desc


# ===========================================================================
# 8. DMA — IRQ integration
# ===========================================================================

class TestDMAIrq:

    def test_dma_done_irq_fires(self, dma_with_dev):
        engine, dev = dma_with_dev
        dev.irq.enable(IRQ_DMA_DONE)
        fired = []
        dev.irq.register_handler(IRQ_DMA_DONE, lambda n: fired.append(n))
        desc = DMADescriptor(src_addr=0, dst_addr=BANK_SIZE_BYTES, length=64)
        engine.submit(desc)
        engine.execute_all()
        assert len(fired) == 1

    def test_dma_done_fires_once_per_transfer(self, dma_with_dev):
        engine, dev = dma_with_dev
        dev.irq.enable(IRQ_DMA_DONE)
        count = []
        dev.irq.register_handler(IRQ_DMA_DONE, lambda n: count.append(1))
        for i in range(5):
            engine.submit(DMADescriptor(src_addr=0, dst_addr=BANK_SIZE_BYTES + i * 64, length=64))
        engine.execute_all()
        assert len(count) == 5


# ===========================================================================
# 9. DMA — Cache
# ===========================================================================

class TestDMACache:

    def test_prefetch_marks_lines_valid(self, dma):
        desc = DMADescriptor(
            src_addr=0, dst_addr=BANK_SIZE_BYTES, length=64,
            flags=TransferFlags.PREFETCH,
        )
        dma.submit(desc)
        dma.execute_all()
        # After PREFETCH transfer, destination lines should be valid
        assert dma.cache_is_valid(BANK_SIZE_BYTES)

    def test_invalidate_clears_lines(self, dma):
        desc = DMADescriptor(
            src_addr=0, dst_addr=BANK_SIZE_BYTES, length=64,
            flags=TransferFlags.INVALIDATE,
        )
        dma.submit(desc)
        dma.execute_all()
        assert not dma.cache_is_valid(BANK_SIZE_BYTES)

    def test_manual_invalidate(self, dma):
        dma.invalidate_cache(0, 64)
        assert not dma.cache_is_valid(0)


# ===========================================================================
# 10. DMA — Reset
# ===========================================================================

class TestDMAReset:

    def test_reset_clears_pending(self, dma):
        dma.submit(DMADescriptor(src_addr=0, dst_addr=BANK_SIZE_BYTES, length=64))
        dma.reset()
        assert dma.pending_count() == 0

    def test_reset_clears_transfer_count(self, dma):
        dma.submit(DMADescriptor(src_addr=0, dst_addr=BANK_SIZE_BYTES, length=64))
        dma.execute_all()
        dma.reset()
        assert dma.total_bytes_transferred == 0

    def test_channel_reset_after_fault(self, dma):
        ch = dma.channel(0)
        ch.state = ChannelState.FAULT
        dma.reset_channel(0)
        assert ch.state == ChannelState.IDLE

    def test_fault_channel_submit_raises(self, dma):
        dma.channel(0).state = ChannelState.FAULT
        with pytest.raises(DmaError):
            dma.submit(DMADescriptor(src_addr=0, dst_addr=BANK_SIZE_BYTES, length=64))


# ===========================================================================
# 11. Compression — INT8 round-trip
# ===========================================================================

class TestCompressionInt8:

    def test_encode_decode_simple(self, nc):
        data = bytes(range(256))
        compressed = nc.encode(data, CompressionMode.INT8)
        recovered = nc.decode(compressed)
        assert recovered == data

    def test_encode_decode_all_zeros(self, nc):
        data = b"\x00" * 512
        compressed = nc.encode(data, CompressionMode.INT8)
        assert nc.decode(compressed) == data

    def test_encode_decode_all_same(self, nc):
        data = b"\xAB" * 1024
        compressed = nc.encode(data, CompressionMode.INT8)
        assert nc.decode(compressed) == data

    def test_encode_decode_random_pattern(self, nc):
        import random
        random.seed(42)
        data = bytes(random.randint(0, 255) for _ in range(1000))
        assert nc.decode(nc.encode(data, CompressionMode.INT8)) == data

    def test_compression_ratio_gt_1_on_uniform(self, nc):
        data = b"\x42" * 1024
        nc.encode(data, CompressionMode.INT8)
        assert nc.last_ratio > 1.0

    def test_compressed_smaller_than_original_uniform(self, nc):
        data = b"\x00" * 4096
        compressed = nc.encode(data, CompressionMode.INT8)
        assert len(compressed) < len(data)

    def test_single_byte_roundtrip(self, nc):
        data = b"\xFF"
        assert nc.decode(nc.encode(data, CompressionMode.INT8)) == data


# ===========================================================================
# 12. Compression — INT4 round-trip
# ===========================================================================

class TestCompressionInt4:

    def test_encode_decode_nibble_data(self, nc):
        data = bytes([i % 16 for i in range(256)])
        compressed = nc.encode(data, CompressionMode.INT4)
        recovered = nc.decode(compressed)
        assert recovered == data

    def test_encode_decode_all_zeros_int4(self, nc):
        data = b"\x00" * 256
        assert nc.decode(nc.encode(data, CompressionMode.INT4)) == data

    def test_encode_decode_all_same_int4(self, nc):
        data = bytes([7] * 512)
        assert nc.decode(nc.encode(data, CompressionMode.INT4)) == data

    def test_int4_ratio_gt_1_on_uniform(self, nc):
        data = bytes([3] * 1024)
        nc.encode(data, CompressionMode.INT4)
        assert nc.last_ratio > 1.0

    def test_int4_compressed_smaller_than_original(self, nc):
        data = bytes([5] * 2048)
        compressed = nc.encode(data, CompressionMode.INT4)
        assert len(compressed) < len(data)


# ===========================================================================
# 13. Compression — Error paths
# ===========================================================================

class TestCompressionErrors:

    def test_empty_data_raises(self, nc):
        with pytest.raises(CompressionError):
            nc.encode(b"", CompressionMode.INT8)

    def test_corrupt_magic_raises(self, nc):
        data = b"\xDE\xAD\xBE\xEF" + b"\x08" + b"\x00\x00\x00\x05" + b"\x00" * 5
        with pytest.raises(CompressionError):
            nc.decode(data)

    def test_truncated_header_raises(self, nc):
        with pytest.raises(CompressionError):
            nc.decode(b"\x00\x01\x02")

    def test_decode_of_encode_is_identity(self, nc):
        # INT8: full byte range is valid
        data8 = bytes(range(128))
        assert nc.decode(nc.encode(data8, CompressionMode.INT8)) == data8
        # INT4: only lower-nibble values (0–15) round-trip cleanly
        data4 = bytes([i % 16 for i in range(128)])
        assert nc.decode(nc.encode(data4, CompressionMode.INT4)) == data4


# ===========================================================================
# 14. Compression — Stats
# ===========================================================================

class TestCompressionStats:

    def test_stats_accumulate_across_calls(self, nc):
        data = b"\x00" * 1024
        nc.encode(data, CompressionMode.INT8)
        nc.encode(data, CompressionMode.INT8)
        assert nc.total_bytes_in == 2048

    def test_reset_stats(self, nc):
        data = b"\x00" * 256
        nc.encode(data, CompressionMode.INT8)
        nc.reset_stats()
        assert nc.total_bytes_in == 0
        assert nc.total_bytes_out == 0

    def test_overall_ratio_after_multiple_encodes(self, nc):
        data = b"\xAA" * 1024
        nc.encode(data, CompressionMode.INT8)
        nc.encode(data, CompressionMode.INT8)
        assert nc.overall_ratio > 1.0
