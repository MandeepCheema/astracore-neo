"""
AstraCore Neo — Camera Detection Receiver cocotb testbench

FIFO_DEPTH = 16 (default). Writes accept camera_detection_t fields in parallel.
Downstream read uses valid/ready handshake.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


FIFO_DEPTH = 16


async def reset_dut(dut):
    dut.rst_n.value            = 0
    dut.wr_valid.value         = 0
    dut.wr_class_id.value      = 0
    dut.wr_confidence.value    = 0
    dut.wr_bbox_x.value        = 0
    dut.wr_bbox_y.value        = 0
    dut.wr_bbox_w.value        = 0
    dut.wr_bbox_h.value        = 0
    dut.wr_timestamp_us.value  = 0
    dut.wr_camera_id.value     = 0
    dut.rd_ready.value         = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def push_det(dut, class_id, conf=128, bbox=(0, 0, 100, 100),
                   ts=1000, cam=0):
    dut.wr_class_id.value     = class_id
    dut.wr_confidence.value   = conf
    dut.wr_bbox_x.value       = bbox[0]
    dut.wr_bbox_y.value       = bbox[1]
    dut.wr_bbox_w.value       = bbox[2]
    dut.wr_bbox_h.value       = bbox[3]
    dut.wr_timestamp_us.value = ts
    dut.wr_camera_id.value    = cam
    dut.wr_valid.value        = 1
    await RisingEdge(dut.clk)
    dut.wr_valid.value        = 0
    await RisingEdge(dut.clk)


async def pop_det(dut):
    """Wait for rd_valid then pop one entry. Returns captured dict."""
    while int(dut.rd_valid.value) == 0:
        await RisingEdge(dut.clk)
    captured = {
        "class_id":     int(dut.rd_class_id.value),
        "confidence":   int(dut.rd_confidence.value),
        "bbox_x":       int(dut.rd_bbox_x.value),
        "bbox_y":       int(dut.rd_bbox_y.value),
        "bbox_w":       int(dut.rd_bbox_w.value),
        "bbox_h":       int(dut.rd_bbox_h.value),
        "timestamp_us": int(dut.rd_timestamp_us.value),
        "camera_id":    int(dut.rd_camera_id.value),
    }
    dut.rd_ready.value = 1
    await RisingEdge(dut.clk)
    dut.rd_ready.value = 0
    await RisingEdge(dut.clk)
    return captured


# ===========================================================================
# Tests
# ===========================================================================

@cocotb.test()
async def test_initial_state(dut):
    """After reset: FIFO empty, not full, wr_ready=1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    assert int(dut.fifo_empty.value) == 1
    assert int(dut.fifo_full.value)  == 0
    assert int(dut.fifo_count.value) == 0
    assert int(dut.wr_ready.value)   == 1
    assert int(dut.rd_valid.value)   == 0
    dut._log.info("initial_state passed")


@cocotb.test()
async def test_single_push_pop(dut):
    """Push one detection, pop it, verify all fields round-trip."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await push_det(dut, class_id=1, conf=200,
                    bbox=(50, 60, 300, 400),
                    ts=0xDEADBEEF, cam=2)
    assert int(dut.fifo_count.value) == 1
    assert int(dut.rd_valid.value)   == 1

    det = await pop_det(dut)
    assert det["class_id"]     == 1
    assert det["confidence"]   == 200
    assert det["bbox_x"]       == 50
    assert det["bbox_y"]       == 60
    assert det["bbox_w"]       == 300
    assert det["bbox_h"]       == 400
    assert det["timestamp_us"] == 0xDEADBEEF
    assert det["camera_id"]    == 2
    assert int(dut.fifo_empty.value) == 1
    assert int(dut.total_received.value) == 1
    dut._log.info("single_push_pop passed")


@cocotb.test()
async def test_push_multiple_pop_in_order(dut):
    """FIFO order preserved for multiple detections."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for i in range(5):
        await push_det(dut, class_id=i + 10, conf=i, cam=i & 3)

    assert int(dut.fifo_count.value) == 5

    for i in range(5):
        det = await pop_det(dut)
        assert det["class_id"]   == i + 10
        assert det["confidence"] == i
        assert det["camera_id"]  == (i & 3)

    assert int(dut.fifo_empty.value) == 1
    dut._log.info("push_multiple_pop_in_order passed")


@cocotb.test()
async def test_fifo_full_and_drop(dut):
    """Fill the FIFO to capacity; extra write is dropped."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for i in range(FIFO_DEPTH):
        await push_det(dut, class_id=i)
    assert int(dut.fifo_full.value)  == 1
    assert int(dut.fifo_count.value) == FIFO_DEPTH
    assert int(dut.wr_ready.value)   == 0

    # Attempt one more write — should be dropped
    await push_det(dut, class_id=99)
    assert int(dut.fifo_count.value) == FIFO_DEPTH
    assert int(dut.total_dropped.value) == 1
    assert int(dut.total_received.value) == FIFO_DEPTH
    dut._log.info("fifo_full_and_drop passed")


@cocotb.test()
async def test_fill_drain_refill(dut):
    """Fill → drain → refill cycles work and pointers wrap correctly."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for i in range(FIFO_DEPTH):
        await push_det(dut, class_id=i)
    for i in range(FIFO_DEPTH):
        det = await pop_det(dut)
        assert det["class_id"] == i
    assert int(dut.fifo_empty.value) == 1

    # Second round — pointers have wrapped
    for i in range(FIFO_DEPTH):
        await push_det(dut, class_id=100 + i)
    for i in range(FIFO_DEPTH):
        det = await pop_det(dut)
        assert det["class_id"] == 100 + i

    assert int(dut.total_received.value) == 2 * FIFO_DEPTH
    assert int(dut.total_dropped.value)  == 0
    dut._log.info("fill_drain_refill passed")


@cocotb.test()
async def test_simultaneous_push_pop(dut):
    """A push and a pop in the same cycle keep count constant and preserve order."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Prime the FIFO with one entry
    await push_det(dut, class_id=1)

    # Simultaneous: push class=2 while popping class=1
    dut.wr_class_id.value = 2
    dut.wr_valid.value    = 1
    dut.rd_ready.value    = 1
    # The current rd_valid should show class=1 right now
    assert int(dut.rd_class_id.value) == 1
    await RisingEdge(dut.clk)
    dut.wr_valid.value = 0
    dut.rd_ready.value = 0
    await RisingEdge(dut.clk)

    # Count should still be 1
    assert int(dut.fifo_count.value) == 1

    det = await pop_det(dut)
    assert det["class_id"] == 2, \
        f"next pop should be class=2, got {det['class_id']}"
    dut._log.info("simultaneous_push_pop passed")
