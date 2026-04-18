"""Python golden reference for rtl/npu_dma/npu_dma.v.

Cycle-accurate mirror of the RTL.  tick() advances one clock edge using
strict snapshot-then-commit semantics so NBA ordering exactly matches
Verilog: all next-state values are computed from PRE-edge state and
committed atomically at the end of tick().

Caller protocol for the source-memory interface:
  - Caller inspects `mem_re` / `mem_raddr` each cycle (combinational).
  - If `mem_re=1`, the caller must drive `mem_rdata` via `set_mem_rdata()`
    BEFORE the next `tick()`, because the RTL's destination write that
    cycle latches the memory data combinationally through dst_wdata.
"""

from __future__ import annotations

from typing import Dict, Optional


def _mask(val: int, width: int) -> int:
    return val & ((1 << width) - 1)


class Dma:
    def __init__(self, *, data_w: int = 8, src_addr_w: int = 32,
                 dst_addr_w: int = 16, len_w: int = 16):
        self.DATA_W = data_w
        self.SRC_ADDR_W = src_addr_w
        self.DST_ADDR_W = dst_addr_w
        self.LEN_W = len_w

        # Latched configuration
        self.src_addr_r = 0
        self.dst_addr_r = 0
        self.tile_h_r = 0
        self.tile_w_r = 0
        self.src_stride_r = 0
        self.pad_top_r = 0
        self.pad_bot_r = 0
        self.pad_left_r = 0
        self.pad_right_r = 0

        # Stage A state
        self.busy_a = 0
        self.i_pos = 0
        self.j_pos = 0
        self.dst_offset_a = 0

        # Stage B state
        self.busy_b = 0
        self.is_real_b = 0
        self.dst_offset_b = 0

        # Control outputs
        self.busy = 0
        self.done = 0

        # External memory bus held by caller
        self._mem_rdata = 0

        # Refresh combinational outputs
        self._refresh()

    # ---------------------------------------------------------------- reset
    def reset(self) -> None:
        self.src_addr_r = 0
        self.dst_addr_r = 0
        self.tile_h_r = 0
        self.tile_w_r = 0
        self.src_stride_r = 0
        self.pad_top_r = 0
        self.pad_bot_r = 0
        self.pad_left_r = 0
        self.pad_right_r = 0
        self.busy_a = 0
        self.i_pos = 0
        self.j_pos = 0
        self.dst_offset_a = 0
        self.busy_b = 0
        self.is_real_b = 0
        self.dst_offset_b = 0
        self.busy = 0
        self.done = 0
        self._mem_rdata = 0
        self._refresh()

    # ---------------------------------------------------- internal utilities
    def _output_h(self) -> int:
        return self.tile_h_r + self.pad_top_r + self.pad_bot_r

    def _output_w(self) -> int:
        return self.tile_w_r + self.pad_left_r + self.pad_right_r

    def _compute_is_real_a(self, busy_a: int, i_pos: int, j_pos: int) -> int:
        if not busy_a:
            return 0
        in_row = self.pad_top_r <= i_pos < self.pad_top_r + self.tile_h_r
        in_col = self.pad_left_r <= j_pos < self.pad_left_r + self.tile_w_r
        return int(bool(in_row and in_col))

    def _refresh(self) -> None:
        """Update combinational outputs (mem_re/raddr and dst_we/addr/data)
        from current registered state."""
        is_real_a = self._compute_is_real_a(self.busy_a, self.i_pos, self.j_pos)
        self.mem_re = is_real_a
        if is_real_a:
            src_row = self.i_pos - self.pad_top_r
            src_col = self.j_pos - self.pad_left_r
            self.mem_raddr = _mask(
                self.src_addr_r + src_row * self.src_stride_r + src_col,
                self.SRC_ADDR_W)
        else:
            self.mem_raddr = 0

        self.dst_we = int(self.busy_b)
        self.dst_waddr = _mask(self.dst_addr_r + self.dst_offset_b,
                               self.DST_ADDR_W)
        self.dst_wdata = (_mask(self._mem_rdata, self.DATA_W)
                          if self.is_real_b else 0)

    # --------------------------------------------------------------------- API
    def set_mem_rdata(self, value: int) -> None:
        self._mem_rdata = _mask(value, self.DATA_W)
        self._refresh()

    def tick(
        self,
        *,
        start: int = 0,
        cfg_src_addr: int = 0,
        cfg_dst_addr: int = 0,
        cfg_tile_h: int = 0,
        cfg_tile_w: int = 0,
        cfg_src_stride: int = 0,
        cfg_pad_top: int = 0,
        cfg_pad_bot: int = 0,
        cfg_pad_left: int = 0,
        cfg_pad_right: int = 0,
    ) -> None:
        """Advance one clock edge.  All NBAs evaluated from pre-edge state."""
        # --- snapshot pre-edge state ---
        p_busy        = self.busy
        p_busy_a      = self.busy_a
        p_busy_b      = self.busy_b
        p_i_pos       = self.i_pos
        p_j_pos       = self.j_pos
        p_dst_off_a   = self.dst_offset_a
        p_is_real_a   = self._compute_is_real_a(p_busy_a, p_i_pos, p_j_pos)

        # --- defaults for next state ---
        n_busy        = p_busy
        n_busy_a      = p_busy_a
        n_i_pos       = p_i_pos
        n_j_pos       = p_j_pos
        n_dst_off_a   = p_dst_off_a
        n_done        = 0

        latching = bool(start and not p_busy)

        # Config latch on start (only when idle)
        if latching:
            self.src_addr_r   = _mask(cfg_src_addr,   self.SRC_ADDR_W)
            self.dst_addr_r   = _mask(cfg_dst_addr,   self.DST_ADDR_W)
            self.tile_h_r     = _mask(cfg_tile_h,     self.LEN_W)
            self.tile_w_r     = _mask(cfg_tile_w,     self.LEN_W)
            self.src_stride_r = _mask(cfg_src_stride, self.LEN_W)
            self.pad_top_r    = cfg_pad_top   & 0xF
            self.pad_bot_r    = cfg_pad_bot   & 0xF
            self.pad_left_r   = cfg_pad_left  & 0xF
            self.pad_right_r  = cfg_pad_right & 0xF
            n_busy      = 1
            n_busy_a    = 1
            n_i_pos     = 0
            n_j_pos     = 0
            n_dst_off_a = 0

        # Stage A advance (uses PRE-edge busy_a; matches RTL's `if (busy_a)`
        # which reads the register value before any NBAs commit)
        if p_busy_a and not latching:
            out_w = self._output_w()
            out_h = self._output_h()
            j_last = (p_j_pos == out_w - 1) if out_w > 0 else True
            i_last = (p_i_pos == out_h - 1) if out_h > 0 else True
            is_last = j_last and i_last
            if is_last:
                n_busy_a = 0
                # i_pos / j_pos / dst_offset_a unchanged
            elif j_last:
                n_j_pos     = 0
                n_i_pos     = p_i_pos + 1
                n_dst_off_a = _mask(p_dst_off_a + 1, self.DST_ADDR_W)
            else:
                n_j_pos     = p_j_pos + 1
                n_dst_off_a = _mask(p_dst_off_a + 1, self.DST_ADDR_W)

        # Pipeline registers always track pre-edge stage A
        n_busy_b     = p_busy_a
        n_is_real_b  = p_is_real_a
        n_dst_off_b  = p_dst_off_a

        # Completion: busy_b was 1 last cycle, busy_a just went 0 → this is
        # the final write cycle; done pulses next cycle.
        if p_busy_b and not p_busy_a:
            n_busy = 0
            n_done = 1

        # --- commit next state atomically ---
        self.busy         = n_busy
        self.busy_a       = n_busy_a
        self.busy_b       = n_busy_b
        self.i_pos        = n_i_pos
        self.j_pos        = n_j_pos
        self.dst_offset_a = n_dst_off_a
        self.is_real_b    = n_is_real_b
        self.dst_offset_b = n_dst_off_b
        self.done         = n_done

        self._refresh()


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    def _run(dma: Dma, cfg: Dict, src: Dict[int, int]) -> Dict[int, int]:
        """Run a complete transfer in the Python model, simulating 1-cycle
        memory-read latency.  Returns captured destination writes as a
        {dst_addr: data} dict."""
        captured: Dict[int, int] = {}

        # Cycle 0: assert start
        dma.tick(start=1, **cfg)
        # Each subsequent cycle: capture request issued by PREV cycle,
        # drive mem_rdata now, capture any write, then tick.
        pending_src: Optional[int] = None

        for _ in range(100_000):
            # Drive mem_rdata from previous cycle's read request (1-cycle latency)
            if pending_src is not None:
                dma.set_mem_rdata(src.get(pending_src, 0))
            else:
                dma.set_mem_rdata(0)
            # Snapshot this-cycle's request (combinational) for next cycle
            pending_src = dma.mem_raddr if dma.mem_re else None
            # Capture a destination write if one is happening
            if dma.dst_we:
                captured[dma.dst_waddr] = dma.dst_wdata
            # Exit AFTER we've captured everything
            if dma.done:
                return captured
            dma.tick()
        raise RuntimeError("DMA did not complete")

    # Test 1: linear 5-word transfer src[100..104] → dst[0..4]
    dma = Dma()
    src = {100 + k: 0x10 + k for k in range(5)}
    dst = _run(dma, dict(cfg_src_addr=100, cfg_dst_addr=0,
                         cfg_tile_h=1, cfg_tile_w=5, cfg_src_stride=5), src)
    expected = {k: 0x10 + k for k in range(5)}
    assert dst == expected, (dst, expected)

    # Test 2: 2D tile 3×4 with stride=10
    dma.reset()
    src = {200 + r * 10 + c: 0x20 + r * 4 + c
           for r in range(3) for c in range(4)}
    dst = _run(dma, dict(cfg_src_addr=200, cfg_dst_addr=0,
                         cfg_tile_h=3, cfg_tile_w=4, cfg_src_stride=10), src)
    expected = {r * 4 + c: 0x20 + r * 4 + c for r in range(3) for c in range(4)}
    assert dst == expected, (dst, expected)

    # Test 3: 2D tile 2×2 + pad 1 each side → 4×4 output, zeros on border
    dma.reset()
    src = {50: 0xA, 51: 0xB, 60: 0xC, 61: 0xD}
    dst = _run(dma, dict(cfg_src_addr=50, cfg_dst_addr=0,
                         cfg_tile_h=2, cfg_tile_w=2, cfg_src_stride=10,
                         cfg_pad_top=1, cfg_pad_bot=1,
                         cfg_pad_left=1, cfg_pad_right=1), src)
    expected = {i: 0 for i in range(16)}
    expected.update({5: 0xA, 6: 0xB, 9: 0xC, 10: 0xD})
    assert dst == expected, (dst, expected)

    # Test 4: asymmetric padding
    dma.reset()
    src = {0: 0x1, 1: 0x2, 2: 0x3}
    dst = _run(dma, dict(cfg_src_addr=0, cfg_dst_addr=0,
                         cfg_tile_h=1, cfg_tile_w=3, cfg_src_stride=3,
                         cfg_pad_top=0, cfg_pad_bot=1,
                         cfg_pad_left=2, cfg_pad_right=0), src)
    # Output is 2 rows × 5 cols.  Row 0 is real (i=0, pad_top=0 → in tile).
    # Inside row 0: cols 0..1 are pad (pad_left=2), col 2..4 are real.
    # Row 1 is all pad (pad_bot=1).
    expected = {
        0: 0, 1: 0,                   # row 0, pad_left
        2: 0x1, 3: 0x2, 4: 0x3,        # row 0, real
        5: 0, 6: 0, 7: 0, 8: 0, 9: 0,  # row 1, all pad
    }
    assert dst == expected, (dst, expected)

    print("dma_ref self-check: PASS")
