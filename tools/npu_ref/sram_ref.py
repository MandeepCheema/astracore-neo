"""Python golden reference for the NPU SRAM subsystem.

Mirrors rtl/npu_sram_bank/npu_sram_bank.v and rtl/npu_sram_ctrl/npu_sram_ctrl.v
cycle-accurately.  Contract matches the RTL:

  - 1-cycle read latency (rdata appears the cycle AFTER re is asserted)
  - rdata holds when re is 0
  - Same-address same-cycle R+W: write wins (read returns new data)
  - Memory contents undefined after reset — caller must initialise
  - rdata register is cleared to 0 on reset (distinct from memory contents)
"""

from __future__ import annotations

from typing import Dict


def _mask(val: int, width: int) -> int:
    return val & ((1 << width) - 1)


class SramBank:
    """Mirror of rtl/npu_sram_bank/npu_sram_bank.v."""

    def __init__(self, data_w: int, depth: int):
        self.DATA_W = data_w
        self.DEPTH = depth
        # Memory contents start as None (== "undefined") so tests that
        # read before writing get a loud failure rather than silent 0.
        self.mem: Dict[int, int] = {}
        self.rdata: int = 0

    def reset(self) -> None:
        # RTL reset clears rdata only — memory contents are NOT cleared,
        # matching real SRAM behaviour.  Callers must initialise.
        self.rdata = 0

    def tick(
        self,
        *,
        we: int = 0,
        waddr: int = 0,
        wdata: int = 0,
        re: int = 0,
        raddr: int = 0,
    ) -> None:
        """Advance one clock edge."""
        # Compute next rdata (NBA semantics — sampled from pre-edge mem).
        next_rdata = self.rdata  # hold when re==0
        if re:
            if we and (waddr == raddr):
                # Write-wins on same-address same-cycle
                next_rdata = _mask(wdata, self.DATA_W)
            else:
                # Reading uninitialised location returns 0 (same as inferred
                # RAM in sim without explicit init).  Tests should initialise
                # to avoid relying on this, but don't crash.
                next_rdata = self.mem.get(raddr, 0)

        # Commit writes (NBA)
        if we:
            self.mem[waddr] = _mask(wdata, self.DATA_W)
        self.rdata = next_rdata


class SramCtrl:
    """Mirror of rtl/npu_sram_ctrl/npu_sram_ctrl.v."""

    def __init__(
        self,
        *,
        data_w: int = 8,
        acc_w: int = 32,
        n_rows: int = 16,
        n_cols: int = 16,
        weight_depth: int = 256,
        act_in_depth: int = 256,
        act_out_depth: int = 256,
        scratch_depth: int = 256,
    ):
        self.DATA_W = data_w
        self.ACC_W = acc_w
        self.N_ROWS = n_rows
        self.N_COLS = n_cols
        self.AI_DATA_W = n_rows * data_w
        self.AO_DATA_W = n_cols * acc_w

        # WA/WB are now N_COLS parallel narrow sub-banks each of depth
        # WEIGHT_DEPTH / N_COLS.  Wide ROW-read, narrow per-weight WRITE.
        assert weight_depth % n_cols == 0, \
            "WEIGHT_DEPTH must be divisible by N_COLS for parallel sub-banks"
        self.W_ROW_DEPTH = weight_depth // n_cols
        self.W_COL_IDX_W = max(1, (n_cols - 1).bit_length())
        self.W_ROW_DATA_W = n_cols * data_w

        self.wa_cols = [SramBank(data_w, self.W_ROW_DEPTH) for _ in range(n_cols)]
        self.wb_cols = [SramBank(data_w, self.W_ROW_DEPTH) for _ in range(n_cols)]
        self.ai = SramBank(self.AI_DATA_W, act_in_depth)
        self.ao = SramBank(self.AO_DATA_W, act_out_depth)
        self.sc = SramBank(data_w, scratch_depth)

        self.w_bank_sel_r: int = 0   # registered copy of sel (1-cycle delay)

    def reset(self) -> None:
        for b in self.wa_cols + self.wb_cols + [self.ai, self.ao, self.sc]:
            b.reset()
        self.w_bank_sel_r = 0

    def tick(
        self,
        *,
        # Weight port
        w_bank_sel: int = 0,
        w_re: int = 0, w_raddr: int = 0,
        w_we: int = 0, w_waddr: int = 0, w_wdata: int = 0,
        # AI port
        ai_re: int = 0, ai_raddr: int = 0,
        ai_we: int = 0, ai_waddr: int = 0, ai_wdata: int = 0,
        # AO port
        ao_re: int = 0, ao_raddr: int = 0,
        ao_we: int = 0, ao_waddr: int = 0, ao_wdata: int = 0,
        # SC port
        sc_re: int = 0, sc_raddr: int = 0,
        sc_we: int = 0, sc_waddr: int = 0, sc_wdata: int = 0,
    ) -> None:
        """Advance one clock edge.  Mirrors the RTL's weight-port routing.

        w_waddr is the LINEAR weight index; it decomposes as
            col_idx  = w_waddr & (N_COLS - 1)
            row_addr = w_waddr >> log2(N_COLS)
        w_raddr is the ROW address (narrower).
        """
        col_mask = (1 << self.W_COL_IDX_W) - 1
        ext_col_idx  = w_waddr & col_mask
        ext_row_addr = w_waddr >> self.W_COL_IDX_W

        # Route write to the correct sub-bank; others see we=0.
        for c in range(self.N_COLS):
            wa_we_c = w_we and (w_bank_sel == 1) and (ext_col_idx == c)
            wa_re_c = w_re and (w_bank_sel == 0)
            wb_we_c = w_we and (w_bank_sel == 0) and (ext_col_idx == c)
            wb_re_c = w_re and (w_bank_sel == 1)
            self.wa_cols[c].tick(we=wa_we_c, waddr=ext_row_addr, wdata=w_wdata,
                                  re=wa_re_c, raddr=w_raddr)
            self.wb_cols[c].tick(we=wb_we_c, waddr=ext_row_addr, wdata=w_wdata,
                                  re=wb_re_c, raddr=w_raddr)

        self.ai.tick(we=ai_we, waddr=ai_waddr, wdata=ai_wdata,
                     re=ai_re, raddr=ai_raddr)
        self.ao.tick(we=ao_we, waddr=ao_waddr, wdata=ao_wdata,
                     re=ao_re, raddr=ao_raddr)
        self.sc.tick(we=sc_we, waddr=sc_waddr, wdata=sc_wdata,
                     re=sc_re, raddr=sc_raddr)

        # Register the bank-sel (NBA) — used next cycle to pick w_rdata mux.
        self.w_bank_sel_r = w_bank_sel & 1

    # ---------------------------------------------------------------- views
    @property
    def w_rdata(self) -> int:
        """Wide ROW read — concatenate the N_COLS sub-bank rdata values.
        Bit [n*DATA_W +: DATA_W] = column n of the selected row.
        """
        cols = self.wa_cols if self.w_bank_sel_r == 0 else self.wb_cols
        out = 0
        for n in range(self.N_COLS):
            out |= (cols[n].rdata & ((1 << self.DATA_W) - 1)) << (n * self.DATA_W)
        return out

    @property
    def ai_rdata(self) -> int:
        return self.ai.rdata

    @property
    def ao_rdata(self) -> int:
        return self.ao.rdata

    @property
    def sc_rdata(self) -> int:
        return self.sc.rdata


# ----------------------------------------------------------------------------
# Self-check
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Single-bank basic read/write
    b = SramBank(data_w=8, depth=16)
    b.reset()
    assert b.rdata == 0
    b.tick(we=1, waddr=3, wdata=0xAB)          # write
    b.tick(re=1, raddr=3)                       # read-request
    assert b.rdata == 0xAB, f"expected 0xAB, got 0x{b.rdata:02x}"
    # Same-cycle R+W to same address → write wins
    b.tick(we=1, waddr=5, wdata=0x55, re=1, raddr=5)
    assert b.rdata == 0x55
    # Read when re=0 holds old value
    b.tick(re=0)
    assert b.rdata == 0x55

    # Controller: double-buffer semantics with WIDE weight read / narrow write.
    # N_COLS=4, DATA_W=8 → w_rdata is 32 bits packed {col3, col2, col1, col0}.
    # WEIGHT_DEPTH=16 → row depth = 16/4 = 4, so valid row addrs are 0..3.
    c = SramCtrl(data_w=8, n_rows=4, n_cols=4,
                 weight_depth=16, act_in_depth=16,
                 act_out_depth=16, scratch_depth=16)
    c.reset()
    # Write WB at linear waddr=7 → (row=1, col=3).  sel=0 routes to WB.
    c.tick(w_bank_sel=0, w_we=1, w_waddr=7, w_wdata=0x33)
    # Read row 1 from WA (sel=0) → all cols zero
    c.tick(w_bank_sel=0, w_re=1, w_raddr=1)
    assert c.w_rdata == 0, f"WA expected 0, got 0x{c.w_rdata:08x}"
    # Read row 1 from WB (sel=1) → col 3 has 0x33, others zero
    # Packing: col n at bits [n*8 +: 8] → 0x33 << 24 = 0x33000000
    c.tick(w_bank_sel=1, w_re=1, w_raddr=1)
    assert c.w_rdata == 0x33000000, f"WB row1 got 0x{c.w_rdata:08x}"

    # Bank isolation: writing scratch doesn't affect weights or activations
    c.tick(sc_we=1, sc_waddr=2, sc_wdata=0x99)
    c.tick(sc_re=1, sc_raddr=2)
    assert c.sc_rdata == 0x99
    # Weight row 1 of WB still holds 0x33000000
    c.tick(w_bank_sel=1, w_re=1, w_raddr=1)
    assert c.w_rdata == 0x33000000

    # Wide bank: AI uses N_ROWS*DATA_W bits = 32 bits for 4x8
    wide_val = (0x11 << 0) | (0x22 << 8) | (0x33 << 16) | (0x44 << 24)
    c.tick(ai_we=1, ai_waddr=0, ai_wdata=wide_val)
    c.tick(ai_re=1, ai_raddr=0)
    assert c.ai_rdata == wide_val, f"ai_rdata=0x{c.ai_rdata:08x} vs 0x{wide_val:08x}"

    print("sram_ref self-check: PASS")
