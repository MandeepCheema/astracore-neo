"""Python golden reference for rtl/npu_tile_ctrl/npu_tile_ctrl.v.

Cycle-accurate FSM mirror.  Each `tick()` call is one clock edge; after
the call the model's public attributes match what the RTL outputs on the
equivalent cycle.

The reference models the FSM only — it does NOT instantiate an SRAM or
systolic array.  Tests therefore verify that the control signals the RTL
drives match the expected sequence cycle-for-cycle.
"""

from __future__ import annotations


# FSM state encoding (must match RTL)
S_IDLE      = 0
S_PRELOAD   = 1
S_EXEC_PREP = 2
S_EXECUTE   = 3
S_DRAIN     = 4
S_STORE     = 5
S_DONE      = 6


def _mask(val: int, width: int) -> int:
    return val & ((1 << width) - 1)


class TileCtrl:
    def __init__(self, *, n_rows: int = 4, n_cols: int = 4, acc_w: int = 32,
                 w_addr_w: int | None = None,
                 ai_addr_w: int = 8, ao_addr_w: int = 8,
                 k_w: int = 16, drain_cycles: int = 2):
        self.N_ROWS = n_rows
        self.N_COLS = n_cols
        self.ACC_W = acc_w
        self.ACC_DATA_W = n_cols * acc_w
        self.W_ADDR_W = w_addr_w if w_addr_w is not None else max(
            1, (n_rows - 1).bit_length())
        self.AI_ADDR_W = ai_addr_w
        self.AO_ADDR_W = ao_addr_w
        self.K_W = k_w
        self.DRAIN_CYCLES = drain_cycles

        self.state = S_IDLE
        self.busy = 0
        self.done = 0
        self.w_bank_sel = 0
        self.w_re = 0
        self.w_raddr = 0
        self.array_load_valid = 0
        self.array_load_cell_addr = 0
        self.array_clear_acc = 0
        self.array_acc_load_valid = 0
        self.array_acc_load_data = 0
        self.ai_re = 0
        self.ai_raddr = 0
        self.array_exec_valid = 0
        self.array_afu_in_valid = 0
        self.ao_we = 0
        self.ao_waddr = 0

        self.preload_cnt = 0
        self.exec_cnt = 0
        self.drain_cnt = 0
        self.preload_addr_r = 0
        self.preload_active_r = 0

        # Latched config (written on start)
        self.cfg_k = 0
        self.cfg_ai_base = 0
        self.cfg_ao_base = 0
        self.cfg_acc_init_mode_r = 0
        self.cfg_acc_init_data_r = 0

    def reset(self) -> None:
        self.__init__(n_rows=self.N_ROWS, n_cols=self.N_COLS,
                      w_addr_w=self.W_ADDR_W, ai_addr_w=self.AI_ADDR_W,
                      ao_addr_w=self.AO_ADDR_W, k_w=self.K_W,
                      drain_cycles=self.DRAIN_CYCLES)

    def tick(self, *, start: int = 0,
             cfg_k: int = 0, cfg_ai_base: int = 0, cfg_ao_base: int = 0,
             cfg_acc_init_mode: int = 0,
             cfg_acc_init_data: int = 0) -> None:
        """Advance one clock edge using snapshot-then-commit semantics
        (matches Verilog NBA ordering)."""
        # --- snapshot pre-edge values ---
        p_state          = self.state
        p_busy           = self.busy
        p_w_bank_sel     = self.w_bank_sel
        p_preload_cnt    = self.preload_cnt
        p_exec_cnt       = self.exec_cnt
        p_drain_cnt      = self.drain_cnt
        p_preload_addr_r = self.preload_addr_r
        p_preload_active_r = self.preload_active_r

        # --- defaults for next state (matches RTL defaults at top of always) ---
        n_state          = p_state
        n_busy           = p_busy
        n_done           = 0
        n_w_bank_sel     = p_w_bank_sel
        n_w_re           = 0
        n_w_raddr        = self.w_raddr             # no default reset for addr
        n_array_load_valid     = p_preload_active_r  # pipeline from A
        n_array_load_cell_addr = p_preload_addr_r
        n_array_clear_acc      = 0
        n_array_acc_load_valid = 0
        n_array_acc_load_data  = 0
        n_ai_re          = 0
        n_ai_raddr       = self.ai_raddr
        n_array_exec_valid = 0
        n_array_afu_in_valid = 0
        n_ao_we          = 0
        n_ao_waddr       = self.ao_waddr
        n_preload_cnt    = p_preload_cnt
        n_exec_cnt       = p_exec_cnt
        n_drain_cnt      = p_drain_cnt
        n_preload_addr_r = p_preload_addr_r
        n_preload_active_r = 0  # default; overridden in PRELOAD case

        if p_state == S_IDLE:
            if start:
                n_busy = 1
                n_state = S_PRELOAD
                n_preload_cnt = 0
                n_w_bank_sel = 0
                self.cfg_k = cfg_k
                self.cfg_ai_base = cfg_ai_base
                self.cfg_ao_base = cfg_ao_base
                self.cfg_acc_init_mode_r = cfg_acc_init_mode & 1
                self.cfg_acc_init_data_r = cfg_acc_init_data & ((1 << self.ACC_DATA_W) - 1)

        elif p_state == S_PRELOAD:
            n_w_re = 1
            n_w_raddr = p_preload_cnt
            n_preload_addr_r = p_preload_cnt
            n_preload_active_r = 1
            if p_preload_cnt == self.N_ROWS - 1:
                n_state = S_EXEC_PREP
                n_preload_cnt = 0
            else:
                n_preload_cnt = p_preload_cnt + 1

        elif p_state == S_EXEC_PREP:
            if self.cfg_acc_init_mode_r:
                n_array_acc_load_valid = 1
                n_array_acc_load_data = self.cfg_acc_init_data_r
            else:
                n_array_clear_acc = 1
            n_state = S_EXECUTE
            n_exec_cnt = 0

        elif p_state == S_EXECUTE:
            n_ai_re = 1
            n_ai_raddr = _mask(
                self.cfg_ai_base + _mask(p_exec_cnt, self.AI_ADDR_W),
                self.AI_ADDR_W)
            n_array_exec_valid = 1
            if p_exec_cnt == self.cfg_k - 1:
                n_state = S_DRAIN
                n_drain_cnt = 0
            else:
                n_exec_cnt = p_exec_cnt + 1

        elif p_state == S_DRAIN:
            if p_drain_cnt == self.DRAIN_CYCLES - 1:
                # Pulse writeback-AFU latch on the last DRAIN cycle (see RTL).
                n_array_afu_in_valid = 1
                n_state = S_STORE
            else:
                n_drain_cnt = p_drain_cnt + 1

        elif p_state == S_STORE:
            n_ao_we = 1
            n_ao_waddr = self.cfg_ao_base
            n_state = S_DONE

        elif p_state == S_DONE:
            n_done = 1
            n_busy = 0
            n_state = S_IDLE

        # --- commit ---
        self.state = n_state
        self.busy = n_busy
        self.done = n_done
        self.w_bank_sel = n_w_bank_sel
        self.w_re = n_w_re
        self.w_raddr = _mask(n_w_raddr, self.W_ADDR_W)
        self.array_load_valid = n_array_load_valid
        self.array_load_cell_addr = _mask(n_array_load_cell_addr, self.W_ADDR_W)
        self.array_clear_acc = n_array_clear_acc
        self.array_acc_load_valid = n_array_acc_load_valid
        self.array_acc_load_data = _mask(n_array_acc_load_data, self.ACC_DATA_W)
        self.ai_re = n_ai_re
        self.ai_raddr = _mask(n_ai_raddr, self.AI_ADDR_W)
        self.array_exec_valid = n_array_exec_valid
        self.array_afu_in_valid = n_array_afu_in_valid
        self.ao_we = n_ao_we
        self.ao_waddr = _mask(n_ao_waddr, self.AO_ADDR_W)
        self.preload_cnt = _mask(n_preload_cnt, self.W_ADDR_W)
        self.exec_cnt = _mask(n_exec_cnt, self.K_W)
        self.drain_cnt = n_drain_cnt & 0xF
        self.preload_addr_r = _mask(n_preload_addr_r, self.W_ADDR_W)
        self.preload_active_r = n_preload_active_r


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Small instance: 2x2 array, K=3 activations
    tc = TileCtrl(n_rows=2, n_cols=2, ai_addr_w=4, ao_addr_w=4, k_w=4)
    tc.reset()
    # Reset state
    assert tc.state == S_IDLE
    assert tc.busy == 0 and tc.done == 0

    # Drive start
    tc.tick(start=1, cfg_k=3, cfg_ai_base=5, cfg_ao_base=9)
    assert tc.state == S_PRELOAD
    assert tc.busy == 1

    # With wide-weight SRAM, PRELOAD walks N_ROWS addresses (= 2 for 2x2).
    preload_addrs_seen = []
    for _ in range(2):
        tc.tick()
        preload_addrs_seen.append((tc.w_re, tc.w_raddr))
    assert tc.state == S_EXEC_PREP, f"state={tc.state}"
    assert preload_addrs_seen == [(1, 0), (1, 1)], preload_addrs_seen

    # Next cycle: EXEC_PREP → array_clear_acc=1, state → EXECUTE
    tc.tick()
    assert tc.state == S_EXECUTE
    assert tc.array_clear_acc == 1

    # 3 cycles of EXECUTE — sample AFTER each tick
    exec_trace = []
    for _ in range(3):
        tc.tick()
        exec_trace.append((tc.ai_re, tc.ai_raddr, tc.array_exec_valid))
    assert tc.state == S_DRAIN, f"state={tc.state}"
    assert exec_trace == [(1, 5, 1), (1, 6, 1), (1, 7, 1)], exec_trace

    # 2 DRAIN cycles (drain_cnt 0→1, then state→STORE)
    tc.tick()
    assert tc.state == S_DRAIN, f"state={tc.state} drain_cnt={tc.drain_cnt}"
    tc.tick()
    assert tc.state == S_STORE, f"state={tc.state}"

    # STORE: pulses ao_we and transitions to DONE
    tc.tick()
    assert tc.ao_we == 1 and tc.ao_waddr == 9
    assert tc.state == S_DONE

    # DONE: pulses done and returns to IDLE
    tc.tick()
    assert tc.done == 1 and tc.state == S_IDLE
    assert tc.busy == 0

    print("tile_ctrl_ref self-check: PASS")
