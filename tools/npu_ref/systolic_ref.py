"""Python golden reference for rtl/npu_systolic_array/npu_systolic_array.v.

Cycle-accurate mirror.  One `tick()` call = one rising clock edge in the RTL.
Post-tick attributes (`c_valid`, `c_vec`) match the RTL values visible after
that edge's NBAs have committed.

Usage:
    sa = SystolicArray(n_rows=4, n_cols=4)
    sa.reset()
    # Load a 4x4 weight tile (row-major)
    for k in range(4):
        for n in range(4):
            sa.tick(w_load=1, w_addr=k*4+n, w_data=W[k][n])
    # Stream activation vectors
    sa.tick(a_valid=1, a_vec=[1, 2, 3, 4])
    # Read output
    assert sa.c_vec == [dot_product_column_0, ..., dot_product_column_3]
"""

from __future__ import annotations

from typing import List, Sequence


def _sign_extend(val: int, width: int) -> int:
    mask = (1 << width) - 1
    val &= mask
    if val & (1 << (width - 1)):
        return val - (1 << width)
    return val


def _mask(val: int, width: int) -> int:
    return val & ((1 << width) - 1)


class SystolicArray:
    """Weight-stationary MVM engine — Python mirror of the RTL."""

    def __init__(self, n_rows: int = 4, n_cols: int = 4,
                 data_w: int = 8, acc_w: int = 32):
        self.N_ROWS = n_rows
        self.N_COLS = n_cols
        self.DATA_W = data_w
        self.ACC_W = acc_w
        self.W: List[int] = [0] * (n_rows * n_cols)   # row-major, stored unsigned
        self.acc: List[int] = [0] * n_cols            # stored unsigned (ACC_W bits)
        self.c_vec: List[int] = [0] * n_cols          # signed ACC_W view after last tick
        self.c_valid: int = 0

    def reset(self) -> None:
        self.W = [0] * (self.N_ROWS * self.N_COLS)
        self.acc = [0] * self.N_COLS
        self.c_vec = [0] * self.N_COLS
        self.c_valid = 0

    # ---------------------------------------------------------------- tick
    def tick(
        self,
        *,
        w_load: int = 0,
        w_addr: int = 0,
        w_data: int = 0,
        clear_acc: int = 0,
        acc_load_valid: int = 0,
        acc_load_data: int = 0,
        a_valid: int = 0,
        a_vec: Sequence[int] | None = None,
        precision_mode: int = 0,   # 00=INT8, 01=INT4, 10=INT2, 11=FP16(placeholder)
        sparse_skip_vec: int = 0,  # N_ROWS-bit mask; bit k = 1 zeroes product for row k
    ) -> None:
        """Advance one clock edge.  All inputs sampled pre-edge."""
        if a_vec is None:
            a_vec = [0] * self.N_ROWS
        assert len(a_vec) == self.N_ROWS, "a_vec length must equal N_ROWS"

        def _int4_pair_prod(w_byte: int, a_byte: int) -> int:
            w_hi = _sign_extend((w_byte >> 4) & 0xF, 4)
            w_lo = _sign_extend(w_byte & 0xF, 4)
            a_hi = _sign_extend((a_byte >> 4) & 0xF, 4)
            a_lo = _sign_extend(a_byte & 0xF, 4)
            return w_hi * a_hi + w_lo * a_lo

        def _int2_quad_prod(w_byte: int, a_byte: int) -> int:
            total = 0
            for sh in (0, 2, 4, 6):
                w2 = _sign_extend((w_byte >> sh) & 0x3, 2)
                a2 = _sign_extend((a_byte >> sh) & 0x3, 2)
                total += w2 * a2
            return total

        # Compute combinational dot products using PRE-edge weights & inputs.
        dots = [0] * self.N_COLS
        for n in range(self.N_COLS):
            s = 0
            for k in range(self.N_ROWS):
                if (sparse_skip_vec >> k) & 0x1:
                    continue   # pruned row; contributes 0 to reduction
                w_byte = _mask(self.W[k * self.N_COLS + n], self.DATA_W)
                a_byte = _mask(a_vec[k], self.DATA_W)
                if precision_mode == 0b01:
                    s += _int4_pair_prod(w_byte, a_byte)
                elif precision_mode == 0b10:
                    s += _int2_quad_prod(w_byte, a_byte)
                else:  # INT8 or FP16-placeholder: full INT8 multiply
                    w = _sign_extend(w_byte, self.DATA_W)
                    a = _sign_extend(a_byte, self.DATA_W)
                    s += w * a
            dots[n] = s

        # Unpack acc_load_data into per-column values
        acc_load_vals = [
            (acc_load_data >> (n * self.ACC_W)) & ((1 << self.ACC_W) - 1)
            for n in range(self.N_COLS)
        ]

        # Next state (NBA semantics with priority: clear > acc_load > accumulate)
        next_acc = list(self.acc)
        next_c_vec = list(self.c_vec)
        if clear_acc:
            next_acc = [0] * self.N_COLS
        elif acc_load_valid:
            next_acc = list(acc_load_vals)
        elif a_valid:
            for n in range(self.N_COLS):
                acc_signed = _sign_extend(self.acc[n], self.ACC_W)
                next_acc[n] = _mask(acc_signed + dots[n], self.ACC_W)

        # c_valid only for real accumulations
        next_c_valid = int(bool(a_valid) and not bool(clear_acc)
                           and not bool(acc_load_valid))
        for n in range(self.N_COLS):
            if clear_acc:
                next_c_vec[n] = 0
            elif acc_load_valid:
                next_c_vec[n] = acc_load_vals[n]
            elif a_valid:
                acc_signed = _sign_extend(self.acc[n], self.ACC_W)
                next_c_vec[n] = _mask(acc_signed + dots[n], self.ACC_W)
            # else: hold

        # Commit state atomically (post-edge view).
        # w_load now latches a FULL ROW of N_COLS weights per cycle.
        # w_data is packed: col n at bits [n*DATA_W +: DATA_W].
        if w_load:
            for n in range(self.N_COLS):
                weight = (w_data >> (n * self.DATA_W)) & ((1 << self.DATA_W) - 1)
                self.W[w_addr * self.N_COLS + n] = weight
        self.acc = next_acc
        self.c_vec = next_c_vec
        self.c_valid = next_c_valid

    # ------------------------------------------------------------ observables
    @property
    def c_vec_signed(self) -> List[int]:
        return [_sign_extend(v, self.ACC_W) for v in self.c_vec]

    def acc_signed(self) -> List[int]:
        return [_sign_extend(v, self.ACC_W) for v in self.acc]


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _pack_row(row: Sequence[int], data_w: int) -> int:
    """Pack a row of weights (LSB-first) into a single wide int."""
    out = 0
    for i, v in enumerate(row):
        out |= (v & ((1 << data_w) - 1)) << (i * data_w)
    return out


if __name__ == "__main__":
    # 2x3 fixture: W = [[1, 2, 3], [4, 5, 6]], a = [7, 8]
    # Expected: [1*7+4*8, 2*7+5*8, 3*7+6*8] = [39, 54, 69]
    sa = SystolicArray(n_rows=2, n_cols=3)
    sa.reset()
    W = [[1, 2, 3], [4, 5, 6]]
    # Wide-row load: one row per cycle, packed
    for k in range(2):
        sa.tick(w_load=1, w_addr=k, w_data=_pack_row(W[k], 8))
    # First activation
    sa.tick(a_valid=1, a_vec=[7, 8])
    assert sa.c_vec_signed == [39, 54, 69], sa.c_vec_signed
    assert sa.c_valid == 1
    # Second activation accumulates
    sa.tick(a_valid=1, a_vec=[1, 1])
    assert sa.c_vec_signed == [39 + 5, 54 + 7, 69 + 9], sa.c_vec_signed
    # Clear
    sa.tick(clear_acc=1)
    assert sa.c_vec_signed == [0, 0, 0]
    assert sa.c_valid == 0
    # Negative operands: W = [[-1]], a = [-5] → 5
    sa2 = SystolicArray(n_rows=1, n_cols=1)
    sa2.reset()
    sa2.tick(w_load=1, w_addr=0, w_data=_pack_row([-1], 8))
    sa2.tick(a_valid=1, a_vec=[-5 & 0xFF])
    assert sa2.c_vec_signed == [5], sa2.c_vec_signed
    # INT8 max: W=127, a=127 → 127*127 = 16129
    sa3 = SystolicArray(n_rows=1, n_cols=1)
    sa3.reset()
    sa3.tick(w_load=1, w_addr=0, w_data=_pack_row([127], 8))
    sa3.tick(a_valid=1, a_vec=[127])
    assert sa3.c_vec_signed == [16129]

    # NEW gap #2: acc_load_valid primes accumulator, next EXECUTE adds on top
    sa4 = SystolicArray(n_rows=2, n_cols=3)
    sa4.reset()
    for k in range(2):
        sa4.tick(w_load=1, w_addr=k, w_data=_pack_row([[1, 2, 3], [4, 5, 6]][k], 8))
    # Prime acc with [100, 200, 300] (packed, each 32-bit)
    packed_primed = (100 & 0xFFFFFFFF) | ((200 & 0xFFFFFFFF) << 32) | \
                    ((300 & 0xFFFFFFFF) << 64)
    sa4.tick(acc_load_valid=1, acc_load_data=packed_primed)
    assert sa4.c_vec_signed == [100, 200, 300], sa4.c_vec_signed
    # Execute with a=[7, 8]: dot = [39, 54, 69], new c_vec = 100+39, 200+54, 300+69
    sa4.tick(a_valid=1, a_vec=[7, 8])
    assert sa4.c_vec_signed == [139, 254, 369], sa4.c_vec_signed
    # clear_acc wins over acc_load_valid
    sa4.tick(clear_acc=1, acc_load_valid=1, acc_load_data=packed_primed)
    assert sa4.c_vec_signed == [0, 0, 0]

    print("systolic_ref self-check: PASS")
