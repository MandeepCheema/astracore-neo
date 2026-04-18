"""Python golden reference for rtl/npu_pe/npu_pe.v.

Cycle-accurate mirror of the RTL PE.  Every tick() call corresponds to one
rising clock edge in the RTL; outputs after tick() match the RTL state
visible after that clock edge's non-blocking assignments have applied.

Usage:
    pe = PE(data_w=8, acc_w=32)
    pe.reset()
    pe.tick(load_w=1, weight_in=3)     # latch weight
    pe.tick(a_valid=1, a_in=5)         # accumulate 3*5 = 15
    pe.tick(a_valid=1, a_in=2)         # accumulate 3*2 = 6, psum = 21
    assert pe.psum_out == 21

The RTL declares all event pulses as 1-cycle NBAs; tick() mirrors that
semantics by computing next-state values from the pre-edge inputs and
then committing them at the end of the call.
"""

from __future__ import annotations


def _sign_extend(val: int, width: int) -> int:
    """Interpret an unsigned bit-vector as signed two's complement."""
    mask = (1 << width) - 1
    val &= mask
    if val & (1 << (width - 1)):
        return val - (1 << width)
    return val


def _mask(val: int, width: int) -> int:
    """Truncate to an unsigned bit-vector of the given width."""
    return val & ((1 << width) - 1)


class PE:
    """Single NPU processing element — weight-stationary MAC.

    Mirrors rtl/npu_pe/npu_pe.v exactly.  V1 implements INT8 only; higher
    precision modes carry through the interface but fall back to INT8.
    """

    def __init__(self, data_w: int = 8, acc_w: int = 32):
        self.DATA_W = data_w
        self.ACC_W = acc_w
        # Stored state (one-to-one with RTL regs)
        self.weight_reg: int = 0          # signed, DATA_W bits
        self.acc: int = 0                 # signed, ACC_W bits
        self.a_out: int = 0               # signed, DATA_W bits
        self.a_valid_out: int = 0
        self.sparse_skip_out: int = 0

    # ------------------------------------------------------------------ reset
    def reset(self) -> None:
        """Mirror the RTL active-low reset: zeroes every register."""
        self.weight_reg = 0
        self.acc = 0
        self.a_out = 0
        self.a_valid_out = 0
        self.sparse_skip_out = 0

    # ------------------------------------------------------------------- tick
    def tick(
        self,
        *,
        load_w: int = 0,
        clear_acc: int = 0,
        weight_in: int = 0,
        a_valid: int = 0,
        a_in: int = 0,
        sparse_skip: int = 0,
        precision_mode: int = 0,   # reserved; v1 always computes INT8
        sparse_en: int = 0,        # reserved; advisory flag
    ) -> None:
        """Advance one clock edge.

        All inputs are sampled BEFORE the edge (like RTL on posedge clk),
        all outputs reflect the values that will be visible AFTER the edge.
        """
        # Snapshot current weight *before* any same-cycle load_w takes effect,
        # matching the RTL where the multiply uses the pre-edge weight_reg.
        w_full = _sign_extend(self.weight_reg, self.DATA_W)
        a_full = _sign_extend(_mask(a_in, self.DATA_W), self.DATA_W)
        w_packed = _mask(self.weight_reg, self.DATA_W)
        a_packed = _mask(a_in, self.DATA_W)

        if precision_mode == 0b01:
            # INT4: 2 packed INT4×INT4 products summed
            w_hi = _sign_extend((w_packed >> 4) & 0xF, 4)
            w_lo = _sign_extend(w_packed & 0xF, 4)
            a_hi = _sign_extend((a_packed >> 4) & 0xF, 4)
            a_lo = _sign_extend(a_packed & 0xF, 4)
            product = w_hi * a_hi + w_lo * a_lo
        elif precision_mode == 0b10:
            # INT2: 4 packed INT2×INT2 products summed
            def _i2(v, shift):
                return _sign_extend((v >> shift) & 0x3, 2)
            product = 0
            for sh in (0, 2, 4, 6):
                product += _i2(w_packed, sh) * _i2(a_packed, sh)
        else:
            # INT8 (and FP16 placeholder): 1 full INT8×INT8 product
            product = w_full * a_full
        do_acc = bool(a_valid) and not bool(sparse_skip)

        # Compute next accumulator value.  clear_acc wins over accumulate.
        if clear_acc:
            next_acc = 0
        elif do_acc:
            acc_signed = _sign_extend(self.acc, self.ACC_W)
            next_acc = _mask(acc_signed + product, self.ACC_W)
        else:
            next_acc = self.acc

        # Commit all state atomically (mirrors RTL NBA phase).
        if load_w:
            self.weight_reg = _mask(weight_in, self.DATA_W)
        self.acc = next_acc
        self.a_out = _mask(a_in, self.DATA_W)
        self.a_valid_out = int(bool(a_valid))
        self.sparse_skip_out = int(bool(sparse_skip))

    # ----------------------------------------------------------- observable
    @property
    def psum_out(self) -> int:
        """Combinational view of accumulator as a signed integer."""
        return _sign_extend(self.acc, self.ACC_W)

    @property
    def a_out_signed(self) -> int:
        return _sign_extend(self.a_out, self.DATA_W)

    @property
    def weight_signed(self) -> int:
        return _sign_extend(self.weight_reg, self.DATA_W)


# ----------------------------------------------------------------------------
# Tiny self-check: runs as a script to prove the reference is internally
# consistent before it is used to validate any RTL.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    pe = PE()
    pe.reset()
    # Load weight = 3
    pe.tick(load_w=1, weight_in=3)
    assert pe.weight_signed == 3
    assert pe.psum_out == 0
    # Accumulate 3*5 = 15
    pe.tick(a_valid=1, a_in=5)
    assert pe.psum_out == 15, pe.psum_out
    # Accumulate 3*2 = 6  → psum = 21
    pe.tick(a_valid=1, a_in=2)
    assert pe.psum_out == 21, pe.psum_out
    # Invalid cycle: psum unchanged
    pe.tick(a_valid=0, a_in=99)
    assert pe.psum_out == 21
    # Sparse-skip: psum unchanged but activation still passes through
    pe.tick(a_valid=1, a_in=7, sparse_skip=1)
    assert pe.psum_out == 21
    assert pe.a_out_signed == 7
    # Clear accumulator
    pe.tick(clear_acc=1)
    assert pe.psum_out == 0
    # Negative operands: -3 * -7 = 21
    pe.tick(load_w=1, weight_in=-3)
    pe.tick(a_valid=1, a_in=-7)
    assert pe.psum_out == 21, pe.psum_out
    # Negative-positive: -3 * 10 = -30
    pe.tick(a_valid=1, a_in=10)
    assert pe.psum_out == -9, pe.psum_out   # 21 + (-30)
    print("pe_ref self-check: PASS")
