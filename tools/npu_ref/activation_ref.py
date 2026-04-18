"""Python golden reference for rtl/npu_activation/npu_activation.v.

Cycle-accurate mirror: each `tick()` is one clock edge.  Post-tick attributes
`out_valid`, `out_data`, `out_saturated` match what the RTL outputs on the
equivalent cycle.
"""

from __future__ import annotations


MODE_PASS           = 0b000
MODE_RELU           = 0b001
MODE_LEAKY_RELU     = 0b010
MODE_CLIP_INT8      = 0b011
MODE_RELU_CLIP_INT8 = 0b100
MODE_SILU           = 0b101
MODE_GELU           = 0b110
MODE_SIGMOID        = 0b111

INT8_MAX = 127
INT8_MIN = -128

# WP-7: LUT-based activations. Lazy-import since this adds module load-time
# cost and not every consumer needs LUTs.
try:
    from .afu_luts import SILU_LUT, GELU_LUT, SIGMOID_LUT
except ImportError:
    # When imported as a top-level module (not a package)
    from afu_luts import SILU_LUT, GELU_LUT, SIGMOID_LUT


def _sign_extend(val: int, width: int) -> int:
    mask = (1 << width) - 1
    val &= mask
    if val & (1 << (width - 1)):
        return val - (1 << width)
    return val


def _mask(val: int, width: int) -> int:
    return val & ((1 << width) - 1)


def _arith_shift_right_3(val: int, width: int) -> int:
    """Arithmetic right-shift by 3 on a signed `width`-bit value.
    Python's `>>` on negative ints is already arithmetic; we just need to
    interpret the result within the same width."""
    signed = _sign_extend(val, width)
    shifted = signed >> 3
    return _mask(shifted, width)


class Activation:
    def __init__(self, *, acc_w: int = 32, out_w: int = 32):
        self.ACC_W = acc_w
        self.OUT_W = out_w
        self.out_valid = 0
        self.out_data = 0
        self.out_saturated = 0

    def reset(self) -> None:
        self.out_valid = 0
        self.out_data = 0
        self.out_saturated = 0

    def tick(self, *, in_valid: int = 0, in_data: int = 0,
             mode: int = MODE_PASS) -> None:
        """Advance one clock edge.  All inputs sampled pre-edge."""
        # Snapshot pre-edge; compute next-state values
        next_out_valid = int(bool(in_valid))
        next_out_data = self.out_data
        next_out_saturated = 0

        if in_valid:
            x = _sign_extend(_mask(in_data, self.ACC_W), self.ACC_W)
            if mode == MODE_PASS:
                y = x
                sat = 0
            elif mode == MODE_RELU:
                y = 0 if x < 0 else x
                sat = 0
            elif mode == MODE_LEAKY_RELU:
                if x < 0:
                    y = _sign_extend(_arith_shift_right_3(_mask(x, self.ACC_W),
                                                          self.ACC_W),
                                     self.ACC_W)
                else:
                    y = x
                sat = 0
            elif mode == MODE_CLIP_INT8:
                if x > INT8_MAX:
                    y, sat = INT8_MAX, 1
                elif x < INT8_MIN:
                    y, sat = INT8_MIN, 1
                else:
                    y, sat = x, 0
            elif mode == MODE_RELU_CLIP_INT8:
                if x < 0:
                    y, sat = 0, 0
                elif x > INT8_MAX:
                    y, sat = INT8_MAX, 1
                else:
                    y, sat = x, 0
            elif mode in (MODE_SILU, MODE_GELU, MODE_SIGMOID):
                # WP-7 LUT modes: saturate to INT8 then look up.
                lut_in = max(-128, min(127, x))
                if mode == MODE_SILU:
                    y = SILU_LUT[lut_in]
                elif mode == MODE_GELU:
                    y = GELU_LUT[lut_in]
                else:
                    y = SIGMOID_LUT[lut_in]
                sat = 0
            else:
                # Any truly-unused mode: pass through unchanged
                y, sat = x, 0

            next_out_data = _mask(y, self.OUT_W)
            next_out_saturated = sat

        # Commit
        self.out_valid = next_out_valid
        self.out_data = next_out_data
        self.out_saturated = next_out_saturated

    # -------------------------------------------------------------- observers
    @property
    def out_data_signed(self) -> int:
        return _sign_extend(self.out_data, self.OUT_W)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    a = Activation()
    a.reset()
    # PASS: y=x
    a.tick(in_valid=1, in_data=5, mode=MODE_PASS)
    assert a.out_data_signed == 5
    assert a.out_saturated == 0
    # PASS negative
    a.tick(in_valid=1, in_data=-5 & 0xFFFFFFFF, mode=MODE_PASS)
    assert a.out_data_signed == -5
    # RELU positive
    a.tick(in_valid=1, in_data=50, mode=MODE_RELU)
    assert a.out_data_signed == 50
    # RELU negative → zero
    a.tick(in_valid=1, in_data=-50 & 0xFFFFFFFF, mode=MODE_RELU)
    assert a.out_data_signed == 0
    # LEAKY_RELU positive → pass
    a.tick(in_valid=1, in_data=16, mode=MODE_LEAKY_RELU)
    assert a.out_data_signed == 16
    # LEAKY_RELU negative: -64 >> 3 = -8
    a.tick(in_valid=1, in_data=-64 & 0xFFFFFFFF, mode=MODE_LEAKY_RELU)
    assert a.out_data_signed == -8, a.out_data_signed
    # LEAKY_RELU negative odd: -63 >> 3 = -8 (arith shift rounds toward -inf)
    a.tick(in_valid=1, in_data=-63 & 0xFFFFFFFF, mode=MODE_LEAKY_RELU)
    assert a.out_data_signed == -8, a.out_data_signed
    # CLIP_INT8 within range → pass
    a.tick(in_valid=1, in_data=100, mode=MODE_CLIP_INT8)
    assert a.out_data_signed == 100
    assert a.out_saturated == 0
    # CLIP_INT8 above → 127
    a.tick(in_valid=1, in_data=500, mode=MODE_CLIP_INT8)
    assert a.out_data_signed == 127
    assert a.out_saturated == 1
    # CLIP_INT8 below → -128
    a.tick(in_valid=1, in_data=-500 & 0xFFFFFFFF, mode=MODE_CLIP_INT8)
    assert a.out_data_signed == -128
    assert a.out_saturated == 1
    # CLIP_INT8 exact boundary (+127, -128) → not saturated
    a.tick(in_valid=1, in_data=127, mode=MODE_CLIP_INT8)
    assert a.out_data_signed == 127 and a.out_saturated == 0
    a.tick(in_valid=1, in_data=-128 & 0xFFFFFFFF, mode=MODE_CLIP_INT8)
    assert a.out_data_signed == -128 and a.out_saturated == 0
    # RELU_CLIP_INT8 negative → 0, no sat
    a.tick(in_valid=1, in_data=-50 & 0xFFFFFFFF, mode=MODE_RELU_CLIP_INT8)
    assert a.out_data_signed == 0 and a.out_saturated == 0
    # RELU_CLIP_INT8 above 127 → 127, sat
    a.tick(in_valid=1, in_data=500, mode=MODE_RELU_CLIP_INT8)
    assert a.out_data_signed == 127 and a.out_saturated == 1
    # RELU_CLIP_INT8 within → pass
    a.tick(in_valid=1, in_data=50, mode=MODE_RELU_CLIP_INT8)
    assert a.out_data_signed == 50 and a.out_saturated == 0
    # in_valid=0 → out_valid=0 next cycle
    a.tick(in_valid=0)
    assert a.out_valid == 0
    print("activation_ref self-check: PASS")
