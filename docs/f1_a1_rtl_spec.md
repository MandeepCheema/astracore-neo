# F1-A1 RTL Specification — Real FP8 / FP16 Datapath

**Status.** Python bit-accurate references shipped 2026-04-18
(`tools/npu_ref/fp_ref.py`, 12 unit tests). RTL landing in a dedicated
RTL session alongside F1-A2 (BF16 / TF32 / FP32) which reuses this
datapath.

Replaces `npu_top.v:111`'s "FP16 placeholder, falls back to INT8" with
an actual floating-point multiply-add datapath.

---

## Precision formats

| Format | Sign | Exp | Mantissa | Bias | Max | Notes |
|---|---|---|---|---|---|---|
| FP8 E4M3 | 1 | 4 | 3 | 7 | 448.0 | OCP MX-FP8; saturating, no Inf |
| FP8 E5M2 | 1 | 5 | 2 | 15 | 57344.0 | OCP MX-FP8; IEEE-style Inf/NaN |
| FP16 | 1 | 5 | 10 | 15 | 65504.0 | IEEE-754 half |

Binding golden: `tools/npu_ref/fp_ref.py` functions
`fp32_to_e4m3`, `fp32_to_e5m2`, `fp32_to_fp16`, and `fp_mac`. Any RTL
divergence from these functions (outside rounding-mode edge cases) is
a bug.

---

## PE changes (`rtl/npu_pe/npu_pe.v`)

Current PE is 8-bit INT with 32-bit accumulator. F1-A1 extends:

### New parameters

```
parameter [2:0] MODE_INT8      = 3'b000;   // existing
parameter [2:0] MODE_INT4      = 3'b001;   // existing
parameter [2:0] MODE_INT2      = 3'b010;   // existing
parameter [2:0] MODE_FP8_E4M3  = 3'b100;   // F1-A1 new
parameter [2:0] MODE_FP8_E5M2  = 3'b101;   // F1-A1 new
parameter [2:0] MODE_FP16      = 3'b110;   // F1-A1 new
// 3'b111 reserved for F1-A2 (BF16/TF32/FP32 via 2-bit subselect)
```

`cfg_precision_mode` widens from 2 bits to 3 bits. Tile controller
and systolic array wiring updated accordingly.

### Datapath

FP mode datapath:

```
a_fp8, b_fp8 → [FP8 → FP16 promotion] → FP16 mul → FP32 add-accum
```

Where:

- **FP8 → FP16 promotion**: 4-cycle pipeline per OCP MX spec.
  Trivial format conversion (sign passthrough, exponent rebias by
  bias_diff, mantissa left-shift).
- **FP16 multiply**: IEEE-754 half-precision multiplier. 3-cycle
  pipeline. Output is FP32 for maximum accumulator precision.
- **FP32 add-accum**: existing INT32 accumulator path extended to FP32.
  Uses a single-precision FMA unit (one DSP48 + LUT glue on UltraScale+).

Pipeline depth: 5 cycles (vs 3 for INT). The systolic array absorbs
the extra latency via forward-flushing at the edge — no correctness
impact, ~2% area impact per PE.

### Accumulator unification

The INT and FP modes share a 32-bit register but interpret it differently:

- INT mode: INT32, sign-extended from PE partial sum
- FP mode: FP32 IEEE-754

AO SRAM writeback keeps its existing format (RTL produces 32-bit words;
compiler reinterprets per cfg_precision_mode). No DMA change required.

---

## Systolic array changes (`rtl/npu_systolic_array/npu_systolic_array.v`)

Minimal — just widen the `cfg_precision_mode` bus from 2 to 3 bits and
forward to each PE. Array-level logic is format-agnostic.

---

## Compiler lowering (`tools/npu_ref/compiler.py`)

Quantiser already supports INT8/INT4/INT2 (F1-C2 + F1-B2). F1-A1
compiler work:

- Extend `PrecisionCode` enum in compiler instructions with FP8_E4M3
  / FP8_E5M2 / FP16 values.
- `compile_matmul_chained` takes a precision parameter that picks the
  right `cfg_precision_mode`.
- Weight packing: FP8 weights pack 4-to-1 in a 32-bit word (8b each);
  FP16 packs 2-to-1. Same memory layout as INT8/INT4 — no SRAM change.

Will land as a compiler PR alongside the RTL.

---

## Quantiser extension (`tools/npu_ref/quantiser.py`)

F1-B2 raises `NotImplementedError` for FP4/FP8 today. F1-A1 lifts this
by routing weight/activation quant through the new `fp_ref` quantisers:

```python
# in quantise_weights:
elif precision == "fp8_e4m3":
    quant = fp32_to_e4m3(w)
    scale = np.ones(w.shape[0], dtype=np.float32)  # FP is its own scale
```

FP formats don't need per-tensor scale (they're already in log space);
`weight_scale` becomes a flag rather than a multiplier when
`QuantParams.precision` is FP*. Compiler handles the dispatch.

---

## Acceptance gate (cocotb)

- **FP16 matmul**: 64×64 tile, random uniform inputs. RTL output must
  match `fp_mac` golden within max error = 2^-10 (1 LSB of FP16).
- **FP8 E4M3 matmul**: same, with max error = 2^-3 (1 LSB of E4M3
  mantissa × max exponent scaling).
- **FP8 E5M2 matmul**: same, max error = 2^-2.
- **Precision-mode mixing**: a program containing an INT8 tile
  followed by an FP16 tile must produce both bit-exact outputs (tests
  that `cfg_precision_mode` routing is clean).
- **End-to-end YOLOv8-N FP16**: whole-model SNR ≥ 45 dB vs FP32
  reference (looser than INT8's 37.5 dB because FP16 has more headroom
  but its own rounding). This is F1-C5's FP16 extension.

---

## Out-of-scope (this spec)

- FP4 E2M1 — F1-A3's sparsity pass depends on FP4 weight-pack
  density; they're better co-designed. Deferred to a joint F1-A1/A3 RTL
  session after F1-A1 lands.
- BF16 / TF32 / FP32 — F1-A2, reuses this datapath's FMA unit.
- FP8 with fine-grained scale (MX-FP8 block-scaled) — needs a separate
  scale-broadcast block. Consider for F1-A1.2 if needed; baseline spec
  is per-tensor scale only.

---

## Files to add when the RTL session starts

- `rtl/npu_fp/fp8_to_fp16.v` — FP8 → FP16 promoter
- `rtl/npu_fp/fp16_mul.v` — half-precision multiplier
- `rtl/npu_fp/fp32_fma.v` — FP32 fused multiply-add (accumulator)
- `rtl/npu_fp/npu_fp_pe.v` — wraps the above; drops into existing
  `npu_pe` per mode
- `sim/npu_fp_pe/test_npu_fp_pe.py` — cocotb testbench using `fp_ref`
  as oracle
