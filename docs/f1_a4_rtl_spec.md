# F1-A4 RTL Specification — Fused Softmax + LayerNorm (V2 Activations)

**Status.** Python golden references shipped 2026-04-18
(`tools/npu_ref/softmax_ref.py`, `tools/npu_ref/layernorm_ref.py`,
14 unit tests). RTL bodies + cocotb fidelity gate land in a dedicated
RTL session.

This doc is the binding interface + numerical contract the RTL will
implement. Any divergence between this spec and the Python reference
is a bug; the golden is authoritative on behavior, the spec is
authoritative on pipelining.

---

## Module 1 — `rtl/npu_softmax/npu_softmax.v`

### Port list

```
module npu_softmax #(
    parameter VEC_LEN         = 64,     // attention K length / softmax axis size
    parameter IN_W            = 32,     // INT32 accumulator input
    parameter OUT_W           = 8,      // Q0.8 softmax output in [0, 255]
    parameter EXP_LUT_DEPTH   = 256,    // exp LUT (Q8.0 → Q0.32)
    parameter RECIP_LUT_DEPTH = 1024    // 1/s LUT (Q10.0 → Q0.32)
) (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       start,      // latch at pass-1 kick
    input  wire                       in_valid,
    input  wire signed [IN_W-1:0]     in_data,    // streamed x[i]
    output reg                        out_valid,
    output reg        [OUT_W-1:0]     out_data,   // unsigned Q0.8
    output reg                        done
);
```

### Pipeline — two-pass streaming

**Pass 1** (VEC_LEN cycles, `in_valid` high each cycle):
1. Track `m = max(x[0..i])` in parallel with
   `s' = Σ_j exp(x[j] − m)` using the online-softmax update:
      when x[i] > m:  s' = s' * exp(m − x[i]) + 1, m = x[i]
      else:          s' = s' + exp(x[i] − m)
2. `exp(...)` is a 256-entry LUT over the clipped input (−128..127);
   out-of-range inputs clip to the LUT endpoints.
3. At end of pass 1, latch `(m, s')` and compute `inv_s = 1 / s'`
   via the 1024-entry reciprocal LUT.

**Pass 2** (VEC_LEN cycles, `in_valid` high each cycle):
1. For each streamed x[i], output `y[i] = exp(x[i] − m) * inv_s`
   truncated to OUT_W bits.
2. Raise `done` one cycle after the final emission.

### Latency

- Pass 1: VEC_LEN + 3 cycles (accumulator pipeline flush)
- Pass 2: VEC_LEN cycles
- Total: 2*VEC_LEN + 3 cycles per row

### LUT spec

| LUT | Depth | Entry type | Index | Golden |
|---|---|---|---|---|
| exp | 256 | Q0.32 | clip(x − m, −128..127) | `np.exp((i − 128) / scale)` where scale = CSR-programmable |
| recip | 1024 | Q0.32 | top-10 bits of `s'` | `1.0 / ((i + 1) * step)` |

LUTs ship as ROM macros in F1-A4's physical-design pass (coordinated
with TW-4). Python-side, they're pre-computed numpy arrays stored in
`tools/npu_ref/softmax_luts.py` (to be added in the RTL session).

### Validation gate

- **Unit (Python)**: 14 tests already green — see
  `tests/test_softmax_layernorm_ref.py`.
- **Cocotb**: replay `softmax_fp32` on 1000 random VEC_LEN=64 vectors;
  require SNR ≥ 30 dB vs oracle.
- **Integration**: on a real ViT-Base self-attention block, require
  ≤1% top-k change on 128 random input patterns.

---

## Module 2 — `rtl/npu_layernorm/npu_layernorm.v`

### Port list

```
module npu_layernorm #(
    parameter VEC_LEN         = 1024,   // ViT token width is 768/1024
    parameter IN_W            = 32,
    parameter OUT_W           = 32,
    parameter RSQRT_LUT_DEPTH = 256,
    parameter MODE_LN         = 0,      // 0 = LayerNorm, 1 = RMSNorm
    parameter EPS_Q           = 32'h3727C5AC // 1e-5 as FP32
) (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       start,
    input  wire [1:0]                 mode,       // {RMSNorm_en, reserved}
    input  wire                       in_valid,
    input  wire signed [IN_W-1:0]     in_data,    // x[i] stream
    input  wire signed [IN_W-1:0]     in_scale,   // γ[i] stream
    input  wire signed [IN_W-1:0]     in_bias,    // β[i] stream (ignored in RMSNorm)
    output reg                        out_valid,
    output reg  signed [OUT_W-1:0]    out_data,
    output reg                        done
);
```

### Pipeline — two-pass streaming

**Pass 1** (VEC_LEN cycles):
1. Accumulate `Σx` (for LayerNorm) and `Σx²` in wide fixed-point
   (64-bit accumulator).
2. At end of pass 1, compute `μ = Σx / VEC_LEN` (zero for RMSNorm),
   `σ² = Σx²/VEC_LEN − μ²` for LayerNorm, or `Σx²/VEC_LEN` for RMSNorm.
3. Compute `inv_sigma = rsqrt(σ² + ε)` via 256-entry LUT.

**Pass 2** (VEC_LEN cycles, all three streams):
1. For each cycle: `y[i] = (x[i] − μ) * inv_sigma * γ[i] + β[i]` in
   LayerNorm mode; `y[i] = x[i] * inv_sigma * γ[i]` in RMSNorm mode.
2. β[i] is ignored (drive-through not required) in RMSNorm mode.
3. Raise `done` one cycle after final emit.

### Latency

- Pass 1: VEC_LEN + 5 cycles (division + rsqrt)
- Pass 2: VEC_LEN cycles
- Total: 2*VEC_LEN + 5 cycles

### LUT spec

| LUT | Depth | Entry type | Index | Golden |
|---|---|---|---|---|
| rsqrt | 256 | Q0.24 | top-8 bits of `σ² + ε` | `1.0 / sqrt(...)` piecewise-linear |

### Validation gate

- **Unit (Python)**: 7 tests green (see softmax+layernorm test file).
- **Cocotb**: replay `layernorm_fp32` + `rmsnorm_fp32` on 500 random
  VEC_LEN={256, 768, 1024} vectors; SNR ≥ 30 dB vs oracle.
- **Integration**: BERT-Base first-LN output bit-exact (within LUT
  error) vs HuggingFace FP32 reference.

---

## Integration into `rtl/npu_tile_ctrl`

Current tile_ctrl has `cfg_afu_mode` selecting single-cycle AFU modes
(PASS/RELU/SILU/GELU/SIGMOID). F1-A4 adds two new **multi-pass** modes:

| Mode | cfg_afu_mode | Uses |
|---|---|---|
| MODE_SOFTMAX | `4'd8` | `npu_softmax` instance |
| MODE_LAYERNORM | `4'd9` | `npu_layernorm` with mode=LN |
| MODE_RMSNORM | `4'd10` | `npu_layernorm` with mode=RMSNorm |

The tile controller recognises these as two-pass and sequences AO→input
→output accordingly. The existing single-cycle AFU modes keep their
encodings — no regression.

### State-machine addition

```
NEW STATES:
    AFU_PASS1  — stream AO values into the multi-pass AFU
    AFU_PASS2  — stream them back while collecting output
    AFU_DONE   — pulse to tile-level done
```

Done-signal from `npu_softmax` / `npu_layernorm` drives the state
transition from PASS1 → PASS2 (via internal `start` pulse after pass 1
completes) and PASS2 → DONE.

---

## Compiler contract (for F1-C3/C4/C5 once RTL lands)

The compiler will emit a `RunMultiPassAFU` instruction for softmax /
layernorm layers (OP_SOFTMAX / OP_LAYERNORM / OP_RMSNORM) with fields:

```
RunMultiPassAFU {
    mode            : {SOFTMAX, LAYERNORM, RMSNORM}
    vec_len         : int
    input_ao_addr   : AO SRAM address (one row)
    scale_ws_addr   : optional (for LN/RMSNorm γ)
    bias_ws_addr    : optional (for LN β)
    output_ao_addr  : where to write the normalised row
}
```

This is the smallest new instruction the compiler needs. F1-C5's
end-to-end cocotb gate validates the host→RTL sequencing.

---

## Deferred (intentional, to the RTL session)

- LUT ROM layout + actual bit patterns (needs physical-design pass).
- Floating-point variants (all F1-A4 math is fixed-point; FP
  layernorm arrives with F1-A1's FP datapath).
- Resource estimates (LUT/DSP/BRAM count) — comes with Vivado out-of-
  context synth run in F1-A4's follow-up.
- AXI-Lite CSR map for the LUT reload path (for debug / post-fab
  tuning). Sketch below; full map lands with the RTL.

### CSR sketch

| Offset | Name | Purpose |
|---|---|---|
| 0x00 | NPU_SM_CTRL | start, clear, soft-reset |
| 0x04 | NPU_SM_STATUS | done, error flags |
| 0x08 | NPU_SM_CONFIG | mode, scale, vec_len |
| 0x40 | NPU_SM_LUT_EXP_DATA | streamed LUT reload |
| 0x44 | NPU_SM_LUT_EXP_ADDR | LUT write address |
| 0x50 | NPU_SM_LUT_RECIP_DATA | |
| 0x54 | NPU_SM_LUT_RECIP_ADDR | |
| 0x60 | NPU_LN_LUT_RSQRT_DATA | |
| 0x64 | NPU_LN_LUT_RSQRT_ADDR | |
