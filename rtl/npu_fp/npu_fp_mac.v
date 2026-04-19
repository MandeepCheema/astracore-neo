`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — NPU FP MAC  (npu_fp_mac.v)  [F1-A1]
// =============================================================================
// Fused FP8/FP16 multiply → FP32 accumulate.  Matches
// `tools/npu_ref/fp_ref.py`'s `fp_mac` oracle within max-error bounds
// specified in `docs/f1_a1_rtl_spec.md`.
//
//   FP16     : max err 2^-10  (1 LSB of FP16)
//   FP8 E4M3 : max err 2^-3
//   FP8 E5M2 : max err 2^-2
//
// ── Scope of this module ───────────────────────────────────────────────────
// This is the **simulation-gate deliverable** for F1-A1.  The datapath uses
// Verilog's `real` type for the FP math inside the always_ff block:
//
//     a_real, b_real ← decoded from raw bit-pattern per precision mode
//     prod_real      = a_real * b_real
//     acc_real       = clear_acc ? 0 : acc_real + prod_real
//     acc_bits       = $realtobits(acc_real)[31:0]   // FP32 output
//
// `real` is simulator-native and not synthesizable.  A fully-synthesizable
// replacement using integer mantissa/exponent arithmetic lands in F1-A1.1.
// The interface, precision encoding, and MAC semantics are identical; the
// follow-up just swaps the `real`-based datapath.
//
// The cocotb fidelity gate in `sim/npu_fp_mac/` validates this sim-model
// bit-exactly (within rounding) against `fp_mac` so the interface contract
// is frozen now.  That lets compiler + PE integration work land in parallel.
//
// ── Precision encoding ─────────────────────────────────────────────────────
//   precision_mode[2:0]  (widened from 2 bits in the spec)
//     3'b100 — FP8 E4M3    (OCP; 1/4/3 bits; bias=7; saturating)
//     3'b101 — FP8 E5M2    (OCP; 1/5/2 bits; bias=15; IEEE Inf/NaN)
//     3'b110 — FP16        (IEEE-754 half; 1/5/10; bias=15)
//     other  — reserved (passes zero to accumulator)
//
//   FP8 modes consume a[7:0] / b[7:0]; upper bits ignored.
//   FP16 consumes a[15:0] / b[15:0].
//
// ── Interface ───────────────────────────────────────────────────────────────
//   clear_acc (1-cycle) : zero the accumulator at next edge.
//   en        (1-cycle) : accumulate a*b into acc at next edge.  clear_acc
//                         takes precedence if both are asserted.
//   acc_out             : combinational FP32 view of current accumulator.
// =============================================================================

module npu_fp_mac (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         en,
    input  wire         clear_acc,
    input  wire [2:0]   precision_mode,
    input  wire [15:0]  a,
    input  wire [15:0]  b,
    // 64-bit $realtobits of the FP64 accumulator. The testbench unpacks
    // this as an IEEE double. A synthesis-ready replacement in F1-A1.1
    // will expose the native 32-bit FP32 accumulator instead.
    output wire [63:0]  acc_out
);

    localparam [2:0] MODE_FP8_E4M3 = 3'b100;
    localparam [2:0] MODE_FP8_E5M2 = 3'b101;
    localparam [2:0] MODE_FP16     = 3'b110;

    // =========================================================================
    // Decoder functions  (simulation-time real, matches fp_ref.py exactly)
    // =========================================================================
    // All three formats saturate (FP8 E4M3) or follow IEEE (FP8 E5M2, FP16).

    function automatic real decode_fp16;
        input [15:0] bits;
        reg         s;
        reg [4:0]   e;
        reg [9:0]   m;
        real        sign_mul, frac, val;
        begin
            s = bits[15];
            e = bits[14:10];
            m = bits[9:0];
            sign_mul = s ? -1.0 : 1.0;
            if (e == 5'd31) begin
                // Inf or NaN
                if (m == 10'd0) decode_fp16 = sign_mul * 1.0e30;  // treat as large
                else             decode_fp16 = 0.0;               // NaN → 0 (propagates harmlessly)
            end else if (e == 5'd0) begin
                // sub-normal or zero
                frac = $itor(m) / 1024.0;
                decode_fp16 = sign_mul * frac * (2.0 ** -14);
            end else begin
                frac = 1.0 + $itor(m) / 1024.0;
                decode_fp16 = sign_mul * frac * (2.0 ** ($signed({1'b0, e}) - 15));
            end
        end
    endfunction

    function automatic real decode_e4m3;
        input [7:0] bits;
        reg         s;
        reg [3:0]   e;
        reg [2:0]   m;
        real        sign_mul, frac;
        begin
            s = bits[7];
            e = bits[6:3];
            m = bits[2:0];
            sign_mul = s ? -1.0 : 1.0;
            if (e == 4'd15 && m == 3'd7) begin
                decode_e4m3 = 0.0;            // NaN slot; propagate as 0 (OCP saturates; NaN rare)
            end else if (e == 4'd0) begin
                frac = $itor(m) / 8.0;
                decode_e4m3 = sign_mul * frac * (2.0 ** -6);   // subnormal
            end else begin
                frac = 1.0 + $itor(m) / 8.0;
                decode_e4m3 = sign_mul * frac * (2.0 ** ($signed({1'b0, e}) - 7));
            end
        end
    endfunction

    function automatic real decode_e5m2;
        input [7:0] bits;
        reg         s;
        reg [4:0]   e;
        reg [1:0]   m;
        real        sign_mul, frac;
        begin
            s = bits[7];
            e = bits[6:2];
            m = bits[1:0];
            sign_mul = s ? -1.0 : 1.0;
            if (e == 5'd31) begin
                if (m == 2'd0) decode_e5m2 = sign_mul * 1.0e30;  // Inf → very large
                else           decode_e5m2 = 0.0;                 // NaN
            end else if (e == 5'd0) begin
                frac = $itor(m) / 4.0;
                decode_e5m2 = sign_mul * frac * (2.0 ** -14);
            end else begin
                frac = 1.0 + $itor(m) / 4.0;
                decode_e5m2 = sign_mul * frac * (2.0 ** ($signed({1'b0, e}) - 15));
            end
        end
    endfunction

    // =========================================================================
    // Multiply + accumulate (all real math)
    // =========================================================================
    real a_real, b_real, prod_real, acc_real;

    always @(posedge clk) begin
        if (!rst_n) begin
            acc_real <= 0.0;
        end else begin
            // decode based on mode
            case (precision_mode)
                MODE_FP16:     begin a_real = decode_fp16(a);           b_real = decode_fp16(b);          end
                MODE_FP8_E4M3: begin a_real = decode_e4m3(a[7:0]);      b_real = decode_e4m3(b[7:0]);     end
                MODE_FP8_E5M2: begin a_real = decode_e5m2(a[7:0]);      b_real = decode_e5m2(b[7:0]);     end
                default:       begin a_real = 0.0;                       b_real = 0.0;                     end
            endcase
            prod_real = a_real * b_real;
            if (clear_acc)      acc_real <= 0.0;
            else if (en)        acc_real <= acc_real + prod_real;
        end
    end

    // =========================================================================
    // Output: 64-bit IEEE double bit-pattern of acc_real. Verilator supports
    // $realtobits; the cocotb test unpacks as `struct.unpack('<d', ...)` to
    // recover the FP value. (Verilator 5.030 does NOT support shortreal
    // conversion, so we publish the double and let the test quantise to
    // FP32 when checking spec bounds.)
    // =========================================================================
    assign acc_out = $realtobits(acc_real);

endmodule
