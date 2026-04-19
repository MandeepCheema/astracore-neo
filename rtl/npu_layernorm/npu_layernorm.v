`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — NPU LayerNorm / RMSNorm  (npu_layernorm.v)  [F1-A4]
// =============================================================================
// Two-pass fused LayerNorm (and RMSNorm via mode bit) for transformer
// features.  Matches tools/npu_ref/layernorm_luts.py bit-for-bit.
//
//   Pass 1 (VEC_LEN cycles):
//     Accumulate Σx (48 bit signed) and Σx² (64 bit unsigned) on each
//     `in_valid` cycle.  `in_scale` and `in_bias` are ignored during
//     pass 1.
//
//   Reciprocal-sqrt sequence (5 fixed cycles):
//     μ        = Σx >>> LOG2_VEC_LEN
//     σ²       = (Σx² >> LOG2_VEC_LEN) − μ²         (Q32.32 unsigned)
//     var_eps  = σ² + ε                              (ε = EPS_Q16_16 << 16)
//     LZC      = count leading zeros of var_eps (64-bit priority encoder)
//     v_norm   = var_eps << LZC                      (MSB at bit 63)
//     idx      = v_norm[63:56] − 128                 (range [0, 127])
//     lut_val  = rsqrt_lut[idx]                      (Q1.15)
//     shift_raw = 32 − LZC                           (signed)
//     if shift_raw is even : inv_sigma = lut_val << (16 − shift_raw/2)
//     else                : inv_sigma = (lut_val*SQRT_HALF_Q15 >> 15)
//                                            << (16 − (shift_raw−1)/2)
//     inv_sigma is saturated to Q1.31 unsigned [0, 2^32).
//
//   Pass 2 (VEC_LEN cycles):
//     y = saturate( ((centered * inv_sigma) >> 31) * γ >> 16 + β , INT32 )
//     where centered = (x − μ) for LN and centered = x for RMSNorm,
//     and β is ignored in RMSNorm mode.
//
// ── Numeric contract ────────────────────────────────────────────────────────
//   All inputs are signed Q(32−IN_FRAC_BITS).IN_FRAC_BITS (default Q16.16).
//   VEC_LEN must be a power of 2 and LOG2_VEC_LEN == log2(VEC_LEN).
//   ε = EPS_Q16_16 × 2^−16 ≈ 1e-4 (default 7).
//
// ── Mode bit ────────────────────────────────────────────────────────────────
//   mode[0] = 0 : LayerNorm (use μ, β).
//   mode[0] = 1 : RMSNorm   (skip mean subtraction; ignore β).
//   mode[1]     : reserved for F1-A4 follow-ups.
// =============================================================================

module npu_layernorm #(
    parameter integer VEC_LEN         = 256,
    parameter integer LOG2_VEC_LEN    = 8,
    parameter integer IN_W            = 32,
    parameter integer IN_FRAC_BITS    = 16,
    parameter integer SUM_W           = 48,
    parameter integer SQ_SUM_W        = 64,
    parameter integer OUT_W           = 32,
    parameter integer EPS_Q16_16      = 7,
    parameter integer RSQRT_LUT_DEPTH = 256,
    parameter         RSQRT_LUT_FILE  = "rsqrt_lut.mem",
    parameter integer COUNT_W         = 12
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          start,
    input  wire [1:0]                    mode,
    input  wire                          in_valid,
    input  wire signed [IN_W-1:0]        in_data,
    input  wire signed [IN_W-1:0]        in_scale,
    input  wire signed [IN_W-1:0]        in_bias,
    output reg                           out_valid,
    output reg  signed [OUT_W-1:0]       out_data,
    output reg                           done
);

    // ========================================================================
    // LUT
    // ========================================================================
    localparam integer RSQRT_Q_FRAC = 15;
    localparam [15:0]  SQRT_HALF_Q15 = 16'h5A82;  // sqrt(0.5) in Q1.15

    reg [15:0] rsqrt_lut [0:RSQRT_LUT_DEPTH-1];
    initial $readmemh(RSQRT_LUT_FILE, rsqrt_lut);

    // ========================================================================
    // State
    // ========================================================================
    localparam [3:0] S_IDLE    = 4'd0;
    localparam [3:0] S_PASS1   = 4'd1;
    localparam [3:0] S_RS0     = 4'd2;  // compute μ, var, var_eps
    localparam [3:0] S_RS1     = 4'd3;  // compute LZC and v_norm
    localparam [3:0] S_RS2     = 4'd4;  // LUT read
    localparam [3:0] S_RS3     = 4'd5;  // apply even/odd shift ⇒ inv_sigma
    localparam [3:0] S_PASS2   = 4'd6;
    localparam [3:0] S_DONE    = 4'd7;

    reg [3:0]                 state;
    reg [1:0]                 mode_r;
    reg signed [SUM_W-1:0]    sum_x;
    reg        [SQ_SUM_W-1:0] sum_x2;
    reg signed [IN_W-1:0]     mu;
    reg        [SQ_SUM_W-1:0] var_eps;
    reg [6:0]                 lzc;
    reg        [SQ_SUM_W-1:0] v_norm;
    reg [7:0]                 rs_idx;
    reg [15:0]                lut_val;
    reg signed [31:0]         inv_sigma;      // Q1.31 unsigned value in a signed wrapper
    reg [COUNT_W-1:0]         count;

    // ========================================================================
    // Pass-1 accumulation combinational helpers
    // ========================================================================
    wire signed [2*IN_W-1:0] x2 = in_data * in_data;                   // Q32.32
    wire signed [SUM_W-1:0]  sum_x_next  = sum_x + in_data;
    wire        [SQ_SUM_W-1:0] sum_x2_next = sum_x2 + x2[SQ_SUM_W-1:0];  // x*x ≥ 0 for any signed x

    // ========================================================================
    // RS0: compute μ, σ², var_eps
    // ========================================================================
    wire signed [IN_W-1:0]    mu_comb        = sum_x >>> LOG2_VEC_LEN;
    wire        [SQ_SUM_W-1:0] mean_x2_comb  = sum_x2 >> LOG2_VEC_LEN;
    wire signed [2*IN_W-1:0]  mu2_comb       = mu_comb * mu_comb;        // Q32.32
    wire        [SQ_SUM_W-1:0] mu2_u         = mu2_comb[SQ_SUM_W-1:0];    // mu² ≥ 0
    wire        [SQ_SUM_W-1:0] var_comb      = (mode_r[0]) ? mean_x2_comb :
                                                (mean_x2_comb > mu2_u ? mean_x2_comb - mu2_u : {SQ_SUM_W{1'b0}});
    wire        [SQ_SUM_W-1:0] eps_q         = { {(SQ_SUM_W-IN_FRAC_BITS-8){1'b0}}, EPS_Q16_16[7:0], {IN_FRAC_BITS{1'b0}} };
    wire        [SQ_SUM_W-1:0] var_eps_comb  = var_comb + eps_q;

    // ========================================================================
    // RS1: leading-zero count (64-bit priority encoder, behavioural)
    // ========================================================================
    function automatic [6:0] lzc64;
        input [SQ_SUM_W-1:0] v;
        integer i;
        reg found;
        begin
            lzc64 = 7'd64;
            found = 1'b0;
            for (i = SQ_SUM_W - 1; i >= 0; i = i - 1) begin
                if (!found && v[i]) begin
                    lzc64 = SQ_SUM_W - 1 - i;
                    found = 1'b1;
                end
            end
        end
    endfunction

    // ========================================================================
    // RS3: apply shift to LUT value
    //   shift_raw = 32 − lzc (can be negative)
    //   even: final = lut_val << (16 − shift_raw/2)  | or shift right if negative
    //   odd : final = (lut_val * SQRT_HALF_Q15 >> 15) << (16 − (shift_raw−1)/2)
    // We pre-compute the odd-branch scaled value.  We track shift_amt as
    // a signed integer that is the left-shift amount to apply to a Q1.15
    // value to land in Q1.31 (positive means left-shift, negative right).
    // ========================================================================
    // signed shift amount expressed relative to a starting Q1.15 value;
    // 16 bits more to get to Q1.31, minus half_shift (which moves the sqrt exp).
    wire signed [7:0] shift_raw = $signed({1'b0, 7'd32}) - $signed({1'b0, lzc});
    wire              shift_odd = shift_raw[0];
    wire signed [7:0] half_shift = shift_odd ? (shift_raw - 8'sd1) >>> 1 : (shift_raw >>> 1);
    wire signed [7:0] total_left = 8'sd16 - half_shift;

    // Explicitly widen the product to 32 bits before shifting — otherwise
    // Verilog does the multiply in the operand width (16 bits) and the upper
    // bits of the 32-bit product are lost.
    wire [31:0] lut_scaled_wide = {16'b0, lut_val} * {16'b0, SQRT_HALF_Q15};
    wire [15:0] lut_scaled_q15  = lut_scaled_wide[30:15];
    wire [15:0] rs_base         = shift_odd ? lut_scaled_q15 : lut_val;

    // Apply variable shift; cap to 32-bit result with saturation.
    reg [63:0] shifted_full;
    always @* begin
        shifted_full = { {48{1'b0}}, rs_base };
        if (total_left >= 0) begin
            shifted_full = shifted_full << total_left[5:0];
        end else begin
            shifted_full = shifted_full >> ((-total_left[5:0]) & 6'h3F);
        end
    end
    wire [31:0] inv_sigma_comb =
          (shifted_full > 64'hFFFFFFFF) ? 32'hFFFFFFFF
        :                                 shifted_full[31:0];

    // ========================================================================
    // Pass-2 output compute
    // ========================================================================
    wire signed [IN_W-1:0]   centered       = mode_r[0] ? in_data : (in_data - mu);
    wire signed [IN_W + 32 - 1:0] centered_mul_inv = centered * $signed({1'b0, inv_sigma});  // Q16.16 * Q1.31 = Q17.47
    wire signed [IN_W-1:0]   norm            = centered_mul_inv >>> 31;                  // back to Q16.16
    wire signed [2*IN_W-1:0] scaled_prod     = norm * in_scale;                           // Q32.32
    wire signed [IN_W + 16 - 1:0] scaled_q16 = scaled_prod >>> IN_FRAC_BITS;              // Q16.16 in wider width

    wire signed [IN_W-1:0]   y_pre           = mode_r[0] ? scaled_q16[IN_W-1:0]
                                                         : (scaled_q16[IN_W-1:0] + in_bias);
    // Saturate to INT32 (the width of scaled_q16 extends into the upper bits;
    // when those represent a number out of INT32 range, clamp).
    wire                     y_overflow_pos  = !scaled_q16[IN_W+16-1] & (|scaled_q16[IN_W+16-2:IN_W-1]);
    wire                     y_overflow_neg  = scaled_q16[IN_W+16-1] & !(&scaled_q16[IN_W+16-2:IN_W-1]);
    wire signed [IN_W-1:0]   y_sat_pre       = y_overflow_pos ? {1'b0, {(IN_W-1){1'b1}}}
                                             : y_overflow_neg ? {1'b1, {(IN_W-1){1'b0}}}
                                             :                  scaled_q16[IN_W-1:0];
    wire signed [IN_W-1:0]   y_final         = mode_r[0] ? y_sat_pre : (y_sat_pre + in_bias);

    // ========================================================================
    // FSM
    // ========================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            mode_r    <= 2'b00;
            sum_x     <= {SUM_W{1'b0}};
            sum_x2    <= {SQ_SUM_W{1'b0}};
            mu        <= {IN_W{1'b0}};
            var_eps   <= {SQ_SUM_W{1'b0}};
            lzc       <= 7'd0;
            v_norm    <= {SQ_SUM_W{1'b0}};
            rs_idx    <= 8'd0;
            lut_val   <= 16'd0;
            inv_sigma <= 32'd0;
            count     <= {COUNT_W{1'b0}};
            out_valid <= 1'b0;
            out_data  <= {OUT_W{1'b0}};
            done      <= 1'b0;
        end else begin
            out_valid <= 1'b0;
            done      <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (start) begin
                        state  <= S_PASS1;
                        count  <= {COUNT_W{1'b0}};
                        sum_x  <= {SUM_W{1'b0}};
                        sum_x2 <= {SQ_SUM_W{1'b0}};
                        mode_r <= mode;
                    end
                end

                S_PASS1: begin
                    if (in_valid) begin
                        sum_x  <= sum_x_next;
                        sum_x2 <= sum_x2_next;
                        count  <= count + 1'b1;
                        if (count == VEC_LEN[COUNT_W-1:0] - 1) begin
                            state <= S_RS0;
                        end
                    end
                end

                S_RS0: begin
                    mu      <= mu_comb;
                    var_eps <= var_eps_comb;
                    state   <= S_RS1;
                end

                S_RS1: begin
                    lzc    <= lzc64(var_eps);
                    v_norm <= var_eps << lzc64(var_eps);
                    state  <= S_RS2;
                end

                S_RS2: begin
                    rs_idx  <= v_norm[SQ_SUM_W-1:SQ_SUM_W-8];  // top 8 bits
                    // idx is in [128, 255] → array idx = top_8_bits - 128
                    // (guaranteed by normalisation: MSB = bit 63 = 1).
                    lut_val <= rsqrt_lut[v_norm[SQ_SUM_W-1:SQ_SUM_W-8] - 8'd128];
                    state   <= S_RS3;
                end

                S_RS3: begin
                    inv_sigma <= inv_sigma_comb;
                    state     <= S_PASS2;
                    count     <= {COUNT_W{1'b0}};
                end

                S_PASS2: begin
                    if (in_valid) begin
                        out_valid <= 1'b1;
                        out_data  <= y_final;
                        count     <= count + 1'b1;
                        if (count == VEC_LEN[COUNT_W-1:0] - 1) begin
                            state <= S_DONE;
                        end
                    end
                end

                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
