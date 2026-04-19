`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — NPU Softmax  (npu_softmax.v)  [F1-A4]
// =============================================================================
// Two-pass streaming softmax for transformer attention-head K vectors.
//
//   Pass 1 (VEC_LEN cycles, in_valid each cycle):
//     Maintains running max m and running Q8.32 sum s = Σ exp(x_i - m)
//     via the online-softmax update:
//         x_i > m :   s <- s * exp(m - x_i) + 1 ;   m <- x_i
//         else    :   s <- s + exp(x_i - m)
//     All exp args are non-positive so the 256-entry exp LUT indexes a
//     non-negative "shift" in units of 1/EXP_SCALE.
//
//   Reciprocal (1 cycle):
//     Compute inv_s = recip_lut[ round(s >> RECIP_SHIFT) ] in Q0.32.
//     Rounding uses midpoint (s + half_bucket) >> RECIP_SHIFT so that the
//     Q0.32 reciprocal value for bucket k (= 1/((k+0.5)*step)) is chosen
//     for s values closest to the bucket's midpoint, not its lower edge.
//
//   Pass 2 (VEC_LEN cycles, in_valid re-streamed):
//     For each x_i (identical sequence as pass 1) emit
//         y_i = round( exp(x_i - m) * inv_s , Q0.8 )
//     via exp_lut lookup + 32×32 multiply + round-to-nearest truncation.
//
// ── Numeric contract (matches tools/npu_ref/softmax_luts.py) ───────────────
//   EXP_SCALE   = 16  ⇒ LUT step = 1/16 in real-number space.
//   Input       = signed INT32 interpreted as Q(32-F).F with F=log2(EXP_SCALE)
//                 =4 so Q28.4 — but the module only consumes the shift
//                 amount (m - x) which is an integer in LUT-grid units, so
//                 the actual Q-format is caller's convention.
//   exp_lut[i]  = round(exp(-i / EXP_SCALE) * 2^32)   Q0.32 unsigned
//   recip_lut[k] = round(1 / ((k+0.5) / 16) * 2^32)  Q0.32 unsigned
//   Sum s       = Q8.32 over 40 bits (max sum 64 * 1.0 < 2^7).
//   Output      = Q0.8 unsigned, in [0, 255].
//
// ── Interface contract ─────────────────────────────────────────────────────
//   start : 1-cycle pulse before the first pass-1 element.  Resets internal
//           counters + clears m/s.  Caller must then drive in_valid for
//           exactly VEC_LEN cycles (pass 1), wait until `done` drops from
//           the previous run and the pass1→recip transition is complete
//           (deterministic 2 cycles after the VEC_LEN'th in_valid), then
//           stream the same VEC_LEN values again for pass 2.
//   out_valid + out_data : one-cycle valid pulses during pass 2.
//   done  : 1-cycle pulse one cycle after the last pass-2 emit.
//
// Simpler caller pattern (used by the cocotb testbench): wait for `done`
// before re-using the module; between passes, wait 2 deterministic cycles.
// =============================================================================

module npu_softmax #(
    parameter integer VEC_LEN          = 64,
    parameter integer IN_W             = 32,
    parameter integer OUT_W            = 8,
    parameter integer EXP_LUT_DEPTH    = 256,
    parameter integer RECIP_LUT_DEPTH  = 1024,
    parameter integer SUM_W            = 40,   // Q8.32 internal sum
    parameter integer EXP_W            = 32,   // Q0.32 exp LUT value width
    parameter integer SUM_FRAC_BITS    = 32,
    parameter integer RECIP_SHIFT      = 28,
    parameter         EXP_LUT_FILE     = "exp_lut.mem",
    parameter         RECIP_LUT_FILE   = "recip_lut.mem",
    parameter integer COUNT_W          = 8     // > log2(VEC_LEN+1)
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          start,
    input  wire                          in_valid,
    input  wire signed [IN_W-1:0]        in_data,
    output reg                           out_valid,
    output reg  [OUT_W-1:0]              out_data,
    output reg                           done
);

    // ========================================================================
    // LUTs (initialised at time 0 via $readmemh; the .mem files are written
    // to sim_build/ by the cocotb runner before build).
    // ========================================================================
    reg [EXP_W-1:0] exp_lut   [0:EXP_LUT_DEPTH-1];
    reg [EXP_W-1:0] recip_lut [0:RECIP_LUT_DEPTH-1];

    initial begin : lut_init
        $readmemh(EXP_LUT_FILE,   exp_lut);
        $readmemh(RECIP_LUT_FILE, recip_lut);
    end

    // ========================================================================
    // State
    // ========================================================================
    localparam [2:0] S_IDLE   = 3'd0;
    localparam [2:0] S_PASS1  = 3'd1;
    localparam [2:0] S_RECIP  = 3'd2;
    localparam [2:0] S_PASS2  = 3'd3;
    localparam [2:0] S_DONE   = 3'd4;

    reg [2:0]             state;
    reg signed [IN_W-1:0] m;                       // running max
    reg [SUM_W-1:0]       s;                       // running Q8.32 sum
    reg [EXP_W-1:0]       inv_s;                   // Q0.32 reciprocal
    reg [COUNT_W-1:0]     count;                   // element counter (per pass)
    reg                   pass1_first;             // 1 on first pass-1 element

    // ========================================================================
    // Shared shift-index computation (used by pass 1 and pass 2).
    // ========================================================================
    // diff_pos = max(0, m - in_data) clipped to LUT index width.
    // (In pass 1's "x > m" branch we instead use (in_data - m) as shift; it's
    // computed separately below — see the pass 1 update block.)
    wire signed [IN_W:0]  m_minus_x        = $signed({m[IN_W-1], m}) -
                                             $signed({in_data[IN_W-1], in_data});
    wire                  m_minus_x_neg    = m_minus_x[IN_W];           // (m-x) < 0 ⇒ x>m
    wire [IN_W:0]         m_minus_x_abs    = m_minus_x_neg ? (-m_minus_x) : m_minus_x;
    wire                  m_minus_x_sat    = |m_minus_x_abs[IN_W:8];    // bit 8..top nonzero ⇒ saturate
    wire [7:0]            m_minus_x_idx    = m_minus_x_sat ? 8'hFF : m_minus_x_abs[7:0];

    // exp lookup for pass-1-non-newmax / pass-2: shift = m - x (always ≥ 0)
    wire [EXP_W-1:0]      exp_val_rd       = exp_lut[m_minus_x_idx];

    // ========================================================================
    // Pass-1 online-update combinational block
    // ========================================================================
    // Case A: x > m  (m_minus_x_neg=1)
    //   shift_a = clip(x - m, 0, 255) = same index, lookup = exp_shift
    //   s_next  = (s * exp_shift) >> SUM_FRAC + exp_lut[0]
    //   m_next  = in_data
    // Case B: x ≤ m
    //   shift_b = clip(m - x, 0, 255)
    //   s_next  = s + zero_extend( exp_lut[shift_b] )
    //   m_next  = m

    wire [EXP_W-1:0]               exp_shift_val = exp_lut[m_minus_x_idx];
    wire [SUM_W + EXP_W - 1:0]     s_times_exp   = s * exp_shift_val;  // Q8.32 * Q0.32 = Q8.64
    wire [SUM_W-1:0]               s_scaled      = s_times_exp[SUM_W + SUM_FRAC_BITS - 1:
                                                               SUM_FRAC_BITS];
    wire [SUM_W-1:0]               s_caseA       = s_scaled + {{(SUM_W-EXP_W){1'b0}}, exp_lut[0]};
    wire [SUM_W-1:0]               s_caseB       = s + {{(SUM_W-EXP_W){1'b0}}, exp_val_rd};

    wire [SUM_W-1:0]               s_online_next = m_minus_x_neg ? s_caseA : s_caseB;
    wire signed [IN_W-1:0]         m_online_next = m_minus_x_neg ? in_data : m;

    // First-element seed: m = in_data, s = Q8.32(exp_lut[0]) ≈ 1.0
    wire [SUM_W-1:0]               s_first       = {{(SUM_W-EXP_W){1'b0}}, exp_lut[0]};

    // ========================================================================
    // Reciprocal indexing
    // ========================================================================
    localparam [SUM_W-1:0] HALF_BUCKET = {{(SUM_W-RECIP_SHIFT){1'b0}}, 1'b1, {(RECIP_SHIFT-1){1'b0}}};
    wire [SUM_W-1:0]       s_rounded   = s + HALF_BUCKET;
    wire [SUM_W-1:RECIP_SHIFT] r_idx_full = s_rounded[SUM_W-1:RECIP_SHIFT];
    wire                                 r_sat     = |r_idx_full[SUM_W-1:RECIP_SHIFT+10];
    wire [9:0]                           r_idx     = r_sat ? 10'h3FF : r_idx_full[RECIP_SHIFT+9:RECIP_SHIFT];

    // ========================================================================
    // Pass-2 output compute
    // ========================================================================
    // product = exp_val_rd * inv_s  (Q0.32 * Q0.32 = Q0.64)
    // y = round(product, top 8 bits) = (product + half_lsb) >> 56
    wire [2*EXP_W - 1:0]   prod2        = exp_val_rd * inv_s;
    localparam [2*EXP_W-1:0] HALF_LSB   = {{(2*EXP_W - (2*SUM_FRAC_BITS - OUT_W) + 1){1'b0}},
                                            1'b1,
                                            {(2*SUM_FRAC_BITS - OUT_W - 1){1'b0}}};
    wire [2*EXP_W - 1:0]   prod2_rnd    = prod2 + HALF_LSB;
    wire [OUT_W-1:0]       y_raw        = prod2_rnd[2*EXP_W - 1:2*EXP_W - OUT_W];
    // Saturation: if pre-rounding top of product above range, cap at 0xFF.
    // In practice max exp*inv_s ≤ 1*1 = (2^32-1)^2 / 2^64 ≈ 1.0 so no overflow.

    // ========================================================================
    // FSM
    // ========================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            m           <= {IN_W{1'b0}};
            s           <= {SUM_W{1'b0}};
            inv_s       <= {EXP_W{1'b0}};
            count       <= {COUNT_W{1'b0}};
            pass1_first <= 1'b0;
            out_valid   <= 1'b0;
            out_data    <= {OUT_W{1'b0}};
            done        <= 1'b0;
        end else begin
            // defaults (single-cycle pulses)
            out_valid   <= 1'b0;
            done        <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        state       <= S_PASS1;
                        count       <= {COUNT_W{1'b0}};
                        pass1_first <= 1'b1;
                        s           <= {SUM_W{1'b0}};
                        m           <= {IN_W{1'b0}};
                    end
                end

                S_PASS1: begin
                    if (in_valid) begin
                        if (pass1_first) begin
                            m           <= in_data;
                            s           <= s_first;
                            pass1_first <= 1'b0;
                        end else begin
                            m <= m_online_next;
                            s <= s_online_next;
                        end
                        count <= count + 1'b1;
                        if (count == VEC_LEN[COUNT_W-1:0] - 1) begin
                            state <= S_RECIP;
                        end
                    end
                end

                S_RECIP: begin
                    inv_s <= recip_lut[r_idx];
                    state <= S_PASS2;
                    count <= {COUNT_W{1'b0}};
                end

                S_PASS2: begin
                    if (in_valid) begin
                        out_valid <= 1'b1;
                        out_data  <= y_raw;
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
