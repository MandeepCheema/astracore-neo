`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — NPU Tile Controller  (npu_tile_ctrl.v)
// =============================================================================
// Sequencer that runs ONE tile of compute end-to-end on the existing NPU
// datapath (npu_sram_ctrl + npu_systolic_array).  Replaces the hand-written
// cocotb sequencing in npu_tile_harness's testbench with an automated FSM.
//
// ── Tile execution flow ─────────────────────────────────────────────────────
//   IDLE → PRELOAD → EXEC_PREP → EXECUTE → DRAIN → STORE → DONE → IDLE
//
//   PRELOAD   Walk SRAM WA addresses 0..(N_ROWS*N_COLS - 1), reading one
//             weight per cycle.  Array's internal W grid latches each weight
//             1 cycle later (SRAM read latency), so array_load_cell_addr and
//             array_load_valid are the 1-cycle-delayed pipeline of the SRAM
//             read address.
//
//   EXEC_PREP One cycle, pulses array_clear_acc to zero the accumulators.
//
//   EXECUTE   Walk AI addresses (cfg_ai_base, cfg_ai_base + cfg_k - 1), one
//             per cycle.  array_exec_valid follows 1 cycle later (AI read
//             latency) so the array's a_valid and a_vec land in the same
//             cycle.
//
//   DRAIN     Wait 2 cycles for the exec pipeline to settle (exec_valid has
//             1-cycle delay, array's dst_wdata is combinational but the
//             caller wants busy to stay high until no further writes occur).
//
//   STORE     Pulse ao_we for 1 cycle with ao_waddr = cfg_ao_base.  The
//             array's c_vec is combinational on the accumulator, so writing
//             it directly captures the final tile result.
//
//   DONE      Pulse done for 1 cycle.  Return to IDLE.
//
// ── V1 scope ────────────────────────────────────────────────────────────────
//   • One tile per start pulse.  No microcode ROM, no multi-tile sequencing.
//   • No DMA integration — caller must pre-populate SRAM WA (weights) and AI
//     (activation vectors).
//   • No AFU integration — raw array c_vec is written to AO; activation
//     function application is the caller's responsibility until V2.
//   • w_bank_sel is hardcoded to 1'b0 (array reads WA, DMA writes WB).
//
// ── Integration note ───────────────────────────────────────────────────────
//   `array_load_valid` / `array_load_cell_addr` are internally delayed by
//   1 cycle (via preload_active_r / preload_addr_r) so they align with the
//   SRAM's registered w_rdata when wiring tile_ctrl directly to sram_ctrl.
//
//   `array_exec_valid` is driven LIVE (no internal delay).  When wiring
//   directly to the array, the integrator must add a 1-cycle register on
//   this signal (and on ai_re → array a_vec path) so the array's a_valid
//   aligns with the SRAM's ai_rdata.  `npu_tile_harness.v` provides this
//   register; any top-level integration should follow the same pattern.
//
// ── Parameter constraints ──────────────────────────────────────────────────
//   • N_ROWS * N_COLS must be ≤ 2^W_ADDR_W (preload counter wrap).
//   • cfg_k must be ≤ 2^AI_ADDR_W and cfg_ai_base + cfg_k - 1 must fit.
// =============================================================================

module npu_tile_ctrl #(
    parameter integer N_ROWS     = 4,
    parameter integer N_COLS     = 4,
    parameter integer ACC_W      = 32,
    // Narrow row-address: after the wide-weight-SRAM change, PRELOAD
    // walks N_ROWS addresses (one full row of N_COLS weights per cycle).
    parameter integer W_ADDR_W   = (N_ROWS <= 1) ? 1 : $clog2(N_ROWS),
    parameter integer AI_ADDR_W  = 8,
    parameter integer AO_ADDR_W  = 8,
    parameter integer K_W        = 16,
    parameter integer DRAIN_CYCLES = 2,
    parameter integer MP_VEC_W   = 11,            // up to VEC_LEN = 1024
    parameter integer MP_GAP_CYCLES = 6,          // pass 1 → pass 2 gap for AFU
    parameter integer MP_DRAIN_CYCLES = 4,        // post-pass-2 drain
    parameter integer ACC_DATA_W = N_COLS * ACC_W
)(
    input  wire                        clk,
    input  wire                        rst_n,

    // ── Control ─────────────────────────────────────────────────────────────
    input  wire                        start,
    input  wire [K_W-1:0]              cfg_k,
    input  wire [AI_ADDR_W-1:0]        cfg_ai_base,
    input  wire [AO_ADDR_W-1:0]        cfg_ao_base,
    // Gap #2 Phase 2: per-tile accumulator init mode.
    //   0 = clear accumulator at EXEC_PREP (original behaviour)
    //   1 = load accumulator from cfg_acc_init_data at EXEC_PREP
    // Used for software-managed k-tile chaining: after completing a
    // previous k-tile, the caller reads its output from AO and feeds it
    // back as cfg_acc_init_data for the next k-tile's start.
    input  wire                        cfg_acc_init_mode,
    input  wire [ACC_DATA_W-1:0]       cfg_acc_init_data,
    // ── F1-A4 multi-pass AFU mode ───────────────────────────────────────────
    // When cfg_mp_mode == 4'd0 the existing tile-compute flow runs (zero
    // regression risk). Non-zero values select a multi-pass AFU sequence:
    //   4'd8  = softmax
    //   4'd9  = layernorm
    //   4'd10 = rmsnorm
    // In multi-pass mode the normal PRELOAD/EXEC/STORE path is skipped;
    // the controller streams AI[cfg_ai_base .. cfg_ai_base+vec_len-1]
    // into an external AFU via `mp_in_valid`, waits MP_GAP_CYCLES, then
    // streams again for pass 2 and writes the AFU's `mp_out_data` into
    // AO[cfg_ao_base ..] via `mp_ao_we + mp_ao_waddr`.
    input  wire [3:0]                  cfg_mp_mode,
    input  wire [MP_VEC_W-1:0]         cfg_mp_vec_len,
    input  wire                        mp_out_valid,    // from AFU
    output reg                         busy,
    output reg                         done,

    // ── SRAM weight-read interface (drives WA read during PRELOAD) ─────────
    // w_raddr is a ROW address (0..N_ROWS-1).  SRAM returns the full row.
    output reg                         w_bank_sel,
    output reg                         w_re,
    output reg  [W_ADDR_W-1:0]         w_raddr,

    // ── Array weight-preload interface (1-cycle behind SRAM read) ──────────
    // array_load_cell_addr is a ROW index (matches the wide systolic
    // array's w_addr).  One w_load pulse latches one full row of weights.
    output reg                         array_load_valid,
    output reg  [W_ADDR_W-1:0]         array_load_cell_addr,
    output reg                         array_clear_acc,
    output reg                         array_acc_load_valid,
    output reg  [ACC_DATA_W-1:0]       array_acc_load_data,

    // ── SRAM AI-read + array execute interface ─────────────────────────────
    output reg                         ai_re,
    output reg  [AI_ADDR_W-1:0]        ai_raddr,
    output reg                         array_exec_valid,

    // ── Gap #3: writeback-AFU latch pulse ──────────────────────────────────
    // 1-cycle pulse in the LAST DRAIN cycle. By then the array's accumulator
    // holds the final tile sum (stable), so writeback AFUs in npu_top can
    // sample c_vec once and have their registered out_data valid when the
    // FSM reaches S_STORE one cycle later. Keeping it a single-cycle pulse
    // means the AFU latches exactly the final result, not any partial sum.
    output reg                         array_afu_in_valid,

    // ── SRAM AO-write interface ────────────────────────────────────────────
    output reg                         ao_we,
    output reg  [AO_ADDR_W-1:0]        ao_waddr,

    // ── F1-A4 multi-pass AFU control/data interface ────────────────────────
    // mp_start pulses 1 cycle before pass-1 streaming begins.
    // mp_mode forwards cfg_mp_mode to the AFU (latched at start).
    // mp_in_valid pulses once per cycle during pass-1 and pass-2 streaming,
    //   aligned 1 cycle after `ai_re` so the consumer sees ai_rdata on the
    //   same cycle.
    // mp_ao_we + mp_ao_waddr drive AO SRAM writes during pass-2 whenever
    //   the AFU raises mp_out_valid. Writes walk cfg_ao_base forward.
    output reg                         mp_start,
    output reg  [3:0]                  mp_mode,
    output reg                         mp_in_valid,
    output reg                         mp_ao_we,
    output reg  [AO_ADDR_W-1:0]        mp_ao_waddr
);

    // =========================================================================
    // FSM state encoding
    // =========================================================================
    localparam [3:0] S_IDLE      = 4'd0;
    localparam [3:0] S_PRELOAD   = 4'd1;
    localparam [3:0] S_EXEC_PREP = 4'd2;
    localparam [3:0] S_EXECUTE   = 4'd3;
    localparam [3:0] S_DRAIN     = 4'd4;
    localparam [3:0] S_STORE     = 4'd5;
    localparam [3:0] S_DONE      = 4'd6;
    // F1-A4 multi-pass AFU states (only entered when cfg_mp_mode != 0)
    localparam [3:0] S_MP_START  = 4'd8;
    localparam [3:0] S_MP_PASS1  = 4'd9;
    localparam [3:0] S_MP_GAP    = 4'd10;
    localparam [3:0] S_MP_PASS2  = 4'd11;
    localparam [3:0] S_MP_DRAIN  = 4'd12;
    localparam [3:0] S_MP_DONE   = 4'd13;

    reg [3:0] state;

    // Counters
    reg [W_ADDR_W-1:0]     preload_cnt;
    reg [K_W-1:0]          exec_cnt;
    reg [3:0]              drain_cnt;

    // Registered copies of address fields for the 1-cycle-delayed array
    // interface (matches the SRAM read pipeline).
    reg [W_ADDR_W-1:0]     preload_addr_r;
    reg                    preload_active_r;

    // Latched configuration — captured on start so the caller can drop
    // the config inputs (or change them for a pipelined next tile) while
    // the current tile is running.
    reg [K_W-1:0]          cfg_k_r;
    reg [AI_ADDR_W-1:0]    cfg_ai_base_r;
    reg [AO_ADDR_W-1:0]    cfg_ao_base_r;
    reg                    cfg_acc_init_mode_r;
    reg [ACC_DATA_W-1:0]   cfg_acc_init_data_r;

    // Multi-pass AFU counters / latched mode
    reg [3:0]              cfg_mp_mode_r;
    reg [MP_VEC_W-1:0]     cfg_mp_vec_len_r;
    reg [MP_VEC_W-1:0]     mp_cnt;
    reg [3:0]              mp_gap_cnt;
    reg [AO_ADDR_W-1:0]    mp_ao_wcnt;
    reg                    ai_re_r;

    // =========================================================================
    // Sequential FSM
    // =========================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            state            <= S_IDLE;
            busy             <= 1'b0;
            done             <= 1'b0;
            w_bank_sel       <= 1'b0;
            w_re             <= 1'b0;
            w_raddr          <= {W_ADDR_W{1'b0}};
            array_load_valid <= 1'b0;
            array_load_cell_addr <= {W_ADDR_W{1'b0}};
            array_clear_acc  <= 1'b0;
            array_acc_load_valid <= 1'b0;
            array_acc_load_data  <= {ACC_DATA_W{1'b0}};
            ai_re            <= 1'b0;
            ai_raddr         <= {AI_ADDR_W{1'b0}};
            array_exec_valid <= 1'b0;
            array_afu_in_valid <= 1'b0;
            ao_we            <= 1'b0;
            ao_waddr         <= {AO_ADDR_W{1'b0}};
            preload_cnt      <= {W_ADDR_W{1'b0}};
            exec_cnt         <= {K_W{1'b0}};
            drain_cnt        <= 4'd0;
            preload_addr_r   <= {W_ADDR_W{1'b0}};
            preload_active_r <= 1'b0;
            cfg_k_r          <= {K_W{1'b0}};
            cfg_ai_base_r    <= {AI_ADDR_W{1'b0}};
            cfg_ao_base_r    <= {AO_ADDR_W{1'b0}};
            cfg_acc_init_mode_r <= 1'b0;
            cfg_acc_init_data_r <= {ACC_DATA_W{1'b0}};
            mp_start         <= 1'b0;
            mp_mode          <= 4'd0;
            mp_in_valid      <= 1'b0;
            mp_ao_we         <= 1'b0;
            mp_ao_waddr      <= {AO_ADDR_W{1'b0}};
            cfg_mp_mode_r    <= 4'd0;
            cfg_mp_vec_len_r <= {MP_VEC_W{1'b0}};
            mp_cnt           <= {MP_VEC_W{1'b0}};
            mp_gap_cnt       <= 4'd0;
            mp_ao_wcnt       <= {AO_ADDR_W{1'b0}};
            ai_re_r          <= 1'b0;
        end else begin
            // --- default outputs (one-cycle pulses clear each cycle) ---
            done                 <= 1'b0;
            w_re                 <= 1'b0;
            array_load_valid     <= 1'b0;
            array_clear_acc      <= 1'b0;
            array_acc_load_valid <= 1'b0;
            array_acc_load_data  <= {ACC_DATA_W{1'b0}};
            ai_re                <= 1'b0;
            array_exec_valid     <= 1'b0;
            array_afu_in_valid   <= 1'b0;
            ao_we                <= 1'b0;
            mp_start             <= 1'b0;
            // mp_in_valid is a 1-cycle-delayed copy of ai_re so it arrives
            // aligned with SRAM's registered ai_rdata. ai_re is itself
            // registered (written in S_MP_PASS1/PASS2 as <= 1'b1), so
            // scheduling mp_in_valid to track ai_re gives the right phase.
            mp_in_valid          <= ai_re;
            mp_ao_we             <= 1'b0;
            ai_re_r              <= ai_re;

            // Shift the preload-read pipeline one stage: array sees the
            // address that was on the SRAM bus LAST cycle.  The array's
            // w_load and w_addr ports therefore align with the SRAM's
            // registered rdata on the same cycle.
            array_load_valid     <= preload_active_r;
            array_load_cell_addr <= preload_addr_r;
            preload_active_r     <= 1'b0;

            case (state)
                // ─────────────────────────────────────────────────────────
                S_IDLE: begin
                    if (start) begin
                        busy                <= 1'b1;
                        cfg_k_r             <= cfg_k;
                        cfg_ai_base_r       <= cfg_ai_base;
                        cfg_ao_base_r       <= cfg_ao_base;
                        cfg_acc_init_mode_r <= cfg_acc_init_mode;
                        cfg_acc_init_data_r <= cfg_acc_init_data;
                        cfg_mp_mode_r       <= cfg_mp_mode;
                        cfg_mp_vec_len_r    <= cfg_mp_vec_len;
                        if (cfg_mp_mode != 4'd0) begin
                            // Multi-pass AFU flow — skip tile compute.
                            state      <= S_MP_START;
                            mp_cnt     <= {MP_VEC_W{1'b0}};
                            mp_gap_cnt <= 4'd0;
                            mp_ao_wcnt <= cfg_ao_base;
                            mp_start   <= 1'b1;
                            mp_mode    <= cfg_mp_mode;
                        end else begin
                            state               <= S_PRELOAD;
                            preload_cnt         <= {W_ADDR_W{1'b0}};
                            w_bank_sel          <= 1'b0;
                        end
                    end
                end

                // ─────────────────────────────────────────────────────────
                S_PRELOAD: begin
                    // Issue SRAM read for current row counter (wide SRAM
                    // returns a full row of N_COLS weights per cycle).
                    w_re             <= 1'b1;
                    w_raddr          <= preload_cnt;
                    // Register the same address into the 1-cycle-delayed
                    // array load interface
                    preload_addr_r   <= preload_cnt;
                    preload_active_r <= 1'b1;

                    if (preload_cnt == (N_ROWS - 1)) begin
                        state       <= S_EXEC_PREP;
                        preload_cnt <= {W_ADDR_W{1'b0}};
                    end else begin
                        preload_cnt <= preload_cnt + 1'b1;
                    end
                end

                // ─────────────────────────────────────────────────────────
                S_EXEC_PREP: begin
                    // Initialise the array's accumulator.  Two modes:
                    //   cfg_acc_init_mode_r = 0 : clear to zero (default,
                    //                             first tile of a chain).
                    //   cfg_acc_init_mode_r = 1 : load from external data
                    //                             (continuation tile; caller
                    //                             feeds previous c_vec).
                    // The array's priority (clear > acc_load > accumulate)
                    // makes this exclusive by construction.
                    if (cfg_acc_init_mode_r) begin
                        array_acc_load_valid <= 1'b1;
                        array_acc_load_data  <= cfg_acc_init_data_r;
                    end else begin
                        array_clear_acc <= 1'b1;
                    end
                    state           <= S_EXECUTE;
                    exec_cnt        <= {K_W{1'b0}};
                end

                // ─────────────────────────────────────────────────────────
                S_EXECUTE: begin
                    ai_re            <= 1'b1;
                    // Verilog will size the + expression to max(operand, LHS)
                    // width and truncate on assignment.  Caller must ensure
                    // cfg_ai_base + cfg_k - 1 fits in AI_ADDR_W bits.
                    ai_raddr         <= cfg_ai_base_r + exec_cnt;
                    array_exec_valid <= 1'b1;  // registered next cycle by array

                    if (exec_cnt == cfg_k_r - 1) begin
                        state     <= S_DRAIN;
                        drain_cnt <= 4'd0;
                    end else begin
                        exec_cnt <= exec_cnt + 1'b1;
                    end
                end

                // ─────────────────────────────────────────────────────────
                S_DRAIN: begin
                    if (drain_cnt == DRAIN_CYCLES[3:0] - 4'd1) begin
                        // Last drain cycle: accumulator is stable.  Pulse
                        // writeback AFUs' in_valid so their registered
                        // out_data is ready when the next state (S_STORE)
                        // captures ao_wdata.
                        array_afu_in_valid <= 1'b1;
                        state              <= S_STORE;
                    end else begin
                        drain_cnt <= drain_cnt + 4'd1;
                    end
                end

                // ─────────────────────────────────────────────────────────
                S_STORE: begin
                    ao_we    <= 1'b1;
                    ao_waddr <= cfg_ao_base_r;
                    state    <= S_DONE;
                end

                // ─────────────────────────────────────────────────────────
                S_DONE: begin
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= S_IDLE;
                end

                // ─────────────────────────────────────────────────────────
                // Multi-pass AFU flow
                // ─────────────────────────────────────────────────────────
                S_MP_START: begin
                    // mp_start already pulsed in S_IDLE; this state exists
                    // so the AFU latches `start` before pass-1 data arrives.
                    state  <= S_MP_PASS1;
                    mp_cnt <= {MP_VEC_W{1'b0}};
                end

                S_MP_PASS1: begin
                    ai_re    <= 1'b1;
                    ai_raddr <= cfg_ai_base_r + mp_cnt[AI_ADDR_W-1:0];
                    if (mp_cnt == cfg_mp_vec_len_r - 1) begin
                        state      <= S_MP_GAP;
                        mp_gap_cnt <= 4'd0;
                    end else begin
                        mp_cnt <= mp_cnt + 1'b1;
                    end
                end

                S_MP_GAP: begin
                    // Wait MP_GAP_CYCLES for AFU's reciprocal/rsqrt phase.
                    if (mp_gap_cnt == MP_GAP_CYCLES[3:0] - 4'd1) begin
                        state  <= S_MP_PASS2;
                        mp_cnt <= {MP_VEC_W{1'b0}};
                    end else begin
                        mp_gap_cnt <= mp_gap_cnt + 4'd1;
                    end
                end

                S_MP_PASS2: begin
                    ai_re    <= 1'b1;
                    ai_raddr <= cfg_ai_base_r + mp_cnt[AI_ADDR_W-1:0];
                    if (mp_cnt == cfg_mp_vec_len_r - 1) begin
                        state      <= S_MP_DRAIN;
                        mp_gap_cnt <= 4'd0;
                    end else begin
                        mp_cnt <= mp_cnt + 1'b1;
                    end
                    // The AFU's out_valid may land on this or the next
                    // cycle; drive ao_we through the combinational branch
                    // below (outside the case statement).
                end

                S_MP_DRAIN: begin
                    if (mp_gap_cnt == MP_DRAIN_CYCLES[3:0] - 4'd1) begin
                        state <= S_MP_DONE;
                    end else begin
                        mp_gap_cnt <= mp_gap_cnt + 4'd1;
                    end
                end

                S_MP_DONE: begin
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase

            // ─────────────────────────────────────────────────────────
            // mp_ao_we writes:  whenever the AFU pulses mp_out_valid
            // during PASS2 or DRAIN, relay a write into AO at the next
            // sequential address starting from cfg_ao_base.
            // ─────────────────────────────────────────────────────────
            if (mp_out_valid &&
                (state == S_MP_PASS2 || state == S_MP_DRAIN)) begin
                mp_ao_we    <= 1'b1;
                mp_ao_waddr <= mp_ao_wcnt;
                mp_ao_wcnt  <= mp_ao_wcnt + 1'b1;
            end
        end
    end

endmodule
