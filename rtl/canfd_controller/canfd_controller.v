`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — CAN-FD Controller RTL  (Rev 2 — frame interface added)
// =============================================================================
// Rev 1: ISO 11898-1 error counter + bus-state FSM.
// Rev 2: Adds 4-entry RX FIFO (transceiver → decoder) and 4-entry TX FIFO
//        (aeb_controller → transceiver) for frame-level data movement.
//
// ── Error counter (unchanged from Rev 1) ─────────────────────────────────────
//   TEC 9-bit: +8 per tx_error, -1 per tx_success (floor 0), auto-+1 per
//              completed TX frame (via int_tx_success from TX FIFO drain).
//   REC 8-bit: +1 per rx_error (saturates at 255).
//   Bus states: ERROR_ACTIVE(00) / ERROR_PASSIVE(01) / BUS_OFF(10).
//   bus_off_recovery: resets TEC+REC, returns to ERROR_ACTIVE.
//
// ── RX frame FIFO (new) ───────────────────────────────────────────────────────
//   Accepts pre-parsed CAN-FD frames from the CAN transceiver.
//   4-entry FIFO; downstream (can_odometry_decoder) reads via rx_out_* AXI-S.
//   Backpressure: rx_frame_ready = 0 when full (upstream must stall).
//   While BUS_OFF no new frames are accepted (rx_frame_ready = 0).
//
// ── TX frame FIFO (new) ───────────────────────────────────────────────────────
//   Accepts frame requests from aeb_controller via tx_frame_* AXI-S.
//   4-entry FIFO; drains one frame per cycle when not BUS_OFF.
//   tx_frame_done pulses for 1 cycle when a frame exits the TX FIFO.
//   Each tx_frame_done also auto-fires an internal tx_success (updates TEC).
//
// ── Interface ─────────────────────────────────────────────────────────────────
//   REV-1 PORTS (preserved, backward compatible):
//     clk, rst_n
//     tx_success, tx_error, rx_error, bus_off_recovery  ← direct error strobes
//     tec[8:0], rec[7:0], bus_state[1:0]
//
//   REV-2 RX PORTS (new):
//     rx_frame_valid, rx_frame_id[28:0], rx_frame_dlc[3:0], rx_frame_data[63:0]
//     rx_frame_ready                                         ← backpressure out
//     rx_out_valid, rx_out_id[28:0], rx_out_dlc[3:0], rx_out_data[63:0]
//     rx_out_ready                                           ← consumer input
//
//   REV-2 TX PORTS (new):
//     tx_frame_valid, tx_frame_id[28:0], tx_frame_dlc[3:0], tx_frame_data[63:0]
//     tx_frame_ready                                         ← backpressure out
//     tx_frame_done                                          ← 1-cycle completion
// =============================================================================

module canfd_controller (
    input  wire        clk,
    input  wire        rst_n,

    // ── Rev-1: direct error counter strobes (preserved) ──────────────────────
    input  wire        tx_success,
    input  wire        tx_error,
    input  wire        rx_error,
    input  wire        bus_off_recovery,

    output reg  [8:0]  tec,
    output reg  [7:0]  rec,
    output reg  [1:0]  bus_state,

    // ── Rev-2: RX frame FIFO ──────────────────────────────────────────────────
    // Upstream write port (CAN transceiver → this FIFO)
    input  wire        rx_frame_valid,
    input  wire [28:0] rx_frame_id,      // 29-bit extended CAN-FD ID
    input  wire [3:0]  rx_frame_dlc,     // data length code 0–15
    input  wire [63:0] rx_frame_data,    // first 8 bytes of payload
    output wire        rx_frame_ready,   // FIFO not full and not BUS_OFF

    // Downstream read port (can_odometry_decoder ← this FIFO)
    output wire        rx_out_valid,
    output wire [28:0] rx_out_id,
    output wire [3:0]  rx_out_dlc,
    output wire [63:0] rx_out_data,
    input  wire        rx_out_ready,

    // ── Rev-2: TX frame FIFO ──────────────────────────────────────────────────
    // Upstream write port (aeb_controller → this FIFO)
    input  wire        tx_frame_valid,
    input  wire [28:0] tx_frame_id,
    input  wire [3:0]  tx_frame_dlc,
    input  wire [63:0] tx_frame_data,
    output wire        tx_frame_ready,   // FIFO not full and not BUS_OFF

    // Completion
    output reg         tx_frame_done     // 1-cycle pulse: frame left TX FIFO
);

    // =========================================================================
    // 1. Bus-state FSM and error counters (Rev 1 — unchanged logic)
    // =========================================================================
    localparam BUS_ERROR_ACTIVE  = 2'd0;
    localparam BUS_ERROR_PASSIVE = 2'd1;
    localparam BUS_OFF           = 2'd2;

    wire is_bus_off   = (bus_state == BUS_OFF);
    wire do_recovery  = bus_off_recovery && is_bus_off;

    // Internal tx_success pulse from TX FIFO drain (auto-updates TEC)
    wire int_tx_success;

    // Effective error inputs: OR of direct strobes and internal frame signals
    wire eff_tx_success = tx_success | int_tx_success;
    wire eff_tx_error   = tx_error;
    wire eff_rx_error   = rx_error;

    // Next TEC
    wire [8:0] tec_adj =
        eff_tx_error   ? (tec + 9'd8)                    :
        eff_tx_success ? ((tec > 9'd0) ? tec - 9'd1 : 9'd0) :
        tec;
    wire [8:0] next_tec = do_recovery  ? 9'd0         :
                          (!is_bus_off) ? tec_adj       : tec;

    // Next REC
    wire [7:0] next_rec = do_recovery                          ? 8'd0  :
                          (!is_bus_off && eff_rx_error)        ?
                              ((rec == 8'd255) ? 8'd255 : rec + 8'd1) :
                          rec;

    // Next bus state
    wire [1:0] next_bus_state =
        do_recovery            ? BUS_ERROR_ACTIVE  :
        (next_tec >= 9'd256)   ? BUS_OFF           :
        (next_tec >= 9'd128 || next_rec >= 8'd128) ? BUS_ERROR_PASSIVE :
                                 BUS_ERROR_ACTIVE;

    always @(posedge clk) begin
        if (!rst_n) begin
            tec       <= 9'd0;
            rec       <= 8'd0;
            bus_state <= BUS_ERROR_ACTIVE;
        end else begin
            tec       <= next_tec;
            rec       <= next_rec;
            bus_state <= next_bus_state;
        end
    end

    // =========================================================================
    // 2. RX Frame FIFO — 4 entries
    //    Each entry: id[28:0] | dlc[3:0] | data[63:0]  = 96 bits
    // =========================================================================
    localparam FIFO_DEPTH = 4;

    // RX storage
    reg [28:0] rx_id_buf   [0:FIFO_DEPTH-1];
    reg [3:0]  rx_dlc_buf  [0:FIFO_DEPTH-1];
    reg [63:0] rx_data_buf [0:FIFO_DEPTH-1];
    reg [1:0]  rx_wptr;     // write pointer
    reg [1:0]  rx_rptr;     // read pointer
    reg [2:0]  rx_count;    // 0–4

    wire rx_full  = (rx_count == FIFO_DEPTH);
    wire rx_empty = (rx_count == 3'd0);

    // rx_frame_ready: accept only when not full and bus is not BUS_OFF
    assign rx_frame_ready = !rx_full && !is_bus_off;

    // RX FIFO write
    wire rx_wr = rx_frame_valid && rx_frame_ready;
    // RX FIFO read
    wire rx_rd = rx_out_valid && rx_out_ready;

    always @(posedge clk) begin
        if (!rst_n) begin
            rx_wptr  <= 2'd0;
            rx_rptr  <= 2'd0;
            rx_count <= 3'd0;
        end else begin
            if (rx_wr) begin
                rx_id_buf  [rx_wptr] <= rx_frame_id;
                rx_dlc_buf [rx_wptr] <= rx_frame_dlc;
                rx_data_buf[rx_wptr] <= rx_frame_data;
                rx_wptr              <= rx_wptr + 2'd1;  // wraps at 4
            end
            if (rx_rd) begin
                rx_rptr <= rx_rptr + 2'd1;               // wraps at 4
            end
            // Count update
            case ({rx_wr, rx_rd})
                2'b10: rx_count <= rx_count + 3'd1;
                2'b01: rx_count <= rx_count - 3'd1;
                default: ;
            endcase
        end
    end

    assign rx_out_valid = !rx_empty;
    assign rx_out_id    = rx_id_buf  [rx_rptr];
    assign rx_out_dlc   = rx_dlc_buf [rx_rptr];
    assign rx_out_data  = rx_data_buf[rx_rptr];

    // =========================================================================
    // 3. TX Frame FIFO — 4 entries
    //    Drains one frame per cycle when bus is not BUS_OFF.
    //    tx_frame_done pulses when drain occurs → also fires int_tx_success.
    // =========================================================================
    reg [28:0] tx_id_buf   [0:FIFO_DEPTH-1];
    reg [3:0]  tx_dlc_buf  [0:FIFO_DEPTH-1];
    reg [63:0] tx_data_buf [0:FIFO_DEPTH-1];
    reg [1:0]  tx_wptr;
    reg [1:0]  tx_rptr;
    reg [2:0]  tx_count;

    wire tx_full  = (tx_count == FIFO_DEPTH);
    wire tx_empty = (tx_count == 3'd0);

    // tx_frame_ready: accept only when not full and bus is not BUS_OFF
    assign tx_frame_ready = !tx_full && !is_bus_off;

    // TX FIFO write
    wire tx_wr = tx_frame_valid && tx_frame_ready;
    // TX FIFO drain: one frame per cycle when not empty and not BUS_OFF
    wire tx_drain = !tx_empty && !is_bus_off;

    // int_tx_success: fires when TX FIFO drains a frame
    assign int_tx_success = tx_drain;

    always @(posedge clk) begin
        if (!rst_n) begin
            tx_wptr       <= 2'd0;
            tx_rptr       <= 2'd0;
            tx_count      <= 3'd0;
            tx_frame_done <= 1'b0;
        end else begin
            // Single assignment: reflects drain-active state of this cycle
            tx_frame_done <= tx_drain;

            if (tx_wr) begin
                tx_id_buf  [tx_wptr] <= tx_frame_id;
                tx_dlc_buf [tx_wptr] <= tx_frame_dlc;
                tx_data_buf[tx_wptr] <= tx_frame_data;
                tx_wptr              <= tx_wptr + 2'd1;
            end

            if (tx_drain) begin
                tx_rptr <= tx_rptr + 2'd1;
            end

            // Count update (tx_drain and tx_wr may happen same cycle)
            case ({tx_wr, tx_drain})
                2'b10: tx_count <= tx_count + 3'd1;
                2'b01: tx_count <= tx_count - 3'd1;
                2'b11: tx_count <= tx_count;           // simultaneous: no change
                default: ;
            endcase
        end
    end

    // =========================================================================
    // CAN-FD safety invariants — Phase-F formal verification targets
    // =========================================================================
`ifndef SYNTHESIS
`ifndef __ICARUS__
    // Invariant 1: bus_state is one of {ACTIVE, PASSIVE, BUS_OFF}. 2'b11 never.
    property p_bus_state_valid;
        @(posedge clk) disable iff (!rst_n)
        (bus_state == BUS_ERROR_ACTIVE) ||
        (bus_state == BUS_ERROR_PASSIVE) ||
        (bus_state == BUS_OFF);
    endproperty
    a_bus_state_valid: assert property (p_bus_state_valid)
        else $error("CAN-FD: bus_state out of {ACTIVE, PASSIVE, BUS_OFF}");

    // Invariant 2: BUS_OFF implies tec >= 256 (by ISO 11898-1).
    property p_bus_off_iff_tec_sat;
        @(posedge clk) disable iff (!rst_n)
        (bus_state == BUS_OFF) |-> (tec >= 9'd256);
    endproperty
    a_bus_off_iff_tec_sat: assert property (p_bus_off_iff_tec_sat)
        else $error("CAN-FD: BUS_OFF without tec saturated");

    // Invariant 3: no new frames accepted while BUS_OFF.
    property p_no_rx_when_bus_off;
        @(posedge clk) disable iff (!rst_n)
        is_bus_off |-> !rx_frame_ready;
    endproperty
    a_no_rx_when_bus_off: assert property (p_no_rx_when_bus_off)
        else $error("CAN-FD: rx_frame_ready asserted during BUS_OFF");

    // Invariant 4: FIFO counts stay within 0..4.
    property p_rx_count_valid;
        @(posedge clk) disable iff (!rst_n)
        (rx_count <= 3'd4);
    endproperty
    a_rx_count_valid: assert property (p_rx_count_valid)
        else $error("CAN-FD: rx_count > 4 (FIFO depth)");

    property p_tx_count_valid;
        @(posedge clk) disable iff (!rst_n)
        (tx_count <= 3'd4);
    endproperty
    a_tx_count_valid: assert property (p_tx_count_valid)
        else $error("CAN-FD: tx_count > 4 (FIFO depth)");
`endif
`endif

endmodule
