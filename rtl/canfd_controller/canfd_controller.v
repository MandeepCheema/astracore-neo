`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — CAN-FD Controller RTL
// =============================================================================
// Implements the ISO 11898-1 CAN-FD error counter and bus-state FSM.
//
// Error counters:
//   TEC (Transmit Error Counter): 9-bit, increments by 8 on tx_error,
//       decrements by 1 on tx_success (floor 0).
//   REC (Receive Error Counter):  8-bit, increments by 1 on rx_error.
//
// Bus state transitions:
//   ERROR_ACTIVE  (2'b00) — TEC < 128 AND REC < 128
//   ERROR_PASSIVE (2'b01) — TEC >= 128 OR REC >= 128 (but TEC < 256)
//   BUS_OFF       (2'b10) — TEC >= 256
//
// Bus-off recovery:
//   Assert bus_off_recovery when bus_state == BUS_OFF.
//   Resets TEC and REC to 0 and returns to ERROR_ACTIVE.
//
// Interface:
//   clk              — system clock (rising edge active)
//   rst_n            — active-low synchronous reset
//   tx_success       — successful TX frame; decrement TEC by 1
//   tx_error         — TX error; increment TEC by 8
//   rx_error         — RX error; increment REC by 1
//   bus_off_recovery — initiate bus-off recovery sequence
//   tec              — 9-bit Transmit Error Counter
//   rec              — 8-bit Receive Error Counter
//   bus_state        — 2-bit bus state (00=ACTIVE, 01=PASSIVE, 10=BUS_OFF)
// =============================================================================

module canfd_controller (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       tx_success,
    input  wire       tx_error,
    input  wire       rx_error,
    input  wire       bus_off_recovery,

    output reg  [8:0] tec,
    output reg  [7:0] rec,
    output reg  [1:0] bus_state
);

    localparam BUS_ERROR_ACTIVE  = 2'd0;
    localparam BUS_ERROR_PASSIVE = 2'd1;
    localparam BUS_OFF           = 2'd2;

    // -------------------------------------------------------------------------
    // Combinational next-state computation
    // -------------------------------------------------------------------------
    wire is_bus_off = (bus_state == BUS_OFF);
    wire do_recovery = bus_off_recovery && is_bus_off;

    // Next TEC
    wire [8:0] tec_adj =
        tx_error   ? (tec + 9'd8) :
        tx_success ? ((tec > 9'd0) ? tec - 9'd1 : 9'd0) :
        tec;
    wire [8:0] next_tec = do_recovery ? 9'd0 :
                          (!is_bus_off) ? tec_adj : tec;

    // Next REC
    wire [7:0] next_rec = do_recovery ? 8'd0 :
                          (!is_bus_off && rx_error) ?
                              ((rec == 8'd255) ? 8'd255 : rec + 8'd1) :
                          rec;

    // Next bus state — evaluated on next_tec/next_rec
    wire [1:0] next_bus_state =
        do_recovery            ? BUS_ERROR_ACTIVE  :
        (next_tec >= 9'd256)   ? BUS_OFF           :
        (next_tec >= 9'd128 || next_rec >= 8'd128) ? BUS_ERROR_PASSIVE :
                                 BUS_ERROR_ACTIVE;

    // -------------------------------------------------------------------------
    // Sequential registers
    // -------------------------------------------------------------------------
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

endmodule
