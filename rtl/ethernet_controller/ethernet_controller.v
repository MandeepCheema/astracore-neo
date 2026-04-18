`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — Ethernet Controller RTL  (Rev 2 — payload stream + TX added)
// =============================================================================
// Rev 1: Ethernet II frame receiver + validator.
//        Extracts dst MAC, EtherType, validates length (64–1518 B).
// Rev 2: Adds RX payload byte streaming and a TX byte pipeline.
//
// ── Rev-1 (preserved, backward compatible) ───────────────────────────────────
//   clk, rst_n
//   rx_valid, rx_byte, rx_last          ← incoming byte stream from PHY
//   frame_ok, frame_err                 ← 1-cycle pulse after rx_last
//   ethertype[15:0], frame_type[1:0], mac_type[1:0], byte_count[10:0]
//
// ── Rev-2: RX payload stream (new) ───────────────────────────────────────────
//   Combinatorial: bytes 14+ (after 6B dst + 6B src + 2B EtherType) forwarded
//   as they arrive.  No buffering — consumer (lidar_interface) must accept each
//   byte in the same cycle it appears.
//     rx_payload_valid  — 1 when a payload byte is present
//     rx_payload_byte   — payload byte value (= rx_byte, pass-through)
//     rx_payload_last   — 1 with the last payload byte (= rx_last, combinatorial)
//
// ── Rev-2: TX byte pipeline (new) ────────────────────────────────────────────
//   1-cycle registered pipeline: upstream (ptp_clock_sync / aeb_controller)
//   drives tx_valid + tx_byte_in + tx_last; output appears one cycle later.
//     tx_valid, tx_byte_in[7:0], tx_last  ← from upstream
//     tx_ready                            ← always 1 (PHY always ready model)
//     tx_out_valid, tx_out_byte[7:0], tx_out_last  ← to PHY
// =============================================================================

module ethernet_controller (
    input  wire        clk,
    input  wire        rst_n,

    // ── Rev-1: RX byte stream (preserved) ───────────────────────────────────
    input  wire        rx_valid,
    input  wire [7:0]  rx_byte,
    input  wire        rx_last,

    output reg         frame_ok,
    output reg         frame_err,
    output reg  [15:0] ethertype,
    output reg  [1:0]  frame_type,
    output reg  [1:0]  mac_type,
    output reg  [10:0] byte_count,

    // ── Rev-2: RX payload stream (new, combinatorial) ───────────────────────
    // Active for bytes 14+ of each received frame (after 6+6+2 header bytes).
    output wire        rx_payload_valid,  // high when payload byte present
    output wire [7:0]  rx_payload_byte,   // payload byte (= rx_byte)
    output wire        rx_payload_last,   // high with last payload byte

    // ── Rev-2: TX byte pipeline (new) ────────────────────────────────────────
    input  wire        tx_valid,          // upstream has a TX byte to send
    input  wire [7:0]  tx_byte_in,        // TX byte data
    input  wire        tx_last,           // last byte of this TX frame
    output wire        tx_ready,          // always 1 — PHY always ready model

    output reg         tx_out_valid,      // TX byte valid (registered, 1-cycle delay)
    output reg  [7:0]  tx_out_byte,       // TX byte to PHY
    output reg         tx_out_last        // last TX byte pulse
);

    // =========================================================================
    // 1. Rev-1: EtherType constants and frame length bounds (unchanged)
    // =========================================================================
    localparam ET_IPv4 = 16'h0800;
    localparam ET_ARP  = 16'h0806;
    localparam ET_IPv6 = 16'h86DD;

    localparam MIN_FRAME    = 11'd64;
    localparam MAX_FRAME    = 11'd1518;
    localparam PAYLOAD_START= 11'd14;   // byte index of first payload byte

    // =========================================================================
    // 2. Rev-1: Byte counter, field extraction, frame validation (unchanged)
    // =========================================================================
    reg [10:0] rx_count;
    reg [47:0] dst_mac_reg;
    reg [7:0]  et_hi;

    always @(posedge clk) begin
        if (!rst_n) begin
            rx_count    <= 11'd0;
            dst_mac_reg <= 48'h0;
            et_hi       <= 8'h0;
            ethertype   <= 16'h0;
            frame_ok    <= 1'b0;
            frame_err   <= 1'b0;
            frame_type  <= 2'd0;
            mac_type    <= 2'd0;
            byte_count  <= 11'd0;
        end else begin
            frame_ok  <= 1'b0;
            frame_err <= 1'b0;

            if (rx_valid) begin
                rx_count <= rx_count + 11'd1;

                // Capture destination MAC (bytes 0-5)
                case (rx_count)
                    11'd0:  dst_mac_reg[47:40] <= rx_byte;
                    11'd1:  dst_mac_reg[39:32] <= rx_byte;
                    11'd2:  dst_mac_reg[31:24] <= rx_byte;
                    11'd3:  dst_mac_reg[23:16] <= rx_byte;
                    11'd4:  dst_mac_reg[15:8]  <= rx_byte;
                    11'd5:  dst_mac_reg[7:0]   <= rx_byte;
                    // Bytes 6-11: source MAC (counted, not stored)
                    11'd12: et_hi <= rx_byte;
                    11'd13: ethertype <= {et_hi, rx_byte};
                    default: ;
                endcase

                if (rx_last) begin
                    byte_count <= rx_count + 11'd1;

                    if ((rx_count + 11'd1) >= MIN_FRAME &&
                        (rx_count + 11'd1) <= MAX_FRAME) begin
                        frame_ok <= 1'b1;
                    end else begin
                        frame_err <= 1'b1;
                    end

                    case (ethertype)
                        ET_IPv4: frame_type <= 2'd1;
                        ET_ARP:  frame_type <= 2'd2;
                        ET_IPv6: frame_type <= 2'd3;
                        default: frame_type <= 2'd0;
                    endcase

                    if (dst_mac_reg == 48'hFFFFFFFFFFFF)
                        mac_type <= 2'd2;
                    else if (dst_mac_reg[40])
                        mac_type <= 2'd1;
                    else
                        mac_type <= 2'd0;

                    rx_count <= 11'd0;
                end
            end
        end
    end

    // =========================================================================
    // 3. Rev-2: RX payload streaming (combinatorial)
    // Bytes 14+ forwarded in real-time.  rx_payload_byte / rx_payload_last are
    // direct pass-throughs of rx_byte / rx_last; only rx_payload_valid gates
    // whether the byte is part of the payload.
    // =========================================================================
    // rx_count holds the byte index of the CURRENT incoming byte (pre-NBA).
    assign rx_payload_valid = rx_valid && (rx_count >= PAYLOAD_START);
    assign rx_payload_byte  = rx_byte;
    assign rx_payload_last  = rx_last;   // meaningful only when rx_payload_valid

    // =========================================================================
    // 4. Rev-2: TX byte pipeline (1-cycle registered delay)
    // ptp_clock_sync drives tx_valid + tx_byte_in + tx_last.
    // Result appears on tx_out_* one cycle later.
    // tx_ready is permanently 1 (no back-pressure in this model).
    // =========================================================================
    assign tx_ready = 1'b1;

    always @(posedge clk) begin
        if (!rst_n) begin
            tx_out_valid <= 1'b0;
            tx_out_byte  <= 8'h00;
            tx_out_last  <= 1'b0;
        end else begin
            tx_out_valid <= tx_valid;
            tx_out_byte  <= tx_byte_in;
            tx_out_last  <= tx_valid && tx_last;  // last only fires when byte is valid
        end
    end

endmodule
