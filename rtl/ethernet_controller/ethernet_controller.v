`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — Ethernet Controller RTL
// =============================================================================
// Implements an Ethernet II frame receiver and validator.
//
// Receives an incoming byte stream and validates the frame:
//   - Extracts destination MAC (bytes 0-5)
//   - Extracts source MAC (bytes 6-11)
//   - Extracts EtherType (bytes 12-13)
//   - Counts payload bytes
//   - Validates total frame length (min 64B, max 1518B for Ethernet II)
//
// Frame type encoding (frame_type):
//   2'd0  DATA      — unknown / raw
//   2'd1  IPv4      — EtherType = 0x0800
//   2'd2  ARP       — EtherType = 0x0806
//   2'd3  IPv6      — EtherType = 0x86DD
//
// MAC type encoding (mac_type):
//   2'd0  UNICAST   — LSB of dst_mac byte 0 = 0
//   2'd1  MULTICAST — LSB of dst_mac byte 0 = 1
//   2'd2  BROADCAST — dst_mac == FF:FF:FF:FF:FF:FF
//
// Interface:
//   clk          — system clock (rising edge)
//   rst_n        — active-low synchronous reset
//   rx_valid     — high when rx_byte contains a valid byte
//   rx_byte      — incoming byte from PHY
//   rx_last      — asserted with the last byte of the frame
//   frame_ok     — pulsed when frame_done and length is valid (64–1518 bytes)
//   frame_err    — pulsed when frame_done and length is invalid
//   ethertype    — 16-bit EtherType field (valid after frame_ok/frame_err)
//   frame_type   — 2-bit frame type (valid after frame_ok/frame_err)
//   mac_type     — 2-bit MAC address type (valid after frame_ok/frame_err)
//   byte_count   — number of bytes received in this frame
// =============================================================================

module ethernet_controller (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        rx_valid,
    input  wire [7:0]  rx_byte,
    input  wire        rx_last,

    output reg         frame_ok,
    output reg         frame_err,
    output reg  [15:0] ethertype,
    output reg  [1:0]  frame_type,
    output reg  [1:0]  mac_type,
    output reg  [10:0] byte_count
);

    // -------------------------------------------------------------------------
    // EtherType constants
    // -------------------------------------------------------------------------
    localparam ET_IPv4 = 16'h0800;
    localparam ET_ARP  = 16'h0806;
    localparam ET_IPv6 = 16'h86DD;

    // Frame length bounds
    localparam MIN_FRAME = 11'd64;
    localparam MAX_FRAME = 11'd1518;

    // -------------------------------------------------------------------------
    // Byte counter and field extraction
    // -------------------------------------------------------------------------
    reg [10:0] rx_count;    // bytes received so far
    reg [47:0] dst_mac_reg;
    reg [7:0]  et_hi;       // EtherType high byte (byte 12)

    always @(posedge clk) begin
        if (!rst_n) begin
            rx_count   <= 11'd0;
            dst_mac_reg<= 48'h0;
            et_hi      <= 8'h0;
            ethertype  <= 16'h0;
            frame_ok   <= 1'b0;
            frame_err  <= 1'b0;
            frame_type <= 2'd0;
            mac_type   <= 2'd0;
            byte_count <= 11'd0;
        end else begin
            frame_ok  <= 1'b0;
            frame_err <= 1'b0;

            if (rx_valid) begin
                rx_count <= rx_count + 11'd1;

                // Capture destination MAC (bytes 0-5)
                case (rx_count)
                    11'd0: dst_mac_reg[47:40] <= rx_byte;
                    11'd1: dst_mac_reg[39:32] <= rx_byte;
                    11'd2: dst_mac_reg[31:24] <= rx_byte;
                    11'd3: dst_mac_reg[23:16] <= rx_byte;
                    11'd4: dst_mac_reg[15:8]  <= rx_byte;
                    11'd5: dst_mac_reg[7:0]   <= rx_byte;
                    // Bytes 6-11: source MAC (not stored, just count)
                    // Byte 12: EtherType high
                    11'd12: et_hi <= rx_byte;
                    // Byte 13: EtherType low → form complete EtherType
                    11'd13: ethertype <= {et_hi, rx_byte};
                    default: ;
                endcase

                if (rx_last) begin
                    byte_count <= rx_count + 11'd1;  // total byte count

                    // Length validation
                    if ((rx_count + 11'd1) >= MIN_FRAME &&
                        (rx_count + 11'd1) <= MAX_FRAME) begin
                        frame_ok <= 1'b1;
                    end else begin
                        frame_err <= 1'b1;
                    end

                    // EtherType decode — use the stored register (captured at byte 13)
                    case (ethertype)
                        ET_IPv4: frame_type <= 2'd1;
                        ET_ARP:  frame_type <= 2'd2;
                        ET_IPv6: frame_type <= 2'd3;
                        default: frame_type <= 2'd0;
                    endcase

                    // MAC type decode (from dst_mac captured so far)
                    if (dst_mac_reg == 48'hFFFFFFFFFFFF)
                        mac_type <= 2'd2;   // broadcast
                    else if (dst_mac_reg[40])  // multicast bit = bit 0 of first byte
                        mac_type <= 2'd1;   // multicast
                    else
                        mac_type <= 2'd0;   // unicast

                    // Reset for next frame
                    rx_count <= 11'd0;
                end
            end
        end
    end

endmodule
