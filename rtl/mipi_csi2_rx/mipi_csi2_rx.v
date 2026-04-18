`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — MIPI CSI-2 Receiver (Packet Layer)  (mipi_csi2_rx.v)
// =============================================================================
// Layer 1 sensor interface.  Consumes the byte-aligned post-D-PHY stream from
// a MIPI CSI-2 camera source (DMS or front cabin) and extracts:
//   • Frame / line sync events (short packets)
//   • Pixel payload byte stream for long packets
//
// ── Scope of v1 ──────────────────────────────────────────────────────────────
// The D-PHY physical layer (HS/LP mode, clock recovery, per-lane byte deskew,
// 4→1 lane merge) is handled by an external deserialiser that presents a
// single synchronous byte stream with byte_valid.  Sync-byte (0xB8) detection
// and header framing are also assumed done externally — this module starts
// receiving at the first DI byte of each packet.
//
// ECC on the 4-byte header and CRC on the long-packet footer are accepted but
// not validated in v1.  The upgrade path is to insert a Hamming-modified ECC
// checker on HDR3 and a CRC-16 checker across the footer bytes.
//
// ── CSI-2 packet layout ──────────────────────────────────────────────────────
//   Short packet (4 bytes): DI, WC[7:0], WC[15:8], ECC
//     Short packets have DT < 0x10 and carry no payload:
//       DT 0x00 = Frame Start (FS)
//       DT 0x01 = Frame End   (FE)
//       DT 0x02 = Line Start  (LS)
//       DT 0x03 = Line End    (LE)
//     The WC field of a short packet holds frame/line number (not used v1).
//
//   Long packet: DI, WC[7:0], WC[15:8], ECC, <WC payload bytes>, CRC[7:0], CRC[15:8]
//     DT 0x10..0x3F — pixel / image data.  WC gives payload byte count.
//
//   DI byte: {VC[1:0], DT[5:0]}
//
// ── FSM ──────────────────────────────────────────────────────────────────────
//   S_HDR0  : capture DI (VC + DT)
//   S_HDR1  : capture WC[7:0]
//   S_HDR2  : capture WC[15:8] and classify short vs long
//   S_HDR3  : consume ECC byte; on short packet go straight back to HDR0
//             and emit the corresponding event pulse
//   S_DATA  : stream WC payload bytes as pixel_valid (long packets only)
//   S_CRC0  : consume first CRC byte
//   S_CRC1  : consume second CRC byte, back to HDR0
//
// ── Interface ─────────────────────────────────────────────────────────────────
//   byte_valid / byte_data          — post-D-PHY byte stream
//   frame_start / frame_end / line_start / line_end — 1-cycle event pulses
//   pixel_valid / pixel_byte / pixel_last — AXI-S payload stream
//   last_data_type / last_word_count / last_virtual_channel — latched header info
//   frame_count / line_count / error_count — saturating statistics
// =============================================================================

module mipi_csi2_rx (
    input  wire        clk,
    input  wire        rst_n,

    // ── Post-D-PHY byte stream ────────────────────────────────────────────────
    input  wire        byte_valid,
    input  wire [7:0]  byte_data,

    // ── Short-packet event pulses (1 cycle) ──────────────────────────────────
    output reg         frame_start,
    output reg         frame_end,
    output reg         line_start,
    output reg         line_end,

    // ── Header status (latched on each received packet) ─────────────────────
    output reg  [7:0]  last_data_type,
    output reg  [15:0] last_word_count,
    output reg  [1:0]  last_virtual_channel,

    // ── Long-packet payload AXI-Stream ───────────────────────────────────────
    output reg         pixel_valid,
    output reg  [7:0]  pixel_byte,
    output reg         pixel_last,

    // ── Counters ──────────────────────────────────────────────────────────────
    output reg  [15:0] frame_count,
    output reg  [15:0] line_count,
    output reg  [15:0] error_count
);

    // =========================================================================
    // Data-type opcodes
    // =========================================================================
    localparam [5:0] DT_FS = 6'h00;
    localparam [5:0] DT_FE = 6'h01;
    localparam [5:0] DT_LS = 6'h02;
    localparam [5:0] DT_LE = 6'h03;

    // =========================================================================
    // FSM state encoding
    // =========================================================================
    localparam [2:0] S_HDR0 = 3'd0,
                     S_HDR1 = 3'd1,
                     S_HDR2 = 3'd2,
                     S_HDR3 = 3'd3,
                     S_DATA = 3'd4,
                     S_CRC0 = 3'd5,
                     S_CRC1 = 3'd6;

    reg [2:0]  state;
    reg [5:0]  dt_reg;
    reg [1:0]  vc_reg;
    reg [15:0] wc_reg;
    reg [15:0] data_remaining;

    wire is_short_packet = (dt_reg < 6'h10);

    always @(posedge clk) begin
        if (!rst_n) begin
            state                <= S_HDR0;
            dt_reg               <= 6'd0;
            vc_reg               <= 2'd0;
            wc_reg               <= 16'd0;
            data_remaining       <= 16'd0;
            frame_start          <= 1'b0;
            frame_end            <= 1'b0;
            line_start           <= 1'b0;
            line_end             <= 1'b0;
            last_data_type       <= 8'd0;
            last_word_count      <= 16'd0;
            last_virtual_channel <= 2'd0;
            pixel_valid          <= 1'b0;
            pixel_byte           <= 8'd0;
            pixel_last           <= 1'b0;
            frame_count          <= 16'd0;
            line_count           <= 16'd0;
            error_count          <= 16'd0;
        end else begin
            // Default de-assert of all 1-cycle pulses
            frame_start <= 1'b0;
            frame_end   <= 1'b0;
            line_start  <= 1'b0;
            line_end    <= 1'b0;
            pixel_valid <= 1'b0;
            pixel_last  <= 1'b0;

            if (byte_valid) begin
                case (state)
                    // ── DI byte: capture VC + DT ────────────────────────────
                    S_HDR0: begin
                        vc_reg <= byte_data[7:6];
                        dt_reg <= byte_data[5:0];
                        state  <= S_HDR1;
                    end

                    // ── WC low byte ─────────────────────────────────────────
                    S_HDR1: begin
                        wc_reg[7:0] <= byte_data;
                        state <= S_HDR2;
                    end

                    // ── WC high byte ────────────────────────────────────────
                    S_HDR2: begin
                        wc_reg[15:8] <= byte_data;
                        state <= S_HDR3;
                    end

                    // ── ECC byte (consumed; not validated v1) ───────────────
                    S_HDR3: begin
                        // Latch header info for status readout
                        last_data_type       <= {2'b00, dt_reg};
                        last_word_count      <= wc_reg;
                        last_virtual_channel <= vc_reg;

                        if (is_short_packet) begin
                            // Emit the corresponding event pulse
                            case (dt_reg)
                                DT_FS: begin
                                    frame_start <= 1'b1;
                                    if (frame_count != 16'hFFFF)
                                        frame_count <= frame_count + 16'd1;
                                end
                                DT_FE: frame_end  <= 1'b1;
                                DT_LS: begin
                                    line_start <= 1'b1;
                                    if (line_count != 16'hFFFF)
                                        line_count <= line_count + 16'd1;
                                end
                                DT_LE: line_end   <= 1'b1;
                                default: begin
                                    // Reserved short-packet DT
                                    if (error_count != 16'hFFFF)
                                        error_count <= error_count + 16'd1;
                                end
                            endcase
                            state <= S_HDR0;
                        end else begin
                            // Long packet: stream WC payload bytes next
                            data_remaining <= wc_reg;
                            state          <= (wc_reg == 16'd0) ? S_CRC0 : S_DATA;
                        end
                    end

                    // ── Payload byte ────────────────────────────────────────
                    S_DATA: begin
                        pixel_valid <= 1'b1;
                        pixel_byte  <= byte_data;
                        pixel_last  <= (data_remaining == 16'd1);
                        data_remaining <= data_remaining - 16'd1;
                        if (data_remaining == 16'd1)
                            state <= S_CRC0;
                    end

                    // ── CRC bytes (consumed, not validated v1) ──────────────
                    S_CRC0: state <= S_CRC1;
                    S_CRC1: state <= S_HDR0;

                    default: state <= S_HDR0;
                endcase
            end
        end
    end

endmodule
