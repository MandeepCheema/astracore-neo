// =============================================================================
// Simple AXI-Lite register file for the NPU CL.
// Decodes OCL BAR writes into the NPU's configuration ports, routes reads
// back. Minimum-viable implementation — AWS Shell already handles
// decode / strobe at the BAR boundary; we handle only the transaction
// state machine here.
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module axi_lite_regfile #(
    parameter integer ADDR_W = 32,
    parameter integer DATA_W = 32,
    parameter integer N_ROWS = 16,
    parameter integer N_COLS = 16,
    parameter integer DATA_B = 8,
    parameter integer ACC_B  = 32,
    parameter integer K_W    = 16
) (
    input  wire                 clk,
    input  wire                 rst_n,

    // AXI-Lite slave (from AWS Shell OCL BAR)
    input  wire                 s_awvalid,
    input  wire [ADDR_W-1:0]    s_awaddr,
    output reg                  s_awready,
    input  wire                 s_wvalid,
    input  wire [DATA_W-1:0]    s_wdata,
    input  wire [DATA_W/8-1:0]  s_wstrb,
    output reg                  s_wready,
    output reg                  s_bvalid,
    output reg  [1:0]           s_bresp,
    input  wire                 s_bready,
    input  wire                 s_arvalid,
    input  wire [ADDR_W-1:0]    s_araddr,
    output reg                  s_arready,
    output reg                  s_rvalid,
    output reg  [DATA_W-1:0]    s_rdata,
    output reg  [1:0]           s_rresp,
    input  wire                 s_rready,

    // Register outputs to NPU datapath — pulsed OR held depending on
    // the register's nature (see reg-map comments in cl_npu.sv).
    output reg                  cfg_start,
    output reg [K_W-1:0]        cfg_k,
    output reg [15:0]           cfg_ai_base,
    output reg [15:0]           cfg_ao_base,
    output reg [2:0]            cfg_afu_mode,
    output reg                  cfg_acc_init_mode,
    output reg [1:0]            cfg_precision_mode,
    output reg [3:0]            cfg_mp_mode,
    output reg [10:0]           cfg_mp_vec_len,
    output reg [N_COLS*ACC_B-1:0] cfg_acc_init_data,
    output reg                  dma_start,
    output reg [31:0]           dma_src_addr,
    output reg [15:0]           dma_ai_base,
    output reg [15:0]           dma_tile_h,
    output reg [15:0]           dma_src_stride,
    output reg                  ext_w_we,
    output reg [15:0]           ext_w_waddr,
    output reg [DATA_B-1:0]     ext_w_wdata,
    output reg                  ext_ai_we,
    output reg [15:0]           ext_ai_waddr,
    output reg [N_ROWS*DATA_B-1:0] ext_ai_wdata,
    output reg                  ext_ao_re,
    output reg [15:0]           ext_ao_raddr,

    // Status from NPU
    input  wire                 busy,
    input  wire                 done,
    input  wire                 dma_busy,
    input  wire                 dma_done,
    input  wire [N_COLS*ACC_B-1:0] ext_ao_rdata
);

    // ------------------- Write channel state machine -------------------
    reg [ADDR_W-1:0] awaddr_q;
    reg              aw_latched;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_awready   <= 1'b1;
            s_wready    <= 1'b1;
            s_bvalid    <= 1'b0;
            s_bresp     <= 2'b00;
            aw_latched  <= 1'b0;
            // Pulse-style signals deassert each cycle.
            cfg_start   <= 1'b0;
            dma_start   <= 1'b0;
            ext_w_we    <= 1'b0;
            ext_ai_we   <= 1'b0;
            ext_ao_re   <= 1'b0;
            // Held-value registers reset to sane defaults.
            cfg_k              <= {K_W{1'b0}};
            cfg_ai_base        <= 16'h0;
            cfg_ao_base        <= 16'h0;
            cfg_afu_mode       <= 3'b000;
            cfg_acc_init_mode  <= 1'b0;
            cfg_precision_mode <= 2'b00;
            cfg_mp_mode        <= 4'b0000;
            cfg_mp_vec_len     <= 11'h0;
            cfg_acc_init_data  <= {N_COLS*ACC_B{1'b0}};
            dma_src_addr       <= 32'h0;
            dma_ai_base        <= 16'h0;
            dma_tile_h         <= 16'h0;
            dma_src_stride     <= 16'h0;
            ext_w_waddr        <= 16'h0;
            ext_w_wdata        <= {DATA_B{1'b0}};
            ext_ai_waddr       <= 16'h0;
            ext_ai_wdata       <= {N_ROWS*DATA_B{1'b0}};
            ext_ao_raddr       <= 16'h0;
        end else begin
            // Default: pulse signals deassert every cycle.
            cfg_start <= 1'b0;
            dma_start <= 1'b0;
            ext_w_we  <= 1'b0;
            ext_ai_we <= 1'b0;
            ext_ao_re <= 1'b0;

            // Latch address when AW handshake lands.
            if (s_awvalid && s_awready) begin
                awaddr_q   <= s_awaddr;
                aw_latched <= 1'b1;
                s_awready  <= 1'b0;
            end

            // On W handshake, dispatch to the register.
            if (aw_latched && s_wvalid && s_wready) begin
                case (awaddr_q[11:0])
                    12'h000: cfg_start          <= s_wdata[0];           // pulse
                    12'h004: cfg_k              <= s_wdata[K_W-1:0];
                    12'h008: cfg_ai_base        <= s_wdata[15:0];
                    12'h00C: cfg_ao_base        <= s_wdata[15:0];
                    12'h010: cfg_afu_mode       <= s_wdata[2:0];
                    12'h014: cfg_acc_init_mode  <= s_wdata[0];
                    12'h018: cfg_precision_mode <= s_wdata[1:0];
                    12'h01C: {cfg_mp_vec_len, cfg_mp_mode}
                                                 <= {s_wdata[14:4], s_wdata[3:0]};
                    12'h020: dma_start          <= s_wdata[0];           // pulse
                    12'h024: dma_src_addr       <= s_wdata;
                    12'h028: dma_ai_base        <= s_wdata[15:0];
                    12'h02C: dma_tile_h         <= s_wdata[15:0];
                    12'h030: dma_src_stride     <= s_wdata[15:0];
                    12'h040: begin
                        ext_w_waddr <= s_wdata[15:0];
                        ext_w_we    <= 1'b1;                              // pulse
                    end
                    12'h044: ext_w_wdata <= s_wdata[DATA_B-1:0];
                    12'h048: begin
                        ext_ai_waddr <= s_wdata[15:0];
                        ext_ai_we    <= 1'b1;                             // pulse
                    end
                    // AI data is N_ROWS*DATA_B bits; host writes 32-bit
                    // chunks starting at 0x04C, 0x050, ... MSB-first.
                    12'h04C: ext_ai_wdata[DATA_W-1:0] <= s_wdata;
                    12'h050: begin
                        ext_ao_raddr <= s_wdata[15:0];
                        ext_ao_re    <= 1'b1;                             // pulse
                    end
                    // cfg_acc_init_data window: 0x100..0x17C. Writes land
                    // in 32-bit words; word-index = awaddr_q[7:2].
                    // Supports up to 16 words = 512 bits of acc_init_data
                    // (enough for N_COLS*ACC_B up to 512 = 16 cols × 32b).
                    default: begin
                        if (awaddr_q[11:8] == 4'h1 &&
                            awaddr_q[7:6] == 2'b00) begin
                            // (awaddr_q[5:2] * 32) +: 32  — stride is
                            // 32 bits per AXI-Lite word, not DATA_W.
                            case (awaddr_q[5:2])
                                4'd0:  cfg_acc_init_data[ 31:  0] <= s_wdata;
                                4'd1:  cfg_acc_init_data[ 63: 32] <= s_wdata;
                                4'd2:  cfg_acc_init_data[ 95: 64] <= s_wdata;
                                4'd3:  cfg_acc_init_data[127: 96] <= s_wdata;
                                4'd4:  cfg_acc_init_data[159:128] <= s_wdata;
                                4'd5:  cfg_acc_init_data[191:160] <= s_wdata;
                                4'd6:  cfg_acc_init_data[223:192] <= s_wdata;
                                4'd7:  cfg_acc_init_data[255:224] <= s_wdata;
                                4'd8:  cfg_acc_init_data[287:256] <= s_wdata;
                                4'd9:  cfg_acc_init_data[319:288] <= s_wdata;
                                4'd10: cfg_acc_init_data[351:320] <= s_wdata;
                                4'd11: cfg_acc_init_data[383:352] <= s_wdata;
                                4'd12: cfg_acc_init_data[415:384] <= s_wdata;
                                4'd13: cfg_acc_init_data[447:416] <= s_wdata;
                                4'd14: cfg_acc_init_data[479:448] <= s_wdata;
                                4'd15: cfg_acc_init_data[511:480] <= s_wdata;
                            endcase
                        end
                    end
                endcase
                s_wready    <= 1'b0;
                s_bvalid    <= 1'b1;
                s_bresp     <= 2'b00;
                aw_latched  <= 1'b0;
            end

            // B handshake clears response, re-arms channel.
            if (s_bvalid && s_bready) begin
                s_bvalid  <= 1'b0;
                s_awready <= 1'b1;
                s_wready  <= 1'b1;
            end
        end
    end

    // ------------------- Read channel state machine -------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_arready <= 1'b1;
            s_rvalid  <= 1'b0;
            s_rdata   <= {DATA_W{1'b0}};
            s_rresp   <= 2'b00;
        end else begin
            if (s_arvalid && s_arready) begin
                s_arready <= 1'b0;
                s_rvalid  <= 1'b1;
                s_rresp   <= 2'b00;
                case (s_araddr[11:0])
                    12'h01C: s_rdata <= {30'b0, done, busy};
                    12'h034: s_rdata <= {30'b0, dma_done, dma_busy};
                    12'h054: s_rdata <= ext_ao_rdata[31:0];
                    12'h058: s_rdata <= (N_COLS*ACC_B > 32)
                                        ? ext_ao_rdata[63:32] : 32'h0;
                    12'h05C: s_rdata <= (N_COLS*ACC_B > 64)
                                        ? ext_ao_rdata[95:64] : 32'h0;
                    12'h060: s_rdata <= (N_COLS*ACC_B > 96)
                                        ? ext_ao_rdata[127:96] : 32'h0;
                    // Magic: 0xFF0 returns "ASTR" as device-ID probe.
                    12'hFF0: s_rdata <= 32'h41535452;   // "ASTR"
                    default: s_rdata <= 32'hDEADBEEF;
                endcase
            end else if (s_rvalid && s_rready) begin
                s_rvalid  <= 1'b0;
                s_arready <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
