// =============================================================================
// AstraCore Neo — AWS F1 Custom Logic (CL) shell wrapper for npu_top.
//
// Wraps our `rtl/npu_top/npu_top.v` inside the AWS F1 Shell's AXI-Lite +
// AXI-4 slave/master interfaces. The Shell provides PCIe, DDR, interrupts;
// this file exposes our NPU as a memory-mapped device to the host.
//
// Register map (AXI-Lite BAR, OCL space, byte-addressable):
//
//   0x00  R/W   cfg_start             pulse to launch a tile; self-clears
//   0x04  R/W   cfg_k                 16-bit cfg_k
//   0x08  R/W   cfg_ai_base           AI SRAM start addr
//   0x0C  R/W   cfg_ao_base           AO SRAM start addr
//   0x10  R/W   cfg_afu_mode[2:0]     PASS/RELU/LEAKY/CLIP/... / SILU / ...
//   0x14  R/W   cfg_acc_init_mode     0=clear, 1=load cfg_acc_init_data
//   0x18  R/W   cfg_precision_mode    00=INT8, 01=INT4, 10=INT2, 11=FP16
//   0x1C  R     status                {30'b0, done, busy}
//   0x20  R/W   dma_start             pulse
//   0x24  R/W   dma_src_addr          DDR address (host DMAed activations)
//   0x28  R/W   dma_ai_base           AI SRAM base
//   0x2C  R/W   dma_tile_h            rows
//   0x30  R/W   dma_src_stride
//   0x34  R     dma_status            {30'b0, dma_done, dma_busy}
//   0x40  W     ext_w_waddr + we pulse
//   0x44  W     ext_w_wdata
//   0x48  W     ext_ai_waddr + we pulse
//   0x4C  W     ext_ai_wdata          (N_ROWS*DATA_W bits; use two 32b regs if > 32)
//   0x50  W     ext_ao_raddr + re pulse
//   0x54  R     ext_ao_rdata_low      (bottom 32 bits of AO word)
//   0x58  R     ext_ao_rdata_high     (top bits; for N_COLS*ACC_W > 32)
//   ...
//   0x100 R/W   cfg_acc_init_data     (multi-word, N_COLS * ACC_W bits)
//
// Note: in production we'd push activation/weight loading through the AXI
// master (DMA) rather than register pokes — but for first bring-up a
// register-pokable interface is debuggable and sufficient for small shapes.
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module cl_npu #(
    // Sized to match AWS F1 VU9P target: 64x64 production array.
    // For first-cut bring-up we stay smaller to speed synthesis.
    parameter integer N_ROWS        = 16,
    parameter integer N_COLS        = 16,
    parameter integer DATA_W        = 8,
    parameter integer ACC_W         = 32,
    parameter integer WEIGHT_DEPTH  = N_ROWS * N_COLS,
    parameter integer ACT_IN_DEPTH  = 64,
    parameter integer ACT_OUT_DEPTH = 64,
    parameter integer SCRATCH_DEPTH = 64,
    parameter integer K_W           = 16
) (
    // ----- AWS F1 Shell clocks + reset -----
    input  wire        clk_main_a0,     // 250 MHz user clock (shell default)
    input  wire        rst_main_n,

    // ----- AXI-Lite slave for OCL (host-driven register access) -----
    // (Shell provides handshake; we only implement the user-side)
    input  wire        sh_ocl_awvalid,
    input  wire [31:0] sh_ocl_awaddr,
    output wire        sh_ocl_awready,
    input  wire        sh_ocl_wvalid,
    input  wire [31:0] sh_ocl_wdata,
    input  wire  [3:0] sh_ocl_wstrb,
    output wire        sh_ocl_wready,
    output wire        sh_ocl_bvalid,
    output wire  [1:0] sh_ocl_bresp,
    input  wire        sh_ocl_bready,
    input  wire        sh_ocl_arvalid,
    input  wire [31:0] sh_ocl_araddr,
    output wire        sh_ocl_arready,
    output wire        sh_ocl_rvalid,
    output wire [31:0] sh_ocl_rdata,
    output wire  [1:0] sh_ocl_rresp,
    input  wire        sh_ocl_rready,

    // ----- AXI-4 master to DDR (activation / weight DMA) -----
    // Simplified: only read path wired; write-back handled via register-
    // pokes for first bring-up. Full AXI master lands in the next iteration.
    output wire        cl_sh_dma_pcis_awvalid,
    output wire [63:0] cl_sh_dma_pcis_awaddr,
    output wire  [7:0] cl_sh_dma_pcis_awlen,
    output wire  [2:0] cl_sh_dma_pcis_awsize,
    input  wire        cl_sh_dma_pcis_awready,
    // ... (full AXI-4 ports elided for brevity; see AWS HDK cl_dram_dma
    // template for the complete list)

    // ----- Interrupt to host -----
    output wire        cl_sh_apppf_irq_req
);

    // ---------------------------------------------------------------
    // Register file
    // ---------------------------------------------------------------
    localparam integer AI_ADDR_W = (ACT_IN_DEPTH  <= 1) ? 1 : $clog2(ACT_IN_DEPTH);
    localparam integer AO_ADDR_W = (ACT_OUT_DEPTH <= 1) ? 1 : $clog2(ACT_OUT_DEPTH);
    localparam integer W_ADDR_W  = (WEIGHT_DEPTH  <= 1) ? 1 : $clog2(WEIGHT_DEPTH);
    localparam integer AI_DATA_W = N_ROWS * DATA_W;
    localparam integer AO_DATA_W = N_COLS * ACC_W;

    reg        cfg_start_q;
    reg [K_W-1:0]              cfg_k_q;
    reg [AI_ADDR_W-1:0]        cfg_ai_base_q;
    reg [AO_ADDR_W-1:0]        cfg_ao_base_q;
    reg [2:0]                  cfg_afu_mode_q;
    reg                        cfg_acc_init_mode_q;
    reg [1:0]                  cfg_precision_mode_q;
    reg [3:0]                  cfg_mp_mode_q;
    reg [10:0]                 cfg_mp_vec_len_q;
    reg [AO_DATA_W-1:0]        cfg_acc_init_data_q;

    reg        dma_start_q;
    reg [31:0] dma_src_addr_q;
    reg [AI_ADDR_W-1:0] dma_ai_base_q;
    reg [15:0] dma_tile_h_q, dma_src_stride_q;

    reg                  ext_w_we_q;
    reg [W_ADDR_W-1:0]   ext_w_waddr_q;
    reg [DATA_W-1:0]     ext_w_wdata_q;
    reg                  ext_ai_we_q;
    reg [AI_ADDR_W-1:0]  ext_ai_waddr_q;
    reg [AI_DATA_W-1:0]  ext_ai_wdata_q;
    reg                  ext_ao_re_q;
    reg [AO_ADDR_W-1:0]  ext_ao_raddr_q;

    wire busy, done, dma_busy, dma_done;
    wire [AO_DATA_W-1:0] ext_ao_rdata;

    // Simple AXI-Lite register decoder — minimal viable implementation.
    // Production would use a full AXI-Lite state machine; this is the
    // stub that a small Python driver on the host can exercise.
    axi_lite_regfile #(
        .ADDR_W(32), .DATA_W(32)
    ) u_regfile (
        .clk(clk_main_a0), .rst_n(rst_main_n),
        // AXI-Lite slave (from shell OCL)
        .s_awvalid(sh_ocl_awvalid), .s_awaddr(sh_ocl_awaddr),
        .s_awready(sh_ocl_awready),
        .s_wvalid(sh_ocl_wvalid),   .s_wdata(sh_ocl_wdata),
        .s_wstrb(sh_ocl_wstrb),     .s_wready(sh_ocl_wready),
        .s_bvalid(sh_ocl_bvalid),   .s_bresp(sh_ocl_bresp),
        .s_bready(sh_ocl_bready),
        .s_arvalid(sh_ocl_arvalid), .s_araddr(sh_ocl_araddr),
        .s_arready(sh_ocl_arready),
        .s_rvalid(sh_ocl_rvalid),   .s_rdata(sh_ocl_rdata),
        .s_rresp(sh_ocl_rresp),     .s_rready(sh_ocl_rready),
        // Register outputs to the NPU datapath
        .cfg_start(cfg_start_q),
        .cfg_k(cfg_k_q),
        .cfg_ai_base(cfg_ai_base_q),
        .cfg_ao_base(cfg_ao_base_q),
        .cfg_afu_mode(cfg_afu_mode_q),
        .cfg_acc_init_mode(cfg_acc_init_mode_q),
        .cfg_precision_mode(cfg_precision_mode_q),
        .cfg_mp_mode(cfg_mp_mode_q),
        .cfg_mp_vec_len(cfg_mp_vec_len_q),
        .cfg_acc_init_data(cfg_acc_init_data_q),
        .dma_start(dma_start_q),
        .dma_src_addr(dma_src_addr_q),
        .dma_ai_base(dma_ai_base_q),
        .dma_tile_h(dma_tile_h_q),
        .dma_src_stride(dma_src_stride_q),
        .ext_w_we(ext_w_we_q),
        .ext_w_waddr(ext_w_waddr_q),
        .ext_w_wdata(ext_w_wdata_q),
        .ext_ai_we(ext_ai_we_q),
        .ext_ai_waddr(ext_ai_waddr_q),
        .ext_ai_wdata(ext_ai_wdata_q),
        .ext_ao_re(ext_ao_re_q),
        .ext_ao_raddr(ext_ao_raddr_q),
        // Status readbacks
        .busy(busy), .done(done),
        .dma_busy(dma_busy), .dma_done(dma_done),
        .ext_ao_rdata(ext_ao_rdata)
    );

    // ---------------------------------------------------------------
    // NPU top — the core. All our existing RTL flows through here.
    // ---------------------------------------------------------------
    npu_top #(
        .DATA_W(DATA_W), .ACC_W(ACC_W),
        .N_ROWS(N_ROWS), .N_COLS(N_COLS),
        .WEIGHT_DEPTH(WEIGHT_DEPTH),
        .ACT_IN_DEPTH(ACT_IN_DEPTH),
        .ACT_OUT_DEPTH(ACT_OUT_DEPTH),
        .SCRATCH_DEPTH(SCRATCH_DEPTH),
        .K_W(K_W)
    ) u_npu (
        .clk(clk_main_a0), .rst_n(rst_main_n),
        .start             (cfg_start_q),
        .cfg_k             (cfg_k_q),
        .cfg_ai_base       (cfg_ai_base_q),
        .cfg_ao_base       (cfg_ao_base_q),
        .cfg_afu_mode      (cfg_afu_mode_q),
        .cfg_acc_init_mode (cfg_acc_init_mode_q),
        .cfg_acc_init_data (cfg_acc_init_data_q),
        .cfg_precision_mode(cfg_precision_mode_q),
        .cfg_mp_mode       (cfg_mp_mode_q),
        .cfg_mp_vec_len    (cfg_mp_vec_len_q),
        .busy(busy), .done(done),
        .ext_w_we   (ext_w_we_q),
        .ext_w_waddr(ext_w_waddr_q),
        .ext_w_wdata(ext_w_wdata_q),
        .ext_ai_we   (ext_ai_we_q),
        .ext_ai_waddr(ext_ai_waddr_q),
        .ext_ai_wdata(ext_ai_wdata_q),
        .ext_ao_re   (ext_ao_re_q),
        .ext_ao_raddr(ext_ao_raddr_q),
        .ext_ao_rdata(ext_ao_rdata),
        .ext_sparse_skip_vec({N_ROWS{1'b0}}),   // 0 = no skip; ties off
        .dma_start     (dma_start_q),
        .dma_cfg_src_addr(dma_src_addr_q),
        .dma_cfg_ai_base (dma_ai_base_q),
        .dma_cfg_tile_h  (dma_tile_h_q),
        .dma_cfg_src_stride(dma_src_stride_q),
        .dma_busy(dma_busy), .dma_done(dma_done),
        .mem_re(), .mem_raddr(), .mem_rdata(8'h00),
        .afu_out_valid(), .afu_out_data(), .afu_out_saturated()
    );

    // Raise an interrupt when the tile completes — lets the host
    // driver wait rather than poll.
    assign cl_sh_apppf_irq_req = done;

    // TODO(F1-F2): wire cl_sh_dma_pcis_* to an AXI master that pulls
    // activation bytes from DDR through the `mem_re / mem_raddr /
    // mem_rdata` port on npu_top. For first bring-up we leave the DMA
    // channel dormant and poke weights + activations via register
    // writes.
    assign cl_sh_dma_pcis_awvalid = 1'b0;
    assign cl_sh_dma_pcis_awaddr  = 64'h0;
    assign cl_sh_dma_pcis_awlen   = 8'h0;
    assign cl_sh_dma_pcis_awsize  = 3'h0;

endmodule

`default_nettype wire
