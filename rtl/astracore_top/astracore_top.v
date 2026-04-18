`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — Top-Level Integration (FPGA target: Arty A7-35T)
// =============================================================================
// Integrates all 11 RTL subsystem modules behind an AXI4-Lite register file.
//
// AXI4-Lite slave:
//   Data width : 32 bits
//   Address    : 8-bit offset (word-aligned, byte addr >> 2)
//
// Write register map (AXI write → module input):
//   0x00  CTRL       [0]=mod_valid  [1]=sw_rst
//   0x04  GAZE       [7:0]=left_ear [15:8]=right_ear
//   0x08  THERMAL    [7:0]=temp_in
//   0x0C  CANFD      [0]=tx_success [1]=tx_error [2]=rx_error [3]=bus_off_recovery
//   0x10  ECC_LO     [31:0]=data_in[31:0]
//   0x14  ECC_HI     [31:0]=data_in[63:32]
//   0x18  ECC_CTRL   [0]=mode [15:8]=parity_in
//   0x1C  TMR_A      [31:0]=lane_a
//   0x20  TMR_B      [31:0]=lane_b
//   0x24  TMR_C      [31:0]=lane_c
//   0x28  FAULT      [15:0]=fault_value
//   0x2C  HEAD_POSE  [7:0]=yaw [15:8]=pitch [23:16]=roll
//   0x30  PCIE_CTRL  [0]=link_up [1]=link_down [4:2]=tlp_type [5]=tlp_start
//   0x34  PCIE_REQID [15:0]=req_id [23:16]=tag
//   0x38  PCIE_ADDR  [31:0]=addr
//   0x3C  PCIE_LEN   [9:0]=length_dw
//   0x40  ETH        [0]=rx_valid [1]=rx_last [9:2]=rx_byte
//   0x44  MAC        [0]=mac_valid [1]=clear [9:2]=a_byte [17:10]=b_byte
//   0x48  INF_CTRL   [0]=load_start [1]=model_valid [2]=run_start [3]=abort [4]=run_done_in
//
// Read register map (AXI read ← module output):
//   0x80  GAZE_ST    [1:0]=eye_state [7:2]=perclos_num [23:8]=blink_count
//   0x84  THERM_ST   [2:0]=therm_state [3]=throttle_en [4]=shutdown_req
//   0x88  CANFD_ST   [8:0]=tec [17:9]=rec [19:18]=bus_state
//   0x8C  ECC_ST     [0]=single_err [1]=double_err [2]=corrected [9:3]=err_pos [17:10]=parity_out
//   0x90  ECC_DLO    [31:0]=data_out[31:0]
//   0x94  ECC_DHI    [31:0]=data_out[63:32]
//   0x98  TMR_RES    [31:0]=voted
//   0x9C  TMR_ST     [0]=agreement [1]=fault_a [2]=fault_b [3]=fault_c [4]=triple_fault [6:5]=vote_count
//   0xA0  FAULT_ST   [2:0]=risk [3]=alarm [19:4]=rolling_mean
//   0xA4  HEAD_ST    [0]=in_zone [5:1]=distracted_count
//   0xA8  PCIE_ST    [2:0]=link_state [3]=pcie_busy [4]=tlp_done
//   0xAC  PCIE_H0    [31:0]=tlp_hdr[31:0]
//   0xB0  PCIE_H1    [31:0]=tlp_hdr[63:32]
//   0xB4  PCIE_H2    [31:0]=tlp_hdr[95:64]
//   0xB8  ETH_ST     [0]=frame_ok [1]=frame_err [3:2]=frame_type [5:4]=mac_type [16:6]=byte_count
//   0xBC  ETYPE      [15:0]=ethertype
//   0xC0  MAC_RES    [31:0]=mac_result (latched on ready)
//   0xC4  INF_ST     [2:0]=inf_state [3]=inf_busy [4]=session_done [5]=inf_error
//
// Status LEDs (Arty A7 on-board):
//   led[0] = thermal throttle
//   led[1] = fault alarm
//   led[2] = inference busy
//   led[3] = ethernet frame_ok (latched 1s)
// =============================================================================

module astracore_top #(
    parameter CLK_FREQ_HZ = 100_000_000   // 100 MHz Arty A7 oscillator
) (
    // ── Clock & Reset ────────────────────────────────────────────────────────
    input  wire        clk,
    input  wire        rst_n,             // active-low, from push-button or MMCM lock

    // ── AXI4-Lite Slave ──────────────────────────────────────────────────────
    // Write address channel
    input  wire [7:0]  s_axil_awaddr,
    input  wire        s_axil_awvalid,
    output reg         s_axil_awready,

    // Write data channel
    input  wire [31:0] s_axil_wdata,
    input  wire [3:0]  s_axil_wstrb,
    input  wire        s_axil_wvalid,
    output reg         s_axil_wready,

    // Write response channel
    output reg  [1:0]  s_axil_bresp,
    output reg         s_axil_bvalid,
    input  wire        s_axil_bready,

    // Read address channel
    input  wire [7:0]  s_axil_araddr,
    input  wire        s_axil_arvalid,
    output reg         s_axil_arready,

    // Read data channel
    output reg  [31:0] s_axil_rdata,
    output reg  [1:0]  s_axil_rresp,
    output reg         s_axil_rvalid,
    input  wire        s_axil_rready,

    // ── Board I/O (Arty A7) ──────────────────────────────────────────────────
    output wire [3:0]  led           // status LEDs
);

    // =========================================================================
    // Write register file
    // =========================================================================
    reg [31:0] wreg_ctrl;       // 0x00
    reg [31:0] wreg_gaze;       // 0x04
    reg [31:0] wreg_thermal;    // 0x08
    reg [31:0] wreg_canfd;      // 0x0C
    reg [31:0] wreg_ecc_lo;     // 0x10
    reg [31:0] wreg_ecc_hi;     // 0x14
    reg [31:0] wreg_ecc_ctrl;   // 0x18
    reg [31:0] wreg_tmr_a;      // 0x1C
    reg [31:0] wreg_tmr_b;      // 0x20
    reg [31:0] wreg_tmr_c;      // 0x24
    reg [31:0] wreg_fault;      // 0x28
    reg [31:0] wreg_headpose;   // 0x2C
    reg [31:0] wreg_pcie_ctrl;  // 0x30
    reg [31:0] wreg_pcie_reqid; // 0x34
    reg [31:0] wreg_pcie_addr;  // 0x38
    reg [31:0] wreg_pcie_len;   // 0x3C
    reg [31:0] wreg_eth;        // 0x40
    reg [31:0] wreg_mac;        // 0x44
    reg [31:0] wreg_inf;        // 0x48

    // =========================================================================
    // AXI4-Lite write path
    // =========================================================================
    reg [7:0] aw_addr_lat;
    reg       aw_valid_lat;

    // Write strobe application
    function [31:0] apply_strobe;
        input [31:0] old_val;
        input [31:0] new_val;
        input [3:0]  strb;
        integer i;
        begin
            apply_strobe = old_val;
            for (i = 0; i < 4; i = i + 1)
                if (strb[i]) apply_strobe[i*8 +: 8] = new_val[i*8 +: 8];
        end
    endfunction

    always @(posedge clk) begin
        if (!rst_n) begin
            s_axil_awready  <= 1'b0;
            s_axil_wready   <= 1'b0;
            s_axil_bvalid   <= 1'b0;
            s_axil_bresp    <= 2'b00;
            aw_valid_lat    <= 1'b0;
            aw_addr_lat     <= 8'h00;
            wreg_ctrl       <= 32'h0;
            wreg_gaze       <= 32'h0;
            wreg_thermal    <= 32'h0;
            wreg_canfd      <= 32'h0;
            wreg_ecc_lo     <= 32'h0;
            wreg_ecc_hi     <= 32'h0;
            wreg_ecc_ctrl   <= 32'h0;
            wreg_tmr_a      <= 32'h0;
            wreg_tmr_b      <= 32'h0;
            wreg_tmr_c      <= 32'h0;
            wreg_fault      <= 32'h0;
            wreg_headpose   <= 32'h0;
            wreg_pcie_ctrl  <= 32'h0;
            wreg_pcie_reqid <= 32'h0;
            wreg_pcie_addr  <= 32'h0;
            wreg_pcie_len   <= 32'h0;
            wreg_eth        <= 32'h0;
            wreg_mac        <= 32'h0;
            wreg_inf        <= 32'h0;
        end else begin
            // Latch write address
            s_axil_awready <= 1'b0;
            if (s_axil_awvalid && !aw_valid_lat) begin
                aw_addr_lat  <= s_axil_awaddr;
                aw_valid_lat <= 1'b1;
                s_axil_awready <= 1'b1;
            end

            // Accept write data
            s_axil_wready <= 1'b0;
            if (s_axil_wvalid && aw_valid_lat && !s_axil_bvalid) begin
                s_axil_wready <= 1'b1;
                aw_valid_lat  <= 1'b0;
                // Decode address (byte addr, word-aligned; drop low 2 bits)
                case (aw_addr_lat[7:2])
                    6'h00: wreg_ctrl       <= apply_strobe(wreg_ctrl,       s_axil_wdata, s_axil_wstrb);
                    6'h01: wreg_gaze       <= apply_strobe(wreg_gaze,       s_axil_wdata, s_axil_wstrb);
                    6'h02: wreg_thermal    <= apply_strobe(wreg_thermal,    s_axil_wdata, s_axil_wstrb);
                    6'h03: wreg_canfd      <= apply_strobe(wreg_canfd,      s_axil_wdata, s_axil_wstrb);
                    6'h04: wreg_ecc_lo     <= apply_strobe(wreg_ecc_lo,     s_axil_wdata, s_axil_wstrb);
                    6'h05: wreg_ecc_hi     <= apply_strobe(wreg_ecc_hi,     s_axil_wdata, s_axil_wstrb);
                    6'h06: wreg_ecc_ctrl   <= apply_strobe(wreg_ecc_ctrl,   s_axil_wdata, s_axil_wstrb);
                    6'h07: wreg_tmr_a      <= apply_strobe(wreg_tmr_a,      s_axil_wdata, s_axil_wstrb);
                    6'h08: wreg_tmr_b      <= apply_strobe(wreg_tmr_b,      s_axil_wdata, s_axil_wstrb);
                    6'h09: wreg_tmr_c      <= apply_strobe(wreg_tmr_c,      s_axil_wdata, s_axil_wstrb);
                    6'h0A: wreg_fault      <= apply_strobe(wreg_fault,      s_axil_wdata, s_axil_wstrb);
                    6'h0B: wreg_headpose   <= apply_strobe(wreg_headpose,   s_axil_wdata, s_axil_wstrb);
                    6'h0C: wreg_pcie_ctrl  <= apply_strobe(wreg_pcie_ctrl,  s_axil_wdata, s_axil_wstrb);
                    6'h0D: wreg_pcie_reqid <= apply_strobe(wreg_pcie_reqid, s_axil_wdata, s_axil_wstrb);
                    6'h0E: wreg_pcie_addr  <= apply_strobe(wreg_pcie_addr,  s_axil_wdata, s_axil_wstrb);
                    6'h0F: wreg_pcie_len   <= apply_strobe(wreg_pcie_len,   s_axil_wdata, s_axil_wstrb);
                    6'h10: wreg_eth        <= apply_strobe(wreg_eth,        s_axil_wdata, s_axil_wstrb);
                    6'h11: wreg_mac        <= apply_strobe(wreg_mac,        s_axil_wdata, s_axil_wstrb);
                    6'h12: wreg_inf        <= apply_strobe(wreg_inf,        s_axil_wdata, s_axil_wstrb);
                    default: ;
                endcase
                // Send write response
                s_axil_bvalid <= 1'b1;
                s_axil_bresp  <= 2'b00; // OKAY
            end

            // Clear write response
            if (s_axil_bvalid && s_axil_bready)
                s_axil_bvalid <= 1'b0;

            // Auto-clear pulse signals (mod_valid, tx_success, etc.).
            //
            // Integration-test finding: these auto-clears MUST be gated on
            // "no write to the same register happened this cycle", otherwise
            // Verilog NBA semantics make the clear override the write, and
            // pulse bits never reach '1' via AXI.  The condition below only
            // auto-clears when the current AXI transaction is NOT writing
            // to the pulse-bearing register's word-offset.  One-cycle pulse
            // lifetime is preserved because the clear still runs on every
            // subsequent cycle.
            if (!(s_axil_wvalid && aw_valid_lat && !s_axil_bvalid &&
                  aw_addr_lat[7:2] == 6'h00)) begin
                wreg_ctrl[0]      <= 1'b0;  // mod_valid
            end
            if (!(s_axil_wvalid && aw_valid_lat && !s_axil_bvalid &&
                  aw_addr_lat[7:2] == 6'h03)) begin
                wreg_canfd[3:0]   <= 4'h0;  // CAN-FD event strobes
            end
            if (!(s_axil_wvalid && aw_valid_lat && !s_axil_bvalid &&
                  aw_addr_lat[7:2] == 6'h0C)) begin
                wreg_pcie_ctrl[5] <= 1'b0;  // tlp_start
            end
            if (!(s_axil_wvalid && aw_valid_lat && !s_axil_bvalid &&
                  aw_addr_lat[7:2] == 6'h12)) begin
                wreg_inf[0]       <= 1'b0;  // load_start
                wreg_inf[2]       <= 1'b0;  // run_start
                wreg_inf[3]       <= 1'b0;  // abort
                wreg_inf[4]       <= 1'b0;  // run_done_in
            end
        end
    end

    // =========================================================================
    // AXI4-Lite read path
    // =========================================================================
    // Module output wires (declared in submodule section below)
    wire [1:0]  w_eye_state;
    wire [7:0]  w_avg_ear;
    wire [5:0]  w_perclos_num;   // $clog2(30)=5 → [5:0]
    wire [15:0] w_blink_count;
    wire [2:0]  w_therm_state;
    wire        w_throttle_en;
    wire        w_shutdown_req;
    wire [8:0]  w_tec;
    wire [7:0]  w_rec;
    wire [1:0]  w_bus_state;
    wire [63:0] w_ecc_data_out;
    wire [7:0]  w_ecc_parity_out;
    wire        w_ecc_single_err;
    wire        w_ecc_double_err;
    wire        w_ecc_corrected;
    wire [6:0]  w_ecc_err_pos;
    wire [31:0] w_tmr_voted;
    wire        w_tmr_agreement;
    wire        w_tmr_fault_a;
    wire        w_tmr_fault_b;
    wire        w_tmr_fault_c;
    wire        w_tmr_triple_fault;
    wire [1:0]  w_tmr_vote_count;
    wire [2:0]  w_fault_risk;
    wire        w_fault_alarm;
    wire [15:0] w_rolling_mean;
    wire        w_in_zone;
    wire [4:0]  w_distracted_count; // $clog2(15)=4 → [4:0]
    wire [2:0]  w_pcie_link_state;
    wire        w_pcie_busy;
    wire        w_pcie_tlp_done;
    wire [95:0] w_pcie_tlp_hdr;
    wire        w_frame_ok;
    wire        w_frame_err;
    wire [15:0] w_ethertype;
    wire [1:0]  w_frame_type;
    wire [1:0]  w_mac_type;
    wire [10:0] w_byte_count;
    wire [31:0] w_mac_result;
    wire        w_mac_ready;
    wire [2:0]  w_inf_state;
    wire        w_inf_busy;
    wire        w_inf_session_done;
    wire        w_inf_error;

    // Latch MAC result on ready
    reg [31:0] mac_result_lat;
    always @(posedge clk) begin
        if (!rst_n) mac_result_lat <= 32'h0;
        else if (w_mac_ready) mac_result_lat <= w_mac_result;
    end

    // AXI read FSM
    reg [7:0] ar_addr_lat;

    always @(posedge clk) begin
        if (!rst_n) begin
            s_axil_arready <= 1'b0;
            s_axil_rvalid  <= 1'b0;
            s_axil_rresp   <= 2'b00;
            s_axil_rdata   <= 32'h0;
            ar_addr_lat    <= 8'h0;
        end else begin
            s_axil_arready <= 1'b0;

            if (s_axil_arvalid && !s_axil_rvalid) begin
                s_axil_arready <= 1'b1;
                ar_addr_lat    <= s_axil_araddr;
                s_axil_rvalid  <= 1'b1;
                s_axil_rresp   <= 2'b00;

                case (s_axil_araddr[7:2])
                    // ── Write register readback ──────────────────────────────
                    6'h00: s_axil_rdata <= wreg_ctrl;
                    6'h01: s_axil_rdata <= wreg_gaze;
                    6'h02: s_axil_rdata <= wreg_thermal;
                    6'h03: s_axil_rdata <= wreg_canfd;
                    6'h04: s_axil_rdata <= wreg_ecc_lo;
                    6'h05: s_axil_rdata <= wreg_ecc_hi;
                    6'h06: s_axil_rdata <= wreg_ecc_ctrl;
                    6'h07: s_axil_rdata <= wreg_tmr_a;
                    6'h08: s_axil_rdata <= wreg_tmr_b;
                    6'h09: s_axil_rdata <= wreg_tmr_c;
                    6'h0A: s_axil_rdata <= wreg_fault;
                    6'h0B: s_axil_rdata <= wreg_headpose;
                    6'h0C: s_axil_rdata <= wreg_pcie_ctrl;
                    6'h0D: s_axil_rdata <= wreg_pcie_reqid;
                    6'h0E: s_axil_rdata <= wreg_pcie_addr;
                    6'h0F: s_axil_rdata <= wreg_pcie_len;
                    6'h10: s_axil_rdata <= wreg_eth;
                    6'h11: s_axil_rdata <= wreg_mac;
                    6'h12: s_axil_rdata <= wreg_inf;

                    // ── Status registers ─────────────────────────────────────
                    // 0x80 → word offset 0x20
                    6'h20: s_axil_rdata <= {8'h0, w_blink_count, w_perclos_num, w_eye_state};
                    6'h21: s_axil_rdata <= {27'h0, w_shutdown_req, w_throttle_en, w_therm_state};
                    6'h22: s_axil_rdata <= {12'h0, w_bus_state, w_rec, w_tec};
                    6'h23: s_axil_rdata <= {14'h0, w_ecc_parity_out, w_ecc_err_pos, w_ecc_corrected, w_ecc_double_err, w_ecc_single_err};
                    6'h24: s_axil_rdata <= w_ecc_data_out[31:0];
                    6'h25: s_axil_rdata <= w_ecc_data_out[63:32];
                    6'h26: s_axil_rdata <= w_tmr_voted;
                    6'h27: s_axil_rdata <= {26'h0, w_tmr_vote_count, w_tmr_triple_fault, w_tmr_fault_c, w_tmr_fault_b, w_tmr_fault_a, w_tmr_agreement};
                    6'h28: s_axil_rdata <= {12'h0, w_rolling_mean, w_fault_alarm, w_fault_risk};
                    6'h29: s_axil_rdata <= {26'h0, w_distracted_count, w_in_zone};
                    6'h2A: s_axil_rdata <= {27'h0, w_pcie_tlp_done, w_pcie_busy, w_pcie_link_state};
                    6'h2B: s_axil_rdata <= w_pcie_tlp_hdr[31:0];
                    6'h2C: s_axil_rdata <= w_pcie_tlp_hdr[63:32];
                    6'h2D: s_axil_rdata <= w_pcie_tlp_hdr[95:64];
                    6'h2E: s_axil_rdata <= {15'h0, w_byte_count, w_mac_type, w_frame_type, w_frame_err, w_frame_ok};
                    6'h2F: s_axil_rdata <= {16'h0, w_ethertype};
                    6'h30: s_axil_rdata <= mac_result_lat;
                    6'h31: s_axil_rdata <= {26'h0, w_inf_error, w_inf_session_done, w_inf_busy, w_inf_state};
                    default: s_axil_rdata <= 32'hDEAD_BEEF;
                endcase
            end

            if (s_axil_rvalid && s_axil_rready)
                s_axil_rvalid <= 1'b0;
        end
    end

    // =========================================================================
    // Submodule control signals (combinational from write registers)
    // =========================================================================
    wire mod_valid     = wreg_ctrl[0];
    wire sw_rst_n      = rst_n && !wreg_ctrl[1];

    // =========================================================================
    // 1. Gaze Tracker
    // =========================================================================
    gaze_tracker #(
        .WINDOW_SIZE      (30),
        .EAR_CLOSED_THRESH (8'd51),
        .EAR_PARTIAL_THRESH(8'd76)
    ) u_gaze (
        .clk        (clk),
        .rst_n      (sw_rst_n),
        .valid      (mod_valid),
        .left_ear   (wreg_gaze[7:0]),
        .right_ear  (wreg_gaze[15:8]),
        .eye_state  (w_eye_state),
        .avg_ear_out(w_avg_ear),
        .perclos_num(w_perclos_num),
        .blink_count(w_blink_count)
    );

    // =========================================================================
    // 2. Thermal Zone
    // =========================================================================
    thermal_zone #(
        .WARN_THRESH     (8'd75),
        .THROTTLE_THRESH (8'd85),
        .CRITICAL_THRESH (8'd95),
        .SHUTDOWN_THRESH (8'd105)
    ) u_thermal (
        .clk        (clk),
        .rst_n      (sw_rst_n),
        .valid      (mod_valid),
        .temp_in    (wreg_thermal[7:0]),
        .state      (w_therm_state),
        .throttle_en(w_throttle_en),
        .shutdown_req(w_shutdown_req)
    );

    // =========================================================================
    // 3. CAN-FD Controller
    // =========================================================================
    canfd_controller u_canfd (
        .clk             (clk),
        .rst_n           (sw_rst_n),
        .tx_success      (wreg_canfd[0]),
        .tx_error        (wreg_canfd[1]),
        .rx_error        (wreg_canfd[2]),
        .bus_off_recovery(wreg_canfd[3]),
        .tec             (w_tec),
        .rec             (w_rec),
        .bus_state       (w_bus_state),
        .rx_frame_valid  (1'b0),
        .rx_frame_id     (29'd0),
        .rx_frame_dlc    (4'd0),
        .rx_frame_data   (64'd0),
        .rx_frame_ready  (),
        .rx_out_valid    (),
        .rx_out_id       (),
        .rx_out_dlc      (),
        .rx_out_data     (),
        .rx_out_ready    (1'b0),
        .tx_frame_valid  (1'b0),
        .tx_frame_id     (29'd0),
        .tx_frame_dlc    (4'd0),
        .tx_frame_data   (64'd0),
        .tx_frame_ready  (),
        .tx_frame_done   ()
    );

    // =========================================================================
    // 4. ECC SECDED
    // =========================================================================
    ecc_secded u_ecc (
        .clk        (clk),
        .rst_n      (sw_rst_n),
        .valid      (mod_valid),
        .mode       (wreg_ecc_ctrl[0]),
        .data_in    ({wreg_ecc_hi, wreg_ecc_lo}),
        .parity_in  (wreg_ecc_ctrl[15:8]),
        .data_out   (w_ecc_data_out),
        .parity_out (w_ecc_parity_out),
        .single_err (w_ecc_single_err),
        .double_err (w_ecc_double_err),
        .corrected  (w_ecc_corrected),
        .err_pos    (w_ecc_err_pos)
    );

    // =========================================================================
    // 5. TMR Voter
    // =========================================================================
    tmr_voter u_tmr (
        .clk         (clk),
        .rst_n       (sw_rst_n),
        .valid       (mod_valid),
        .lane_a      (wreg_tmr_a),
        .lane_b      (wreg_tmr_b),
        .lane_c      (wreg_tmr_c),
        .voted       (w_tmr_voted),
        .agreement   (w_tmr_agreement),
        .fault_a     (w_tmr_fault_a),
        .fault_b     (w_tmr_fault_b),
        .fault_c     (w_tmr_fault_c),
        .triple_fault(w_tmr_triple_fault),
        .vote_count  (w_tmr_vote_count)
    );

    // =========================================================================
    // 6. Fault Predictor
    // =========================================================================
    fault_predictor #(
        .WARN_THRESH     (16'd50),
        .CRITICAL_THRESH (16'd100),
        .WINDOW_SIZE     (16),
        .SPIKE_OFFSET    (16'd30)
    ) u_fault (
        .clk         (clk),
        .rst_n       (sw_rst_n),
        .valid       (mod_valid),
        .value       (wreg_fault[15:0]),
        .risk        (w_fault_risk),
        .alarm       (w_fault_alarm),
        .rolling_mean(w_rolling_mean)
    );

    // =========================================================================
    // 7. Head Pose Tracker
    // =========================================================================
    head_pose_tracker #(
        .YAW_THRESH   (7'd30),
        .PITCH_THRESH (7'd20),
        .ROLL_THRESH  (7'd20),
        .WINDOW_SIZE  (15)
    ) u_headpose (
        .clk             (clk),
        .rst_n           (sw_rst_n),
        .valid           (mod_valid),
        .yaw             (wreg_headpose[7:0]),
        .pitch           (wreg_headpose[15:8]),
        .roll            (wreg_headpose[23:16]),
        .in_zone         (w_in_zone),
        .distracted_count(w_distracted_count)
    );

    // =========================================================================
    // 8. PCIe Controller
    // =========================================================================
    pcie_controller u_pcie (
        .clk       (clk),
        .rst_n     (sw_rst_n),
        .link_up   (wreg_pcie_ctrl[0]),
        .link_down (wreg_pcie_ctrl[1]),
        .tlp_type  (wreg_pcie_ctrl[3:2]),
        .tlp_start (wreg_pcie_ctrl[5]),
        .req_id    (wreg_pcie_reqid[15:0]),
        .tag       (wreg_pcie_reqid[23:16]),
        .addr      (wreg_pcie_addr),
        .length_dw (wreg_pcie_len[9:0]),
        .link_state(w_pcie_link_state),
        .busy      (w_pcie_busy),
        .tlp_done  (w_pcie_tlp_done),
        .tlp_hdr   (w_pcie_tlp_hdr)
    );

    // =========================================================================
    // 9. Ethernet Controller
    // =========================================================================
    ethernet_controller u_eth (
        .clk             (clk),
        .rst_n           (sw_rst_n),
        .rx_valid        (wreg_eth[0]),
        .rx_last         (wreg_eth[1]),
        .rx_byte         (wreg_eth[9:2]),
        .frame_ok        (w_frame_ok),
        .frame_err       (w_frame_err),
        .ethertype       (w_ethertype),
        .frame_type      (w_frame_type),
        .mac_type        (w_mac_type),
        .byte_count      (w_byte_count),
        .rx_payload_valid(),
        .rx_payload_byte (),
        .rx_payload_last (),
        .tx_valid        (1'b0),
        .tx_byte_in      (8'd0),
        .tx_last         (1'b0),
        .tx_ready        (),
        .tx_out_valid    (),
        .tx_out_byte     (),
        .tx_out_last     ()
    );

    // =========================================================================
    // 10. MAC Array
    // =========================================================================
    mac_array u_mac (
        .clk   (clk),
        .rst_n (sw_rst_n),
        .valid (wreg_mac[0]),
        .clear (wreg_mac[1]),
        .a     (wreg_mac[9:2]),
        .b     (wreg_mac[17:10]),
        .result(w_mac_result),
        .ready (w_mac_ready)
    );

    // =========================================================================
    // 11. Inference Runtime
    // =========================================================================
    inference_runtime u_inf (
        .clk         (clk),
        .rst_n       (sw_rst_n),
        .load_start  (wreg_inf[0]),
        .model_valid (wreg_inf[1]),
        .run_start   (wreg_inf[2]),
        .abort       (wreg_inf[3]),
        .run_done_in (wreg_inf[4]),
        .state       (w_inf_state),
        .busy        (w_inf_busy),
        .session_done(w_inf_session_done),
        .error       (w_inf_error)
    );

    // =========================================================================
    // Status LEDs
    // =========================================================================
    // led[3]: frame_ok pulse stretcher (visible on board)
    reg [23:0] led3_stretch;
    always @(posedge clk) begin
        if (!rst_n)          led3_stretch <= 24'h0;
        else if (w_frame_ok) led3_stretch <= 24'hFFFFFF;
        else if (|led3_stretch) led3_stretch <= led3_stretch - 24'h1;
    end

    assign led[0] = w_throttle_en;
    assign led[1] = w_fault_alarm;
    assign led[2] = w_inf_busy;
    assign led[3] = |led3_stretch;

endmodule
