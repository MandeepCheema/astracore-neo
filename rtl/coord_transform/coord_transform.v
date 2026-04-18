`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — Coordinate Transform  (coord_transform.v)
// =============================================================================
// Layer 2 fusion module.  Transforms 3-D object detections from each sensor's
// local coordinate frame into the vehicle body frame.
//
// ── Design ───────────────────────────────────────────────────────────────────
//   Each sensor has a static calibration: translation (x_mm, y_mm, z_mm) and
//   a heading rotation (yaw) stored as Q15 sin/cos in AXI-Lite registers.
//
//   2-stage pipelined transform (2-cycle latency):
//     Stage 1 — multiply:
//       xcos = det_x * cos_yaw   (48-bit signed Q15 product)
//       xsin = det_x * sin_yaw
//       ycos = det_y * cos_yaw
//       ysin = det_y * sin_yaw
//     Stage 2 — accumulate, Q15 descale, add offset:
//       body_x = ((xcos - ysin) >>> 15) + off_x
//       body_y = ((xsin + ycos) >>> 15) + off_y
//       body_z = det_z + off_z
//
//   Fixed-point: positions in mm (32-bit signed).  cos/sin in Q15
//   (32767 ≈ 1.0, -32768 = -1.0).  Stage-1 products: 32b × 16b → 48b.
//
// ── AXI-Lite calibration register map ────────────────────────────────────────
//   4 sensors × 5 registers = 20 words (word-addressed, [6:2] = index 0-19)
//   Sensor k base index = k * 5:
//     base+0  offset_x_mm   [31:0] signed, vehicle-frame X of sensor origin
//     base+1  offset_y_mm   [31:0] signed, vehicle-frame Y of sensor origin
//     base+2  offset_z_mm   [31:0] signed, vehicle-frame Z of sensor origin
//     base+3  cos_yaw_q15   [15:0] signed Q15 cosine of sensor heading
//     base+4  sin_yaw_q15   [15:0] signed Q15 sine of sensor heading
//
//   Default (power-on): identity transform — no rotation (cos=32767, sin=0),
//   no translation (offsets = 0).
//
// ── Interface ─────────────────────────────────────────────────────────────────
//   det_valid / det_sensor_id[1:0] / det_x_mm, det_y_mm, det_z_mm
//     → 1-cycle detection pulse with sensor-frame coordinates
//   out_valid / out_sensor_id[1:0] / out_x_mm, out_y_mm, out_z_mm
//     → result pulse 2 cycles after det_valid (pipeline latency = 2)
//   cal_we / cal_addr[6:0] / cal_wdata[31:0]
//     → AXI-Lite style calibration register write (word-addressed)
// =============================================================================

module coord_transform (
    input  wire        clk,
    input  wire        rst_n,

    // ── Detection input ───────────────────────────────────────────────────────
    input  wire        det_valid,
    input  wire [1:0]  det_sensor_id,
    input  wire signed [31:0] det_x_mm,
    input  wire signed [31:0] det_y_mm,
    input  wire signed [31:0] det_z_mm,

    // ── Transformed output ────────────────────────────────────────────────────
    output reg         out_valid,
    output reg  [1:0]  out_sensor_id,
    output reg  signed [31:0] out_x_mm,
    output reg  signed [31:0] out_y_mm,
    output reg  signed [31:0] out_z_mm,

    // ── AXI-Lite calibration register write port ─────────────────────────────
    input  wire        cal_we,          // write enable (1 cycle)
    input  wire [6:0]  cal_addr,        // word address; [6:2] = reg index 0-19
    input  wire [31:0] cal_wdata        // write data
);

    // =========================================================================
    // 1. Calibration register file (4 sensors × 5 regs = 20 words)
    //    Synth-safe: identity defaults are loaded from the reset branch rather
    //    than from an `initial` block (which is not valid for ASIC flows).
    // =========================================================================
    reg [31:0] cal_regs [0:19];

    integer ri;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (ri = 0; ri < 20; ri = ri + 1)
                cal_regs[ri] <= 32'd0;
            // cos_yaw = 1.0 in Q15 = 32767 for sensors 0-3 (reg index 3,8,13,18)
            cal_regs[3]  <= 32'd32767;
            cal_regs[8]  <= 32'd32767;
            cal_regs[13] <= 32'd32767;
            cal_regs[18] <= 32'd32767;
        end else if (cal_we) begin
            cal_regs[cal_addr[6:2]] <= cal_wdata;
        end
    end

    // =========================================================================
    // 2. Calibration parameter extraction (combinatorial)
    //    base = sensor_id * 5  (max index = 3*5+4 = 19 — fits in 5 bits)
    // =========================================================================
    wire [4:0] base = {3'b000, det_sensor_id} * 5'd5;

    wire signed [31:0] off_x   = $signed(cal_regs[base]);
    wire signed [31:0] off_y   = $signed(cal_regs[base + 5'd1]);
    wire signed [31:0] off_z   = $signed(cal_regs[base + 5'd2]);
    wire signed [15:0] cos_yaw = $signed(cal_regs[base + 5'd3][15:0]);
    wire signed [15:0] sin_yaw = $signed(cal_regs[base + 5'd4][15:0]);

    // Sign-extend cos/sin to 48 bits for correct signed multiplication
    wire signed [47:0] cos_ext = {{32{cos_yaw[15]}}, cos_yaw};
    wire signed [47:0] sin_ext = {{32{sin_yaw[15]}}, sin_yaw};

    // =========================================================================
    // 3. Stage 1 — Multiply (1-cycle latency)
    //    Compute the four rotation partial products (48-bit signed Q15).
    // =========================================================================
    reg         s1_valid;
    reg  [1:0]  s1_sensor_id;
    reg  signed [47:0] s1_xcos;
    reg  signed [47:0] s1_xsin;
    reg  signed [47:0] s1_ycos;
    reg  signed [47:0] s1_ysin;
    reg  signed [31:0] s1_z;
    reg  signed [31:0] s1_off_x;
    reg  signed [31:0] s1_off_y;
    reg  signed [31:0] s1_off_z;

    always @(posedge clk) begin
        if (!rst_n) begin
            s1_valid     <= 1'b0;
            s1_sensor_id <= 2'd0;
            s1_xcos      <= 48'd0;
            s1_xsin      <= 48'd0;
            s1_ycos      <= 48'd0;
            s1_ysin      <= 48'd0;
            s1_z         <= 32'd0;
            s1_off_x     <= 32'd0;
            s1_off_y     <= 32'd0;
            s1_off_z     <= 32'd0;
        end else begin
            s1_valid     <= det_valid;
            s1_sensor_id <= det_sensor_id;
            // Signed 32b × signed 48b: Verilog evaluates in 48-bit signed context
            // (LHS width).  Products are Q15-scaled mm values.
            s1_xcos      <= det_x_mm * cos_ext;
            s1_xsin      <= det_x_mm * sin_ext;
            s1_ycos      <= det_y_mm * cos_ext;
            s1_ysin      <= det_y_mm * sin_ext;
            s1_z         <= det_z_mm;
            s1_off_x     <= off_x;
            s1_off_y     <= off_y;
            s1_off_z     <= off_z;
        end
    end

    // =========================================================================
    // 4. Stage 2 — Accumulate + Q15 descale + offset (1-cycle latency)
    //    body_x = ((xcos - ysin) >>> 15) + off_x
    //    body_y = ((xsin + ycos) >>> 15) + off_y
    //    body_z = z + off_z
    //    Parentheses critical: >>> has lower precedence than + in Verilog.
    // =========================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            out_valid     <= 1'b0;
            out_sensor_id <= 2'd0;
            out_x_mm      <= 32'd0;
            out_y_mm      <= 32'd0;
            out_z_mm      <= 32'd0;
        end else begin
            out_valid     <= s1_valid;
            out_sensor_id <= s1_sensor_id;
            out_x_mm      <= ($signed(s1_xcos - s1_ysin) >>> 15) + s1_off_x;
            out_y_mm      <= ($signed(s1_xsin + s1_ycos) >>> 15) + s1_off_y;
            out_z_mm      <= s1_z + s1_off_z;
        end
    end

endmodule
