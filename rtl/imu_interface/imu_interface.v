`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — IMU Interface  (imu_interface.v)
// =============================================================================
// Layer 1 sensor interface.  Deserialises a SPI-framed 6-DOF IMU packet into
// accel X/Y/Z (milli-g) and gyro X/Y/Z (millidegrees/s) registers, and pulses
// imu_valid when a complete, well-formed frame has been received.
//
// ── Interface model ──────────────────────────────────────────────────────────
// The SPI physical layer (SCLK / MOSI / MISO / CS) is handled externally by a
// small synchroniser + byte-assembler that exposes:
//   spi_byte_valid   — 1-cycle pulse with the next assembled byte
//   spi_byte[7:0]    — the byte value
//   spi_frame_end    — 1-cycle pulse when CS deasserts (packet boundary)
//                      expected 1+ clocks after the last spi_byte_valid pulse.
//
// ── Frame format (13 bytes) ──────────────────────────────────────────────────
//   Byte  0      : 0x3A   header / command identifier
//   Bytes 1..2   : accel_x_mg   (signed big-endian, milli-g)
//   Bytes 3..4   : accel_y_mg
//   Bytes 5..6   : accel_z_mg
//   Bytes 7..8   : gyro_x_mdps  (signed big-endian, millideg/s)
//   Bytes 9..10  : gyro_y_mdps
//   Bytes 11..12 : gyro_z_mdps
//
//   On spi_frame_end: if 13 bytes have been received AND rx_buf[0] == 0x3A,
//   the payload is committed to the 6 output registers and imu_valid pulses
//   for 1 cycle.  Otherwise error_count increments and the frame is dropped.
//
// ── Interface ─────────────────────────────────────────────────────────────────
//   spi_byte_valid, spi_byte[7:0], spi_frame_end
//   imu_valid,
//   accel_x_mg, accel_y_mg, accel_z_mg,   (signed [15:0])
//   gyro_x_mdps, gyro_y_mdps, gyro_z_mdps,
//   frame_count, error_count              (saturating u16)
// =============================================================================

module imu_interface (
    input  wire        clk,
    input  wire        rst_n,

    // ── SPI byte stream (synchronous to clk) ─────────────────────────────────
    input  wire        spi_byte_valid,
    input  wire [7:0]  spi_byte,
    input  wire        spi_frame_end,

    // ── 6-DOF output registers ────────────────────────────────────────────────
    output reg         imu_valid,
    output reg signed [15:0] accel_x_mg,
    output reg signed [15:0] accel_y_mg,
    output reg signed [15:0] accel_z_mg,
    output reg signed [15:0] gyro_x_mdps,
    output reg signed [15:0] gyro_y_mdps,
    output reg signed [15:0] gyro_z_mdps,

    output reg  [15:0] frame_count,
    output reg  [15:0] error_count
);

    localparam [7:0] HDR_BYTE  = 8'h3A;
    localparam [3:0] FRAME_LEN = 4'd13;

    reg [7:0] rx_buf [0:12];
    reg [3:0] byte_count;

    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < 13; i = i + 1)
                rx_buf[i] <= 8'd0;
            byte_count  <= 4'd0;
            imu_valid   <= 1'b0;
            accel_x_mg  <= 16'sd0;
            accel_y_mg  <= 16'sd0;
            accel_z_mg  <= 16'sd0;
            gyro_x_mdps <= 16'sd0;
            gyro_y_mdps <= 16'sd0;
            gyro_z_mdps <= 16'sd0;
            frame_count <= 16'd0;
            error_count <= 16'd0;
        end else begin
            imu_valid <= 1'b0;   // default de-assert

            // Accumulate bytes
            if (spi_byte_valid) begin
                if (byte_count < FRAME_LEN) begin
                    rx_buf[byte_count] <= spi_byte;
                    byte_count         <= byte_count + 4'd1;
                end
                // Bytes beyond 13 silently overflow (frame will be rejected)
            end

            // Commit or reject on frame boundary
            if (spi_frame_end) begin
                if (byte_count == FRAME_LEN && rx_buf[0] == HDR_BYTE) begin
                    accel_x_mg  <= {rx_buf[1],  rx_buf[2]};
                    accel_y_mg  <= {rx_buf[3],  rx_buf[4]};
                    accel_z_mg  <= {rx_buf[5],  rx_buf[6]};
                    gyro_x_mdps <= {rx_buf[7],  rx_buf[8]};
                    gyro_y_mdps <= {rx_buf[9],  rx_buf[10]};
                    gyro_z_mdps <= {rx_buf[11], rx_buf[12]};
                    imu_valid   <= 1'b1;
                    if (frame_count != 16'hFFFF)
                        frame_count <= frame_count + 16'd1;
                end else begin
                    if (error_count != 16'hFFFF)
                        error_count <= error_count + 16'd1;
                end
                byte_count <= 4'd0;
            end
        end
    end

endmodule
