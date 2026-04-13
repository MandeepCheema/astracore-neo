`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — HeadPoseTracker RTL
// =============================================================================
// Implements the DMS head-pose attention zone classifier.
//
// Accepts per-frame yaw / pitch / roll angles as signed 8-bit integers
// (degrees, range -128 to +127).
//
// Attention zone: |yaw| <= YAW_THRESH AND |pitch| <= PITCH_THRESH AND
//                 |roll| <= ROLL_THRESH
//
// A rolling window of WINDOW_SIZE frames tracks whether each frame was in zone.
// distracted_count = number of OUT-OF-ZONE frames in the current window.
//
// Interface:
//   clk              — system clock (rising edge active)
//   rst_n            — active-low synchronous reset
//   valid            — pulse high for one cycle when new pose is ready
//   yaw              — signed 8-bit yaw angle (degrees)
//   pitch            — signed 8-bit pitch angle (degrees)
//   roll             — signed 8-bit roll angle (degrees)
//   in_zone          — registered: 1 if current frame is in attention zone
//   distracted_count — number of out-of-zone frames in rolling window
// =============================================================================

module head_pose_tracker #(
    parameter YAW_THRESH   = 7'd30,    // |yaw| <= 30°
    parameter PITCH_THRESH = 7'd20,    // |pitch| <= 20°
    parameter ROLL_THRESH  = 7'd20,    // |roll| <= 20°
    parameter WINDOW_SIZE  = 15        // distraction rolling window depth
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid,
    input  wire signed [7:0] yaw,
    input  wire signed [7:0] pitch,
    input  wire signed [7:0] roll,

    output reg         in_zone,
    output wire [$clog2(WINDOW_SIZE):0] distracted_count
);

    // -------------------------------------------------------------------------
    // Absolute value (combinational)
    // -------------------------------------------------------------------------
    wire [6:0] abs_yaw   = yaw[7]   ? (-yaw[6:0])   : yaw[6:0];
    wire [6:0] abs_pitch = pitch[7] ? (-pitch[6:0]) : pitch[6:0];
    wire [6:0] abs_roll  = roll[7]  ? (-roll[6:0])  : roll[6:0];

    // -------------------------------------------------------------------------
    // Zone classification (combinational)
    // -------------------------------------------------------------------------
    wire in_zone_next = (abs_yaw   <= YAW_THRESH)   &&
                        (abs_pitch <= PITCH_THRESH) &&
                        (abs_roll  <= ROLL_THRESH);

    // -------------------------------------------------------------------------
    // Rolling window shift register
    // -------------------------------------------------------------------------
    // 1 = in zone, 0 = distracted
    reg [WINDOW_SIZE-1:0] zone_window;

    // Popcount: count OUT-OF-ZONE (0) bits = distracted frames
    // distracted = WINDOW_SIZE - count_of_in_zone_bits
    integer pw;
    reg [$clog2(WINDOW_SIZE):0] in_zone_count;
    always @(*) begin
        in_zone_count = 0;
        for (pw = 0; pw < WINDOW_SIZE; pw = pw + 1)
            in_zone_count = in_zone_count + zone_window[pw];
    end

    wire [$clog2(WINDOW_SIZE):0] dist_count =
        WINDOW_SIZE[($clog2(WINDOW_SIZE)):0] - in_zone_count;
    assign distracted_count = dist_count;

    // -------------------------------------------------------------------------
    // Sequential logic
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            in_zone     <= 1'b1;    // default: in zone at reset
            zone_window <= {WINDOW_SIZE{1'b1}};  // all in-zone initially
        end else if (valid) begin
            in_zone     <= in_zone_next;
            // Shift window left, insert new result at bit 0
            zone_window <= {zone_window[WINDOW_SIZE-2:0], in_zone_next};
        end
    end

endmodule
