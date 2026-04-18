`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — Lane Fusion  (lane_fusion.v)
// =============================================================================
// Layer 2 fusion module.  Combines camera lane detections (cam_detection_
// receiver output) with HD-map lane geometry into a robust fused lane estimate
// that feeds LDW/LKA and the plausibility_checker.
//
// ── Design ───────────────────────────────────────────────────────────────────
//   Two input streams:
//     Camera: cam_left_mm, cam_right_mm (lateral offset from ego center,
//             +ve right), cam_confidence[7:0]
//     HD Map: map_left_mm, map_right_mm  (HD map nominal lane geometry —
//             assumed high quality, no per-sample confidence)
//
//   Confidence-weighted blend:
//     w_cam = last_cam_conf >> 1        (range 0..127)
//     w_map = 128 - w_cam                (range 1..128)
//     fused_left  = (cam_left * w_cam + map_left * w_map) >>> 7
//     fused_right analogously
//     → at high cam confidence the camera dominates; at low confidence the
//       HD map carries the estimate.
//
//   Stale fallback: if one source has gone stale (no valid pulse for
//   STALE_CYCLES), its weight is forced to 0 and the other source is used
//   exclusively.  If both stale, fused_valid never pulses and outputs hold.
//
//   Derived quantities:
//     fused_lane_width_mm     = fused_right - fused_left
//     fused_center_offset_mm  = (fused_left + fused_right) >>> 1
//       → ego lateral offset relative to lane center; key input for LKA.
//
//   Pipeline latency: 2 clock cycles from cam_valid/map_valid to fused_valid.
//     Stage 1: register the 4 multiplier outputs (DSP48E1 MREG slot)
//     Stage 2: sum, descale, compute derived quantities, register output
//
//   fusion_source[1:0]:
//     2'b00 = no data (both sources stale/never seen)
//     2'b01 = map-only   (camera stale)
//     2'b10 = cam-only   (map stale)
//     2'b11 = blended    (both sources fresh, contributing per weights)
//
// ── Parameters ────────────────────────────────────────────────────────────────
//   STALE_CYCLES — stale threshold (default 500 for sim;
//                  production: 500_000 = 10ms @ 50MHz)
//
// ── Interface ─────────────────────────────────────────────────────────────────
//   cam_valid, cam_left_mm[31:0], cam_right_mm[31:0], cam_confidence[7:0]
//   map_valid, map_left_mm[31:0], map_right_mm[31:0]
//   fused_valid, fused_left_mm, fused_right_mm, fused_lane_width_mm,
//     fused_center_offset_mm, fusion_source[1:0]
//   sensor_stale[1:0]  ([0]=camera stale, [1]=HD map stale)
// =============================================================================

module lane_fusion #(
    parameter [23:0] STALE_CYCLES = 24'd500
)(
    input  wire        clk,
    input  wire        rst_n,

    // ── Camera lane detection input ───────────────────────────────────────────
    input  wire        cam_valid,
    input  wire signed [31:0] cam_left_mm,
    input  wire signed [31:0] cam_right_mm,
    input  wire [7:0]  cam_confidence,

    // ── HD map lane input ─────────────────────────────────────────────────────
    input  wire        map_valid,
    input  wire signed [31:0] map_left_mm,
    input  wire signed [31:0] map_right_mm,

    // ── Fused lane output ─────────────────────────────────────────────────────
    output reg         fused_valid,
    output reg  signed [31:0] fused_left_mm,
    output reg  signed [31:0] fused_right_mm,
    output reg  signed [31:0] fused_lane_width_mm,
    output reg  signed [31:0] fused_center_offset_mm,
    output reg  [1:0]  fusion_source,

    // ── Stale watchdog ────────────────────────────────────────────────────────
    output reg  [1:0]  sensor_stale
);

    // =========================================================================
    // 1. Latched last measurements + data-available flags
    // =========================================================================
    reg signed [31:0] last_cam_left;
    reg signed [31:0] last_cam_right;
    reg [7:0]         last_cam_conf;
    reg               cam_has_data;

    reg signed [31:0] last_map_left;
    reg signed [31:0] last_map_right;
    reg               map_has_data;

    // =========================================================================
    // 2. Stale watchdog counters
    // =========================================================================
    reg [23:0] cam_cnt;
    reg [23:0] map_cnt;

    always @(posedge clk) begin
        if (!rst_n) begin
            cam_cnt      <= 24'd0;
            map_cnt      <= 24'd0;
            sensor_stale <= 2'b00;
        end else begin
            if (cam_valid) begin
                cam_cnt        <= 24'd0;
                sensor_stale[0] <= 1'b0;
            end else begin
                if (cam_cnt < STALE_CYCLES)
                    cam_cnt <= cam_cnt + 24'd1;
                else
                    sensor_stale[0] <= 1'b1;
            end

            if (map_valid) begin
                map_cnt        <= 24'd0;
                sensor_stale[1] <= 1'b0;
            end else begin
                if (map_cnt < STALE_CYCLES)
                    map_cnt <= map_cnt + 24'd1;
                else
                    sensor_stale[1] <= 1'b1;
            end
        end
    end

    // =========================================================================
    // 3. Effective current values: use the freshly-arriving input (if valid this
    //    cycle) in preference to the latched copy.  This lets the fusion output
    //    reflect a new measurement on the same clock edge it arrives.
    // =========================================================================
    wire signed [31:0] cur_cam_left  = cam_valid ? cam_left_mm    : last_cam_left;
    wire signed [31:0] cur_cam_right = cam_valid ? cam_right_mm   : last_cam_right;
    wire [7:0]         cur_cam_conf  = cam_valid ? cam_confidence : last_cam_conf;
    wire               cur_cam_has   = cam_valid | cam_has_data;

    wire signed [31:0] cur_map_left  = map_valid ? map_left_mm    : last_map_left;
    wire signed [31:0] cur_map_right = map_valid ? map_right_mm   : last_map_right;
    wire               cur_map_has   = map_valid | map_has_data;

    // Stale after this cycle's pulse: a new valid pulse clears stale
    wire cam_eff_stale = sensor_stale[0] && !cam_valid;
    wire map_eff_stale = sensor_stale[1] && !map_valid;

    // =========================================================================
    // 4. Confidence-weighted blend weights (signed 9-bit, sum = 128)
    //    Stale handling forces all weight to the surviving source.
    // =========================================================================
    wire [7:0] base_w_cam = {1'b0, cur_cam_conf[7:1]};   // cam_conf >> 1 ∈ [0,127]

    wire [7:0] w_cam_u8 =
        (cam_eff_stale || !cur_cam_has) ? 8'd0   :
        (map_eff_stale || !cur_map_has) ? 8'd128 :
                                          base_w_cam;
    wire [7:0] w_map_u8 = 8'd128 - w_cam_u8;

    wire signed [9:0] w_cam_s = $signed({2'b00, w_cam_u8});
    wire signed [9:0] w_map_s = $signed({2'b00, w_map_u8});

    // =========================================================================
    // 5. Combinatorial blend products
    //    32b signed × 10b signed → 42b signed intermediates.
    //    These feed the stage-1 registers below, letting Vivado use the
    //    DSP48E1 MREG (post-multiply register) for fMax headroom.
    // =========================================================================
    wire signed [41:0] prod_cam_left_c  = $signed(cur_cam_left)  * w_cam_s;
    wire signed [41:0] prod_map_left_c  = $signed(cur_map_left)  * w_map_s;
    wire signed [41:0] prod_cam_right_c = $signed(cur_cam_right) * w_cam_s;
    wire signed [41:0] prod_map_right_c = $signed(cur_map_right) * w_map_s;

    // =========================================================================
    // 6. Fusion-source classification (combinatorial — rides along to stage 1)
    // =========================================================================
    wire [1:0] src_comb =
        (!cur_cam_has && !cur_map_has) ? 2'b00 :   // no data at all
        (cam_eff_stale || !cur_cam_has) ? 2'b01 :  // map-only
        (map_eff_stale || !cur_map_has) ? 2'b10 :  // cam-only
                                          2'b11;   // blended

    wire s1_fire_comb = (cam_valid || map_valid) && (cur_cam_has || cur_map_has);

    // =========================================================================
    // 7. Stage 1 — register the 4 multiplier outputs + ride-along metadata.
    //    This is where the DSP48E1 MREG lands after synthesis.
    // =========================================================================
    reg               s1_valid;
    reg signed [41:0] s1_prod_cam_left, s1_prod_map_left;
    reg signed [41:0] s1_prod_cam_right, s1_prod_map_right;
    reg [1:0]         s1_src;

    // =========================================================================
    // 8. Latch updates + stage-1 pipeline + stage-2 output register
    //    Everything happens in one always block so the reset path is atomic,
    //    but the dataflow is clearly 2 pipeline stages.
    // =========================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            last_cam_left           <= 32'd0;
            last_cam_right          <= 32'd0;
            last_cam_conf           <= 8'd0;
            cam_has_data            <= 1'b0;
            last_map_left           <= 32'd0;
            last_map_right          <= 32'd0;
            map_has_data            <= 1'b0;

            s1_valid                <= 1'b0;
            s1_prod_cam_left        <= 42'sd0;
            s1_prod_map_left        <= 42'sd0;
            s1_prod_cam_right       <= 42'sd0;
            s1_prod_map_right       <= 42'sd0;
            s1_src                  <= 2'b00;

            fused_valid             <= 1'b0;
            fused_left_mm           <= 32'd0;
            fused_right_mm          <= 32'd0;
            fused_lane_width_mm     <= 32'd0;
            fused_center_offset_mm  <= 32'd0;
            fusion_source           <= 2'b00;
        end else begin
            // Update input latches on valid pulses
            if (cam_valid) begin
                last_cam_left  <= cam_left_mm;
                last_cam_right <= cam_right_mm;
                last_cam_conf  <= cam_confidence;
                cam_has_data   <= 1'b1;
            end
            if (map_valid) begin
                last_map_left  <= map_left_mm;
                last_map_right <= map_right_mm;
                map_has_data   <= 1'b1;
            end

            // ── Stage 1: register products ──────────────────────────────────
            s1_valid          <= s1_fire_comb;
            if (s1_fire_comb) begin
                s1_prod_cam_left  <= prod_cam_left_c;
                s1_prod_map_left  <= prod_map_left_c;
                s1_prod_cam_right <= prod_cam_right_c;
                s1_prod_map_right <= prod_map_right_c;
                s1_src            <= src_comb;
            end

            // ── Stage 2: sum + descale + output register ────────────────────
            fused_valid <= s1_valid;   // 1-cycle pulse delayed one more clock
            if (s1_valid) begin
                // 43-bit signed sums; shift right by 7 to drop Q7 blend weight
                fused_left_mm  <=
                    ($signed(s1_prod_cam_left  + s1_prod_map_left)  >>> 7);
                fused_right_mm <=
                    ($signed(s1_prod_cam_right + s1_prod_map_right) >>> 7);
                fused_lane_width_mm <=
                    ($signed(s1_prod_cam_right + s1_prod_map_right) >>> 7) -
                    ($signed(s1_prod_cam_left  + s1_prod_map_left)  >>> 7);
                fused_center_offset_mm <=
                    (($signed(s1_prod_cam_right + s1_prod_map_right) >>> 7) +
                     ($signed(s1_prod_cam_left  + s1_prod_map_left)  >>> 7)) >>> 1;
                fusion_source <= s1_src;
            end
        end
    end

endmodule
