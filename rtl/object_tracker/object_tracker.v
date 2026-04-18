`timescale 1ns/1ps
// =============================================================================
// AstraCore Neo — Object Tracker  (object_tracker.v)
// =============================================================================
// Layer 2 fusion module.  Maintains a fixed-size table of active object tracks
// that are matched to incoming detections via a spatial association gate.
// Tracks accumulate contributions from multiple sensors and are pruned after
// a configurable number of "tick" periods without an update.
//
// ── Design ───────────────────────────────────────────────────────────────────
//   Storage: NUM_TRACKS entries (default 8), each holding valid, id, x, y,
//   age, sensor_mask, class_id, confidence.
//
//   Detection pipeline (2-stage, 1 detection per clock, 2-cycle latency):
//     S1 (combinatorial → registered):
//       1. Compute |dx|, |dy| to every track (bounding-box gate, no multiply)
//       2. Priority-encode first in-gate track → match_idx
//       3. Priority-encode first invalid slot → alloc_idx
//       4. Register all results + detection fields into s1_* pipeline regs
//     S2 (registered → track update):
//       5. On s1_valid:
//          match_found  → update track[s1_match_idx] (50/50 position blend,
//                         sensor_mask |=, confidence = max, age = 0)
//                         det_matched pulses 1 cycle
//          else alloc    → allocate track[s1_alloc_idx], assign new track_id
//                          det_allocated pulses
//          else          → det_dropped pulses (table full)
//
//   Aging (tick_valid): every valid track age++.  When age reaches MAX_AGE,
//   the track is invalidated and freed for reuse.
//
//   Query interface (combinatorial): query_idx selects one track for readout.
//
//   num_active_tracks is a registered popcount of track_valid (1-cycle delay).
//
// ── Parameters ────────────────────────────────────────────────────────────────
//   NUM_TRACKS  (default 8)   — must be 8 for current priority encoders
//   GATE_MM     (default 2000) — half-width of association box in mm
//   MAX_AGE     (default 10)  — tick counts before a track is pruned
//
// ── Interface ─────────────────────────────────────────────────────────────────
//   det_valid, det_sensor_id[1:0], det_x_mm[31:0], det_y_mm[31:0],
//     det_class_id[7:0], det_confidence[7:0]          → new detection
//   tick_valid                                         → age + prune pulse
//   det_matched / det_allocated / det_dropped         → 1-cycle result pulses
//   num_active_tracks[7:0]                            → registered popcount
//   query_idx[2:0]                                    → track readout selector
//   query_valid / query_track_id / query_x_mm /
//     query_y_mm / query_age / query_sensor_mask /
//     query_class_id / query_confidence                → combinatorial fields
// =============================================================================

module object_tracker #(
    parameter integer NUM_TRACKS = 8,
    parameter signed [31:0] GATE_MM = 32'sd2000,
    parameter [7:0]         MAX_AGE = 8'd10
)(
    input  wire        clk,
    input  wire        rst_n,

    // ── Detection input ───────────────────────────────────────────────────────
    input  wire        det_valid,
    input  wire [1:0]  det_sensor_id,
    input  wire signed [31:0] det_x_mm,
    input  wire signed [31:0] det_y_mm,
    input  wire [7:0]  det_class_id,
    input  wire [7:0]  det_confidence,

    // ── Tick for age advance / pruning ────────────────────────────────────────
    input  wire        tick_valid,

    // ── Event pulses (registered, 1 cycle latency from det_valid) ─────────────
    output reg         det_matched,
    output reg         det_allocated,
    output reg         det_dropped,

    // ── Post-update sensor_mask of the affected track (aligned with pulses) ──
    // Valid for one cycle when det_matched or det_allocated asserts.  Holds the
    // track's sensor_mask AFTER the current detection has been OR'd in.  Lets
    // downstream plausibility use a true per-object accumulated mask instead of
    // a wall-clock activity heuristic.  Undefined when neither pulse fires.
    output reg  [3:0]  det_sensor_mask,

    // ── Count (combinatorial popcount of track_valid) ─────────────────────────
    output wire [7:0]  num_active_tracks,

    // ── Query interface (combinatorial) ───────────────────────────────────────
    input  wire [2:0]  query_idx,
    output wire        query_valid,
    output wire [15:0] query_track_id,
    output wire signed [31:0] query_x_mm,
    output wire signed [31:0] query_y_mm,
    output wire signed [31:0] query_vx_mm_per_update,
    output wire signed [31:0] query_vy_mm_per_update,
    output wire [7:0]  query_age,
    output wire [3:0]  query_sensor_mask,
    output wire [7:0]  query_class_id,
    output wire [7:0]  query_confidence
);

    // =========================================================================
    // 1. Track storage (flop arrays)
    // =========================================================================
    reg                track_valid       [0:NUM_TRACKS-1];
    reg [15:0]         track_id_r        [0:NUM_TRACKS-1];
    reg signed [31:0]  track_x           [0:NUM_TRACKS-1];
    reg signed [31:0]  track_y           [0:NUM_TRACKS-1];
    reg signed [31:0]  track_vx          [0:NUM_TRACKS-1];  // mm per update
    reg signed [31:0]  track_vy          [0:NUM_TRACKS-1];  // mm per update
    reg [7:0]          track_age         [0:NUM_TRACKS-1];
    reg [3:0]          track_sensor_mask [0:NUM_TRACKS-1];
    reg [7:0]          track_class_id    [0:NUM_TRACKS-1];
    reg [7:0]          track_confidence  [0:NUM_TRACKS-1];

    reg [15:0] next_id_ctr;

    // =========================================================================
    // 2. PIPELINE STAGE 1 — gate computation + priority encode (combinatorial)
    //    Registered into s1_* at the clock edge to cut the critical path.
    // =========================================================================
    wire signed [32:0] diff_x [0:NUM_TRACKS-1];
    wire signed [32:0] diff_y [0:NUM_TRACKS-1];
    wire [31:0]        abs_dx [0:NUM_TRACKS-1];
    wire [31:0]        abs_dy [0:NUM_TRACKS-1];
    wire               in_gate_raw [0:NUM_TRACKS-1];
    wire [NUM_TRACKS-1:0] in_gate_vec;
    wire [NUM_TRACKS-1:0] invalid_vec;

    genvar gi;
    generate
        for (gi = 0; gi < NUM_TRACKS; gi = gi + 1) begin : GATE
            assign diff_x[gi] = $signed({det_x_mm[31], det_x_mm}) -
                                 $signed({track_x[gi][31], track_x[gi]});
            assign diff_y[gi] = $signed({det_y_mm[31], det_y_mm}) -
                                 $signed({track_y[gi][31], track_y[gi]});
            assign abs_dx[gi] = diff_x[gi][32] ? (~diff_x[gi][31:0] + 32'd1)
                                               : diff_x[gi][31:0];
            assign abs_dy[gi] = diff_y[gi][32] ? (~diff_y[gi][31:0] + 32'd1)
                                               : diff_y[gi][31:0];
            assign in_gate_raw[gi] = track_valid[gi] &&
                                     ($signed(abs_dx[gi]) <= GATE_MM) &&
                                     ($signed(abs_dy[gi]) <= GATE_MM);
            assign in_gate_vec[gi] = in_gate_raw[gi];
            assign invalid_vec[gi] = !track_valid[gi];
        end
    endgenerate

    wire match_found_comb = |in_gate_vec;
    wire alloc_found_comb = |invalid_vec;

    wire [2:0] match_idx_comb =
        in_gate_vec[0] ? 3'd0 :
        in_gate_vec[1] ? 3'd1 :
        in_gate_vec[2] ? 3'd2 :
        in_gate_vec[3] ? 3'd3 :
        in_gate_vec[4] ? 3'd4 :
        in_gate_vec[5] ? 3'd5 :
        in_gate_vec[6] ? 3'd6 :
        3'd7;

    wire [2:0] alloc_idx_comb =
        invalid_vec[0] ? 3'd0 :
        invalid_vec[1] ? 3'd1 :
        invalid_vec[2] ? 3'd2 :
        invalid_vec[3] ? 3'd3 :
        invalid_vec[4] ? 3'd4 :
        invalid_vec[5] ? 3'd5 :
        invalid_vec[6] ? 3'd6 :
        3'd7;

    // ── S1 pipeline registers ────────────────────────────────────────────────
    reg        s1_valid;
    reg        s1_match_found;
    reg        s1_alloc_found;
    reg [2:0]  s1_match_idx;
    reg [2:0]  s1_alloc_idx;
    reg signed [31:0] s1_det_x_mm;
    reg signed [31:0] s1_det_y_mm;
    reg [1:0]  s1_det_sensor_id;
    reg [7:0]  s1_det_class_id;
    reg [7:0]  s1_det_confidence;

    always @(posedge clk) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
        end else begin
            s1_valid         <= det_valid;
            s1_match_found   <= match_found_comb;
            s1_alloc_found   <= alloc_found_comb;
            s1_match_idx     <= match_idx_comb;
            s1_alloc_idx     <= alloc_idx_comb;
            s1_det_x_mm      <= det_x_mm;
            s1_det_y_mm      <= det_y_mm;
            s1_det_sensor_id  <= det_sensor_id;
            s1_det_class_id   <= det_class_id;
            s1_det_confidence <= det_confidence;
        end
    end

    // =========================================================================
    // 3. Combinatorial popcount → num_active_tracks output
    // =========================================================================
    assign num_active_tracks =
        {7'd0, track_valid[0]} + {7'd0, track_valid[1]} +
        {7'd0, track_valid[2]} + {7'd0, track_valid[3]} +
        {7'd0, track_valid[4]} + {7'd0, track_valid[5]} +
        {7'd0, track_valid[6]} + {7'd0, track_valid[7]};

    // =========================================================================
    // 4. PIPELINE STAGE 2 — track update using registered S1 results
    //    Velocity delta is computed from S1-registered detection position
    //    vs current track position (1 cycle fresher than the gate check,
    //    which is acceptable — the gate already confirmed proximity).
    // =========================================================================
    wire signed [32:0] vel_dx = $signed({s1_det_x_mm[31], s1_det_x_mm}) -
                                 $signed({track_x[s1_match_idx][31], track_x[s1_match_idx]});
    wire signed [32:0] vel_dy = $signed({s1_det_y_mm[31], s1_det_y_mm}) -
                                 $signed({track_y[s1_match_idx][31], track_y[s1_match_idx]});

    integer k;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (k = 0; k < NUM_TRACKS; k = k + 1) begin
                track_valid[k]       <= 1'b0;
                track_id_r[k]        <= 16'd0;
                track_x[k]           <= 32'd0;
                track_y[k]           <= 32'd0;
                track_vx[k]          <= 32'd0;
                track_vy[k]          <= 32'd0;
                track_age[k]         <= 8'd0;
                track_sensor_mask[k] <= 4'd0;
                track_class_id[k]    <= 8'd0;
                track_confidence[k]  <= 8'd0;
            end
            next_id_ctr       <= 16'd1;
            det_matched       <= 1'b0;
            det_allocated     <= 1'b0;
            det_dropped       <= 1'b0;
            det_sensor_mask   <= 4'd0;
        end else begin
            det_matched   <= 1'b0;
            det_allocated <= 1'b0;
            det_dropped   <= 1'b0;

            // ── S2: detection handling from registered S1 results ───────────
            if (s1_valid) begin
                if (s1_match_found) begin
                    track_x[s1_match_idx] <=
                        ($signed(track_x[s1_match_idx]) + $signed(s1_det_x_mm)) >>> 1;
                    track_y[s1_match_idx] <=
                        ($signed(track_y[s1_match_idx]) + $signed(s1_det_y_mm)) >>> 1;
                    track_vx[s1_match_idx] <=
                        (track_vx[s1_match_idx] + vel_dx) >>> 1;
                    track_vy[s1_match_idx] <=
                        (track_vy[s1_match_idx] + vel_dy) >>> 1;
                    track_age[s1_match_idx]          <= 8'd0;
                    track_sensor_mask[s1_match_idx]  <=
                        track_sensor_mask[s1_match_idx] | (4'd1 << s1_det_sensor_id);
                    if (s1_det_confidence > track_confidence[s1_match_idx])
                        track_confidence[s1_match_idx] <= s1_det_confidence;
                    det_matched     <= 1'b1;
                    det_sensor_mask <=
                        track_sensor_mask[s1_match_idx] | (4'd1 << s1_det_sensor_id);

                end else if (s1_alloc_found) begin
                    track_valid[s1_alloc_idx]       <= 1'b1;
                    track_id_r[s1_alloc_idx]        <= next_id_ctr;
                    track_x[s1_alloc_idx]           <= s1_det_x_mm;
                    track_y[s1_alloc_idx]           <= s1_det_y_mm;
                    track_vx[s1_alloc_idx]          <= 32'sd0;
                    track_vy[s1_alloc_idx]          <= 32'sd0;
                    track_age[s1_alloc_idx]         <= 8'd0;
                    track_sensor_mask[s1_alloc_idx] <= (4'd1 << s1_det_sensor_id);
                    track_class_id[s1_alloc_idx]    <= s1_det_class_id;
                    track_confidence[s1_alloc_idx]  <= s1_det_confidence;
                    next_id_ctr                     <= next_id_ctr + 16'd1;
                    det_allocated                   <= 1'b1;
                    det_sensor_mask                 <= (4'd1 << s1_det_sensor_id);

                end else begin
                    det_dropped <= 1'b1;
                end
            end

            // ── Tick handling: age every valid track, prune at MAX_AGE ──────
            if (tick_valid) begin
                for (k = 0; k < NUM_TRACKS; k = k + 1) begin
                    if (track_valid[k]) begin
                        if (track_age[k] == MAX_AGE - 8'd1) begin
                            track_valid[k] <= 1'b0;
                            track_age[k]   <= 8'd0;
                        end else begin
                            track_age[k]   <= track_age[k] + 8'd1;
                        end
                    end
                end
            end

        end
    end

    // =========================================================================
    // 6. Query interface (pure combinatorial array read)
    // =========================================================================
    assign query_valid            = track_valid[query_idx];
    assign query_track_id         = track_id_r[query_idx];
    assign query_vx_mm_per_update = track_vx[query_idx];
    assign query_vy_mm_per_update = track_vy[query_idx];
    assign query_x_mm        = track_x[query_idx];
    assign query_y_mm        = track_y[query_idx];
    assign query_age         = track_age[query_idx];
    assign query_sensor_mask = track_sensor_mask[query_idx];
    assign query_class_id    = track_class_id[query_idx];
    assign query_confidence  = track_confidence[query_idx];

    // =========================================================================
    // Object-tracker safety invariants — Phase-F formal verification targets
    // =========================================================================
`ifndef SYNTHESIS
`ifndef __ICARUS__
    // Invariant 1: num_active_tracks never exceeds NUM_TRACKS.
    property p_num_active_bounded;
        @(posedge clk) disable iff (!rst_n)
        (num_active_tracks <= NUM_TRACKS[$clog2(NUM_TRACKS+1)-1:0]);
    endproperty
    a_num_active_bounded: assert property (p_num_active_bounded)
        else $error("object_tracker: num_active_tracks > NUM_TRACKS");

    // Invariant 2: det_allocated and det_matched are mutually exclusive
    // (a detection is either allocated to a new slot or matched to existing,
    // never both in the same cycle).
    property p_alloc_match_mutex;
        @(posedge clk) disable iff (!rst_n)
        !(det_allocated && det_matched);
    endproperty
    a_alloc_match_mutex: assert property (p_alloc_match_mutex)
        else $error("object_tracker: det_allocated and det_matched both set");

    // Invariant 3: det_dropped is mutually exclusive with alloc/match
    // (drop means nothing happened with the detection).
    property p_drop_exclusive;
        @(posedge clk) disable iff (!rst_n)
        det_dropped |-> (!det_allocated && !det_matched);
    endproperty
    a_drop_exclusive: assert property (p_drop_exclusive)
        else $error("object_tracker: det_dropped set with alloc or match");
`endif
`endif

endmodule
