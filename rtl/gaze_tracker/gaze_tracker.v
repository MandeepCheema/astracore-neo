// =============================================================================
// AstraCore Neo — GazeTracker RTL
// =============================================================================
// Implements the DMS eye-state classification and PERCLOS engine.
//
// EAR (Eye Aspect Ratio) is represented as an 8-bit unsigned integer
// where 0x00 = 0.0 and 0xFF = 255/255 ≈ 1.0.
//
// Average EAR = (left_ear + right_ear) >> 1  (truncating)
//
// EyeState classification:
//   avg < EAR_CLOSED_THRESH          → CLOSED  (2'b10)
//   avg < EAR_PARTIAL_THRESH         → PARTIAL (2'b01)
//   avg >= EAR_PARTIAL_THRESH        → OPEN    (2'b00)
//
// PERCLOS:
//   A shift register of depth WINDOW_SIZE tracks whether each frame was CLOSED.
//   perclos_num = popcount(shift register) = number of closed frames in window.
//
// Blink counter:
//   Counts CLOSED → OPEN transitions.
//
// Interface:
//   clk         — system clock (rising edge active)
//   rst_n       — active-low synchronous reset
//   valid       — pulse high for one cycle when new EAR values are ready
//   left_ear    — 8-bit EAR for left eye
//   right_ear   — 8-bit EAR for right eye
//   eye_state   — 2-bit output: 00=OPEN, 01=PARTIAL, 10=CLOSED
//   avg_ear_out — 8-bit average EAR (registered)
//   perclos_num — number of CLOSED frames in current window (log2(W)+1 bits)
//   blink_count — total blink events since reset
// =============================================================================

module gaze_tracker #(
    parameter WINDOW_SIZE      = 30,    // PERCLOS rolling window depth
    parameter EAR_CLOSED_THRESH  = 8'd51, // 0.20 * 255 = 51.0
    parameter EAR_PARTIAL_THRESH = 8'd76  // 0.30 * 255 = 76.5 → 76
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid,
    input  wire [7:0]  left_ear,
    input  wire [7:0]  right_ear,

    output reg  [1:0]  eye_state,      // 00=OPEN, 01=PARTIAL, 10=CLOSED
    output reg  [7:0]  avg_ear_out,
    output wire [$clog2(WINDOW_SIZE):0] perclos_num,
    output reg  [15:0] blink_count
);

    // -------------------------------------------------------------------------
    // EAR average (combinational)
    // -------------------------------------------------------------------------
    wire [8:0] ear_sum  = {1'b0, left_ear} + {1'b0, right_ear};
    wire [7:0] avg_ear  = ear_sum[8:1];   // divide by 2 (truncate)

    // -------------------------------------------------------------------------
    // EyeState classification (combinational)
    // -------------------------------------------------------------------------
    wire is_closed  = (avg_ear < EAR_CLOSED_THRESH);
    wire is_partial = (!is_closed) && (avg_ear < EAR_PARTIAL_THRESH);
    // is_open      = (!is_closed) && (!is_partial)

    wire [1:0] next_eye_state = is_closed  ? 2'b10 :
                                is_partial ? 2'b01 :
                                             2'b00;

    // -------------------------------------------------------------------------
    // PERCLOS shift register
    // -------------------------------------------------------------------------
    reg [WINDOW_SIZE-1:0] closed_window;

    // popcount — count set bits in closed_window
    integer i;
    reg [$clog2(WINDOW_SIZE):0] pop_count;
    always @(*) begin
        pop_count = 0;
        for (i = 0; i < WINDOW_SIZE; i = i + 1)
            pop_count = pop_count + closed_window[i];
    end
    assign perclos_num = pop_count;

    // -------------------------------------------------------------------------
    // Sequential logic
    // -------------------------------------------------------------------------
    // eye_state holds the PREVIOUS frame's classification (it is updated at the
    // end of each valid cycle).  Blink detection therefore compares eye_state
    // (last frame) with next_eye_state (this frame) — no extra pipeline stage
    // is needed.

    always @(posedge clk) begin
        if (!rst_n) begin
            eye_state     <= 2'b00;
            avg_ear_out   <= 8'h00;
            closed_window <= {WINDOW_SIZE{1'b0}};
            blink_count   <= 16'h0000;
        end else if (valid) begin
            // Register outputs
            avg_ear_out <= avg_ear;
            eye_state   <= next_eye_state;

            // Shift PERCLOS window: shift left, insert new is_closed at bit 0
            closed_window <= {closed_window[WINDOW_SIZE-2:0], is_closed};

            // Blink: CLOSED (2'b10) → OPEN (2'b00)
            // eye_state is the registered PREVIOUS frame state; next_eye_state
            // is the combinational result for THIS frame.
            if (eye_state == 2'b10 && next_eye_state == 2'b00)
                blink_count <= blink_count + 1;
        end
    end

endmodule
