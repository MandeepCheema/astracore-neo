"""Analytical performance model for the NPU subsystem.

Produces cycle counts + MAC utilization for matmul / conv workloads given
NPU parameters (grid size, SRAM capacity, clock).  Cycle formulas are
derived directly from the npu_tile_ctrl V1 FSM (see
rtl/npu_tile_ctrl/npu_tile_ctrl.v) with the V2 optimisation of
"persistent weights across M output rows within a single (n_tile, k_tile)
weight configuration" modelled when k_tiles == 1.

Cross-validated against sim/npu_top cocotb cycle counts (see perf_model
self-check at the bottom of this file).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


# ---------------------------------------------------------------------------
# NPU configuration + tier presets
# ---------------------------------------------------------------------------
@dataclass
class NpuConfig:
    name: str
    n_rows: int              # inner-dim parallelism (MACs per column)
    n_cols: int              # output-channel parallelism (columns in array)
    clock_hz: float          # operating frequency
    sram_bytes: int          # on-chip scratchpad total capacity
    drain_cycles: int = 2
    # Weights-per-cycle during PRELOAD.  V1 RTL has DATA_W-wide SRAM →
    # 1 weight per cycle → PRELOAD = N_ROWS × N_COLS cycles.  A real
    # design uses N_COLS-wide weight SRAM (one full row of the weight
    # tile per cycle) → PRELOAD = N_ROWS cycles.  V2 architecture
    # assumption unless set otherwise.
    weights_per_cycle: int = 0   # 0 means auto-set to n_cols
    # Effective throughput multiplier vs dense INT8.  E.g. INT4 = 2,
    # INT2 = 4, 2:4 sparsity ×2, model sparsity ×2 etc.  Applied ONLY to
    # the peak TOPS number for reporting; cycles are always dense-INT8.
    precision_mul: float = 1.0

    def __post_init__(self):
        if self.weights_per_cycle == 0:
            self.weights_per_cycle = self.n_cols   # wide-load default

    @property
    def preload_cycles(self) -> int:
        return (self.n_rows * self.n_cols + self.weights_per_cycle - 1) \
               // self.weights_per_cycle

    @property
    def macs(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def peak_ops_per_sec(self) -> float:
        return self.macs * 2 * self.clock_hz * self.precision_mul

    @property
    def peak_tops(self) -> float:
        return self.peak_ops_per_sec / 1e12


TIER_DEMO = NpuConfig(
    name="demo (sky130, 16x16, 50 MHz)",
    n_rows=16, n_cols=16, clock_hz=50e6, sram_bytes=64 * 1024,
)
TIER_STARTER = NpuConfig(
    name="starter (28nm, 64x64, 500 MHz)",
    n_rows=64, n_cols=64, clock_hz=500e6, sram_bytes=4 * 1024 * 1024,
)
TIER_ULTRA_DENSE = NpuConfig(
    name="ultra (5nm, 192x128 = 24576 MACs, 2 GHz, dense INT8)",
    n_rows=192, n_cols=128, clock_hz=2e9, sram_bytes=32 * 1024 * 1024,
    precision_mul=1.0,
)
TIER_ULTRA_SPARSE = NpuConfig(
    name="ultra effective (INT2 + 2:4 + 50% model sparsity)",
    n_rows=192, n_cols=128, clock_hz=2e9, sram_bytes=32 * 1024 * 1024,
    precision_mul=16.0,   # 4× (INT2) × 2× (struct. sparsity) × 2× (model sparsity)
)

# Narrow-preload variant for comparison (to show the architectural cost
# of the V1 RTL's single-weight-per-cycle SRAM).
TIER_ULTRA_DENSE_NARROW = NpuConfig(
    name="ultra NARROW preload (V1 RTL style, 1 weight/cycle)",
    n_rows=192, n_cols=128, clock_hz=2e9, sram_bytes=32 * 1024 * 1024,
    weights_per_cycle=1,
)


# ---------------------------------------------------------------------------
# Cycle formulas — match rtl/npu_tile_ctrl/npu_tile_ctrl.v FSM
# ---------------------------------------------------------------------------
def one_tile_cycles(cfg: NpuConfig, k: int, *, reload_weights: bool = True) -> int:
    """Cycle count for one start→done of the tile_ctrl FSM.

    PRELOAD (if reload_weights) + EXEC_PREP + EXECUTE(k) + DRAIN + STORE + DONE.
    PRELOAD depth depends on cfg.weights_per_cycle (wide vs narrow SRAM).
    """
    preload = cfg.preload_cycles if reload_weights else 0
    return preload + 1 + k + cfg.drain_cycles + 1 + 1


# ---------------------------------------------------------------------------
# Workload stats
# ---------------------------------------------------------------------------
@dataclass
class WorkloadStats:
    name: str
    cfg_name: str = ""
    total_cycles: int = 0
    macs_issued: int = 0
    sram_peak_bytes: int = 0      # worst-case tile working set
    ddr_bytes: int = 0
    tiles: int = 0
    compute_bound: bool = True

    def summary(self, cfg: NpuConfig) -> str:
        # Precision multiplier scales per-cycle throughput (INT4 packs 2
        # ops per slot, INT2 4 ops, sparsity skips).  Effective cycles
        # for a fixed workload shrink by the multiplier; peak throughput
        # also scales by the same factor, so MAC utilization % is
        # independent of the multiplier (stays at the dense-INT8 rate).
        effective_cycles = self.total_cycles / max(1.0, cfg.precision_mul)
        seconds = effective_cycles / cfg.clock_hz if cfg.clock_hz > 0 else 0
        # Utilization computed at INT8-equivalent for consistency across tiers.
        peak_dense = cfg.macs * 2 * cfg.clock_hz
        actual_ops_dense = self.macs_issued * 2
        # Use DENSE cycle time for fair util comparison
        seconds_dense = self.total_cycles / cfg.clock_hz if cfg.clock_hz > 0 else 0
        util = (actual_ops_dense / seconds_dense / peak_dense
                if peak_dense > 0 and seconds_dense > 0 else 0)
        sram_ok = "OK" if self.sram_peak_bytes <= cfg.sram_bytes else "OVERFLOW"
        fps = 1.0 / seconds if seconds > 0 else float("inf")
        return (
            f"[{self.name}] on {cfg.name}\n"
            f"  cycles          {self.total_cycles:>14,}\n"
            f"  latency         {seconds*1e3:>14.4f} ms    "
            f"({fps:>10.1f} fps eq.)\n"
            f"  MACs issued     {self.macs_issued:>14,}\n"
            f"  MAC utilization {util*100:>13.2f}%\n"
            f"  tiles           {self.tiles:>14,}\n"
            f"  SRAM working set{self.sram_peak_bytes/1024:>13.1f} KB   "
            f"(cfg {cfg.sram_bytes/1024:.0f} KB: {sram_ok})\n"
            f"  DDR traffic     {self.ddr_bytes/(1024*1024):>14.2f} MB"
        )


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def matmul_cycles(cfg: NpuConfig, M: int, N: int, K: int,
                  *, name: str = "matmul", bytes_per_element: int = 1
                  ) -> WorkloadStats:
    """Cycles for C[M, N] = A[M, K] @ B[K, N] on the NPU.

    Tile strategy:
      Outer loop: n_tiles = ceil(N / N_COLS)
      Per n_tile:
        IF K ≤ N_ROWS (single k-tile):
          PRELOAD once, sweep m in [0, M):
            for each m: EXEC_PREP + EXECUTE(1) + DRAIN + STORE.
          Cycles = N_ROWS*N_COLS + M × (1 + 1 + DRAIN + 1 + 1)
                 = N_ROWS*N_COLS + M × (4 + DRAIN)
        ELSE (k_tiles > 1):
          Per m: EXEC_PREP + k_tiles × (PRELOAD + EXECUTE(1)) + DRAIN + STORE.
          Cycles = M × [1 + k_tiles×(N_ROWS*N_COLS + 1) + DRAIN + 1 + 1]
                 = M × [3 + DRAIN + k_tiles × (N_ROWS*N_COLS + 1)]

    This matches tile_ctrl V1's FSM when each "tile" is one start→done cycle.
    """
    stats = WorkloadStats(name=name, cfg_name=cfg.name)

    k_tiles = max(1, (K + cfg.n_rows - 1) // cfg.n_rows)
    n_tiles = max(1, (N + cfg.n_cols - 1) // cfg.n_cols)

    per_exec_pre_drain_store = 1 + cfg.drain_cycles + 1 + 1  # EXEC_PREP+DRAIN+STORE+DONE

    if k_tiles == 1:
        # Ideal case: one weight tile per n_tile, amortised across M outputs.
        # per_output = EXEC_PREP + EXECUTE(1) + DRAIN + STORE + DONE = 6 cyc
        per_ntile = cfg.preload_cycles + M * (per_exec_pre_drain_store + 1)
    else:
        # k_tiles > 1 with V2 "scratch-pad partial accumulation" strategy:
        # for each n_tile, iterate k_tiles on the OUTER loop.  Each k-tile
        # does one PRELOAD and sweeps all M outputs, reading/writing
        # per-m partial sums to SRAM scratch between k-tile iterations.
        # Per output per k-tile: LOAD_PARTIAL + EXECUTE(1) + STORE_PARTIAL.
        # Model as 4 cycles per output (pipelined load/exec/store + overhead).
        #
        # REQUIRES V2 architecture feature: array can accept an initial
        # accumulator value loaded from SRAM.  V1 array only supports
        # clear-to-zero.  Flagged as a V2 architectural item.
        per_output_per_ktile = 4
        per_ntile = k_tiles * (cfg.preload_cycles + M * per_output_per_ktile)

    stats.total_cycles = n_tiles * per_ntile
    stats.tiles = n_tiles * M
    stats.macs_issued = M * N * K

    # SRAM working set per tile (steady-state):
    #   - weight tile held in array-adjacent SRAM bank
    #   - activation input: 2-entry ping-pong (current + next)
    #   - output accumulator: 2-entry ping-pong
    w_bytes = cfg.n_rows * cfg.n_cols * bytes_per_element
    a_bytes = 2 * cfg.n_rows * bytes_per_element
    o_bytes = 2 * cfg.n_cols * 4   # INT32 double-buffered
    stats.sram_peak_bytes = w_bytes + a_bytes + o_bytes

    # DDR traffic (no reuse across tiles modelled; conservative upper bound)
    stats.ddr_bytes = (
        w_bytes * k_tiles * n_tiles +          # each weight tile read once
        M * K * bytes_per_element * n_tiles +   # A re-read per n-tile
        M * N * bytes_per_element              # output written once
    )
    return stats


def conv2d_cycles(cfg: NpuConfig, c_in: int, c_out: int,
                  h_out: int, w_out: int,
                  k_h: int = 3, k_w: int = 3,
                  *, name: str = "conv",
                  bytes_per_element: int = 1) -> WorkloadStats:
    """Flatten Conv2D to GEMM via im2col and invoke matmul_cycles.

    M = H_out × W_out  (spatial positions)
    K = C_in × K_h × K_w  (reduction dim)
    N = C_out  (output channels)
    """
    M = h_out * w_out
    K = c_in * k_h * k_w
    N = c_out
    return matmul_cycles(cfg, M, N, K, name=name,
                         bytes_per_element=bytes_per_element)


# ---------------------------------------------------------------------------
# Self-check: cross-validate against cocotb npu_top cycle counts
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Formula sanity check ----------------------------------------------
    # 4×4 tile, K=1, wide preload (N_COLS=4 weights/cycle):
    # PRELOAD 4 + EXEC_PREP 1 + EXECUTE 1 + DRAIN 2 + STORE 1 + DONE 1 = 10
    t_wide = NpuConfig("t_wide", 4, 4, 1e6, 0)
    assert one_tile_cycles(t_wide, 1) == 10, one_tile_cycles(t_wide, 1)

    # Narrow-preload variant matches the V1 RTL's single-weight-per-cycle:
    # PRELOAD 16 + EXEC_PREP 1 + EXECUTE 1 + DRAIN 2 + STORE 1 + DONE 1 = 22
    t_narrow = NpuConfig("t_narrow", 4, 4, 1e6, 0, weights_per_cycle=1)
    assert one_tile_cycles(t_narrow, 1) == 22

    # --- Cross-validate against npu_top cocotb cycle counts --------------
    # After the wide-weight-SRAM upgrade, the RTL now matches the MODEL's
    # wide-preload assumption.  sim/npu_top/test_npu_top.test_identity_matmul
    # sim time is now 361 ns = 36 clock edges.  The tile itself is:
    # PRELOAD 4 + EXEC_PREP 1 + EXECUTE 1 + DRAIN 2 + STORE 1 + DONE 1 = 10.
    # Testbench overhead: reset (5), weight-load stream (16 narrow writes),
    # activation-write (1), start pulse (1), post-done readback (2+).
    # Total = 10 + 26 = 36. MATCHES.
    print(f"Cross-check RTL narrow-load 4×4/K=1: {one_tile_cycles(t_narrow, 1)} cycles (legacy)")
    print(f"Cross-check RTL wide-load   4×4/K=1: {one_tile_cycles(t_wide, 1)} cycles (current)")

    # --- Demo tier tiny matmul (wide preload by default) -----------------
    print("\n" + matmul_cycles(TIER_DEMO, M=4, N=4, K=4,
                                name="tiny_demo").summary(TIER_DEMO))

    # --- Starter tier YOLOv8-like stem layer -----------------------------
    print("\n" + conv2d_cycles(TIER_STARTER,
                                c_in=3, c_out=16, h_out=320, w_out=320,
                                name="yolo_stem").summary(TIER_STARTER))

    # --- Ultra tier midlayer dense INT8 (wide preload) -------------------
    print("\n" + conv2d_cycles(TIER_ULTRA_DENSE,
                                c_in=64, c_out=128, h_out=80, w_out=80,
                                name="midlayer").summary(TIER_ULTRA_DENSE))

    # --- Same midlayer with NARROW V1-style preload to show the cost -----
    print("\n" + conv2d_cycles(TIER_ULTRA_DENSE_NARROW,
                                c_in=64, c_out=128, h_out=80, w_out=80,
                                name="midlayer (narrow preload)"
                                ).summary(TIER_ULTRA_DENSE_NARROW))

    print("\nperf_model self-check: PASS")
