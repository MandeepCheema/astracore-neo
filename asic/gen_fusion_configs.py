#!/usr/bin/env python3
"""
AstraCore Neo — OpenLane 2 config generator for the fusion-pipeline
RTL modules.

Die-area sizing is driven by **actual sky130 post-synthesis cell areas**
from Yosys (OpenLane step 06), collected 2026-04-16.  The previous
approach (Vivado LUT count x 1.5) underestimated by 5-400x because
sky130 standard-cell multipliers are far larger than FPGA DSP48E1
primitives.

Calibration against 11 proven ASIC-passing base modules shows that
the ratio  cell_area / die_area  must stay below ~0.22 for reliable
placement.  We target <=0.18 for small designs and <=0.14 for large
ones (more routing / buffer insertion overhead).

Tiers (by synthesis area in um^2):
  tiny    (<  5 000)  -> target ratio 0.18  -> 150-200 um die
  small   (5k-15k)    -> target ratio 0.18  -> 200-300 um die
  medium  (15k-50k)   -> target ratio 0.16  -> 300-400 um die
  large   (50k+)      -> target ratio 0.14  -> 600+ um die

Run from the asic/ directory:
    python3 gen_fusion_configs.py
"""

import math
from pathlib import Path

ASIC_DIR = Path(__file__).parent

CONFIG_TEMPLATE = """\
# =============================================================================
# AstraCore Neo - {name} single-module OpenLane 2 config
# Target: sky130A | sky130_fd_sc_hd | 50 MHz
# Sky130 synthesis area: {sky130_area:.0f} um^2 ({sky130_cells} cells)
# Die: {die} x {die} um^2 (tier: {tier})
# =============================================================================

DESIGN_NAME: {name}
VERILOG_FILES: dir::../rtl/{name}/{name}.v

CLOCK_PORT: clk
CLOCK_PERIOD: 20.0   # 50 MHz

PNR_SDC_FILE: dir::scripts/base.sdc
SIGNOFF_SDC_FILE: dir::scripts/base.sdc

FP_SIZING: absolute
DIE_AREA: [0, 0, {die}, {die}]
FP_CORE_UTIL: {core_util}
FP_ASPECT_RATIO: 1

PL_TARGET_DENSITY_PCT: {density_pct}

SYNTH_STRATEGY: DELAY 0
MAX_FANOUT_CONSTRAINT: 10

RUN_CTS: true

DRT_THREADS: 4

pdk::sky130*:
  FP_PDN_VOFFSET: 5
  FP_PDN_HOFFSET: 5
  GPL_CELL_PADDING: {cell_padding}
  DPL_CELL_PADDING: {cell_padding}
"""

# (module_name, sky130_cells, sky130_area_um2) — from OpenLane Yosys
# synthesis reports (stat.json / run logs), collected 2026-04-16.
# Sorted by area ascending.
MODULES = [
    ("plausibility_checker",     213,    2394.80),
    ("safe_state_controller",    222,    2551.20),
    ("aeb_controller",           233,    2682.57),
    ("dms_fusion",               549,    6127.13),
    ("ldw_lka_controller",       640,    6353.59),
    ("mipi_csi2_rx",             582,    7041.75),
    ("det_arbiter",              813,    8769.66),
    ("ego_motion_estimator",     882,    9539.15),
    ("ptp_clock_sync",           772,   10098.44),
    ("imu_interface",            743,   10181.01),
    ("can_odometry_decoder",     927,   10409.98),
    ("gnss_interface",           943,   11564.84),
    ("sensor_sync",             1869,   19358.57),
    ("ultrasonic_interface",    1570,   20958.85),
    ("ttc_calculator",          5676,   59196.77),
    ("lidar_interface",         4620,   67638.62),
    ("radar_interface",         4977,   75595.00),
    ("cam_detection_receiver",  5681,   91133.65),
    ("lane_fusion",             9994,  106497.14),
    ("object_tracker",         11280,  132351.94),
    ("coord_transform",        22840,  247788.90),
]


def die_size(area_um2: float) -> tuple[str, int, int, int]:
    """Return (tier_name, die_um, core_util_pct, density_pct) for a given
    post-synthesis cell area.

    Calibrated against 11 proven ASIC-passing modules:
      - tmr_voter:       area/die^2 = 0.222 (tight, 150x150)
      - fault_predictor: area/die^2 = 0.217 (tight, 250x250)
      - ecc_secded:      area/die^2 = 0.152 (comfortable, 300x300)

    We target conservative ratios below the proven limit to ensure
    reliable placement with routing margin.
    """
    if area_um2 < 5_000:
        tier, ratio = "tiny", 0.18
    elif area_um2 < 15_000:
        tier, ratio = "small", 0.18
    elif area_um2 < 50_000:
        tier, ratio = "medium", 0.16
    else:
        tier, ratio = "large", 0.14

    raw = math.sqrt(area_um2 / ratio)
    die = max(150, int(math.ceil(raw / 50.0)) * 50)

    # Adjust placement parameters by tier
    if tier == "large":
        return tier, die, 40, 45
    elif tier == "medium":
        return tier, die, 42, 48
    else:
        return tier, die, 45, 50


def main() -> None:
    out_dir = ASIC_DIR
    written = 0
    total_area = 0.0

    for name, cells, area in MODULES:
        tier, die, core_util, density_pct = die_size(area)
        # Larger modules benefit from reduced cell padding to ease congestion
        cell_padding = 2 if tier == "large" else 4

        content = CONFIG_TEMPLATE.format(
            name=name,
            sky130_area=area,
            sky130_cells=cells,
            die=die,
            tier=tier,
            core_util=core_util,
            density_pct=density_pct,
            cell_padding=cell_padding,
        )
        path = out_dir / f"config_{name}.yaml"
        path.write_text(content, encoding="utf-8")
        total_area += die * die
        print(f"  [{tier:6}] {die:>5}x{die:<5}  {name:<28}  "
              f"({cells:>5} cells, {area:>10.0f} um^2)")
        written += 1

    total_mm2 = total_area / 1e6
    print(f"\nWrote {written} module configs to {out_dir}/")
    print(f"Total die area (sum of all modules): {total_mm2:.2f} mm^2")


if __name__ == "__main__":
    main()
