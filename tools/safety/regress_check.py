"""FMEDA regression check.

Runs the FMEDA tool against every module that has at least one entry
in ``tools/safety/failure_modes.yaml``, compares against a committed
baseline JSON, and exits non-zero if any module's SPFM or LFM has
dropped more than ``--max-drop-pct`` (default 1.0 percentage points)
or its PMHF has risen more than ``--max-pmhf-rise-fit`` (default
0.001 FIT).

Intended for CI: run on every PR that touches ``rtl/``,
``tools/safety/``, or ``docs/safety/``. Exit 0 = pass, 1 = regression
detected, 2 = baseline / catalog error.

Reproduce locally:
    python -m tools.safety.regress_check \\
        --baseline docs/safety/fmeda/baseline.json \\
        --asil ASIL-B

Generate a new baseline (after intentional improvement, with founder
sign-off):
    python -m tools.safety.regress_check --emit-baseline > docs/safety/fmeda/baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from tools.safety.fmeda import (
    DEFAULT_FAILURE_MODES,
    DEFAULT_MECHANISMS,
    compute_aggregate_fmeda,
    compute_fmeda,
    load_failure_modes,
    load_mechanisms,
)


AGGREGATE_KEY = "aggregate"


@dataclass(frozen=True)
class ModuleSnapshot:
    module: str
    asil_target: str
    spfm_pct: float
    lfm_pct: float
    pmhf_fit: float


def discover_modules(failure_modes_path: Path) -> List[str]:
    """List every distinct module that has at least one failure-mode row."""
    fms = load_failure_modes(failure_modes_path)
    return sorted({fm.module for fm in fms})


def snapshot_all(
    failure_modes_path: Path,
    mechanisms_path: Path,
    asil_target: str,
    *,
    include_aggregate: bool = True,
) -> List[ModuleSnapshot]:
    """Compute per-module FMEDAs, optionally with the cross-module aggregate.

    The aggregate row uses ``module = "aggregate"`` (the AGGREGATE_KEY
    constant) and is computed via ``compute_aggregate_fmeda`` over the
    per-module results. Including it in the baseline + regression-check
    gate catches *aggregate* SPFM/LFM/PMHF drift that per-module checks
    would miss (e.g. a new failure mode that's small per-module but
    significant at item level).
    """
    mechs = load_mechanisms(mechanisms_path)
    per_module_results = []
    snaps: List[ModuleSnapshot] = []
    for mod in discover_modules(failure_modes_path):
        fms = load_failure_modes(failure_modes_path, module_filter=mod)
        result = compute_fmeda(fms, mechs, module=mod, asil_target=asil_target)
        per_module_results.append(result)
        snaps.append(
            ModuleSnapshot(
                module=mod,
                asil_target=asil_target,
                spfm_pct=result.spfm_pct,
                lfm_pct=result.lfm_pct,
                pmhf_fit=result.pmhf_fit,
            )
        )
    if include_aggregate and per_module_results:
        agg = compute_aggregate_fmeda(per_module_results, asil_target=asil_target)
        snaps.append(
            ModuleSnapshot(
                module=AGGREGATE_KEY,
                asil_target=asil_target,
                spfm_pct=agg.spfm_pct,
                lfm_pct=agg.lfm_pct,
                pmhf_fit=agg.pmhf_fit,
            )
        )
    return snaps


def load_baseline(path: Path) -> Dict[str, ModuleSnapshot]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, ModuleSnapshot] = {}
    for entry in raw["modules"]:
        snap = ModuleSnapshot(
            module=entry["module"],
            asil_target=entry["asil_target"],
            spfm_pct=float(entry["spfm_pct"]),
            lfm_pct=float(entry["lfm_pct"]),
            pmhf_fit=float(entry["pmhf_fit"]),
        )
        out[snap.module] = snap
    return out


def emit_baseline_json(snapshots: List[ModuleSnapshot], asil_target: str) -> str:
    return json.dumps(
        {
            "asil_target": asil_target,
            "modules": [asdict(s) for s in snapshots],
        },
        indent=2,
    )


def compare(
    current: List[ModuleSnapshot],
    baseline: Dict[str, ModuleSnapshot],
    *,
    max_drop_pct: float,
    max_pmhf_rise_fit: float,
) -> List[str]:
    """Return a list of regression strings; empty list = pass."""
    regressions: List[str] = []
    seen = set()
    for cur in current:
        seen.add(cur.module)
        base = baseline.get(cur.module)
        if base is None:
            # New module — add to the baseline next time, but don't fail.
            continue
        spfm_drop = base.spfm_pct - cur.spfm_pct
        lfm_drop = base.lfm_pct - cur.lfm_pct
        pmhf_rise = cur.pmhf_fit - base.pmhf_fit
        if spfm_drop > max_drop_pct:
            regressions.append(
                f"{cur.module}: SPFM dropped {spfm_drop:+.2f} pp "
                f"(was {base.spfm_pct:.2f} %, now {cur.spfm_pct:.2f} %; "
                f"limit -{max_drop_pct:.2f})"
            )
        if lfm_drop > max_drop_pct:
            regressions.append(
                f"{cur.module}: LFM dropped {lfm_drop:+.2f} pp "
                f"(was {base.lfm_pct:.2f} %, now {cur.lfm_pct:.2f} %; "
                f"limit -{max_drop_pct:.2f})"
            )
        if pmhf_rise > max_pmhf_rise_fit:
            regressions.append(
                f"{cur.module}: PMHF rose {pmhf_rise:+.6f} FIT "
                f"(was {base.pmhf_fit:.6f}, now {cur.pmhf_fit:.6f}; "
                f"limit +{max_pmhf_rise_fit:.6f})"
            )
    # Modules removed from the catalog don't fail — explicit flag if needed
    return regressions


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--failure-modes",
        type=Path,
        default=DEFAULT_FAILURE_MODES,
    )
    p.add_argument("--mechanisms", type=Path, default=DEFAULT_MECHANISMS)
    p.add_argument(
        "--asil",
        choices=["QM", "ASIL-A", "ASIL-B", "ASIL-C", "ASIL-D"],
        default="ASIL-B",
        help="ASIL target used for FMEDA (does not gate metric thresholds)",
    )
    p.add_argument(
        "--baseline",
        type=Path,
        default=Path("docs/safety/fmeda/baseline.json"),
    )
    p.add_argument(
        "--max-drop-pct",
        type=float,
        default=1.0,
        help="Fail if SPFM or LFM drops more than this (percentage points)",
    )
    p.add_argument(
        "--max-pmhf-rise-fit",
        type=float,
        default=1e-3,
        help="Fail if PMHF rises more than this (FIT)",
    )
    p.add_argument(
        "--emit-baseline",
        action="store_true",
        help="Print a fresh baseline JSON to stdout (use to regenerate after intentional change)",
    )
    args = p.parse_args(argv)

    snapshots = snapshot_all(args.failure_modes, args.mechanisms, args.asil)

    if args.emit_baseline:
        print(emit_baseline_json(snapshots, args.asil))
        return 0

    if not args.baseline.exists():
        print(
            f"error: baseline not found at {args.baseline}. "
            f"Generate one with --emit-baseline.",
            file=sys.stderr,
        )
        return 2

    baseline = load_baseline(args.baseline)
    regressions = compare(
        snapshots,
        baseline,
        max_drop_pct=args.max_drop_pct,
        max_pmhf_rise_fit=args.max_pmhf_rise_fit,
    )
    if regressions:
        print("FMEDA regression detected:", file=sys.stderr)
        for r in regressions:
            print(f"  - {r}", file=sys.stderr)
        return 1
    # Plain-ASCII message: Windows cp1252 console cannot encode U+2264.
    print(
        f"FMEDA OK: {len(snapshots)} module(s) within "
        f"thresholds (drop <= {args.max_drop_pct} pp, "
        f"PMHF rise <= {args.max_pmhf_rise_fit} FIT) "
        f"vs baseline at {args.baseline}.",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
