"""FMEDA — Failure Modes, Effects, and Diagnostic Analysis.

Implements the quantitative analysis required by ISO 26262-5 §7.4.5,
§8.4.1-§8.4.4, with the semiconductor refinements from ISO 26262-11
§4.6 and §7.

Inputs
------
``failure_modes.yaml``
    Per-module catalog of failure modes. Each entry declares a base
    failure rate (FIT), a distribution across modes, a classification
    (safe / dangerous / no-effect), and the safety mechanism (if any)
    that covers it.

``safety_mechanisms.yaml``
    Catalog of safety mechanisms with their declared diagnostic
    coverage (DC). Coverage values here are *targets* until validated
    by the fault-injection campaign — see ``docs/safety/fault_injection/``.

Outputs
-------
A markdown report at the requested path. The report contains:

* The per-failure-mode table (the FMEDA spreadsheet equivalent).
* Resulting λ values per ISO 26262-5 §8.4 (S, DD, DU, LF, DPF).
* Aggregate Single-Point Fault Metric (SPFM), Latent-Fault Metric
  (LFM), and Probabilistic Metric for random Hardware Failures
  (PMHF).
* Pass / fail against the configured ASIL target (defaults to ASIL-B).

CLI
---
::

    python -m tools.safety.fmeda \\
        --module dms_fusion \\
        --output docs/safety/fmeda/dms_fusion_fmeda.md

Limitations (v0.1)
------------------
* Failure-rate baselines are placeholder values sourced from IEC 62380
  and SN29500 norms scaled to a 7 nm digital-logic assumption. Real
  numbers come from the licensee's silicon characterization and the
  AstraCore SER analysis (Track 2 W10 deliverable).
* Dual-point fault rate (λ_DPF) uses the ISO 26262-5 §8.4.4 simplified
  formula with a 1-year service-interval assumption. The full formula
  is licensee-specific (depends on operational profile).
* Common-cause failures are not modelled here; see the W14-W15 CCF
  analysis (separate deliverable).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore[import-untyped]
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required (pip install pyyaml). "
        "Already declared in pyproject.toml."
    ) from e


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FAILURE_MODES = REPO_ROOT / "tools" / "safety" / "failure_modes.yaml"
DEFAULT_MECHANISMS = REPO_ROOT / "tools" / "safety" / "safety_mechanisms.yaml"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

CLASSIFICATIONS = {"safe", "dangerous", "no-effect"}


@dataclass(frozen=True)
class SafetyMechanism:
    """A safety mechanism with a declared diagnostic coverage target."""

    id: str
    name: str
    description: str
    target_dc_pct: float  # 0..100
    target_dc_lf_pct: float  # latent-fault DC, 0..100
    applies_to: Tuple[str, ...]  # failure-mode IDs this mechanism covers

    def __post_init__(self) -> None:
        if not 0.0 <= self.target_dc_pct <= 100.0:
            raise ValueError(
                f"mechanism {self.id}: target_dc_pct out of range [0,100]: "
                f"{self.target_dc_pct}"
            )
        if not 0.0 <= self.target_dc_lf_pct <= 100.0:
            raise ValueError(
                f"mechanism {self.id}: target_dc_lf_pct out of range [0,100]: "
                f"{self.target_dc_lf_pct}"
            )


@dataclass(frozen=True)
class FailureMode:
    """A single failure mode of a sub-element of a module.

    Attributes
    ----------
    id : str
        Stable identifier, e.g. ``dms_fusion.wdog_cnt.seu``.
    module : str
        RTL module name (e.g. ``dms_fusion``).
    subpart : str
        Sub-element within the module (e.g. ``wdog_cnt`` flip-flop).
    mode : str
        Human-readable failure mode (e.g. ``single-event upset on
        watchdog counter``).
    lambda_fit : float
        Base failure rate in FIT (failures per 1e9 hours) for the
        sub-part. The ``distribution`` then apportions this across
        the named failure mode.
    distribution : float
        Fraction of ``lambda_fit`` attributable to *this* failure mode.
        Sum of distributions across all failure modes of one sub-part
        must be ≤ 1.0; the remainder is "no failure mode considered"
        (typically benign).
    classification : str
        ``"safe"`` (cannot cause hazard), ``"dangerous"`` (can violate
        a safety goal), or ``"no-effect"`` (does not affect output).
    mechanism_id : str | None
        ID of the covering safety mechanism, or ``None`` if uncovered.
    rationale : str
        One-line justification of classification + mechanism choice.
    """

    id: str
    module: str
    subpart: str
    mode: str
    lambda_fit: float
    distribution: float
    classification: str
    mechanism_id: Optional[str]
    rationale: str

    def __post_init__(self) -> None:
        if self.classification not in CLASSIFICATIONS:
            raise ValueError(
                f"failure mode {self.id}: classification "
                f"'{self.classification}' not in {CLASSIFICATIONS}"
            )
        if self.lambda_fit < 0:
            raise ValueError(f"failure mode {self.id}: negative lambda_fit")
        if not 0.0 <= self.distribution <= 1.0:
            raise ValueError(
                f"failure mode {self.id}: distribution out of [0,1]: "
                f"{self.distribution}"
            )

    @property
    def lambda_mode(self) -> float:
        """Failure rate apportioned to this specific mode (FIT)."""
        return self.lambda_fit * self.distribution


@dataclass
class ModeResult:
    """Per-mode FMEDA computation result."""

    mode: FailureMode
    mechanism: Optional[SafetyMechanism]
    lambda_mode: float
    lambda_safe: float          # λ_S
    lambda_dd: float            # dangerous-detected
    lambda_du: float            # dangerous-undetected (single-point fault)
    lambda_lf: float            # latent-fault residual

    @property
    def dc_applied_pct(self) -> float:
        if self.mechanism is None:
            return 0.0
        return self.mechanism.target_dc_pct


@dataclass
class FmedaResult:
    """Aggregate FMEDA result for a module (or set of modules)."""

    module: str
    asil_target: str
    per_mode: List[ModeResult]

    # Sums (FIT)
    lambda_total: float
    lambda_safe_total: float
    lambda_dangerous_total: float
    lambda_dd_total: float
    lambda_du_total: float
    lambda_lf_total: float
    lambda_dpf_total: float  # estimated dual-point fault rate

    # Metrics (per ISO 26262-5 §8.4)
    spfm_pct: float       # 1 - λ_DU / λ_dangerous
    lfm_pct: float        # 1 - λ_LF / (λ_dangerous - λ_DU)
    pmhf_fit: float       # λ_DU + 0.5 * λ_DPF (simplified)

    asil_target_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def passes_asil_target(self) -> bool:
        spfm_target = self.asil_target_metrics.get("spfm_min_pct", 0.0)
        lfm_target = self.asil_target_metrics.get("lfm_min_pct", 0.0)
        pmhf_target = self.asil_target_metrics.get("pmhf_max_fit", math.inf)
        return (
            self.spfm_pct >= spfm_target
            and self.lfm_pct >= lfm_target
            and self.pmhf_fit <= pmhf_target
        )


# ASIL target metrics per ISO 26262-5 Annex C, Tables 4 + 5 + 6.
ASIL_TARGETS: Dict[str, Dict[str, float]] = {
    "QM": {"spfm_min_pct": 0.0, "lfm_min_pct": 0.0, "pmhf_max_fit": math.inf},
    "ASIL-A": {"spfm_min_pct": 0.0, "lfm_min_pct": 0.0, "pmhf_max_fit": 1000.0},
    "ASIL-B": {"spfm_min_pct": 90.0, "lfm_min_pct": 60.0, "pmhf_max_fit": 100.0},
    "ASIL-C": {"spfm_min_pct": 97.0, "lfm_min_pct": 80.0, "pmhf_max_fit": 100.0},
    "ASIL-D": {"spfm_min_pct": 99.0, "lfm_min_pct": 90.0, "pmhf_max_fit": 10.0},
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_mechanisms(path: Path) -> Dict[str, SafetyMechanism]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    mechanisms: Dict[str, SafetyMechanism] = {}
    for entry in raw.get("mechanisms", []):
        sm = SafetyMechanism(
            id=entry["id"],
            name=entry["name"],
            description=entry["description"],
            target_dc_pct=float(entry["target_dc_pct"]),
            target_dc_lf_pct=float(entry.get("target_dc_lf_pct", entry["target_dc_pct"])),
            applies_to=tuple(entry.get("applies_to", [])),
        )
        if sm.id in mechanisms:
            raise ValueError(f"duplicate mechanism id: {sm.id}")
        mechanisms[sm.id] = sm
    return mechanisms


def load_failure_modes(path: Path, module_filter: Optional[str] = None) -> List[FailureMode]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    out: List[FailureMode] = []
    for entry in raw.get("failure_modes", []):
        if module_filter is not None and entry["module"] != module_filter:
            continue
        fm = FailureMode(
            id=entry["id"],
            module=entry["module"],
            subpart=entry["subpart"],
            mode=entry["mode"],
            lambda_fit=float(entry["lambda_fit"]),
            distribution=float(entry["distribution"]),
            classification=entry["classification"],
            mechanism_id=entry.get("mechanism_id"),
            rationale=entry.get("rationale", ""),
        )
        out.append(fm)
    return out


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def compute_mode(
    fm: FailureMode,
    mechanisms: Dict[str, SafetyMechanism],
) -> ModeResult:
    mech = mechanisms.get(fm.mechanism_id) if fm.mechanism_id else None
    lam = fm.lambda_mode
    if fm.classification == "safe" or fm.classification == "no-effect":
        return ModeResult(
            mode=fm,
            mechanism=mech,
            lambda_mode=lam,
            lambda_safe=lam,
            lambda_dd=0.0,
            lambda_du=0.0,
            lambda_lf=0.0,
        )
    # dangerous
    if mech is None:
        return ModeResult(
            mode=fm,
            mechanism=None,
            lambda_mode=lam,
            lambda_safe=0.0,
            lambda_dd=0.0,
            lambda_du=lam,
            lambda_lf=lam,
        )
    dc = mech.target_dc_pct / 100.0
    dc_lf = mech.target_dc_lf_pct / 100.0
    return ModeResult(
        mode=fm,
        mechanism=mech,
        lambda_mode=lam,
        lambda_safe=0.0,
        lambda_dd=lam * dc,
        lambda_du=lam * (1.0 - dc),
        lambda_lf=lam * (1.0 - dc) * (1.0 - dc_lf),
    )


def compute_aggregate_fmeda(
    per_module_results: List[FmedaResult],
    asil_target: str = "ASIL-B",
) -> FmedaResult:
    """Combine per-module FmedaResult objects into a cross-module aggregate.

    Per ISO 26262-5 §8.4.4, the item-level (or here IP-block-level)
    FMEDA aggregates the per-module λ values and recomputes SPFM / LFM
    / PMHF from the totals.

    For v0.1 we use a **straight sum** without cross-module re-crediting.
    Some module-level λ_DU contributions could be re-classified as
    aggregate-level λ_DD when the `safe_state_controller` mechanism
    catches them at item scope; that re-crediting requires Phase C
    fault-injection campaigns to measure the actual aggregate-coverage
    rate. v0.2 of this function will accept a configurable per-row
    ``aggregate_credit`` mapping to apply that re-crediting honestly.

    Returns a FmedaResult where ``module = "aggregate"`` and ``per_mode``
    is the concatenation of all input per_mode lists (preserving full
    traceability per ISO 26262-8 §6.4.4).
    """
    if not per_module_results:
        raise ValueError("at least one per-module result required for aggregation")
    if asil_target not in ASIL_TARGETS:
        raise ValueError(f"unknown ASIL target: {asil_target}")

    lambda_total = sum(r.lambda_total for r in per_module_results)
    lambda_safe = sum(r.lambda_safe_total for r in per_module_results)
    lambda_dangerous = sum(r.lambda_dangerous_total for r in per_module_results)
    lambda_dd = sum(r.lambda_dd_total for r in per_module_results)
    lambda_du = sum(r.lambda_du_total for r in per_module_results)
    lambda_lf = sum(r.lambda_lf_total for r in per_module_results)
    lambda_dpf = sum(r.lambda_dpf_total for r in per_module_results)

    # SPFM / LFM / PMHF recomputed from aggregated sums per ISO 26262-5 §8.4
    if lambda_dangerous > 0:
        spfm = max(0.0, 1.0 - lambda_du / lambda_dangerous)
    else:
        spfm = 1.0
    detectable_residual = lambda_dangerous - lambda_du
    if detectable_residual > 0:
        lfm = max(0.0, 1.0 - lambda_lf / detectable_residual)
    else:
        lfm = 1.0 if lambda_lf == 0 else 0.0
    pmhf = lambda_du + 0.5 * lambda_dpf

    per_mode_concat: List[ModeResult] = []
    for r in per_module_results:
        per_mode_concat.extend(r.per_mode)

    return FmedaResult(
        module="aggregate",
        asil_target=asil_target,
        per_mode=per_mode_concat,
        lambda_total=lambda_total,
        lambda_safe_total=lambda_safe,
        lambda_dangerous_total=lambda_dangerous,
        lambda_dd_total=lambda_dd,
        lambda_du_total=lambda_du,
        lambda_lf_total=lambda_lf,
        lambda_dpf_total=lambda_dpf,
        spfm_pct=spfm * 100.0,
        lfm_pct=lfm * 100.0,
        pmhf_fit=pmhf,
        asil_target_metrics=ASIL_TARGETS[asil_target],
    )


def compute_fmeda(
    failure_modes: List[FailureMode],
    mechanisms: Dict[str, SafetyMechanism],
    module: str,
    asil_target: str = "ASIL-B",
    service_interval_hours: float = 8760.0,  # 1 year
) -> FmedaResult:
    if asil_target not in ASIL_TARGETS:
        raise ValueError(f"unknown ASIL target: {asil_target}")

    per_mode = [compute_mode(fm, mechanisms) for fm in failure_modes]
    lambda_total = sum(r.lambda_mode for r in per_mode)
    lambda_safe = sum(r.lambda_safe for r in per_mode)
    lambda_dd = sum(r.lambda_dd for r in per_mode)
    lambda_du = sum(r.lambda_du for r in per_mode)
    lambda_lf = sum(r.lambda_lf for r in per_mode)
    lambda_dangerous = lambda_dd + lambda_du

    # SPFM = 1 - λ_DU / λ_dangerous (clamped to [0,1])
    if lambda_dangerous > 0:
        spfm = max(0.0, 1.0 - lambda_du / lambda_dangerous)
    else:
        spfm = 1.0

    # LFM = 1 - λ_LF / (λ_dangerous - λ_DU)
    detectable_residual = lambda_dangerous - lambda_du
    if detectable_residual > 0:
        lfm = max(0.0, 1.0 - lambda_lf / detectable_residual)
    else:
        # If everything dangerous is undetected, latent-fault metric
        # is undefined; conservatively report 0.
        lfm = 1.0 if lambda_lf == 0 else 0.0

    # Dual-point fault contribution per ISO 26262-5 §8.4.4 simplified:
    # λ_DPF ≈ λ_LF × (T_service / 2). (Service-interval halving accounts
    # for average exposure window.) Detectable latent faults are
    # bounded by τ_lifetime in FIT-equivalent terms.
    hours_per_fit = 1.0e9
    lambda_dpf = lambda_lf * (service_interval_hours / (2.0 * hours_per_fit))

    pmhf = lambda_du + 0.5 * lambda_dpf

    return FmedaResult(
        module=module,
        asil_target=asil_target,
        per_mode=per_mode,
        lambda_total=lambda_total,
        lambda_safe_total=lambda_safe,
        lambda_dangerous_total=lambda_dangerous,
        lambda_dd_total=lambda_dd,
        lambda_du_total=lambda_du,
        lambda_lf_total=lambda_lf,
        lambda_dpf_total=lambda_dpf,
        spfm_pct=spfm * 100.0,
        lfm_pct=lfm * 100.0,
        pmhf_fit=pmhf,
        asil_target_metrics=ASIL_TARGETS[asil_target],
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(x: float, digits: int = 4) -> str:
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e6:
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


def render_markdown(result: FmedaResult, *, doc_id: str, generated_on: str) -> str:
    lines: List[str] = []
    lines.append(f"# FMEDA — {result.module}")
    lines.append("")
    lines.append(f"**Document ID:** {doc_id}  ")
    lines.append(f"**Generated:** {generated_on}  ")
    lines.append(f"**Standard:** ISO 26262-5 §7.4.5, §8.4 + ISO 26262-11 §4.6  ")
    lines.append(f"**Module:** `rtl/{result.module}/`  ")
    lines.append(f"**ASIL target:** {result.asil_target}  ")
    lines.append(
        "**Status:** v0.1 — failure-rate baselines are placeholders sourced "
        "from IEC 62380 / SN29500 scaled to 7 nm digital-logic assumptions. "
        "Diagnostic-coverage values are *targets* until validated by the "
        "fault-injection campaign documented in "
        "`docs/safety/fault_injection/`."
    )
    lines.append("")
    lines.append("## 1. Per-failure-mode table")
    lines.append("")
    lines.append(
        "| ID | Sub-part | Failure mode | λ_FM (FIT) | Class | Mechanism | DC (%) | λ_S | λ_DD | λ_DU | λ_LF |"
    )
    lines.append(
        "|---|---|---|---:|---|---|---:|---:|---:|---:|---:|"
    )
    for r in result.per_mode:
        mech = r.mechanism.name if r.mechanism else "—"
        lines.append(
            f"| `{r.mode.id}` "
            f"| {r.mode.subpart} "
            f"| {r.mode.mode} "
            f"| {_fmt(r.lambda_mode)} "
            f"| {r.mode.classification} "
            f"| {mech} "
            f"| {_fmt(r.dc_applied_pct, 1)} "
            f"| {_fmt(r.lambda_safe)} "
            f"| {_fmt(r.lambda_dd)} "
            f"| {_fmt(r.lambda_du)} "
            f"| {_fmt(r.lambda_lf)} |"
        )
    lines.append("")
    lines.append("All λ values in FIT (failures per 10⁹ hours).")
    lines.append("")
    lines.append("## 2. Aggregates")
    lines.append("")
    lines.append("| Quantity | Value (FIT) |")
    lines.append("|---|---:|")
    lines.append(f"| λ_total (all failure modes) | {_fmt(result.lambda_total)} |")
    lines.append(f"| λ_S (safe) | {_fmt(result.lambda_safe_total)} |")
    lines.append(f"| λ_dangerous (DD + DU) | {_fmt(result.lambda_dangerous_total)} |")
    lines.append(f"| λ_DD (dangerous-detected) | {_fmt(result.lambda_dd_total)} |")
    lines.append(f"| λ_DU (dangerous-undetected, SPF) | {_fmt(result.lambda_du_total)} |")
    lines.append(f"| λ_LF (latent-fault residual) | {_fmt(result.lambda_lf_total)} |")
    lines.append(f"| λ_DPF (estimated dual-point) | {_fmt(result.lambda_dpf_total)} |")
    lines.append("")
    lines.append("## 3. Metrics vs ASIL target")
    lines.append("")
    targets = result.asil_target_metrics
    lines.append("| Metric | Computed | Target ({asil}) | Pass? |".format(asil=result.asil_target))
    lines.append("|---|---:|---:|:---:|")
    lines.append(
        f"| SPFM | {result.spfm_pct:.2f} % | ≥ {targets['spfm_min_pct']:.0f} % | "
        f"{'✅' if result.spfm_pct >= targets['spfm_min_pct'] else '❌'} |"
    )
    lines.append(
        f"| LFM  | {result.lfm_pct:.2f} % | ≥ {targets['lfm_min_pct']:.0f} % | "
        f"{'✅' if result.lfm_pct >= targets['lfm_min_pct'] else '❌'} |"
    )
    pmhf_target = targets["pmhf_max_fit"]
    pmhf_target_str = f"≤ {pmhf_target:.0f} FIT" if pmhf_target != math.inf else "n/a"
    lines.append(
        f"| PMHF | {result.pmhf_fit:.4f} FIT | {pmhf_target_str} | "
        f"{'✅' if result.pmhf_fit <= pmhf_target else '❌'} |"
    )
    lines.append("")
    lines.append(
        f"**Overall:** {'✅ passes' if result.passes_asil_target else '❌ fails'} "
        f"{result.asil_target} target."
    )
    lines.append("")
    lines.append("## 4. Findings and next actions")
    lines.append("")
    if result.passes_asil_target:
        lines.append(
            f"Module-level FMEDA passes the {result.asil_target} target. "
            "Confirm at aggregate scope after fault-injection campaign "
            "validates the declared DC numbers."
        )
    else:
        targets = result.asil_target_metrics
        failures: List[str] = []
        if result.spfm_pct < targets["spfm_min_pct"]:
            failures.append(
                f"SPFM {result.spfm_pct:.2f} % below target "
                f"{targets['spfm_min_pct']:.0f} %"
            )
        if result.lfm_pct < targets["lfm_min_pct"]:
            failures.append(
                f"LFM {result.lfm_pct:.2f} % below target "
                f"{targets['lfm_min_pct']:.0f} %"
            )
        if result.pmhf_fit > targets["pmhf_max_fit"]:
            failures.append(
                f"PMHF {result.pmhf_fit:.4f} FIT above target "
                f"{targets['pmhf_max_fit']:.0f} FIT"
            )
        lines.append(
            f"Module-level FMEDA does **not** meet the "
            f"{result.asil_target} target ({'; '.join(failures)})."
        )
        # Identify top-3 dangerous-undetected contributors.
        top = sorted(
            (r for r in result.per_mode if r.lambda_du > 0),
            key=lambda r: r.lambda_du,
            reverse=True,
        )[:3]
        if top:
            lines.append("")
            lines.append(
                "Top dangerous-undetected contributors (drive the SPFM gap):"
            )
            lines.append("")
            for r in top:
                mech = r.mechanism.name if r.mechanism else "no mechanism"
                lines.append(
                    f"- `{r.mode.id}` — λ_DU = {_fmt(r.lambda_du)} FIT "
                    f"({mech}, DC = {r.dc_applied_pct:.1f} %)"
                )
            lines.append("")
            lines.append("Closure options:")
            lines.append(
                "1. Add a module-level mechanism for any uncovered "
                "(`no mechanism`) row. The most common pattern is parity "
                "or duplication on a single critical FF."
            )
            lines.append(
                "2. Improve the declared DC of the named mechanism by "
                "running the fault-injection campaign and demonstrating "
                "higher actual coverage than the conservative target."
            )
            lines.append(
                "3. Demonstrate aggregate coverage via the "
                "`safe_state_controller` cross-module roll-up — a "
                "module-level SPF that escalates to safe-state within "
                "the FTTI may be reclassified as DD at aggregate scope."
            )
    lines.append("")
    lines.append("## 5. Cross-references")
    lines.append("")
    lines.append("- `docs/safety/seooc_declaration_v0_1.md` §5 (declared mechanism coverage)")
    lines.append("- `docs/safety/iso26262_gap_analysis_v0_1.md` Part 5 (FMEDA gap closure)")
    lines.append("- `docs/safety/fault_injection/` (campaigns that will validate DC numbers)")
    lines.append("- `tools/safety/failure_modes.yaml` (input catalog)")
    lines.append("- `tools/safety/safety_mechanisms.yaml` (mechanism catalog)")
    lines.append("")
    lines.append("## 6. Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m tools.safety.fmeda \\")
    lines.append(f"    --module {result.module} \\")
    lines.append(f"    --asil {result.asil_target} \\")
    lines.append(f"    --output docs/safety/fmeda/{result.module}_fmeda.md")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def discover_modules(failure_modes_path: Path) -> List[str]:
    """List every distinct module that has at least one failure-mode row."""
    fms = load_failure_modes(failure_modes_path)
    return sorted({fm.module for fm in fms})


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--module", help="RTL module name (e.g. dms_fusion). Mutually exclusive with --aggregate.")
    p.add_argument("--aggregate", action="store_true", help="Aggregate FMEDA across every module in failure_modes.yaml.")
    p.add_argument(
        "--failure-modes",
        type=Path,
        default=DEFAULT_FAILURE_MODES,
        help="Path to failure_modes.yaml (default: tools/safety/failure_modes.yaml)",
    )
    p.add_argument(
        "--mechanisms",
        type=Path,
        default=DEFAULT_MECHANISMS,
        help="Path to safety_mechanisms.yaml (default: tools/safety/safety_mechanisms.yaml)",
    )
    p.add_argument(
        "--asil",
        choices=list(ASIL_TARGETS),
        default="ASIL-B",
        help="Target ASIL (default ASIL-B)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown path (default stdout)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of markdown",
    )
    p.add_argument(
        "--doc-id",
        default=None,
        help="Override generated document ID",
    )
    p.add_argument(
        "--generated-on",
        default=None,
        help="Override timestamp (default: today)",
    )
    args = p.parse_args(argv)

    if args.aggregate and args.module:
        print("error: --aggregate and --module are mutually exclusive", file=sys.stderr)
        return 2
    if not args.aggregate and not args.module:
        print("error: one of --module or --aggregate is required", file=sys.stderr)
        return 2

    mechanisms = load_mechanisms(args.mechanisms)

    if args.aggregate:
        modules = discover_modules(args.failure_modes)
        if not modules:
            print(f"error: no modules found in {args.failure_modes}", file=sys.stderr)
            return 2
        per_module_results: List[FmedaResult] = []
        for mod in modules:
            fms = load_failure_modes(args.failure_modes, module_filter=mod)
            per_module_results.append(
                compute_fmeda(fms, mechanisms, module=mod, asil_target=args.asil)
            )
        result = compute_aggregate_fmeda(per_module_results, asil_target=args.asil)
    else:
        failure_modes = load_failure_modes(args.failure_modes, module_filter=args.module)
        if not failure_modes:
            print(
                f"error: no failure modes found for module '{args.module}' "
                f"in {args.failure_modes}",
                file=sys.stderr,
            )
            return 2
        result = compute_fmeda(failure_modes, mechanisms, module=args.module, asil_target=args.asil)

    if args.json:
        payload = {
            "module": result.module,
            "asil_target": result.asil_target,
            "lambda_total_fit": result.lambda_total,
            "lambda_safe_fit": result.lambda_safe_total,
            "lambda_dangerous_fit": result.lambda_dangerous_total,
            "lambda_dd_fit": result.lambda_dd_total,
            "lambda_du_fit": result.lambda_du_total,
            "lambda_lf_fit": result.lambda_lf_total,
            "lambda_dpf_fit": result.lambda_dpf_total,
            "spfm_pct": result.spfm_pct,
            "lfm_pct": result.lfm_pct,
            "pmhf_fit": result.pmhf_fit,
            "passes_target": result.passes_asil_target,
        }
        text = json.dumps(payload, indent=2)
    else:
        from datetime import date

        doc_id = args.doc_id or f"ASTR-FMEDA-{result.module.upper()}-V0.1"
        gen = args.generated_on or date.today().isoformat()
        text = render_markdown(result, doc_id=doc_id, generated_on=gen)

    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
