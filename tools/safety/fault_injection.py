"""Fault-injection campaign planner + result aggregator.

This module is the **RTL-agnostic, framework-agnostic** half of the
fault-injection harness. It plans campaigns, validates the plan,
iterates injection specs, and aggregates results into the diagnostic
coverage numbers the FMEDA tool consumes.

The **simulator-side** half lives in ``sim/fault_injection/`` and uses
cocotb's ``Force`` / ``Release`` API on Verilator. That code only runs
under WSL because cocotb requires a POSIX simulator. This module runs
anywhere (Windows / Linux / mac) and is therefore the unit-testable
piece — the planner and aggregator have no dependency on cocotb or
Verilator.

Why split this way
------------------
Fault-injection has two failure modes for the *tooling itself*:

1. **Plan errors**: bad target paths, off-by-one bit indices,
   classification mismatches, or expected-response logic errors.
   These are caught at planning time. We unit-test the planner so
   campaign authors get fast feedback on Windows before incurring a
   WSL+cocotb cycle.
2. **Aggregation errors**: cocotb emits one JSON record per injection;
   we aggregate them into a per-campaign coverage report. Bugs here
   would silently mis-quote diagnostic coverage and corrupt the
   safety case. We unit-test the aggregator with synthetic JSON
   records.

What runs in cocotb
-------------------
cocotb owns:
- selecting the target signal at the simulator level
- forcing / releasing values (``cocotb.handle.SimHandle.value`` and
  ``Force`` / ``Release``)
- driving stimulus
- observing the oracle signal (e.g. ``fault_detected``)
- writing one ``InjectionResult`` JSON record per injection

What runs here
--------------
This module owns:
- parsing and validating ``campaigns/*.yaml`` campaign specs
- iterating all planned injections (used by the cocotb runner)
- aggregating cocotb-emitted JSONL into ``CampaignResult`` objects
- rendering markdown reports for ``docs/safety/fault_injection/``

Reference
---------
ISO 26262-5 §11 (HW integration testing) and ISO 26262-11 §7 (soft
errors) require a fault-injection campaign with documented coverage
to claim DC numbers in the FMEDA. This tool produces that evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

try:
    import yaml  # type: ignore[import-untyped]
except ImportError as e:  # pragma: no cover
    raise SystemExit("PyYAML required (already in pyproject.toml)") from e


class InjectionKind(str, Enum):
    """Categories of injection the cocotb runner supports."""

    STUCK_0 = "stuck_0"        # Force net to 0 for `duration_cycles`
    STUCK_1 = "stuck_1"        # Force net to 1 for `duration_cycles`
    BIT_FLIP = "bit_flip"      # XOR mask on net for `duration_cycles`
    TRANSIENT = "transient"    # 1-cycle bit-flip then release


@dataclass(frozen=True)
class InjectionSpec:
    """A single fault to inject in one simulation run.

    Attributes
    ----------
    target_path : str
        Hierarchical RTL path to the target net or register, e.g.
        ``"u_dut.u_tmr.lane_a_reg"``. The cocotb runner resolves this
        via ``simulator.get_root_handle()``.
    bit_index : int | None
        Which bit of the target to perturb. ``None`` means "the whole
        target" (useful for 1-bit nets).
    kind : InjectionKind
        How to perturb (see :class:`InjectionKind`).
    start_cycle : int
        Cycle at which to apply the perturbation.
    duration_cycles : int
        How many cycles the perturbation persists. Ignored for
        ``TRANSIENT`` (always 1).
    expected_detection : bool
        True if the campaign's oracle is *expected* to detect this
        fault (i.e. it should be classified as detected after the run).
        False if the fault is below the campaign's coverage scope.

    Notes
    -----
    Verilator's ``Force`` does not reach port-constant drivers (memory
    pin in ``feedback_verilator_force.md``). Pick *internal* nets for
    injection, not module ports.
    """

    target_path: str
    bit_index: Optional[int]
    kind: InjectionKind
    start_cycle: int
    duration_cycles: int
    expected_detection: bool = True

    def __post_init__(self) -> None:
        if not self.target_path or "." not in self.target_path:
            raise ValueError(
                f"target_path must be a hierarchical RTL path with at "
                f"least one '.' separator (got: {self.target_path!r})"
            )
        if self.bit_index is not None and self.bit_index < 0:
            raise ValueError(f"bit_index must be ≥ 0 or None (got {self.bit_index})")
        if self.start_cycle < 0:
            raise ValueError(f"start_cycle must be ≥ 0 (got {self.start_cycle})")
        if self.duration_cycles < 1:
            raise ValueError(f"duration_cycles must be ≥ 1 (got {self.duration_cycles})")
        if not isinstance(self.kind, InjectionKind):
            raise ValueError(f"kind must be an InjectionKind (got {type(self.kind).__name__})")


@dataclass(frozen=True)
class Campaign:
    """One fault-injection campaign on one DUT.

    Attributes
    ----------
    name : str
        Human-readable campaign ID (used as filename).
    target_module : str
        RTL module name (e.g. ``tmr_voter``). Used for traceability
        back to the FMEDA failure_modes catalog.
    oracle_signal : str
        RTL path to the fault-detection signal (e.g.
        ``"u_dut.tmr_fault"``). The cocotb runner samples this at the
        end of each injection's duration to decide detected vs missed.
    expected_safe_response : str
        Human description of what the SUT must do on detection (e.g.
        "tmr_fault asserts within 1 cycle of voted disagreement").
        Carried into the report for the safety case.
    injections : tuple[InjectionSpec, ...]
        The planned injections. Iterating yields each in order.
    fmeda_failure_mode_ids : tuple[str, ...]
        IDs from ``failure_modes.yaml`` this campaign validates DC for.
    """

    name: str
    target_module: str
    oracle_signal: str
    expected_safe_response: str
    injections: Sequence[InjectionSpec]
    fmeda_failure_mode_ids: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("campaign name required")
        if not self.target_module:
            raise ValueError("target_module required")
        if not self.oracle_signal or "." not in self.oracle_signal:
            raise ValueError("oracle_signal must be a hierarchical RTL path")
        if not self.injections:
            raise ValueError(f"campaign {self.name}: at least one injection required")

    def __iter__(self):
        return iter(self.injections)

    def __len__(self) -> int:
        return len(self.injections)


@dataclass(frozen=True)
class InjectionResult:
    """Result of running one injection in cocotb."""

    spec_index: int           # index into Campaign.injections
    target_path: str
    detected: bool             # oracle signal asserted
    detection_cycle: Optional[int]   # cycle the oracle fired (None if not)
    perturbed_outputs: bool    # any monitored output changed (i.e. injection had effect)


@dataclass
class CampaignResult:
    """Aggregated result of running a Campaign through cocotb."""

    campaign: Campaign
    per_injection: List[InjectionResult]

    @property
    def total_runs(self) -> int:
        return len(self.per_injection)

    @property
    def perturbed(self) -> int:
        return sum(1 for r in self.per_injection if r.perturbed_outputs)

    @property
    def detected(self) -> int:
        return sum(1 for r in self.per_injection if r.perturbed_outputs and r.detected)

    @property
    def missed(self) -> int:
        return sum(1 for r in self.per_injection if r.perturbed_outputs and not r.detected)

    @property
    def benign(self) -> int:
        """Injections that produced no observable output perturbation."""
        return self.total_runs - self.perturbed

    @property
    def false_positives(self) -> int:
        """Detections fired but no output was perturbed."""
        return sum(
            1
            for r in self.per_injection
            if not r.perturbed_outputs and r.detected
        )

    @property
    def coverage_pct(self) -> float:
        """Diagnostic coverage = detected / perturbed (×100)."""
        if self.perturbed == 0:
            return 100.0  # nothing to detect, vacuously covered
        return 100.0 * self.detected / self.perturbed


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_campaign(path: Path) -> Campaign:
    """Load a campaign from YAML.

    YAML schema (see also ``sim/fault_injection/campaigns/*.yaml``):

    .. code-block:: yaml

        name: tmr_voter_seu_1k
        target_module: tmr_voter
        oracle_signal: u_dut.tmr_fault
        expected_safe_response: >
          tmr_fault asserts on the lane that disagrees within 1 cycle
        fmeda_failure_mode_ids:
          - dms_fusion.dal_lane.seu
        injections:
          - target_path: u_dut.lane_a_reg
            bit_index: 3
            kind: bit_flip
            start_cycle: 100
            duration_cycles: 1
            expected_detection: true
          # ... more rows
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _campaign_from_dict(raw, source=path)


def _campaign_from_dict(raw: Dict, *, source: Optional[Path] = None) -> Campaign:
    where = f" (in {source})" if source else ""
    try:
        injections_raw = raw["injections"]
    except KeyError as e:
        raise ValueError(f"campaign missing 'injections'{where}") from e
    injections = []
    for i, entry in enumerate(injections_raw):
        try:
            spec = InjectionSpec(
                target_path=entry["target_path"],
                bit_index=entry.get("bit_index"),
                kind=InjectionKind(entry["kind"]),
                start_cycle=int(entry["start_cycle"]),
                duration_cycles=int(entry["duration_cycles"]),
                expected_detection=bool(entry.get("expected_detection", True)),
            )
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"campaign injection #{i}{where}: {e}"
            ) from e
        injections.append(spec)
    return Campaign(
        name=raw["name"],
        target_module=raw["target_module"],
        oracle_signal=raw["oracle_signal"],
        expected_safe_response=raw["expected_safe_response"],
        injections=tuple(injections),
        fmeda_failure_mode_ids=tuple(raw.get("fmeda_failure_mode_ids", [])),
    )


def load_results_jsonl(path: Path, campaign: Campaign) -> CampaignResult:
    """Load cocotb-emitted JSONL results into a CampaignResult.

    Each line of the JSONL file is one ``InjectionResult`` produced by
    the cocotb runner. We re-pair them with the campaign for context.
    """
    results: List[InjectionResult] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            results.append(
                InjectionResult(
                    spec_index=int(obj["spec_index"]),
                    target_path=obj["target_path"],
                    detected=bool(obj["detected"]),
                    detection_cycle=obj.get("detection_cycle"),
                    perturbed_outputs=bool(obj["perturbed_outputs"]),
                )
            )
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"{path}:{line_no}: {e}") from e
    return CampaignResult(campaign=campaign, per_injection=results)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_markdown(
    result: CampaignResult, *, doc_id: str, generated_on: str
) -> str:
    c = result.campaign
    lines: List[str] = []
    lines.append(f"# Fault-Injection Campaign — {c.name}")
    lines.append("")
    lines.append(f"**Document ID:** {doc_id}  ")
    lines.append(f"**Generated:** {generated_on}  ")
    lines.append(f"**Standard:** ISO 26262-5 §11 + ISO 26262-11 §7  ")
    lines.append(f"**Target module:** `rtl/{c.target_module}/`  ")
    lines.append(f"**Oracle signal:** `{c.oracle_signal}`  ")
    lines.append(f"**Expected safe response:** {c.expected_safe_response}")
    lines.append("")
    lines.append("## 1. Aggregate")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Planned injections | {len(c)} |")
    lines.append(f"| Runs completed | {result.total_runs} |")
    lines.append(f"| Perturbed an output | {result.perturbed} |")
    lines.append(f"| Detected by oracle | {result.detected} |")
    lines.append(f"| Missed by oracle | {result.missed} |")
    lines.append(f"| Benign (no output change) | {result.benign} |")
    lines.append(f"| False positives | {result.false_positives} |")
    lines.append(f"| **Diagnostic coverage** | **{result.coverage_pct:.2f} %** |")
    lines.append("")
    lines.append("## 2. Per-injection results (first 50)")
    lines.append("")
    lines.append("| # | Target | Detected? | Cycle | Perturbed? |")
    lines.append("|---|---|:---:|---:|:---:|")
    for r in result.per_injection[:50]:
        det = "✅" if r.detected else "❌"
        pert = "yes" if r.perturbed_outputs else "no"
        cyc = str(r.detection_cycle) if r.detection_cycle is not None else "—"
        lines.append(
            f"| {r.spec_index} | `{r.target_path}` | {det} | {cyc} | {pert} |"
        )
    if len(result.per_injection) > 50:
        lines.append(f"")
        lines.append(f"... ({len(result.per_injection) - 50} more rows omitted)")
    lines.append("")
    if c.fmeda_failure_mode_ids:
        lines.append("## 3. FMEDA traceability")
        lines.append("")
        lines.append(
            "This campaign validates the diagnostic-coverage assumption for the "
            "following failure-mode rows in `tools/safety/failure_modes.yaml`:"
        )
        lines.append("")
        for fm_id in c.fmeda_failure_mode_ids:
            lines.append(f"- `{fm_id}`")
        lines.append("")
        lines.append(
            f"After this report is filed, update the corresponding mechanism "
            f"`target_dc_pct` in `tools/safety/safety_mechanisms.yaml` to the "
            f"measured **{result.coverage_pct:.2f} %** if it differs from the "
            f"declared target by more than 5 percentage points (per SEooC §9.1 "
            f"revision trigger #2)."
        )
        lines.append("")
    lines.append("## 4. Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("# (WSL Ubuntu 22.04, Verilator 5.030, cocotb 2.0.1)")
    lines.append(f"cd sim/fault_injection")
    lines.append(f"make CAMPAIGN={c.name}")
    lines.append("python -m tools.safety.fault_injection \\")
    lines.append(f"    --campaign campaigns/{c.name}.yaml \\")
    lines.append(f"    --results out/{c.name}.jsonl \\")
    lines.append(f"    --output ../../docs/safety/fault_injection/{c.name}.md")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI (aggregation only — campaign execution is the cocotb runner's job)
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    from datetime import date

    p = argparse.ArgumentParser(description="Aggregate cocotb fault-injection results into a markdown report.")
    p.add_argument("--campaign", type=Path, required=True, help="Path to campaign YAML")
    p.add_argument("--results", type=Path, required=True, help="Path to cocotb JSONL output")
    p.add_argument("--output", type=Path, default=None, help="Markdown output path")
    p.add_argument("--doc-id", default=None)
    p.add_argument("--generated-on", default=None)
    args = p.parse_args(argv)

    campaign = load_campaign(args.campaign)
    result = load_results_jsonl(args.results, campaign)
    doc_id = args.doc_id or f"ASTR-FI-{campaign.name.upper()}-V0.1"
    gen = args.generated_on or date.today().isoformat()
    text = render_markdown(result, doc_id=doc_id, generated_on=gen)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
