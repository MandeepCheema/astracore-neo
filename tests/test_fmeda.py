"""Tests for the FMEDA tool (tools/safety/fmeda.py).

Covers:
- per-failure-mode classification math (safe / no-effect / dangerous)
- coverage application (uncovered / 100 % covered / partial)
- aggregate SPFM / LFM / PMHF formulas vs hand calculation
- ASIL target gating
- input validation (distribution range, classification enum)
- end-to-end YAML load + render of the dms_fusion FMEDA
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tools.safety import fmeda as fmeda_mod


# ---------------------------------------------------------------------------
# Per-mode computation
# ---------------------------------------------------------------------------


def _mech(id_: str, dc: float, dc_lf: float = None) -> fmeda_mod.SafetyMechanism:
    return fmeda_mod.SafetyMechanism(
        id=id_,
        name=f"mech-{id_}",
        description="test",
        target_dc_pct=dc,
        target_dc_lf_pct=dc if dc_lf is None else dc_lf,
        applies_to=(),
    )


def _fm(
    id_: str,
    classification: str,
    lambda_fit: float,
    distribution: float = 1.0,
    mechanism_id: str | None = None,
) -> fmeda_mod.FailureMode:
    return fmeda_mod.FailureMode(
        id=id_,
        module="m",
        subpart="sp",
        mode="x",
        lambda_fit=lambda_fit,
        distribution=distribution,
        classification=classification,
        mechanism_id=mechanism_id,
        rationale="test",
    )


def test_safe_mode_routes_to_lambda_safe():
    fm = _fm("safe1", "safe", 1.0)
    r = fmeda_mod.compute_mode(fm, mechanisms={})
    assert r.lambda_safe == 1.0
    assert r.lambda_dd == 0.0
    assert r.lambda_du == 0.0
    assert r.lambda_lf == 0.0


def test_no_effect_mode_routes_to_lambda_safe():
    fm = _fm("ne1", "no-effect", 2.5)
    r = fmeda_mod.compute_mode(fm, mechanisms={})
    assert r.lambda_safe == 2.5
    assert r.lambda_du == 0.0


def test_dangerous_uncovered_routes_to_lambda_du_and_lf():
    fm = _fm("d1", "dangerous", 4.0)
    r = fmeda_mod.compute_mode(fm, mechanisms={})
    assert r.lambda_dd == 0.0
    assert r.lambda_du == 4.0
    assert r.lambda_lf == 4.0


def test_dangerous_full_coverage_routes_to_lambda_dd():
    mech = _mech("perfect", 100.0, 100.0)
    fm = _fm("d2", "dangerous", 4.0, mechanism_id="perfect")
    r = fmeda_mod.compute_mode(fm, mechanisms={"perfect": mech})
    assert r.lambda_dd == pytest.approx(4.0)
    assert r.lambda_du == pytest.approx(0.0)
    assert r.lambda_lf == pytest.approx(0.0)


def test_dangerous_partial_coverage_splits_correctly():
    mech = _mech("half", 50.0, 50.0)
    fm = _fm("d3", "dangerous", 10.0, mechanism_id="half")
    r = fmeda_mod.compute_mode(fm, mechanisms={"half": mech})
    assert r.lambda_dd == pytest.approx(5.0)
    assert r.lambda_du == pytest.approx(5.0)
    # lf = lambda * (1-dc) * (1-dc_lf) = 10 * 0.5 * 0.5
    assert r.lambda_lf == pytest.approx(2.5)


def test_distribution_apportions_lambda():
    fm = _fm("d4", "dangerous", 8.0, distribution=0.25)
    assert fm.lambda_mode == pytest.approx(2.0)
    r = fmeda_mod.compute_mode(fm, mechanisms={})
    assert r.lambda_du == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def test_aggregate_metrics_match_hand_calculation():
    perfect = _mech("perfect", 100.0, 100.0)
    half = _mech("half", 50.0, 50.0)
    modes = [
        _fm("a", "dangerous", 10.0, mechanism_id="perfect"),  # all DD
        _fm("b", "dangerous", 10.0, mechanism_id="half"),     # 5 DD, 5 DU
        _fm("c", "dangerous", 10.0),                            # 10 DU
        _fm("d", "safe", 5.0),                                  # 5 safe
    ]
    result = fmeda_mod.compute_fmeda(
        modes, {"perfect": perfect, "half": half}, module="t", asil_target="ASIL-B"
    )
    assert result.lambda_total == pytest.approx(35.0)
    assert result.lambda_safe_total == pytest.approx(5.0)
    assert result.lambda_dangerous_total == pytest.approx(30.0)
    assert result.lambda_dd_total == pytest.approx(15.0)
    assert result.lambda_du_total == pytest.approx(15.0)
    # SPFM = 1 - 15/30 = 0.5 -> 50 %
    assert result.spfm_pct == pytest.approx(50.0)
    # LFM = 1 - lambda_LF / (lambda_dangerous - lambda_DU)
    # lambda_LF: a=0 (full), b=10*0.5*0.5=2.5, c=10 (uncovered)
    # but lambda_LF only summed where mechanism exists meaningfully —
    # the formula in compute_mode sets lf for dangerous-uncovered to
    # lambda_mode (worst case). Total lf = 0 + 2.5 + 10 = 12.5
    # detectable_residual = 30 - 15 = 15
    # LFM = 1 - 12.5/15 = 0.1667 -> 16.67 %
    assert result.lfm_pct == pytest.approx((1 - 12.5 / 15.0) * 100.0, abs=0.01)


def test_no_dangerous_modes_yields_perfect_spfm():
    modes = [_fm("safe1", "safe", 1.0), _fm("ne1", "no-effect", 2.0)]
    result = fmeda_mod.compute_fmeda(modes, {}, module="t", asil_target="ASIL-D")
    assert result.spfm_pct == 100.0
    assert result.lfm_pct == 100.0
    assert result.pmhf_fit == 0.0
    assert result.passes_asil_target


def test_asil_targets_table_present():
    for asil in ("QM", "ASIL-A", "ASIL-B", "ASIL-C", "ASIL-D"):
        assert asil in fmeda_mod.ASIL_TARGETS


def test_asil_d_more_strict_than_asil_b():
    b = fmeda_mod.ASIL_TARGETS["ASIL-B"]
    d = fmeda_mod.ASIL_TARGETS["ASIL-D"]
    assert d["spfm_min_pct"] >= b["spfm_min_pct"]
    assert d["lfm_min_pct"] >= b["lfm_min_pct"]
    assert d["pmhf_max_fit"] <= b["pmhf_max_fit"]


def test_passes_asil_target_gates_correctly():
    perfect = _mech("perfect", 100.0, 100.0)
    modes = [_fm("a", "dangerous", 1.0, mechanism_id="perfect")]
    r = fmeda_mod.compute_fmeda(modes, {"perfect": perfect}, "t", asil_target="ASIL-D")
    assert r.passes_asil_target  # 100% covered, well under PMHF ceiling


def test_unknown_asil_raises():
    with pytest.raises(ValueError):
        fmeda_mod.compute_fmeda([], {}, "t", asil_target="ASIL-Z")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_distribution_must_be_in_unit_range():
    with pytest.raises(ValueError):
        _fm("bad", "dangerous", 1.0, distribution=1.5)


def test_classification_enum_validated():
    with pytest.raises(ValueError):
        fmeda_mod.FailureMode(
            id="x", module="m", subpart="sp", mode="m",
            lambda_fit=1.0, distribution=1.0,
            classification="meh",
            mechanism_id=None, rationale="",
        )


def test_negative_lambda_rejected():
    with pytest.raises(ValueError):
        _fm("neg", "dangerous", -1.0)


def test_mechanism_dc_range_validated():
    with pytest.raises(ValueError):
        fmeda_mod.SafetyMechanism(
            id="bad", name="x", description="x",
            target_dc_pct=120.0, target_dc_lf_pct=50.0,
            applies_to=(),
        )


# ---------------------------------------------------------------------------
# YAML load + end-to-end on dms_fusion (the real catalog)
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parent.parent
FAILURE_MODES = REPO_ROOT / "tools" / "safety" / "failure_modes.yaml"
MECHANISMS = REPO_ROOT / "tools" / "safety" / "safety_mechanisms.yaml"


def test_yaml_load_roundtrip():
    mechs = fmeda_mod.load_mechanisms(MECHANISMS)
    fms = fmeda_mod.load_failure_modes(FAILURE_MODES, module_filter="dms_fusion")
    assert "tmr_voter" in mechs
    assert "watchdog_sensor" in mechs
    assert any(fm.id == "dms_fusion.dal_lane.seu" for fm in fms)
    assert all(fm.module == "dms_fusion" for fm in fms)


def test_dms_fusion_fmeda_runs_end_to_end():
    mechs = fmeda_mod.load_mechanisms(MECHANISMS)
    fms = fmeda_mod.load_failure_modes(FAILURE_MODES, module_filter="dms_fusion")
    result = fmeda_mod.compute_fmeda(fms, mechs, module="dms_fusion", asil_target="ASIL-D")
    # Numbers documented in docs/safety/fmeda/dms_fusion_fmeda.md.
    # Updated 2026-04-20 after F4-A-5 (tmr_valid_r shadow-register fix
    # in rtl/dms_fusion/dms_fusion.v) lifted SPFM 84.70 → 85.52 %.
    assert result.lambda_total == pytest.approx(0.0517, abs=0.001)
    assert result.spfm_pct == pytest.approx(85.5, abs=0.5)
    assert result.lfm_pct == pytest.approx(93.0, abs=0.5)
    # At ASIL-D module-level alone, dms_fusion does not pass SPFM (99% target).
    # This is a real, expected finding — additional aggregate coverage
    # via safe_state_controller is required.
    assert not result.passes_asil_target


def test_dms_fusion_fmeda_at_asil_b_still_fails_spfm():
    mechs = fmeda_mod.load_mechanisms(MECHANISMS)
    fms = fmeda_mod.load_failure_modes(FAILURE_MODES, module_filter="dms_fusion")
    result = fmeda_mod.compute_fmeda(fms, mechs, module="dms_fusion", asil_target="ASIL-B")
    # ASIL-B SPFM target is 90 %; dms_fusion at module scope is ~84.7 %
    # so it still fails. Aggregate FMEDA must lift this.
    assert not result.passes_asil_target
    # But LFM (60% target) and PMHF (100 FIT) both pass at ASIL-B
    assert result.lfm_pct >= 60.0
    assert result.pmhf_fit < 100.0


def test_render_markdown_has_required_sections():
    mechs = fmeda_mod.load_mechanisms(MECHANISMS)
    fms = fmeda_mod.load_failure_modes(FAILURE_MODES, module_filter="dms_fusion")
    result = fmeda_mod.compute_fmeda(fms, mechs, module="dms_fusion", asil_target="ASIL-B")
    md = fmeda_mod.render_markdown(result, doc_id="TEST", generated_on="2026-04-20")
    for required in (
        "# FMEDA — dms_fusion",
        "## 1. Per-failure-mode table",
        "## 2. Aggregates",
        "## 3. Metrics vs ASIL target",
        "## 4. Findings and next actions",
        "## 5. Cross-references",
        "## 6. Reproduce",
        "SPFM",
        "LFM",
        "PMHF",
        "λ_DU",
    ):
        assert required in md, f"missing section/keyword: {required}"


def test_findings_section_lists_top_uncovered_when_failing():
    """When the FMEDA fails its target, the report must call out top SPF contributors."""
    mechs = fmeda_mod.load_mechanisms(MECHANISMS)
    fms = fmeda_mod.load_failure_modes(FAILURE_MODES, module_filter="dms_fusion")
    result = fmeda_mod.compute_fmeda(fms, mechs, module="dms_fusion", asil_target="ASIL-D")
    md = fmeda_mod.render_markdown(result, doc_id="TEST", generated_on="2026-04-20")
    assert not result.passes_asil_target  # precondition for this assertion
    assert "Top dangerous-undetected contributors" in md
    # tmr_valid_r is the largest uncovered SPF by lambda_DU; must appear.
    assert "tmr_valid" in md or "dms_fusion.tmr_valid" in md
    # Closure options should be enumerated.
    assert "Closure options" in md


def test_findings_section_short_when_passing():
    """When the FMEDA passes, the report should NOT enumerate closure options."""
    perfect = _mech("perfect", 100.0, 100.0)
    modes = [_fm("a", "dangerous", 1.0, mechanism_id="perfect")]
    result = fmeda_mod.compute_fmeda(modes, {"perfect": perfect}, "t", asil_target="ASIL-B")
    md = fmeda_mod.render_markdown(result, doc_id="TEST", generated_on="2026-04-20")
    assert result.passes_asil_target
    assert "passes" in md.lower()
    assert "Closure options" not in md


def test_module_filter_excludes_other_modules():
    fms = fmeda_mod.load_failure_modes(FAILURE_MODES, module_filter="some_module_that_does_not_exist")
    assert fms == []


# ---------------------------------------------------------------------------
# Aggregate FMEDA
# ---------------------------------------------------------------------------


def test_aggregate_requires_at_least_one_result():
    with pytest.raises(ValueError, match="at least one"):
        fmeda_mod.compute_aggregate_fmeda([], asil_target="ASIL-B")


def test_aggregate_unknown_asil_raises():
    perfect = _mech("perfect", 100.0, 100.0)
    modes = [_fm("a", "dangerous", 1.0, mechanism_id="perfect")]
    r = fmeda_mod.compute_fmeda(modes, {"perfect": perfect}, "t", asil_target="ASIL-B")
    with pytest.raises(ValueError, match="unknown ASIL"):
        fmeda_mod.compute_aggregate_fmeda([r], asil_target="ASIL-Z")


def test_aggregate_sums_lambdas_correctly():
    perfect = _mech("perfect", 100.0, 100.0)
    half = _mech("half", 50.0, 50.0)
    # Module 1: 10 FIT all-DD (perfect coverage)
    m1_modes = [_fm("m1.a", "dangerous", 10.0, mechanism_id="perfect")]
    r1 = fmeda_mod.compute_fmeda(m1_modes, {"perfect": perfect}, module="m1", asil_target="ASIL-B")
    # Module 2: 10 FIT half-covered = 5 DD / 5 DU
    m2_modes = [_fm("m2.a", "dangerous", 10.0, mechanism_id="half")]
    r2 = fmeda_mod.compute_fmeda(m2_modes, {"half": half}, module="m2", asil_target="ASIL-B")
    agg = fmeda_mod.compute_aggregate_fmeda([r1, r2], asil_target="ASIL-B")
    assert agg.module == "aggregate"
    assert agg.lambda_total == pytest.approx(20.0)
    assert agg.lambda_dangerous_total == pytest.approx(20.0)
    assert agg.lambda_dd_total == pytest.approx(15.0)
    assert agg.lambda_du_total == pytest.approx(5.0)
    # SPFM = 1 - 5/20 = 75%
    assert agg.spfm_pct == pytest.approx(75.0)


def test_aggregate_concatenates_per_mode():
    perfect = _mech("perfect", 100.0, 100.0)
    m1 = [_fm("m1.a", "dangerous", 1.0, mechanism_id="perfect"), _fm("m1.b", "safe", 1.0)]
    m2 = [_fm("m2.a", "dangerous", 1.0, mechanism_id="perfect")]
    r1 = fmeda_mod.compute_fmeda(m1, {"perfect": perfect}, "m1", asil_target="ASIL-B")
    r2 = fmeda_mod.compute_fmeda(m2, {"perfect": perfect}, "m2", asil_target="ASIL-B")
    agg = fmeda_mod.compute_aggregate_fmeda([r1, r2], asil_target="ASIL-B")
    # 2 modes from m1 + 1 mode from m2 = 3 total
    assert len(agg.per_mode) == 3


def test_aggregate_fmeda_on_real_catalog():
    """End-to-end aggregate over every module in failure_modes.yaml."""
    mechs = fmeda_mod.load_mechanisms(MECHANISMS)
    modules = fmeda_mod.discover_modules(FAILURE_MODES)
    assert "dms_fusion" in modules
    assert "npu_top" in modules
    assert len(modules) >= 6  # 4 safety primitives + dms_fusion + npu_top
    per_module: list = []
    for mod in modules:
        fms = fmeda_mod.load_failure_modes(FAILURE_MODES, module_filter=mod)
        per_module.append(
            fmeda_mod.compute_fmeda(fms, mechs, module=mod, asil_target="ASIL-B")
        )
    agg = fmeda_mod.compute_aggregate_fmeda(per_module, asil_target="ASIL-B")
    # Aggregate λ_total must equal sum of per-module
    assert agg.lambda_total == pytest.approx(sum(r.lambda_total for r in per_module))
    # Aggregate SPFM is driven by the largest λ_DU contributor (npu_top today)
    # and is necessarily ≤ the per-module worst SPFM in some sense — verify
    # it falls within a reasonable range without locking in a specific number
    # that would tie this test to current YAML values
    assert 0.0 <= agg.spfm_pct <= 100.0
    assert 0.0 <= agg.lfm_pct <= 100.0
    assert agg.pmhf_fit >= 0.0
    # At today's baseline, aggregate fails ASIL-B (npu_top SPFM 2.08% dominates)
    assert not agg.passes_asil_target


def test_aggregate_render_markdown_works():
    mechs = fmeda_mod.load_mechanisms(MECHANISMS)
    modules = fmeda_mod.discover_modules(FAILURE_MODES)
    per_module = [
        fmeda_mod.compute_fmeda(
            fmeda_mod.load_failure_modes(FAILURE_MODES, module_filter=mod),
            mechs, module=mod, asil_target="ASIL-B",
        )
        for mod in modules
    ]
    agg = fmeda_mod.compute_aggregate_fmeda(per_module, asil_target="ASIL-B")
    md = fmeda_mod.render_markdown(agg, doc_id="TEST-AGG", generated_on="2026-04-20")
    assert "# FMEDA — aggregate" in md
    assert "λ_DU" in md
    # Verify "Findings and next actions" section appears (since agg fails)
    assert "## 4. Findings and next actions" in md
