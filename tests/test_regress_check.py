"""Tests for tools/safety/regress_check.py — FMEDA regression gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.safety.regress_check import (
    AGGREGATE_KEY,
    ModuleSnapshot,
    compare,
    discover_modules,
    emit_baseline_json,
    load_baseline,
    snapshot_all,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
FAILURE_MODES = REPO_ROOT / "tools" / "safety" / "failure_modes.yaml"
MECHANISMS = REPO_ROOT / "tools" / "safety" / "safety_mechanisms.yaml"
BASELINE = REPO_ROOT / "docs" / "safety" / "fmeda" / "baseline.json"


def test_discover_modules_finds_at_least_dms_fusion_and_npu_top():
    modules = discover_modules(FAILURE_MODES)
    assert "dms_fusion" in modules
    assert "npu_top" in modules


def test_snapshot_all_returns_one_per_module():
    snaps = snapshot_all(FAILURE_MODES, MECHANISMS, "ASIL-B")
    modules = {s.module for s in snaps}
    assert "dms_fusion" in modules
    assert "npu_top" in modules
    # Each snapshot has the requested asil target.
    assert all(s.asil_target == "ASIL-B" for s in snaps)


def test_emit_baseline_round_trip(tmp_path):
    snaps = snapshot_all(FAILURE_MODES, MECHANISMS, "ASIL-B")
    json_text = emit_baseline_json(snaps, "ASIL-B")
    p = tmp_path / "baseline.json"
    p.write_text(json_text, encoding="utf-8")
    loaded = load_baseline(p)
    assert "dms_fusion" in loaded
    assert loaded["dms_fusion"].spfm_pct == pytest.approx(snaps[0].spfm_pct if snaps[0].module == "dms_fusion" else next(s for s in snaps if s.module == "dms_fusion").spfm_pct)


def test_baseline_file_committed_and_parses():
    assert BASELINE.exists(), f"committed baseline missing at {BASELINE}"
    loaded = load_baseline(BASELINE)
    assert "dms_fusion" in loaded
    assert "npu_top" in loaded


def test_compare_passes_when_within_threshold():
    base = {
        "m": ModuleSnapshot("m", "ASIL-B", spfm_pct=90.0, lfm_pct=70.0, pmhf_fit=10.0),
    }
    cur = [ModuleSnapshot("m", "ASIL-B", spfm_pct=89.5, lfm_pct=70.0, pmhf_fit=10.0)]
    regressions = compare(cur, base, max_drop_pct=1.0, max_pmhf_rise_fit=0.001)
    assert regressions == []


def test_compare_flags_spfm_drop_beyond_threshold():
    base = {"m": ModuleSnapshot("m", "ASIL-B", 90.0, 70.0, 10.0)}
    cur = [ModuleSnapshot("m", "ASIL-B", 85.0, 70.0, 10.0)]  # drop 5 pp
    regressions = compare(cur, base, max_drop_pct=1.0, max_pmhf_rise_fit=0.001)
    assert len(regressions) == 1
    assert "SPFM dropped" in regressions[0]
    assert "85.00 %" in regressions[0]


def test_compare_flags_lfm_drop():
    base = {"m": ModuleSnapshot("m", "ASIL-B", 90.0, 70.0, 10.0)}
    cur = [ModuleSnapshot("m", "ASIL-B", 90.0, 60.0, 10.0)]  # drop 10 pp
    regressions = compare(cur, base, max_drop_pct=1.0, max_pmhf_rise_fit=0.001)
    assert any("LFM dropped" in r for r in regressions)


def test_compare_flags_pmhf_rise():
    base = {"m": ModuleSnapshot("m", "ASIL-B", 90.0, 70.0, 10.0)}
    cur = [ModuleSnapshot("m", "ASIL-B", 90.0, 70.0, 11.0)]  # +1 FIT
    regressions = compare(cur, base, max_drop_pct=1.0, max_pmhf_rise_fit=0.001)
    assert any("PMHF rose" in r for r in regressions)


def test_compare_ignores_new_module_in_current():
    """A new module in `current` not present in baseline does not fail."""
    base = {"m1": ModuleSnapshot("m1", "ASIL-B", 90.0, 70.0, 10.0)}
    cur = [
        ModuleSnapshot("m1", "ASIL-B", 90.0, 70.0, 10.0),
        ModuleSnapshot("m2", "ASIL-B", 50.0, 30.0, 100.0),  # new
    ]
    regressions = compare(cur, base, max_drop_pct=1.0, max_pmhf_rise_fit=0.001)
    assert regressions == []


def test_committed_baseline_matches_current_state_no_drift():
    """Sanity: the baseline JSON in the repo matches the current FMEDA output.

    This guards against accidental drift — if someone tweaks the
    failure_modes / mechanisms YAML without regenerating the baseline,
    this test fails and reminds them to refresh.
    """
    cur = snapshot_all(FAILURE_MODES, MECHANISMS, "ASIL-B")
    base = load_baseline(BASELINE)
    for c in cur:
        if c.module not in base:
            pytest.fail(f"module {c.module} present in current but not baseline")
        b = base[c.module]
        assert c.spfm_pct == pytest.approx(b.spfm_pct, abs=0.01), (
            f"{c.module}: SPFM drift {b.spfm_pct} → {c.spfm_pct}; "
            f"regenerate baseline if change is intentional"
        )
        assert c.lfm_pct == pytest.approx(b.lfm_pct, abs=0.01), (
            f"{c.module}: LFM drift"
        )
        assert c.pmhf_fit == pytest.approx(b.pmhf_fit, abs=1e-6), (
            f"{c.module}: PMHF drift"
        )


def test_snapshot_includes_aggregate_by_default():
    snaps = snapshot_all(FAILURE_MODES, MECHANISMS, "ASIL-B")
    modules = {s.module for s in snaps}
    assert AGGREGATE_KEY in modules, "aggregate row missing from snapshot"


def test_snapshot_omits_aggregate_when_requested():
    snaps = snapshot_all(FAILURE_MODES, MECHANISMS, "ASIL-B", include_aggregate=False)
    modules = {s.module for s in snaps}
    assert AGGREGATE_KEY not in modules


def test_aggregate_in_committed_baseline():
    """Aggregate row must be in the committed baseline so CI catches drift."""
    base = load_baseline(BASELINE)
    assert AGGREGATE_KEY in base, (
        "aggregate row missing from baseline.json; regenerate with "
        "`python -m tools.safety.regress_check --emit-baseline > docs/safety/fmeda/baseline.json`"
    )


def test_aggregate_baseline_matches_recomputed():
    """Aggregate baseline must equal the value recomputed from per-module sums."""
    cur = snapshot_all(FAILURE_MODES, MECHANISMS, "ASIL-B")
    agg_cur = next(s for s in cur if s.module == AGGREGATE_KEY)
    base = load_baseline(BASELINE)
    agg_base = base[AGGREGATE_KEY]
    assert agg_cur.spfm_pct == pytest.approx(agg_base.spfm_pct, abs=0.01)
    assert agg_cur.lfm_pct == pytest.approx(agg_base.lfm_pct, abs=0.01)
    assert agg_cur.pmhf_fit == pytest.approx(agg_base.pmhf_fit, abs=1e-6)


def test_compare_flags_aggregate_drift():
    """Regression: a per-module change small enough to slip the per-module
    threshold can still produce a measurable aggregate drift. Verify the
    aggregate-row check catches it."""
    base = {
        "m1": ModuleSnapshot("m1", "ASIL-B", 90.0, 70.0, 10.0),
        "m2": ModuleSnapshot("m2", "ASIL-B", 90.0, 70.0, 10.0),
        AGGREGATE_KEY: ModuleSnapshot(AGGREGATE_KEY, "ASIL-B", 90.0, 70.0, 20.0),
    }
    # Each module drops 0.5 pp — below per-module 1 pp threshold — but
    # the synthetic aggregate row "drops" 2 pp (caught).
    cur = [
        ModuleSnapshot("m1", "ASIL-B", 89.5, 70.0, 10.0),
        ModuleSnapshot("m2", "ASIL-B", 89.5, 70.0, 10.0),
        ModuleSnapshot(AGGREGATE_KEY, "ASIL-B", 88.0, 70.0, 20.0),
    ]
    regs = compare(cur, base, max_drop_pct=1.0, max_pmhf_rise_fit=0.001)
    assert any("aggregate" in r and "SPFM" in r for r in regs)
