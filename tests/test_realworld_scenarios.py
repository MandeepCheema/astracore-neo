"""Smoke tests for scripts/run_realworld_scenarios.py.

The full 5-scenario driver takes ~48 s; too slow for CI. We verify:
- The script imports cleanly.
- Scenario 4 (alarm, <1 s) runs end-to-end and produces a valid result.
- The rendered markdown doesn't lose scenarios.

Scenarios 1/2/3/5 exercised by the on-demand full run under
``reports/realworld_scenarios/``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "run_realworld_scenarios.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location(
        "run_realworld_scenarios", SCRIPT,
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["run_realworld_scenarios"] = m
    spec.loader.exec_module(m)
    return m


def test_module_imports(mod):
    # 5 scenarios registered.
    assert len(mod.SCENARIOS) == 5
    names = [s[0] for s in mod.SCENARIOS]
    assert "image_inference" in names
    assert "yolo_detection_sweep" in names
    assert "perception_presets" in names
    assert "alarm_scenarios" in names
    assert "yolo_resolution_sweep" in names


def test_percentile_helper(mod):
    s = mod._percentile_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["n"] == 5
    assert s["mean_ms"] == pytest.approx(3.0)
    assert s["p50_ms"] == pytest.approx(3.0)


def test_scenario_4_alarm_end_to_end(mod):
    """Scenario 4 is fast + self-contained; run it and validate shape."""
    data = mod.scenario_4_alarm_scenarios()
    assert data["scenario"] == "fusion_alarm_scenarios"
    subs = data["subscenarios"]
    assert len(subs) == 4

    # Parking crawl always yields exactly the histogram we pinned in
    # the alarm regression test.
    park = next(s for s in subs
                if "parking" in s["subscenario"])
    assert park["histogram"]["CRITICAL"] == 1
    assert park["histogram"]["WARNING"] == 1
    assert park["histogram"]["CAUTION"] == 2

    # Emergency brake gates: CRITICAL >= 1
    em = next(s for s in subs if "emergency_brake" in s["subscenario"])
    assert em["histogram"]["CRITICAL"] >= 1
    assert em.get("passed") is True

    # US dropout gate: CAUTION >= 1
    dr = next(s for s in subs if "us_dropout" in s["subscenario"])
    assert dr["histogram"]["CAUTION"] >= 1
    assert dr.get("passed") is True


def test_scenario_4_markdown_renders(mod):
    data = mod.scenario_4_alarm_scenarios()
    md = mod.render_s4(data)
    assert "parking_crawl_5_to_0p5_kph" in md
    assert "highway_cruise_100_kph_clear_road" in md
    assert "emergency_brake_60_kph_to_35_kph" in md
    assert "us_dropout_lidar_only_detection" in md
    assert "PASS" in md  # all four gates pass today


def test_scenario_5_yolo_baseline_is_quick(mod):
    """Scenario 5 runs 30 iters of static-640 YOLO — should take <5 s."""
    data = mod.scenario_5_yolo_resolution_sweep()
    if "skipped" in data:
        pytest.skip(data["skipped"])
    s = data["baseline_640_static"]
    assert s["n"] == 30
    assert s["p50_ms"] > 0
    assert s["p99_ms"] >= s["p50_ms"]


def test_committed_report_exists():
    """The committed report under reports/realworld_scenarios/ must
    have the 5 sub-reports + summary."""
    base = REPO / "reports" / "realworld_scenarios"
    for i, name in [
        (1, "image_inference"),
        (2, "yolo_detection_sweep"),
        (3, "perception_presets"),
        (4, "alarm_scenarios"),
        (5, "yolo_resolution_sweep"),
    ]:
        assert (base / f"{i}_{name}.json").exists(), f"missing {i}_{name}.json"
        assert (base / f"{i}_{name}.md").exists(), f"missing {i}_{name}.md"
    assert (base / "summary.md").exists()
    assert (base / "README.md").exists()


def test_summary_documents_highway_bug_and_fix():
    """Summary must document the bug AND the applied fix.

    The original version of this test gated on the bug being present;
    after the fix landed 2026-04-20, we gate on BOTH the bug narrative
    and the fix-applied note existing so nobody silently rewrites the
    history.
    """
    summary = (REPO / "reports" / "realworld_scenarios" / "summary.md").read_text(
        encoding="utf-8"
    )
    low = summary.lower()
    assert "highway" in low
    assert "bug" in low
    assert "no-echo" in low or "no echo" in low
    # Fix status must be reachable.
    assert "fix applied" in low or "fix landed" in low or "done" in low
