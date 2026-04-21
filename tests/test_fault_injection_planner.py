"""Tests for the fault-injection planner + aggregator.

These tests cover the *RTL-agnostic* half of the harness — the cocotb
side lives in sim/fault_injection/runner.py and runs only under WSL.
The split is documented in tools/safety/fault_injection.py module
docstring.

Coverage:
- InjectionSpec validation (target path, bit index, kind, cycles)
- Campaign YAML parsing (incl. error paths)
- CampaignResult aggregation: detected / missed / benign / coverage
- Markdown render presence-checks for required sections
- The shipped sample campaign (tmr_voter_seu_1k) parses cleanly
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.safety.fault_injection import (
    Campaign,
    CampaignResult,
    InjectionKind,
    InjectionResult,
    InjectionSpec,
    _campaign_from_dict,
    load_campaign,
    load_results_jsonl,
    render_markdown,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CAMPAIGN = REPO_ROOT / "sim" / "fault_injection" / "campaigns" / "tmr_voter_seu_1k.yaml"


# ---------------------------------------------------------------------------
# InjectionSpec validation
# ---------------------------------------------------------------------------


def test_injection_spec_basic():
    s = InjectionSpec(
        target_path="u_dut.lane_a_reg",
        bit_index=3,
        kind=InjectionKind.BIT_FLIP,
        start_cycle=10,
        duration_cycles=1,
    )
    assert s.expected_detection is True
    assert s.kind == InjectionKind.BIT_FLIP


def test_target_path_must_be_hierarchical():
    with pytest.raises(ValueError, match="hierarchical"):
        InjectionSpec(
            target_path="lane_a",  # no dot
            bit_index=0,
            kind=InjectionKind.STUCK_0,
            start_cycle=0,
            duration_cycles=1,
        )


def test_negative_bit_index_rejected():
    with pytest.raises(ValueError, match="bit_index"):
        InjectionSpec(
            target_path="u_dut.x",
            bit_index=-1,
            kind=InjectionKind.STUCK_0,
            start_cycle=0,
            duration_cycles=1,
        )


def test_zero_duration_rejected():
    with pytest.raises(ValueError, match="duration_cycles"):
        InjectionSpec(
            target_path="u_dut.x",
            bit_index=None,
            kind=InjectionKind.TRANSIENT,
            start_cycle=0,
            duration_cycles=0,
        )


def test_negative_start_cycle_rejected():
    with pytest.raises(ValueError, match="start_cycle"):
        InjectionSpec(
            target_path="u_dut.x",
            bit_index=None,
            kind=InjectionKind.TRANSIENT,
            start_cycle=-5,
            duration_cycles=1,
        )


def test_kind_must_be_enum():
    with pytest.raises(ValueError, match="InjectionKind"):
        InjectionSpec(
            target_path="u_dut.x",
            bit_index=None,
            kind="bit_flip",  # type: ignore[arg-type]  # str not allowed
            start_cycle=0,
            duration_cycles=1,
        )


def test_whole_net_injection_allows_none_bit_index():
    s = InjectionSpec(
        target_path="u_dut.scalar_flag",
        bit_index=None,
        kind=InjectionKind.STUCK_1,
        start_cycle=0,
        duration_cycles=10,
    )
    assert s.bit_index is None


# ---------------------------------------------------------------------------
# Campaign validation + parsing
# ---------------------------------------------------------------------------


def _basic_campaign(injections=None) -> Campaign:
    return Campaign(
        name="test",
        target_module="tmr_voter",
        oracle_signal="u_dut.fault_a",
        expected_safe_response="fault_a asserts within 1 cycle",
        injections=tuple(
            injections
            or [
                InjectionSpec(
                    target_path="u_dut.lane_a",
                    bit_index=0,
                    kind=InjectionKind.BIT_FLIP,
                    start_cycle=10,
                    duration_cycles=1,
                )
            ]
        ),
    )


def test_campaign_iter_and_len():
    c = _basic_campaign(
        injections=[
            InjectionSpec("u.x", 0, InjectionKind.BIT_FLIP, 0, 1),
            InjectionSpec("u.y", 1, InjectionKind.STUCK_1, 0, 1),
        ]
    )
    assert len(c) == 2
    assert [s.bit_index for s in c] == [0, 1]


def test_campaign_requires_injections():
    with pytest.raises(ValueError, match="at least one injection"):
        Campaign(
            name="empty",
            target_module="foo",
            oracle_signal="u_dut.fault",
            expected_safe_response="x",
            injections=(),
        )


def test_campaign_oracle_must_be_hierarchical():
    with pytest.raises(ValueError, match="hierarchical"):
        Campaign(
            name="x",
            target_module="foo",
            oracle_signal="fault",  # no dot
            expected_safe_response="x",
            injections=(InjectionSpec("u.x", 0, InjectionKind.STUCK_0, 0, 1),),
        )


def test_campaign_from_dict_parses_full_yaml_shape():
    raw = {
        "name": "tmr_voter_test",
        "target_module": "tmr_voter",
        "oracle_signal": "u_dut.fault_a",
        "expected_safe_response": "asserts on disagreement",
        "fmeda_failure_mode_ids": ["dms_fusion.dal_lane.seu"],
        "injections": [
            {
                "target_path": "u_dut.lane_a",
                "bit_index": 3,
                "kind": "bit_flip",
                "start_cycle": 100,
                "duration_cycles": 1,
                "expected_detection": True,
            }
        ],
    }
    c = _campaign_from_dict(raw)
    assert c.name == "tmr_voter_test"
    assert c.fmeda_failure_mode_ids == ("dms_fusion.dal_lane.seu",)
    assert c.injections[0].kind == InjectionKind.BIT_FLIP


def test_campaign_from_dict_reports_injection_index_on_error():
    raw = {
        "name": "x",
        "target_module": "y",
        "oracle_signal": "u.f",
        "expected_safe_response": "z",
        "injections": [
            {
                "target_path": "u.lane",
                "bit_index": 0,
                "kind": "not-a-real-kind",
                "start_cycle": 0,
                "duration_cycles": 1,
            }
        ],
    }
    with pytest.raises(ValueError, match="injection #0"):
        _campaign_from_dict(raw)


def test_sample_campaign_yaml_parses():
    """The shipped sample tmr_voter_seu_1k.yaml must always parse."""
    c = load_campaign(SAMPLE_CAMPAIGN)
    assert c.name == "tmr_voter_seu_1k"
    assert c.target_module == "tmr_voter"
    assert "dms_fusion.dal_lane.seu" in c.fmeda_failure_mode_ids
    # At least one stuck-at injection (longer duration) and at least one
    # not-expected-to-detect (CCF) injection.
    assert any(s.kind == InjectionKind.STUCK_1 for s in c.injections)
    assert any(s.expected_detection is False for s in c.injections)
    # Verify the testbench module-name convention: tb_<dut>_fi
    assert all(
        s.target_path.startswith("tb_tmr_voter_fi.") for s in c.injections
    ), "tmr_voter campaign should reference tb_tmr_voter_fi-prefixed paths"


def test_all_shipped_campaigns_parse():
    """Every YAML in sim/fault_injection/campaigns/ must parse without error.

    This is a structural regression gate — if anyone drops a malformed
    YAML or renames a field, this fires immediately on Windows without
    needing a WSL cocotb run.
    """
    campaigns_dir = REPO_ROOT / "sim" / "fault_injection" / "campaigns"
    yamls = sorted(campaigns_dir.glob("*.yaml"))
    assert len(yamls) >= 4, (
        f"expected at least 4 campaign YAMLs (tmr_voter, ecc_secded, "
        f"dms_fusion, safe_state_controller), found {len(yamls)}: {yamls}"
    )
    for p in yamls:
        c = load_campaign(p)
        assert len(c) > 0, f"{p.name}: no injections"
        # Every campaign must declare at least one FMEDA failure-mode ID
        # for traceability — otherwise the report can't tie back to the
        # safety case.
        assert len(c.fmeda_failure_mode_ids) > 0, (
            f"{p.name}: no fmeda_failure_mode_ids declared — campaign "
            f"results would be untraceable to FMEDA"
        )


def test_all_campaign_fmeda_ids_exist_in_failure_modes_yaml():
    """Every fmeda_failure_mode_id named in a campaign must exist in
    failure_modes.yaml — otherwise the traceability is broken."""
    import yaml as _yaml
    fm_yaml = _yaml.safe_load(
        (REPO_ROOT / "tools" / "safety" / "failure_modes.yaml").read_text(encoding="utf-8")
    )
    known_fm_ids = {entry["id"] for entry in fm_yaml.get("failure_modes", [])}
    campaigns_dir = REPO_ROOT / "sim" / "fault_injection" / "campaigns"
    for p in sorted(campaigns_dir.glob("*.yaml")):
        c = load_campaign(p)
        for fm_id in c.fmeda_failure_mode_ids:
            assert fm_id in known_fm_ids, (
                f"{p.name}: declares fmeda_failure_mode_id {fm_id!r} "
                f"that is not in tools/safety/failure_modes.yaml — "
                f"traceability broken; either add the failure mode or "
                f"correct the campaign reference"
            )


# ---------------------------------------------------------------------------
# CampaignResult aggregation
# ---------------------------------------------------------------------------


def _result(detected: bool, perturbed: bool, idx: int = 0, target="u.x") -> InjectionResult:
    return InjectionResult(
        spec_index=idx,
        target_path=target,
        detected=detected,
        detection_cycle=10 if detected else None,
        perturbed_outputs=perturbed,
    )


def test_coverage_full_when_all_perturbed_detected():
    c = _basic_campaign()
    cr = CampaignResult(
        campaign=c,
        per_injection=[_result(True, True), _result(True, True)],
    )
    assert cr.detected == 2
    assert cr.missed == 0
    assert cr.coverage_pct == 100.0


def test_coverage_partial_when_some_missed():
    c = _basic_campaign()
    cr = CampaignResult(
        campaign=c,
        per_injection=[
            _result(True, True),
            _result(False, True),
            _result(True, True),
            _result(False, True),
        ],
    )
    assert cr.detected == 2
    assert cr.missed == 2
    assert cr.coverage_pct == 50.0


def test_benign_injections_excluded_from_coverage():
    """Injections that perturbed nothing are not counted in DC."""
    c = _basic_campaign()
    cr = CampaignResult(
        campaign=c,
        per_injection=[
            _result(True, True),
            _result(False, False),  # benign — does not pull DC down
            _result(False, False),
        ],
    )
    assert cr.benign == 2
    assert cr.perturbed == 1
    assert cr.coverage_pct == 100.0


def test_false_positives_counted_separately():
    """Detection without perturbation is a false positive, not coverage."""
    c = _basic_campaign()
    cr = CampaignResult(
        campaign=c,
        per_injection=[_result(True, False)],  # oracle fired but no perturbation
    )
    assert cr.false_positives == 1
    assert cr.detected == 0  # only counts perturbed-and-detected
    assert cr.coverage_pct == 100.0  # vacuous (perturbed == 0)


def test_zero_perturbed_is_vacuously_full_coverage():
    c = _basic_campaign()
    cr = CampaignResult(campaign=c, per_injection=[_result(False, False)])
    assert cr.coverage_pct == 100.0


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------


def test_jsonl_loader_round_trip(tmp_path):
    c = _basic_campaign()
    rec = {
        "spec_index": 0,
        "target_path": "u_dut.lane_a_reg",
        "detected": True,
        "detection_cycle": 100,
        "perturbed_outputs": True,
    }
    p = tmp_path / "results.jsonl"
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    cr = load_results_jsonl(p, c)
    assert cr.total_runs == 1
    assert cr.detected == 1


def test_jsonl_loader_skips_blank_lines(tmp_path):
    c = _basic_campaign()
    rec = {
        "spec_index": 0,
        "target_path": "u_dut.x",
        "detected": False,
        "detection_cycle": None,
        "perturbed_outputs": True,
    }
    p = tmp_path / "results.jsonl"
    p.write_text("\n" + json.dumps(rec) + "\n\n", encoding="utf-8")
    cr = load_results_jsonl(p, c)
    assert cr.total_runs == 1


def test_jsonl_loader_reports_line_number_on_error(tmp_path):
    c = _basic_campaign()
    p = tmp_path / "bad.jsonl"
    p.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=":1:"):
        load_results_jsonl(p, c)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def test_render_markdown_has_required_sections():
    c = _basic_campaign()
    cr = CampaignResult(campaign=c, per_injection=[_result(True, True)])
    md = render_markdown(cr, doc_id="TEST", generated_on="2026-04-20")
    for required in (
        f"# Fault-Injection Campaign — {c.name}",
        "## 1. Aggregate",
        "## 2. Per-injection results",
        "## 4. Reproduce",
        "Diagnostic coverage",
        "ISO 26262-5 §11",
    ):
        assert required in md, f"missing: {required}"


def test_render_markdown_includes_traceability_when_fmeda_ids_present():
    c = Campaign(
        name="x",
        target_module="m",
        oracle_signal="u_dut.fault",
        expected_safe_response="r",
        injections=(InjectionSpec("u.x", 0, InjectionKind.BIT_FLIP, 0, 1),),
        fmeda_failure_mode_ids=("dms_fusion.dal_lane.seu",),
    )
    cr = CampaignResult(campaign=c, per_injection=[_result(True, True)])
    md = render_markdown(cr, doc_id="T", generated_on="2026-04-20")
    assert "## 3. FMEDA traceability" in md
    assert "dms_fusion.dal_lane.seu" in md
    assert "5 percentage points" in md


def test_render_markdown_omits_traceability_section_when_no_ids():
    c = _basic_campaign()  # no fmeda_failure_mode_ids
    cr = CampaignResult(campaign=c, per_injection=[_result(True, True)])
    md = render_markdown(cr, doc_id="T", generated_on="2026-04-20")
    assert "## 3. FMEDA traceability" not in md


def test_truncation_note_when_more_than_50_results():
    c = _basic_campaign(
        injections=[
            InjectionSpec(f"u.x{i % 4}", i % 8, InjectionKind.BIT_FLIP, 0, 1)
            for i in range(60)
        ]
    )
    cr = CampaignResult(
        campaign=c,
        per_injection=[_result(True, True, idx=i) for i in range(60)],
    )
    md = render_markdown(cr, doc_id="T", generated_on="2026-04-20")
    assert "10 more rows omitted" in md
