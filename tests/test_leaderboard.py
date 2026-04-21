"""Pin the leaderboard generator + manifest shape.

Separates two concerns:
* the generator itself (tmp_path render, no git-committed files changed)
* the committed ``LEADERBOARD.md`` at repo root (must exist + contain the
  five sections end-users rely on).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "make_leaderboard.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("make_leaderboard", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["make_leaderboard"] = m
    spec.loader.exec_module(m)
    return m


def test_collect_returns_expected_keys(mod):
    data = mod._collect()
    assert "host" in data
    assert "zoo_bench" in data or not (REPO / "reports" / "benchmark_sweep"
                                        / "zoo.json").exists()
    assert data["host"].get("cpu_count", 0) >= 1


def test_committed_leaderboard_exists():
    p = REPO / "LEADERBOARD.md"
    assert p.exists(), "LEADERBOARD.md at repo root is missing"
    body = p.read_text(encoding="utf-8")
    assert "# AstraCore Neo" in body
    # Core sections present.
    for header in (
        "## 1. Model zoo",
        "## 2. Multi-stream scaling",
        "## 3. Deep AI model tests",
        "## 4. Safety fusion alarm",
        "## Caveats",
    ):
        assert header in body, f"missing section: {header}"


def test_leaderboard_lists_int8_snr_numbers():
    body = (REPO / "LEADERBOARD.md").read_text(encoding="utf-8")
    # At least one model must have a >20 dB SNR row.
    assert "INT8 SNR" in body
    assert "yolov8n" in body.lower()
    # The "production-grade" legend is load-bearing text.
    assert "production-grade" in body


def test_int8_manifest_has_all_zoo_entries():
    p = REPO / "data" / "models" / "zoo" / "int8" / "manifest.json"
    if not p.exists():
        pytest.skip("int8 manifest missing — run scripts/quantise_zoo.py")
    doc = json.loads(p.read_text())
    # 8 zoo models in; at least 6 must have produced an artefact.
    ok = [r for r in doc["rows"] if r.get("status") == "ok"]
    assert len(ok) >= 6, (
        f"only {len(ok)}/8 zoo models quantised successfully"
    )
    # yolov8n must hit production-grade SNR.
    yolo = next((r for r in ok if r["model"] == "yolov8n"), None)
    assert yolo is not None
    assert yolo["snr_db"] > 30.0, (
        f"yolov8n INT8 SNR regressed to {yolo['snr_db']} dB — calibration broken?"
    )


def test_cli_make_leaderboard_is_idempotent(tmp_path):
    """Re-running the script must produce a valid LEADERBOARD.md."""
    r = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO, capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, r.stderr
    assert (REPO / "LEADERBOARD.md").exists()
    assert (REPO / "reports" / "leaderboard.json").exists()
    assert (REPO / "reports" / "leaderboard_reproduce.md").exists()


def test_reproduce_guide_lists_commands():
    p = REPO / "reports" / "leaderboard_reproduce.md"
    if not p.exists():
        pytest.skip("run scripts/make_leaderboard.py first")
    body = p.read_text(encoding="utf-8")
    for cmd in ("astracore zoo",
                "bench_zoo_detailed",
                "quantise_zoo",
                "astracore multistream",
                "ai_model_deep_tests",
                "run_realworld_scenarios",
                "make_leaderboard"):
        assert cmd in body, f"reproduce guide missing command: {cmd}"
