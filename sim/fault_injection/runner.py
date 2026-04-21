"""cocotb fault-injection runner.

Reads a campaign YAML, applies each injection in turn against the
DUT, samples the oracle signal, and emits one JSON record per
injection to ``out/<campaign>.jsonl`` for the aggregator
(``tools/safety/fault_injection.py``) to consume.

Runs only under cocotb (WSL Ubuntu 22.04 + Verilator 5.030 +
cocotb 2.0.1 per `memory/wsl_verilator_setup.md`).

Invoked via the Makefile:

    cd sim/fault_injection
    make CAMPAIGN=tmr_voter_seu_1k

The Makefile sets the COCOTB_TOPLEVEL, COCOTB_TEST_MODULES, and
ASTRACORE_FI_CAMPAIGN environment variables; this module reads
them on import.

NOTE: This file is an executable scaffold. It runs end-to-end only
with cocotb installed in WSL. The campaign planner + aggregator
(`tools/safety/fault_injection.py`) is unit-tested independently on
Windows so campaign authoring + result interpretation do not require
a WSL round-trip.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# This file is `sim/fault_injection/runner.py`; the planner is at
# `tools/safety/fault_injection.py`. Add repo root to sys.path so the
# planner module imports cleanly under cocotb's runtime.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.safety.fault_injection import (  # noqa: E402
    Campaign,
    InjectionKind,
    InjectionSpec,
    load_campaign,
)


# cocotb is only importable inside the cocotb test runner; guard the
# import so this file can be linted / partially imported on Windows.
try:
    import cocotb  # type: ignore[import-untyped]
    from cocotb.clock import Clock  # type: ignore[import-untyped]
    from cocotb.triggers import RisingEdge, Timer  # type: ignore[import-untyped]
    _HAS_COCOTB = True
except ImportError:  # pragma: no cover
    _HAS_COCOTB = False


def _resolve_handle(dut, hierarchical_path: str):
    """Walk ``u_dut.lane_a_reg`` style paths down from ``dut``.

    The first component is the toplevel (``u_dut``) and is dropped if
    it matches the dut's name; otherwise we treat the whole path as
    relative to ``dut``.
    """
    parts = hierarchical_path.split(".")
    if parts and parts[0] == dut._name:
        parts = parts[1:]
    h = dut
    for p in parts:
        h = getattr(h, p)
    return h


def _apply_injection(handle, spec: InjectionSpec) -> None:
    """Force the perturbation. Caller is responsible for releasing."""
    if spec.bit_index is None:
        # Whole-net injection
        if spec.kind == InjectionKind.STUCK_0:
            handle.value = 0
        elif spec.kind == InjectionKind.STUCK_1:
            handle.value = 1
        elif spec.kind in (InjectionKind.BIT_FLIP, InjectionKind.TRANSIENT):
            handle.value = (~int(handle.value)) & 0x1
    else:
        cur = int(handle.value)
        mask = 1 << spec.bit_index
        if spec.kind == InjectionKind.STUCK_0:
            new = cur & ~mask
        elif spec.kind == InjectionKind.STUCK_1:
            new = cur | mask
        elif spec.kind in (InjectionKind.BIT_FLIP, InjectionKind.TRANSIENT):
            new = cur ^ mask
        else:  # pragma: no cover
            raise ValueError(f"unsupported injection kind: {spec.kind}")
        handle.value = new


def _release_injection(handle, original_value: int) -> None:
    """Restore the pre-injection value (cocotb has no Release for value=)."""
    handle.value = original_value


@cocotb.test() if _HAS_COCOTB else (lambda fn: fn)  # type: ignore[misc]
async def run_campaign(dut):  # pragma: no cover  (only runs in WSL+cocotb)
    """Cocotb test entrypoint. Iterates the campaign and emits JSONL."""
    campaign_yaml = os.environ.get("ASTRACORE_FI_CAMPAIGN")
    if not campaign_yaml:
        raise RuntimeError(
            "ASTRACORE_FI_CAMPAIGN env var must point to the campaign YAML"
        )
    campaign = load_campaign(Path(campaign_yaml))
    out_dir = Path(os.environ.get("ASTRACORE_FI_OUTDIR", "out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{campaign.name}.jsonl"

    # Free-running 100 MHz clock so cycles map to a known time.
    # cocotb 2.0 uses unit= (singular); 1.9 used units= (plural).
    clk = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clk.start())

    # Start with a clean reset.
    dut.rst_n.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1

    oracle = _resolve_handle(dut, campaign.oracle_signal)

    with out_path.open("w", encoding="utf-8") as f:
        for idx, spec in enumerate(campaign):
            handle = _resolve_handle(dut, spec.target_path)
            # Save original for release
            await RisingEdge(dut.clk)  # settle into a known cycle
            for _ in range(spec.start_cycle):
                await RisingEdge(dut.clk)
            original = int(handle.value)
            _apply_injection(handle, spec)
            detected = False
            detection_cycle = None
            for k in range(spec.duration_cycles):
                await RisingEdge(dut.clk)
                if int(oracle.value) and detection_cycle is None:
                    detected = True
                    detection_cycle = spec.start_cycle + k
            _release_injection(handle, original)
            # Settle one more cycle to capture late assertions
            await RisingEdge(dut.clk)
            if int(oracle.value) and detection_cycle is None:
                detected = True
                detection_cycle = spec.start_cycle + spec.duration_cycles

            # Did the injection actually perturb anything? Use the
            # voted output as a proxy (only meaningful for tmr_voter;
            # other DUTs will need their own perturbation oracle).
            perturbed = bool(int(handle.value) != original) or detected

            record = {
                "spec_index": idx,
                "target_path": spec.target_path,
                "detected": detected,
                "detection_cycle": detection_cycle,
                "perturbed_outputs": perturbed,
            }
            f.write(json.dumps(record) + "\n")

    dut._log.info("Wrote %d records to %s", len(campaign), out_path)
