# Fault-Injection Harness

cocotb + Verilator runner for the fault-injection campaigns required by ISO 26262-5 §11 and ISO 26262-11 §7.

## Layout

```
sim/fault_injection/
├── README.md            ← this file
├── Makefile             ← per-campaign run targets
├── runner.py            ← cocotb test entrypoint (resolves spec + applies forces)
├── campaigns/
│   └── tmr_voter_seu_1k.yaml   ← W4 sample campaign (smallest target)
└── out/                 ← cocotb writes one .jsonl per run; gitignored
```

## What runs where

| Layer | Where it lives | Runs on |
|---|---|---|
| Campaign planning + validation | `tools/safety/fault_injection.py` | Windows / Linux / macOS (no cocotb dep) |
| Cocotb test driver | `sim/fault_injection/runner.py` | WSL Ubuntu 22.04 + Verilator 5.030 + cocotb 2.0.1 |
| Aggregation + markdown report | `tools/safety/fault_injection.py` | Windows / Linux / macOS |

The split is deliberate: the planner + aggregator have unit tests
(`tests/test_fault_injection.py`) that run on Windows. Only the cocotb
adapter requires WSL.

## End-to-end flow

```bash
# 1. Author / edit a campaign on any platform
$EDITOR sim/fault_injection/campaigns/tmr_voter_seu_1k.yaml

# 2. Validate the plan (Windows OK)
python -m tools.safety.fault_injection \
    --campaign sim/fault_injection/campaigns/tmr_voter_seu_1k.yaml \
    --results /dev/null --output /dev/null  # validates parse

# 3. Run the campaign (WSL required)
cd sim/fault_injection
make CAMPAIGN=tmr_voter_seu_1k
# produces out/tmr_voter_seu_1k.jsonl

# 4. Aggregate into the safety case (any platform)
python -m tools.safety.fault_injection \
    --campaign sim/fault_injection/campaigns/tmr_voter_seu_1k.yaml \
    --results sim/fault_injection/out/tmr_voter_seu_1k.jsonl \
    --output docs/safety/fault_injection/tmr_voter_seu_1k.md
```

## Campaign YAML schema

```yaml
name: tmr_voter_seu_1k
target_module: tmr_voter
oracle_signal: u_dut.tmr_fault
expected_safe_response: >
  tmr_fault asserts on the lane that disagrees within 1 cycle of vote
fmeda_failure_mode_ids:
  - dms_fusion.dal_lane.seu  # closes one row of the FMEDA
injections:
  - target_path: u_dut.lane_a_reg
    bit_index: 3
    kind: bit_flip          # one of: stuck_0 stuck_1 bit_flip transient
    start_cycle: 100
    duration_cycles: 1
    expected_detection: true
```

See `tools/safety/fault_injection.py` `InjectionSpec` for the full
dataclass and `InjectionKind` for the supported kinds.

## Verilator caveat

`cocotb.handle.SimHandle.value` setters apply via Verilator's `Force`
backend. Per `memory/feedback_verilator_force.md`:

> Verilator's `Force` does not reach **port-constant** drivers (memory pin
> in older Verilator). Pick *internal* nets for injection, not module
> ports.

When in doubt, target a register inside the DUT (`u_dut.lane_a_reg`),
not a top-level port (`lane_a`).

## Status (2026-04-20)

- ✅ Planner + aggregator (`tools/safety/fault_injection.py`)
- ✅ Sample campaign (`campaigns/tmr_voter_seu_1k.yaml`)
- 🟡 Cocotb runner skeleton (`runner.py`) — needs WSL run to validate
- 🟡 Makefile — needs WSL run to validate
- pending: campaigns for ECC SECDED, dms_fusion, npu_top SRAM (per `docs/safety/fault_injection/README.md` schedule)
