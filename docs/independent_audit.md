# AstraCore Neo — Independent Code Audit

*Conducted 2026-04-17. This audit is deliberately decoupled from the
memory system's claims — it verifies what actually exists in the repo,
runs what actually passes, and flags gaps the memory / prior
documentation under-represents. The only input from memory is the
project vision statement itself; everything else is verified against
source.*

---

## Summary

Twelve concrete findings below, grouped A–L. **None invalidate the
top-level architectural claims**, but several change the fidelity of
claims that would appear in investor diligence. The three most
material:

1. **1 test fails in batch, passes in isolation.** Fusion-top
   `test_ego_motion_chain` — test-ordering state-leakage, not a
   silicon bug, but publicly the suite is not 36/36 clean.
2. **Zero SystemVerilog assertions across 9,085 lines of RTL.** Phase
   F (formal verification) starts from a blank slate — no invariants
   yet in the RTL for any of the ASIL-D safety properties.
3. **`astracore_top` (the tape-out-ready legacy chip) has no
   top-level integration test.** Its 11 sub-modules are individually
   tested, but the AXI-Lite wrapper connecting them is functionally
   unverified end-to-end.

---

## Methodology

- No memory entries read during the audit, other than acknowledging
  the project goal ("automotive AI SoC with NPU + ASIL-D fusion").
- RTL inventory via `ls rtl/` and `find`.
- Test inventory via `find sim/ -name 'test_*.py'`.
- Grep searches for TODO / FIXME / placeholder / V1 / deferred / no
  implementation markers.
- Grep for SystemVerilog assertions / formal properties.
- Fresh regression runs via cocotb + iverilog for representative
  modules.
- Targeted bisection of the one observed test failure.

---

## A. RTL / Test Inventory

| Category           | Count   | Notes                                |
|--------------------|---------|--------------------------------------|
| RTL modules (`rtl/`) | **44** | Doc previously said 39 — undercount |
| Test files (`sim/`)  | **42** | `astracore_top` (legacy) and `npu_sram_bank` have no test files |
| Test functions (cocotb) | **~426** | Doc previously said ~370 |
| Tools / Python refs  | 6 RTL-mirroring refs + 3 model traces + perf model | All self-checks PASS |
| Total RTL lines      | 9,085  | Largest: `astracore_fusion_top.v` 991; `astracore_top.v` 592; `npu_top.v` 356 |

Four distinct top-level integrations exist, not one:

- `astracore_top` — 11 legacy modules, AXI4-Lite. **Tape-out clean
  sky130 (physical)**, **no top-level functional integration test.**
- `astracore_fusion_top` — 20 fusion modules, dataflow. 36 integration
  tests (see §C).
- `astracore_system_top` — wraps the two above. 2/2 smoke tests PASS.
- `npu_top` — 8 NPU modules. 9/9 integration tests PASS.
- `npu_system_top` — **does not exist**; the intended final product
  top that wraps NPU + fusion is a future file.

## B. Fresh regression results (verified in this audit)

| Subsystem                          | Tests run              | Result      |
|------------------------------------|------------------------|-------------|
| NPU unit + integration (8 modules) | 59                     | **59/59 PASS** |
| Spot-check safety modules          | 87 (gaze, thermal, tmr, ecc, plaus, ttc, aeb, safe_state) | **87/87 PASS** |
| Fusion top (batch run)             | 36                     | **35/36 PASS, 1 FAIL** (see §C) |
| Fusion top (`ego_motion_chain` alone) | 1                   | **PASS**    |
| System top                         | 2                      | **2/2 PASS**|
| Python ref self-checks (6 files)   | —                      | **All PASS**|

## C. Finding: test-ordering state-leakage in fusion_top

- `test_ego_motion_chain` fails when the full 36-test batch runs.
- It passes in isolation.
- It also passes when just its immediate predecessor
  (`test_camera_only_closing_target`) runs before it.
- Therefore: some earlier test (positions 1–6) leaves DUT state the
  `reset_dut()` helper does not clear.
- Candidate culprits: tests that write to CAN-FD FIFO state, IMU
  register state, or object tracker slot state. Would need a proper
  bisection to identify.
- **Severity:** moderate. Not a silicon bug. Does invalidate claims
  of "36/36 fusion_top tests PASS" and the memory-recorded
  "31/31 scenarios PASS". An investor running the suite themselves
  will see 35/36 and ask.
- **Fix path:** make `reset_dut()` explicitly clear all module-level
  registers (FIFOs, counters, latches) between test functions. Or
  use `@cocotb.test(reset_after=True)` equivalent harness pattern.

## D. Finding: `astracore_top` has no integration test

- The 11 modules behind `astracore_top`'s AXI4-Lite wrapper are each
  individually tested (gaze_tracker 11 tests, thermal_zone 10, etc.).
- The wrapper itself (`rtl/astracore_top/astracore_top.v`, 592 lines
  including the AXI register map and submodule instantiations) has
  **no cocotb test file**.
- Memory's "108/108 tests PASS" refers to the sum of per-sub-module
  tests, not the wrapper.
- Physical signoff (DRC, LVS, timing at sky130) is clean, but
  functional integration at the AXI layer is unverified. If the
  register map is miswired, silicon boots but hangs.
- **Severity:** medium-to-high for investor confidence; high for
  actual tape-out insurance.
- **Fix path:** a ~200-line cocotb test that pulses each AXI-write
  register, reads each AXI-read register, and asserts the bit
  patterns propagate correctly. Standard register-map verification.

## E. Finding: zero SystemVerilog assertions in 9,085 lines of RTL

- `grep -r 'assert property\|assume\|cover\|endproperty'` returns
  zero matches across all 44 modules.
- ASIL-D safety properties (AEB latency bounded, safe_state
  permanence, TMR majority, DMA bounds) are documented in memory as
  formal proof targets but **not encoded in the RTL**.
- Phase F (formal verification, SymbiYosys → JasperGold) cannot
  start without these.
- **Severity:** high for Phase F timeline, medium for short-term
  credibility. This doesn't break anything now but materially
  extends the safety-qualification work.
- **Scope estimate:** writing SVA for 20 safety-critical modules is
  ~2 engineer-months of focused work. Currently not in any roadmap
  memory I am aware of.

## F. Finding: confirmed multi-precision RTL is placeholder

- `rtl/npu_pe/npu_pe.v` line 25: "v1 ignores precision_mode and
  always runs INT8. The placeholder encodings ..."
- Lines 21–23 show INT4 / INT2 / FP16 as PLACEHOLDER only.
- Simulation-only `$display` warning fires if anyone asserts
  non-INT8 precision mode at the port (line 141).
- The 1,250 effective TOPS premium-tier claim depends entirely on
  INT2 + 2:4 structured sparsity working — and the arithmetic RTL to
  do that does not yet exist.
- **Severity:** the number is defensible architecturally (same math
  NVIDIA uses for Thor's 2,000 TOPS claim) but the investor answer
  is "the TOPS number is a specification, not yet validated in RTL".
- **Already flagged** in investor doc §11.1 as "load-bearing
  unvalidated assumption" after this finding.

## G. Finding: AFU placeholder modes for transformer-critical activations

- `rtl/npu_activation/npu_activation.v`: SiLU / GELU / Sigmoid / Tanh
  reserved encodings at mode bits `3'b101, 110, 111` — no
  implementation.
- Affects: ViT-B/16 uses GELU in every MLP block; LLaMA uses SwiGLU
  (SiLU × gate) in every attention block. Neither workload can run
  end-to-end without these modes.
- Workaround: software implements activations between NPU tile
  invocations (host-side) — adds latency, defeats one of the NPU's
  points.
- **Severity:** medium. Must land before any real LLaMA / ViT
  benchmark can claim end-to-end performance.

## H. Finding: CAN-FD / Ethernet are protocol-level, not bit-level

- `rtl/canfd_controller/canfd_controller.v` Rev 2 accepts
  pre-parsed 29-bit-ID + 4-bit DLC + 64-bit data via AXI-stream;
  there is **no bit-level serialisation, no baud rate generator,
  no wire-level CAN-FD frame encoding**.
- Production silicon needs an external transceiver (TJA1043T etc.)
  to handle PHY-layer CAN-FD. This is a common pattern but should
  be documented.
- Ethernet controller has the same caveat (scope was not verified
  further in this audit).
- **Severity:** low (standard architecture; every SoC does this).
  But "real CAN-FD PHY" remains an external dependency that isn't
  on-die.

## I. Finding: official batch runner covers 10/44 modules (23%)

- `run_all_sims.sh` `MODULES=` list contains only:
  `thermal_zone canfd_controller ecc_secded tmr_voter fault_predictor
   head_pose_tracker pcie_controller ethernet_controller mac_array
   inference_runtime`
- That's 10 modules — and **gaze_tracker** (a legacy module with 11
  tests) is NOT in the batch. Nor are any fusion, NPU, or top-level
  modules.
- The ~426 total test figure relies on per-module manual runs, not
  the automated batch.
- **Severity:** medium. A clean CI needs a runner that covers 44
  modules, not 10. Current state hides regressions that the
  per-module runs individually catch.
- **Fix path:** update `MODULES=` list + add NPU and fusion sources
  to the runner. ~1 day of work.

## J. Finding: no skipped tests, no disabled code paths

- `grep @cocotb.test\(skip=True\)` returns zero matches. This is good
  — no hidden "we'll get to it later" tests in the suite.

## K. Finding: Python refs all self-check clean

- All 6 NPU Python reference files (`activation_ref.py`, `dma_ref.py`,
  `pe_ref.py`, `sram_ref.py`, `systolic_ref.py`, `tile_ctrl_ref.py`)
  self-check PASS when run standalone.
- This confirms the NPU bit-exact RTL-vs-reference discipline is
  operational. No ref drift from RTL.

## L. Finding: unreviewed top-level modules

- `astracore_fusion_top.v` (991 lines) and `astracore_top.v` (592
  lines) are the largest RTL files by far. Both are primarily
  instantiation + routing, but they are the integration glue that
  carries risk (signal wiring errors, width mismatches, typo'd
  register map bits).
- I did not do a line-by-line audit of either in this pass. An
  investor-grade diligence would want this, especially for
  `astracore_top` given finding §D (no integration test).
- **Severity:** unknown — noted as a "check again" item.

---

## Unanswered questions / re-check list

*These are items I formed a hypothesis about but did not fully verify
in this session. They should be revisited before the next investor
engagement.*

1. **Which specific test is leaking state to break
   `test_ego_motion_chain`?** I bisected partway (ran direct
   predecessor alone — both PASS) but did not fully identify the
   culprit. Likely candidates: tests that touch CAN-FD FIFO state,
   IMU register state, or object-tracker slot state.

2. **Does `ecc_secded` actually detect double-bit errors reliably
   across the full data-width space?** 9 tests exist; I didn't
   inspect coverage. For an ASIL-D module, corner-case coverage
   matters.

3. **Does the `dms_fusion` output need a TMR voter for ASIL-D?**
   Earlier memory audit said yes (currently single non-redundant
   output). I didn't independently verify the current state of the
   RTL on this.

4. **Is the `astracore_fusion_top` lint actually clean at Verilator
   5.030 (per earlier claim)?** I did not re-run Verilator lint in
   this audit; the 36-test iverilog run does not surface Verilator
   warnings.

5. **How does NPU behave at the production-scale grid
   (192×128)?** Tests run at 4×4 / 8×16 / 16×8. No evidence the
   RTL elaborates cleanly at 192×128 — iverilog may run out of
   memory; Verilator would be the right tool for that scale.

6. **Does the `npu_sram_ctrl.v` default-parameter change from this
   morning interact with any other instantiator I missed?** I grep'd
   for instantiators and found only `npu_top` and `npu_tile_harness`,
   but a broader repo-wide search could surface stale references.

7. **Are there any tests that depend on specific OS / path /
   toolchain quirks that might fail in CI on Linux?** I ran
   everything in Git Bash on Windows with iverilog. Cross-platform
   reproducibility is unverified.

---

## Delta to investor brief

Items to reflect in `docs/investor_brief.md` after this audit:

- §4 test-results table: fusion_top is 35/36 (with batch-order caveat),
  not 36/36. Add note.
- §4 test-results table: `astracore_top` physical signoff ≠ functional
  integration test. Clarify.
- §11.2: add "write SystemVerilog assertions for all ASIL-D modules"
  as a Phase F prerequisite, ~2 engineer-months.
- §11.2: add "fix test-ordering state leakage in fusion_top test
  harness" as a near-term cleanup.
- §11.2: add "integration test for `astracore_top` AXI register map"
  as a near-term cleanup.
- §11.1: confirm multi-precision RTL is placeholder (already in the
  brief after the prior re-audit).
- §7: update "Tests that matter" totals to ~426 cases, not ~370.

Overall: the project's engineering posture is genuinely good — the
memory's claims are substantially accurate, the bug-catching
discipline is real (the 4 fusion bugs were real), the NPU RTL is
bit-exact against its reference. The audit surfaces rigour debts and
infrastructure gaps, not architectural flaws.
