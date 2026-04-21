# Tool Confidence Level (TCL) Evaluations

**Document ID:** ASTR-SAFETY-TCL-V0.1
**Date:** 2026-04-20
**Standard:** ISO 26262-8:2018 §11 (Confidence in the use of software tools) + ISO 26262-11:2018 §6 (Application to semiconductor digital design tools)
**Element:** Software tools used in AstraCore Neo development
**Status:** v0.1 — first formal classification. Closes the open item flagged in `docs/safety/iso26262_gap_analysis_v0_1.md` Part 8 §11 + Part 11 §6.
**Classification:** Internal — pre-engagement draft for TÜV SÜD India workshop and first NDA evaluation licensee.
**Author:** TBD (Track 2 lead) — currently founder + collaborator
**Reviewer:** TBD (independent reviewer per ISO 26262-2 §7)
**Approver:** TBD (Safety Manager)

---

## 0. Purpose and framing

ISO 26262-8 §11 requires that **every software tool** whose output influences the development of a safety-relevant element be qualified to a level of confidence appropriate to its potential impact. The Tool Confidence Level (TCL) is determined by combining:

- **Tool Impact (TI):** could a malfunction in the tool cause a safety violation in the developed element that could go undetected?
- **Tool error Detection (TD):** how confident are we that a malfunction in the tool would be detected by independent checks?

> **What this document does:** classifies each tool used in `tools/`, `sim/`, `asic/`, and `astracore/` development flows; assigns TCL per the Table 4 matrix; lists qualification activities for tools with TCL > 1; documents compensating measures.
>
> **What this document does NOT do:** it does not cover hardware components such as OpenTitan crypto IP or licensed memory PHY — those are subject to ISO 26262-8 §13 (Qualification of hardware components), tracked separately. It also does not cover off-the-shelf SW components like ONNX Runtime in their *runtime* role inside the licensee's product (that is an ISO 26262-8 §12 evaluation, also tracked separately).

### 0.1 Companion documents

- `docs/safety/iso26262_gap_analysis_v0_1.md` Part 8 §11 + Part 11 §6 — gap rows this evaluation closes
- `docs/safety/seooc_declaration_v0_1.md` §4.3 ASR-PROC-04 — the Assumed Safety Requirement requiring tool qualification
- `docs/safety/safety_manual_v0_5.md` §13 (revision triggers) — when this doc must be re-issued

---

## 1. Methodology (ISO 26262-8 §11)

### 1.1 Tool Impact (TI)

| Class | Description |
|---|---|
| **TI1** | The tool malfunction *cannot* introduce or fail to detect an error in the safety-related element under development |
| **TI2** | The tool malfunction *could* introduce or fail to detect an error in the safety-related element under development |

### 1.2 Tool error Detection (TD)

| Class | Description |
|---|---|
| **TD1** | High degree of confidence that a tool malfunction would be prevented or detected (e.g., independent cross-check, redundant tools, downstream test gate that cannot be silenced) |
| **TD2** | Medium degree of confidence (some indirect verification, e.g., third-party static analysis, or non-redundant cross-check) |
| **TD3** | Lower degree of confidence (no direct verification path; depends on downstream functional testing on a per-feature basis) |

### 1.3 TCL determination matrix (Table 4)

| TI \ TD | TD1 | TD2 | TD3 |
|:---:|:---:|:---:|:---:|
| **TI1** | TCL1 | TCL1 | TCL1 |
| **TI2** | TCL1 | TCL2 | TCL3 |

**TCL1** = no qualification activity required. **TCL2** and **TCL3** require formal qualification per ISO 26262-8 §11.4.7 (Validation of the Tool with established methods, Tool Development Process Audit, Increased confidence from use).

### 1.4 Qualification methods (when TCL > 1)

Per ISO 26262-8 §11.4.7, four methods are recognised:

1. **Increased confidence from use** — tool has been in use for a long time across many projects with documented absence of relevant errors
2. **Evaluation of tool development process** — the tool's developer follows a documented dev process suitable for the criticality
3. **Validation of the tool** — the tool is validated against a representative test suite that exercises the safety-relevant features
4. **Development in accordance with a safety standard** — the tool itself was developed per a recognised safety standard

For ASIL-D / ASIL-C, multiple methods may be required; for ASIL-A / ASIL-B, one method per ISO 26262-8 Table 5 may suffice.

---

## 2. Tools in scope

The 9 software tools used in the AstraCore safety-relevant development flow:

| # | Tool | Version | Used for |
|---:|---|---|---|
| 1 | Verilator | 5.030 | RTL simulation; fault-injection harness |
| 2 | cocotb | 2.0.1 | Test framework wrapping Verilator |
| 3 | Yosys | latest stable | Logic synthesis (sky130 today; ASAP7 planned) |
| 4 | Symbiyosys (SBY) | latest stable | Formal verification (planned: F4-D-1, F4-D-2) |
| 5 | OpenROAD | latest stable | Physical synthesis + STA (planned: ASAP7 projection) |
| 6 | ASAP7 PDK | open-source academic | 7 nm process projection |
| 7 | pytest | 9.0+ | Python test runner |
| 8 | numpy | 1.26+ | Bit-exact reference math in `tools/npu_ref/` |
| 9 | onnx + onnxruntime | 1.14+ / 1.16+ | Model frontend + FP32 oracle for quantisation tests |

OpenTitan crypto IP is **out of scope** of TCL (it is a hardware component qualified under ISO 26262-8 §13, tracked separately).

---

## 3. Per-tool evaluations

### 3.1 Verilator 5.030

| Field | Value |
|---|---|
| Vendor | Wilson Snyder + open-source community |
| License | LGPL-3.0-or-later / Artistic 2.0 |
| AstraCore use | RTL simulation under cocotb (`tools/run_verilator_*.sh`); fault-injection harness backend (`sim/fault_injection/`) |

**TI classification: TI2.** A Verilator bug could mis-simulate the RTL, causing the bit-exact mirror tests to silently agree with both the wrong RTL behaviour and the wrong Python reference (if the Python uses incorrect semantics). It could also cause fault-injection campaigns to under-report or over-report diagnostic coverage. Either pathway leads to a wrong FMEDA result and a wrong safety claim.

**TD classification: TD1.** Three independent verification paths catch a Verilator bug:

1. **Bit-exact Python mirror.** `tools/npu_ref/` provides bit-exact Python implementations of every safety-critical RTL operation. Tests in `tests/` compare RTL output (via Verilator) against Python output for every supported configuration. A Verilator bug would manifest as a divergence between these two paths.
2. **Multiple test backends.** The Python tests run on the host platform without Verilator; the cocotb tests run via Verilator; the SDK runs via ONNX Runtime as a third backend (`astracore.backends.ort`). A Verilator-only bug would not affect the other two paths' agreement.
3. **Verilator's own validation.** Verilator's regression suite runs millions of tests on every release; bugs that affect basic SystemVerilog semantics are caught upstream and patched.

**Resulting TCL: TCL1.** No qualification activity required. Compensating measures (the three paths above) are documented for the safety case.

**Open items:** none.

---

### 3.2 cocotb 2.0.1

| Field | Value |
|---|---|
| Vendor | cocotb open-source community |
| License | BSD-3-Clause |
| AstraCore use | Test framework wrapping Verilator; coordinates RTL stimulus, response, and assertion checking; used by every cocotb test in `sim/*` and the fault-injection harness in `sim/fault_injection/` |

**TI classification: TI2.** A cocotb bug could mask test failures (e.g., assertion silently swallowed, test marked as PASS when it should FAIL).

**TD classification: TD1.** Two compensating measures:

1. **Pass/fail count is gated externally.** The test scripts (`tools/run_verilator_*.sh`) parse the cocotb summary line; a missing pass/fail line itself causes the script to exit non-zero. Silent test loss would be detected.
2. **cocotb is broadly used** (Synopsys, Cadence, Renesas, multiple open-source projects); failure modes are well-known and patched within community release cycles.

**Resulting TCL: TCL1.** No qualification activity required.

**Open items:** none.

---

### 3.3 Yosys

| Field | Value |
|---|---|
| Vendor | Claire Wolf + YosysHQ + open-source community |
| License | ISC |
| AstraCore use | Logic synthesis (sky130 130 nm via OpenLane today; ASAP7 7 nm planned for F4 phase). Frontend reads SystemVerilog → produces gate-level netlist + cell-mapping. |

**TI classification: TI2.** A Yosys bug could:
- Optimise away a safety-critical net (e.g., remove a TMR voter line because it appears to be "redundant logic")
- Mis-map a SystemVerilog construct to incorrect cells
- Lose the SVA assertions during synthesis (assertions are simulation-only; this is a known Yosys limitation, not a bug — but worth flagging)

Either of the first two pathways would silently produce a synthesised design that violates the safety case.

**TD classification: TD2.** One independent verification path + one indirect path:

1. **Gate-level simulation** of the Yosys output against the same cocotb test suite that validated the RTL. This is **planned but not yet shipped** — gate-level cocotb runs are a Phase B WP. Until then, only the OpenLane batch reports (32/32 modules close on sky130) provide indirect evidence.
2. **Alternative synthesis flow.** A second synthesis pass via a commercial flow (Synopsys Design Compiler / Cadence Genus) would provide an independent cross-check; not currently licensed at AstraCore. Acceptable compensating measure: licensee runs their own synthesis with their preferred flow and reports back.

**Resulting TCL: TCL2.** Qualification activity required.

**Qualification approach (per ISO 26262-8 §11.4.7):**
- **Method 3 (Validation of the tool):** Phase B work package — set up gate-level simulation of Yosys output for every safety-critical module; cross-check against RTL cocotb gate. WP est. 5 days.
- **Method 1 (Increased confidence from use):** Yosys is used by lowRISC OpenTitan (production-grade safety-relevant project), Pulpissimo, and ETH Zurich research chips. Documented usage history available.

**Compensating measure until qualification ships:** safety case cites OpenLane batch report (32/32 modules close on sky130) + planned gate-sim WP.

**Open items:**
- WP for gate-level cocotb in W5-W8 (Phase B)
- Document Yosys usage history collation in safety case at v1.0

---

### 3.4 Symbiyosys (SBY)

| Field | Value |
|---|---|
| Vendor | YosysHQ + open-source community |
| License | MIT |
| AstraCore use | Formal verification driver — proves SVA properties on `tmr_voter`, `ecc_secded`, and `dms_fusion` (per F4-D-1, F4-D-2 in remediation plan). Currently planned, not yet shipped. |

**TI classification: TI2.** A SBY bug could produce a false-positive proof — the tool reports "property holds" when in fact the property does not hold. We would then claim formal coverage that does not exist, undermining the ASIL-D safety case.

**TD classification: TD1.** SBY operates by emitting SMT instances to a backend solver (Z3, Yices, Boolector); a false-positive at the SBY level would require the backend solver to also be wrong. The SAT/SMT space is one of the most rigorously tested areas in computer science; widespread cross-validation between solvers makes false-positive proofs extremely rare. Additionally:

1. **Counter-example inspection.** When a property does not hold, SBY produces a counter-example trace; we can manually inspect the trace.
2. **Cross-checking with commercial formal tools** (Cadence Jasper, Synopsys VC Formal) is feasible if the safety case requires; not currently licensed at AstraCore.
3. **Bounded model checking + induction** can be combined; passing both methods provides redundant verification.

**Resulting TCL: TCL1.** No qualification activity required, *provided* the AstraCore formal-flow process documents:
- That counter-examples are inspected when proofs fail
- That at least bounded model checking AND induction are run on every property
- That the SVA property text is reviewed independently from the RTL author

These three process requirements become part of the W7 formal-flow setup deliverable.

**Open items:**
- Formal flow setup at W7 must document the three process requirements above
- Cross-tool validation with commercial formal optional (compensating measure for ASIL-D)

---

### 3.5 OpenROAD

| Field | Value |
|---|---|
| Vendor | The OpenROAD Project + UCSD + open-source community |
| License | BSD-3-Clause + Apache-2.0 |
| AstraCore use | Physical synthesis + static timing analysis on ASAP7 7 nm projection (planned for F4 7 nm characterization). Not yet exercised. |

**TI classification: TI2.** An OpenROAD bug could:
- Place / route logic in a way that creates timing violations missed by static checks
- Generate wrong SDF (Standard Delay Format) files used in gate-level timing simulation
- Mis-report STA results, leading us to claim timing closure that does not hold on silicon

**TD classification: TD2.** Verification paths:

1. **Independent STA tool** (e.g., Cadence Tempus, Synopsys PrimeTime) — not currently licensed.
2. **Gate-level simulation with backannotated SDF** — provides functional cross-check; planned in same Phase B WP as the gate-sim for Yosys.
3. **OpenROAD's own regression suite** — well-tested but not yet exercised on our specific design.

**Resulting TCL: TCL2.** Qualification activity required.

**Qualification approach:**
- **Method 3 (Validation of the tool):** Phase B / Phase D WP — exercise OpenROAD on a known-good reference design (e.g., a small RISC-V core) with published timing numbers; validate that AstraCore's flow reproduces those numbers within tolerance.
- **Method 1 (Increased confidence from use):** OpenROAD is used by Google (TPU, Tensorflow Edge), academic chips (FreePDK15 reference, Sky130 community shuttles); documented usage history.
- **Method 4 (Development per a safety standard):** OpenROAD is *not* developed per ISO 26262; this method does not apply.

**Compensating measure:** spec sheet rev 1.4 explicitly labels every ASAP7-derived number as "tape-out target" / "projection" — the licensee's eventual production tape-out using their licensed commercial flow is the authoritative validation.

**Open items:**
- Validation WP scheduled for Phase D (W13-W18 per remediation plan §3.4); currently un-scoped — open as F4-D-7
- Cross-tool STA validation optional (deferred until commercial license available)

---

### 3.6 ASAP7 PDK

| Field | Value |
|---|---|
| Vendor | Arizona State University + Greg Yeric (academic) |
| License | Open-source academic |
| AstraCore use | Process Design Kit for 7 nm projection — provides cell library, parasitics models, design rules; consumed by Yosys + OpenROAD for area / timing / power projection |

**TI classification: TI2.** An ASAP7 inaccuracy (which is *expected* — it's an academic predictor, not a foundry PDK) could cause:
- Area projections to differ materially from production silicon
- Timing projections to be optimistic or pessimistic vs production
- Power projections to mis-represent dynamic / leakage components

These do not affect *correctness* of the IP block (Yosys output is functionally correct regardless of PDK), but they affect the headline numbers cited in the spec sheet.

**TD classification: TD3.** No production silicon exists at AstraCore today against which to validate ASAP7 numbers. The PDK's own published "average over commercial PDKs" calibration is the only reference. TD3 is the appropriate honesty rating.

**Resulting TCL: TCL3.** Qualification activity required.

**Qualification approach (TCL3 strictly requires multiple methods):**
- **Method 1 (Increased confidence from use):** ASAP7 has been used in dozens of academic publications; its area/timing predictions are typically within 10-20 % of commercial PDKs at the same node.
- **Method 3 (Validation of the tool):** AstraCore validates ASAP7 numbers against published benchmarks (e.g., the ASAP7 RISC-V reference designs Arizona State publishes); deltas are documented.

**Compensating measure (the strong one):** **the spec sheet rev 1.4 (`docs/spec_sheet_rev_1_4.md`) carries the ASAP7-derived numbers in the "Tape-out target" column with explicit "projection" labelling — never in the "Validated today" column.** The IP datasheet framing makes the limitation explicit to every external reader. This is the load-bearing mitigation.

**Open items:**
- Validation against ASAP7 RISC-V reference (Phase D); open as F4-D-8
- Document delta vs commercial PDKs in safety case at v1.0

---

### 3.7 pytest

| Field | Value |
|---|---|
| Vendor | pytest open-source community |
| License | MIT |
| AstraCore use | Python test runner for the 1352-test suite, including all 208 safety-tooling tests |

**TI classification: TI1.** pytest is the runner, not the test logic. A pytest bug could cause test discovery to miss tests, but discovered-and-run tests use Python's native `assert` and `pytest.raises` mechanisms whose failure modes are explicit (raise AssertionError → test fails). Silent test pass on incorrect logic is not a pytest bug; it's a test-author bug.

**TD classification: n/a** for TI1 (TCL is TCL1 regardless of TD per Table 4).

**Resulting TCL: TCL1.** No qualification activity required.

**Compensating measure (for the test-discovery edge case):** the test count is monitored across runs (today's count is 1352); any unexpected drop in test count is investigated.

**Open items:** none.

---

### 3.8 numpy

| Field | Value |
|---|---|
| Vendor | NumFOCUS + numpy open-source community |
| License | BSD-3-Clause |
| AstraCore use | Bit-exact reference math in `tools/npu_ref/` (PE arithmetic, accumulator semantics, AFU LUTs); also used by `astracore` for tensor manipulation |

**TI classification: TI2.** A numpy bug in elementary operations (e.g., int32 overflow handling, signed vs unsigned reduction) could cause the bit-exact reference to differ from RTL silently. This would propagate to the FMEDA + fault-injection coverage measurements.

**TD classification: TD1.** Three compensating paths:

1. **numpy is one of the most widely-used scientific libraries in existence.** Bugs in elementary arithmetic are detected within days by the broader user base.
2. **AstraCore uses only stable releases** (1.26+); experimental flags are not enabled.
3. **Bit-exact tests cross-check against documented behaviour** — for every PE operation, the Python reference is unit-tested against hand-calculated examples (`test_pe_ref_int8_signed_negative_product` etc.). A numpy bug would cause these tests to fail.

**Resulting TCL: TCL1.** No qualification activity required.

**Open items:** none.

---

### 3.9 onnx + onnxruntime

| Field | Value |
|---|---|
| Vendor | Linux Foundation (ONNX) + Microsoft (ONNX Runtime) |
| License | MIT (ONNX) + MIT (ORT) |
| AstraCore use | Model frontend (ONNX 2.0 loader at `tools/npu_ref/onnx_loader.py`); FP32 oracle for quantisation tests (`astracore.backends.ort` runs the same model in ORT and compares against the NPU-sim backend) |

**TI classification: TI2.** A bug in either:
- ONNX loader → mis-parses the model graph, leading to wrong quantisation targets
- ONNX Runtime → produces wrong FP32 oracle, masking quantisation errors

Either pathway would silently inflate the measured quantisation accuracy (`98.4/96.0/91.2 %` numbers in spec sheet rev 1.4 §3 / §10).

**TD classification: TD1.** Three compensating paths:

1. **Multiple ONNX Runtime versions tested.** Tests run against current stable + previous stable; divergence flagged.
2. **Cross-backend comparison.** `astracore.backends` supports CUDA, TensorRT, OpenVINO, QNN, CPU, etc. via the ORT EP façade (`step1_ort_multi_ep_complete.md`); a bug in any one backend would not affect agreement with the others.
3. **Bit-exact reference path** (`tools/npu_ref/`) is independent of ONNX Runtime — quantisation tests compare against this Python reference, not ORT, for the bit-exact cases.

**Resulting TCL: TCL1.** No qualification activity required.

**Open items:** none.

---

## 4. Aggregate summary

| # | Tool | TI | TD | TCL | Qualification required? |
|---:|---|:---:|:---:|:---:|:---:|
| 1 | Verilator 5.030 | TI2 | TD1 | TCL1 | No |
| 2 | cocotb 2.0.1 | TI2 | TD1 | TCL1 | No |
| 3 | Yosys | TI2 | TD2 | **TCL2** | Yes — gate-sim WP (Phase B) |
| 4 | Symbiyosys (SBY) | TI2 | TD1 | TCL1 | No (process requirements documented at W7) |
| 5 | OpenROAD | TI2 | TD2 | **TCL2** | Yes — F4-D-7 (Phase D) |
| 6 | ASAP7 PDK | TI2 | TD3 | **TCL3** | Yes — F4-D-8 + spec-sheet labelling (already in place) |
| 7 | pytest | TI1 | n/a | TCL1 | No |
| 8 | numpy | TI2 | TD1 | TCL1 | No |
| 9 | onnx + onnxruntime | TI2 | TD1 | TCL1 | No |

**3 tools require formal qualification activities** (Yosys, OpenROAD, ASAP7). Yosys qualification is in Phase B scope (W5-W8); OpenROAD + ASAP7 qualification is in Phase D scope (W13-W18, currently un-scheduled — opening as F4-D-7 / F4-D-8).

---

## 5. Qualification activities (consolidated)

| Tool | Method | WP | Effort | Schedule |
|---|---|---|---|---|
| Yosys | Validation of tool (gate-sim cross-check) | Phase B follow-on (open as **F4-B-7**) | 5 days | W5-W8 |
| Yosys | Increased confidence from use (collation document) | Doc work | 1 day | v1.0 safety case |
| OpenROAD | Validation of tool (RISC-V reference cross-check) | Phase D (open as **F4-D-7**) | 4 days | W13-W18 |
| ASAP7 | Validation of tool (academic-benchmark cross-check) | Phase D (open as **F4-D-8**) | 3 days | W13-W18 |
| ASAP7 | Spec-sheet labelling (compensating measure) | Already shipped in `docs/spec_sheet_rev_1_4.md` | 0 | ✅ done |

Three new WP IDs proposed (F4-B-7 + F4-D-7 + F4-D-8); should be added to `docs/safety/findings_remediation_plan_v0_1.md` Phase B / Phase D tables in a follow-up edit.

---

## 6. Open items for v0.2

These items must close before this TCL evaluation can be approved for external assessment:

1. **Named Track 2 lead, independent reviewer, Safety Manager / Approver** — currently TBD per §0
2. **Confirmation review** per ISO 26262-2 §7 (highly recommended for ASIL-B, mandatory for ASIL-C/D — we have ASIL-D safety goals)
3. **Pinned tool versions** — current entries say "latest stable" for Yosys / SBY / OpenROAD; v0.2 must record exact versions in use at the W6 TÜV pre-engagement workshop
4. **Yosys validation WP (F4-B-7) opened** in remediation plan
5. **OpenROAD + ASAP7 qualification WPs (F4-D-7, F4-D-8) opened** in remediation plan
6. **TI/TD justification refinement** for any tool the licensee challenges during NDA evaluation (TCL is the IP supplier's classification; the licensee may revise within their item-level safety case)

---

## 7. Revision triggers

This TCL evaluation is re-issued (with revision bump) on any of:

1. New tool added to the `tools/`, `sim/`, `asic/`, or `astracore/` development flow
2. Tool version upgrade that changes feature scope (minor patch bumps don't trigger; major version bumps do)
3. Tool removal from the flow
4. Qualification activity completes (TCL1 status with "no quals required" must be revised if a qualification was done; TCL2/TCL3 status must be revised when the qualification completes)
5. Discovered tool error that materially affects safety claims — bumps revision and triggers re-classification of the tool
6. Confirmation review feedback that changes any TI / TD assignment

---

## 8. Document control

| Field | Value |
|---|---|
| Document ID | ASTR-SAFETY-TCL-V0.1 |
| Revision | 0.1 |
| Revision date | 2026-04-20 |
| Author | TBD (Track 2 lead) — currently founder + collaborator |
| Reviewer | TBD |
| Approver | TBD |
| Distribution | Internal + TÜV SÜD India + first NDA evaluation licensee |
| Retention | 15 years post-product-discontinuation per ISO 26262-8 §10 |
| Supersedes | None — this is v0.1; supersedes the tentative classification table in `docs/safety/tcl/README.md` |
