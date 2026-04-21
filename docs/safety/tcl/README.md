# Tool Confidence Level (TCL) Evaluations

**Purpose.** Per ISO 26262-8 §11 and ISO 26262-11 §6, every software tool used in the development of safety-relevant elements requires a TCL evaluation.

**Status:** ✅ v0.1 consolidated evaluation shipped 2026-04-20 at `tcl_evaluations_v0_1.md`. Three tools require formal qualification (Yosys → TCL2, OpenROAD → TCL2, ASAP7 → TCL3); remaining six are TCL1.

## TCL formula

```
TCL = max( TI ranking , TD ranking )
```
- **TI (Tool Impact):** TI1 = no impact on safety-relevant element; TI2 = could introduce or fail to detect an error.
- **TD (Tool error Detection):** TD1 = high confidence error is detected; TD2 = medium; TD3 = low.

Then map (TI, TD) → TCL1 / TCL2 / TCL3 per Table 4 of 26262-8 §11. TCL1 = no qualification needed; TCL3 = full qualification required.

## Tools evaluated (final classifications)

See `tcl_evaluations_v0_1.md` for full justification per tool.

| # | Tool | TI | TD | TCL | Qualification required? |
|---:|---|:---:|:---:|:---:|:---:|
| 1 | Verilator 5.030 | TI2 | TD1 | TCL1 | No |
| 2 | cocotb 2.0.1 | TI2 | TD1 | TCL1 | No |
| 3 | Yosys | TI2 | TD2 | **TCL2** | Yes — gate-sim WP F4-B-7 (Phase B) |
| 4 | Symbiyosys (SBY) | TI2 | TD1 | TCL1 | No (process requirements documented at W7) |
| 5 | OpenROAD | TI2 | TD2 | **TCL2** | Yes — F4-D-7 (Phase D) |
| 6 | ASAP7 PDK | TI2 | TD3 | **TCL3** | Yes — F4-D-8 + spec-sheet labelling (already in place) |
| 7 | pytest | TI1 | n/a | TCL1 | No |
| 8 | numpy | TI2 | TD1 | TCL1 | No |
| 9 | onnx + onnxruntime | TI2 | TD1 | TCL1 | No |
| — | OpenTitan crypto IP | (not a tool — qualification per ISO 26262-8 §13 hardware component) | n/a | n/a | n/a |

## Output per tool

For each tool, produce `<tool>_tcl.md` with:
1. Tool description, version, vendor
2. Use-case in AstraCore development flow
3. Tool Impact (TI) classification + justification
4. Tool error Detection (TD) classification + justification
5. Resulting TCL
6. Qualification activities (if TCL > 1)
7. Compensating measures

## Reference

ISO 26262-8 §11; ISO 26262-11 §6.

See also: `docs/safety/iso26262_gap_analysis_v0_1.md` Part 8 row.
