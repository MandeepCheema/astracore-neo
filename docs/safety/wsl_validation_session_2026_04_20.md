# WSL Validation Session — 2026-04-20

**Document ID:** ASTR-SAFETY-WSL-VAL-2026-04-20
**Date:** 2026-04-20
**Purpose:** Record the WSL cocotb regression run that validates the F4-A-5 RTL change + the F4-A-1 ECC wrapper + the 4 fault-injection campaigns against live simulation evidence.
**Status:** IN PROGRESS — Verilator 5.036 build in flight; results land as each suite completes.

---

## 0. Environment

| Component | Version | Source |
|---|---|---|
| OS | WSL Ubuntu-22.04 on Windows 11 | `wsl --status` |
| Initial Verilator | 5.030 (2024-10-27) | `verilator --version` |
| Initial cocotb (system) | 2.0.1 | `python3 -c "import cocotb"` |
| Initial cocotb (user, fallback) | 1.9.2 | `pip3 install --user cocotb==1.9.2` |
| Target Verilator (building) | v5.036 | `git clone --depth 1 --branch v5.036 https://github.com/verilator/verilator.git` |

**Root version conflict:** cocotb 2.0.1 (system) requires Verilator ≥ 5.036; WSL had 5.030. cocotb 1.9.2 (user-install) works with 5.030 but uses the older `units=` Clock API, while most cocotb test files in `sim/*` were authored for cocotb 2.0's `unit=` API. Resolution: build Verilator 5.036 from source to `~/.local`, then revert to system cocotb 2.0.1.

---

## 1. F4-A-5 validation (dms_fusion shadow-register SEU detection)

### 1.1 Working-copy RTL run (with F4-A-5)

Ran `sim/dms_fusion/test_dms_fusion.py` against the working-copy `rtl/dms_fusion/dms_fusion.v` (including F4-A-5 shadow register) using cocotb 1.9.2 + Verilator 5.030.

```
** TESTS=14 PASS=13 FAIL=1 SKIP=0 **
```

The 1 FAIL is `test_reset_state` at line 100:
```
AssertionError: Expected confidence=100 after reset, got 00000000
```

### 1.2 HEAD RTL run (WITHOUT F4-A-5) — diagnostic

To verify whether the failure was caused by F4-A-5, re-ran the same test suite against the committed HEAD RTL (pre-F4-A-5) via `git show HEAD:rtl/dms_fusion/dms_fusion.v > /tmp/astracore_head_rtl/dms_fusion.v` and explicit `VERILOG_SOURCES` override:

```
** TESTS=14 PASS=13 FAIL=1 SKIP=0 **
```

Same result — `test_reset_state` FAIL on HEAD RTL too.

### 1.3 Finding: F4-A-5 is a clean non-regression

The `test_reset_state` failure is **pre-existing** in both HEAD and working-copy RTL. It is unrelated to F4-A-5.

**F4-A-5 validated:** my shadow-register change adds 1 FF + 1 comparator wire + OR into `tmr_fault`. Does not touch `dms_confidence` or any other output the failing test asserts against.

### 1.4 Root cause of the pre-existing `test_reset_state` failure

The test asserts `dms_confidence == 100` immediately after `rst_n` deasserts. Architecturally:

- `rtl/dms_fusion/dms_fusion.v` lines 246-271 — TMR lanes `dal_a/b/c` reset to `ATTENTIVE=0` and `conf_a/b/c` reset to `100`
- `rtl/tmr_voter/tmr_voter.v` lines 73-91 — the voter's **output register `voted` resets to `32'h0`**, independent of the input lanes' reset values
- `assign dms_confidence = tmr_voted[10:3]` — feeds from the voter's output register

Therefore after reset, `voted = 32'h0` → `dms_confidence = 0`, not 100. On the first cycle where `valid=1` (i.e., the first gaze_valid or pose_valid pulse), the voter computes `voted_next` from the lanes and latches it — at that point `dms_confidence` becomes 100.

**Three possible fixes (Licensee / safety-engineering decision):**

1. Change `rtl/tmr_voter/tmr_voter.v` to reset `voted` in a way that reflects the input-lane reset values. Clean but requires voter parametrisation.
2. Change the test to wait one `valid=1` cycle before asserting.
3. Accept the one-cycle cold-start window where `dms_confidence=0` as a design decision. Downstream consumers should ignore the cold-start cycle anyway.

Proposal: **option 2** (test fix), classified as a test-asset WP. Filing as `F4-TEST-1`.

### 1.5 All other dms_fusion cocotb tests pass ✅

| Test | Status |
|---|:---:|
| `test_attentive_path` | ✅ |
| `test_drowsy_perclos_threshold` | ✅ |
| `test_critical_perclos_threshold` | ✅ |
| `test_critical_continuous_closed` | ✅ |
| `test_distracted_continuous_out_of_zone` | ✅ |
| `test_return_to_attentive` | ✅ |
| `test_iir_smoothing_no_single_frame_alert` | ✅ |
| `test_sensor_fail_watchdog` | ✅ |
| `test_sensor_fail_immediate_override` | ✅ |
| `test_dms_alert_deasserted_when_attentive` | ✅ |
| `test_gaze_only_updates` | ✅ |
| `test_pose_only_updates` | ✅ |
| `test_blink_rate_elevation` | ✅ |

All 13 of these exercise the IIR path, TMR path (incl. my F4-A-5 shadow comparator's feed), watchdog, blink-window logic. **No regression from F4-A-5.**

---

## 2. Remaining suites — partial results (Verilator 5.036 built + installed)

Verilator 5.036 + cocotb 2.0.1 combo working post-build. Results:

| Suite | Result | Evidence |
|---|---|---|
| `sim/tmr_voter/test_tmr_voter.py` | **9/9 PASS** ✅ | `logs/wsl_cocotb_2026_04_20/tmr_voter.log` line 82: `TESTS=9 PASS=9 FAIL=0 SKIP=0` |
| `sim/dms_fusion/test_dms_fusion.py` (post-F4-A-5) | **13/14 PASS** (1 pre-existing FAIL) | `dms_fusion_v2.log` line 130. F4-A-5 confirmed non-regression. |
| `sim/npu_top/test_npu_compiled.py` | **6/6 PASS** ✅ | `tools/run_verilator_npu_top.sh` output: `TESTS=6 PASS=6 FAIL=0 SKIP=0` |
| `sim/ecc_secded/test_ecc_secded.py` | g++ -Os pathological hang on Vtop__ALL.cpp; killed at 7+ min. Retry with `OPT_FAST=-O0 OPT_SLOW=-O0` in flight | known issue: Verilator 5.036-generated C++ + g++ 11 -Os interaction on bit-manipulation-heavy modules |
| `sim/fault_injection/campaigns/tmr_voter_seu_1k.yaml` | testbench fix landed (delays removed, runner.py units→unit) — re-run pending | sample campaign, 17 injections |
| `sim/fault_injection/campaigns/ecc_secded_bf_10k.yaml` | same | sample campaign, 21 injections |
| `sim/fault_injection/campaigns/dms_fusion_inj_5k.yaml` | same | sample campaign, 21 injections |
| `sim/fault_injection/campaigns/safe_state_controller_inj_1k.yaml` | same | sample campaign, 9 injections |

### 2.1 Validation chain summary

| Track 2 milestone | WSL evidence | Status |
|---|---|:---:|
| F4-A-5 (dms_fusion shadow comparator) does not regress existing tests | dms_fusion 13/14 same as HEAD pre-F4-A-5 | ✅ confirmed |
| tmr_voter passes existing functional gates under Verilator 5.036 | tmr_voter 9/9 | ✅ confirmed |
| npu_top passes the documented 6/6 cocotb gate under Verilator 5.036 | tools/run_verilator_npu_top.sh PASS | ✅ confirmed |
| ecc_secded compiles + runs under Verilator 5.036 | g++ -Os pathological case; retry in flight | ⚠ build issue |
| Fault-injection campaigns produce measured DC | testbench + runner fixes landed; re-run pending | ⚠ pending |
| F4-A-1 (`npu_sram_bank_ecc` wrapper) integration into npu_top | F4-A-1.1 follow-up not run this session | deferred |

## 3. F4-A-1 wrapper (npu_sram_bank_ecc) — deferred to npu_top integration

The `rtl/npu_sram_bank_ecc/` wrapper was shipped as a new module earlier today. Running it standalone requires a cocotb testbench (none exists yet). The wrapper's functional validation is covered by:

1. `tools/safety/ecc_ref.py` Python mirror — 147 passing tests
2. Deferred npu_top integration (F4-A-1.1) — will validate via existing npu_top 6/6 cocotb suite once swap lands

---

## 4. Next steps (after Verilator 5.036 lands)

1. Re-run all 4 remaining suites above
2. Publish measured DC for each mechanism mapped in the fault-injection campaigns
3. Update `safety_mechanisms.yaml` target DC values if measured differs > 5 pp (per SEooC §9.1 revision trigger #2)
4. Re-run FMEDA + regenerate baseline.json
5. Update this WSL validation report with full results
6. File F4-TEST-1 in remediation plan (test fix for `test_reset_state` cold-start window)

---

## 5. Document control

| Field | Value |
|---|---|
| Document ID | ASTR-SAFETY-WSL-VAL-2026-04-20 |
| Revision | 0.1 |
| Revision date | 2026-04-20 |
| Author | Track 2 collaborator |
| Supersedes | None — first WSL validation report |
