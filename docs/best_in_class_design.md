# AstraCore Neo — Best-in-Class Design + Audit

**Date:** 2026-04-19
**Author:** Strategy + audit pass (Claude collaborator)
**Status:** Working document — overrides memory snapshots. Verified against repo.

---

## 0. How to read this document

Three things, in order:

1. **§1 — Audit.** What is *actually* true in the repo today, verified by running code, reading RTL, listing files. Replaces stale memory claims (memory said 1119 → 1128 → 1140 tests; we treat README's *honest matrix* as canonical).
2. **§2 — Best-in-class thesis.** What "best-in-class" means *for the lane we can credibly own*, not what it would mean for NVIDIA.
3. **§3 — Plan.** Prioritized 6 / 12 / 26-week work to make the thesis defensible. Includes what to defer, partner, or buy.

If you're short on time: skip to §2.4 (the one-page positioning statement) and §3.1 (the next 6 weeks).

---

## 1. Audit — what is actually true today (2026-04-19)

### 1.1 Verified by running it

| Claim | Verified? | Evidence |
|---|---|---|
| Python test suite collects 1140 tests | ✅ | `pytest --collect-only` → `1140 tests collected` |
| Suite passes | ⚠️ Mostly | First failure observed at `test_pytorch_frontend.py::test_gelu_exports_to_op_gelu` — needs `torch` extra. `792 passed` before stop on first failure. README claim "1025 tests pass" understates collection but matches a passing subset (`-m "not integration"`). |
| 48 RTL modules in `rtl/` | ✅ | `ls rtl/` — see Appendix A |
| TMR voter, ECC SECDED, safe-state ctrl, plausibility checker, fault predictor exist | ✅ | `rtl/tmr_voter/`, `rtl/ecc_secded/`, `rtl/safe_state_controller/`, `rtl/plausibility_checker/`, `rtl/fault_predictor/` |
| `dms_fusion` uses `u_tmr_dal` | ✅ | `rtl/dms_fusion/dms_fusion.v` |
| SDK has plugin entry-points for ops/quantisers/backends | ✅ | `pyproject.toml` `[project.entry-points."astracore.ops"]` etc. |
| Two backends ship: `npu_sim` + `ort` | ✅ | `astracore/backends/npu_sim.py`, `astracore/backends/ort.py` |
| Tier-1 ADAS YAML drives full pipeline (4 cams + 1 lidar + 6 radars + 12 US + thermal + event + ToF + 2 CAN + GNSS + IMU + 3 models + safety policies) | ✅ | `examples/tier1_adas.yaml` + `reports/benchmark_sweep/apply_tier1/` |
| Custom-sensor fusion example runs end-to-end | ✅ | `examples/ultrasonic_proximity_alarm.py` — 4 alarm bands measured |

### 1.2 Validated buyer-DD findings (from `docs/buyer_dd_findings_2026_04_19.md`)

These are *real* gaps. Fixing them — or honestly framing them — is mandatory to win serious automotive evaluations.

| Severity | Gap | Implication |
|---|---|---|
| **FATAL F1** | Validated at 16-MAC default (8×8 = 64 max). Spec sheet says 24,576 MACs. **384× gap.** | No external buyer will accept the TOPS claim. |
| **FATAL F2** | Synthesis only on sky130 130nm @ ~100MHz target. Spec says 7nm @ 2.5–3.2 GHz. | The "1258 TOPS" headline depends on silicon that doesn't exist. |
| **FATAL F3** | "1258 TOPS INT8 / 2516 TOPS INT4" multiplies unmeasured projections × an 8:1 sparsity engine that **isn't built and shows 0% accuracy at 2:4 magnitude pruning**. | Headline TOPS is fiction until QAT + sparsity engine exist. |
| **FATAL F4** | Zero crypto RTL (no AES, RSA, SHA-256, Kyber, Dilithium). Spec sheet promises all of these. | Blocks automotive secure-boot, V2X, OTA story. |
| **FATAL F5** | Zero memory controller RTL. SRAM bank default is 256 *bytes* (spec: 128 MB). | Bandwidth claims (LPDDR5X 400 GB/s, HBM3 750 GB/s) are licensed-IP placeholders. |
| **FATAL F6** | No FMEDA, no fault-injection harness, no DC/LFM/SPFM numbers. | ASIL-D claim cannot be defended. |
| **FATAL F7** | FP MAC RTL exists in `rtl/npu_fp/` but is *not wired into* `npu_top`/`systolic_array`. PE falls back to INT8 on FP16 mode. | All FP precision rows on the spec sheet are non-functional. |
| HIGH H1–H9 | Pipeline depth, transformer engine MHSA tile, ISP, AXI security snoop, on-chip training, raw-image eval corpus, UCIe/PHY/V2X licensed-IP claims | Each blocks a slice of the spec sheet. |
| MED M1–M8 | README staleness, missing LICENSE root, INT2 undersell, TensorRT mis-categorization, etc. | Doc hygiene; cheap to fix. |

### 1.3 What this means

**The asset is a mature small-NPU IP + a serious sensor-fusion RTL portfolio + a clean SDK with extensibility designed in.** Not "almost ready to tape out a 1258-TOPS 7nm chip". The spec sheet is written for a chip that needs ~18–24 months of additional silicon work, ~$25–50M, and licensed PHY / memory / crypto IP.

**The good news** — the SDK side is further along than the buyer DD foregrounds. Plugin entry-points exist. Two backends ship. Tier-1 YAML works. Custom fusion examples run. The path to a *credible best-in-class software product* is much shorter than the path to a *credible best-in-class silicon product*.

---

## 2. Best-in-class thesis

### 2.1 Why "best-in-class" cannot mean "beat NVIDIA across the board"

NVIDIA owns:
- CUDA + 15 years of devtool mindshare
- Training at hyperscale
- Datacenter inference at top dollar
- Even **automotive central compute** (DRIVE Orin shipping in Mercedes EQS, Volvo EX90, Polestar 3)

Trying to beat NVIDIA in any of those lanes requires capital and time AstraCore does not have. Pick what NVIDIA *cannot* serve well and dominate it.

### 2.2 The lane AstraCore can credibly own

> **Safety-certifiable, fusion-native edge AI for the zonal/ECU tier of automotive + adjacent industrial/medical edge.**
> Sub-10W. Sub-$50 BOM. ISO-26262 process from day one. Multi-sensor fusion is *built in*, not bolted on.

This lane works because:
- **Hailo** ships excellent perf/W but *no fusion, no safety story* — customers must integrate themselves.
- **NVIDIA DRIVE Orin** is $400+ at the module level and 15–60 W — wrong cost/power envelope for zonal controllers.
- **Qualcomm Snapdragon Ride** competes at central compute, not zonal.
- **Coral / Apple ANE** have no automotive safety story.
- **Mobileye EyeQ** is locked to Mobileye perception stack — no openness for OEM custom models.

The wedge: **OEMs assembling zonal architectures (Volvo SPA2, BMW Neue Klasse, VW SSP) need many small AI nodes near sensors with safety primitives, not one big central compute.** Today they use Renesas R-Car / Infineon AURIX / TI Jacinto + bolted-on accelerators. AstraCore replaces the bolt-on with a single-die fusion-native NPU.

### 2.3 What "best-in-class" actually means for that lane

Five claim categories an OEM evaluator will check, in order of weight:

| # | Claim | Why it ranks here |
|---|---|---|
| 1 | **Safety primitives + ISO 26262 process evidence** | If you cannot show this, conversation ends. |
| 2 | **Sensor fusion that works on real data** | This is the visible differentiator. Every competitor has TOPS; few have fusion. |
| 3 | **Reproducible perf/W numbers vs Hailo / Jetson Nano on a fixed model zoo** | "Trust me" is not a perf claim. |
| 4 | **Extensibility — OEM can plug in custom models, custom calibration, custom backend without forking** | Decides whether the SDK survives 5-year automotive design cycles. |
| 5 | **Tooling DX — `pip install`, model imports, profiler, sane errors** | Decides whether engineers actually use it. |

Notice what is *not* in the top 5: peak TOPS, peak GHz, peak SRAM. Those are spec-sheet vanity for this lane. Ship lower numbers honestly and you win.

### 2.4 One-page positioning statement (use this verbatim with OEMs)

> AstraCore Neo is a **small, safety-shaped NPU IP + Python/C++ SDK** purpose-built for **zonal AI controllers** in automotive (and the same envelope works for industrial PLCs and medical edge).
>
> **Validated today (FPGA + simulation):** INT8 / INT4 / INT2 compute, end-to-end ONNX → quantize → run on YOLOv8-N. ECC SECDED, TMR voter, safe-state controller, plausibility checker, fault predictor — all present in RTL. Driver-monitoring fusion engine with explicit ASIL-shaped state machine. 48 RTL modules synthesized in OpenLane sky130. SDK ships with plug-in op / quantiser / backend registries; two backends ready (NPU sim + ONNX Runtime fallback).
>
> **Roadmap, dated:** AWS F1 bring-up of 64×64 array (4096 MACs, ~16% of full-spec) for measured silicon-scale latency by Q3 2026. C++ runtime + MLPerf Tiny submission in the same window. ISO 26262 process audit (TÜV/SGS) targeting ASIL-B initially, ASIL-D as a follow-on safety extension. Crypto, memory PHY, 7nm synthesis are explicit dependencies on licensed IP + Series-B capital, scoped separately.

### 2.5 What you stop claiming

| Old claim | Replacement |
|---|---|
| 1258 TOPS INT8 | "Up to 64 INT8 MACs validated, AWS F1 4096-MAC bring-up Q3 2026; full-spec 24,576 MACs is a tape-out target." |
| 2.5–3.2 GHz | "100 MHz validated on Artix-7; 7nm timing closure is a tape-out target." |
| LPDDR5X 400 GB/s | "Memory PHY is licensed-IP, not in current scope." |
| AES-256 + RSA-2048 + PQC RTL | "Software crypto today (BoringSSL/mbedTLS via host CPU); RTL crypto is a Series-B work package." |
| ASIL-D certified | "ISO 26262 software process from day one. ASIL-B target Q4 2026; ASIL-D requires post-silicon FMEDA + fault injection." |
| ISP-Pro, 8K HDR | Drop. Partner with Sony / OmniVision. |
| UCIe chiplet 2000+ TOPS | Drop. This is post-Series-B. |

This is *not* surrender. It's **honesty as a competitive weapon.** OEMs have been burned by spec-sheet inflation from every AI silicon vendor. A vendor that says exactly what they have wins the trust deltas that close deals.

---

## 3. Plan — 6 / 12 / 26 weeks

### 3.1 Next 6 weeks (closes the first credibility gap)

**Goal:** Take an OEM evaluator from `pip install astracore` to *measured* silicon-scale latency + an ISO 26262 process foothold, in one session.

| Week | Track A — SDK/demo | Track B — silicon credibility | Track C — safety process |
|---|---|---|---|
| 1 | `pip install astracore-sdk` works from PyPI test index. README front-page rewritten with §2.4 positioning. README "1025 tests" updated to whatever the real `pytest --no-header -q` number is. | F1-F1: Vivado synthesis of 64×64 NPU on AWS F1 VU9P (target 4096 MACs, ~150 MHz). Use existing `WEIGHT_DEPTH=4096` parameterization. | Open ISO 26262 *software* process gap analysis. List MISRA-C/SystemVerilog gaps in `tools/`+`rtl/`. One-page action plan. |
| 2 | HuggingFace + ONNX Hub direct import: `astracore.compile("meta-llama/Llama-3.2-1B")` works. Wire to existing INT4 quantizer. | F1-F2: AWS F1 image build + `bitstream.tar.gz` artefact. | Draft *Safety Manual v0.1* (template from ISO 26262-10 §6). Even 20 pages is enough to start a TÜV conversation. |
| 3 | One MLPerf Tiny benchmark E2E on the npu-sim backend: pick **visual wake words** (smallest model, fastest to ship). Reproducible script + numbers in `reports/mlperf_tiny/`. | F1-F3: Run yolov8n on AWS F1 image, capture cycle counts + bandwidth. First *measured* silicon-scale latency number. | Add MISRA-C gates to CI for `tools/` (start with style-only rules, ratchet up). |
| 4 | Profiler: per-op cycle counts + memory bandwidth heatmap. CLI: `astracore profile model.onnx`. Use the AWS F1 measurement loop. | Document `reports/aws_f1_v0/` with reproducible commands. Compare measured FPS vs Python-projected FPS. | Run first FMEDA-shaped failure-mode review on `dms_fusion`. Output: mechanism table (TMR coverage %, ECC coverage %, safe-state coverage %). |
| 5 | C++ runtime stub via pybind11. Just enough to load a compiled program and run one op. Wire as third backend (`astracore.backends.cpp`). | Synthesis report (timing, area, utilization) for AWS F1 build, published in `docs/silicon_evidence_q3_2026.md`. | First fault-injection harness: stuck-at + bit-flip in `tmr_voter` testbench. Even 100 injections produces a defensible coverage number. |
| 6 | **Demo day artefact:** single 5-minute screen-record showing: pip install → load YOLOv8 → quantize → run on npu-sim → run on AWS F1 (same compiled program) → profiler heatmap → safety-mechanism report. | (Same artefact closes silicon track.) | (Same artefact closes safety track.) |

**Cost of this 6 weeks vs current `next_session_handoff.md` plan:** delays F1-A1 (FP8/FP16 RTL synthesizable) by ~6 weeks. That's the only major slip. F1-A1 stays Tier-3 because no OEM in your wedge cares about FP RTL until LLMs cross from useful-novelty to required-feature, which is post-Series-B for automotive zonal.

### 3.2 Weeks 7–12 (deepens the moat)

| Track | Work |
|---|---|
| **SDK** | Add 3 more frontends: PyTorch direct (skip ONNX export), TensorFlow Lite import, GGUF for LLMs. The plugin registry already supports this — wire each as an `astracore.frontends` entry-point. |
| **Quantization moat** | Implement SmoothQuant + AWQ for INT4 LLM. Add per-layer mixed precision search. This is the *real* differentiator vs Hailo (whose INT8 story is fine but who has no INT4 LLM story). |
| **C++ runtime** | Reach Python-runtime parity on the 8-model zoo. Run regression of all 1140 tests through the C++ backend. |
| **Silicon** | F1-A4 → F1-A5 path: wire MHSA tile (8× attention heads) into AWS F1 image. Measured transformer block latency. |
| **Safety** | Complete FMEDA on dms_fusion + lane_fusion. Start ECC SECDED diagnostic-coverage formal proof (Symbiyosys / Jasper). |
| **Benchmarks** | Submit MLPerf Tiny v2.0 (Apr/Oct 2026 deadlines). Even *one* submission category puts you on the public board. |
| **Customer** | Pick **one** Tier-1 design partner (suggested: a brake-by-wire / steering-by-wire vendor; their AI needs are real and their safety bar is high). Ship them a NDA'd evaluation kit = AWS F1 access + SDK + safety manual draft. |

### 3.3 Weeks 13–26 (decides whether the chip ever exists)

This is the fork in the road. Two valid paths; pick one.

**Path A — Software-first IP licensing.** Stop trying to tape out. Sell the IP + SDK to OEMs who already have silicon partners (Renesas, Infineon, TI, NXP). Revenue model: per-unit royalty on licensed IP. Weeks 13–26 = sales motion + 3 paid IP evaluations.

**Path B — Tape-out preparation.** Raise Series B (~$25–50M for 7nm shuttle), license PHY + memory + crypto IP, hire 12–15 silicon engineers, target tape-out 2027 H2. Weeks 13–26 = fundraise narrative built on §3.1 + §3.2 evidence.

**Recommendation:** Path A first, Path B as a Series B narrative that is justified *because* Path A produced 3+ paying IP evaluations. Going straight to Path B without paying customers makes the round 3× harder.

---

## 4. What to defer, partner, or buy

| Capability | Build / Defer / Partner / Buy |
|---|---|
| ISP / image signal processor | **Partner** (Sony IMX series ships with ISP; OmniVision ditto). Drop "ISP-Pro" claim. |
| LPDDR5X / HBM3 PHY | **Buy** (Synopsys, Cadence licensed IP) post-Series-B. Defer claim until then. |
| AES-256 / RSA-2048 RTL | **Buy** (Synopsys DesignWare crypto IP) post-Series-B. Today: software crypto on host CPU. |
| NIST PQC (Kyber, Dilithium) | **Defer** to 2028. Standards still settling. |
| UCIe chiplet | **Defer** post-Series-B. |
| 8K HDR / AI denoising | **Drop**. Not your lane. |
| MLPerf submission | **Build** in §3.1 wk 3. Cheap, high credibility ROI. |
| ISO 26262 cert | **Partner** with TÜV SÜD / SGS-TÜV for audit; build process internally. |
| C++ runtime | **Build** in §3.1 wk 5 + §3.2. |
| Custom op extensibility | **Already built**. Document better. |
| HuggingFace import | **Build** in §3.1 wk 2. Two days of work. |
| QAT / sparsity-aware training | **Build** in Phase C as planned. Differentiator. |
| RISC-V control core | **Buy** (SiFive, Andes) or open-source (CV32E40P) when needed. |
| Vision pipeline (NMS, NMS-free, DFL decode) | **Build**. Already partial in `tools/npu_ref/yolo_decoder.py`. |

---

## 5. Risks that kill the "best-in-class" claim

Sorted by probability × damage:

1. **Buyer DD without §2.5 honesty.** If the spec sheet keeps the inflated numbers and an OEM does the same audit `docs/buyer_dd_findings_2026_04_19.md` did, you lose the deal *and* a reputation. Fix this **week 1**.
2. **No measured silicon-scale latency.** Every conversation hits the same wall: "what's the actual throughput?" — and Python projections won't satisfy. AWS F1 closes this.
3. **No safety process evidence.** OEMs ask for the safety manual on the first call. A 20-page v0.1 + a clear ASIL-B path beats a polished 0-page silence.
4. **Spreading thin across 4 vertical markets.** If you try to be best-in-class for automotive AND industrial AND medical AND robotics in the same year, you'll be mediocre in all four. Pick automotive zonal first; the others adopt the same SDK later.
5. **Building the C++ runtime before there's a contract that needs it.** It's the right work eventually. It's the wrong work in week 1 because nobody buys *because* you have a C++ runtime — they buy because of safety + fusion + perf.

---

## 6. Open questions for you (the founder)

These I cannot answer; you can:

1. **Path A (IP licensing) vs Path B (tape-out).** Which fundraise narrative do you want? This decides whether week 7+ work optimizes for revenue evidence or for tape-out readiness.
2. **First Tier-1 design partner.** Do you have a candidate? If yes, §3.2 changes from "find one" to "ship to the named one."
3. **Geographic safety-cert strategy.** TÜV SÜD (Germany) vs SGS-TÜV (Germany) vs UL (US) vs DEKRA — which target market drives the choice?
4. **Spec-sheet rewrite authority.** Are you willing to ship rev 1.4 with §2.5 honesty? This is the single highest-leverage move in the document.

---

## 7. Founder direction (locked 2026-04-19)

Three decisions taken:

1. **Path A (IP licensing) is the primary business model.** Path B (in-house tape-out) becomes a Series-B narrative justified by Path A revenue evidence.
2. **Spec sheet rev 1.4 reframes as IP datasheet, not chip datasheet.** Headline TOPS/clock/precision claims retained but explicitly conditional on the licensee's 7nm tape-out + their PHY/memory/package IP. Variance budget vs current rev 1.3 numbers: ≤10% on items AstraCore controls (compute datapath, sparsity, crypto, FP, safety mechanisms). Items the licensee provides (LPDDR5X PHY, ISP, UCIe, package thermal, final tape-out) move to a "Reference SoC integration" appendix as licensee-supplied.
3. **Safety-cert is the lead track.** All other tracks must produce artefacts that feed the safety case (FMEDA inputs, fault-injection coverage, MISRA-shaped code, traceability matrix). ASIL-B target by W12, ASIL-D safety case document by W20, TÜV SÜD pre-engagement by W6.

### 7.1 Why this works as a coherent IP licensing story

Standard IP licensing practice (Synopsys ARC NPX, Arm Ethos, Imagination NN-A, Cadence Tensilica) is exactly this pattern: vendor ships RTL + compiler + safety artefacts; licensee provides PHYs, memory, package, tape-out. AstraCore's differentiation in this market: **(a)** sensor-fusion RTL portfolio competitors don't have, **(b)** ISO 26262 process from day one, not retrofit, **(c)** open-source SDK with extensibility entry-points already designed in.

### 7.2 Revised 16-week plan (3 parallel tracks, safety-cert lead)

| Wk | Track 2 — Safety-cert (LEAD) | Track 1 — Compute IP credibility | Track 3 — Crypto + IP collateral |
|---:|---|---|---|
| 1 | ISO 26262 software process gap analysis. Safety Manual v0.1 outline. FMEDA tool stub (Python). | F7: wire `rtl/npu_fp/` MAC + PE into `npu_top` and systolic array. | OpenTitan repo vendored to `third_party/opentitan/`. AES-256 GCM block selected. |
| 2 | First FMEDA on `dms_fusion`. Failure-mode catalog v0.1. | F1 prep: parameter sweep `npu_top` to 16×16, validate via Verilator. | OpenTitan AES integration + AXI wrapper. |
| 3 | First FMEDA on `npu_top`. DC/LFM hand-calculation. Cocotb fault-injection harness scaffold. | F1: Verilator at 32×32 (1024 MACs). Throughput regression. | OpenTitan SHA-256 + HMAC integration. |
| 4 | 1000-fault campaign on `tmr_voter` + `ecc_secded`. First defensible diagnostic-coverage number. | F1: Verilator at 48×48 (2304 MACs). | OpenTitan RSA-2048 integration + key-storage stub. |
| 5 | Engage TÜV SÜD India (Bangalore) for pre-assessment workshop. Send Safety Manual v0.1 + first FMEDA. | F2: ASAP7 PDK + OpenROAD installed. First trial synthesis of `npu_pe` at 7nm. | TRNG (PTRNG/CSRNG from OpenTitan). |
| 6 | TÜV pre-assessment workshop. Output: prioritized gap list + audit timeline. | F1: Verilator at 64×64 (4096 MACs). F2: synthesis of full systolic array tile at 7nm; report timing/area. | Secure-boot reference flow document. |
| 7 | Formal verification setup (Symbiyosys / Yosys-SBY): properties for `tmr_voter` (no-stuck-fault, single-fault-tolerant). | F3 begins: 8:1 sparsity engine RTL — metadata decoder. | IP licensing collateral kickoff: datasheet rev 1.4 outline. |
| 8 | Formal proofs for `ecc_secded` (single-bit-correct, double-bit-detect). | F3: zero-skip MAC integration. | Datasheet rev 1.4 v0.1 draft. Reference SoC integration appendix. |
| 9 | Fault-injection campaign extended to `dms_fusion`, `safe_state_controller`, `plausibility_checker`. 10K injections. | F3: QAT pipeline — first sparsity-aware training on YOLOv8-N. | NDA evaluation kit: assemble RTL + testbenches + safety artefacts + reference SoC. |
| 10 | DC/LFM/SPFM aggregate report v0.1 across all safety modules. | F1: scale Verilator past VU9P cap — full 24,576 MACs simulation (slow but completes). | NDA evaluation kit packaging + signing flow. |
| 11 | Safety case document outline (per ISO 26262-10). | F2: full systolic array timing closure at 7nm @ 2.5 GHz target (relaxed from 3.2 GHz spec). | First IP licensee outreach (Renesas, Infineon, Indian/Asian fabless). |
| 12 | **ASIL-B safety case v1.0.** TÜV SÜD interim review. | F3: measured throughput at 8:1 sparsity vs projection. Update spec sheet variance numbers. | Second outreach round; respond to first NDA queries. |
| 13 | Safety manual v1.0 (production version). | F1+F2 combined: full-array @ 7nm power/area characterization report. | First NDA technical evaluation begins. |
| 14 | Begin ASIL-D extension: extended fault campaigns, common-cause failure analysis. | F3: QAT accuracy report on 8-model zoo at INT4 + 8:1 sparsity. | Second NDA technical evaluation begins. |
| 15 | CCF analysis + FMEDA v2.0. | Compiler scheduler stress test at full-array scale. | Third outreach; first evaluation feedback loop. |
| 16 | **ASIL-D safety case document v1.0.** TÜV SÜD pre-audit checklist. | Spec sheet rev 1.4 final variance numbers measured. | First evaluation produces actionable feedback; iterate. |

### 7.3 What happens after Week 16

- TÜV SÜD full audit (3–6 month engagement) leads to ASIL-B certification mark.
- ASIL-D requires post-silicon fault-injection campaign at the licensee's tape-out. We provide the safety case + test plan; licensee runs on their silicon.
- 3 NDA technical evaluations either close to a paid IP license (Path A success) or produce specific blockers we feed back into the next 16-week cycle.
- Series B narrative becomes: "3 IP licensees signed, ASIL-B in hand, ASIL-D safety case ready, here is the capital ask for our own tape-out."

### 7.4 Variance accounting (the ≤10% promise)

| Spec sheet claim | Strategy to hit ≤10% variance | Owner |
|---|---|---|
| 24,576 MACs (48×512) | Verilator simulation at full scale by W10. Synthesis projection to 7nm by W13. Variance measured against simulation throughput × characterized clock. | Track 1 |
| INT8 / INT4 / INT2 / FP8 / FP16 precision | F7 (FP wire-in) + existing INT paths. ≤10% accuracy variance vs FP32 reference on 8-model zoo. | Track 1 |
| 1258 TOPS INT8 @ 3.2 GHz × 8:1 sparsity | F3 sparsity engine + characterized clock from F2. Headline variance = (measured TOPS / 1258) — target ≥ 0.90. | Track 1 |
| 2.5–3.2 GHz | F2 7nm synthesis at 2.5 GHz target initially; 3.2 GHz becomes "high-corner stretch", documented. | Track 1 |
| LPDDR5X / HBM3 PHY | **Reframed as licensee-supplied.** Variance N/A on AstraCore IP; we expose AXI/CHI interface. | Datasheet |
| 128 MB SRAM | Parameterized; full count is silicon area decision. Variance = SRAM controller correctness at parameter ≥ 90% of 128 MB. | Track 1 |
| ASIL-D | Process documented + safety case in place by W20. Cert mark requires post-silicon work; **we ship the artefacts that make cert achievable**. | Track 2 |
| AES-256 / RSA-2048 / SHA-256 / TRNG | OpenTitan integration. Same algorithm coverage as spec, throughput depends on AXI clock. | Track 3 |
| PQC (Kyber, Dilithium) | Defer to W17+ or Series B (standards still maturing). Spec sheet caveats accordingly. | Track 3 |
| ISP-Pro, 8K HDR, UCIe, V2X | **Reframed as licensee-supplied or partner-supplied**. | Datasheet |

### 7.5 Resourcing reality

This 16-week plan needs:

- **Track 1 (compute):** 2 RTL engineers + 1 verification engineer + access to ASAP7 PDK (free, academic). Optional commercial EDA (~$50–100k for 16-week eval license).
- **Track 2 (safety):** 1 functional-safety engineer (ideally with prior ISO 26262 experience) + ~$30–60k TÜV SÜD pre-engagement fee.
- **Track 3 (crypto + collateral):** 1 RTL engineer for OpenTitan integration (4–6 weeks) + 1 product/marketing person for datasheet + 1 BD person for IP outreach.

If headcount is the constraint, the priority order for hiring is **(1) functional-safety engineer, (2) verification engineer, (3) BD person.** Track 2 (safety) has the slowest serial dependency (TÜV engagement is calendar-bound, not headcount-bound after kickoff) and the biggest blocker on closing IP deals. Compute scaling can flex; safety cannot.

### 7.6 Concrete artefacts delivered this session (Track 2)

| Artefact | Path | Plan Wk | Status |
|---|---|---|---|
| ISO 26262 process gap analysis | `docs/safety/iso26262_gap_analysis_v0_1.md` | W1 | ✅ v0.1 complete |
| SEooC declaration (per ISO 26262-10 §9 + 26262-11 §4.6) | `docs/safety/seooc_declaration_v0_1.md` | W2 | ✅ v0.1 complete |
| Safety Manual outline (per ISO 26262-11 §4.7) | `docs/safety/safety_manual_v0_1.md` | W1-W12 | 🟡 v0.1 skeleton; sections fill in W2-W12 per completion plan in §13.1 |
| FMEDA reports directory + index | `docs/safety/fmeda/README.md` | W2-W10 | 🟡 directory + index only; reports populate per schedule |
| Fault-injection campaign reports directory + index | `docs/safety/fault_injection/README.md` | W3-W10 | 🟡 directory + index only; harness at `sim/fault_injection/` to be created W3 |
| Tool Confidence Level evaluations directory + index | `docs/safety/tcl/README.md` | W7 | 🟡 directory + tentative TI/TD/TCL classification table |

---

## Appendix A — RTL module census (48 modules)

**Compute (10):** `npu_pe`, `npu_systolic_array`, `npu_tile_ctrl`, `npu_tile_harness`, `npu_top`, `npu_system_top`, `npu_dma`, `npu_sram_bank`, `npu_sram_ctrl`, `npu_softmax` + `npu_layernorm` + `npu_activation` + `npu_fp` (last 4 are AFUs).

**Sensor I/O (9):** `mipi_csi2_rx`, `radar_interface`, `lidar_interface`, `imu_interface`, `ultrasonic_interface`, `gnss_interface`, `canfd_controller`, `ethernet_controller`, `pcie_controller`.

**Fusion (3):** `dms_fusion`, `lane_fusion`, `astracore_fusion_top`. (Memory's "20 fusion modules" includes I/O + decoders; the strict-fusion count is 3.)

**Perception primitives (5):** `gaze_tracker`, `head_pose_tracker`, `cam_detection_receiver`, `det_arbiter`, `object_tracker`.

**Decoding/transform (3):** `can_odometry_decoder`, `coord_transform`, `ego_motion_estimator`.

**Safety (5):** `tmr_voter`, `ecc_secded`, `safe_state_controller`, `plausibility_checker`, `fault_predictor`.

**Vehicle dynamics / ADAS app (3):** `aeb_controller`, `ldw_lka_controller`, `ttc_calculator`.

**Infrastructure (5):** `inference_runtime`, `mac_array`, `sensor_sync`, `ptp_clock_sync`, `thermal_zone`.

**Top (3):** `astracore_top`, `astracore_system_top`, `astracore_fusion_top`.

---

## Appendix B — What I changed about prior advice in this conversation

| Earlier statement | Correction |
|---|---|
| "NVIDIA cannot follow you into automotive" | Wrong. NVIDIA DRIVE Orin ships in production cars. The defensible claim is **zonal/ECU tier**, not "automotive". |
| "Jetson is 15–60W" | Incomplete. Jetson Orin Nano starts ~7W. Hailo edge is real but smaller than implied. |
| "TensorRT INT4 is weak" | Outdated. TensorRT-LLM has improved (W4A16, AWQ, GPTQ). Still weaker than a clean per-channel INT4 SDK could be, but "weak" overstates it. |
| "20 fusion modules" (from memory) | Strict fusion count is 3 (`dms_fusion`, `lane_fusion`, `astracore_fusion_top`); broader sensor-related count is ~17 if you include I/O + decoders + perception primitives. |
| "Memory says 1119/1128 tests pass" | Verified collection is 1140; full pass count needs `torch` extra installed. README's 1025 number reflects `pytest -m "not integration"`. |
