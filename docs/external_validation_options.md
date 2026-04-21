# External Validation Options for AstraCore Neo IP

**Date:** 2026-04-21
**Purpose:** Catalogue of third-party / online validation services + benchmarks that complement the in-house cocotb + FMEDA + fault-injection evidence already in the safety case.
**Audience:** Founder + product/marketing/BD when planning the next 12 months of evidence-gathering.

> **Framing.** All of these produce *external* evidence — numbers / certs from a party other than AstraCore. External evidence is what OEM evaluators trust, what TÜV cites, and what closes IP-licensing deals. Internal evidence (our own tests) is necessary but not sufficient.

---

## 1. Performance benchmarking (perf-per-watt, latency, throughput)

These are **online services** that take your ONNX/TFLite model and report performance on real silicon. Useful for spec-sheet credibility.

| Service | What it measures | Cost | Effort | Relevance to Path A | Status |
|---|---|---|---|:---:|:---:|
| **Qualcomm AI Hub** ([aihub.qualcomm.com](https://aihub.qualcomm.com)) | Inference latency on Snapdragon devices (Hexagon NPU + Adreno GPU + Kryo CPU) for any uploaded ONNX/TFLite/PyTorch model. Free tier: 25 jobs/month. Paid: ~$50-200/job. | Free / low | 1-2 days to set up + per-model job | **HIGH** — automotive Tier-1s (Renesas, Continental) routinely reference Snapdragon AI Hub numbers as the public yardstick | not yet attempted |
| **MLPerf Inference** ([mlcommons.org](https://mlcommons.org)) | Standardised benchmark suite (ResNet-50, BERT, RetinaNet, GPT-J, …) with reference accuracy thresholds. Datacentre + Edge submission categories | $5k+ membership + ~10-20 engineer-weeks per submission round | high | **VERY HIGH** for Series-B narrative — without MLPerf numbers, perf claims are marketing | already in design doc §3.1 plan |
| **MLPerf Tiny** ([mlcommons.org/en/inference-tiny](https://mlcommons.org/en/inference-tiny)) | Small-NPU benchmark — visual wake words, image classification, anomaly detection, keyword spotting. Sub-1W power envelope. Submission rounds Apr/Oct | Same membership; ~2-4 engineer-weeks per submission | medium | **VERY HIGH** for zonal-controller positioning vs Hailo / Coral / NXP ENS24x | already in design doc §3.1 plan |
| **AI Benchmark (ETH Zurich)** ([ai-benchmark.com](https://ai-benchmark.com)) | Mobile NPU benchmark; long history; widely cited in academic + Android-OEM literature | Free (academic submissions) | low (1-2 weeks) | medium — mobile-focused; less weight in automotive | optional |
| **EEMBC MLMark** | Edge AI benchmark from EEMBC consortium; similar to MLPerf Tiny but membership-based | $5-20k/yr membership | medium | medium (specific automotive-supplier weight in Europe + Japan) | optional |
| **AWS Sagemaker Neo + Triton Model Analyzer** | Compiles + benchmarks ONNX models on AWS instances (CPU/GPU/Inferentia) | AWS pay-per-use (~$0.50-5/job) | low | low for IP-licensing claims (AWS infrastructure not the target silicon) | not relevant for IP datasheet |
| **OctoML / Octoflow** (now part of Microsoft) | Compile + benchmark across CPU/GPU/NPU; spec-sheet-friendly automated reports | Subscription (commercial) | low | medium | optional — only if BD cites it |
| **Hugging Face Inference Endpoints** | Benchmark transformer models on managed inference; rate-cards for cost/perf | Pay-per-use | low | low (datacentre-focused) | not relevant |

**Recommendation per `docs/best_in_class_design.md` §3.1 W3:** start with **MLPerf Tiny** (visual wake words is the smallest-effort entry; gets us on the public scoreboard alongside Hailo / Syntiant / NXP). After that, **Qualcomm AI Hub** for cross-comparison numbers OEMs already trust.

---

## 2. Automotive functional-safety certification

These are **assessment programmes** by accredited bodies that turn the safety case (HARA + FSC + TSC + FMEDA + fault-injection + …) into a binding cert mark. Required for any production OEM use.

| Service | What it certifies | Cost | Effort | Relevance | Status |
|---|---|---|---|:---:|:---:|
| **TÜV SÜD ASIL Concept Assessment** | Reviews safety case for ASIL-B / ASIL-C / ASIL-D readiness; issues a formal letter of conformance | $30-80k pre-engagement; $200-500k full audit | 6-12 months elapsed | **CRITICAL** for Path A — proposed default reviewer in `docs/safety/confirmation_review_checklist_v0_1.md` §1.3 | scheduled W6 per design doc §7.2 |
| **TÜV SÜD ISO 21434 Assessment** | Cybersecurity engineering assessment (companion to ASIL); required for automotive after July 2024 EU regulation | $50-100k | 6 months | HIGH | not yet scheduled |
| **TÜV Rheinland / SGS-TÜV / DEKRA** | Same scope as TÜV SÜD; competing accredited bodies | Similar cost band | Similar | medium (vendor-neutral; choice is BD-led) | optional |
| **UL 4600** (Standard for Safety for Autonomous Vehicles) | Vehicle-level autonomy safety — beyond ASIL scope | $100k+ | 12+ months | LOW for AstraCore IP (vehicle OEM scope) | not relevant |
| **AEC-Q100 / Q104** | Automotive electronics qualification (silicon-level; temperature, vibration, EMC) | $200-500k per silicon revision | 6-12 months | **Licensee-side** at their tape-out; AstraCore advisory only | Licensee scope per spec sheet rev 1.4 §12 |
| **IATF 16949** | Automotive quality management (org-level; not product) | $20-60k/year | ongoing | LOW for IP supplier; HIGH for licensee | optional later |

**Recommendation:** TÜV SÜD India pre-engagement is already W6 in plan. Add **ISO 21434 assessment** to the Q4 2026 milestone (post-OpenTitan crypto integration in Track 3) — the same TÜV office handles both for ~30% combined-engagement discount.

---

## 3. Hardware / silicon validation

These are **physical-silicon programmes** that test the IP on real fab silicon. Mostly Licensee-side or expensive shuttles.

| Service | What it tests | Cost | Effort | Relevance | Status |
|---|---|---|---|:---:|:---:|
| **AWS F1 (FPGA-as-a-service)** | Run RTL on VU9P (4096 MACs) at production-grade clocks; measured silicon-scale latency | $1.65/hr per f1.2xlarge instance; ~$1-3k/month for sustained dev | medium | **HIGH** — first measured silicon-scale latency without tape-out | F1-F1..F3 in original plan; deferred for safety-cert focus |
| **Intel Agilex / Achronix Speedster cloud** | Equivalent FPGA-as-a-service on different toolchain | similar cost | medium | medium (alternative if AWS F1 has issues) | optional |
| **Tiny Tapeout** ([tinytapeout.com](https://tinytapeout.com)) | Open-source shuttle program; sky130 small-area tapeouts | $300/tile | low (~weeks per shuttle) | LOW (130 nm; not production-relevant) but cheap to demonstrate first-silicon | optional showcase |
| **Efabless Open MPW shuttle** | Caravel SoC integration on sky130; free shuttles via Google sponsorship | $0 (sponsored) for accepted designs | medium-high | medium (proves the IP integrates into a real SoC; sky130 still not production-relevant) | optional |
| **Tape-out at MPW (multi-project wafer) on TSMC N7 / N5** | Shared-wafer production-grade shuttle | $500k-2M per shuttle slot | 12-18 months | **Path B territory** — Series-B funded | post-Series-B per design doc §3.3 |
| **Cadence / Synopsys EDA cloud trials** | Free 30-90 day trials of commercial synthesis + STA tools (closes Yosys/OpenROAD TCL2/3 gap) | Free trial; ~$50-200k/year per seat after | low (trial setup) | **MEDIUM** — helps close TCL2 (Yosys → commercial cross-check) | F4-B-7 in remediation plan |

**Recommendation:** start AWS F1 bring-up (F1-F1..F3) as soon as bandwidth allows — first measured silicon-scale latency unblocks the Series-A→B narrative. Tiny Tapeout is a cheap showcase (~$300, 1-2 weeks) for demo days but doesn't move the IP-licensing needle on its own.

---

## 4. Open-source / community evaluation

These are **community-run benchmarks** with low cost + high signal-to-noise for early traction.

| Service | What it provides | Cost | Effort | Relevance | Status |
|---|---|---|---|:---:|:---:|
| **OpenTitan compatibility self-cert** ([opentitan.org](https://opentitan.org)) | Verify our crypto integration matches OpenTitan reference behaviour | Free | Track 3 deliverable | HIGH (closes ASR-HW-12, ASR-HW-13) | scoped Track 3 |
| **MLIR / IREE conformance suite** | Validate our compiler frontend against the MLIR community's regression suite | Free | medium | medium (IP licensee may use IREE as their compiler) | optional |
| **ONNX Runtime EP conformance** | We already plug in via `astracore.backends.ort`; conformance test from Microsoft validates EP API correctness | Free | low (already in flight per memory note `step1_ort_multi_ep_complete.md`) | HIGH | partially in place |
| **Hugging Face Optimum integration** | Submit ONNX-graph-rewrite plugin so HF transformers can target AstraCore directly; gets us on the HF model card | Free | medium (~2 engineer-weeks) | HIGH for visibility (every HF download becomes a touchpoint) | optional Track 3 |
| **TVM BYOC backend** | Build Apache TVM bring-your-own-codegen path for AstraCore; gets us on the TVM website + papers | Free | medium-high (~4-6 weeks) | medium (academic + open-hardware visibility) | optional |
| **NIST CAVP / CMVP** ([csrc.nist.gov](https://csrc.nist.gov)) | Crypto algorithm validation — validates AES / RSA / SHA-256 implementations conform to FIPS-197 / FIPS-186 / FIPS-180 | Free for self-test, $10-50k for cert | medium | HIGH if defence/medical use cases enter scope | post-OpenTitan integration |

**Recommendation:** **Hugging Face Optimum integration** is the single highest-ROI item here — every HF model card with an "Inference Provider: AstraCore" tag is a free OEM touchpoint. ~2-week engineering effort.

---

## 5. Recommended sequencing (next 12 weeks)

| Wk | Test/cert | Why now |
|---:|---|---|
| 5-6 | **MLPerf Tiny** — visual wake words submission (smallest entry) | Ahead of October submission deadline; first public scoreboard appearance |
| 6 | **TÜV SÜD India** pre-engagement workshop | Already in plan |
| 6-7 | **Qualcomm AI Hub** — submit YOLOv8-N + GPT-2 from the model zoo as cross-reference benchmarks | Free; takes a couple of days; produces directly comparable Snapdragon numbers |
| 7-9 | **AWS F1** F1-F1..F3 bring-up | First measured silicon-scale latency |
| 8 | **Hugging Face Optimum** AstraCore EP plugin | Visibility multiplier |
| 10-12 | **MLPerf Inference Edge** submission preparation | Q1 2027 round; Series-B ready evidence |
| 14 | **TÜV SÜD ISO 21434** kickoff (post-OpenTitan integration) | Bundle with ASIL-B audit for combined discount |
| 16 | **Cadence / Synopsys EDA cloud trial** for F4-B-7 (Yosys TCL2 cross-check) | Closes the TCL2 qualification gap |

---

## 6. What is NOT useful (avoid)

- **Generic "AI accelerator benchmark" tools that don't publish methodology** — anything from a vendor blog without a public test suite is marketing noise.
- **Datacentre-only benchmarks** (Hugging Face Inference Endpoints, AWS Sagemaker Neo) — measured numbers don't translate to automotive zonal envelope.
- **UL 4600** at the IP level — that's vehicle-level autonomy scope; out of AstraCore's licensable IP.
- **AEC-Q100 attempted at the IP level** — silicon-level qualification at Licensee tape-out only.
- **Whitepaper "best-in-class" claims without third-party numbers** — destroys credibility on first OEM DD pass.

---

## 7. Document control

| Field | Value |
|---|---|
| Document ID | ASTR-EXT-VAL-V0.1 |
| Revision | 0.1 |
| Revision date | 2026-04-21 |
| Author | Track 2 collaborator + founder |
| Distribution | Internal — strategy + BD + product |
| Supersedes | None |
