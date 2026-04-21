# AstraCore Neo — External Buyer Technical DD Findings (2026-04-19)

**Scope.** Five-pass audit of AstraCore Neo spec sheet rev 1.3 (10.07.2025) against the repo, performed as an independent external-buyer (automotive OEM) would.

**Methodology.** No trust in internal `.md` summaries. Every claim verified by running code, reading RTL, inspecting synthesis artefacts, or measuring primary data.

**Fact-base (reproducible commands):**

| # | Verification | Command | Result |
|---|---|---|---|
| 1 | Python test suite | `pytest -m "not integration"` | **1025 pass, 1 skip, 7 deselected** (166 s) |
| 2 | RTL cocotb (4×4 NPU top) | `wsl bash tools/run_verilator_npu_top.sh` | **6/6 PASS** |
| 3 | Peak TOPS formula | `24576 × 2 × 3.2 GHz × 8 ÷ 1e12` | **1258.3** (arithmetic correct) |
| 4 | Validated array size | `grep parameter N_ROWS rtl/npu_top/npu_top.v` | **4×4 default; 8×8 max validated** |
| 5 | Synthesis technology | `asic/runs/*/config.json::STD_CELL_LIBRARY` | `sky130_fd_sc_hd` (**130 nm, 1.8 V**) only |
| 6 | FPGA target | `constraints/astracore_fusion_top.xdc` | **Artix-7 @ 100 MHz** |
| 7 | Crypto RTL presence | `grep -riE 'aes|rsa|sha256|kyber|dilithium' rtl/` | **0 hits** |
| 8 | Memory controller RTL | `find rtl -type d -iname '*ddr*|*hbm*|*mem_ctrl*'` | **0 modules** |
| 9 | FP in main datapath | `grep npu_fp_mac rtl/npu_top/ rtl/npu_systolic_array/` | **0 references** |
| 10 | FMEDA / fault injection | `find -iname '*fmeda*|*fault_inj*|*stuck_at*|*SEU*'` | **0 files** |

## FATAL (F1–F7) — block any automotive-OEM technical DD

### F1 — Validated at 0.26 % of claimed compute scale
PDF: 24 576 MACs. RTL default: `N_ROWS=4, N_COLS=4 = 16 MACs`. Largest validated cocotb: 8×8 = 64 MACs. Gap: **384×**. Parameterisation ≠ validation — no evidence the compiler / SRAM ctrl / DMA / tile scheduler behave correctly at spec scale.

### F2 — No 7 nm (or any sub-GHz) synthesis evidence
Only synthesis flow is OpenLane on **sky130 130 nm, 1.8 V**. FPGA constraint: **Artix-7 @ 100 MHz**. Claim: 2.5–3.2 GHz. Gap ≥ 10× to validated frequency, ≥ 32× to claimed.

### F3 — "1258 TOPS" multiplies two unmeasured projections
157 TOPS (silicon-scale, no FPGA proof) × 8 (8:1 sparsity multiplier with no QAT — `reports/pruning_accuracy.json` shows **0 % detection match** at 2:4/2:8/1:8 magnitude pruning).

### F4 — Zero cryptographic RTL
PDF: *"AES-256, RSA-2048, NIST PQC (Kyber, Dilithium), hardware key storage"*. `grep rtl/` for any crypto primitive → **0 hits**. `src/security/*.py` are Python skeletons with no RTL substrate.

### F5 — Zero memory controller RTL for claimed bandwidth
PDF: 400 GB/s LPDDR5X / 750 GB/s HBM3 / 128 MB on-chip SRAM.
- No `rtl/ddr*`, `rtl/lpddr*`, `rtl/hbm*` anywhere.
- SRAM bank default `DEPTH=256` bytes (`rtl/npu_sram_bank.v:32-33`) — **500 000× smaller** than claim.

### F6 — No ISO 26262 safety case
ASIL-D requires per Part 5 §7 a systematic fault-injection campaign + FMEDA with DC / LFM / SPFM + CCF analysis. Repo: no fault-injection files, no FMEDA, no DC/LFM numbers. TMR voter + ECC blocks are mechanisms, not a safety case.

### F7 — Floating-point MACs not wired into the main datapath
`rtl/npu_fp/npu_fp_mac.v` and `npu_fp_pe.v` exist but have **zero references** in `rtl/npu_top/` or `rtl/npu_systolic_array/`. `npu_pe.v:179-182` falls back to INT8 on `precision_mode=2'b11` (FP16). Result: **all FP modes on PDF row (FP4, FP8 E4M3/E5M2, FP16, BF16, TF32, FP32) are non-functional in the compute array.**

## HIGH (H1–H9)

| # | Finding |
|---|---|
| H1 | "5-stage PE pipeline" is Python-only (`src/compute/mac_array.py`). RTL has 1 stage (`npu_pe.v:44`) |
| H2 | Transformer Engine "dynamic sparsity" + "dedicated sparsity engine" + 8×MHSA tile have no RTL. Softmax/LayerNorm/GeLU DO exist |
| H3 | "Typical 500–700 TOPS YOLOv8" contradicted by measured 6.14 % single-stream util → ~77 TOPS |
| H4 | "<0.5 ms latency" — repo perf model projects YOLOv8-N at 1.45 ms silicon-scale |
| H5 | "ISP-Pro, 8K HDR, AI denoising" — zero RTL, zero Python model |
| H6 | "Runtime Protection: AXI snooping" — no AXI bus exists in `rtl/` |
| H7 | "On-Chip Training" — no backward-pass RTL, no gradient datapath |
| H8 | YOLOv8 eval corpus is 28 preprocessed `.npz` blobs, not raw images |
| H9 | UCIe, chiplet 2000+ TOPS, C-PHY, 10/100 G Ethernet PHY, C-V2X, 8K MIPI, Gen4 PHY all depend on unlicensed silicon IP |

## MEDIUM (M1–M8)

- **M1** `README.md` says every module "PENDING" — contradicts 1025 passing tests
- **M2** No root `LICENSE` despite "Partially Open-Source SDK" claim
- **M3** INT2 implemented in RTL but absent from PDF precision list (undersell)
- **M4** TensorRT listed as compiler front-end (it's a runtime)
- **M5** "Energy Harvesting" is board-level, category error on a chip spec
- **M6** "Cloud Platform" is a services offering, not a chip feature
- **M7** "Predictive ML-based fault/thermal" is rule-based thresholding
- **M8** Stray `|` typo in PDF page 3 Typical Power row

## What actually ships today (buyer-truth)

- INT8 / INT4 / INT2 MAC datapath validated 4×4 / 8×8, bit-exact vs Python mirror
- Compiler + INT8 quantiser + Python runtime end-to-end on YOLOv8-N
- 20 sensor-fusion RTL modules, 32/32 OpenLane sky130 ASIC batch pass
- Softmax + LayerNorm RTL (F1-A4) bit-exact vs mirror
- ECC SECDED + TMR voter RTL
- 1025 Python tests + 6/6 cocotb npu_top (confirmed this session)

## Verdict

Credible **small INT8 NPU IP + working compiler + good sensor-fusion RTL** validated at prototype scale on an academic 130 nm flow. The spec sheet is written for a product that needs **≈18–24 months** of additional work plus licensed memory/crypto/PHY IP plus a 7 nm tape-out to match. The gap between document and repo is the majority of the claim surface.

**Recommended rewrite pattern:** two-column spec à la NVIDIA Thor / Qualcomm Ride / Mobileye EyeQ — *"Validated today"* (~25 % of current text) and *"Tape-out target (projected)"* with ⚠️ footnote.
