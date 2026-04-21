# AstraCore Neo — NPU IP + Software Simulation SDK

Python + SystemVerilog reference implementation of the AstraCore A2 Neo automotive AI inference accelerator.

**Chip target:** A2 Neo (7 nm, tape-out pending)
**Spec sheet rev:** 1.3 (10.07.2025)
**Validation status:** 1025 Python tests + 6/6 cocotb RTL tests passing as of 2026-04-19
**See also:** [`LEADERBOARD.md`](LEADERBOARD.md) · [`cpp/README.md`](cpp/README.md) · [`docs/spec_sheet_provenance.md`](docs/spec_sheet_provenance.md) · [`docs/buyer_dd_findings_2026_04_19.md`](docs/buyer_dd_findings_2026_04_19.md) · [`docs/cloud_readiness_playbook.md`](docs/cloud_readiness_playbook.md)

---

## What works today (validated)

| Area | Component | Status | Evidence |
|---|---|---|---|
| Compute | INT8 / INT4 / INT2 MAC datapath | ✅ | `rtl/npu_pe/npu_pe.v`, cocotb 6/6 PASS |
| Compute | Systolic array (4×4, 8×8 validated) | ✅ | `rtl/npu_systolic_array/`, parameterised |
| Compute | Softmax RTL | ✅ | `rtl/npu_softmax/`, bit-exact vs Python mirror |
| Compute | LayerNorm / RMSNorm RTL | ✅ | `rtl/npu_layernorm/`, bit-exact vs mirror |
| Compute | Activation unit (ReLU, SiLU, Sigmoid, GeLU, Tanh LUT) | ✅ | `rtl/npu_activation/`, `afu_luts.vh` |
| Compiler | ONNX 2.0 loader | ✅ | `tools/npu_ref/onnx_loader.py`, 233-node yolov8n loads cleanly |
| Compiler | INT8 quantiser (per-channel weights, per-tensor percentile activations) | ✅ | `tools/npu_ref/quantiser.py` |
| Compiler | INT4 quantiser (SNR 15.7 dB on yolov8n) | ✅ | F1-B2 |
| Compiler | SiLU fusion pass | ✅ | `tools/npu_ref/fusion.py` |
| Compiler | Im2col conv2d → matmul with K/N tiling | ✅ | F1-C3, F1-C4 |
| Frontend | PyTorch, TVM, XLA, MLIR, NNEF (share ONNX path) | ✅ | `tools/frontends/` |
| Runtime | Python end-to-end inference | ✅ | `tools/npu_ref/nn_runtime.py` |
| Safety | ECC SECDED (Hamming 72/64) | ✅ | `rtl/ecc_secded/` |
| Safety | TMR voter, used in dms_fusion | ✅ | `rtl/tmr_voter/` |
| Safety | Safe-state controller, plausibility checker | ✅ | `rtl/safe_state_controller/`, `rtl/plausibility_checker/` |
| Sensor Fusion | 20 RTL modules (camera/radar/lidar/IMU/GNSS/DMS) | ✅ | 32/32 OpenLane sky130 ASIC batch PASS |
| Sim | Verilator 5.030 + cocotb 2.0.1 via WSL Ubuntu 22.04 | ✅ | `tools/run_verilator_*.sh` |

## What is in-flight (WPs open, closable pre-tape-out)

| WP | Scope |
|---|---|
| F1-A1.1 | Synthesizable FP8 / FP16 datapath (sim-gate already 5/5 PASS; **not yet wired into `npu_top`**) |
| F1-A2 | BF16 / TF32 / FP32 (blocked on F1-A1.1) |
| F1-A3 | 8:1 sparsity engine (metadata decoder + compaction pipeline + QAT pipeline) |
| F1-A5 | 8× MHSA tile (blocked on F1-A1.1 / A4) |
| F1-A6 | AES-256 + RSA-2048 RTL (no current hits in `rtl/` for any crypto) |
| F1-A7 | NIST PQC (Kyber, Dilithium) |
| F1-A8 | Weight compression 3–5× |
| F1-A9 | ML-based thermal / fault predictor (currently rule-based) |
| F1-B3 | C++ runtime (parity with Python) |
| F1-B6 | Broader model library validation (EfficientNet-B7, ViT-Large, BEVFormer, BERT, Swin) |
| F1-F1..F3 | FPGA bring-up (Vivado → AWS F1 VU9P, 64×64 = 4 096 MACs cap) |

## What is post-tape-out (silicon-only, not demonstrable on FPGA)

| Claim | Why |
|---|---|
| 24 576-MAC array (48 × 512) | Silicon area budget; FPGA cap 4 096 MACs on VU9P |
| 2.5–3.2 GHz clock | 7 nm timing closure; FPGA runs at 100 MHz |
| 1258 TOPS (INT8) / 2516 TOPS (INT4) | Requires 24 576 MACs × 3.2 GHz × 8:1 sparsity on silicon |
| 15–30 TOPS/W, 40–50 W typical | Silicon power sign-off |
| 128 MB on-chip SRAM | Silicon area; RTL bank is parameterised but default 256 bytes |
| LPDDR5X 400 GB/s / HBM3 750 GB/s | Licensed PHY + memory-controller IP; not in current plan |
| UCIe chiplet 2000+ TOPS | Licensed UCIe IP + package |
| MIPI C-PHY, 10/100 G Ethernet PHY, PCIe Gen4 PHY | Silicon IP |
| ISP-Pro, 8K HDR, AI denoising | No `rtl/isp/` today |
| –40 °C to +125 °C operating range | Silicon + package + board |
| ASIL-D formal certification | FMEDA + fault-injection + external certifier (post-tape-out program) |

## Quick start

```bash
# Python simulation (Windows / Linux / macOS)
python -m venv .venv && .venv/bin/pip install -r requirements.txt
pytest -m "not integration"                    # 1025 tests
python tools/npu_ref/nn_runtime.py --help      # end-to-end yolov8n demo

# RTL cocotb (WSL Ubuntu 22.04, Verilator 5.030)
bash tools/run_verilator_npu_top.sh            # 6/6 PASS
bash tools/run_verilator_npu_top_8x8.sh        # 2/2 PASS
bash tools/run_verilator_npu_softmax.sh        # F1-A4 softmax bit-exact gate
bash tools/run_verilator_npu_layernorm.sh      # F1-A4 layernorm/rmsnorm gate
```

## Repository layout

```
rtl/                  SystemVerilog design (48 modules: 14 NPU + 20 fusion + 14 IO/safety/top)
sim/                  cocotb testbenches (53 test modules)
tools/npu_ref/        Python reference: loader, quantiser, runtime, compiler, perf model
tools/frontends/      PyTorch, TVM, XLA, MLIR, NNEF → ONNX glue
src/                  Python platform SDK (hal, memory, compute, safety, security, telemetry, dms)
asic/                 OpenLane flow (sky130 130 nm prototyping; NOT the 7 nm tape-out flow)
constraints/          FPGA timing constraints (Artix-7 @ 100 MHz)
fpga/aws_f1/          AWS F1 VU9P bring-up package (CL wrapper, driver, makefile)
data/calibration/     Preprocessed .npz blobs for YOLOv8 calibration + eval (28 images)
reports/              JSON outputs: yolov8n eval, pruning baseline, INT2 probe
docs/                 Architecture, provenance, audit findings
memory/               Session handoff notes (not indexed by code)
```

## License

See [`LICENSE`](LICENSE) — the Python SDK (`src/`, `tools/`) is Apache-2.0; the RTL (`rtl/`, `constraints/`, `fpga/`, `asic/`) is source-available under a non-commercial evaluation licence pending a formal commercial posture decision.

## Getting help

- **Architecture**: `docs/architecture.md`, `docs/astracore_v2_npu_architecture.md`
- **Spec-sheet provenance**: `docs/spec_sheet_provenance.md` (maps each spec-sheet claim to RTL, WP, or post-tape-out)
- **Buyer-level DD findings (2026-04-19)**: `docs/buyer_dd_findings_2026_04_19.md`
- **Sensor fusion**: `docs/sensor_fusion_architecture.md`
