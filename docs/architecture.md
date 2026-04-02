# AstraCore Neo — System Architecture

**Chip:** A2 Neo | **Rev:** 1.3  
**Target:** ISO 26262 ASIL-D | 1258 TOPS (INT8) | L2+–L4 ADAS

---

## Overview

The AstraCore Neo SDK is a Python functional simulation of the A2 Neo inference accelerator. It mirrors the chip's hardware architecture as a layered software stack, enabling:

- Algorithm validation before silicon tape-out
- SDK/driver development against a faithful hardware model
- ASIL-D safety verification via simulation
- Integration testing with ADAS pipelines

---

## Layer Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│          (ADAS pipelines, smart cockpit apps)            │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                   Model Library (11)                     │
│    YOLOv8 · BEVFormer · LLaMA-13B · Stable Diffusion    │
└────┬──────────────┬────────────────────────────────────┘
     │              │
┌────▼─────┐  ┌─────▼──────────────────────────────────┐
│Inference │  │         Perception (5)                   │
│   (4)    │  │  Camera · Lidar · Radar · Fusion         │
│Compiler  │  └────────────────┬───────────────────────┘
│Quantizer │                   │
│Runtime   │         ┌─────────▼──────┐
└────┬─────┘         │    DMS (9)      │
     │               │ Alertness·LP   │
┌────▼──────────────────────────────────────────────────┐
│                   Compute (3)                          │
│       MAC Array · Sparsity Engine · Transformer       │
└──────────────────────┬────────────────────────────────┘
                       │
┌──────────────────────▼────────────────────────────────┐
│                   Memory (2)                           │
│          128MB SRAM · DMA · Neural Compression         │
└──────────────────────┬────────────────────────────────┘
                       │
     ┌─────────────────┼─────────────────────┐
     │                 │                     │
┌────▼────┐    ┌───────▼──────┐    ┌─────────▼──────────┐
│Safety(6)│    │ Security (7) │    │  Connectivity (10)  │
│ECC·TMR  │    │ Boot·TEE·OTA │    │ V2X·CAN-FD·ETH·PCIe│
│Watchdog │    └──────────────┘    └────────────────────┘
└────┬────┘
     │
┌────▼──────────────────────────────────────────────────┐
│                  Telemetry (8)                         │
│       Logger · Fault Predictor · Thermal Control      │
└──────────────────────┬────────────────────────────────┘
                       │
┌──────────────────────▼────────────────────────────────┐
│              HAL — Hardware Abstraction Layer (1)      │
│    AstraCoreDevice · RegisterFile · InterruptCtrl      │
└───────────────────────────────────────────────────────┘
```

---

## Module Summary

| # | Module | Key Specs from Chip |
|---|--------|---------------------|
| 1 | hal | Device state machine, register map, 32-bit IRQ controller |
| 2 | memory | 128MB SRAM (16×8MB), 400GB/s LPDDR5X, 3–5× neural compression |
| 3 | compute | 24,576 MACs, 8:1 sparsity, transformer engine (8×MHSA) |
| 4 | inference | ONNX/TVM/MLIR compiler, INT4/FP8 quantizer, C++/Python runtime |
| 5 | perception | MIPI CSI-2, 4D lidar, radar-camera-lidar fusion |
| 6 | safety | ECC, TMR (triple-modular redundancy), watchdog, ASIL-D |
| 7 | security | AES-256/RSA-2048/PQC boot, secure enclave, delta OTA |
| 8 | telemetry | Real-time logging, ML fault prediction, thermal control |
| 9 | dms | BlazeFace alertness, 256 MAC @ 500 MHz always-on mode |
| 10 | connectivity | C-V2X, 2×CAN-FD, 1/10/100G Ethernet TSN, PCIe Gen4 |
| 11 | models | YOLOv8, BEVFormer, LLaMA-13B, Stable Diffusion (quantized) |

---

## Key Design Principles

1. **HAL is the only hardware boundary.** No module above HAL touches registers or interrupts directly.
2. **Safety is cross-cutting.** Module 6 (safety) wraps any compute path used in ASIL-D contexts.
3. **DMS is always-on and independent.** Runs in LOW_POWER mode, never blocked by inference pipeline.
4. **Quantization is compile-time.** Precision (INT4/INT8/FP8) is a compiler artifact, not a runtime switch.
5. **Telemetry is async.** Fault prediction runs as a background service off hardware counters.
