# AstraCore Neo — Software Simulation SDK

Python-based functional simulation of the AstraCore Neo automotive AI inference accelerator.

**Chip:** A2 Neo | **Rev:** 1.3 | **Date:** 10.07.2025  
**Target:** ISO 26262 ASIL-D | 1258 TOPS (INT8) | L2+–L4 ADAS

## Modules

| # | Module | Description | Status |
|---|--------|-------------|--------|
| 1 | hal | Hardware Abstraction Layer | PENDING |
| 2 | memory | SRAM, DMA, Compression | PENDING |
| 3 | compute | MAC Array, Sparsity, Transformer Engine | PENDING |
| 4 | inference | Compiler, Quantizer, Runtime | PENDING |
| 5 | perception | Camera, Lidar, Radar, Fusion | PENDING |
| 6 | safety | ECC, TMR, Watchdog, Clock Monitor | PENDING |
| 7 | security | Secure Boot, TEE, OTA | PENDING |
| 8 | telemetry | Logger, Fault Predictor, Thermal | PENDING |
| 9 | dms | Driver Monitoring System | PENDING |
| 10 | connectivity | V2X, CAN-FD, Ethernet, PCIe | PENDING |
| 11 | models | Pre-validated Model Library | PENDING |

## Setup

```bash
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/test_<module>.py -v 2>&1 | tee logs/test_<module>.log
```

## Session Protocol

Each session builds one module. Start by reading `PLAN.md`.
