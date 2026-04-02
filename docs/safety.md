# Module 6 — Safety

**Status:** DONE | **Tests:** 92/92 (100%) | **Date:** 2026-04-02

## Overview

The Safety module implements the ASIL-D safety mechanisms required for automotive chip certification (ISO 26262). It provides four subsystems — ECC memory protection, TMR voting, watchdog timer, and clock monitoring — plus a central safety manager that aggregates events and drives the chip safety state machine.

## Sub-modules

### ecc.py — Error Correcting Code (SECDED)
- **ECCEngine** — SECDED(72,64): 64 data bits + 8 parity bits
- **encode(data)** → 72-bit codeword
- **decode(codeword)** → CorrectionResult (auto-corrects single-bit, raises on double-bit)
- **scrub_bank(bank, words)** — background scrubbing, re-encodes corrected words
- Per-bank counters: single-bit errors, double-bit errors, corrections, scrub count
- **ECCConfig** — configures data_bits, parity_bits, banks, scrub_interval_words
- **BitFlipType** — NONE, SINGLE_BIT, DOUBLE_BIT, MULTI_BIT

### tmr.py — Triple Modular Redundancy
- **TMRVoter** — majority vote across three independent compute lanes (A, B, C)
- Supports scalars (int, float) and numpy arrays
- Float comparison uses configurable `float_atol` tolerance
- On disagreement: flags faulty lane, returns majority value
- On triple disagreement: raises **TMRError**
- Per-lane fault counters, total vote count, triple disagreement counter
- **TMRResult** — voted_value, faulty_lane, vote_count, agreement flag

### watchdog.py — Windowed Watchdog Timer
- **WatchdogTimer** — must be serviced within [window_open_ms, timeout_ms] window
- Too-early service (before window opens): raises WatchdogError (window violation)
- Service with wrong token: raises WatchdogError
- Timeout (no service): raises WatchdogError, increments timeout counter
- **escalation_level()** — INTERRUPT → NMI → RESET based on timeout count
- Deterministic testing via `_inject_elapsed(ms)` — no real sleeping needed
- **WatchdogConfig** — timeout_ms, window_open_ms, token, max_timeouts_before_reset

### clock_monitor.py — Clock Monitor
- **ClockMonitor** — monitors multiple clock domains for faults
- **check_frequency(domain, measured_mhz)** — detects FREQ_HIGH / FREQ_LOW
- **check_clock_loss(domain, elapsed_us)** — detects CLOCK_LOSS
- **check_glitch(domain, pulse_width_ns)** — detects GLITCH
- **ClockFaultType** — NONE, FREQ_HIGH, FREQ_LOW, CLOCK_LOSS, GLITCH, JITTER_EXCESS
- Fault log with history, any_fault() convenience check
- **ClockMonitorConfig** — expected_mhz, tolerance_pct, loss_timeout_us, min_pulse_width_ns

### safety_manager.py — Central Safety Coordinator
- **SafetyManager** — aggregates events, drives chip safety state machine
- **SafetyState** — NORMAL → DEGRADED → SAFE_STATE → SHUTDOWN (escalates only)
- **report_event(source, severity, message)** — logs event, updates state
- **SafetySeverity** — INFO (NORMAL), WARNING (DEGRADED), CRITICAL (SAFE_STATE), FATAL (SHUTDOWN)
- **is_safe_to_operate()** — True in NORMAL or DEGRADED states only
- Event log queryable by source, severity; highest_severity() convenience method

## Dependencies
- Needs: HAL (module 1), Compute (module 3)
- Required by: Telemetry (module 8)

## Critical Design Notes
- **SECDED overall parity bug to avoid:** P7 must be computed as XOR of ALL codeword bits (data + P0-P6). A naive approach computing P7 from data + P0-P6 in one pass causes P0-flip and data-flip to cancel, making single-bit errors appear as double-bit. Fix: separate `_compute_hamming()` (7 bits) from `_compute_parity()` (adds P7), and in decode recompute overall parity across all received bits including received Hamming bits.
- TMR with numpy arrays compares using `np.allclose` for floats and `np.array_equal` for integers
- Watchdog window violation (too-early kick) is a safety fault, not just ignored
