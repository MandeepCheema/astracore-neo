# inference_runtime — RTL Module

**File:** `rtl/inference_runtime/inference_runtime.v`
**Purpose:** Inference session state machine — manages AI model load and run lifecycle

## What it does
Controls the full lifecycle of an AI inference session: load a model, run it, handle completion and errors. Acts as the job manager that coordinates with the MAC array.

## States
| Value | State | Meaning |
|-------|-------|---------|
| 3'd0 | UNLOADED | No model loaded |
| 3'd1 | LOADED | Model ready, waiting to run |
| 3'd2 | RUNNING | Inference in progress |
| 3'd3 | DONE | Session complete |
| 3'd4 | ERROR | Fault state |

## State transitions
```
UNLOADED + load_start + model_valid  → LOADED
UNLOADED + load_start + !model_valid → ERROR
LOADED   + run_start                 → RUNNING
RUNNING  + run_done_in               → DONE
DONE     + load_start                → LOADED  (reload)
DONE     + run_start                 → RUNNING (re-run same model)
any      + abort                     → ERROR
ERROR    + load_start                → UNLOADED (recovery)
```

## Ports
| Port | Dir | Width | Description |
|------|-----|-------|-------------|
| clk | in | 1 | System clock |
| rst_n | in | 1 | Active-low synchronous reset |
| load_start | in | 1 | Request to load a model |
| model_valid | in | 1 | Model data is valid (checked at load_start) |
| run_start | in | 1 | Request to start inference |
| abort | in | 1 | Abort current session (goes to ERROR) |
| run_done_in | in | 1 | Signal from inference engine: job complete |
| state | out | 3 | Current state (0–4) |
| busy | out | 1 | High when LOADED or RUNNING |
| session_done | out | 1 | Pulsed one cycle on RUNNING→DONE |
| error | out | 1 | High when state == ERROR |

## Key implementation details
- `abort` takes priority over all other inputs in next-state logic
- `busy` is combinational: `(state == LOADED) || (state == RUNNING)`
- `error` is combinational: `(state == ERROR)`
- `session_done` pulses exactly one cycle: registered `(state==RUNNING) && (next_state==DONE)`
- All input strobes auto-clear next cycle in the top-level (wreg_inf bits 0,2,3,4)

## AXI4-Lite connection (in astracore_top)
- Write: `INF_CTRL` register (0x48) — `[0]=load_start`, `[1]=model_valid`, `[2]=run_start`, `[3]=abort`, `[4]=run_done_in`
- Read: `INF_ST` register (0xC4) — `[2:0]=state`, `[3]=busy`, `[4]=session_done`, `[5]=error`
- LED: `led[2]` = `busy`
