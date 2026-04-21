# Leaderboard reproduce guide

Exact commands to regenerate every row in `LEADERBOARD.md`.
Run in order; total wall time ~5 minutes on a typical laptop.

## Prerequisites

```bash
pip install -e .
python scripts/fetch_model_zoo.py          # downloads ~1.5 GB of ONNX
```

## Generate the artefacts

```bash
# §1 — zoo latency matrix
astracore zoo --iter 3 \
    --out reports/benchmark_sweep/zoo.json \
    --md-out reports/benchmark_sweep/zoo.md

# §1 — latency distribution
python scripts/bench_zoo_detailed.py \
    --threads 1,2,4 --batch 1,2,4 --gopt basic,extended,all \
    --warmup 3 --iter 20

# §1 — INT8 manifest for all 8 models
python scripts/quantise_zoo.py --cal-samples 50

# §2 — multi-stream scaling (yolov8n, shufflenet, mobilenet)
astracore multistream --model data/models/yolov8n.onnx \
    --streams 1,2,4,8 --duration 3.0 \
    --out reports/benchmark_sweep/multistream_yolov8n.json \
    --md-out reports/benchmark_sweep/multistream_yolov8n.md
astracore multistream --model data/models/zoo/shufflenet-v2-10.onnx \
    --streams 1,2,4,8 --duration 2.0 \
    --out reports/benchmark_sweep/multistream_shufflenet.json \
    --md-out reports/benchmark_sweep/multistream_shufflenet.md
astracore multistream --model data/models/zoo/mobilenetv2-7.onnx \
    --streams 1,2,4,8 --duration 2.0 \
    --out reports/benchmark_sweep/multistream_mobilenet.json \
    --md-out reports/benchmark_sweep/multistream_mobilenet.md

# §3 — deep AI tests (determinism + drift)
python scripts/ai_model_deep_tests.py

# §4 — safety fusion scenarios
python scripts/run_realworld_scenarios.py --only 4

# Assemble LEADERBOARD.md + reports/leaderboard.json
python scripts/make_leaderboard.py
```

## Verify

```bash
pytest -m 'not integration' -q      # ≥ 1370 tests should pass
```

## Troubleshooting

- If `astracore quantise` complains about Dropout/BatchNorm, pass `--engine ort` explicitly.
- If a deep-test JSON is missing, re-run the specific driver (`ai_model_deep_tests.py`, `run_realworld_scenarios.py`).
- Fingerprints drift between ORT versions — see `tests/yolo_fingerprint_baseline.json` and regenerate if needed.
