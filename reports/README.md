# reports/ — measured outputs

Every JSON here is produced by a script in `tests/` or `tools/`. Re-run the
cited script to regenerate. These files are committed so that audit /
due-diligence reviewers can reproduce without running the full pipeline.

## Files

### `yolov8n_eval.json`

End-to-end detection accuracy of the AstraCore Neo INT8 PTQ recipe vs FP32
onnxruntime reference, on **28 preprocessed COCO-128 evaluation images**.

- Calibration set: 100 COCO-128 images (separate from eval set).
- Weight calibration: per-channel max-abs.
- Activation calibration: per-tensor percentile-99.9999.
- NPU score threshold: 0.20 (asymmetric vs ORT 0.25 — a production-recipe
  knob introduced to offset minor quantisation-induced score shift).
- Reproduce: `pytest -m "not integration" tests/test_yolov8n_eval_suite.py`.

Key numbers (aggregate over 28 images):
- IoU ≥ 0.50 match rate: **98.4 %**
- IoU ≥ 0.70 match rate: **96.0 %**
- IoU ≥ 0.90 match rate: **91.2 %**
- Tensor SNR vs FP32: min 24.1 / median 28.3 / max 32.8 dB

**Known limitation.** 28 images is a small sample for a "production-grade"
ML accuracy claim. A ≥ 500-image run should be executed before any external
publication of these numbers (see `docs/buyer_dd_findings_2026_04_19.md`
H8).

### `yolov8n_eval_100cal_maxabs.json` / `_pct9999.json` / `_pct99999.json` / `_pct999999.json`

Ablation of activation calibration method on the same eval set (100-image
calibration × {max-abs, 99.99th / 99.999th / 99.9999th percentile}).
Percentile-99.9999 is the recipe used in the production `yolov8n_eval.json`
above; the other three are the evidence that led to the choice.

### `pruning_accuracy.json` — **negative-test baseline, expected 0 %**

This file deliberately shows 0 % detection match for 2:4, 2:8, and 1:8
magnitude-only pruning on yolov8n. **The zeros are the intended result** —
they demonstrate that magnitude pruning *without QAT* collapses detection
accuracy, which is why the F1-A3 sparsity WP includes a QAT pipeline as a
blocker.

- Reproduce: `pytest tests/test_pruning_accuracy.py`.
- **Do not interpret as a bug** in the sparsity path. The RTL
  `ext_sparse_skip_vec` port works correctly (see
  `test_sparse_skip_zeros_products`); this file is about *model accuracy*
  at aggressive sparsity ratios, not RTL behaviour.

### `int2_baseline.json`

INT2 precision probe on yolov8n. Shows that INT2 saturates bbox regression
(representation wall at 3 effective grid levels) and detections collapse.
Shippable only with QAT + detection-head-at-higher-precision, same recipe
as 8:1 sparsity.

### `int2_probe_artefacts/`

Per-image intermediates from the INT2 probe above (tensor histograms,
per-layer SNR).

### `_run.log`

Rolling stdout from the last `pytest` invocation that wrote any of the
files above. Useful for debugging flaky runs; not authoritative.

## How to regenerate everything in one go

```bash
cd <repo-root>
pytest -m "not integration" \
       tests/test_yolov8n_eval_suite.py \
       tests/test_pruning_accuracy.py \
       tests/test_int2_probe.py
```
