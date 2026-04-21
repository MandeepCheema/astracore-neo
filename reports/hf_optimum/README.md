# reports/hf_optimum/ — Hugging Face Optimum integration evidence

This folder records the first validation pass against Track 3 item 8 of
`docs/external_validation_options.md` — "Hugging Face Optimum AstraCore
EP plugin".

## What was tested

Two reference exports produced by `optimum-cli export onnx` for the
`question-answering` task, shipped under
`data/models/zoo/optimum_exports/`:

| Export             | Producer       | Opset | Inputs                                      | Size     |
|--------------------|---------------|-------|---------------------------------------------|----------|
| `distilbert-squad` | pytorch 2.11  | 17    | input_ids, attention_mask                   | 265 MB   |
| `bert-large-squad` | pytorch 2.11  | 17    | input_ids, attention_mask, token_type_ids   | 1.28 GB  |

## Suite

`tests/test_hf_optimum.py`, 19 test cases split:

- **Default** (`pytest tests/test_hf_optimum.py`): 19 pass in ~47 s.
- **Integration** (`pytest -m integration tests/test_hf_optimum.py`): 3
  heavyweight BERT-large cases pass in ~39 s.

Per-case coverage, harness fix, and the full environment snapshot live in
`results.json` alongside this file.

## Reproduce

```bash
# Fast loop (CPU-only, < 1 min)
pytest tests/test_hf_optimum.py

# Heavy loop (downloads 1.3 GB BERT-large export if absent)
pytest -m integration tests/test_hf_optimum.py
```

## What the suite does NOT yet cover

- No real AstraCore EP (`AstraCoreExecutionProvider`) registration — ORT
  still falls back to `CPUExecutionProvider` because the provider
  library isn't built yet. This is the ~2-engineer-week deliverable
  that row 8 of the external-validation doc calls out.
- No HF Hub model-card upload. Once the EP lands, buyers can add an
  "Inference Provider: AstraCore" tag to their fine-tuned checkpoints.

## Notable latent fix shipped alongside the suite

`astracore/benchmark.py::_gen_input_for` used to emit random token IDs
in `[0, 30000)` for every integer input. That overflows BERT's
`token_type_embeddings` table (size 2) and ORT's Gather bounds check
throws on opset ≥ 17 (the newer Optimum exports). Fix: a name-aware
range helper `_int_range_for_input` that routes `attention_mask`,
`token_type_ids`, `segment_ids`, `*_mask` → `[0, 2)` and `position_ids`
→ `[0, 512)`. Regression test
`test_bert_large_astracore_benchmark_harness` (integration-marked) is
the guard.
