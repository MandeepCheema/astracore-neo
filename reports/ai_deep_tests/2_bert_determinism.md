# Deep test 2 — BERT-Squad answer-span determinism

Same canned Q+A input, 10 runs. Start/end token indices and start-logit score must be bit-identical across runs.

- Start tokens observed: [9]
- End tokens observed:   [9]
- Start-score stdev:     0.0000
- Deterministic?         **YES**
