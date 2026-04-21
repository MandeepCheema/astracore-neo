# Deep test 3 — GPT-2 ' Paris' rank consistency

Canned prompt 'The capital of France is'; BPE token 6342 corresponds to ' Paris'. After LM-head projection, Paris should be ranked near the top and stable across runs.

- Ranks observed across 10 runs: [5]
- Paris probability (%):         [3.22, 3.22, 3.22, 3.22, 3.22, 3.22, 3.22, 3.22, 3.22, 3.22]
- Deterministic?                 **YES**
- Inside top-10?                 **YES**
