# MLIR / IREE / TVM conformance — AstraCore

- Generated: `2026-04-21T10:01:29Z`
- Suite size: **15** canonical ops
- Adapters: `mlir-stablehlo`, `tvm-relay`, `jax-xla`
- Result: **45 / 45 pass**

| case | group | mlir-stablehlo | tvm-relay | jax-xla |
|---|---|---|---|---|
| add | arith | PASS | PASS | PASS |
| mul | arith | PASS | PASS | PASS |
| matmul | linalg | PASS | PASS | PASS |
| gemm | linalg | PASS | PASS | PASS |
| conv | linalg | PASS | PASS | PASS |
| relu | act | PASS | PASS | PASS |
| sigmoid | act | PASS | PASS | PASS |
| softmax | act | PASS | PASS | PASS |
| gelu | act | PASS | PASS | PASS |
| layernorm | norm | PASS | PASS | PASS |
| reshape | shape | PASS | PASS | PASS |
| transpose | shape | PASS | PASS | PASS |
| concat | shape | PASS | PASS | PASS |
| maxpool | pool | PASS | PASS | PASS |
| globalavgpool | pool | PASS | PASS | PASS |
