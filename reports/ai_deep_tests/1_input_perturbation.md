# Deep test 1 — Input perturbation robustness

Gaussian pixel noise σ ∈ {0, 0.01, 0.05, 0.1, 0.2} (σ expressed as fraction of 255). Bus image, ImageNet classifiers. Top-1 prediction monitored per σ.

| Model | σ=0 top1 | σ=0.01 | σ=0.05 | σ=0.1 | σ=0.2 | Survives up to |
|---|---:|---:|---:|---:|---:|---:|
| squeezenet-1.1 | 8.5% | 10.9% ✗ | 12.2% ✗ | 28.6% ✗ | 13.0% ✗ | σ=0.0 |
| mobilenetv2-7 | 77.0% | 84.3% ✓ | 96.5% ✓ | 48.4% ✓ | 16.9% ✗ | σ=0.1 |
| shufflenet-v2-10 | 43.9% | 37.1% ✗ | 81.2% ✓ | 55.7% ✗ | 95.5% ✗ | σ=0.05 |

✓ = top-1 label unchanged from noise-free baseline. Higher `Survives up to σ` = more robust model.