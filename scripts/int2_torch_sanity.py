"""OPT-D phase 1 sanity — is INT2 fake-quant *at all survivable* on
trained yolov8n.pt weights, and what's the "distance to recovery"?

This is the cheapest signal on whether Option D (INT2 + 2:4) is
realistic before we commit QAT compute:

  1. Load yolov8n.pt (dense FP32 trained).
  2. Run one forward pass on bus.jpg → record class-score range.
  3. Monkey-patch every Conv2d to fake-quant weights to INT2 per-output
     channel before matmul. No QAT.
  4. Run the same forward pass → compare class scores + bbox.
  5. Also try INT4 for comparison.

A "best-case PTQ" for INT2. If even with per-channel best-case
calibration INT2 forward collapses to random, QAT will have a very
steep hill; we should pivot. If INT2 output is recognisably
correlated with dense (e.g. class argmax matches on ~50% of anchors),
QAT has something to work with.

Runs from .venv-export (has torch + ultralytics).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _fake_quant_per_channel(w: torch.Tensor, qmax: float) -> torch.Tensor:
    """Symmetric per-output-channel fake-quant. scale = max(|w|) / qmax
    per channel, clamp to [-qmax, qmax]. STE in backward via detach."""
    C_out = w.shape[0]
    flat = w.reshape(C_out, -1)
    max_abs = flat.abs().amax(dim=1).clamp_min(1e-12)
    scale = max_abs / qmax
    scale_reshaped = scale.view(C_out, *([1] * (w.dim() - 1)))
    q = torch.clamp(torch.round(w / scale_reshaped), -qmax, qmax)
    return q * scale_reshaped


class Int2ConvWrapper(nn.Module):
    """Wraps an nn.Conv2d so its weight is INT2 fake-quanted at each
    forward. Bias is unquanted (matches F1-C2 audit H1 recipe)."""
    def __init__(self, conv: nn.Conv2d, qmax: float):
        super().__init__()
        self.conv = conv
        self.qmax = qmax

    def forward(self, x):
        w_q = _fake_quant_per_channel(self.conv.weight, self.qmax)
        return F.conv2d(x, w_q, self.conv.bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation,
                        self.conv.groups)


def wrap_model_conv_weights(model: nn.Module, qmax: float) -> int:
    """Replace every nn.Conv2d inside `model` with Int2ConvWrapper.
    Collect targets first to avoid mutating the tree while walking it
    (the wrapper itself *contains* a Conv2d, which would otherwise
    recurse infinitely when the iterator revisits it)."""
    to_swap = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if type(child) is nn.Conv2d:   # strict: skip already-wrapped
                to_swap.append((module, child_name, child))
    for parent, child_name, child in to_swap:
        setattr(parent, child_name, Int2ConvWrapper(child, qmax))
    return len(to_swap)


def summarise_output(y, label: str):
    """y can be a tuple (training-mode) or tensor (inference)."""
    if isinstance(y, (tuple, list)):
        y = y[0]
    print(f"  {label:20s} shape={tuple(y.shape)}  "
          f"range=[{y.min().item():8.3f}, {y.max().item():8.3f}]  "
          f"mean={y.mean().item():7.3f}  std={y.std().item():7.3f}")


def main() -> int:
    from ultralytics import YOLO

    bus_npz = REPO / "data" / "calibration" / "bus.npz"
    pt_path = REPO / "data" / "models" / "yolov8n.pt"
    if not bus_npz.exists() or not pt_path.exists():
        print("missing artefacts", file=sys.stderr)
        return 2

    x_np = np.load(bus_npz)["image"]  # (1, 3, 640, 640) fp32
    x = torch.from_numpy(x_np)

    print("=== dense FP32 ===")
    model = YOLO(str(pt_path)).model
    model.eval()
    with torch.no_grad():
        y_dense = model(x)
    summarise_output(y_dense, "dense FP32")

    for label, qmax in (("INT8 (qmax=127)", 127.0),
                         ("INT4 (qmax=7)",   7.0),
                         ("INT2 (qmax=1)",   1.0)):
        print(f"\n=== per-channel fake-quant {label} ===")
        model2 = YOLO(str(pt_path)).model
        model2.eval()
        n = wrap_model_conv_weights(model2, qmax)
        with torch.no_grad():
            y_q = model2(x)
        print(f"  wrapped {n} Conv2d layers")
        summarise_output(y_q, label)

        # Cosine & SNR vs dense on the output tensor.
        if isinstance(y_q, (tuple, list)):
            y_q_t = y_q[0]
        else:
            y_q_t = y_q
        if isinstance(y_dense, (tuple, list)):
            y_d_t = y_dense[0]
        else:
            y_d_t = y_dense
        a = y_d_t.flatten().double()
        b = y_q_t.flatten().double()
        cos = float((a @ b) / (a.norm() * b.norm() + 1e-30))
        err = a - b
        snr = 10.0 * float(torch.log10(
            (a ** 2).mean() / (err ** 2).mean().clamp_min(1e-30)
        ))
        print(f"  cosine vs dense: {cos:.6f}   SNR: {snr:.2f} dB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
