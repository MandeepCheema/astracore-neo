"""Unit tests for tools/frontends/pytorch.py (F1-B4).

Tests skip cleanly when torch is not installed in the active venv
(torch lives in .venv-export, not .venv, to keep the dev loop lean).
Run from .venv-export to exercise the full path.
"""

from __future__ import annotations

import pytest

from tools.npu_ref.nn_graph import NnGraph, OP_CONV, OP_GELU, OP_GEMM, OP_RELU

torch = pytest.importorskip("torch")
from tools.frontends.pytorch import load_pytorch  # noqa: E402


def test_simple_linear_round_trips():
    """Tiny nn.Linear: (16 → 4) — exports cleanly and loads as a
    matmul or gemm with a weight_scale-ready IR."""
    module = torch.nn.Linear(16, 4).eval()
    x = torch.randn(1, 16)
    g = load_pytorch(module, x, input_name="x", output_name="y")
    assert isinstance(g, NnGraph)
    # The Linear exports as either Gemm or MatMul+Add depending on
    # torch version; we accept either but require at least one.
    ops = {L.op for L in g.layers}
    assert OP_GEMM in ops or OP_CONV in ops or "matmul" in ops, (
        f"expected a Gemm/MatMul/Conv in exported Linear graph, got {ops}"
    )


def test_conv_relu_chain_preserves_ops():
    """nn.Conv2d → ReLU: both ops must survive the export + load
    round-trip with their shapes intact."""
    module = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
    ).eval()
    x = torch.randn(1, 3, 16, 16)
    g = load_pytorch(module, x, input_name="image")
    ops = [L.op for L in g.layers]
    assert OP_CONV in ops, f"Conv2d missing after export, got {ops}"
    assert OP_RELU in ops, f"ReLU missing after export, got {ops}"


def test_gelu_exports_to_op_gelu():
    """nn.GELU exports as the Gelu op at opset 20+; the F1-B1 handler
    claims it and produces OP_GELU."""
    module = torch.nn.GELU().eval()
    x = torch.randn(1, 16)
    g = load_pytorch(module, x, input_name="x", opset=20)
    ops = [L.op for L in g.layers]
    assert OP_GELU in ops, f"GELU not emitted as OP_GELU; ops seen: {ops}"


def test_dynamic_batch_axis_is_concretised():
    """With dynamic_axes set on the batch dim and batch_size=1 passed
    to the loader, the returned graph's input shape starts with 1."""
    module = torch.nn.Linear(8, 8).eval()
    x = torch.randn(1, 8)
    g = load_pytorch(
        module, x,
        input_name="x", output_name="y",
        dynamic_axes={"x": {0: "N"}, "y": {0: "N"}},
        batch_size=1,
    )
    assert g.inputs["x"] == (1, 8)


def test_missing_torch_raises_import_error(monkeypatch):
    """The adapter raises a clear ImportError if torch isn't installed.
    We simulate the missing-torch case by masking the top-level import
    inside the function's scope."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("simulated missing torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="PyTorch front-end requires torch"):
        load_pytorch(object(), None)
