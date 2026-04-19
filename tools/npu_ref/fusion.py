"""Graph-level fusion passes (F1-C1c).

Each fusion is a mutate-in-place NnGraph rewrite:

    fuse_silu(graph)   — Sigmoid + Mul → SiLU, when the Mul consumes
                         both the Sigmoid's output and the Sigmoid's
                         input. Matches the pattern emitted by
                         Ultralytics YOLOv8 (58 instances) and other
                         SiLU-heavy CNNs.

    fuse_all(graph)    — run every fusion pass we have. Adding new
                         fusions (Conv+ReLU, Add+ReLU, etc.) means
                         writing a `fuse_*` function and listing it
                         in `fuse_all`.

Why fusion lives outside the loader: the loader is a faithful
ONNX-to-IR translator. Fusion is a structural rewrite that only makes
sense once the full graph is in memory; doing it inline would couple
two concerns and make both harder to test. The downstream tiler
(F1-C3/C4) will read OP_SILU and emit a single RTL tile with
`cfg_afu_mode = MODE_SILU` — two operations turn into one.

Pre/post-conditions:
  - Input graph must be topologically ordered (NnGraph invariant).
  - Output graph stays topologically ordered.
  - Tensor names are preserved or safely rerouted; no layer ends up
    referencing a dangling tensor name.
  - Learned weights on unaffected layers are untouched.
  - QuantParams are untouched — fusion is expected to run before
    F1-C2 quantisation; calling it after is legal but has no effect
    on the fused SiLU layer's (still empty) quant field.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .nn_graph import NnGraph, NnLayer, OP_MUL, OP_SIGMOID, OP_SILU


# ---------------------------------------------------------------------------
# SiLU
# ---------------------------------------------------------------------------
def _build_consumer_map(layers: List[NnLayer]) -> Dict[str, List[int]]:
    """For each tensor name, list the indices of layers that consume
    it as an input. Used to decide whether a Sigmoid's output flows to
    exactly one place (the Mul)."""
    m: Dict[str, List[int]] = defaultdict(list)
    for i, L in enumerate(layers):
        for name in L.inputs:
            m[name].append(i)
    return m


def _build_producer_map(layers: List[NnLayer]) -> Dict[str, int]:
    m: Dict[str, int] = {}
    for i, L in enumerate(layers):
        for name in L.outputs:
            m[name] = i
    return m


def fuse_silu(graph: NnGraph) -> NnGraph:
    """Replace Sigmoid+Mul SiLU patterns with a single OP_SILU layer.

    A Sigmoid layer is fused iff:
      1. Its output has exactly one consumer.
      2. That consumer is a Mul with exactly two inputs.
      3. The Mul's other input is the same tensor fed into the Sigmoid
         (i.e. `x` feeds both sides — `sigmoid(x) * x`).

    When those hold, the Sigmoid+Mul pair is replaced by a single
    SiLU layer that consumes `x` and produces the Mul's output.

    Records the fusion count at `graph.metadata["silu_fusions"]` so
    downstream reporting can compare against the ONNX Sigmoid count.
    """
    consumers = _build_consumer_map(graph.layers)

    fused_mul_idx: set[int] = set()
    new_layers: List[NnLayer] = list(graph.layers)

    for i, layer in enumerate(graph.layers):
        if layer.op != OP_SIGMOID:
            continue
        if i in fused_mul_idx:
            continue
        if not layer.inputs or not layer.outputs:
            continue
        sig_input = layer.inputs[0]
        sig_output = layer.outputs[0]

        cons = consumers.get(sig_output, [])
        if len(cons) != 1:
            continue
        mul_idx = cons[0]
        if mul_idx == i:
            continue
        mul = graph.layers[mul_idx]
        if mul.op != OP_MUL or len(mul.inputs) != 2:
            continue

        # Find the "other" input of the Mul (the one that isn't the
        # Sigmoid's output) and check it's sig_input.
        other = mul.inputs[0] if mul.inputs[1] == sig_output else mul.inputs[1]
        if other != sig_input:
            continue

        silu = NnLayer(
            name=(layer.name + "+" + mul.name) if layer.name and mul.name
                 else f"silu_{mul.outputs[0]}",
            op=OP_SILU,
            inputs=[sig_input],
            outputs=list(mul.outputs),
            in_shapes={sig_input: layer.in_shapes[sig_input]}
                       if sig_input in layer.in_shapes else {},
            out_shapes=dict(mul.out_shapes),
            attrs={},
        )
        new_layers[i] = silu
        fused_mul_idx.add(mul_idx)

    if fused_mul_idx:
        graph.layers = [L for idx, L in enumerate(new_layers)
                        if idx not in fused_mul_idx]

    graph.metadata["silu_fusions"] = len(fused_mul_idx)
    return graph


def fuse_all(graph: NnGraph) -> NnGraph:
    """Run every available fusion pass. Extend this when new
    fusions land."""
    fuse_silu(graph)
    return graph
