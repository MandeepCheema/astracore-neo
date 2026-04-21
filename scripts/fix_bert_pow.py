"""Replace BERT's GELU-related Pow(x, N) with repeated Mul for QNN compatibility.

QNN HTP's TFLite delegate rejects Pow ops with small integer exponents
(seen on BERT-Squad node 667 on both QCS8550 and Samsung S24). The
offending Pow lives inside BERT's tanh-approximation GELU:

    0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

Replacing ``Pow(x, 3.0)`` with ``Mul(Mul(x, x), x)`` gives
numerically-equivalent output and uses only QNN-supported ops.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper


def rewrite_pow_as_mul(model: onnx.ModelProto,
                       max_exponent: int = 4) -> int:
    """Replace Pow(x, k) with k-1 Mul ops for integer-valued k ≤ max_exponent.

    Returns number of replacements made.
    """
    # Map initializer name -> numpy value (for constant-exponent detection)
    inits = {init.name: numpy_helper.to_array(init)
             for init in model.graph.initializer}

    new_nodes = []
    replaced = 0
    for node in model.graph.node:
        if node.op_type != "Pow" or len(node.input) != 2:
            new_nodes.append(node)
            continue
        # The exponent must be an initializer with integer-valued scalar.
        exp_name = node.input[1]
        if exp_name not in inits:
            new_nodes.append(node)
            continue
        arr = inits[exp_name]
        if arr.size != 1:
            new_nodes.append(node)
            continue
        exp = float(arr.flatten()[0])
        if not exp.is_integer() or not (2 <= int(exp) <= max_exponent):
            new_nodes.append(node)
            continue

        k = int(exp)
        x = node.input[0]
        out = node.output[0]
        # Chain: tmp_1 = x * x ; tmp_2 = tmp_1 * x ; ... ; out = tmp_{k-1} * x
        prev = x
        for i in range(k - 1):
            is_last = (i == k - 2)
            step_out = out if is_last else f"{out}__pow_step_{i}"
            new_nodes.append(helper.make_node(
                "Mul", [prev, x], [step_out],
                name=f"{node.name or out}__mul_{i}",
            ))
            prev = step_out
        replaced += 1

    if replaced:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
    return replaced


def main(argv):
    src = Path(argv[1])
    dst = Path(argv[2])
    m = onnx.load(str(src))
    n = rewrite_pow_as_mul(m)
    onnx.save(m, str(dst))
    print(f"{src.name} -> {dst.name}   replaced {n} Pow -> Mul chains")


if __name__ == "__main__":
    main(sys.argv)
