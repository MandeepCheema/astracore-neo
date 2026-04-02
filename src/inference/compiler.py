"""
AstraCore Neo Inference — Graph Compiler.

Simulates the chip's AI-driven compiler toolchain:
  - Computation graph (DAG of typed operator nodes)
  - Operator fusion: conv+relu, matmul+add, layernorm+gelu, etc.
  - Auto-tiling: splits large ops across MAC cores and SRAM banks
  - Topological scheduling with dependency tracking
  - Target precision selection (INT4/INT8/FP8/FP16/FP32)
  - Outputs a CompiledModel ready for the runtime

Chip spec: "ONNX 2.0, PyTorch, TensorRT, TVM, XLA, MLIR, NNEF,
            AI-driven scheduling"

In simulation we model the compiler's logical pipeline — graph
parsing, fusion, tiling, scheduling — without actually JIT-compiling
to machine code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .exceptions import CompilerError, FusionError, TilingError

# ---------------------------------------------------------------------------
# Operator types
# ---------------------------------------------------------------------------

class OpType(Enum):
    # Compute
    MATMUL    = "matmul"
    CONV2D    = "conv2d"
    ELEMWISE  = "elemwise"
    # Activation
    RELU      = "relu"
    GELU      = "gelu"
    SIGMOID   = "sigmoid"
    TANH      = "tanh"
    # Normalisation
    LAYERNORM = "layernorm"
    BATCHNORM = "batchnorm"
    SOFTMAX   = "softmax"
    # Reshape
    RESHAPE   = "reshape"
    TRANSPOSE = "transpose"
    CONCAT    = "concat"
    SPLIT     = "split"
    # Pooling
    MAXPOOL   = "maxpool"
    AVGPOOL   = "avgpool"
    # Attention
    ATTENTION = "attention"
    # Memory
    LOAD      = "load"
    STORE     = "store"
    # Fused (produced by compiler)
    FUSED_CONV_RELU     = "fused_conv_relu"
    FUSED_MATMUL_ADD    = "fused_matmul_add"
    FUSED_LAYERNORM_GELU = "fused_layernorm_gelu"
    FUSED_ATTENTION_SOFTMAX = "fused_attention_softmax"


# Fusion rules: (op_a, op_b) → fused_op
_FUSION_RULES: Dict[Tuple[OpType, OpType], OpType] = {
    (OpType.CONV2D,    OpType.RELU):      OpType.FUSED_CONV_RELU,
    (OpType.MATMUL,    OpType.ELEMWISE):  OpType.FUSED_MATMUL_ADD,
    (OpType.LAYERNORM, OpType.GELU):      OpType.FUSED_LAYERNORM_GELU,
    (OpType.ATTENTION, OpType.SOFTMAX):   OpType.FUSED_ATTENTION_SOFTMAX,
}


# ---------------------------------------------------------------------------
# Compiler target precision
# ---------------------------------------------------------------------------

class CompilerTarget(Enum):
    INT4  = "int4"
    INT8  = "int8"
    FP8   = "fp8"
    FP16  = "fp16"
    FP32  = "fp32"


# ---------------------------------------------------------------------------
# Graph primitives
# ---------------------------------------------------------------------------

@dataclass
class TensorShape:
    dims: Tuple[int, ...]

    def numel(self) -> int:
        result = 1
        for d in self.dims:
            result *= d
        return result

    def __str__(self) -> str:
        return f"({', '.join(str(d) for d in self.dims)})"


@dataclass
class GraphNode:
    """One operator node in the computation graph."""
    node_id:   str
    op_type:   OpType
    inputs:    List[str]          # names of input tensors / preceding node ids
    outputs:   List[str]          # names of output tensors
    attrs:     Dict[str, Any] = field(default_factory=dict)
    shape_in:  Optional[TensorShape] = None
    shape_out: Optional[TensorShape] = None
    fused_from: List[str] = field(default_factory=list)  # source node ids if fused
    tiled:     bool = False
    tile_size: Optional[int] = None

    def is_fused(self) -> bool:
        return bool(self.fused_from)


@dataclass
class CompiledModel:
    """Output of the compiler — ready for the runtime to execute."""
    name:           str
    target:         CompilerTarget
    schedule:       List[GraphNode]     # topologically sorted, fused, tiled
    input_names:    List[str]
    output_names:   List[str]
    # Compilation stats
    original_nodes: int = 0
    fused_nodes:    int = 0
    tiled_nodes:    int = 0
    estimated_tops: float = 0.0
    memory_bytes:   int = 0

    @property
    def node_count(self) -> int:
        return len(self.schedule)

    @property
    def fusion_savings(self) -> int:
        return self.original_nodes - self.node_count


# ---------------------------------------------------------------------------
# Tiling config
# ---------------------------------------------------------------------------

# Max tile size (elements) fitting in one SRAM bank's scratchpad
# 8MB bank, FP32 = 4B/elem → 2M elems; use 256K as safe tile for L1
DEFAULT_TILE_SIZE = 256 * 1024


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

class AstraCoreCompiler:
    """
    AI-driven graph compiler for the AstraCore Neo.

    Pipeline:
      parse → validate → fuse → tile → schedule → emit CompiledModel

    Usage::

        compiler = AstraCoreCompiler()
        graph    = compiler.parse(node_dicts)
        model    = compiler.compile(graph, target=CompilerTarget.INT8)
        print(model.fusion_savings)   # nodes eliminated by fusion
    """

    def __init__(
        self,
        tile_size:  int = DEFAULT_TILE_SIZE,
        enable_fusion: bool = True,
        enable_tiling: bool = True,
    ) -> None:
        self.tile_size     = tile_size
        self.enable_fusion = enable_fusion
        self.enable_tiling = enable_tiling
        self.compile_count = 0

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    def parse(self, node_defs: List[Dict[str, Any]]) -> List[GraphNode]:
        """
        Parse a list of node definition dicts into GraphNode objects.

        Each dict must have:
          - "id":      str
          - "op":      str  (matches OpType value)
          - "inputs":  List[str]
          - "outputs": List[str]
          - "attrs":   Dict  (optional)
          - "shape_in":  tuple  (optional)
          - "shape_out": tuple  (optional)
        """
        if not node_defs:
            raise CompilerError("Empty graph — nothing to compile")
        nodes: List[GraphNode] = []
        seen_ids: Set[str] = set()
        for d in node_defs:
            nid = d.get("id")
            op_str = d.get("op")
            if not nid:
                raise CompilerError("Node missing 'id' field")
            if nid in seen_ids:
                raise CompilerError(f"Duplicate node id: {nid!r}")
            try:
                op = OpType(op_str)
            except ValueError:
                raise CompilerError(f"Unknown op type: {op_str!r}")
            si = d.get("shape_in")
            so = d.get("shape_out")
            nodes.append(GraphNode(
                node_id=nid,
                op_type=op,
                inputs=d.get("inputs", []),
                outputs=d.get("outputs", []),
                attrs=d.get("attrs", {}),
                shape_in=TensorShape(tuple(si)) if si else None,
                shape_out=TensorShape(tuple(so)) if so else None,
            ))
            seen_ids.add(nid)
        return nodes

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def compile(
        self,
        nodes: List[GraphNode],
        name:  str = "model",
        target: CompilerTarget = CompilerTarget.INT8,
        input_names:  Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ) -> CompiledModel:
        """
        Full compilation pipeline: validate → fuse → tile → schedule.

        Returns a CompiledModel ready for InferenceRuntime.
        """
        if not nodes:
            raise CompilerError("Cannot compile an empty node list")

        original_count = len(nodes)

        # 1. Validate DAG (no cycles via simple DFS)
        self._validate_dag(nodes)

        # 2. Operator fusion
        if self.enable_fusion:
            nodes = self._fuse(nodes)

        fused_count = original_count - len(nodes)

        # 3. Auto-tiling
        tiled_count = 0
        if self.enable_tiling:
            for node in nodes:
                if self._needs_tiling(node):
                    node.tiled     = True
                    node.tile_size = self.tile_size
                    tiled_count   += 1

        # 4. Topological schedule
        schedule = self._topological_sort(nodes)

        # 5. Estimate TOPS and memory
        est_tops = self._estimate_tops(schedule, target)
        mem      = self._estimate_memory(schedule, target)

        self.compile_count += 1

        return CompiledModel(
            name=name,
            target=target,
            schedule=schedule,
            input_names=input_names or [],
            output_names=output_names or [],
            original_nodes=original_count,
            fused_nodes=fused_count,
            tiled_nodes=tiled_count,
            estimated_tops=est_tops,
            memory_bytes=mem,
        )

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def _fuse(self, nodes: List[GraphNode]) -> List[GraphNode]:
        """Single-pass adjacent pair fusion."""
        if len(nodes) < 2:
            return nodes
        result: List[GraphNode] = []
        i = 0
        while i < len(nodes):
            if i + 1 < len(nodes):
                a, b = nodes[i], nodes[i + 1]
                fused_op = _FUSION_RULES.get((a.op_type, b.op_type))
                if fused_op and self._can_fuse(a, b):
                    fused = GraphNode(
                        node_id=f"{a.node_id}+{b.node_id}",
                        op_type=fused_op,
                        inputs=a.inputs,
                        outputs=b.outputs,
                        attrs={**a.attrs, **b.attrs},
                        shape_in=a.shape_in,
                        shape_out=b.shape_out,
                        fused_from=[a.node_id, b.node_id],
                    )
                    result.append(fused)
                    i += 2
                    continue
            result.append(nodes[i])
            i += 1
        return result

    def _can_fuse(self, a: GraphNode, b: GraphNode) -> bool:
        """Nodes can fuse if b's inputs are a's outputs (direct chain)."""
        return bool(set(a.outputs) & set(b.inputs))

    # ------------------------------------------------------------------
    # Tiling
    # ------------------------------------------------------------------

    def _needs_tiling(self, node: GraphNode) -> bool:
        """Tile if output tensor exceeds tile_size elements."""
        if node.shape_out is None:
            return False
        return node.shape_out.numel() > self.tile_size

    # ------------------------------------------------------------------
    # Topological sort (Kahn's algorithm)
    # ------------------------------------------------------------------

    def _topological_sort(self, nodes: List[GraphNode]) -> List[GraphNode]:
        id_to_node = {n.node_id: n for n in nodes}
        # Build in-degree map
        in_degree: Dict[str, int] = {n.node_id: 0 for n in nodes}
        dependents: Dict[str, List[str]] = {n.node_id: [] for n in nodes}

        for node in nodes:
            for inp in node.inputs:
                if inp in id_to_node:
                    in_degree[node.node_id] += 1
                    dependents[inp].append(node.node_id)

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: List[GraphNode] = []

        while queue:
            nid = queue.pop(0)
            order.append(id_to_node[nid])
            for dep in dependents[nid]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        if len(order) != len(nodes):
            raise CompilerError("Cycle detected in computation graph")
        return order

    # ------------------------------------------------------------------
    # DAG validation
    # ------------------------------------------------------------------

    def _validate_dag(self, nodes: List[GraphNode]) -> None:
        """Detect cycles using DFS coloring."""
        WHITE, GRAY, BLACK = 0, 1, 2
        id_to_node = {n.node_id: n for n in nodes}
        color: Dict[str, int] = {n.node_id: WHITE for n in nodes}

        def dfs(nid: str) -> None:
            color[nid] = GRAY
            node = id_to_node[nid]
            for inp in node.inputs:
                if inp not in color:
                    continue
                if color[inp] == GRAY:
                    raise CompilerError(f"Cycle detected involving node {inp!r}")
                if color[inp] == WHITE:
                    dfs(inp)
            color[nid] = BLACK

        for n in nodes:
            if color[n.node_id] == WHITE:
                dfs(n.node_id)

    # ------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------

    _TOPS_PER_OP = {
        OpType.MATMUL: 1.0, OpType.CONV2D: 1.0, OpType.ATTENTION: 1.5,
        OpType.FUSED_CONV_RELU: 1.2, OpType.FUSED_MATMUL_ADD: 1.1,
        OpType.FUSED_ATTENTION_SOFTMAX: 1.6, OpType.FUSED_LAYERNORM_GELU: 0.3,
    }
    _PRECISION_MUL = {
        CompilerTarget.INT4: 2.0, CompilerTarget.INT8: 1.0,
        CompilerTarget.FP8: 1.0,  CompilerTarget.FP16: 0.5,
        CompilerTarget.FP32: 0.25,
    }
    _BYTES_PER_ELEM = {
        CompilerTarget.INT4: 0.5, CompilerTarget.INT8: 1,
        CompilerTarget.FP8: 1,    CompilerTarget.FP16: 2,
        CompilerTarget.FP32: 4,
    }

    def _estimate_tops(self, nodes: List[GraphNode], target: CompilerTarget) -> float:
        base = sum(self._TOPS_PER_OP.get(n.op_type, 0.1) for n in nodes)
        return base * self._PRECISION_MUL[target]

    def _estimate_memory(self, nodes: List[GraphNode], target: CompilerTarget) -> int:
        bpe = self._BYTES_PER_ELEM[target]
        total = 0
        for n in nodes:
            if n.shape_out:
                total += int(n.shape_out.numel() * bpe)
        return total
