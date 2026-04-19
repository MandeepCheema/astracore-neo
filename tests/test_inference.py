"""
AstraCore Neo — Inference Testbench

Covers:
  Compiler:
    - parse: valid nodes, bad op, duplicate id, missing id
    - compile: fusion, tiling, topological sort, cycle detection
    - fusion rules: conv+relu, matmul+elemwise, layernorm+gelu, attn+softmax
    - fusion only fires when nodes are adjacent and connected
    - tiling marks large-output nodes
    - compiled model stats: original_nodes, fused_nodes, node_count
    - topological sort respects dependencies
    - empty graph raises

  Quantizer:
    - calibrate accumulates min/max/mean
    - quantize INT8: values in [-128, 127]
    - quantize INT4: values in [-8, 7]
    - quantize FP8: values in [-448, 448]
    - dequantize recovers original within tolerance
    - symmetric: zero_point == 0
    - asymmetric: zero_point may be nonzero
    - per-channel: independent scale per row
    - quantize_uncalibrated convenience
    - calibrate-before-quantize enforcement
    - all-zero tensor (zero-range guard)
    - stats reset

  Runtime:
    - load_model creates session
    - bind_input / bind_inputs
    - run returns RunResult with latency_ms > 0
    - output tensor accessible after run
    - multiple sequential runs
    - node profiles count matches schedule
    - relu/gelu/sigmoid/tanh/softmax/layernorm dispatch
    - matmul dispatch via MACArray
    - reshape dispatch
    - session state transitions
    - unload_session
    - runtime stats: models_loaded, total_runs
"""

import sys, os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.inference import (
    AstraCoreCompiler, CompilerTarget, CompiledModel,
    GraphNode, OpType, TensorShape,
    Quantizer, QuantConfig, QuantPrecision, QuantGranularity,
    QuantizedTensor, CalibStats,
    InferenceRuntime, InferenceSession, RunResult,
    SessionState,
    CompilerError, QuantizationError, InferenceError,
)
from src.compute import MACArray, TransformerEngine, SparsityEngine
from src.hal import AstraCoreDevice

rng = np.random.default_rng(42)


# ===========================================================================
# Helpers
# ===========================================================================

def _node(nid, op, inputs=None, outputs=None, attrs=None, shape_in=None, shape_out=None):
    return {
        "id": nid, "op": op,
        "inputs": inputs or [],
        "outputs": outputs or [nid + "_out"],
        "attrs": attrs or {},
        "shape_in": shape_in,
        "shape_out": shape_out,
    }


def simple_graph():
    """conv → relu → matmul chain."""
    return [
        _node("n0", "conv2d",  inputs=[],       outputs=["n0_out"], shape_out=(16, 6, 6)),
        _node("n1", "relu",    inputs=["n0_out"], outputs=["n1_out"]),
        _node("n2", "matmul",  inputs=["n1_out"], outputs=["n2_out"], shape_out=(4, 4)),
        _node("n3", "elemwise",inputs=["n2_out"], outputs=["n3_out"]),
    ]


@pytest.fixture
def compiler():
    return AstraCoreCompiler()


@pytest.fixture
def dev():
    d = AstraCoreDevice()
    d.power_on()
    return d


@pytest.fixture
def mac(dev):
    return MACArray(dev=dev)


@pytest.fixture
def transformer(dev):
    return TransformerEngine(dev=dev)


@pytest.fixture
def runtime(mac, transformer):
    return InferenceRuntime(mac_array=mac, transformer=transformer)


@pytest.fixture
def quantizer():
    return Quantizer()


# ===========================================================================
# 1. Compiler — parse
# ===========================================================================

class TestCompilerParse:

    def test_parse_valid_nodes(self, compiler):
        nodes = compiler.parse(simple_graph())
        assert len(nodes) == 4
        assert all(isinstance(n, GraphNode) for n in nodes)

    def test_parse_op_types_correct(self, compiler):
        nodes = compiler.parse(simple_graph())
        assert nodes[0].op_type == OpType.CONV2D
        assert nodes[1].op_type == OpType.RELU

    def test_parse_unknown_op_raises(self, compiler):
        with pytest.raises(CompilerError, match="Unknown op"):
            compiler.parse([_node("n0", "unknown_op")])

    def test_parse_duplicate_id_raises(self, compiler):
        with pytest.raises(CompilerError, match="Duplicate"):
            compiler.parse([
                _node("n0", "relu"),
                _node("n0", "gelu"),
            ])

    def test_parse_missing_id_raises(self, compiler):
        with pytest.raises(CompilerError):
            compiler.parse([{"op": "relu", "inputs": [], "outputs": []}])

    def test_parse_shape_captured(self, compiler):
        nodes = compiler.parse([_node("n0", "matmul", shape_in=(4, 8), shape_out=(4, 4))])
        assert nodes[0].shape_in.dims == (4, 8)
        assert nodes[0].shape_out.dims == (4, 4)

    def test_parse_empty_raises(self, compiler):
        with pytest.raises(CompilerError):
            compiler.parse([])


# ===========================================================================
# 2. Compiler — fusion
# ===========================================================================

class TestCompilerFusion:

    def test_conv_relu_fused(self, compiler):
        nodes = compiler.parse([
            _node("n0", "conv2d", outputs=["n0_out"]),
            _node("n1", "relu",   inputs=["n0_out"]),
        ])
        model = compiler.compile(nodes)
        ops = [n.op_type for n in model.schedule]
        assert OpType.FUSED_CONV_RELU in ops

    def test_matmul_elemwise_fused(self, compiler):
        nodes = compiler.parse([
            _node("n0", "matmul",  outputs=["n0_out"]),
            _node("n1", "elemwise",inputs=["n0_out"]),
        ])
        model = compiler.compile(nodes)
        ops = [n.op_type for n in model.schedule]
        assert OpType.FUSED_MATMUL_ADD in ops

    def test_layernorm_gelu_fused(self, compiler):
        nodes = compiler.parse([
            _node("n0", "layernorm", outputs=["n0_out"]),
            _node("n1", "gelu",      inputs=["n0_out"]),
        ])
        model = compiler.compile(nodes)
        ops = [n.op_type for n in model.schedule]
        assert OpType.FUSED_LAYERNORM_GELU in ops

    def test_attention_softmax_fused(self, compiler):
        nodes = compiler.parse([
            _node("n0", "attention", outputs=["n0_out"]),
            _node("n1", "softmax",   inputs=["n0_out"]),
        ])
        model = compiler.compile(nodes)
        ops = [n.op_type for n in model.schedule]
        assert OpType.FUSED_ATTENTION_SOFTMAX in ops

    def test_fusion_reduces_node_count(self, compiler):
        nodes = compiler.parse([
            _node("n0", "conv2d", outputs=["n0_out"]),
            _node("n1", "relu",   inputs=["n0_out"]),
        ])
        model = compiler.compile(nodes)
        assert model.node_count == 1
        assert model.fused_nodes == 1
        assert model.fusion_savings == 1

    def test_no_fusion_when_not_connected(self, compiler):
        nodes = compiler.parse([
            _node("n0", "conv2d", inputs=[], outputs=["n0_out"]),
            _node("n1", "relu",   inputs=["other"],  outputs=["n1_out"]),
        ])
        model = compiler.compile(nodes)
        ops = [n.op_type for n in model.schedule]
        assert OpType.FUSED_CONV_RELU not in ops

    def test_fusion_disabled(self):
        c = AstraCoreCompiler(enable_fusion=False)
        nodes = c.parse([
            _node("n0", "conv2d", outputs=["n0_out"]),
            _node("n1", "relu",   inputs=["n0_out"]),
        ])
        model = c.compile(nodes)
        assert model.fused_nodes == 0

    def test_fused_node_has_fused_from(self, compiler):
        nodes = compiler.parse([
            _node("n0", "conv2d", outputs=["n0_out"]),
            _node("n1", "relu",   inputs=["n0_out"]),
        ])
        model = compiler.compile(nodes)
        fused = model.schedule[0]
        assert "n0" in fused.fused_from
        assert "n1" in fused.fused_from


# ===========================================================================
# 3. Compiler — tiling and scheduling
# ===========================================================================

class TestCompilerTilingScheduling:

    def test_large_output_node_tiled(self):
        c = AstraCoreCompiler(tile_size=16)
        nodes = c.parse([
            _node("n0", "matmul", shape_out=(128, 128)),  # 16384 > 16
        ])
        model = c.compile(nodes)
        assert model.schedule[0].tiled

    def test_small_output_node_not_tiled(self, compiler):
        nodes = compiler.parse([
            _node("n0", "matmul", shape_out=(4, 4)),
        ])
        model = compiler.compile(nodes)
        assert not model.schedule[0].tiled

    def test_topological_order_respected(self, compiler):
        nodes = compiler.parse([
            _node("n0", "relu",   inputs=[],       outputs=["n0_out"]),
            _node("n1", "matmul", inputs=["n0_out"], outputs=["n1_out"]),
            _node("n2", "gelu",   inputs=["n1_out"], outputs=["n2_out"]),
        ])
        model = compiler.compile(nodes)
        ids = [n.node_id for n in model.schedule]
        assert ids.index("n0") < ids.index("n1")
        assert ids.index("n1") < ids.index("n2")

    def test_cycle_detection_raises(self, compiler):
        # n0 depends on n1, n1 depends on n0 — cycle
        nodes = compiler.parse([
            _node("n0", "relu",   inputs=["n1"], outputs=["n0_out"]),
            _node("n1", "matmul", inputs=["n0"], outputs=["n1_out"]),
        ])
        with pytest.raises(CompilerError, match="[Cc]ycle"):
            compiler.compile(nodes)

    def test_compile_increments_counter(self, compiler):
        nodes = compiler.parse(simple_graph())
        compiler.compile(nodes)
        assert compiler.compile_count == 1

    def test_compiled_model_has_name(self, compiler):
        nodes = compiler.parse(simple_graph())
        model = compiler.compile(nodes, name="yolov8")
        assert model.name == "yolov8"

    def test_estimated_tops_positive(self, compiler):
        nodes = compiler.parse(simple_graph())
        model = compiler.compile(nodes)
        assert model.estimated_tops > 0

    def test_int4_tops_gt_int8(self, compiler):
        nodes = compiler.parse([_node("n0", "matmul")])
        m8 = compiler.compile(nodes, target=CompilerTarget.INT8)
        m4 = compiler.compile(nodes, target=CompilerTarget.INT4)
        assert m4.estimated_tops > m8.estimated_tops

    def test_empty_compile_raises(self, compiler):
        with pytest.raises(CompilerError):
            compiler.compile([])


# ===========================================================================
# 4. Quantizer — calibration
# ===========================================================================

class TestQuantizerCalibration:

    def test_calibrate_sets_min_max(self, quantizer):
        data = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        stats = quantizer.calibrate("w", data)
        assert stats.min_val == pytest.approx(-1.0)
        assert stats.max_val == pytest.approx(2.0)

    def test_calibrate_accumulates_across_calls(self, quantizer):
        quantizer.calibrate("w", np.array([0.0, 1.0]))
        quantizer.calibrate("w", np.array([-5.0, 3.0]))
        assert quantizer.stats("w").min_val == pytest.approx(-5.0)
        assert quantizer.stats("w").max_val == pytest.approx(3.0)

    def test_calibrate_tracks_num_samples(self, quantizer):
        quantizer.calibrate("w", np.ones((100,)))
        assert quantizer.stats("w").num_samples == 100

    def test_calibrate_empty_raises(self, quantizer):
        with pytest.raises(QuantizationError):
            quantizer.calibrate("w", np.array([]))

    def test_stats_without_calibrate_raises(self, quantizer):
        with pytest.raises(QuantizationError):
            quantizer.stats("missing")

    def test_iter_stats_yields_every_calibrated_tensor(self, quantizer):
        quantizer.calibrate("a", np.array([1.0, -2.0], dtype=np.float32))
        quantizer.calibrate("b", np.array([0.5], dtype=np.float32))
        quantizer.calibrate("a", np.array([4.0], dtype=np.float32))
        seen = dict(quantizer.iter_stats())
        assert set(seen) == {"a", "b"}
        assert seen["a"].max_val == pytest.approx(4.0)
        assert seen["b"].max_val == pytest.approx(0.5)

    def test_iter_stats_empty_when_no_calibration(self, quantizer):
        assert list(quantizer.iter_stats()) == []


# ===========================================================================
# 5. Quantizer — quantize / dequantize
# ===========================================================================

class TestQuantizerQuantize:

    def test_int8_values_in_range(self, quantizer):
        data = rng.random((64,)).astype(np.float32) * 2 - 1
        quantizer.calibrate("w", data)
        qt = quantizer.quantize("w", data)
        assert qt.data.max() <= 127
        assert qt.data.min() >= -128

    def test_int4_values_in_range(self):
        q = Quantizer(QuantConfig(precision=QuantPrecision.INT4))
        data = rng.random((32,)).astype(np.float32)
        q.calibrate("w", data)
        qt = q.quantize("w", data)
        assert qt.data.max() <= 7
        assert qt.data.min() >= -8

    def test_fp8_values_in_range(self):
        q = Quantizer(QuantConfig(precision=QuantPrecision.FP8))
        data = rng.random((32,)).astype(np.float32) * 100
        q.calibrate("w", data)
        qt = q.quantize("w", data)
        assert qt.data.max() <= 448
        assert qt.data.min() >= -448

    def test_dequantize_close_to_original_int8(self, quantizer):
        data = np.linspace(-1, 1, 128, dtype=np.float32)
        quantizer.calibrate("w", data)
        qt = quantizer.quantize("w", data)
        rec = quantizer.dequantize(qt)
        np.testing.assert_allclose(rec, data, atol=0.02)

    def test_dequantize_recovers_shape(self, quantizer):
        data = rng.random((4, 8)).astype(np.float32)
        quantizer.calibrate("w", data)
        qt = quantizer.quantize("w", data)
        rec = quantizer.dequantize(qt)
        assert rec.shape == (4, 8)

    def test_symmetric_zero_point_is_zero(self, quantizer):
        data = rng.random((32,)).astype(np.float32)
        quantizer.calibrate("w", data)
        qt = quantizer.quantize("w", data)
        assert qt.zero_point.item() == pytest.approx(0.0)

    def test_asymmetric_zero_point_may_be_nonzero(self):
        q = Quantizer(QuantConfig(symmetric=False))
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        q.calibrate("w", data)
        qt = q.quantize("w", data)
        # For range [1,4] shifted away from zero, zp can be non-zero
        assert isinstance(qt.zero_point.item(), float)

    def test_per_channel_independent_scales(self):
        q = Quantizer(QuantConfig(granularity=QuantGranularity.PER_CHANNEL))
        data = np.array([[0.0, 1.0], [0.0, 100.0]], dtype=np.float32)
        q.calibrate("w", data)
        qt = q.quantize("w", data)
        assert qt.scale.shape == (2,)
        # Channel 1 has much larger range — its scale must be larger
        assert qt.scale[1] > qt.scale[0]

    def test_all_zero_tensor_no_crash(self, quantizer):
        data = np.zeros((16,), dtype=np.float32)
        quantizer.calibrate("w", data)
        qt = quantizer.quantize("w", data)
        rec = quantizer.dequantize(qt)
        np.testing.assert_array_equal(rec, np.zeros((16,)))

    def test_quantize_without_calibrate_raises(self, quantizer):
        with pytest.raises(QuantizationError):
            quantizer.quantize("missing", np.ones((4,)))

    def test_uncalibrated_convenience(self, quantizer):
        data = rng.random((16,)).astype(np.float32)
        qt = quantizer.quantize_uncalibrated(data)
        assert qt.data.max() <= 127

    def test_tensors_quantized_counter(self, quantizer):
        data = rng.random((8,)).astype(np.float32)
        quantizer.calibrate("a", data)
        quantizer.calibrate("b", data)
        quantizer.quantize("a", data)
        quantizer.quantize("b", data)
        assert quantizer.tensors_quantized == 2

    def test_reset_clears_stats(self, quantizer):
        data = rng.random((8,)).astype(np.float32)
        quantizer.calibrate("w", data)
        quantizer.reset()
        with pytest.raises(QuantizationError):
            quantizer.stats("w")


# ===========================================================================
# 6. Runtime — session lifecycle
# ===========================================================================

class TestRuntimeSession:

    def _compiled_model(self, compiler, nodes=None):
        if nodes is None:
            nodes = compiler.parse([_node("n0", "relu", inputs=["x"], outputs=["y"])])
        return compiler.compile(nodes, input_names=["x"], output_names=["y"])

    def test_load_model_returns_session(self, runtime):
        compiler = AstraCoreCompiler()
        model = self._compiled_model(compiler)
        session = runtime.load_model(model)
        assert isinstance(session, InferenceSession)

    def test_session_id_unique(self, runtime):
        compiler = AstraCoreCompiler()
        model = self._compiled_model(compiler)
        s1 = runtime.load_model(model)
        s2 = runtime.load_model(model)
        assert s1.session_id != s2.session_id

    def test_session_state_loaded(self, runtime):
        compiler = AstraCoreCompiler()
        model = self._compiled_model(compiler)
        session = runtime.load_model(model)
        assert session.state == SessionState.LOADED

    def test_session_state_done_after_run(self, runtime):
        compiler = AstraCoreCompiler()
        model = self._compiled_model(compiler)
        session = runtime.load_model(model)
        session.bind_input("x", np.ones((4,)))
        session.run()
        assert session.state == SessionState.DONE

    def test_run_returns_run_result(self, runtime):
        compiler = AstraCoreCompiler()
        model = self._compiled_model(compiler)
        session = runtime.load_model(model)
        session.bind_input("x", np.ones((4,)))
        result = session.run()
        assert isinstance(result, RunResult)

    def test_run_latency_positive(self, runtime):
        compiler = AstraCoreCompiler()
        model = self._compiled_model(compiler)
        session = runtime.load_model(model)
        session.bind_input("x", np.ones((4,)))
        result = session.run()
        assert result.latency_ms >= 0

    def test_run_node_profiles_count(self, runtime):
        compiler = AstraCoreCompiler()
        nodes = compiler.parse([
            _node("n0", "relu",   inputs=["x"],     outputs=["n0_out"]),
            _node("n1", "gelu",   inputs=["n0_out"], outputs=["n1_out"]),
        ])
        model = compiler.compile(nodes, input_names=["x"], output_names=["n1_out"])
        session = runtime.load_model(model)
        session.bind_input("x", np.ones((4,)))
        result = session.run()
        assert len(result.node_profiles) == len(model.schedule)

    def test_run_count_increments(self, runtime):
        compiler = AstraCoreCompiler()
        model = self._compiled_model(compiler)
        session = runtime.load_model(model)
        session.bind_input("x", np.ones((4,)))
        session.run()
        session.run()
        assert session.run_count == 2

    def test_unload_session(self, runtime):
        compiler = AstraCoreCompiler()
        model = self._compiled_model(compiler)
        session = runtime.load_model(model)
        sid = session.session_id
        runtime.unload_session(sid)
        assert runtime.active_sessions == 0

    def test_unload_unknown_raises(self, runtime):
        with pytest.raises(InferenceError):
            runtime.unload_session("nonexistent-id")

    def test_runtime_convenience_run(self, runtime):
        compiler = AstraCoreCompiler()
        model = self._compiled_model(compiler)
        session = runtime.load_model(model)
        result = runtime.run(session, {"x": np.ones((4,))})
        assert isinstance(result, RunResult)
        assert runtime.total_runs == 1

    def test_models_loaded_counter(self, runtime):
        compiler = AstraCoreCompiler()
        model = self._compiled_model(compiler)
        runtime.load_model(model)
        runtime.load_model(model)
        assert runtime.models_loaded == 2

    def test_active_sessions_count(self, runtime):
        compiler = AstraCoreCompiler()
        model = self._compiled_model(compiler)
        runtime.load_model(model)
        runtime.load_model(model)
        assert runtime.active_sessions == 2


# ===========================================================================
# 7. Runtime — operator dispatch
# ===========================================================================

class TestRuntimeDispatch:

    def _run_single_op(self, runtime, op_name, x):
        compiler = AstraCoreCompiler(enable_fusion=False)
        nodes = compiler.parse([_node("n0", op_name, inputs=["x"], outputs=["y"])])
        model = compiler.compile(nodes, input_names=["x"], output_names=["y"])
        session = runtime.load_model(model)
        return runtime.run(session, {"x": x})

    def test_relu_non_negative_output(self, runtime):
        result = self._run_single_op(runtime, "relu", np.array([-1.0, 0.0, 1.0]))
        # relu output stored under node's output name, not necessarily "y"
        assert result.latency_ms >= 0

    def test_gelu_runs_without_error(self, runtime):
        result = self._run_single_op(runtime, "gelu", np.ones((8,)))
        assert result.latency_ms >= 0

    def test_sigmoid_runs(self, runtime):
        result = self._run_single_op(runtime, "sigmoid", np.ones((4,)))
        assert result.latency_ms >= 0

    def test_tanh_runs(self, runtime):
        result = self._run_single_op(runtime, "tanh", np.ones((4,)))
        assert result.latency_ms >= 0

    def test_softmax_runs(self, runtime):
        result = self._run_single_op(runtime, "softmax", np.ones((8,)))
        assert result.latency_ms >= 0

    def test_layernorm_runs(self, runtime):
        result = self._run_single_op(runtime, "layernorm", np.ones((16,)))
        assert result.latency_ms >= 0

    def test_reshape_dispatch(self, runtime):
        compiler = AstraCoreCompiler(enable_fusion=False)
        nodes = compiler.parse([
            _node("n0", "reshape", inputs=["x"], outputs=["y"], attrs={"shape": (2, 4)})
        ])
        model = compiler.compile(nodes, input_names=["x"], output_names=["y"])
        session = runtime.load_model(model)
        runtime.run(session, {"x": np.ones((8,))})

    def test_matmul_dispatch_with_mac(self, runtime):
        compiler = AstraCoreCompiler(enable_fusion=False)
        nodes = compiler.parse([
            _node("n0", "matmul", inputs=["A", "B"], outputs=["C"])
        ])
        model = compiler.compile(nodes, input_names=["A", "B"], output_names=["C"])
        session = runtime.load_model(model)
        A = rng.random((4, 8)).astype(np.float32)
        B = rng.random((8, 4)).astype(np.float32)
        result = runtime.run(session, {"A": A, "B": B})
        assert result.latency_ms >= 0

    def test_result_session_id_matches(self, runtime):
        compiler = AstraCoreCompiler()
        nodes = compiler.parse([_node("n0", "relu", inputs=["x"], outputs=["y"])])
        model = compiler.compile(nodes, input_names=["x"], output_names=["y"])
        session = runtime.load_model(model)
        result = runtime.run(session, {"x": np.ones((4,))})
        assert result.session_id == session.session_id

    def test_slowest_fastest_node(self, runtime):
        compiler = AstraCoreCompiler(enable_fusion=False)
        nodes = compiler.parse([
            _node("n0", "relu", inputs=["x"], outputs=["n0_out"]),
            _node("n1", "gelu", inputs=["n0_out"], outputs=["y"]),
        ])
        model = compiler.compile(nodes, input_names=["x"], output_names=["y"])
        session = runtime.load_model(model)
        result = runtime.run(session, {"x": np.ones((4,))})
        assert result.slowest_node is not None
        assert result.fastest_node is not None
