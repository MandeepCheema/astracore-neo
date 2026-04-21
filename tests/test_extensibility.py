"""Pillar 1 proof — customer can extend AstraCore without editing our source.

This test file must **not** import anything from ``tools.npu_ref`` or
``src.*``; it acts the way an external Tier-1 integrator's pip package
would, using only the public ``astracore.*`` API.

If any assertion fails, the extensibility claim in the SDK datasheet is
not defensible — a customer would have to fork the repo.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict

import numpy as np
import pytest

import astracore
from astracore import (
    Backend,
    BackendReport,
    Quantiser,
    get_backend,
    get_op,
    get_quantiser,
    list_backends,
    list_ops,
    list_quantisers,
    register_backend,
    register_op,
    register_quantiser,
)
from astracore.registry import _reset_for_tests


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test starts with a clean registry (prevents cross-test bleed)."""
    # We don't globally wipe — we want the built-in backends still visible
    # for end-to-end tests. Instead we snapshot and restore.
    from astracore.registry import _ops, _quantisers, _backends
    snap_ops = dict(_ops._items)
    snap_quant = dict(_quantisers._items)
    snap_be = dict(_backends._items)
    yield
    _ops._items.clear(); _ops._items.update(snap_ops)
    _quantisers._items.clear(); _quantisers._items.update(snap_quant)
    _backends._items.clear(); _backends._items.update(snap_be)


# ---------------------------------------------------------------------------
# Pillar 1a — Custom op registration
# ---------------------------------------------------------------------------

def test_custom_op_registration_is_discoverable():
    """An external package can register an op handler and it shows up."""
    @register_op("MyOEMCustomNormalisation")
    def handle_custom_norm(node, graph):
        return {"handled": True, "node": node, "graph": graph}

    assert "MyOEMCustomNormalisation" in list_ops()
    handler = get_op("MyOEMCustomNormalisation")
    result = handler(node={"name": "n0"}, graph=None)
    assert result["handled"] is True


def test_custom_op_overrides_registry():
    """A second registration under the same name wins (last-writer-wins)."""
    @register_op("OverrideMe")
    def v1(node, graph):
        return "v1"

    @register_op("OverrideMe")
    def v2(node, graph):
        return "v2"

    assert get_op("OverrideMe")(None, None) == "v2"


def test_missing_op_raises_helpful_error():
    with pytest.raises(KeyError) as excinfo:
        get_op("NotInAnyRegistry")
    assert "Registered:" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Pillar 1b — Custom quantiser
# ---------------------------------------------------------------------------

def test_custom_quantiser_subclass_registers():
    """An OEM can subclass Quantiser + plug in their calibration recipe."""

    @register_quantiser("oem_percentile_98")
    class OEMPercentile98(Quantiser):
        def _percentile(self):
            return 98.0

    assert "oem_percentile_98" in list_quantisers()
    cls = get_quantiser("oem_percentile_98")
    q = cls()
    assert q._percentile() == 98.0
    # Default (overridable) knobs are present
    assert hasattr(q, "config")
    assert q.config.precision == "INT8"


def test_quantiser_config_is_dataclass_and_extensible():
    """OEM can add their own config fields without breaking ours."""
    from astracore.quantiser import QuantiserConfig
    cfg = QuantiserConfig(precision="INT4",
                          activation_calibration="percentile_99",
                          extra={"oem_knob_x": 0.5})
    assert cfg.precision == "INT4"
    assert cfg.extra["oem_knob_x"] == 0.5


# ---------------------------------------------------------------------------
# Pillar 1c — Custom backend
# ---------------------------------------------------------------------------

def test_custom_backend_plugs_in_and_runs():
    """An OEM's backend class satisfies the protocol and reports results."""

    @register_backend("tier1-oem-mock")
    class Tier1OemMockBackend:
        """Pretends to run on a 500 TOPS INT8 target silicon."""
        name = "tier1-oem-mock"
        silicon_profile = "tier1-oem-mock-500tops"

        def __init__(self):
            self._last = BackendReport(backend=self.name,
                                       silicon_profile=self.silicon_profile)

        def compile(self, graph, *, precision="INT8", sparsity="dense"):
            self._last.precision = precision
            self._last.sparsity = sparsity
            return {"compiled": True, "graph": graph}

        def run(self, program, inputs: Dict[str, np.ndarray]):
            # Pretend each run does exactly 1 GMACs for the test.
            self._last.n_inferences += 1
            self._last.wall_s_total += 0.001  # 1 ms
            self._last.wall_ms_per_inference = 1.0
            self._last.mac_ops_effective = 1_000_000_000
            self._last.delivered_tops = 1.0
            return {"y": np.zeros((1,), dtype=np.float32)}

        def report_last(self) -> BackendReport:
            return self._last

    assert "tier1-oem-mock" in list_backends()
    backend_cls = get_backend("tier1-oem-mock")
    be = backend_cls()
    # Protocol check: the mock must satisfy the Backend protocol.
    assert isinstance(be, Backend)

    program = be.compile("some-graph", precision="INT4", sparsity="2:4")
    assert program["compiled"]
    be.run(program, {"x": np.zeros((1,), dtype=np.float32)})
    rep = be.report_last()
    assert rep.backend == "tier1-oem-mock"
    assert rep.precision == "INT4"
    assert rep.sparsity == "2:4"
    assert rep.delivered_tops == 1.0


def test_backend_protocol_runtime_check_rejects_garbage():
    """Something that doesn't have compile / run / report_last fails the
    isinstance(_, Backend) check."""

    class NotABackend:
        pass

    assert not isinstance(NotABackend(), Backend)


# ---------------------------------------------------------------------------
# Pillar 1d — End-to-end: custom backend gets used by the benchmark harness
# ---------------------------------------------------------------------------

def test_benchmark_harness_picks_up_custom_backend(tmp_path):
    """Prove the `astracore bench` code path uses a plugin backend."""
    from astracore.benchmark import benchmark_model
    import onnx
    from onnx import helper, TensorProto

    # Build a trivial ONNX: y = x (Identity) so we don't need real weights.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "test", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)

    model_path = tmp_path / "identity.onnx"
    onnx.save(model, str(model_path))

    # Register a backend that records it was called.
    calls = {"compile": 0, "run": 0}

    @register_backend("counter-backend")
    class CounterBackend:
        name = "counter-backend"
        silicon_profile = "counter"

        def __init__(self):
            self._last = BackendReport(backend=self.name,
                                       silicon_profile=self.silicon_profile)

        def compile(self, graph, *, precision="INT8", sparsity="dense"):
            calls["compile"] += 1
            self._last.precision = precision
            return {"graph": graph}

        def run(self, program, inputs):
            calls["run"] += 1
            self._last.n_inferences += 1
            self._last.wall_s_total += 0.001
            self._last.wall_ms_per_inference = 1.0
            return {"y": np.zeros((1, 1), dtype=np.float32)}

        def report_last(self):
            return self._last

    report = benchmark_model(model_path, backend="counter-backend",
                             precision="INT8", n_iter=3, warmup=1)
    assert calls["compile"] == 1
    # warmup + 3 iters = 4 runs
    assert calls["run"] == 4
    assert report.backend == "counter-backend"
    assert report.n_inferences == 3


# ---------------------------------------------------------------------------
# Smoke tests — built-ins still present after plugin registration
# ---------------------------------------------------------------------------

def test_builtin_backends_still_discoverable():
    assert "npu-sim" in list_backends()
    assert "onnxruntime" in list_backends()


def test_package_has_stable_public_api():
    """Freeze the public API. Anything removed here breaks OEM code."""
    public = {
        "__version__",
        "register_op", "register_quantiser", "register_backend",
        "get_op", "get_quantiser", "get_backend",
        "list_ops", "list_quantisers", "list_backends",
        "Backend", "BackendReport", "Quantiser",
    }
    for name in public:
        assert hasattr(astracore, name), f"Missing public API: {name}"
