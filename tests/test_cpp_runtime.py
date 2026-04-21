"""Tests for the C++ runtime via the pybind11 binding.

These tests skip cleanly if ``cpp/astracore_runtime*.so`` (or .pyd
on Windows) hasn't been built yet — see ``cpp/README.md`` for build
steps. Once built, they validate:

1. The extension loads + reports a version.
2. ``make_backend('onnxruntime', ['cpu'])`` round-trips.
3. End-to-end YOLOv8n: same model + same input through C++ and Python
   OrtBackend produces bit-identical output (the cross-runtime
   conformance gate).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parent.parent
CPP_DIR = REPO / "cpp"
YOLO = REPO / "data" / "models" / "yolov8n.onnx"


def _import_extension():
    """Try to import the built C++ extension, return module or None."""
    sys.path.insert(0, str(CPP_DIR))
    try:
        import astracore_runtime          # noqa: F401
        return astracore_runtime
    except Exception:
        return None
    finally:
        if str(CPP_DIR) in sys.path:
            try:
                sys.path.remove(str(CPP_DIR))
            except ValueError:
                pass


_ext = _import_extension()
_skip_no_ext = pytest.mark.skipif(
    _ext is None,
    reason="C++ extension not built — run "
           "`cd cpp && python setup.py build_ext --inplace`",
)
_skip_no_yolo = pytest.mark.skipif(
    not YOLO.exists(),
    reason="yolov8n.onnx missing",
)


# ---------------------------------------------------------------------------
# Scaffold-only tests — pass even when the extension is absent.
# ---------------------------------------------------------------------------

def test_cpp_directory_layout():
    """The C++ source tree must be present + structured even if not built."""
    must_exist = [
        CPP_DIR / "include" / "astracore" / "runtime.hpp",
        CPP_DIR / "src" / "runtime.cpp",
        CPP_DIR / "python" / "bindings.cpp",
        CPP_DIR / "setup.py",
        CPP_DIR / "README.md",
        CPP_DIR / "third_party" / "onnxruntime" / "onnxruntime_c_api.h",
        CPP_DIR / "third_party" / "onnxruntime" / "onnxruntime_cxx_api.h",
    ]
    missing = [str(p) for p in must_exist if not p.exists()]
    assert not missing, f"missing C++ scaffold files: {missing}"


def test_runtime_header_declares_public_api():
    """Header must export the symbols pybind11 binds against."""
    body = (CPP_DIR / "include" / "astracore" / "runtime.hpp").read_text()
    for symbol in ("class Backend", "class Tensor", "struct Report",
                   "make_backend", "version()", "DType"):
        assert symbol in body, f"missing public symbol: {symbol}"


def test_pybind_binding_exposes_make_backend():
    """The pybind11 binding source must register make_backend + Backend."""
    body = (CPP_DIR / "python" / "bindings.cpp").read_text()
    assert 'PYBIND11_MODULE(astracore_runtime' in body
    assert '"make_backend"' in body
    assert 'py::class_<ac::Backend>' in body


# ---------------------------------------------------------------------------
# Live tests — only when the extension is built.
# ---------------------------------------------------------------------------

@_skip_no_ext
def test_extension_version_reachable():
    assert hasattr(_ext, "version")
    # v0.2 adds make_backend_with_options but keeps v0.1 API.
    assert _ext.version() in ("0.1.0", "0.2.0")


@_skip_no_ext
def test_v02_per_ep_options_binding_present():
    """v0.2: pybind binding must expose make_backend_with_options."""
    # If the extension is v0.1 that's OK — this is the one path we
    # explicitly allow to auto-upgrade when rebuilt.
    if _ext.version() == "0.1.0":
        pytest.skip("extension is v0.1 — rebuild for v0.2")
    assert callable(getattr(_ext, "make_backend_with_options", None))
    # Sanity: asking for CUDA on a CPU-only build must still produce
    # a runnable backend (CPU fallback).
    be = _ext.make_backend_with_options(
        "onnxruntime",
        [("cuda", {"device_id": "0"}), ("cpu", {})],
    )
    assert be.name() == "onnxruntime"


@_skip_no_ext
def test_make_backend_returns_object():
    be = _ext.make_backend("onnxruntime", ["cpu"])
    assert hasattr(be, "compile")
    assert hasattr(be, "run")
    assert be.name() == "onnxruntime"


@_skip_no_ext
@_skip_no_yolo
def test_cpp_yolo_runs_end_to_end():
    be = _ext.make_backend("onnxruntime", ["cpu"])
    program = be.compile(str(YOLO))
    x = np.random.default_rng(0).standard_normal(
        (1, 3, 640, 640)).astype(np.float32)
    out = be.run(program, {"images": x})
    assert out, "C++ run returned no outputs"
    arr = next(iter(out.values()))
    assert arr.shape == (1, 84, 8400)


@_skip_no_ext
@_skip_no_yolo
def test_cpp_and_python_outputs_are_bit_identical():
    """Cross-runtime conformance gate.

    Same ONNX, same input, same backend (ORT CPU FP32) must produce
    bit-identical numerics whether driven from C++ or Python.
    """
    from astracore.backends.ort import OrtBackend

    rng = np.random.default_rng(42)
    x = rng.standard_normal((1, 3, 640, 640)).astype(np.float32)

    cpp_be = _ext.make_backend("onnxruntime", ["cpu"])
    cpp_program = cpp_be.compile(str(YOLO))
    cpp_out = next(iter(cpp_be.run(cpp_program, {"images": x}).values()))

    py_be = OrtBackend(providers=["cpu"])
    py_program = py_be.compile(str(YOLO))
    py_out = next(iter(py_be.run(py_program, {"images": x.copy()}).values()))

    np.testing.assert_array_equal(
        np.asarray(cpp_out), np.asarray(py_out),
        err_msg="C++ and Python OrtBackend produced different outputs — "
                "the cross-runtime conformance gate is broken",
    )
