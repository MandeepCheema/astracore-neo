"""Build the ``astracore_runtime`` C++ extension via pybind11.

Usage (Linux / WSL Ubuntu):

    cd cpp
    pip install pybind11 onnxruntime          # if not already
    python setup.py build_ext --inplace

That produces ``astracore_runtime.cpython-<py>-<arch>.so`` in this
directory. Add the directory to ``sys.path`` to import it:

    import sys
    sys.path.insert(0, "cpp")
    import astracore_runtime

The Python tests under ``tests/test_cpp_runtime.py`` skip cleanly when
the extension is not built.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
import pybind11
import onnxruntime


HERE = Path(__file__).resolve().parent

# ORT shared library lives under the wheel's `capi/` directory. We
# link against it AND tell the runtime loader where to find it via
# rpath ($ORIGIN/.../capi/).
ORT_PKG_DIR = Path(onnxruntime.__file__).resolve().parent
ORT_CAPI_DIR = ORT_PKG_DIR / "capi"

# Find the actual library file (libonnxruntime.so.X.Y.Z on Linux,
# libonnxruntime.dylib on macOS, onnxruntime.dll on Windows).
def _find_ort_lib() -> Path:
    candidates = []
    for name in ("libonnxruntime.so", "libonnxruntime.dylib",
                 "onnxruntime.dll"):
        candidates.append(ORT_CAPI_DIR / name)
    # Versioned form on Linux.
    for child in ORT_CAPI_DIR.iterdir():
        if child.name.startswith("libonnxruntime.so.") or \
           child.name.startswith("libonnxruntime.") and child.suffix == ".dylib":
            candidates.append(child)
    for c in candidates:
        if c.exists():
            return c
    raise RuntimeError(
        f"could not find onnxruntime shared library under {ORT_CAPI_DIR}"
    )


ort_lib = _find_ort_lib()
print(f"[setup] Using ONNX Runtime lib: {ort_lib}")
print(f"[setup] Using pybind11 headers: {pybind11.get_include()}")

# On Linux, libonnxruntime.so.X.Y.Z is what's on disk but ld-link wants
# `-lonnxruntime` (i.e. libonnxruntime.so). Symlink in a build/ dir.
build_link_dir = HERE / "build" / "link"
build_link_dir.mkdir(parents=True, exist_ok=True)
soname_link = build_link_dir / "libonnxruntime.so"
if sys.platform == "linux" and not soname_link.exists():
    try:
        soname_link.symlink_to(ort_lib)
    except FileExistsError:
        pass

extra_link_args: list = []
extra_compile_args = [
    "-std=c++17",
    "-O2",
    "-Wall",
    "-Wextra",
    "-Wno-unused-parameter",
]
if sys.platform == "linux":
    # rpath so the loaded .so can find ORT next to itself OR in the
    # caller-supplied LD_LIBRARY_PATH.
    extra_link_args = [
        f"-Wl,-rpath,{ORT_CAPI_DIR}",
        f"-Wl,-rpath,$ORIGIN",
    ]
elif sys.platform == "darwin":
    extra_compile_args.remove("-Wno-unused-parameter")
    extra_link_args = [
        f"-Wl,-rpath,{ORT_CAPI_DIR}",
    ]


ext = Extension(
    name="astracore_runtime",
    sources=[
        str(HERE / "src" / "runtime.cpp"),
        str(HERE / "python" / "bindings.cpp"),
    ],
    include_dirs=[
        str(HERE / "include"),
        str(HERE / "third_party" / "onnxruntime"),
        pybind11.get_include(),
        sysconfig.get_path("include"),
    ],
    library_dirs=[
        str(build_link_dir),    # symlinked libonnxruntime.so
        str(ORT_CAPI_DIR),
    ],
    libraries=["onnxruntime"],
    language="c++",
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name="astracore-runtime",
    version="0.1.0",
    description="AstraCore Neo C++ runtime (pybind11 binding)",
    ext_modules=[ext],
    zip_safe=False,
)
