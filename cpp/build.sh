#!/usr/bin/env bash
#
# Build the C++ runtime extension. Detects missing prereqs up-front so
# the actual setup.py build doesn't fail with cryptic errors.
#
# Usage:   ./cpp/build.sh
# Linux / WSL Ubuntu / macOS.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

err() { echo "ERROR: $*" >&2; exit 1; }

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || err "no $PY in PATH"

# 1. C++ compiler
if ! command -v g++ >/dev/null && ! command -v clang++ >/dev/null; then
  err "no C++ compiler found (need g++ or clang++)"
fi

# 2. Python.h (python3-dev on Debian/Ubuntu).
#    If missing AND we're on Debian-family WITHOUT sudo (typical WSL
#    dev setup), auto-fetch the .deb to $HOME/.local/pydev/ and export
#    CPATH so the build finds the header without root.
PYINC="$($PY -c 'import sysconfig; print(sysconfig.get_path("include"))')"
if [[ ! -f "$PYINC/Python.h" ]]; then
  if command -v apt-get >/dev/null; then
    PYVER="$($PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    LOCAL_PYDEV="$HOME/.local/pydev"
    if [[ ! -f "$LOCAL_PYDEV/usr/include/python${PYVER}/Python.h" ]]; then
      echo "Python.h missing; attempting sudo-free user-local install..."
      mkdir -p "$LOCAL_PYDEV"
      (cd /tmp && apt-get download "libpython${PYVER}-dev" "python${PYVER}-dev" 2>/dev/null) || {
        echo "apt-get download failed; falling back to sudo path" >&2
      }
      for deb in /tmp/libpython${PYVER}-dev*.deb /tmp/python${PYVER}-dev*.deb; do
        [[ -f "$deb" ]] && dpkg -x "$deb" "$LOCAL_PYDEV"
      done
    fi
    if [[ -f "$LOCAL_PYDEV/usr/include/python${PYVER}/Python.h" ]]; then
      export CPATH="$LOCAL_PYDEV/usr/include:$LOCAL_PYDEV/usr/include/python${PYVER}:${CPATH:-}"
      echo "Python.h located at $LOCAL_PYDEV/usr/include/python${PYVER}/ (CPATH exported)"
    else
      cat >&2 <<EOF
ERROR: Python.h not available.

Try (in order):
  sudo apt-get install -y python${PYVER}-dev       # Debian / Ubuntu / WSL
  sudo dnf install python3-devel                    # Fedora / RHEL
  macOS (Apple Python):  Python.h ships in the framework; verify
                         /Library/Frameworks/Python.framework/...

EOF
      exit 2
    fi
  else
    echo "ERROR: Python.h missing under $PYINC; no apt-get for auto-fetch" >&2
    exit 2
  fi
fi

# 3. pybind11
if ! "$PY" -c 'import pybind11' 2>/dev/null; then
  echo "Installing pybind11..."
  "$PY" -m pip install --user pybind11
fi

# 4. onnxruntime
if ! "$PY" -c 'import onnxruntime' 2>/dev/null; then
  echo "Installing onnxruntime..."
  "$PY" -m pip install --user onnxruntime
fi

# 5. ORT SONAME symlink. The pip wheel ships libonnxruntime.so.X.Y.Z
# but the library's SONAME is libonnxruntime.so.1, which the linker
# needs at runtime. Create a local symlink so rpath finds it.
ORT_CAPI="$("$PY" -c 'import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), "capi"))')"
if [[ -d "$ORT_CAPI" ]]; then
  if [[ ! -e "$ORT_CAPI/libonnxruntime.so.1" ]]; then
    # Locate the versioned .so and symlink
    VER_SO="$(ls "$ORT_CAPI"/libonnxruntime.so.* 2>/dev/null | head -1)"
    if [[ -n "$VER_SO" ]]; then
      ln -sf "$(basename "$VER_SO")" "$ORT_CAPI/libonnxruntime.so.1" 2>/dev/null || true
      echo "Symlinked SONAME: libonnxruntime.so.1 -> $(basename "$VER_SO")"
    fi
  fi
fi

cd "$HERE"
echo
echo "Building astracore_runtime extension..."
"$PY" setup.py build_ext --inplace

echo
echo "Build complete. Smoke test:"
"$PY" -c "
import sys; sys.path.insert(0, '$HERE')
import astracore_runtime as ar
print('  astracore_runtime', ar.version())
be = ar.make_backend('onnxruntime', ['cpu'])
print('  backend name:', be.name())
"
