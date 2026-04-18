#!/bin/bash
# Build Verilator 5.x from source in WSL.
# Installs to /usr/local (takes precedence over /usr/bin).
# Expected runtime: 5-12 minutes on 12 cores.
set -e

VERILATOR_TAG="${VERILATOR_TAG:-v5.024}"

echo "=== Installing build deps ==="
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git autoconf g++ flex bison help2man \
    libfl2 libfl-dev zlib1g zlib1g-dev ccache 2>&1 | tail -5

echo "=== Cloning Verilator $VERILATOR_TAG ==="
BUILD=/opt/verilator-src
if [ -d "$BUILD/.git" ]; then
    cd "$BUILD"
    git fetch --tags --depth=1 origin "$VERILATOR_TAG"
else
    rm -rf "$BUILD"
    git clone --depth=1 --branch "$VERILATOR_TAG" https://github.com/verilator/verilator.git "$BUILD"
    cd "$BUILD"
fi
git checkout "$VERILATOR_TAG"
git describe --tags

echo "=== autoconf ==="
autoconf

echo "=== configure (prefix=/usr/local) ==="
./configure --prefix=/usr/local 2>&1 | tail -5

echo "=== make -j$(nproc) ==="
make -j"$(nproc)" 2>&1 | tail -10

echo "=== make install ==="
make install 2>&1 | tail -5

echo "=== verify ==="
hash -r
/usr/local/bin/verilator --version
which verilator
verilator --version

echo "=== DONE ==="
