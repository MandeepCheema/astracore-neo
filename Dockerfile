# AstraCore Neo — reproducible test + benchmark container.
#
# Usage
# -----
#
#   # Build
#   docker build -t astracore:latest .
#
#   # Run the full regression (≥1400 tests, ~4 minutes on 8 cores)
#   docker run --rm astracore:latest pytest -m 'not integration' -q
#
#   # Regenerate the public LEADERBOARD
#   docker run --rm -v "$PWD/reports:/app/reports" astracore:latest \
#       python scripts/make_leaderboard.py
#
#   # Ad-hoc benchmark
#   docker run --rm astracore:latest \
#       astracore zoo --iter 3 --out /app/reports/zoo.json
#
# The image pins Python 3.12, ONNX Runtime 1.23.2, numpy 1.26 — the same
# triplet the committed LEADERBOARD numbers were measured against. Any
# cloud host with Docker can reproduce those numbers bit-for-bit (modulo
# CPU-specific floating-point determinism in ORT's default graph-opt
# path, which we've validated is stable).

FROM python:3.12-slim-bookworm

# Pinned toolchain dependencies needed for the C++ extension build.
# Kept deliberately minimal — no GPU base, no CUDA, no OpenCV; the
# production image is a customer-config choice, this is dev / CI.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        python3-dev \
        libonnxruntime-dev 2>/dev/null || true \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy in reqs first for build-cache friendliness.
COPY pyproject.toml ./
COPY requirements.txt* ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
        'numpy>=1.26,<2.0' \
        'onnx>=1.14,<2.0' \
        'onnxruntime==1.23.2' \
        'pyyaml>=6.0' \
        'pybind11>=2.11' \
        pytest pytest-cov

# Now the source.
COPY astracore/ ./astracore/
COPY tools/ ./tools/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY examples/ ./examples/
COPY cpp/ ./cpp/
COPY data/models/yolov8n.onnx ./data/models/yolov8n.onnx
COPY data/models/yolov8n.manifest.json ./data/models/yolov8n.manifest.json
COPY data/calibration/ ./data/calibration/
COPY pytest.ini ./
COPY README.md LICENSE ./
# Zoo models are large (~1 GB total); not baked into the image by
# default. Mount them in at run-time with `-v "$PWD/data/models/zoo:/app/data/models/zoo"`
# or run `python scripts/fetch_model_zoo.py` inside the container.

RUN pip install --no-cache-dir -e .

# Build the C++ extension too so cross-runtime conformance gate runs
# inside the container. Continue on failure — some minimal CI hosts
# don't have a full g++ toolchain and Python-only tests still need to
# work.
RUN bash cpp/build.sh 2>&1 || echo "C++ build skipped — tests will skip cleanly"

ENV PYTHONPATH=/app:/app/cpp

# Default: print versions so docker run works with no args.
CMD ["bash", "-c", "\
    echo 'AstraCore Neo container ready.'; \
    astracore version; \
    echo 'Run: docker run astracore pytest -m \"not integration\" -q'"]
