"""
AstraCore Neo — Reference Model Library.

Pre-validated model descriptors for the five canonical ADAS inference tasks
on the AstraCore Neo chip.  These represent the "golden" baseline models that
ship with the chip SDK and have been validated against the AstraCore Neo A2
hardware specification (1258 TOPS INT8, 128 MB on-chip SRAM, 16 ms latency).

Usage::

    from models.reference_models import build_default_catalog, ASTRA_HW_SPEC

    catalog = build_default_catalog()
    detect = catalog.find("astra_detect_v1", "1.0.0")
"""

from __future__ import annotations

from .model_descriptor import (
    ModelDescriptor, ModelTask, ModelPrecision, TensorSpec,
)
from .model_catalog import ModelCatalog
from .model_validator import HardwareSpec


# ---------------------------------------------------------------------------
# AstraCore Neo A2 hardware specification
# ---------------------------------------------------------------------------

ASTRA_HW_SPEC = HardwareSpec(
    compute_tops=1258.0,
    memory_mb=128.0,
    max_latency_ms=16.0,         # 60 fps ≈ 16.7 ms/frame budget
)


# ---------------------------------------------------------------------------
# Reference model descriptors
# ---------------------------------------------------------------------------

# 1. Object Detection — YOLOv8-nano-class INT8, 416×416 input
ASTRA_DETECT_V1 = ModelDescriptor(
    name="astra_detect_v1",
    version="1.0.0",
    task=ModelTask.OBJECT_DETECTION,
    precision=ModelPrecision.INT8,
    input_specs=[
        TensorSpec(name="image", shape=[1, 3, 416, 416], dtype="uint8"),
    ],
    output_specs=[
        TensorSpec(name="boxes", shape=[1, 8400, 4], dtype="float32"),
        TensorSpec(name="scores", shape=[1, 8400, 80], dtype="float32"),
    ],
    params_M=3.2,
    ops_G=8.7,
    memory_mb=12.0,
    max_latency_ms=10.0,
    description="Compact object detection, 80-class COCO, 416×416 INT8.",
)

# 2. Lane Detection — lightweight BEV lane poly-line model INT8
ASTRA_LANE_V1 = ModelDescriptor(
    name="astra_lane_v1",
    version="1.0.0",
    task=ModelTask.LANE_DETECTION,
    precision=ModelPrecision.INT8,
    input_specs=[
        TensorSpec(name="image", shape=[1, 3, 320, 640], dtype="uint8"),
    ],
    output_specs=[
        TensorSpec(name="lane_heatmap", shape=[1, 6, 80, 160], dtype="float32"),
        TensorSpec(name="lane_params", shape=[1, 6, 4], dtype="float32"),
    ],
    params_M=1.8,
    ops_G=4.2,
    memory_mb=8.0,
    max_latency_ms=8.0,
    description="6-lane detection from front camera, 320×640 INT8.",
)

# 3. Depth Estimation — monocular depth INT8
ASTRA_DEPTH_V1 = ModelDescriptor(
    name="astra_depth_v1",
    version="1.0.0",
    task=ModelTask.DEPTH_ESTIMATION,
    precision=ModelPrecision.INT8,
    input_specs=[
        TensorSpec(name="image", shape=[1, 3, 192, 640], dtype="uint8"),
    ],
    output_specs=[
        TensorSpec(name="depth_map", shape=[1, 1, 192, 640], dtype="float32"),
    ],
    params_M=6.5,
    ops_G=14.1,
    memory_mb=18.0,
    max_latency_ms=14.0,
    description="Monocular depth estimation, 192×640 INT8.",
)

# 4. Driver Monitoring — facial landmark and gaze estimator INT8 (feeds DMS module)
ASTRA_DMS_V1 = ModelDescriptor(
    name="astra_dms_v1",
    version="1.0.0",
    task=ModelTask.DRIVER_MONITORING,
    precision=ModelPrecision.INT8,
    input_specs=[
        TensorSpec(name="face_crop", shape=[1, 1, 64, 64], dtype="uint8"),
    ],
    output_specs=[
        TensorSpec(name="landmarks_68", shape=[1, 68, 2], dtype="float32"),
        TensorSpec(name="head_pose", shape=[1, 3], dtype="float32"),  # yaw, pitch, roll
        TensorSpec(name="eye_aspect_ratio", shape=[1, 2], dtype="float32"),
    ],
    params_M=0.9,
    ops_G=0.6,
    memory_mb=2.0,
    max_latency_ms=4.0,
    description="68-point face landmark + head pose + EAR for DMS pipeline, 64×64 INT8.",
)

# 5. Occupancy Grid — BEV occupancy from multi-camera FP16
ASTRA_OCCGRID_V1 = ModelDescriptor(
    name="astra_occgrid_v1",
    version="1.0.0",
    task=ModelTask.OCCUPANCY_GRID,
    precision=ModelPrecision.FP16,
    input_specs=[
        TensorSpec(name="surround_views", shape=[6, 3, 256, 512], dtype="float16"),
    ],
    output_specs=[
        TensorSpec(name="occupancy_bev", shape=[1, 16, 200, 200], dtype="float16"),
    ],
    params_M=14.2,
    ops_G=35.0,
    memory_mb=52.0,
    max_latency_ms=15.0,
    description="6-camera surround BEV occupancy grid, FP16, 200×200 output.",
)


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------

ALL_REFERENCE_MODELS: list[ModelDescriptor] = [
    ASTRA_DETECT_V1,
    ASTRA_LANE_V1,
    ASTRA_DEPTH_V1,
    ASTRA_DMS_V1,
    ASTRA_OCCGRID_V1,
]


def build_default_catalog() -> ModelCatalog:
    """
    Build and return a ModelCatalog pre-loaded with all reference models.

    All five models are pre-validated against ASTRA_HW_SPEC.
    """
    catalog = ModelCatalog()
    for model in ALL_REFERENCE_MODELS:
        catalog.register(model)
    return catalog
