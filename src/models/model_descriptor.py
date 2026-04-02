"""
AstraCore Neo — Model Descriptor.

Defines the metadata schema for AI models in the AstraCore Neo model library.
A ModelDescriptor is hardware-agnostic: it describes what a model does and what
compute/memory resources it requires, without embedding any weights or runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .exceptions import ModelValidationError


class ModelTask(Enum):
    """High-level task category for a model."""
    OBJECT_DETECTION   = auto()   # 2-D bounding box detection
    LANE_DETECTION     = auto()   # Road lane segmentation / polyline
    DEPTH_ESTIMATION   = auto()   # Monocular or stereo depth map
    DRIVER_MONITORING  = auto()   # Facial landmark / gaze / pose for DMS
    OCCUPANCY_GRID     = auto()   # BEV (bird's-eye-view) occupancy prediction
    CLASSIFICATION     = auto()   # Single-label or multi-label classification
    SEGMENTATION       = auto()   # Pixel-wise semantic segmentation


class ModelPrecision(Enum):
    """Numerical precision of model weights and activations."""
    FP32 = auto()
    FP16 = auto()
    INT8 = auto()
    INT4 = auto()


@dataclass
class TensorSpec:
    """Specification for one model input or output tensor."""
    name: str
    shape: list[int]          # e.g. [1, 3, 416, 416] for a batched RGB image
    dtype: str                # "float32", "float16", "int8", "int4", "uint8"

    def numel(self) -> int:
        """Total number of elements (product of all dimensions)."""
        result = 1
        for d in self.shape:
            result *= d
        return result

    def size_bytes(self) -> int:
        """Byte size based on dtype."""
        _dtype_bytes = {
            "float32": 4, "float16": 2,
            "int8": 1, "uint8": 1,
            "int4": 1,   # packed: 2 values per byte in HW; report as 1 for simplicity
        }
        return self.numel() * _dtype_bytes.get(self.dtype, 1)


@dataclass
class ModelDescriptor:
    """
    Full metadata descriptor for one versioned AI model.

    Parameters
    ----------
    name           : unique model name (e.g. "astra_detect_v1")
    version        : semantic version string (e.g. "1.0.0")
    task           : high-level task category
    precision      : dominant weight/activation precision
    input_specs    : list of input TensorSpec (one per model input)
    output_specs   : list of output TensorSpec (one per model output)
    params_M       : total parameter count in millions
    ops_G          : compute requirement in giga-ops (GOPs) per inference
    memory_mb      : peak activation memory in MB
    max_latency_ms : target maximum latency in ms (0 = no constraint)
    description    : optional human-readable description
    """
    name: str
    version: str
    task: ModelTask
    precision: ModelPrecision
    input_specs: list[TensorSpec]
    output_specs: list[TensorSpec]
    params_M: float            # parameters in millions
    ops_G: float               # giga-ops per inference
    memory_mb: float           # peak activation memory in MB
    max_latency_ms: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ModelValidationError("ModelDescriptor.name must not be empty")
        if not self.version:
            raise ModelValidationError("ModelDescriptor.version must not be empty")
        if self.params_M < 0:
            raise ModelValidationError(f"params_M must be >= 0, got {self.params_M}")
        if self.ops_G < 0:
            raise ModelValidationError(f"ops_G must be >= 0, got {self.ops_G}")
        if self.memory_mb < 0:
            raise ModelValidationError(f"memory_mb must be >= 0, got {self.memory_mb}")
        if not self.input_specs:
            raise ModelValidationError("ModelDescriptor must have at least one input_spec")
        if not self.output_specs:
            raise ModelValidationError("ModelDescriptor must have at least one output_spec")

    @property
    def model_id(self) -> str:
        """Unique identifier: name@version."""
        return f"{self.name}@{self.version}"

    def __repr__(self) -> str:
        return (
            f"ModelDescriptor('{self.model_id}', task={self.task.name}, "
            f"precision={self.precision.name}, "
            f"params={self.params_M:.1f}M, ops={self.ops_G:.2f}G)"
        )
