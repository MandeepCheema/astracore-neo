"""
AstraCore Neo Models Subsystem.

Pre-validated AI model library: descriptors, hardware validation, and catalog.

Public API::

    from models import ModelDescriptor, ModelTask, ModelPrecision, TensorSpec
    from models import HardwareSpec, ModelValidator, ValidationResult
    from models import ModelCatalog
    from models import build_default_catalog, ALL_REFERENCE_MODELS, ASTRA_HW_SPEC
    from models import (ASTRA_DETECT_V1, ASTRA_LANE_V1, ASTRA_DEPTH_V1,
                         ASTRA_DMS_V1, ASTRA_OCCGRID_V1)
    from models import ModelsBaseError, ModelRegistrationError, ModelNotFoundError, ModelValidationError
"""

from .model_descriptor import (
    ModelDescriptor, ModelTask, ModelPrecision, TensorSpec,
)
from .model_validator import (
    HardwareSpec, ModelValidator, ValidationResult,
)
from .model_catalog import ModelCatalog
from .reference_models import (
    build_default_catalog, ALL_REFERENCE_MODELS, ASTRA_HW_SPEC,
    ASTRA_DETECT_V1, ASTRA_LANE_V1, ASTRA_DEPTH_V1,
    ASTRA_DMS_V1, ASTRA_OCCGRID_V1,
)
from .exceptions import (
    ModelsBaseError, ModelRegistrationError, ModelNotFoundError, ModelValidationError,
)

__all__ = [
    # Descriptor
    "ModelDescriptor", "ModelTask", "ModelPrecision", "TensorSpec",
    # Validator
    "HardwareSpec", "ModelValidator", "ValidationResult",
    # Catalog
    "ModelCatalog",
    # Reference models
    "build_default_catalog", "ALL_REFERENCE_MODELS", "ASTRA_HW_SPEC",
    "ASTRA_DETECT_V1", "ASTRA_LANE_V1", "ASTRA_DEPTH_V1",
    "ASTRA_DMS_V1", "ASTRA_OCCGRID_V1",
    # Exceptions
    "ModelsBaseError", "ModelRegistrationError", "ModelNotFoundError", "ModelValidationError",
]
