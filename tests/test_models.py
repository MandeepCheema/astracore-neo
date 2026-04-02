"""
AstraCore Neo — Models testbench.

Coverage:
  - ModelDescriptor: construction, validation, field access, TensorSpec
  - HardwareSpec: construction, validation
  - ModelValidator: all violation types (precision, memory, latency, compute)
  - ModelCatalog: register, unregister, find, find_latest, filter_by_task,
                   filter_by_precision, get_recommended, __contains__, __len__
  - Reference models: all 5 descriptors present and pass ASTRA_HW_SPEC validation
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from models import (
    ModelDescriptor, ModelTask, ModelPrecision, TensorSpec,
    HardwareSpec, ModelValidator, ValidationResult,
    ModelCatalog,
    build_default_catalog, ALL_REFERENCE_MODELS, ASTRA_HW_SPEC,
    ASTRA_DETECT_V1, ASTRA_LANE_V1, ASTRA_DEPTH_V1,
    ASTRA_DMS_V1, ASTRA_OCCGRID_V1,
    ModelsBaseError, ModelRegistrationError, ModelNotFoundError, ModelValidationError,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _simple_desc(
    name="test_model",
    version="1.0.0",
    task=ModelTask.CLASSIFICATION,
    precision=ModelPrecision.INT8,
    params_M=1.0,
    ops_G=1.0,
    memory_mb=4.0,
    max_latency_ms=10.0,
) -> ModelDescriptor:
    return ModelDescriptor(
        name=name,
        version=version,
        task=task,
        precision=precision,
        input_specs=[TensorSpec("input", [1, 3, 224, 224], "uint8")],
        output_specs=[TensorSpec("logits", [1, 1000], "float32")],
        params_M=params_M,
        ops_G=ops_G,
        memory_mb=memory_mb,
        max_latency_ms=max_latency_ms,
    )


def _hw(
    compute_tops=100.0,
    memory_mb=64.0,
    max_latency_ms=20.0,
    precisions=None,
) -> HardwareSpec:
    kw = dict(compute_tops=compute_tops, memory_mb=memory_mb, max_latency_ms=max_latency_ms)
    if precisions is not None:
        from frozenset import __init__  # just to ensure import
    if precisions is not None:
        kw["supported_precisions"] = frozenset(precisions)
    return HardwareSpec(**kw)


# ===========================================================================
# TensorSpec
# ===========================================================================

class TestTensorSpec:

    def test_numel(self):
        t = TensorSpec("x", [2, 3, 4], "float32")
        assert t.numel() == 24

    def test_size_bytes_float32(self):
        t = TensorSpec("x", [1, 3, 224, 224], "float32")
        assert t.size_bytes() == 1 * 3 * 224 * 224 * 4

    def test_size_bytes_int8(self):
        t = TensorSpec("x", [1, 3, 224, 224], "int8")
        assert t.size_bytes() == 1 * 3 * 224 * 224 * 1

    def test_size_bytes_float16(self):
        t = TensorSpec("x", [1, 256], "float16")
        assert t.size_bytes() == 256 * 2


# ===========================================================================
# ModelDescriptor
# ===========================================================================

class TestModelDescriptor:

    def test_model_id(self):
        d = _simple_desc(name="foo", version="2.0.0")
        assert d.model_id == "foo@2.0.0"

    def test_repr_contains_name(self):
        d = _simple_desc()
        assert "test_model" in repr(d)

    def test_empty_name_raises(self):
        with pytest.raises(ModelValidationError):
            ModelDescriptor(
                name="", version="1.0.0",
                task=ModelTask.CLASSIFICATION, precision=ModelPrecision.INT8,
                input_specs=[TensorSpec("x", [1], "int8")],
                output_specs=[TensorSpec("y", [1], "float32")],
                params_M=1.0, ops_G=1.0, memory_mb=1.0,
            )

    def test_empty_version_raises(self):
        with pytest.raises(ModelValidationError):
            ModelDescriptor(
                name="test", version="",
                task=ModelTask.CLASSIFICATION, precision=ModelPrecision.INT8,
                input_specs=[TensorSpec("x", [1], "int8")],
                output_specs=[TensorSpec("y", [1], "float32")],
                params_M=1.0, ops_G=1.0, memory_mb=1.0,
            )

    def test_negative_params_raises(self):
        with pytest.raises(ModelValidationError):
            _simple_desc(params_M=-1.0)

    def test_no_input_specs_raises(self):
        with pytest.raises(ModelValidationError):
            ModelDescriptor(
                name="test", version="1.0.0",
                task=ModelTask.CLASSIFICATION, precision=ModelPrecision.INT8,
                input_specs=[],
                output_specs=[TensorSpec("y", [1], "float32")],
                params_M=1.0, ops_G=1.0, memory_mb=1.0,
            )

    def test_task_and_precision_stored(self):
        d = _simple_desc(task=ModelTask.OBJECT_DETECTION, precision=ModelPrecision.FP16)
        assert d.task == ModelTask.OBJECT_DETECTION
        assert d.precision == ModelPrecision.FP16


# ===========================================================================
# HardwareSpec
# ===========================================================================

class TestHardwareSpec:

    def test_valid_hw_spec(self):
        hw = HardwareSpec(compute_tops=100.0, memory_mb=64.0)
        assert hw.compute_tops == 100.0

    def test_zero_compute_tops_raises(self):
        with pytest.raises(ModelValidationError):
            HardwareSpec(compute_tops=0.0, memory_mb=64.0)

    def test_zero_memory_raises(self):
        with pytest.raises(ModelValidationError):
            HardwareSpec(compute_tops=100.0, memory_mb=0.0)


# ===========================================================================
# ModelValidator
# ===========================================================================

class TestModelValidatorPass:

    def _hw(self):
        return HardwareSpec(compute_tops=1000.0, memory_mb=256.0, max_latency_ms=20.0)

    def test_valid_model_passes(self):
        validator = ModelValidator(self._hw())
        result = validator.validate(_simple_desc())
        assert result.passed is True
        assert result.violations == []

    def test_bool_true_when_passed(self):
        validator = ModelValidator(self._hw())
        assert bool(validator.validate(_simple_desc())) is True

    def test_model_id_in_result(self):
        d = _simple_desc(name="m", version="1.0.0")
        validator = ModelValidator(self._hw())
        result = validator.validate(d)
        assert result.model_id == "m@1.0.0"


class TestModelValidatorViolations:

    def test_unsupported_precision_violation(self):
        hw = HardwareSpec(
            compute_tops=1000.0, memory_mb=256.0,
            supported_precisions=frozenset({ModelPrecision.INT8}),
        )
        d = _simple_desc(precision=ModelPrecision.FP32)
        result = ModelValidator(hw).validate(d)
        assert not result.passed
        assert any("Precision" in v for v in result.violations)

    def test_memory_exceeded_violation(self):
        hw = HardwareSpec(compute_tops=1000.0, memory_mb=2.0)
        d = _simple_desc(memory_mb=10.0)
        result = ModelValidator(hw).validate(d)
        assert not result.passed
        assert any("memory" in v.lower() for v in result.violations)

    def test_latency_target_exceeded_violation(self):
        hw = HardwareSpec(compute_tops=1000.0, memory_mb=256.0, max_latency_ms=5.0)
        d = _simple_desc(max_latency_ms=10.0)  # model wants 10 ms but HW limit is 5 ms
        result = ModelValidator(hw).validate(d)
        assert not result.passed
        assert any("latency" in v.lower() for v in result.violations)

    def test_no_latency_violation_when_hw_limit_zero(self):
        # HW max_latency_ms=0 means unlimited → no latency check
        hw = HardwareSpec(compute_tops=1000.0, memory_mb=256.0, max_latency_ms=0.0)
        d = _simple_desc(max_latency_ms=100.0)
        result = ModelValidator(hw).validate(d)
        assert result.passed

    def test_multiple_violations(self):
        hw = HardwareSpec(
            compute_tops=1.0, memory_mb=1.0, max_latency_ms=1.0,
            supported_precisions=frozenset({ModelPrecision.INT4}),
        )
        d = _simple_desc(precision=ModelPrecision.FP32, memory_mb=10.0, max_latency_ms=5.0)
        result = ModelValidator(hw).validate(d)
        assert not result.passed
        assert len(result.violations) >= 2

    def test_validate_all(self):
        hw = HardwareSpec(compute_tops=1000.0, memory_mb=256.0)
        validator = ModelValidator(hw)
        models = [_simple_desc(name=f"m{i}", version="1.0.0") for i in range(3)]
        results = validator.validate_all(models)
        assert len(results) == 3
        assert all(r.passed for r in results)


# ===========================================================================
# ModelCatalog
# ===========================================================================

class TestModelCatalogBasic:

    def test_empty_catalog(self):
        c = ModelCatalog()
        assert len(c) == 0

    def test_register_and_len(self):
        c = ModelCatalog()
        c.register(_simple_desc())
        assert len(c) == 1

    def test_find_registered(self):
        c = ModelCatalog()
        d = _simple_desc(name="foo", version="1.0.0")
        c.register(d)
        found = c.find("foo", "1.0.0")
        assert found is d

    def test_find_not_registered_raises(self):
        c = ModelCatalog()
        with pytest.raises(ModelNotFoundError):
            c.find("ghost", "1.0.0")

    def test_duplicate_registration_raises(self):
        c = ModelCatalog()
        c.register(_simple_desc(name="a", version="1.0.0"))
        with pytest.raises(ModelRegistrationError):
            c.register(_simple_desc(name="a", version="1.0.0"))

    def test_unregister(self):
        c = ModelCatalog()
        c.register(_simple_desc(name="a", version="1.0.0"))
        c.unregister("a", "1.0.0")
        assert len(c) == 0

    def test_unregister_not_found_raises(self):
        c = ModelCatalog()
        with pytest.raises(ModelNotFoundError):
            c.unregister("ghost", "1.0.0")

    def test_contains_operator(self):
        c = ModelCatalog()
        c.register(_simple_desc(name="a", version="1.0.0"))
        assert "a@1.0.0" in c
        assert "a@2.0.0" not in c

    def test_all_models(self):
        c = ModelCatalog()
        for i in range(3):
            c.register(_simple_desc(name=f"m{i}", version="1.0.0"))
        assert len(c.all_models()) == 3


class TestModelCatalogFilter:

    def _populated_catalog(self):
        c = ModelCatalog()
        c.register(_simple_desc(name="od1", task=ModelTask.OBJECT_DETECTION, precision=ModelPrecision.INT8))
        c.register(_simple_desc(name="od2", task=ModelTask.OBJECT_DETECTION, precision=ModelPrecision.FP16, version="1.0.0"))
        c.register(_simple_desc(name="lane1", task=ModelTask.LANE_DETECTION, precision=ModelPrecision.INT8))
        c.register(_simple_desc(name="cls1", task=ModelTask.CLASSIFICATION, precision=ModelPrecision.FP32))
        return c

    def test_filter_by_task(self):
        c = self._populated_catalog()
        od = c.filter_by_task(ModelTask.OBJECT_DETECTION)
        assert len(od) == 2

    def test_filter_by_task_empty(self):
        c = self._populated_catalog()
        assert c.filter_by_task(ModelTask.DEPTH_ESTIMATION) == []

    def test_filter_by_precision(self):
        c = self._populated_catalog()
        int8 = c.filter_by_precision(ModelPrecision.INT8)
        assert len(int8) == 2

    def test_filter_by_task_and_precision(self):
        c = self._populated_catalog()
        result = c.filter_by_task_and_precision(ModelTask.OBJECT_DETECTION, ModelPrecision.INT8)
        assert len(result) == 1

    def test_find_latest(self):
        c = ModelCatalog()
        c.register(_simple_desc(name="m", version="1.0.0"))
        c.register(_simple_desc(name="m", version="2.0.0"))
        c.register(_simple_desc(name="m", version="1.5.0"))
        latest = c.find_latest("m")
        assert latest.version == "2.0.0"

    def test_find_latest_not_found_raises(self):
        c = ModelCatalog()
        with pytest.raises(ModelNotFoundError):
            c.find_latest("ghost")


class TestModelCatalogRecommendation:

    def test_get_recommended_returns_fastest(self):
        c = ModelCatalog()
        c.register(_simple_desc(name="slow", task=ModelTask.CLASSIFICATION, max_latency_ms=15.0))
        c.register(_simple_desc(name="fast", task=ModelTask.CLASSIFICATION, max_latency_ms=5.0))
        rec = c.get_recommended(ModelTask.CLASSIFICATION)
        assert rec.name == "fast"

    def test_get_recommended_no_task_returns_none(self):
        c = ModelCatalog()
        assert c.get_recommended(ModelTask.SEGMENTATION) is None

    def test_get_recommended_with_validator_filters_invalid(self):
        c = ModelCatalog()
        # Model needs 100 MB but HW has 20 MB
        c.register(_simple_desc(
            name="big", task=ModelTask.CLASSIFICATION, memory_mb=100.0
        ))
        hw = HardwareSpec(compute_tops=1000.0, memory_mb=20.0)
        validator = ModelValidator(hw)
        assert c.get_recommended(ModelTask.CLASSIFICATION, validator=validator) is None

    def test_get_recommended_with_validator_picks_valid(self):
        c = ModelCatalog()
        c.register(_simple_desc(name="big", task=ModelTask.CLASSIFICATION, memory_mb=100.0, max_latency_ms=5.0))
        c.register(_simple_desc(name="small", task=ModelTask.CLASSIFICATION, memory_mb=4.0, max_latency_ms=8.0))
        hw = HardwareSpec(compute_tops=1000.0, memory_mb=20.0)
        validator = ModelValidator(hw)
        rec = c.get_recommended(ModelTask.CLASSIFICATION, validator=validator)
        assert rec.name == "small"


# ===========================================================================
# Reference Models
# ===========================================================================

class TestReferenceModels:

    def test_all_five_models_in_library(self):
        assert len(ALL_REFERENCE_MODELS) == 5

    def test_detect_v1_task(self):
        assert ASTRA_DETECT_V1.task == ModelTask.OBJECT_DETECTION

    def test_lane_v1_task(self):
        assert ASTRA_LANE_V1.task == ModelTask.LANE_DETECTION

    def test_depth_v1_task(self):
        assert ASTRA_DEPTH_V1.task == ModelTask.DEPTH_ESTIMATION

    def test_dms_v1_task(self):
        assert ASTRA_DMS_V1.task == ModelTask.DRIVER_MONITORING

    def test_occgrid_v1_precision(self):
        assert ASTRA_OCCGRID_V1.precision == ModelPrecision.FP16

    def test_all_reference_models_int8_except_occgrid(self):
        int8_models = [
            ASTRA_DETECT_V1, ASTRA_LANE_V1, ASTRA_DEPTH_V1, ASTRA_DMS_V1
        ]
        for m in int8_models:
            assert m.precision == ModelPrecision.INT8, f"{m.name} should be INT8"

    def test_default_catalog_has_all_five(self):
        catalog = build_default_catalog()
        assert len(catalog) == 5

    def test_default_catalog_all_models_findable(self):
        catalog = build_default_catalog()
        for m in ALL_REFERENCE_MODELS:
            found = catalog.find(m.name, m.version)
            assert found.model_id == m.model_id

    def test_all_reference_models_pass_astra_hw_validation(self):
        validator = ModelValidator(ASTRA_HW_SPEC)
        for model in ALL_REFERENCE_MODELS:
            result = validator.validate(model)
            assert result.passed, (
                f"Model {model.model_id} FAILED validation: {result.violations}"
            )

    def test_astra_hw_spec_values(self):
        assert ASTRA_HW_SPEC.compute_tops == 1258.0
        assert ASTRA_HW_SPEC.memory_mb == 128.0
        assert ASTRA_HW_SPEC.max_latency_ms == 16.0

    def test_detect_v1_input_shape(self):
        spec = ASTRA_DETECT_V1.input_specs[0]
        assert spec.shape == [1, 3, 416, 416]

    def test_dms_v1_outputs_three_tensors(self):
        assert len(ASTRA_DMS_V1.output_specs) == 3
