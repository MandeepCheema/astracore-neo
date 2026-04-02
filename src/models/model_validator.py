"""
AstraCore Neo — Model Validator.

Checks a ModelDescriptor against a HardwareSpec to determine whether the model
can be deployed on the target chip configuration.  Returns a ValidationResult
that lists any constraint violations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model_descriptor import ModelDescriptor, ModelPrecision
from .exceptions import ModelValidationError


@dataclass
class HardwareSpec:
    """
    Hardware capability constraints for model deployment.

    Parameters
    ----------
    compute_tops      : peak compute in TOPS (tera-ops/sec)
    memory_mb         : available SRAM/on-chip memory in MB
    max_latency_ms    : maximum acceptable inference latency in ms (0 = unlimited)
    supported_precisions : set of ModelPrecision values the hardware supports
    """
    compute_tops: float
    memory_mb: float
    max_latency_ms: float = 0.0
    supported_precisions: frozenset[ModelPrecision] = field(
        default_factory=lambda: frozenset({
            ModelPrecision.INT8, ModelPrecision.INT4,
            ModelPrecision.FP16, ModelPrecision.FP32,
        })
    )

    def __post_init__(self) -> None:
        if self.compute_tops <= 0:
            raise ModelValidationError(f"compute_tops must be > 0, got {self.compute_tops}")
        if self.memory_mb <= 0:
            raise ModelValidationError(f"memory_mb must be > 0, got {self.memory_mb}")


@dataclass
class ValidationResult:
    """Result of validating a model against a HardwareSpec."""
    model_id: str
    passed: bool
    violations: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


class ModelValidator:
    """
    Validates ModelDescriptors against a HardwareSpec.

    Usage::

        hw = HardwareSpec(compute_tops=1258.0, memory_mb=128.0, max_latency_ms=16.0)
        validator = ModelValidator(hw)
        result = validator.validate(descriptor)
        if not result:
            print(result.violations)
    """

    def __init__(self, hw: HardwareSpec) -> None:
        self._hw = hw

    def validate(self, descriptor: ModelDescriptor) -> ValidationResult:
        """Check descriptor against all hardware constraints."""
        violations: list[str] = []

        # Precision support
        if descriptor.precision not in self._hw.supported_precisions:
            violations.append(
                f"Precision {descriptor.precision.name} not supported by hardware "
                f"(supported: {[p.name for p in self._hw.supported_precisions]})"
            )

        # Memory check
        if descriptor.memory_mb > self._hw.memory_mb:
            violations.append(
                f"Model requires {descriptor.memory_mb:.1f} MB activation memory; "
                f"hardware provides {self._hw.memory_mb:.1f} MB"
            )

        # Latency check (only when both model and HW have a constraint)
        if self._hw.max_latency_ms > 0 and descriptor.max_latency_ms > 0:
            if descriptor.max_latency_ms > self._hw.max_latency_ms:
                violations.append(
                    f"Model latency target {descriptor.max_latency_ms:.1f} ms exceeds "
                    f"hardware limit {self._hw.max_latency_ms:.1f} ms"
                )

        # Compute throughput check
        # Minimum compute needed = ops_G / (latency_ms / 1000)
        # We check whether HW TOPS is sufficient to meet the model's latency target
        if descriptor.max_latency_ms > 0 and self._hw.max_latency_ms > 0:
            latency_budget_s = min(descriptor.max_latency_ms, self._hw.max_latency_ms) / 1000.0
            required_tops = (descriptor.ops_G * 1e9) / latency_budget_s / 1e12  # to TOPS
            if required_tops > self._hw.compute_tops:
                violations.append(
                    f"Model requires {required_tops:.1f} TOPS to meet latency; "
                    f"hardware provides {self._hw.compute_tops:.1f} TOPS"
                )

        return ValidationResult(
            model_id=descriptor.model_id,
            passed=len(violations) == 0,
            violations=violations,
        )

    def validate_all(self, descriptors: list[ModelDescriptor]) -> list[ValidationResult]:
        """Validate a list of descriptors; returns one result per descriptor."""
        return [self.validate(d) for d in descriptors]

    @property
    def hardware_spec(self) -> HardwareSpec:
        return self._hw
