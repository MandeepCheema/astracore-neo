"""
AstraCore Neo — Model Catalog.

Maintains a registry of versioned ModelDescriptors.  Provides search and
filtering capabilities so callers can discover available models by task,
precision, or hardware suitability.
"""

from __future__ import annotations

from typing import Optional

from .model_descriptor import ModelDescriptor, ModelTask, ModelPrecision
from .model_validator import ModelValidator
from .exceptions import ModelRegistrationError, ModelNotFoundError


class ModelCatalog:
    """
    Registry of versioned AI model descriptors.

    Usage::

        catalog = ModelCatalog()
        catalog.register(descriptor)
        desc = catalog.find("astra_detect_v1", "1.0.0")
        od_models = catalog.filter_by_task(ModelTask.OBJECT_DETECTION)
    """

    def __init__(self) -> None:
        # key: (name, version) → ModelDescriptor
        self._registry: dict[tuple[str, str], ModelDescriptor] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, descriptor: ModelDescriptor) -> None:
        """Add a model to the catalog. Raises if already registered."""
        key = (descriptor.name, descriptor.version)
        if key in self._registry:
            raise ModelRegistrationError(
                f"Model '{descriptor.model_id}' is already registered"
            )
        self._registry[key] = descriptor

    def unregister(self, name: str, version: str) -> None:
        """Remove a model from the catalog."""
        key = (name, version)
        if key not in self._registry:
            raise ModelNotFoundError(f"Model '{name}@{version}' not found in catalog")
        del self._registry[key]

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def find(self, name: str, version: str) -> ModelDescriptor:
        """Return the descriptor for a specific model version."""
        key = (name, version)
        if key not in self._registry:
            raise ModelNotFoundError(f"Model '{name}@{version}' not found in catalog")
        return self._registry[key]

    def find_latest(self, name: str) -> ModelDescriptor:
        """Return the lexicographically latest version of a named model."""
        candidates = [v for (n, v) in self._registry if n == name]
        if not candidates:
            raise ModelNotFoundError(f"No versions of model '{name}' found in catalog")
        latest_version = sorted(candidates)[-1]
        return self._registry[(name, latest_version)]

    def all_models(self) -> list[ModelDescriptor]:
        """Return all registered descriptors (order is insertion order)."""
        return list(self._registry.values())

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_task(self, task: ModelTask) -> list[ModelDescriptor]:
        """Return all models for a given task."""
        return [d for d in self._registry.values() if d.task == task]

    def filter_by_precision(self, precision: ModelPrecision) -> list[ModelDescriptor]:
        """Return all models with a given precision."""
        return [d for d in self._registry.values() if d.precision == precision]

    def filter_by_task_and_precision(
        self, task: ModelTask, precision: ModelPrecision
    ) -> list[ModelDescriptor]:
        return [
            d for d in self._registry.values()
            if d.task == task and d.precision == precision
        ]

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def get_recommended(
        self,
        task: ModelTask,
        validator: Optional[ModelValidator] = None,
    ) -> Optional[ModelDescriptor]:
        """
        Return the best model for a task.

        If a validator is supplied, only models that pass validation are
        considered.  Among valid candidates, the model with the lowest
        `max_latency_ms` (or lowest `ops_G` if no latency target) is returned.
        Returns None if no suitable model is found.
        """
        candidates = self.filter_by_task(task)
        if not candidates:
            return None

        if validator is not None:
            candidates = [d for d in candidates if validator.validate(d).passed]

        if not candidates:
            return None

        # Prefer models with an explicit latency target; then by ops_G
        def _sort_key(d: ModelDescriptor) -> tuple[float, float]:
            latency = d.max_latency_ms if d.max_latency_ms > 0 else float("inf")
            return (latency, d.ops_G)

        return min(candidates, key=_sort_key)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, model_id: str) -> bool:
        """Support `'name@version' in catalog` syntax."""
        parts = model_id.split("@", 1)
        if len(parts) != 2:
            return False
        return (parts[0], parts[1]) in self._registry

    def __repr__(self) -> str:
        return f"ModelCatalog(models={len(self._registry)})"
