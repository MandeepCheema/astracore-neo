"""Plugin registry — the extension contract for downstream packages.

Three parallel registries:
  * ``ops`` — ONNX op-type string → handler(node, graph) -> graph node
  * ``quantisers`` — calibration-strategy name → ``Quantiser`` subclass
  * ``backends`` — target-silicon name → ``Backend`` subclass

Each has a ``@register_*`` decorator (for in-process registration, e.g.
tests) and an entry-point loader (for installed packages). Both paths
write into the same underlying dict; last-writer-wins.

Entry-points are loaded lazily on first ``get_*`` / ``list_*`` call so
importing ``astracore`` stays cheap.
"""

from __future__ import annotations

from typing import Callable, Dict, Type, TypeVar, Generic, List, Optional
import importlib.metadata
import logging

_log = logging.getLogger(__name__)

T = TypeVar("T")


class _Registry(Generic[T]):
    """A single named registry plus its entry-point group."""

    def __init__(self, name: str, entry_point_group: str):
        self._name = name
        self._group = entry_point_group
        self._items: Dict[str, T] = {}
        self._entry_points_loaded = False

    def register(self, key: str, item: T) -> T:
        self._items[key] = item
        return item

    def get(self, key: str) -> T:
        self._ensure_entry_points()
        if key not in self._items:
            raise KeyError(
                f"No {self._name} registered for key {key!r}. "
                f"Registered: {sorted(self._items)}"
            )
        return self._items[key]

    def has(self, key: str) -> bool:
        self._ensure_entry_points()
        return key in self._items

    def list(self) -> List[str]:
        self._ensure_entry_points()
        return sorted(self._items)

    def _ensure_entry_points(self) -> None:
        if self._entry_points_loaded:
            return
        self._entry_points_loaded = True
        try:
            eps = importlib.metadata.entry_points()
            # Python 3.10 / 3.12 API: select by group.
            if hasattr(eps, "select"):
                group_eps = eps.select(group=self._group)
            else:
                group_eps = eps.get(self._group, [])
            for ep in group_eps:
                try:
                    item = ep.load()
                    if ep.name not in self._items:
                        # In-process @register_* wins over entry-points.
                        self._items[ep.name] = item
                except Exception as exc:  # pragma: no cover - plugin fault
                    _log.warning(
                        "Failed to load %s plugin %r: %s",
                        self._name, ep.name, exc,
                    )
        except Exception as exc:  # pragma: no cover
            _log.debug("Entry-point discovery failed for %s: %s",
                      self._name, exc)


# ---------------------------------------------------------------------------
# Module-level registries
# ---------------------------------------------------------------------------

_ops: _Registry[Callable] = _Registry("op", "astracore.ops")
_quantisers: _Registry[Type] = _Registry("quantiser", "astracore.quantisers")
_backends: _Registry[Type] = _Registry("backend", "astracore.backends")


# ---------------------------------------------------------------------------
# Decorators (in-process registration — for tests and user code)
# ---------------------------------------------------------------------------

def register_op(op_type: str) -> Callable[[Callable], Callable]:
    """Register an ONNX op handler.

    Usage::

        @astracore.register_op("MyCustomOp")
        def handle_my_custom_op(node, graph):
            ...
    """
    def _decorator(fn: Callable) -> Callable:
        _ops.register(op_type, fn)
        return fn
    return _decorator


def register_quantiser(name: str) -> Callable[[Type], Type]:
    """Register a calibration-strategy class.

    Usage::

        @astracore.register_quantiser("percentile_98")
        class Percentile98(Quantiser):
            ...
    """
    def _decorator(cls: Type) -> Type:
        _quantisers.register(name, cls)
        return cls
    return _decorator


def register_backend(name: str) -> Callable[[Type], Type]:
    """Register a target-silicon backend class.

    Usage::

        @astracore.register_backend("tensorrt")
        class TensorRTBackend(Backend):
            ...
    """
    def _decorator(cls: Type) -> Type:
        _backends.register(name, cls)
        return cls
    return _decorator


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_op(op_type: str) -> Callable:
    return _ops.get(op_type)


def get_quantiser(name: str) -> Type:
    return _quantisers.get(name)


def get_backend(name: str) -> Type:
    return _backends.get(name)


def list_ops() -> List[str]:
    return _ops.list()


def list_quantisers() -> List[str]:
    return _quantisers.list()


def list_backends() -> List[str]:
    return _backends.list()


# ---------------------------------------------------------------------------
# Test / debug
# ---------------------------------------------------------------------------

def _reset_for_tests() -> None:
    """Clear all registries. Test fixture only; do not call in production."""
    for reg in (_ops, _quantisers, _backends):
        reg._items.clear()
        reg._entry_points_loaded = False
