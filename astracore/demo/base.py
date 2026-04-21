"""Demo dispatch + shared types.

A demo handler takes (zoo_entry, onnx_path, input_spec, backend_name)
and returns a :class:`DemoResult` with both raw numeric output and a
human-readable summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class DemoError(RuntimeError):
    """Raised when a demo can't run (missing file, bad input, etc)."""


@dataclass
class DemoResult:
    model: str
    family: str
    backend: str
    input_source: str                     # filename or "canned:<description>"
    wall_ms: float
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    raw_shape: Optional[List[int]] = None
    ok: bool = True
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


_HANDLERS: Dict[str, Callable] = {}


def register_demo_family(family: str) -> Callable[[Callable], Callable]:
    """Register a demo handler for a zoo family name."""
    def _dec(fn: Callable) -> Callable:
        _HANDLERS[family] = fn
        return fn
    return _dec


def get_demo_handler(family: str) -> Callable:
    if family not in _HANDLERS:
        raise DemoError(
            f"No demo handler for family {family!r}. "
            f"Registered: {sorted(_HANDLERS)}"
        )
    return _HANDLERS[family]


def run_demo(zoo_entry, onnx_path: Path, *,
             input_spec: Optional[str] = None,
             backend_name: str = "onnxruntime",
             warmup: int = 0) -> DemoResult:
    """Route a zoo entry to its family-specific demo handler.

    Parameters
    ----------
    warmup
        Number of *extra* warmup runs before the timed call. ORT compiles
        kernels on first use, so a cold first call can be 50-400× slower
        than steady-state — pass ``warmup=3`` to publish a realistic
        per-inference number instead of the cold-start bump.
    """
    # Side-effect imports: register all handlers.
    from astracore.demo import vision_classifier  # noqa: F401
    from astracore.demo import vision_detector    # noqa: F401
    from astracore.demo import text_models        # noqa: F401

    handler = get_demo_handler(zoo_entry.family)
    try:
        for _ in range(max(0, warmup)):
            # Discard the warmup result; we only want the side-effect of
            # ORT's kernel compilation + cache warming.
            handler(zoo_entry, Path(onnx_path),
                    input_spec=input_spec, backend_name=backend_name)
        return handler(zoo_entry, Path(onnx_path),
                       input_spec=input_spec, backend_name=backend_name)
    except DemoError:
        raise
    except Exception as exc:
        return DemoResult(
            model=zoo_entry.name, family=zoo_entry.family,
            backend=backend_name,
            input_source=input_spec or "default",
            wall_ms=0.0, ok=False, error=repr(exc),
        )
