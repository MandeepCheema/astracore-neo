"""Quantiser protocol — the calibration-strategy contract.

Thin wrapper around ``tools.npu_ref.quantiser``. Exposes a single class
``Quantiser`` that downstream packages subclass to plug in their own
calibration behaviour without forking.

Built-in recipe:
  * per-channel max-abs for weights
  * per-tensor percentile-99.9999 for activations
  * NPU score threshold 0.20 (asymmetric vs 0.25 ORT)
  * 100-image calibration set

Override by subclassing + ``@register_quantiser("your-name")``.
"""

from __future__ import annotations

from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class QuantiserConfig:
    """Knobs every quantiser understands. Subclasses may add their own."""

    precision: str = "INT8"        # INT8 | INT4 | INT2
    weight_calibration: str = "per_channel_max_abs"
    activation_calibration: str = "percentile_99_9999"
    calibration_images: int = 100
    score_threshold: float = 0.20
    extra: Optional[dict] = None


class Quantiser:
    """Base class for calibration strategies.

    Subclasses override :meth:`calibrate` (compute scales / zero-points
    from a calibration set) and optionally :meth:`fake_quantise`
    (produce a fake-quantised graph for accuracy evaluation).
    """

    config_cls = QuantiserConfig

    def __init__(self, config: Optional[QuantiserConfig] = None):
        self.config = config or self.config_cls()

    # ------------------------------------------------------------------
    # Subclass API
    # ------------------------------------------------------------------

    def calibrate(self, graph: Any, calibration_set) -> Any:
        """Return a quantised graph + scale/zp table.

        Default implementation delegates to the reference recipe in
        ``tools/npu_ref/quantiser.py``.
        """
        from tools.npu_ref.quantiser import quantise_model  # lazy import
        return quantise_model(graph, calibration_set,
                              calib_percentile=self._percentile())

    def _percentile(self) -> float:
        """Map the string ``activation_calibration`` knob to the percentile."""
        mapping = {
            "max_abs": 100.0,
            "percentile_99": 99.0,
            "percentile_9999": 99.99,
            "percentile_99999": 99.999,
            "percentile_999999": 99.9999,
            "percentile_99_9999": 99.9999,
        }
        return mapping.get(self.config.activation_calibration, 99.9999)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__} "
                f"precision={self.config.precision} "
                f"activation={self.config.activation_calibration}>")
