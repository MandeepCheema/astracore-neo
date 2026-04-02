"""
AstraCore Neo — Fault Predictor simulation.

Models a lightweight predictive fault detection engine:
  - Tracks per-metric time series (ECC rate, temperature, voltage, utilization)
  - Detects anomalies: threshold breach, trend acceleration, sudden spike
  - Confidence-scored fault predictions with escalation levels
  - Rolling statistics (mean, std dev, trend slope) using sliding window
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import math

from .exceptions import FaultPredictorError


class FaultRisk(Enum):
    NONE     = auto()   # metric is healthy
    LOW      = auto()   # mild anomaly, watch
    MEDIUM   = auto()   # notable anomaly, investigate
    HIGH     = auto()   # strong signal, act soon
    CRITICAL = auto()   # imminent fault predicted


@dataclass
class MetricConfig:
    """Configuration for a tracked metric."""
    name: str
    warning_threshold: float
    critical_threshold: float
    window_size: int = 32          # rolling window for stats
    spike_std_multiplier: float = 3.0   # value > mean + N*std is a spike


@dataclass
class FaultPrediction:
    """Result of a fault prediction evaluation."""
    metric: str
    risk: FaultRisk
    current_value: float
    mean: float
    std_dev: float
    trend_slope: float             # positive = rising
    confidence: float              # 0–1
    reason: str


class MetricTracker:
    """
    Tracks a single metric and predicts fault risk.
    """

    def __init__(self, config: MetricConfig) -> None:
        self._cfg = config
        self._window: deque[float] = deque(maxlen=config.window_size)
        self._total_samples: int = 0

    def push(self, value: float) -> FaultPrediction:
        """Add a new sample and return current fault prediction."""
        self._window.append(value)
        self._total_samples += 1
        return self._evaluate(value)

    def _stats(self) -> tuple[float, float]:
        """Return (mean, std_dev) of window. Returns (value, 0) for single sample."""
        data = list(self._window)
        n = len(data)
        if n == 0:
            return 0.0, 0.0
        mean = sum(data) / n
        if n == 1:
            return mean, 0.0
        variance = sum((x - mean) ** 2 for x in data) / n
        return mean, math.sqrt(variance)

    def _slope(self) -> float:
        """Linear trend slope over the window."""
        data = list(self._window)
        n = len(data)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(data) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(data))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den != 0 else 0.0

    def _evaluate(self, value: float) -> FaultPrediction:
        cfg = self._cfg
        mean, std = self._stats()
        slope = self._slope()

        risk = FaultRisk.NONE
        confidence = 0.0
        reason = "nominal"

        # Threshold breach
        if value >= cfg.critical_threshold:
            risk = FaultRisk.CRITICAL
            confidence = 0.95
            reason = f"value {value:.2f} >= critical threshold {cfg.critical_threshold:.2f}"
        elif value >= cfg.warning_threshold:
            # Scale risk by how far above warning we are
            ratio = (value - cfg.warning_threshold) / max(
                cfg.critical_threshold - cfg.warning_threshold, 1e-9
            )
            if ratio >= 0.7:
                risk = FaultRisk.HIGH
                confidence = 0.80
            elif ratio >= 0.3:
                risk = FaultRisk.MEDIUM
                confidence = 0.60
            else:
                risk = FaultRisk.LOW
                confidence = 0.40
            reason = f"value {value:.2f} above warning {cfg.warning_threshold:.2f}"

        # Spike detection (only if we have enough history)
        if len(self._window) >= 4 and std > 0:
            z_score = (value - mean) / std
            if z_score > cfg.spike_std_multiplier and risk.value < FaultRisk.HIGH.value:
                risk = FaultRisk.MEDIUM
                confidence = max(confidence, 0.55)
                reason = f"spike detected: z={z_score:.1f}"

        # Rising trend escalation (only if slope is significant)
        if slope > 0 and len(self._window) >= cfg.window_size // 2:
            # If trend would reach critical in <window samples, escalate
            if std > 0:
                samples_to_critical = (cfg.critical_threshold - value) / max(slope, 1e-9)
                if 0 < samples_to_critical < cfg.window_size and risk.value < FaultRisk.HIGH.value:
                    risk = FaultRisk.HIGH
                    confidence = max(confidence, 0.70)
                    reason = f"rising trend: slope={slope:.3f}, ~{samples_to_critical:.0f} samples to critical"

        return FaultPrediction(
            metric=cfg.name,
            risk=risk,
            current_value=value,
            mean=mean,
            std_dev=std,
            trend_slope=slope,
            confidence=confidence,
            reason=reason,
        )

    @property
    def sample_count(self) -> int:
        return self._total_samples

    @property
    def config(self) -> MetricConfig:
        return self._cfg

    def window_data(self) -> list[float]:
        return list(self._window)


class FaultPredictor:
    """
    Multi-metric fault prediction engine.

    Tracks multiple chip health metrics and provides risk assessments.

    Usage::

        predictor = FaultPredictor()
        predictor.add_metric(MetricConfig("ecc_rate", warning_threshold=5.0, critical_threshold=20.0))
        predictor.push("ecc_rate", 8.0)
        prediction = predictor.latest("ecc_rate")
        assert prediction.risk == FaultRisk.LOW
    """

    def __init__(self) -> None:
        self._trackers: dict[str, MetricTracker] = {}
        self._latest: dict[str, FaultPrediction] = {}

    def add_metric(self, config: MetricConfig) -> None:
        """Register a new metric for tracking."""
        if config.name in self._trackers:
            raise FaultPredictorError(f"Metric '{config.name}' already registered")
        self._trackers[config.name] = MetricTracker(config)

    def remove_metric(self, name: str) -> None:
        if name not in self._trackers:
            raise FaultPredictorError(f"Metric '{name}' not registered")
        del self._trackers[name]
        self._latest.pop(name, None)

    def push(self, metric: str, value: float) -> FaultPrediction:
        """Push a new value for a metric. Returns the current prediction."""
        if metric not in self._trackers:
            raise FaultPredictorError(f"Metric '{metric}' not registered")
        prediction = self._trackers[metric].push(value)
        self._latest[metric] = prediction
        return prediction

    def latest(self, metric: str) -> FaultPrediction:
        """Return the most recent prediction for a metric."""
        if metric not in self._latest:
            raise FaultPredictorError(f"No predictions yet for metric '{metric}'")
        return self._latest[metric]

    def highest_risk(self) -> Optional[FaultPrediction]:
        """Return the prediction with the highest risk across all metrics."""
        if not self._latest:
            return None
        return max(self._latest.values(), key=lambda p: p.risk.value)

    def any_high_risk(self) -> bool:
        return any(
            p.risk in (FaultRisk.HIGH, FaultRisk.CRITICAL)
            for p in self._latest.values()
        )

    def metric_names(self) -> list[str]:
        return list(self._trackers.keys())

    def tracker(self, name: str) -> MetricTracker:
        if name not in self._trackers:
            raise FaultPredictorError(f"Metric '{name}' not registered")
        return self._trackers[name]

    def __repr__(self) -> str:
        high = sum(
            1 for p in self._latest.values()
            if p.risk in (FaultRisk.HIGH, FaultRisk.CRITICAL)
        )
        return f"FaultPredictor(metrics={len(self._trackers)}, high_risk={high})"
