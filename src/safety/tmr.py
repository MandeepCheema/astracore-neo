"""
AstraCore Neo — TMR (Triple Modular Redundancy) voter simulation.

Models the TMR voting logic used in ASIL-D safety-critical paths:
  - Three independent compute lanes produce the same result
  - Majority voter selects the correct output
  - Disagreement patterns are logged for fault isolation
  - Supports integer, float, and numpy array voting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

import numpy as np

from .exceptions import TMRError


@dataclass
class TMRResult:
    """Result of a TMR vote."""
    voted_value: Any              # winning value from majority vote
    lane_a: Any
    lane_b: Any
    lane_c: Any
    agreement: bool               # True if at least 2 lanes agreed
    faulty_lane: Optional[str]    # "A", "B", "C", or None if all agree
    vote_count: int               # which majority won (2 or 3)


class TMRVoter:
    """
    Triple Modular Redundancy voter.

    Three lanes (A, B, C) independently compute the same result.
    The voter selects the majority output and flags the disagreeing lane.

    Usage::

        voter = TMRVoter()
        result = voter.vote(42, 42, 43)
        assert result.voted_value == 42
        assert result.faulty_lane == "C"

    For numpy arrays, comparison is element-wise with a configurable tolerance.
    """

    def __init__(self, float_atol: float = 1e-6) -> None:
        self._float_atol = float_atol
        self._total_votes: int = 0
        self._fault_counts: dict[str, int] = {"A": 0, "B": 0, "C": 0}
        self._triple_disagreements: int = 0

    def _equal(self, x: Any, y: Any) -> bool:
        """Compare two values (handles scalars, floats, and numpy arrays)."""
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            if x.shape != y.shape:
                return False
            if np.issubdtype(x.dtype, np.floating):
                return bool(np.allclose(x, y, atol=self._float_atol))
            return bool(np.array_equal(x, y))
        if isinstance(x, float) or isinstance(y, float):
            return abs(float(x) - float(y)) <= self._float_atol
        return x == y

    def vote(self, lane_a: Any, lane_b: Any, lane_c: Any) -> TMRResult:
        """
        Perform majority vote across three lanes.

        Returns TMRResult with the winning value and fault information.
        Raises TMRError if all three lanes disagree (no majority possible).
        """
        self._total_votes += 1

        ab = self._equal(lane_a, lane_b)
        ac = self._equal(lane_a, lane_c)
        bc = self._equal(lane_b, lane_c)

        if ab and ac:
            # All three agree
            return TMRResult(
                voted_value=lane_a,
                lane_a=lane_a, lane_b=lane_b, lane_c=lane_c,
                agreement=True, faulty_lane=None, vote_count=3,
            )

        if ab:
            # A and B agree, C is faulty
            self._fault_counts["C"] += 1
            return TMRResult(
                voted_value=lane_a,
                lane_a=lane_a, lane_b=lane_b, lane_c=lane_c,
                agreement=True, faulty_lane="C", vote_count=2,
            )

        if ac:
            # A and C agree, B is faulty
            self._fault_counts["B"] += 1
            return TMRResult(
                voted_value=lane_a,
                lane_a=lane_a, lane_b=lane_b, lane_c=lane_c,
                agreement=True, faulty_lane="B", vote_count=2,
            )

        if bc:
            # B and C agree, A is faulty
            self._fault_counts["A"] += 1
            return TMRResult(
                voted_value=lane_b,
                lane_a=lane_a, lane_b=lane_b, lane_c=lane_c,
                agreement=True, faulty_lane="A", vote_count=2,
            )

        # All three disagree — unrecoverable
        self._triple_disagreements += 1
        raise TMRError(
            f"TMR triple disagreement: lane_a={lane_a!r}, "
            f"lane_b={lane_b!r}, lane_c={lane_c!r}"
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def fault_count(self, lane: str) -> int:
        """Return number of faults detected on a specific lane ('A', 'B', or 'C')."""
        if lane not in self._fault_counts:
            raise ValueError(f"Lane must be 'A', 'B', or 'C', got {lane!r}")
        return self._fault_counts[lane]

    @property
    def total_votes(self) -> int:
        return self._total_votes

    @property
    def triple_disagreements(self) -> int:
        return self._triple_disagreements

    def reset_counters(self) -> None:
        self._total_votes = 0
        self._fault_counts = {"A": 0, "B": 0, "C": 0}
        self._triple_disagreements = 0

    def __repr__(self) -> str:
        return (
            f"TMRVoter(votes={self._total_votes}, "
            f"faults=A:{self._fault_counts['A']} "
            f"B:{self._fault_counts['B']} "
            f"C:{self._fault_counts['C']})"
        )
