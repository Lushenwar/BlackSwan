"""
Detectors for general numerical failures: NaN, Inf, and division instability.
"""

import math
from typing import Any

import numpy as np

from .base import FailureDetector, Finding


class NaNInfDetector(FailureDetector):
    """
    Detects NaN or infinite values in any part of a function's output.

    Traverses scalars, NumPy arrays, and dicts of either. Returns a critical
    Finding on the first bad value found; returns None when the output is
    entirely finite.

    Severity: critical — NaN/Inf propagate silently through downstream
    calculations and corrupt every result that depends on them.
    """

    FAILURE_TYPE = "nan_inf"

    def check(self, inputs: dict, output: Any, iteration: int) -> Finding | None:
        bad = self._find_bad_value(output)
        if bad is None:
            return None

        return Finding(
            failure_type=self.FAILURE_TYPE,
            severity="critical",
            message=f"Output contains {bad}. NaN/Inf propagate silently and corrupt all downstream calculations.",
            iteration=iteration,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_bad_value(self, value: Any) -> str | None:
        """
        Recursively search `value` for any NaN or Inf.

        Returns a short description of the bad value found ('NaN', '+Inf',
        '-Inf'), or None if everything is finite.
        """
        if isinstance(value, dict):
            for v in value.values():
                result = self._find_bad_value(v)
                if result is not None:
                    return result
            return None

        if isinstance(value, np.ndarray):
            if np.any(np.isnan(value)):
                return "NaN"
            if np.any(np.isposinf(value)):
                return "+Inf"
            if np.any(np.isneginf(value)):
                return "-Inf"
            return None

        # Scalar: Python float, numpy scalar, or int
        try:
            if math.isnan(value):
                return "NaN"
            if math.isinf(value):
                return "+Inf" if value > 0 else "-Inf"
        except (TypeError, ValueError):
            # Non-numeric types (str, None, etc.) cannot be NaN/Inf
            pass

        return None


class DivisionStabilityDetector(FailureDetector):
    """
    Detects denominators approaching zero before a division is performed.

    The runner is expected to pass the denominator value(s) in inputs under
    the key 'denominator'. The detector fires when |denominator| <= epsilon,
    indicating the division result would be numerically unreliable or infinite.

    Design note: the detector inspects inputs, not output. This lets it flag
    danger before the division happens so the runner can record the iteration
    without executing an unsafe division.

    Severity: critical — near-zero denominators produce values that are orders
    of magnitude too large, silently destroying downstream risk metrics.
    """

    FAILURE_TYPE = "division_by_zero"
    DEFAULT_EPSILON = 1e-10

    def __init__(self, epsilon: float = DEFAULT_EPSILON) -> None:
        self.epsilon = epsilon

    def check(self, inputs: dict, output: Any, iteration: int) -> Finding | None:
        denom = inputs.get("denominator")
        if denom is None:
            return None

        if isinstance(denom, np.ndarray):
            bad_mask = np.abs(denom) <= self.epsilon
            if not np.any(bad_mask):
                return None
            bad_val = float(denom[bad_mask][0])
        else:
            try:
                bad_val = float(denom)
            except (TypeError, ValueError):
                return None
            if abs(bad_val) > self.epsilon:
                return None

        return Finding(
            failure_type=self.FAILURE_TYPE,
            severity="critical",
            message=(
                f"Denominator value {bad_val:.3e} is within epsilon ({self.epsilon:.0e}) of zero. "
                "Division would produce an unreliable or infinite result."
            ),
            iteration=iteration,
        )
