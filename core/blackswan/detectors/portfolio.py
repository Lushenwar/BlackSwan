"""
Detectors for portfolio-level plausibility: output bounds checking.
"""

from typing import Any

import numpy as np

from .base import FailureDetector, Finding


# Default plausibility bounds per recognised output key.
# Keys not in this map are silently ignored — no false positives on unknown fields.
_DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "weights": (-2.0, 2.0),   # leveraged long/short; beyond ±2x is implausible
    "var":     (0.0, 1.0),    # Value-at-Risk expressed as a fraction [0, 1]
    "sharpe":  (-5.0, 5.0),   # Sharpe ratio; beyond ±5 is implausible in practice
}


class BoundsDetector(FailureDetector):
    """
    Flags output values that exceed configurable plausibility bounds.

    Checks known keys in a dict output (weights, var, sharpe) against their
    expected ranges. Unknown keys are silently skipped — this detector never
    fires on fields it doesn't recognise, keeping false-positive rate at zero
    for custom outputs.

    Bounds are configurable via the constructor so scenario YAML can tighten
    or widen them. The defaults represent sane outer limits for a V1 portfolio
    risk model.

    Severity: warning — values outside bounds are suspicious but could
    theoretically be correct (e.g. a stress scenario producing extreme Sharpe).
    The finding surfaces the anomaly for human review.
    """

    FAILURE_TYPE = "bounds_exceeded"

    def __init__(self, bounds: dict[str, tuple[float, float]] | None = None) -> None:
        self.bounds = bounds if bounds is not None else dict(_DEFAULT_BOUNDS)

    def check(self, inputs: dict, output: Any, iteration: int) -> Finding | None:
        if not isinstance(output, dict):
            return None

        for key, (lo, hi) in self.bounds.items():
            value = output.get(key)
            if value is None:
                continue

            violation = self._find_violation(value, lo, hi)
            if violation is not None:
                bad_val, direction = violation
                return Finding(
                    failure_type=self.FAILURE_TYPE,
                    severity="warning",
                    message=(
                        f"Output '{key}' value {bad_val:.4g} is {direction} "
                        f"the plausible range [{lo}, {hi}]."
                    ),
                    iteration=iteration,
                )

        return None

    def _find_violation(
        self, value: Any, lo: float, hi: float
    ) -> tuple[float, str] | None:
        """
        Return (offending_value, 'above'|'below') if value is outside [lo, hi],
        else None.
        """
        if isinstance(value, np.ndarray):
            if np.any(value > hi):
                idx = int(np.argmax(value > hi))
                return (float(value.flat[idx]), "above")
            if np.any(value < lo):
                idx = int(np.argmax(value < lo))
                return (float(value.flat[idx]), "below")
            return None

        try:
            v = float(value)
        except (TypeError, ValueError):
            return None

        if v > hi:
            return (v, "above")
        if v < lo:
            return (v, "below")
        return None
