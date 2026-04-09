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


class ExplodingGradientDetector(FailureDetector):
    """
    Flags iterations where ||Δoutput|| / ||Δinput|| exceeds a threshold.

    Call set_baseline(base_inputs, base_output) before the run loop.
    Without a baseline this detector is silently disabled.

    The ratio measures how much the output magnitude changed relative to the
    input magnitude change. A very high ratio indicates the model is amplifying
    small input perturbations into large output swings — a sign of numerical
    instability that can destroy downstream risk metrics.

    Severity: critical — explosive sensitivity means the model's outputs are
    dominated by numerical noise rather than the underlying financial signal.
    """

    FAILURE_TYPE = "nan_inf"
    DEFAULT_THRESHOLD = 100.0
    _EPSILON = 1e-12

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold
        self._base_inputs: dict | None = None
        self._base_output: Any = None

    def set_baseline(self, base_inputs: dict, base_output: Any) -> None:
        """Store the unperturbed inputs and output for ratio computation."""
        self._base_inputs = base_inputs
        self._base_output = base_output

    def check(self, inputs: dict, output: Any, iteration: int) -> Finding | None:
        if self._base_inputs is None:
            return None

        input_delta = self._vector_norm(inputs) - self._vector_norm(self._base_inputs)
        output_delta = self._array_norm(output) - self._array_norm(self._base_output)

        ratio = abs(output_delta) / (abs(input_delta) + self._EPSILON)

        if ratio <= self.threshold:
            return None

        return Finding(
            failure_type=self.FAILURE_TYPE,
            severity="critical",
            message=(
                f"Exploding gradient detected: output change / input change = {ratio:.1f}x "
                f"(threshold = {self.threshold}x). Small input perturbations produce "
                f"disproportionately large output swings, indicating numerical instability."
            ),
            iteration=iteration,
        )

    @staticmethod
    def _vector_norm(inputs: dict) -> float:
        """Flatten all numeric values in inputs dict to a single L2 norm."""
        sum_sq = 0.0
        for v in inputs.values():
            if isinstance(v, np.ndarray):
                n = float(np.linalg.norm(v.ravel()))
                sum_sq += n * n
            else:
                try:
                    f = float(v)
                    sum_sq += f * f
                except (TypeError, ValueError):
                    pass
        return math.sqrt(sum_sq)

    @staticmethod
    def _array_norm(output: Any) -> float:
        """Return L2 norm of output (scalar or array)."""
        if isinstance(output, np.ndarray):
            return float(np.linalg.norm(output.ravel()))
        try:
            return abs(float(output))
        except (TypeError, ValueError):
            return 0.0


class RegimeShiftDetector(FailureDetector):
    """
    Flags when output deviates more than z_threshold standard deviations
    from the running mean across all iterations seen so far.

    Stateful: accumulates output norms. Call reset() before each run.
    Not triggered until min_history observations collected.

    Severity: warning — a statistical outlier relative to the run's own
    distribution. Indicates a possible regime shift but not an outright
    numerical failure like NaN/Inf.
    """

    FAILURE_TYPE = "nan_inf"
    DEFAULT_Z_THRESHOLD = 4.0
    DEFAULT_MIN_HISTORY = 30
    _EPSILON = 1e-12

    def __init__(
        self,
        z_threshold: float = DEFAULT_Z_THRESHOLD,
        min_history: int = DEFAULT_MIN_HISTORY,
    ) -> None:
        self.z_threshold = z_threshold
        self.min_history = min_history
        self._history: list[float] = []

    def reset(self) -> None:
        """Clear accumulated history."""
        self._history.clear()

    def check(self, inputs: dict, output: Any, iteration: int) -> Finding | None:
        norm = self._output_norm(output)
        self._history.append(norm)

        if len(self._history) < self.min_history:
            return None

        mean = np.mean(self._history)
        std = np.std(self._history)

        if std < self._EPSILON:
            return None

        z = abs(norm - mean) / std

        if z <= self.z_threshold:
            return None

        return Finding(
            failure_type=self.FAILURE_TYPE,
            severity="warning",
            message=(
                f"Regime shift detected: Z-score={z:.2f} exceeds threshold={self.z_threshold}. "
                f"Output norm={norm:.4g} deviates significantly from running mean={mean:.4g} "
                f"(std={std:.4g}) over {len(self._history)} iterations."
            ),
            iteration=iteration,
        )

    @staticmethod
    def _output_norm(output: Any) -> float:
        """Return L2 norm of output (ndarray), abs value (scalar), or 0.0."""
        if isinstance(output, np.ndarray):
            return float(np.linalg.norm(output.ravel()))
        try:
            return abs(float(output))
        except (TypeError, ValueError):
            return 0.0
