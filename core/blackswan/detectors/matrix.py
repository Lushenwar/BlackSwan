"""
Detectors for matrix-specific numerical failures: non-PSD and ill-conditioning.
"""

from typing import Any

import numpy as np

from .base import FailureDetector, Finding, TriggerDisclosure


class MatrixPSDDetector(FailureDetector):
    """
    Detects loss of positive semi-definiteness in covariance/correlation matrices.

    Computes the minimum eigenvalue of any square 2D NumPy array in the output.
    Fires when min(eigenvalue) < -epsilon, which means the matrix is no longer
    PSD and cannot serve as a valid covariance matrix (Cholesky will fail,
    portfolio variance can go negative).

    Traverses dict outputs to find the first offending matrix.

    Severity: critical — a non-PSD covariance matrix corrupts all risk metrics
    derived from it.
    """

    FAILURE_TYPE = "non_psd_matrix"
    DEFAULT_EPSILON = 1e-10

    def __init__(self, epsilon: float = DEFAULT_EPSILON) -> None:
        self.epsilon = epsilon

    def check(self, inputs: dict, output: Any, iteration: int) -> Finding | None:
        result = self._check_value(output)
        if result is None:
            return None

        mat, min_eigval = result
        return Finding(
            failure_type=self.FAILURE_TYPE,
            severity="critical",
            message=(
                f"Matrix ({mat.shape[0]}x{mat.shape[1]}) is not positive semi-definite. "
                f"Minimum eigenvalue: {min_eigval:.4e}. "
                "A non-PSD covariance matrix will cause Cholesky decomposition to fail "
                "and portfolio variance to go negative."
            ),
            iteration=iteration,
            trigger_disclosure=TriggerDisclosure(
                detector_name="MatrixPSDDetector",
                observed_value=round(min_eigval, 10),
                threshold=-self.epsilon,
                comparison="<",
                explanation=(
                    f"Minimum eigenvalue {min_eigval:.4e} fell below threshold "
                    f"{-self.epsilon:.0e}. A PSD matrix requires all eigenvalues >= 0."
                ),
            ),
        )

    def _check_value(self, value: Any) -> tuple[np.ndarray, float] | None:
        """Return (matrix, min_eigenvalue) for the first non-PSD matrix found, else None."""
        if isinstance(value, dict):
            for v in value.values():
                result = self._check_value(v)
                if result is not None:
                    return result
            return None

        if not isinstance(value, np.ndarray):
            return None
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            return None  # not a square matrix — skip

        try:
            eigvals = np.linalg.eigvalsh(value)
        except np.linalg.LinAlgError:
            return None

        min_eigval = float(eigvals.min())
        if min_eigval < -self.epsilon:
            return (value, min_eigval)

        return None


class ConditionNumberDetector(FailureDetector):
    """
    Detects ill-conditioned matrices where inversion would be numerically unreliable.

    Computes np.linalg.cond() on every square 2D array in the output and fires
    when the condition number exceeds the threshold (default 1e12). Matrices
    with condition numbers this high amplify floating-point errors by ~1e12x
    during inversion, making the result essentially garbage.

    Severity: warning (not critical) — the inversion may succeed but with
    significant loss of precision. The caller should regularise or avoid
    inverting the matrix.
    """

    FAILURE_TYPE = "ill_conditioned_matrix"
    DEFAULT_THRESHOLD = 1e12

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold

    def check(self, inputs: dict, output: Any, iteration: int) -> Finding | None:
        result = self._check_value(output)
        if result is None:
            return None

        mat, cond = result
        return Finding(
            failure_type=self.FAILURE_TYPE,
            severity="warning",
            message=(
                f"Matrix ({mat.shape[0]}x{mat.shape[1]}) condition number {cond:.3e} "
                f"exceeds threshold {self.threshold:.0e}. "
                "Inversion will amplify floating-point errors by this factor, "
                "producing an unreliable result."
            ),
            iteration=iteration,
            trigger_disclosure=TriggerDisclosure(
                detector_name="ConditionNumberDetector",
                observed_value=round(cond, 3),
                threshold=self.threshold,
                comparison=">",
                explanation=(
                    f"Condition number {cond:.3e} exceeded threshold {self.threshold:.0e}. "
                    "Matrices with high condition numbers amplify floating-point errors "
                    "proportionally during inversion."
                ),
            ),
        )

    def _check_value(self, value: Any) -> tuple[np.ndarray, float] | None:
        """Return (matrix, condition_number) for the first ill-conditioned matrix found, else None."""
        if isinstance(value, dict):
            for v in value.values():
                result = self._check_value(v)
                if result is not None:
                    return result
            return None

        if not isinstance(value, np.ndarray):
            return None
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            return None

        try:
            cond = float(np.linalg.cond(value))
        except np.linalg.LinAlgError:
            return None

        if cond > self.threshold:
            return (value, cond)

        return None
