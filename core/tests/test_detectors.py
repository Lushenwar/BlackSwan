"""
Tests for all FailureDetector implementations.

Each detector requires:
  - 5 true-positive cases (must fire)
  - 5 true-negative cases (must stay silent)
"""

import pytest
import numpy as np

from blackswan.detectors.base import FailureDetector, Finding
from blackswan.detectors.numerical import NaNInfDetector, DivisionStabilityDetector
from blackswan.detectors.matrix import MatrixPSDDetector, ConditionNumberDetector
from blackswan.detectors.portfolio import BoundsDetector


# ---------------------------------------------------------------------------
# Finding dataclass contract
# ---------------------------------------------------------------------------

class TestFinding:
    def test_finding_has_required_fields(self):
        f = Finding(
            failure_type="nan_inf",
            severity="critical",
            message="NaN detected in output",
            iteration=7,
        )
        assert f.failure_type == "nan_inf"
        assert f.severity == "critical"
        assert f.message == "NaN detected in output"
        assert f.iteration == 7

    def test_finding_line_and_column_default_to_none(self):
        f = Finding(failure_type="nan_inf", severity="critical", message="x", iteration=0)
        assert f.line is None
        assert f.column is None

    def test_finding_accepts_explicit_line_and_column(self):
        f = Finding(
            failure_type="nan_inf", severity="critical", message="x",
            iteration=0, line=42, column=8,
        )
        assert f.line == 42
        assert f.column == 8


# ---------------------------------------------------------------------------
# FailureDetector abstract base class contract
# ---------------------------------------------------------------------------

class TestFailureDetectorBase:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            FailureDetector()

    def test_concrete_subclass_missing_check_cannot_be_instantiated(self):
        class BadDetector(FailureDetector):
            pass  # missing check()

        with pytest.raises(TypeError):
            BadDetector()

    def test_concrete_subclass_with_check_can_be_instantiated(self):
        class OkDetector(FailureDetector):
            def check(self, inputs, output, iteration):
                return None

        d = OkDetector()
        assert d.check({}, 0.0, 0) is None


# ---------------------------------------------------------------------------
# NaNInfDetector — TRUE POSITIVES (must fire)
# ---------------------------------------------------------------------------

class TestNaNInfDetectorFires:
    def setup_method(self):
        self.detector = NaNInfDetector()

    def test_fires_on_nan_scalar(self):
        """float('nan') output must trigger a Finding."""
        result = self.detector.check(inputs={}, output=float("nan"), iteration=1)
        assert result is not None
        assert result.failure_type == "nan_inf"
        assert result.severity == "critical"
        assert result.iteration == 1

    def test_fires_on_positive_inf_scalar(self):
        """Positive infinity must trigger a Finding."""
        result = self.detector.check(inputs={}, output=np.inf, iteration=2)
        assert result is not None
        assert result.failure_type == "nan_inf"

    def test_fires_on_negative_inf_scalar(self):
        """-Inf (e.g. from log(0)) must trigger a Finding."""
        result = self.detector.check(inputs={}, output=-np.inf, iteration=3)
        assert result is not None
        assert result.failure_type == "nan_inf"

    def test_fires_on_array_containing_nan(self):
        """A single NaN anywhere in a NumPy array must trigger a Finding."""
        arr = np.array([1.0, 2.0, np.nan, 4.0])
        result = self.detector.check(inputs={}, output=arr, iteration=4)
        assert result is not None
        assert result.failure_type == "nan_inf"

    def test_fires_on_dict_output_with_inf_value(self):
        """Inf nested inside a dict value (e.g. portfolio result dict) must fire."""
        output = {
            "weights": np.array([0.3, 0.4, 0.3]),
            "var": np.inf,
        }
        result = self.detector.check(inputs={}, output=output, iteration=5)
        assert result is not None
        assert result.failure_type == "nan_inf"


# ---------------------------------------------------------------------------
# NaNInfDetector — TRUE NEGATIVES (must stay silent)
# ---------------------------------------------------------------------------

class TestNaNInfDetectorSilent:
    def setup_method(self):
        self.detector = NaNInfDetector()

    def test_silent_on_normal_float(self):
        """Ordinary finite float must not fire."""
        result = self.detector.check(inputs={}, output=0.42, iteration=1)
        assert result is None

    def test_silent_on_very_large_but_finite_float(self):
        """1e308 is the largest representable finite float64 — must not fire."""
        result = self.detector.check(inputs={}, output=1e308, iteration=2)
        assert result is None

    def test_silent_on_all_finite_array(self):
        """Array of normal values must not fire."""
        arr = np.array([0.1, 0.2, 0.3, 0.4])
        result = self.detector.check(inputs={}, output=arr, iteration=3)
        assert result is None

    def test_silent_on_negative_finite_values(self):
        """Negative finite numbers (e.g. short positions, losses) must not fire."""
        arr = np.array([-1.0, -999.0, -0.001])
        result = self.detector.check(inputs={}, output=arr, iteration=4)
        assert result is None

    def test_silent_on_zero_array(self):
        """All-zero array must not fire — zero is a valid finite value."""
        arr = np.zeros(10)
        result = self.detector.check(inputs={}, output=arr, iteration=5)
        assert result is None


# ---------------------------------------------------------------------------
# DivisionStabilityDetector — TRUE POSITIVES (must fire)
# ---------------------------------------------------------------------------

class TestDivisionStabilityDetectorFires:
    def setup_method(self):
        self.detector = DivisionStabilityDetector()

    def test_fires_when_denominator_is_zero(self):
        """Exact zero denominator must fire."""
        result = self.detector.check(
            inputs={"denominator": 0.0}, output=None, iteration=1
        )
        assert result is not None
        assert result.failure_type == "division_by_zero"
        assert result.severity == "critical"

    def test_fires_when_denominator_is_very_small_positive(self):
        """1e-11 is below the default epsilon of 1e-10 — must fire."""
        result = self.detector.check(
            inputs={"denominator": 1e-11}, output=None, iteration=2
        )
        assert result is not None
        assert result.failure_type == "division_by_zero"

    def test_fires_when_denominator_is_very_small_negative(self):
        """-1e-11 is within epsilon of zero — must fire."""
        result = self.detector.check(
            inputs={"denominator": -1e-11}, output=None, iteration=3
        )
        assert result is not None
        assert result.failure_type == "division_by_zero"

    def test_fires_on_array_denominator_with_one_near_zero(self):
        """Array where one element is near zero must fire."""
        result = self.detector.check(
            inputs={"denominator": np.array([1.0, 2.0, 5e-12])}, output=None, iteration=4
        )
        assert result is not None
        assert result.failure_type == "division_by_zero"

    def test_fires_exactly_at_epsilon_boundary(self):
        """Value equal to epsilon must fire (boundary inclusive)."""
        result = self.detector.check(
            inputs={"denominator": 1e-10}, output=None, iteration=5
        )
        assert result is not None
        assert result.failure_type == "division_by_zero"


# ---------------------------------------------------------------------------
# DivisionStabilityDetector — TRUE NEGATIVES (must stay silent)
# ---------------------------------------------------------------------------

class TestDivisionStabilityDetectorSilent:
    def setup_method(self):
        self.detector = DivisionStabilityDetector()

    def test_silent_when_denominator_is_one(self):
        result = self.detector.check(
            inputs={"denominator": 1.0}, output=None, iteration=1
        )
        assert result is None

    def test_silent_when_denominator_is_small_but_above_epsilon(self):
        """1e-9 > 1e-10 epsilon — small but safe."""
        result = self.detector.check(
            inputs={"denominator": 1e-9}, output=None, iteration=2
        )
        assert result is None

    def test_silent_when_denominator_is_negative_and_safe(self):
        """-0.5 is far from zero — must not fire."""
        result = self.detector.check(
            inputs={"denominator": -0.5}, output=None, iteration=3
        )
        assert result is None

    def test_silent_when_no_denominator_key_in_inputs(self):
        """If inputs don't contain 'denominator', no failure to detect."""
        result = self.detector.check(
            inputs={"weights": np.array([0.5, 0.5])}, output=None, iteration=4
        )
        assert result is None

    def test_silent_on_array_all_elements_safe(self):
        """Array where every element is well above epsilon must not fire."""
        result = self.detector.check(
            inputs={"denominator": np.array([0.5, 1.0, 2.5, 0.01])}, output=None, iteration=5
        )
        assert result is None


# ---------------------------------------------------------------------------
# MatrixPSDDetector — TRUE POSITIVES (must fire)
# ---------------------------------------------------------------------------

class TestMatrixPSDDetectorFires:
    def setup_method(self):
        self.detector = MatrixPSDDetector()

    def _make_non_psd(self):
        """2x2 matrix with a small negative eigenvalue."""
        return np.array([[1.0, 1.1], [1.1, 1.0]])

    def test_fires_on_matrix_with_negative_eigenvalue(self):
        """[[1, 1.1],[1.1, 1]] has eigenvalues 2.1 and -0.1 — must fire."""
        result = self.detector.check(inputs={}, output=self._make_non_psd(), iteration=1)
        assert result is not None
        assert result.failure_type == "non_psd_matrix"
        assert result.severity == "critical"

    def test_fires_when_min_eigenvalue_just_below_zero(self):
        """Eigenvalue -1e-9 is negative — must fire even if tiny."""
        # Construct symmetric matrix with eigenvalue slightly below zero
        eigvals = np.array([2.0, 1.0, -1e-9])
        Q = np.eye(3)  # orthonormal (identity for simplicity)
        mat = Q @ np.diag(eigvals) @ Q.T
        result = self.detector.check(inputs={}, output=mat, iteration=2)
        assert result is not None
        assert result.failure_type == "non_psd_matrix"

    def test_fires_on_large_non_psd_matrix(self):
        """5x5 symmetric matrix with an explicit negative eigenvalue must fire.

        Note: uniform correlation matrices (all off-diag = r) are always PSD
        for r in (-1/(n-1), 1). To reliably produce a non-PSD matrix we use
        eigenvalue decomposition directly.
        """
        rng = np.random.default_rng(42)
        Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
        eigvals = np.array([3.0, 1.5, 0.8, 0.2, -0.05])
        mat = Q @ np.diag(eigvals) @ Q.T
        result = self.detector.check(inputs={}, output=mat, iteration=3)
        assert result is not None
        assert result.failure_type == "non_psd_matrix"

    def test_fires_on_dict_output_containing_non_psd_matrix(self):
        """Non-PSD matrix nested inside a result dict must fire."""
        output = {"cov_matrix": self._make_non_psd(), "var": 0.05}
        result = self.detector.check(inputs={}, output=output, iteration=4)
        assert result is not None
        assert result.failure_type == "non_psd_matrix"

    def test_fires_when_matrix_has_all_negative_eigenvalues(self):
        """Completely inverted matrix — clearly non-PSD."""
        mat = np.array([[-1.0, 0.0], [0.0, -2.0]])
        result = self.detector.check(inputs={}, output=mat, iteration=5)
        assert result is not None
        assert result.failure_type == "non_psd_matrix"


# ---------------------------------------------------------------------------
# MatrixPSDDetector — TRUE NEGATIVES (must stay silent)
# ---------------------------------------------------------------------------

class TestMatrixPSDDetectorSilent:
    def setup_method(self):
        self.detector = MatrixPSDDetector()

    def test_silent_on_identity_matrix(self):
        result = self.detector.check(inputs={}, output=np.eye(4), iteration=1)
        assert result is None

    def test_silent_on_well_conditioned_covariance_matrix(self):
        """Diagonal covariance — all eigenvalues positive."""
        cov = np.diag([0.04, 0.09, 0.01, 0.16])
        result = self.detector.check(inputs={}, output=cov, iteration=2)
        assert result is None

    def test_silent_on_psd_with_zero_eigenvalue(self):
        """Zero eigenvalue is valid for PSD (positive SEMI-definite)."""
        mat = np.array([[1.0, 1.0], [1.0, 1.0]])  # eigenvalues: 2 and 0
        result = self.detector.check(inputs={}, output=mat, iteration=3)
        assert result is None

    def test_silent_on_non_square_array(self):
        """Non-square arrays cannot be covariance matrices — skip silently."""
        result = self.detector.check(inputs={}, output=np.ones((3, 5)), iteration=4)
        assert result is None

    def test_silent_on_1d_array_output(self):
        """1D array is not a matrix — skip silently."""
        result = self.detector.check(inputs={}, output=np.array([1.0, 2.0, 3.0]), iteration=5)
        assert result is None


# ---------------------------------------------------------------------------
# ConditionNumberDetector — TRUE POSITIVES (must fire)
# ---------------------------------------------------------------------------

class TestConditionNumberDetectorFires:
    def setup_method(self):
        self.detector = ConditionNumberDetector()

    def _ill_conditioned(self):
        """Near-singular 2x2 matrix with condition number >> 1e12."""
        return np.array([[1.0, 1.0], [1.0, 1.0 + 1e-15]])

    def test_fires_on_near_singular_matrix(self):
        result = self.detector.check(inputs={}, output=self._ill_conditioned(), iteration=1)
        assert result is not None
        assert result.failure_type == "ill_conditioned_matrix"
        assert result.severity == "warning"

    def test_fires_on_matrix_with_one_tiny_eigenvalue(self):
        """One very small eigenvalue → enormous condition number."""
        mat = np.diag([1.0, 1.0, 1e-14])
        result = self.detector.check(inputs={}, output=mat, iteration=2)
        assert result is not None
        assert result.failure_type == "ill_conditioned_matrix"

    def test_fires_on_large_matrix_with_near_zero_eigenvalue(self):
        """6x6 matrix with one eigenvalue of 1e-14 → cond ≈ 5e14 >> 1e12.

        Note: uniform correlation matrices have cond = (1+(n-1)r)/(1-r).
        For r=0.9999, n=6 that is only ~6e4, well below 1e12. We use
        eigenvalue decomposition to guarantee the desired condition number.
        """
        rng = np.random.default_rng(0)
        Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        eigvals = np.array([5.0, 2.0, 1.0, 0.5, 0.1, 1e-14])
        mat = Q @ np.diag(eigvals) @ Q.T
        result = self.detector.check(inputs={}, output=mat, iteration=3)
        assert result is not None
        assert result.failure_type == "ill_conditioned_matrix"

    def test_fires_on_dict_with_ill_conditioned_matrix(self):
        output = {"cov_inv_input": self._ill_conditioned(), "weights": np.array([0.5, 0.5])}
        result = self.detector.check(inputs={}, output=output, iteration=4)
        assert result is not None
        assert result.failure_type == "ill_conditioned_matrix"

    def test_fires_exactly_at_threshold(self):
        """Condition number just above 1e12 must fire."""
        mat = np.diag([1.0, 1e-13])  # cond ≈ 1e13 > 1e12
        result = self.detector.check(inputs={}, output=mat, iteration=5)
        assert result is not None
        assert result.failure_type == "ill_conditioned_matrix"


# ---------------------------------------------------------------------------
# ConditionNumberDetector — TRUE NEGATIVES (must stay silent)
# ---------------------------------------------------------------------------

class TestConditionNumberDetectorSilent:
    def setup_method(self):
        self.detector = ConditionNumberDetector()

    def test_silent_on_identity_matrix(self):
        result = self.detector.check(inputs={}, output=np.eye(3), iteration=1)
        assert result is None

    def test_silent_on_well_conditioned_diagonal(self):
        """cond = max/min eigenvalue = 100/1 = 100 — well below 1e12."""
        mat = np.diag([100.0, 10.0, 1.0])
        result = self.detector.check(inputs={}, output=mat, iteration=2)
        assert result is None

    def test_silent_on_1d_array(self):
        result = self.detector.check(inputs={}, output=np.array([1.0, 2.0]), iteration=3)
        assert result is None

    def test_silent_on_non_square_matrix(self):
        result = self.detector.check(inputs={}, output=np.ones((2, 4)), iteration=4)
        assert result is None

    def test_silent_on_scalar_output(self):
        result = self.detector.check(inputs={}, output=0.05, iteration=5)
        assert result is None


# ---------------------------------------------------------------------------
# BoundsDetector — TRUE POSITIVES (must fire)
# ---------------------------------------------------------------------------

class TestBoundsDetectorFires:
    def setup_method(self):
        # Default bounds: weights [-2, 2], var [0, 1], sharpe [-5, 5]
        self.detector = BoundsDetector()

    def test_fires_when_weight_exceeds_upper_bound(self):
        """Weight of 3.0 exceeds max of 2.0."""
        result = self.detector.check(
            inputs={}, output={"weights": np.array([3.0, 0.5, -0.5])}, iteration=1
        )
        assert result is not None
        assert result.failure_type == "bounds_exceeded"

    def test_fires_when_weight_below_lower_bound(self):
        """Weight of -3.0 is below min of -2.0."""
        result = self.detector.check(
            inputs={}, output={"weights": np.array([-3.0, 0.5, 0.5])}, iteration=2
        )
        assert result is not None
        assert result.failure_type == "bounds_exceeded"

    def test_fires_when_var_exceeds_upper_bound(self):
        """VaR of 1.5 exceeds max of 1.0 (150% loss is implausible)."""
        result = self.detector.check(
            inputs={}, output={"var": 1.5}, iteration=3
        )
        assert result is not None
        assert result.failure_type == "bounds_exceeded"

    def test_fires_when_var_is_negative(self):
        """Negative VaR (profit at risk) violates [0, 1] bounds."""
        result = self.detector.check(
            inputs={}, output={"var": -0.1}, iteration=4
        )
        assert result is not None
        assert result.failure_type == "bounds_exceeded"

    def test_fires_when_sharpe_exceeds_upper_bound(self):
        """Sharpe of 10.0 is implausible — above max of 5.0."""
        result = self.detector.check(
            inputs={}, output={"sharpe": 10.0}, iteration=5
        )
        assert result is not None
        assert result.failure_type == "bounds_exceeded"


# ---------------------------------------------------------------------------
# BoundsDetector — TRUE NEGATIVES (must stay silent)
# ---------------------------------------------------------------------------

class TestBoundsDetectorSilent:
    def setup_method(self):
        self.detector = BoundsDetector()

    def test_silent_on_normal_long_only_weights(self):
        """Weights summing to 1, all in [0, 1] — clearly safe."""
        result = self.detector.check(
            inputs={}, output={"weights": np.array([0.3, 0.4, 0.3])}, iteration=1
        )
        assert result is None

    def test_silent_on_intentional_short_within_bounds(self):
        """Small short position (-0.5) is within [-2, 2] — must not fire."""
        result = self.detector.check(
            inputs={}, output={"weights": np.array([-0.5, 0.8, 0.7])}, iteration=2
        )
        assert result is None

    def test_silent_on_var_in_valid_range(self):
        """VaR of 5% is normal and within [0, 1]."""
        result = self.detector.check(
            inputs={}, output={"var": 0.05}, iteration=3
        )
        assert result is None

    def test_silent_on_output_with_no_known_keys(self):
        """Dict with unrecognised keys — no bounds defined, stay silent."""
        result = self.detector.check(
            inputs={}, output={"some_custom_metric": 999.0}, iteration=4
        )
        assert result is None

    def test_silent_on_sharpe_within_bounds(self):
        """Sharpe of 1.2 is within [-5, 5]."""
        result = self.detector.check(
            inputs={}, output={"sharpe": 1.2}, iteration=5
        )
        assert result is None
