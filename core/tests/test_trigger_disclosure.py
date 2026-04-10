"""
Tests for TriggerDisclosure — Phase 5B detector transparency layer.

Every concrete detector that fires on a threshold-based condition must attach
a TriggerDisclosure to its Finding. This makes the decision boundary fully
auditable without reading source code.
"""

from __future__ import annotations

import numpy as np
import pytest

from blackswan.detectors.base import Finding, TriggerDisclosure
from blackswan.detectors.matrix import ConditionNumberDetector, MatrixPSDDetector
from blackswan.detectors.numerical import DivisionStabilityDetector, NaNInfDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _non_psd_matrix(n: int = 3) -> np.ndarray:
    """Return a matrix that is guaranteed to be non-PSD."""
    mat = np.full((n, n), 0.99)
    np.fill_diagonal(mat, 1.0)
    # Force it non-PSD by making off-diagonals too large
    mat[0, 1] = mat[1, 0] = 1.5
    return mat


def _ill_conditioned_matrix(n: int = 3) -> np.ndarray:
    """Return a matrix with condition number >> 1e12."""
    mat = np.eye(n, dtype=float)
    mat[0, 0] = 1e-15   # tiny leading value → huge condition number
    return mat


def _psd_matrix(n: int = 3) -> np.ndarray:
    """Return a clean PSD matrix."""
    a = np.random.default_rng(0).standard_normal((n, n))
    return a @ a.T + np.eye(n)


# ---------------------------------------------------------------------------
# MatrixPSDDetector — TriggerDisclosure
# ---------------------------------------------------------------------------

class TestMatrixPSDDetectorDisclosure:
    def test_firing_finding_has_trigger_disclosure(self):
        det = MatrixPSDDetector()
        finding = det.check(inputs={}, output=_non_psd_matrix(), iteration=0)
        assert finding is not None
        assert finding.trigger_disclosure is not None

    def test_trigger_disclosure_type(self):
        det = MatrixPSDDetector()
        finding = det.check(inputs={}, output=_non_psd_matrix(), iteration=0)
        assert isinstance(finding.trigger_disclosure, TriggerDisclosure)

    def test_detector_name_is_correct(self):
        det = MatrixPSDDetector()
        finding = det.check(inputs={}, output=_non_psd_matrix(), iteration=0)
        assert finding.trigger_disclosure.detector_name == "MatrixPSDDetector"

    def test_comparison_is_less_than(self):
        det = MatrixPSDDetector()
        finding = det.check(inputs={}, output=_non_psd_matrix(), iteration=0)
        assert finding.trigger_disclosure.comparison == "<"

    def test_threshold_matches_epsilon(self):
        epsilon = 1e-8
        det = MatrixPSDDetector(epsilon=epsilon)
        finding = det.check(inputs={}, output=_non_psd_matrix(), iteration=0)
        assert finding.trigger_disclosure.threshold == pytest.approx(-epsilon)

    def test_observed_value_is_negative(self):
        det = MatrixPSDDetector()
        finding = det.check(inputs={}, output=_non_psd_matrix(), iteration=0)
        assert finding.trigger_disclosure.observed_value < 0

    def test_observed_value_matches_actual_min_eigenvalue(self):
        mat = _non_psd_matrix()
        expected_min = float(np.linalg.eigvalsh(mat).min())
        det = MatrixPSDDetector()
        finding = det.check(inputs={}, output=mat, iteration=0)
        assert finding.trigger_disclosure.observed_value == pytest.approx(expected_min, rel=1e-6)

    def test_explanation_is_non_empty_string(self):
        det = MatrixPSDDetector()
        finding = det.check(inputs={}, output=_non_psd_matrix(), iteration=0)
        assert isinstance(finding.trigger_disclosure.explanation, str)
        assert len(finding.trigger_disclosure.explanation) > 10

    def test_no_disclosure_when_silent(self):
        """When the matrix is PSD, check() returns None — no disclosure at all."""
        det = MatrixPSDDetector()
        result = det.check(inputs={}, output=_psd_matrix(), iteration=0)
        assert result is None

    def test_disclosure_present_in_dict_output(self):
        det = MatrixPSDDetector()
        finding = det.check(
            inputs={}, output={"cov": _non_psd_matrix()}, iteration=0
        )
        assert finding is not None
        assert finding.trigger_disclosure is not None


# ---------------------------------------------------------------------------
# ConditionNumberDetector — TriggerDisclosure
# ---------------------------------------------------------------------------

class TestConditionNumberDetectorDisclosure:
    def test_firing_finding_has_trigger_disclosure(self):
        det = ConditionNumberDetector()
        finding = det.check(inputs={}, output=_ill_conditioned_matrix(), iteration=0)
        assert finding is not None
        assert finding.trigger_disclosure is not None

    def test_trigger_disclosure_type(self):
        det = ConditionNumberDetector()
        finding = det.check(inputs={}, output=_ill_conditioned_matrix(), iteration=0)
        assert isinstance(finding.trigger_disclosure, TriggerDisclosure)

    def test_detector_name_is_correct(self):
        det = ConditionNumberDetector()
        finding = det.check(inputs={}, output=_ill_conditioned_matrix(), iteration=0)
        assert finding.trigger_disclosure.detector_name == "ConditionNumberDetector"

    def test_comparison_is_greater_than(self):
        det = ConditionNumberDetector()
        finding = det.check(inputs={}, output=_ill_conditioned_matrix(), iteration=0)
        assert finding.trigger_disclosure.comparison == ">"

    def test_threshold_matches_detector_threshold(self):
        threshold = 1e10
        det = ConditionNumberDetector(threshold=threshold)
        finding = det.check(inputs={}, output=_ill_conditioned_matrix(), iteration=0)
        assert finding.trigger_disclosure.threshold == pytest.approx(threshold)

    def test_observed_value_exceeds_threshold(self):
        det = ConditionNumberDetector()
        finding = det.check(inputs={}, output=_ill_conditioned_matrix(), iteration=0)
        assert finding.trigger_disclosure.observed_value > det.threshold

    def test_explanation_mentions_condition_number(self):
        det = ConditionNumberDetector()
        finding = det.check(inputs={}, output=_ill_conditioned_matrix(), iteration=0)
        assert "condition" in finding.trigger_disclosure.explanation.lower()

    def test_no_disclosure_when_silent(self):
        det = ConditionNumberDetector()
        well_conditioned = np.eye(3) * 2.0
        result = det.check(inputs={}, output=well_conditioned, iteration=0)
        assert result is None


# ---------------------------------------------------------------------------
# DivisionStabilityDetector — TriggerDisclosure
# ---------------------------------------------------------------------------

class TestDivisionStabilityDetectorDisclosure:
    def test_firing_finding_has_trigger_disclosure(self):
        det = DivisionStabilityDetector()
        finding = det.check(inputs={"denominator": 1e-15}, output=None, iteration=0)
        assert finding is not None
        assert finding.trigger_disclosure is not None

    def test_detector_name_is_correct(self):
        det = DivisionStabilityDetector()
        finding = det.check(inputs={"denominator": 0.0}, output=None, iteration=0)
        assert finding.trigger_disclosure.detector_name == "DivisionStabilityDetector"

    def test_comparison_is_less_than_or_equal(self):
        det = DivisionStabilityDetector()
        finding = det.check(inputs={"denominator": 0.0}, output=None, iteration=0)
        assert finding.trigger_disclosure.comparison == "<="

    def test_threshold_matches_epsilon(self):
        epsilon = 1e-8
        det = DivisionStabilityDetector(epsilon=epsilon)
        finding = det.check(inputs={"denominator": 0.0}, output=None, iteration=0)
        assert finding.trigger_disclosure.threshold == pytest.approx(epsilon)

    def test_observed_value_near_zero(self):
        det = DivisionStabilityDetector()
        finding = det.check(inputs={"denominator": 5e-15}, output=None, iteration=0)
        # observed_value should be the denominator value
        assert abs(finding.trigger_disclosure.observed_value) < 1e-12

    def test_explanation_is_non_empty_string(self):
        det = DivisionStabilityDetector()
        finding = det.check(inputs={"denominator": 0.0}, output=None, iteration=0)
        assert isinstance(finding.trigger_disclosure.explanation, str)
        assert len(finding.trigger_disclosure.explanation) > 10

    def test_no_disclosure_when_silent(self):
        det = DivisionStabilityDetector()
        result = det.check(inputs={"denominator": 1.0}, output=None, iteration=0)
        assert result is None


# ---------------------------------------------------------------------------
# NaNInfDetector — no TriggerDisclosure (observation-based, no threshold)
# ---------------------------------------------------------------------------

class TestNaNInfDetectorNoDisclosure:
    def test_nan_finding_has_no_trigger_disclosure(self):
        """NaNInfDetector detects an observation (NaN present), not a threshold crossing."""
        det = NaNInfDetector()
        finding = det.check(inputs={}, output=float("nan"), iteration=0)
        assert finding is not None
        # NaN detection has no configurable threshold — disclosure is not applicable
        assert finding.trigger_disclosure is None

    def test_inf_finding_has_no_trigger_disclosure(self):
        det = NaNInfDetector()
        finding = det.check(inputs={}, output=float("inf"), iteration=0)
        assert finding is not None
        assert finding.trigger_disclosure is None


# ---------------------------------------------------------------------------
# TriggerDisclosure dataclass contract
# ---------------------------------------------------------------------------

class TestTriggerDisclosureContract:
    def test_all_required_fields_present(self):
        td = TriggerDisclosure(
            detector_name="TestDetector",
            observed_value=3.14,
            threshold=2.0,
            comparison=">",
            explanation="3.14 exceeded 2.0",
        )
        assert td.detector_name == "TestDetector"
        assert td.observed_value == 3.14
        assert td.threshold == 2.0
        assert td.comparison == ">"
        assert td.explanation == "3.14 exceeded 2.0"

    def test_observed_value_can_be_string(self):
        td = TriggerDisclosure(
            detector_name="D",
            observed_value="NaN",
            threshold="finite",
            comparison="!=",
            explanation="Not finite",
        )
        assert td.observed_value == "NaN"

    def test_finding_trigger_disclosure_defaults_to_none(self):
        """New Finding objects must have trigger_disclosure=None by default."""
        f = Finding(
            failure_type="nan_inf",
            severity="critical",
            message="test",
            iteration=0,
        )
        assert f.trigger_disclosure is None
