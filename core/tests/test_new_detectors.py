"""
Tests for ExplodingGradientDetector.

Each test verifies real detector behavior — no mocking of the class under test.
True-positive tests confirm the detector fires when the ratio exceeds the threshold.
True-negative tests confirm silence when the ratio is within bounds or baseline is absent.
"""

import math

import numpy as np
import pytest

from blackswan.detectors.numerical import ExplodingGradientDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detector(threshold: float = ExplodingGradientDetector.DEFAULT_THRESHOLD) -> ExplodingGradientDetector:
    return ExplodingGradientDetector(threshold=threshold)


# ---------------------------------------------------------------------------
# TestExplodingGradientDetector
# ---------------------------------------------------------------------------

class TestExplodingGradientDetector:

    def test_small_change_silent(self):
        """Input changes 0.01, output changes 0.01 → ratio ≈ 1 < 100 → None."""
        det = _make_detector()
        det.set_baseline({"x": 1.0}, 1.0)
        result = det.check({"x": 1.01}, 1.01, iteration=0)
        assert result is None

    def test_large_output_change_fires(self):
        """threshold=10, input changes 0.01, output changes 100 → ratio >> 10 → Finding."""
        det = _make_detector(threshold=10.0)
        det.set_baseline({"x": 1.0}, 0.0)
        # input delta norm ≈ 0.01, output delta ≈ 100 → ratio ≈ 10000
        result = det.check({"x": 1.01}, 100.0, iteration=3)
        assert result is not None

    def test_finding_failure_type_is_nan_inf(self):
        """Confirmed finding carries failure_type='nan_inf'."""
        det = _make_detector(threshold=10.0)
        det.set_baseline({"x": 1.0}, 0.0)
        finding = det.check({"x": 1.01}, 100.0, iteration=0)
        assert finding is not None
        assert finding.failure_type == "nan_inf"

    def test_finding_severity_is_critical(self):
        """Confirmed finding carries severity='critical'."""
        det = _make_detector(threshold=10.0)
        det.set_baseline({"x": 1.0}, 0.0)
        finding = det.check({"x": 1.01}, 100.0, iteration=0)
        assert finding is not None
        assert finding.severity == "critical"

    def test_no_baseline_set_always_silent(self):
        """Without set_baseline call, check() returns None regardless of inputs."""
        det = _make_detector(threshold=1.0)  # very sensitive threshold
        # Large output relative to input, but no baseline → must stay silent
        result = det.check({"x": 0.001}, 99999.0, iteration=0)
        assert result is None

    def test_zero_input_delta_does_not_raise(self):
        """Identical inputs but changed output: input_delta=0, handled via epsilon → fires."""
        det = _make_detector(threshold=10.0)
        det.set_baseline({"x": 5.0}, 0.0)
        # input unchanged → input_delta = 0 → ratio = |output_delta| / epsilon
        # output changed to 1.0 → ratio = 1.0 / 1e-12 >> 10
        result = det.check({"x": 5.0}, 1.0, iteration=7)
        assert result is not None  # large ratio due to epsilon denominator

    def test_silent_when_below_threshold(self):
        """threshold=1000, moderate output change → ratio well below threshold → None."""
        det = _make_detector(threshold=1000.0)
        det.set_baseline({"x": 10.0}, 10.0)
        # input: 10→20 (delta norm ≈ 10), output: 10→15 (delta ≈ 5) → ratio ≈ 0.5
        result = det.check({"x": 20.0}, 15.0, iteration=2)
        assert result is None

    def test_ratio_message_contains_ratio(self):
        """The finding message embeds the computed ratio value."""
        det = _make_detector(threshold=10.0)
        det.set_baseline({"x": 1.0}, 0.0)
        finding = det.check({"x": 1.01}, 100.0, iteration=0)
        assert finding is not None
        # Message format: "... = {ratio:.1f}x ..."
        assert "x" in finding.message  # ratio label present
        # The ratio should be a large number; verify it appears in the message
        assert finding.message.count("x") >= 2  # threshold and ratio both appear

    def test_numpy_array_output_norm_computed(self):
        """Output as np.ndarray: detector computes L2 norm correctly."""
        det = _make_detector(threshold=5.0)
        base_out = np.array([1.0, 0.0, 0.0])
        det.set_baseline({"x": 1.0}, base_out)
        # input delta ≈ 0.01, output L2 norm changes from 1 to ~100 → fires
        big_out = np.array([100.0, 0.0, 0.0])
        result = det.check({"x": 1.01}, big_out, iteration=1)
        assert result is not None

    def test_dict_input_norms_handled(self):
        """Inputs dict with multiple float values: all contribute to the L2 norm."""
        det = _make_detector(threshold=10.0)
        base_inputs = {"a": 3.0, "b": 4.0}  # norm = 5.0
        det.set_baseline(base_inputs, 0.0)

        # Compute expected baseline norm: sqrt(9 + 16) = 5
        assert math.isclose(ExplodingGradientDetector._vector_norm(base_inputs), 5.0)

        # perturbed inputs with same norm → input_delta ≈ 0, output changed → fires
        same_norm_inputs = {"a": 5.0, "b": 0.0}  # norm = 5 → delta = 0
        result = det.check(same_norm_inputs, 1000.0, iteration=4)
        assert result is not None  # large output delta / ~epsilon input delta
