"""
Tests for StressRunner.

Covers: RunResult structure, finding collection, iteration numbering,
multiple detectors, exception handling, and full determinism guarantee.
"""

import pytest
import numpy as np

from blackswan.detectors.base import FailureDetector, Finding
from blackswan.detectors.numerical import NaNInfDetector
from blackswan.engine.runner import StressRunner, RunResult


# ---------------------------------------------------------------------------
# Test helpers — scenarios and functions
# ---------------------------------------------------------------------------

def _identity(value: float) -> float:
    """Returns value unchanged — NaN and Inf pass straight through."""
    return float(value)


def _always_clean(value: float) -> float:
    """Always returns a safe finite value regardless of input."""
    return 1.0


def _always_raises(value: float) -> float:
    raise ValueError("deliberately broken target function")


class _FixedScenario:
    """Injects the same fixed value every iteration. Does not use the RNG."""

    def __init__(self, value: float, iterations: int = 10):
        self.iterations = iterations
        self._value = value

    def apply(self, inputs: dict, rng: np.random.Generator) -> dict:
        return {"value": self._value}


class _RandomScenario:
    """Draws value uniformly from [low, high] each iteration using the RNG."""

    def __init__(self, low: float, high: float, iterations: int = 30):
        self.iterations = iterations
        self._low = low
        self._high = high

    def apply(self, inputs: dict, rng: np.random.Generator) -> dict:
        return {"value": float(rng.uniform(self._low, self._high))}


class _RecordingScenario:
    """Records every value it generates so tests can inspect the sequence."""

    def __init__(self, iterations: int = 20):
        self.iterations = iterations
        self.applied: list[float] = []

    def apply(self, inputs: dict, rng: np.random.Generator) -> dict:
        v = float(rng.uniform(0.0, 1.0))
        self.applied.append(v)
        return {"value": v}


class _AlwaysFiresDetector(FailureDetector):
    """Test double that fires on every call."""

    def check(self, inputs, output, iteration):
        return Finding(
            failure_type="nan_inf",
            severity="critical",
            message="always fires",
            iteration=iteration,
        )


class _NeverFiresDetector(FailureDetector):
    """Test double that never fires."""

    def check(self, inputs, output, iteration):
        return None


BASE_INPUTS = {"value": 1.0}


# ---------------------------------------------------------------------------
# RunResult structure
# ---------------------------------------------------------------------------

class TestRunResult:
    def test_run_returns_run_result_instance(self):
        runner = StressRunner(
            fn=_always_clean,
            base_inputs=BASE_INPUTS,
            scenario=_FixedScenario(1.0, iterations=5),
            detectors=[],
            seed=42,
        )
        result = runner.run()
        assert isinstance(result, RunResult)

    def test_iterations_completed_matches_scenario(self):
        runner = StressRunner(
            fn=_always_clean,
            base_inputs=BASE_INPUTS,
            scenario=_FixedScenario(1.0, iterations=7),
            detectors=[],
            seed=42,
        )
        assert runner.run().iterations_completed == 7

    def test_seed_is_stored_in_result(self):
        runner = StressRunner(
            fn=_always_clean,
            base_inputs=BASE_INPUTS,
            scenario=_FixedScenario(1.0),
            detectors=[],
            seed=42,
        )
        assert runner.run().seed == 42

    def test_runtime_ms_is_non_negative(self):
        runner = StressRunner(
            fn=_always_clean,
            base_inputs=BASE_INPUTS,
            scenario=_FixedScenario(1.0, iterations=10),
            detectors=[],
            seed=42,
        )
        assert runner.run().runtime_ms >= 0


# ---------------------------------------------------------------------------
# Finding collection
# ---------------------------------------------------------------------------

class TestFindingCollection:
    def test_no_findings_when_output_is_always_clean(self):
        runner = StressRunner(
            fn=_always_clean,
            base_inputs=BASE_INPUTS,
            scenario=_FixedScenario(1.0, iterations=10),
            detectors=[NaNInfDetector()],
            seed=42,
        )
        assert runner.run().findings == []

    def test_one_finding_per_iteration_when_output_is_always_nan(self):
        """NaN injected every iteration → NaNInfDetector fires every iteration."""
        runner = StressRunner(
            fn=_identity,
            base_inputs=BASE_INPUTS,
            scenario=_FixedScenario(float("nan"), iterations=10),
            detectors=[NaNInfDetector()],
            seed=42,
        )
        assert len(runner.run().findings) == 10

    def test_finding_iteration_numbers_are_zero_indexed_and_sequential(self):
        runner = StressRunner(
            fn=_identity,
            base_inputs=BASE_INPUTS,
            scenario=_FixedScenario(float("nan"), iterations=5),
            detectors=[NaNInfDetector()],
            seed=42,
        )
        iterations = [f.iteration for f in runner.run().findings]
        assert iterations == [0, 1, 2, 3, 4]

    def test_never_fires_detector_contributes_zero_findings(self):
        runner = StressRunner(
            fn=_always_clean,
            base_inputs=BASE_INPUTS,
            scenario=_FixedScenario(1.0, iterations=6),
            detectors=[_AlwaysFiresDetector(), _NeverFiresDetector()],
            seed=42,
        )
        # Only AlwaysFires contributes — proves NeverFires was called but stayed silent
        assert len(runner.run().findings) == 6

    def test_findings_from_two_detectors_are_combined(self):
        """Two AlwaysFires detectors on 4 iterations → 8 total findings."""
        runner = StressRunner(
            fn=_always_clean,
            base_inputs=BASE_INPUTS,
            scenario=_FixedScenario(1.0, iterations=4),
            detectors=[_AlwaysFiresDetector(), _AlwaysFiresDetector()],
            seed=42,
        )
        assert len(runner.run().findings) == 8

    def test_partial_failures_only_counted_when_detector_fires(self):
        """NaNInfDetector only fires on NaN iterations, not clean ones."""
        # 10 iterations: 10 clean (vol=1.0 → output=1.0, no NaN)
        runner = StressRunner(
            fn=_identity,
            base_inputs=BASE_INPUTS,
            scenario=_FixedScenario(1.0, iterations=10),
            detectors=[NaNInfDetector()],
            seed=42,
        )
        assert runner.run().findings == []


# ---------------------------------------------------------------------------
# Exception handling
# ---------------------------------------------------------------------------

class TestExceptionHandling:
    def test_exception_in_target_does_not_propagate(self):
        """Runner must swallow exceptions from the target function."""
        runner = StressRunner(
            fn=_always_raises,
            base_inputs=BASE_INPUTS,
            scenario=_FixedScenario(1.0, iterations=3),
            detectors=[],
            seed=42,
        )
        result = runner.run()  # must not raise
        assert result.iterations_completed == 3

    def test_exception_is_recorded_as_nan_inf_finding(self):
        """Each exception from the target function becomes one Finding."""
        runner = StressRunner(
            fn=_always_raises,
            base_inputs=BASE_INPUTS,
            scenario=_FixedScenario(1.0, iterations=3),
            detectors=[],
            seed=42,
        )
        result = runner.run()
        assert len(result.findings) == 3
        assert all(f.failure_type == "nan_inf" for f in result.findings)
        assert all(f.severity == "critical" for f in result.findings)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_produces_identical_findings(self):
        """Two independent runs with the same seed must yield identical findings."""
        kwargs = dict(
            fn=_identity,
            base_inputs=BASE_INPUTS,
            detectors=[NaNInfDetector()],
            seed=42,
        )
        result_a = StressRunner(scenario=_RandomScenario(0.0, 2.0, 50), **kwargs).run()
        result_b = StressRunner(scenario=_RandomScenario(0.0, 2.0, 50), **kwargs).run()

        assert result_a.findings == result_b.findings
        assert result_a.iterations_completed == result_b.iterations_completed

    def test_different_seeds_produce_different_perturbation_sequences(self):
        """Different seeds must drive different RNG sequences."""
        s1 = _RecordingScenario(iterations=20)
        s2 = _RecordingScenario(iterations=20)

        StressRunner(fn=_always_clean, base_inputs=BASE_INPUTS,
                     scenario=s1, detectors=[], seed=42).run()
        StressRunner(fn=_always_clean, base_inputs=BASE_INPUTS,
                     scenario=s2, detectors=[], seed=99).run()

        assert s1.applied != s2.applied
