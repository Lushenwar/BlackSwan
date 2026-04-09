"""
Tests for PlausibilityValidator and its integration with StressRunner.

Covers:
    PlausibilityConstraint — floor-only, ceiling-only, both, missing-key
    PlausibilityValidator  — aggregation, short-circuit, never-raises contract
    StressRunner           — skips invalid iterations, calls fn on valid ones,
                             works with plain duck-typed scenarios (no make_validator)
"""

import pytest
import numpy as np

from blackswan.engine.validator import (
    PlausibilityConstraint,
    PlausibilityValidator,
    ValidationError,
)
from blackswan.engine.runner import StressRunner, RunResult
from blackswan.detectors.base import FailureDetector, Finding


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_constraint(target, *, min_value=None, max_value=None):
    return PlausibilityConstraint(target=target, min_value=min_value, max_value=max_value)


# ---------------------------------------------------------------------------
# TestConstraintFloor — min_value enforcement
# ---------------------------------------------------------------------------

class TestConstraintFloor:
    def test_value_above_floor_passes(self):
        c = _make_constraint("price", min_value=0.0)
        c.check({"price": 1.0})  # must not raise

    def test_value_at_floor_passes(self):
        c = _make_constraint("price", min_value=0.0)
        c.check({"price": 0.0})  # exactly at floor — inclusive bound

    def test_value_below_floor_raises(self):
        c = _make_constraint("price", min_value=0.0)
        with pytest.raises(ValidationError) as exc_info:
            c.check({"price": -0.01})
        assert "price" in str(exc_info.value)

    def test_missing_key_is_ignored(self):
        c = _make_constraint("price", min_value=0.0)
        # inputs only has "vol" — no "price" key → silent no-op
        c.check({"vol": 0.5})  # must not raise

    def test_min_only_no_ceiling_check(self):
        # Even a very large value should pass if only min_value is set
        c = _make_constraint("price", min_value=0.0)
        c.check({"price": 1_000_000.0})  # must not raise


# ---------------------------------------------------------------------------
# TestConstraintCeiling — max_value enforcement
# ---------------------------------------------------------------------------

class TestConstraintCeiling:
    def test_value_below_ceiling_passes(self):
        c = _make_constraint("vol", max_value=1.0)
        c.check({"vol": 0.5})  # must not raise

    def test_value_at_ceiling_passes(self):
        c = _make_constraint("vol", max_value=1.0)
        c.check({"vol": 1.0})  # exactly at ceiling — inclusive bound

    def test_value_above_ceiling_raises(self):
        c = _make_constraint("vol", max_value=1.0)
        with pytest.raises(ValidationError) as exc_info:
            c.check({"vol": 1.01})
        assert "vol" in str(exc_info.value)

    def test_max_only_no_floor_check(self):
        # Even a very large negative value should pass if only max_value is set
        c = _make_constraint("vol", max_value=1.0)
        c.check({"vol": -999.0})  # must not raise


# ---------------------------------------------------------------------------
# TestConstraintBoth — both min_value and max_value enforced
# ---------------------------------------------------------------------------

class TestConstraintBoth:
    def test_in_range_passes(self):
        c = _make_constraint("rate", min_value=-0.05, max_value=0.10)
        c.check({"rate": 0.02})  # must not raise

    def test_below_min_raises(self):
        c = _make_constraint("rate", min_value=-0.05, max_value=0.10)
        with pytest.raises(ValidationError):
            c.check({"rate": -0.06})

    def test_above_max_raises(self):
        c = _make_constraint("rate", min_value=-0.05, max_value=0.10)
        with pytest.raises(ValidationError):
            c.check({"rate": 0.11})

    def test_boundary_values_pass(self):
        c = _make_constraint("rate", min_value=-0.05, max_value=0.10)
        c.check({"rate": -0.05})  # exactly at min — must not raise
        c.check({"rate": 0.10})   # exactly at max — must not raise


# ---------------------------------------------------------------------------
# TestValidator — aggregation and never-raises contract
# ---------------------------------------------------------------------------

class TestValidator:
    def test_empty_constraints_always_passes(self):
        v = PlausibilityValidator([])
        assert v.validate({"price": -999_999.0, "vol": float("inf")}) is True

    def test_all_constraints_satisfied(self):
        v = PlausibilityValidator([
            _make_constraint("price", min_value=0.0),
            _make_constraint("vol", max_value=1.0),
        ])
        assert v.validate({"price": 1.0, "vol": 0.5}) is True

    def test_first_violation_returns_false(self):
        v = PlausibilityValidator([
            _make_constraint("price", min_value=0.0),
        ])
        result = v.validate({"price": -1.0})
        assert result is False

    def test_second_constraint_violation_returns_false(self):
        v = PlausibilityValidator([
            _make_constraint("price", min_value=0.0),   # passes: price=1.0
            _make_constraint("vol", max_value=0.5),      # fails:  vol=0.9
        ])
        result = v.validate({"price": 1.0, "vol": 0.9})
        assert result is False

    def test_validate_never_raises(self):
        # Even with None values in the dict, validate() must return a bool, not raise.
        v = PlausibilityValidator([
            _make_constraint("x", min_value=0.0),
        ])
        # None < 0.0 raises TypeError in Python — validator must catch this.
        try:
            result = v.validate({"x": None})
            assert isinstance(result, bool)
        except Exception as exc:
            pytest.fail(f"validate() raised unexpectedly: {exc!r}")

    def test_multiple_constraints_all_pass(self):
        v = PlausibilityValidator([
            _make_constraint("a", min_value=0.0, max_value=10.0),
            _make_constraint("b", min_value=-1.0, max_value=1.0),
            _make_constraint("c", min_value=0.5),
        ])
        assert v.validate({"a": 5.0, "b": 0.0, "c": 1.0}) is True


# ---------------------------------------------------------------------------
# TestValidatorIntegrationWithStressRunner
# ---------------------------------------------------------------------------

class _CallCountingFn:
    """Wraps a simple function and counts how many times it has been called."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, **kwargs):
        self.call_count += 1
        return 1.0


class _ValidatorScenario:
    """
    Scenario whose apply() always returns a fixed dict.
    Exposes make_validator() so StressRunner wires the validator.
    """

    def __init__(self, fixed_inputs: dict, constraint: PlausibilityConstraint, iterations: int = 10):
        self.iterations = iterations
        self._fixed_inputs = fixed_inputs
        self._constraint = constraint

    def apply(self, inputs: dict, rng: np.random.Generator) -> dict:
        return dict(self._fixed_inputs)

    def make_validator(self) -> PlausibilityValidator:
        return PlausibilityValidator([self._constraint])


class _PlainScenario:
    """Duck-typed scenario with NO make_validator — mimics pre-Task-A1 objects."""

    def __init__(self, iterations: int = 5):
        self.iterations = iterations

    def apply(self, inputs: dict, rng: np.random.Generator) -> dict:
        return {"value": 1.0}


class _AlwaysFiresDetector(FailureDetector):
    def check(self, inputs, output, iteration):
        return Finding(
            failure_type="nan_inf",
            severity="critical",
            message="always fires",
            iteration=iteration,
        )


class TestValidatorIntegrationWithStressRunner:
    def test_invalid_inputs_are_skipped_not_counted(self):
        """
        Scenario always produces price=-1.0; validator requires price >= 0.
        The target function should never be called, so findings list must be empty
        and iterations_completed must be 0 (skipped iterations are not counted).
        """
        fn = _CallCountingFn()
        scenario = _ValidatorScenario(
            fixed_inputs={"price": -1.0},
            constraint=_make_constraint("price", min_value=0.0),
            iterations=10,
        )
        runner = StressRunner(
            fn=fn,
            base_inputs={"price": 1.0},
            scenario=scenario,
            detectors=[_AlwaysFiresDetector()],
            seed=42,
        )
        result = runner.run()
        # All 10 iterations were rejected by the validator.
        assert result.findings == [], (
            f"Expected no findings when all inputs are invalid, got {result.findings}"
        )
        # Skipped iterations must not count toward iterations_completed.
        assert result.iterations_completed == 0, (
            f"Expected iterations_completed=0 (all skipped), got {result.iterations_completed}"
        )
        # The counting function must not have been called during the loop
        # (it IS called once during the pre-loop baseline pass, which uses base_inputs
        # not the perturbed inputs, so we accept at most 1 call).
        assert fn.call_count <= 1, (
            f"Target function called {fn.call_count} times; expected ≤1 (baseline only)"
        )

    def test_valid_inputs_reach_the_function(self):
        """
        Scenario always produces price=5.0; validator requires price >= 0.
        The function must be called on every iteration.
        """
        fn = _CallCountingFn()
        scenario = _ValidatorScenario(
            fixed_inputs={"price": 5.0},
            constraint=_make_constraint("price", min_value=0.0),
            iterations=10,
        )
        runner = StressRunner(
            fn=fn,
            base_inputs={"price": 1.0},
            scenario=scenario,
            detectors=[],
            seed=42,
        )
        runner.run()
        # Baseline call (base_inputs) + 10 valid iterations = 11 calls total.
        assert fn.call_count == 11, (
            f"Expected 11 calls (1 baseline + 10 iterations), got {fn.call_count}"
        )

    def test_hasattr_guard_works_with_plain_scenario(self):
        """
        A scenario without make_validator must not cause StressRunner to raise.
        The guard uses hasattr(), so plain duck-typed scenarios must still work.
        """
        fn = _CallCountingFn()
        scenario = _PlainScenario(iterations=5)

        runner = StressRunner(
            fn=fn,
            base_inputs={"value": 1.0},
            scenario=scenario,
            detectors=[],
            seed=0,
        )
        result = runner.run()  # must not raise
        assert isinstance(result, RunResult)
        assert result.iterations_completed == 5

    def test_baseline_established_false_when_fn_raises_on_base_inputs(self):
        """
        When the target function raises on unperturbed base_inputs, baseline_established
        must be False so callers know the baseline was never set.
        """
        def always_raises(**kwargs):
            raise ValueError("always bad")

        class SimpleScenario:
            iterations = 3

            def apply(self, inputs, rng):
                return dict(inputs)

        runner = StressRunner(
            fn=always_raises,
            base_inputs={"x": 1.0},
            scenario=SimpleScenario(),
            detectors=[],
            seed=42,
        )
        result = runner.run()
        assert result.baseline_established is False

    def test_iterations_completed_counts_only_executed_not_skipped(self):
        """
        Validator rejects every perturbed input → 0 iterations are executed.
        iterations_completed must be 0, not the scenario's .iterations value.
        """
        def fn(**kwargs):
            return 1.0

        class ScenarioWithValidator:
            iterations = 5

            def apply(self, inputs, rng):
                return {"price": -1.0}  # always invalid

            def make_validator(self):
                return PlausibilityValidator(
                    [PlausibilityConstraint("price", min_value=0.0)]
                )

        runner = StressRunner(
            fn=fn,
            base_inputs={"price": 1.0},
            scenario=ScenarioWithValidator(),
            detectors=[],
            seed=42,
        )
        result = runner.run()
        assert result.iterations_completed == 0, (
            f"Expected 0 executed iterations (all skipped), got {result.iterations_completed}"
        )
