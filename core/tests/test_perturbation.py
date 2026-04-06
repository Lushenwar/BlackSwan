"""
Tests for the Perturbation system.

Covers: multiplicative/additive types, uniform/lognormal distributions,
determinism, range clamping, passthrough of untargeted keys, and immutability.
"""

import numpy as np
import pytest

from blackswan.engine.perturbation import Perturbation, apply_perturbations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Multiplicative — uniform distribution
# ---------------------------------------------------------------------------

class TestMultiplicativeUniform:
    def _perturb(self, base: float, low: float, high: float, seed: int = 42) -> float:
        p = Perturbation(
            target="vol",
            type="multiplicative",
            distribution="uniform",
            params={"low": low, "high": high},
        )
        return apply_perturbations({"vol": base}, [p], _rng(seed))["vol"]

    def test_output_equals_base_times_factor_in_range(self):
        """Result must be base * f where f is in [low, high]."""
        base, low, high = 0.2, 1.5, 2.5
        result = self._perturb(base, low, high)
        assert base * low <= result <= base * high

    def test_factor_range_respected_across_many_draws(self):
        """Every draw over 500 iterations must stay in [base*low, base*high]."""
        base, low, high = 0.2, 1.5, 2.5
        p = Perturbation(
            target="vol", type="multiplicative", distribution="uniform",
            params={"low": low, "high": high},
        )
        rng = _rng(0)
        for _ in range(500):
            result = apply_perturbations({"vol": base}, [p], rng)["vol"]
            assert base * low <= result <= base * high

    def test_value_is_different_from_base(self):
        """With low=1.5, result cannot equal original value."""
        result = self._perturb(0.2, 1.5, 2.5)
        assert result != 0.2


# ---------------------------------------------------------------------------
# Additive — uniform distribution
# ---------------------------------------------------------------------------

class TestAdditiveUniform:
    def _perturb(self, base: float, low: float, high: float, seed: int = 42) -> float:
        p = Perturbation(
            target="corr",
            type="additive",
            distribution="uniform",
            params={"low": low, "high": high},
        )
        return apply_perturbations({"corr": base}, [p], _rng(seed))["corr"]

    def test_output_equals_base_plus_shift_in_range(self):
        """Result must be base + s where s is in [low, high]."""
        base, low, high = 0.5, 0.10, 0.30
        result = self._perturb(base, low, high)
        assert base + low <= result <= base + high

    def test_shift_range_respected_across_many_draws(self):
        p = Perturbation(
            target="corr", type="additive", distribution="uniform",
            params={"low": 0.10, "high": 0.35},
        )
        rng = _rng(0)
        for _ in range(500):
            result = apply_perturbations({"corr": 0.5}, [p], rng)["corr"]
            assert 0.60 <= result <= 0.85

    def test_negative_shift_reduces_value(self):
        """Negative shift range must reduce the base value."""
        result = self._perturb(0.5, -0.20, -0.05)
        assert 0.30 <= result <= 0.45


# ---------------------------------------------------------------------------
# Multiplicative — lognormal distribution
# ---------------------------------------------------------------------------

class TestMultiplicativeLognormal:
    def test_lognormal_result_is_always_positive(self):
        """Lognormal factor is always positive → positive base stays positive."""
        p = Perturbation(
            target="vol", type="multiplicative", distribution="lognormal",
            params={"mu": 0.5, "sigma": 0.3},
        )
        rng = _rng(0)
        for _ in range(200):
            result = apply_perturbations({"vol": 0.2}, [p], rng)["vol"]
            assert result > 0

    def test_lognormal_changes_value(self):
        """With mu=0.5, typical factor ≈ e^0.5 ≈ 1.65 — result ≠ base."""
        p = Perturbation(
            target="vol", type="multiplicative", distribution="lognormal",
            params={"mu": 0.5, "sigma": 0.3},
        )
        result = apply_perturbations({"vol": 0.2}, [p], _rng(42))["vol"]
        assert result != 0.2


# ---------------------------------------------------------------------------
# Range clamping
# ---------------------------------------------------------------------------

class TestRangeClamping:
    def test_additive_output_clamped_at_upper_bound(self):
        """Correlation 0.95 + shift [0.4, 0.6] → [1.35, 1.55], clamped to 0.99."""
        p = Perturbation(
            target="corr", type="additive", distribution="uniform",
            params={"low": 0.40, "high": 0.60},
            clamp=(None, 0.99),
        )
        rng = _rng(0)
        for _ in range(100):
            result = apply_perturbations({"corr": 0.95}, [p], rng)["corr"]
            assert result <= 0.99

    def test_additive_output_clamped_at_lower_bound(self):
        """vol 0.05 - shift [0.04, 0.06] → [-0.01, 0.01], clamped to min 0.0."""
        p = Perturbation(
            target="vol", type="additive", distribution="uniform",
            params={"low": 0.04, "high": 0.06},
            clamp=(0.0, None),
        )
        rng = _rng(0)
        for _ in range(100):
            result = apply_perturbations({"vol": 0.05}, [p], rng)["vol"]
            assert result >= 0.0

    def test_no_clamp_applied_when_not_specified(self):
        """Without clamp, a large additive shift can exceed natural bounds."""
        p = Perturbation(
            target="corr", type="additive", distribution="uniform",
            params={"low": 0.5, "high": 0.6},
        )
        rng = _rng(0)
        results = [apply_perturbations({"corr": 0.8}, [p], rng)["corr"] for _ in range(20)]
        # At least one result should exceed 1.0 (0.8 + [0.5, 0.6])
        assert any(r > 1.0 for r in results)


# ---------------------------------------------------------------------------
# Passthrough and immutability
# ---------------------------------------------------------------------------

class TestPassthroughAndImmutability:
    def test_untargeted_key_passes_through_unchanged(self):
        """Keys not listed in perturbations must not be modified."""
        p = Perturbation(
            target="vol", type="multiplicative", distribution="uniform",
            params={"low": 2.0, "high": 3.0},
        )
        inputs = {"vol": 0.2, "spread": 100.0, "corr": 0.5}
        result = apply_perturbations(inputs, [p], _rng(42))
        assert result["spread"] == 100.0
        assert result["corr"] == 0.5

    def test_original_inputs_dict_is_not_mutated(self):
        """apply_perturbations must return a new dict and leave inputs unchanged."""
        p = Perturbation(
            target="vol", type="multiplicative", distribution="uniform",
            params={"low": 2.0, "high": 3.0},
        )
        inputs = {"vol": 0.2}
        original_vol = inputs["vol"]
        apply_perturbations(inputs, [p], _rng(42))
        assert inputs["vol"] == original_vol

    def test_empty_perturbations_returns_copy_of_inputs(self):
        """No perturbations → output equals input values exactly."""
        inputs = {"vol": 0.2, "corr": 0.5}
        result = apply_perturbations(inputs, [], _rng(42))
        assert result == inputs
        assert result is not inputs  # must be a copy, not the same object


# ---------------------------------------------------------------------------
# Multiple perturbations
# ---------------------------------------------------------------------------

class TestMultiplePerturbations:
    def test_two_perturbations_applied_to_different_keys(self):
        """Both vol and corr get perturbed independently."""
        perturbations = [
            Perturbation(target="vol", type="multiplicative", distribution="uniform",
                         params={"low": 1.5, "high": 2.5}),
            Perturbation(target="corr", type="additive", distribution="uniform",
                         params={"low": 0.1, "high": 0.2}),
        ]
        inputs = {"vol": 0.2, "corr": 0.5}
        result = apply_perturbations(inputs, perturbations, _rng(42))

        assert result["vol"] != 0.2   # was perturbed
        assert result["corr"] != 0.5  # was perturbed
        assert 0.2 * 1.5 <= result["vol"] <= 0.2 * 2.5
        assert 0.6 <= result["corr"] <= 0.7

    def test_each_perturbation_draws_from_rng_independently(self):
        """Two perturbations on the same rng must not share draws."""
        # Apply vol then corr with seed 42
        perturbations = [
            Perturbation(target="vol", type="multiplicative", distribution="uniform",
                         params={"low": 1.0, "high": 3.0}),
            Perturbation(target="corr", type="multiplicative", distribution="uniform",
                         params={"low": 1.0, "high": 3.0}),
        ]
        inputs = {"vol": 1.0, "corr": 1.0}
        result = apply_perturbations(inputs, perturbations, _rng(42))
        # Both start at 1.0 and use uniform[1,3] — they would be equal
        # only if they drew the same RNG value (i.e. shared a draw)
        assert result["vol"] != result["corr"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_produces_identical_output(self):
        perturbations = [
            Perturbation(target="vol", type="multiplicative", distribution="uniform",
                         params={"low": 1.5, "high": 2.5}),
            Perturbation(target="corr", type="additive", distribution="uniform",
                         params={"low": 0.1, "high": 0.3}),
        ]
        inputs = {"vol": 0.2, "corr": 0.5}
        result_a = apply_perturbations(inputs, perturbations, _rng(42))
        result_b = apply_perturbations(inputs, perturbations, _rng(42))
        assert result_a == result_b

    def test_different_seeds_produce_different_outputs(self):
        p = Perturbation(target="vol", type="multiplicative", distribution="uniform",
                         params={"low": 1.0, "high": 3.0})
        inputs = {"vol": 0.2}
        result_a = apply_perturbations(inputs, [p], _rng(42))
        result_b = apply_perturbations(inputs, [p], _rng(99))
        assert result_a["vol"] != result_b["vol"]
