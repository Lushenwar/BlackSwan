"""Tests for blackswan.engine.adversarial — GA primitive types (Task B1)."""

import numpy as np
import pytest

from blackswan.detectors.base import Finding
from blackswan.engine.adversarial import Individual, compute_fitness, crossover, mutate


class TestIndividual:
    def test_params_stored(self):
        ind = Individual(params={"vol_low": 1.5})
        assert ind.params["vol_low"] == 1.5

    def test_default_fitness_is_zero(self):
        ind = Individual(params={})
        assert ind.fitness == 0.0

    def test_fitness_settable(self):
        ind = Individual(params={})
        ind.fitness = 5.0
        assert ind.fitness == 5.0


class TestComputeFitness:
    def _finding(self, failure_type: str = "non_psd_matrix", severity: str = "critical") -> Finding:
        return Finding(
            failure_type=failure_type,
            severity=severity,
            message="test finding",
            iteration=0,
        )

    def test_no_findings_gives_zero(self):
        assert compute_fitness([]) == 0.0

    def test_critical_finding_positive(self):
        assert compute_fitness([self._finding(severity="critical")]) > 0.0

    def test_critical_higher_than_warning(self):
        critical_score = compute_fitness([self._finding(failure_type="nan_inf", severity="critical")])
        warning_score = compute_fitness([self._finding(failure_type="nan_inf", severity="warning")])
        assert critical_score > warning_score

    def test_more_findings_higher_score(self):
        one = compute_fitness([self._finding()])
        two = compute_fitness([self._finding(), self._finding()])
        assert two > one

    def test_non_psd_bonus_applied(self):
        non_psd_score = compute_fitness([self._finding(failure_type="non_psd_matrix", severity="critical")])
        nan_inf_score = compute_fitness([self._finding(failure_type="nan_inf", severity="critical")])
        # non_psd_matrix has a bonus of 1.5 vs no bonus (1.0) for nan_inf
        assert non_psd_score >= nan_inf_score

    def test_ill_conditioned_bonus_applied(self):
        ill_cond_score = compute_fitness([self._finding(failure_type="ill_conditioned_matrix", severity="critical")])
        plain_warning_score = compute_fitness([self._finding(failure_type="nan_inf", severity="warning")])
        # ill_conditioned_matrix: 3.0 * 1.2 = 3.6 vs nan_inf warning: 1.5 * 1.0 = 1.5
        assert ill_cond_score > plain_warning_score


class TestCrossover:
    def _make_rng(self, seed: int = 42) -> np.random.Generator:
        return np.random.default_rng(seed)

    def test_output_keys_match_parents(self):
        p1 = Individual(params={"a": 1.0, "b": 2.0})
        p2 = Individual(params={"a": 10.0, "b": 20.0})
        child = crossover(p1, p2, self._make_rng())
        assert set(child.params.keys()) == set(p1.params.keys())

    def test_each_param_from_one_parent(self):
        p1 = Individual(params={"a": 0.0})
        p2 = Individual(params={"a": 100.0})
        rng = self._make_rng(seed=7)
        for _ in range(30):
            child = crossover(p1, p2, rng)
            assert child.params["a"] in (0.0, 100.0), (
                f"Expected 0.0 or 100.0, got {child.params['a']}"
            )

    def test_child_fitness_starts_zero(self):
        p1 = Individual(params={"a": 1.0}, fitness=99.0)
        p2 = Individual(params={"a": 2.0}, fitness=99.0)
        child = crossover(p1, p2, self._make_rng())
        assert child.fitness == 0.0

    def test_original_parents_not_modified(self):
        p1 = Individual(params={"a": 1.0, "b": 2.0})
        p2 = Individual(params={"a": 10.0, "b": 20.0})
        p1_params_before = dict(p1.params)
        p2_params_before = dict(p2.params)
        crossover(p1, p2, self._make_rng())
        assert p1.params == p1_params_before
        assert p2.params == p2_params_before


class TestMutate:
    def _make_rng(self, seed: int = 0) -> np.random.Generator:
        return np.random.default_rng(seed)

    def test_mutation_changes_values(self):
        ind = Individual(params={"a": 1.0, "b": 2.0})
        mutated = mutate(ind, self._make_rng(seed=0), noise_scale=1.0)
        # With noise_scale=1.0 and seed=0, at least one value should differ
        assert mutated.params != ind.params

    def test_keys_preserved(self):
        ind = Individual(params={"vol_low": 1.5, "vol_high": 3.0})
        mutated = mutate(ind, self._make_rng())
        assert set(mutated.params.keys()) == set(ind.params.keys())

    def test_original_not_mutated(self):
        ind = Individual(params={"a": 1.0, "b": 2.0})
        original_params = dict(ind.params)
        mutate(ind, self._make_rng(), noise_scale=1.0)
        assert ind.params == original_params

    def test_child_fitness_starts_zero(self):
        ind = Individual(params={"a": 1.0}, fitness=42.0)
        mutated = mutate(ind, self._make_rng())
        assert mutated.fitness == 0.0
