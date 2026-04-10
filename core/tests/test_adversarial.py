"""Tests for blackswan.engine.adversarial — GA primitive types (Task B1) and EvolutionaryStressRunner (Task B2)."""

import numpy as np
import pytest

from blackswan.detectors.base import Finding
from blackswan.detectors.matrix import MatrixPSDDetector
from blackswan.engine.adversarial import (
    Individual,
    EvolutionaryStressRunner,
    compute_fitness,
    crossover,
    mutate,
)
from blackswan.engine.runner import RunResult
from blackswan.scenarios.registry import load_scenario


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


# ---------------------------------------------------------------------------
# Inline fixture for EvolutionaryStressRunner tests
# ---------------------------------------------------------------------------

def _broken_cov(
    weights=np.array([0.3, 0.3, 0.4]),
    vol=np.array([0.15, 0.20, 0.10]),
    correlation=0.0,
) -> np.ndarray:
    n = len(vol)
    corr_val = 0.8 + correlation
    corr_matrix = np.full((n, n), corr_val)
    np.fill_diagonal(corr_matrix, 1.0)
    return np.diag(vol) @ corr_matrix @ np.diag(vol)


class TestEvolutionaryStressRunner:
    def _runner(self, generations=5, pop_size=20):
        scenario = load_scenario("liquidity_crash")
        return EvolutionaryStressRunner(
            fn=_broken_cov,
            base_inputs={"correlation": 0.0},
            scenario=scenario,
            detectors=[MatrixPSDDetector()],
            seed=42,
            n_generations=generations,
            population_size=pop_size,
            elite_fraction=0.2,
        )

    def test_returns_run_result(self):
        result = self._runner().run()
        assert isinstance(result, RunResult)

    def test_runtime_ms_non_negative(self):
        result = self._runner().run()
        assert result.runtime_ms >= 0

    def test_seed_is_stored(self):
        result = self._runner().run()
        assert result.seed == 42

    def test_finds_failures_on_broken_fixture(self):
        result = self._runner(generations=5, pop_size=20).run()
        assert len(result.findings) > 0
        assert any(f.failure_type == "non_psd_matrix" for f in result.findings)

    def test_same_seed_deterministic(self):
        r1 = self._runner(generations=5, pop_size=20).run()
        r2 = self._runner(generations=5, pop_size=20).run()
        assert len(r1.findings) == len(r2.findings)

    def test_iterations_completed_equals_generations_times_population(self):
        result = self._runner(generations=3, pop_size=10).run()
        assert result.iterations_completed == 3 * 10

    def test_baseline_established_true_on_clean_fn(self):
        result = self._runner().run()
        assert result.baseline_established is True

    def test_finds_failure_under_500_iterations(self):
        # 25 generations × 20 pop = 500 evaluations max
        runner = EvolutionaryStressRunner(
            fn=_broken_cov,
            base_inputs={"correlation": 0.0},
            scenario=load_scenario("liquidity_crash"),
            detectors=[MatrixPSDDetector()],
            seed=42,
            n_generations=25,
            population_size=20,
            elite_fraction=0.2,
        )
        result = runner.run()
        psd = [f for f in result.findings if f.failure_type == "non_psd_matrix"]
        assert len(psd) > 0
        assert result.iterations_completed <= 500
