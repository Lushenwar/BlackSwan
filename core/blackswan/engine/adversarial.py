"""
Evolutionary (Genetic Algorithm) stress runner for BlackSwan.

Replaces random Monte Carlo sampling with a population-based search that
evolves perturbation sets toward maximum failure severity.

Public API (built across tasks B1–B3):
    Individual                — one candidate perturbation parameter set
    compute_fitness(findings) — score a list of findings (higher = worse)
    crossover(p1, p2, rng)   — produce a child by mixing two parents
    mutate(ind, rng, scale)  — apply Gaussian noise to individual's params
    EvolutionaryStressRunner  — drop-in replacement for StressRunner (added B2)
"""

import copy
import time
import traceback as tb
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..detectors.base import FailureDetector, Finding
from ..engine.perturbation import Perturbation, apply_perturbations
from .runner import RunResult


@dataclass
class Individual:
    """
    One candidate in the GA population.

    params: dict mapping parameter names to float values.
            Key convention: "<perturbation_target>_<param_name>"
            e.g. {"vol_low": 1.5, "vol_high": 3.0, "corr_low": 0.1}
    fitness: Score after evaluation. Higher = caused more severe failure.
             Starts at 0.0; set by the runner after evaluation.
    """

    params: dict[str, float]
    fitness: float = 0.0


_SEVERITY_WEIGHT = {"critical": 3.0, "warning": 1.5, "info": 0.5}
_TYPE_BONUS = {"non_psd_matrix": 1.5, "ill_conditioned_matrix": 1.2}


def compute_fitness(findings: list[Finding]) -> float:
    """
    Score a list of findings. Higher = more severe failures.
    Returns 0.0 for empty list.
    Each finding contributes: severity_weight * type_bonus (1.0 if no bonus).
    """
    if not findings:
        return 0.0

    total = 0.0
    for finding in findings:
        weight = _SEVERITY_WEIGHT.get(finding.severity, 0.0)
        bonus = _TYPE_BONUS.get(finding.failure_type, 1.0)
        total += weight * bonus

    return total


def crossover(parent1: Individual, parent2: Individual, rng: np.random.Generator) -> Individual:
    """
    Produce a child by randomly choosing each param from one parent (50/50 coin flip).
    Child fitness starts at 0.0 (not yet evaluated).
    Original parents are not modified.
    Keys come from parent1 — both parents must have same keys.
    """
    child_params: dict[str, float] = {}
    for key in parent1.params:
        if rng.random() < 0.5:
            child_params[key] = parent1.params[key]
        else:
            child_params[key] = parent2.params[key]
    return Individual(params=child_params, fitness=0.0)


def mutate(individual: Individual, rng: np.random.Generator, noise_scale: float = 0.1) -> Individual:
    """
    Apply Gaussian noise to every param value. Returns a NEW Individual.
    Original is never modified.
    noise_scale is std dev of noise relative to the absolute value of each param:
        new_val = val + normal(0, noise_scale * max(abs(val), 1e-6))
    Child fitness starts at 0.0.
    """
    new_params: dict[str, float] = {}
    for key, val in individual.params.items():
        std = noise_scale * max(abs(val), 1e-6)
        noise = rng.normal(0.0, std)
        new_params[key] = val + noise
    return Individual(params=new_params, fitness=0.0)


# ---------------------------------------------------------------------------
# Module-level helpers for EvolutionaryStressRunner
# ---------------------------------------------------------------------------

def _extract_param_ranges(scenario: Any) -> dict[str, tuple[float, float]]:
    """
    Build a search space from a Scenario's perturbation list.

    For each Perturbation:
      distribution="uniform"   → keys "<target>_low" and "<target>_high"
                                  ranges: (low*0.5, low*2.0) and (hi*0.5, hi*2.0)
      distribution="lognormal" → keys "<target>_mu" and "<target>_sigma"
                                  ranges: (mu*0.5, mu*2.0) and (sigma*0.5, sigma*2.0)
    Other distributions: skip silently.
    """
    ranges: dict[str, tuple[float, float]] = {}
    for p in scenario.perturbations:
        if p.distribution == "uniform":
            low = p.params["low"]
            high = p.params["high"]
            ranges[f"{p.target}_low"] = (low * 0.5, low * 2.0)
            ranges[f"{p.target}_high"] = (high * 0.5, high * 2.0)
        elif p.distribution == "lognormal":
            mu = p.params["mu"]
            sigma = p.params["sigma"]
            ranges[f"{p.target}_mu"] = (mu * 0.5, mu * 2.0)
            ranges[f"{p.target}_sigma"] = (sigma * 0.5, sigma * 2.0)
        # other distributions: skip silently
    return ranges


def _initialise_population(
    param_ranges: dict[str, tuple[float, float]],
    size: int,
    rng: np.random.Generator,
) -> list[Individual]:
    """
    Generate `size` random Individuals with params drawn uniformly from param_ranges.
    Each Individual's fitness starts at 0.0.
    """
    population: list[Individual] = []
    for _ in range(size):
        params: dict[str, float] = {}
        for key, (lo, hi) in param_ranges.items():
            params[key] = float(rng.uniform(lo, hi))
        population.append(Individual(params=params, fitness=0.0))
    return population


def _apply_individual(
    base_inputs: dict,
    scenario: Any,
    individual: Individual,
    rng: np.random.Generator,
) -> dict:
    """
    Build perturbed inputs from base_inputs using an Individual's param overrides.

    For each Perturbation in scenario.perturbations:
      - Build new_params from p.params
      - If distribution="uniform": override "low"/"high" from individual if present
      - If distribution="lognormal": override "mu"/"sigma" from individual if present
      - Build a new Perturbation with the overridden params (same target, type, distribution, clamp)
    Call apply_perturbations(base_inputs, overridden_perturbations, rng) and return the result.
    """
    overridden: list[Perturbation] = []
    for p in scenario.perturbations:
        new_params = dict(p.params)
        if p.distribution == "uniform":
            low_key = f"{p.target}_low"
            high_key = f"{p.target}_high"
            if low_key in individual.params:
                new_params["low"] = individual.params[low_key]
            if high_key in individual.params:
                new_params["high"] = individual.params[high_key]
        elif p.distribution == "lognormal":
            mu_key = f"{p.target}_mu"
            sigma_key = f"{p.target}_sigma"
            if mu_key in individual.params:
                new_params["mu"] = individual.params[mu_key]
            if sigma_key in individual.params:
                new_params["sigma"] = individual.params[sigma_key]
        overridden.append(Perturbation(
            target=p.target,
            type=p.type,
            distribution=p.distribution,
            params=new_params,
            clamp=p.clamp,
        ))
    return apply_perturbations(base_inputs, overridden, rng)


# ---------------------------------------------------------------------------
# HardnessAdaptor
# ---------------------------------------------------------------------------

class HardnessAdaptor:
    """
    Adaptive intensity controller for EvolutionaryStressRunner.

    When the GA makes no progress (best fitness == 0.0 in a generation),
    hardness increases by `step`, capped at `max_hardness`.
    `range_scale()` returns a multiplier [1.0, 2.0] used to widen
    perturbation parameter ranges so the engine escapes flat search regions.

    Formula: range_scale = 1.0 + hardness
    """

    def __init__(self, step: float = 0.1, max_hardness: float = 1.0) -> None:
        self._step = step
        self._max = max_hardness
        self.hardness: float = 0.0

    def update(self, best_fitness: float) -> None:
        """Increase hardness when best_fitness == 0.0 (no failure found this generation)."""
        if best_fitness == 0.0:
            self.hardness = min(self.hardness + self._step, self._max)

    def range_scale(self) -> float:
        """Multiplier to apply to param ranges. Always >= 1.0."""
        return 1.0 + self.hardness


# ---------------------------------------------------------------------------
# EvolutionaryStressRunner
# ---------------------------------------------------------------------------

class EvolutionaryStressRunner:
    """
    Genetic Algorithm stress runner. Drop-in replacement for StressRunner.
    Returns RunResult with findings, runtime_ms, seed, iterations_completed,
    baseline_established.
    """

    def __init__(
        self,
        fn: Callable,
        base_inputs: dict,
        scenario: Any,
        detectors: list[FailureDetector],
        seed: int,
        n_generations: int = 20,
        population_size: int = 100,
        elite_fraction: float = 0.2,
        noise_scale: float = 0.1,
    ) -> None:
        self.fn = fn
        self.base_inputs = base_inputs
        self.scenario = scenario
        self.detectors = detectors
        self.seed = seed
        self.n_generations = n_generations
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.noise_scale = noise_scale
        self._hardness = HardnessAdaptor(step=0.05, max_hardness=1.0)

    def run(self) -> RunResult:
        """
        GA loop:
        1. rng = np.random.default_rng(self.seed)
        2. Call detector.reset() via hasattr on all detectors
        3. Run fn(**base_inputs) to get base_output; if raises, baseline_established=False
        4. Call detector.set_baseline(base_inputs, base_output) via hasattr if base_output not None
        5. Extract param_ranges from scenario via _extract_param_ranges
        6. Initialise population of size population_size
        7. n_elites = max(1, int(population_size * elite_fraction))
        8. For each generation:
            a. Evaluate each individual: _apply_individual → fn(**perturbed) → detectors → compute_fitness
            b. Store all findings in all_findings list
            c. Sort population by fitness descending, take top n_elites as elites
            d. Build next_pop: start with deep copies of elites, fill rest via crossover+mutate
            e. population = next_pop
        9. Return RunResult(...)
        """
        rng = np.random.default_rng(self.seed)
        start = time.monotonic()
        all_findings: list[Finding] = []

        # --- Reset detectors and establish baseline ---
        baseline_established = True
        for detector in self.detectors:
            if hasattr(detector, "reset"):
                detector.reset()

        try:
            base_output = self.fn(**self.base_inputs)
        except Exception:
            base_output = None
            baseline_established = False

        if base_output is not None:
            for detector in self.detectors:
                if hasattr(detector, "set_baseline"):
                    detector.set_baseline(self.base_inputs, base_output)

        # --- Build search space and initialise population ---
        param_ranges = _extract_param_ranges(self.scenario)
        population = _initialise_population(param_ranges, self.population_size, rng)
        n_elites = max(1, int(self.population_size * self.elite_fraction))
        total_iterations = 0

        # --- GA loop ---
        for generation in range(self.n_generations):
            # Evaluate each individual
            for idx, individual in enumerate(population):
                iteration_id = generation * self.population_size + idx
                perturbed = _apply_individual(
                    self.base_inputs, self.scenario, individual, rng
                )
                findings = self._evaluate_individual(perturbed, iteration_id)
                individual.fitness = compute_fitness(findings)
                all_findings.extend(findings)
                total_iterations += 1

            # Sort descending by fitness, select elites
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            elites = [copy.deepcopy(ind) for ind in population[:n_elites]]

            # Update hardness adaptor based on best elite fitness
            best_gen_fitness = max((ind.fitness for ind in population[:n_elites]), default=0.0)
            self._hardness.update(best_gen_fitness)
            scale = self._hardness.range_scale()

            # Build next generation
            next_pop: list[Individual] = list(elites)
            while len(next_pop) < self.population_size:
                # Pick two parents from elites (or full population if elites insufficient)
                parent_pool = elites if len(elites) >= 2 else population
                i1, i2 = rng.choice(len(parent_pool), size=2, replace=False)
                child = crossover(parent_pool[i1], parent_pool[i2], rng)
                child = mutate(child, rng, self.noise_scale * scale)
                next_pop.append(child)

            population = next_pop

        runtime_ms = int((time.monotonic() - start) * 1000)

        from .cluster import cluster_findings
        buckets = cluster_findings(
            findings=all_findings,
            total_iterations=total_iterations,
        )

        return RunResult(
            iterations_completed=total_iterations,
            findings=all_findings,
            root_cause_buckets=buckets,
            runtime_ms=runtime_ms,
            seed=self.seed,
            baseline_established=baseline_established,
            mode="adversarial",
        )

    def _evaluate_individual(self, perturbed: dict, iteration: int) -> list[Finding]:
        """
        Call fn(**perturbed), run all detectors, return findings list.
        If fn raises: capture traceback frames, return [Finding(failure_type="nan_inf", ...)].
        """
        try:
            output = self.fn(**perturbed)
        except Exception as exc:
            frames = [
                (frame.filename, frame.lineno)
                for frame in tb.extract_tb(exc.__traceback__)
            ]
            return [Finding(
                failure_type="nan_inf",
                severity="critical",
                message=f"{type(exc).__name__}: {exc}",
                iteration=iteration,
                exc_frames=frames,
            )]

        findings: list[Finding] = []
        for detector in self.detectors:
            finding = detector.check(
                inputs=perturbed,
                output=output,
                iteration=iteration,
            )
            if finding is not None:
                findings.append(finding)
        return findings
