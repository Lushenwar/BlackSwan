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
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..detectors.base import Finding


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
