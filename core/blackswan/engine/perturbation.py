"""
Perturbation primitives for BlackSwan stress tests.

A Perturbation describes one rule for modifying a single input value.
apply_perturbations() applies a list of rules to an inputs dict and returns
a new dict — the original is never mutated.

Supported types:
    multiplicative  new_val = base * factor     (factor drawn from distribution)
    additive        new_val = base + shift       (shift drawn from distribution)

Supported distributions:
    uniform         params: {"low": float, "high": float}
    lognormal       params: {"mu": float, "sigma": float}

Optional clamping:
    clamp=(lo, hi)  clips the output value; use None for an open bound.
                    e.g. clamp=(None, 0.99) → cap at 0.99
                         clamp=(0.0, None)  → floor at 0.0
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Perturbation:
    """
    A single perturbation rule applied to one key in the inputs dict.

    Attributes:
        target:       Key in the inputs dict to perturb.
        type:         "multiplicative" or "additive".
        distribution: "uniform" or "lognormal".
        params:       Distribution parameters.
                        uniform:   {"low": float, "high": float}
                        lognormal: {"mu": float, "sigma": float}
        clamp:        Optional (lo, hi) tuple to clip the output value.
                      Either element may be None for an open bound.
    """

    target: str
    type: str
    distribution: str
    params: dict
    clamp: tuple[float | None, float | None] | None = None


def apply_perturbations(
    inputs: dict,
    perturbations: list[Perturbation],
    rng: np.random.Generator,
) -> dict:
    """
    Apply a list of perturbation rules to an inputs dict.

    Returns a new dict. Keys not targeted by any rule are copied unchanged.
    The original dict is never modified. Each perturbation draws exactly one
    value from rng in the order they appear in the list.

    Args:
        inputs:       Base input values.
        perturbations: Ordered list of rules to apply.
        rng:          NumPy Generator (e.g. np.random.default_rng(seed)).
                      Caller owns the generator; results are deterministic
                      provided rng is seeded identically across calls.

    Returns:
        New dict with perturbed values.
    """
    result = dict(inputs)

    for p in perturbations:
        if p.target not in result:
            continue

        base = result[p.target]
        draw = _draw(p, rng)

        if p.type == "multiplicative":
            new_val = base * draw
        elif p.type == "additive":
            new_val = base + draw
        else:
            raise ValueError(
                f"Unknown perturbation type {p.type!r}. "
                "Expected 'multiplicative' or 'additive'."
            )

        if p.clamp is not None:
            lo, hi = p.clamp
            if lo is not None:
                new_val = max(new_val, lo)
            if hi is not None:
                new_val = min(new_val, hi)

        result[p.target] = new_val

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw(p: Perturbation, rng: np.random.Generator) -> float:
    """Draw a single sample from the perturbation's distribution."""
    if p.distribution == "uniform":
        return float(rng.uniform(p.params["low"], p.params["high"]))

    if p.distribution == "lognormal":
        return float(rng.lognormal(p.params["mu"], p.params["sigma"]))

    raise ValueError(
        f"Unknown distribution {p.distribution!r}. "
        "Expected 'uniform' or 'lognormal'."
    )
