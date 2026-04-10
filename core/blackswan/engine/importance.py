"""
ImportanceSampler — biased perturbation sampling for BlackSwan.

Standard Monte Carlo samples perturbations uniformly from the scenario's
defined ranges. This wastes evaluations in the center of the distribution
where failures are rare. ImportanceSampler maintains a lightweight record
of where failures have been observed and biases subsequent draws toward those
regions.

Strategy:
  - No failures seen yet      → uniform sampling (identical to Monte Carlo)
  - Failures observed (≥5)    → 70% tail-biased (near known failure points),
                                  30% uniform exploration
  - Bandwidth controls spread of tail samples relative to each parameter's range

The sampler is stateless from the caller's perspective: call record_failure()
after each detected failure, then sample() for the next perturbation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ImportanceSampler:
    """
    Biases perturbation sampling toward regions where failures have been observed.

    Falls back to uniform sampling until min_failures_before_bias observations
    have been collected (default: 5). This prevents premature exploitation of
    a single lucky failure hit.

    Thread-safety: instances are not thread-safe. Create one per runner.
    """

    def __init__(
        self,
        param_ranges: dict[str, tuple[float, float]],
        bandwidth: float = 0.2,
        tail_fraction: float = 0.7,
        min_failures_before_bias: int = 5,
    ) -> None:
        """
        Args:
            param_ranges: dict mapping param name → (lo, hi) bounds.
            bandwidth:    Controls spread of tail samples as a fraction of each
                          parameter's range. 0.2 = samples within ±20% of the
                          known failure point's range.
            tail_fraction: Probability of sampling near a known failure point
                           (vs. uniform exploration). Default 0.7.
            min_failures_before_bias: Number of recorded failures required before
                                      tail sampling activates.
        """
        self._ranges = param_ranges
        self._bandwidth = bandwidth
        self._tail_fraction = tail_fraction
        self._min = min_failures_before_bias
        self._failure_points: list[dict[str, float]] = []

    def record_failure(self, inputs: dict[str, Any]) -> None:
        """
        Record a set of perturbed inputs that caused a failure.
        Only scalar float/int values are recorded; arrays are skipped.
        """
        point: dict[str, float] = {}
        for k, v in inputs.items():
            if k in self._ranges and isinstance(v, (int, float)):
                point[k] = float(v)
        if point:
            self._failure_points.append(point)

    def sample(self, rng: np.random.Generator) -> dict[str, float]:
        """
        Draw one parameter set from the importance-weighted distribution.

        Returns a dict with the same keys as param_ranges, all values clamped
        to [lo, hi] for each parameter.
        """
        if len(self._failure_points) < self._min or rng.random() > self._tail_fraction:
            return self._uniform_sample(rng)
        return self._tail_sample(rng)

    def reset(self) -> None:
        """Clear all recorded failure points (useful between GA runs)."""
        self._failure_points.clear()

    @property
    def failure_count(self) -> int:
        return len(self._failure_points)

    # ------------------------------------------------------------------

    def _uniform_sample(self, rng: np.random.Generator) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, (lo, hi) in self._ranges.items():
            result[name] = float(rng.uniform(lo, hi))
        return result

    def _tail_sample(self, rng: np.random.Generator) -> dict[str, float]:
        """
        Pick a random known failure point, add Gaussian noise scaled to
        bandwidth × parameter range, clamp to [lo, hi].
        """
        idx = int(rng.integers(0, len(self._failure_points)))
        center = self._failure_points[idx]

        result: dict[str, float] = {}
        for name, (lo, hi) in self._ranges.items():
            spread = self._bandwidth * (hi - lo)
            base = center.get(name, rng.uniform(lo, hi))
            sample = float(rng.normal(base, spread))
            result[name] = float(np.clip(sample, lo, hi))
        return result
