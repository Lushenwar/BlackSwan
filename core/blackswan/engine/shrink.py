"""
ShrinkingEngine — reduces a failing input to its minimal reproducing form.

Given an input dict that causes a failure, the shrinker iteratively removes
irrelevant dimensions, bisects magnitudes toward base values, and rounds
to fewest significant figures — all while verifying the failure still occurs.

Three-layer pipeline (applied in sequence):
  Layer 1: Dimensional reduction — remove parameters unnecessary for failure.
  Layer 2: Magnitude reduction  — binary-search each param toward its base value.
  Layer 3: Precision reduction  — round values to fewest significant figures.

The shrinker is constraint-aware: PlausibilityValidator is consulted before
each candidate is tested so invalid inputs (e.g. correlation > 1) are never
passed to the target function.

ConstraintRepairPipeline is a separate component used by the GA (adversarial.py)
to repair mutated individuals. It is defined here because both the shrinker
and the GA need it.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..detectors.base import FailureDetector, Finding
from ..engine.validator import PlausibilityValidator


# ---------------------------------------------------------------------------
# ConstraintRepairPipeline
# ---------------------------------------------------------------------------

@dataclass
class RepairResult:
    params: dict[str, Any]
    repair_applied: bool    # True if any value was changed during repair


class ConstraintRepairPipeline:
    """
    Repairs a parameter dict so it satisfies known mathematical constraints.

    Used by:
      - ShrinkingEngine: before testing each candidate
      - EvolutionaryStressRunner: after every crossover + mutation step

    Repairs are domain-specific clamps and projections. If a repair changes a
    value by more than 10% (relative), repair_applied is set to True in the result
    so callers can flag it in reports.
    """

    def repair(self, params: dict[str, Any]) -> RepairResult:
        """
        Apply all repair rules and return the repaired params with a flag
        indicating whether any repair changed a value materially (>10% relative).
        """
        repaired = dict(params)
        changed = False

        repaired, c = self._clamp_correlations(repaired)
        changed = changed or c

        repaired, c = self._clamp_volatilities(repaired)
        changed = changed or c

        repaired, c = self._clamp_probabilities(repaired)
        changed = changed or c

        return RepairResult(params=repaired, repair_applied=changed)

    def _clamp_correlations(self, params: dict) -> tuple[dict, bool]:
        """Correlation values must be in (-1, 1)."""
        changed = False
        for key in list(params.keys()):
            if "corr" in key.lower() or "correlation" in key.lower():
                val = params[key]
                if isinstance(val, (int, float)):
                    clamped = float(np.clip(val, -0.999, 0.999))
                    if abs(clamped - val) > 0.1 * max(abs(val), 1e-9):
                        changed = True
                    params[key] = clamped
        return params, changed

    def _clamp_volatilities(self, params: dict) -> tuple[dict, bool]:
        """Volatility values must be strictly positive."""
        changed = False
        for key in list(params.keys()):
            if "vol" in key.lower() or "sigma" in key.lower():
                val = params[key]
                if isinstance(val, (int, float)) and val <= 0:
                    params[key] = 1e-8
                    changed = True
        return params, changed

    def _clamp_probabilities(self, params: dict) -> tuple[dict, bool]:
        """Parameters named 'prob', 'weight', 'rate' should be in [0, 1]."""
        changed = False
        for key in list(params.keys()):
            lower = key.lower()
            if any(tok in lower for tok in ("prob", "weight", "rate", "frac")):
                val = params[key]
                if isinstance(val, (int, float)):
                    clamped = float(np.clip(val, 0.0, 1.0))
                    if abs(clamped - val) > 0.1 * max(abs(val), 1e-9):
                        changed = True
                    params[key] = clamped
        return params, changed


# ---------------------------------------------------------------------------
# ShrinkingEngine
# ---------------------------------------------------------------------------

_REPAIR_PIPELINE = ConstraintRepairPipeline()


class ShrinkingEngine:
    """
    Reduces a failing input dict to its minimal reproducing form.

    Usage:
        shrinker = ShrinkingEngine(fn, base_inputs, detectors, validator, seed=42)
        minimal = shrinker.shrink(failing_inputs, original_finding)

    Guarantees:
      - Output always causes the same failure_type as original_finding.
      - Output always passes PlausibilityValidator (if provided).
      - Output is as close to base_inputs as possible while still failing.
    """

    def __init__(
        self,
        fn: Callable,
        base_inputs: dict[str, Any],
        detectors: list[FailureDetector],
        validator: PlausibilityValidator | None = None,
        max_steps: int = 500,
        seed: int = 0,
    ) -> None:
        self._fn = fn
        self._base_inputs = base_inputs
        self._detectors = detectors
        self._validator = validator
        self._max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        self._step_count = 0

    def shrink(
        self,
        failing_inputs: dict[str, Any],
        original_finding: Finding,
    ) -> dict[str, Any]:
        """
        Run all three shrinking layers and return the smallest failing input.
        If no layer reduces the input further, returns failing_inputs unchanged.
        """
        self._step_count = 0
        self._original_finding = original_finding

        current = copy.deepcopy(failing_inputs)

        current = self._layer_dimensional_reduction(current)
        current = self._layer_magnitude_reduction(current)
        current = self._layer_precision_reduction(current)

        return current

    # ------------------------------------------------------------------
    # Layer 1: Dimensional reduction
    # ------------------------------------------------------------------

    def _layer_dimensional_reduction(self, inputs: dict) -> dict:
        """
        For each perturbed parameter, try resetting it to its base value.
        If the failure still occurs, keep the reset — the parameter is irrelevant.
        Repeat until no further reduction is possible.
        """
        current = dict(inputs)
        improved = True
        while improved and self._step_count < self._max_steps:
            improved = False
            for key in list(current.keys()):
                if key not in self._base_inputs:
                    continue
                base_val = self._base_inputs[key]
                if _values_equal(current[key], base_val):
                    continue  # already at base
                candidate = dict(current)
                candidate[key] = base_val
                if self._still_fails(candidate):
                    current = candidate
                    improved = True
                    self._step_count += 1
        return current

    # ------------------------------------------------------------------
    # Layer 2: Magnitude reduction (binary search)
    # ------------------------------------------------------------------

    def _layer_magnitude_reduction(self, inputs: dict) -> dict:
        """
        Binary-search each parameter between its current failing value and its
        base value. Find the smallest deviation from base that still causes failure.
        """
        current = dict(inputs)
        for key in list(current.keys()):
            if key not in self._base_inputs:
                continue
            if self._step_count >= self._max_steps:
                break
            base_val = self._base_inputs[key]
            failing_val = current[key]
            if not isinstance(failing_val, (int, float)):
                continue
            if _values_equal(failing_val, base_val):
                continue

            # Bisect: shrink toward base while failure persists
            lo, hi = float(base_val), float(failing_val)
            for _ in range(20):
                if self._step_count >= self._max_steps:
                    break
                mid = (lo + hi) / 2.0
                candidate = dict(current)
                candidate[key] = mid
                if self._still_fails(candidate):
                    hi = mid     # closer to base still fails — keep shrinking
                else:
                    lo = mid     # too close to base, failure disappeared
                self._step_count += 1

            current[key] = hi    # last value that still caused failure

        return current

    # ------------------------------------------------------------------
    # Layer 3: Precision reduction
    # ------------------------------------------------------------------

    def _layer_precision_reduction(self, inputs: dict) -> dict:
        """
        Round each scalar parameter to progressively fewer significant figures.
        Finds the coarsest representation that still causes failure.
        """
        current = dict(inputs)
        for key in list(current.keys()):
            if self._step_count >= self._max_steps:
                break
            val = current[key]
            if not isinstance(val, (int, float)):
                continue
            for sig_figs in [4, 3, 2, 1]:
                if self._step_count >= self._max_steps:
                    break
                rounded = _round_significant(val, sig_figs)
                candidate = dict(current)
                candidate[key] = rounded
                if self._still_fails(candidate):
                    current[key] = rounded
                    val = rounded
                self._step_count += 1

        return current

    # ------------------------------------------------------------------
    # Internal: failure verification
    # ------------------------------------------------------------------

    def _still_fails(self, inputs: dict) -> bool:
        """
        Return True iff calling fn(**repaired_inputs) produces the same
        failure_type as the original finding, and the inputs pass validation.
        """
        # Apply constraint repair
        repair_result = _REPAIR_PIPELINE.repair(dict(inputs))
        repaired = repair_result.params

        # Validate against plausibility constraints
        if self._validator is not None and not self._validator.validate(repaired):
            return False

        try:
            output = self._fn(**repaired)
        except Exception as exc:
            # If original was an exception-class failure, any exception counts
            if self._original_finding.exc_frames:
                return True
            # Otherwise check if any detector fires
            return False

        # Check if any detector fires with the same failure_type
        for detector in self._detectors:
            finding = detector.check(inputs=repaired, output=output, iteration=-1)
            if finding is not None and finding.failure_type == self._original_finding.failure_type:
                return True

        return False


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _round_significant(value: float, sig_figs: int) -> float:
    """Round value to sig_figs significant figures."""
    if value == 0.0:
        return 0.0
    try:
        magnitude = math.floor(math.log10(abs(value)))
        factor = 10 ** (sig_figs - 1 - magnitude)
        return round(value * factor) / factor
    except (ValueError, OverflowError):
        return value


def _values_equal(a: Any, b: Any) -> bool:
    """Equality check that handles numpy arrays and scalars."""
    try:
        import numpy as np
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np.array_equal(a, b)
    except ImportError:
        pass
    return a == b
