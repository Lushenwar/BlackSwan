"""
PlausibilityValidator for BlackSwan stress tests.

Before each iteration's perturbed inputs are passed to the target function,
the validator checks that all constrained keys remain within plausible bounds.
Iterations that violate a constraint are silently skipped — they do not count
toward findings and do not increment the completed-iteration counter.

Public API:
    ValidationError               — raised by PlausibilityConstraint.check()
    PlausibilityConstraint        — one bound check on one input key
    PlausibilityValidator         — owns a list of constraints; never raises
"""

from dataclasses import dataclass


class ValidationError(Exception):
    """Raised by PlausibilityConstraint.check() when a value violates a bound."""


@dataclass(frozen=True)
class PlausibilityConstraint:
    """
    A single plausibility bound on one input key.

    Attributes:
        target:    Key in the inputs dict to inspect.
        min_value: Inclusive lower bound. None = no lower bound.
        max_value: Inclusive upper bound. None = no upper bound.

    If the target key is absent from the inputs dict the constraint is a no-op
    (silently ignored). This mirrors the behaviour of Perturbation.target when
    a key is absent from the inputs.
    """

    target: str
    min_value: float | None = None
    max_value: float | None = None

    def check(self, inputs: dict) -> None:
        """
        Validate one inputs dict against this constraint.

        Args:
            inputs: The perturbed inputs dict for one simulation iteration.

        Raises:
            ValidationError: If the value violates the bound.
        """
        if self.target not in inputs:
            return  # no-op: key absent

        val = inputs[self.target]

        if self.min_value is not None and val < self.min_value:
            raise ValidationError(
                f"'{self.target}' = {val:.6g} is below minimum {self.min_value}"
            )
        if self.max_value is not None and val > self.max_value:
            raise ValidationError(
                f"'{self.target}' = {val:.6g} exceeds maximum {self.max_value}"
            )


class PlausibilityValidator:
    """
    Runs a list of PlausibilityConstraints against an inputs dict.

    Design notes:
    - validate() never raises — it returns a bool so that the runner loop
      can use a simple conditional without a try/except at the call site.
    - Each constraint is checked in list order; the first violation short-circuits.
    """

    def __init__(self, constraints: list[PlausibilityConstraint]) -> None:
        self._constraints = constraints

    def validate(self, inputs: dict) -> bool:
        """
        Check all constraints against inputs.

        Returns:
            True  — all constraints satisfied (iteration should proceed).
            False — at least one constraint violated (iteration should be skipped).

        Never raises. Any ValidationError from a constraint is caught and
        converted to a False return value.
        """
        for constraint in self._constraints:
            try:
                constraint.check(inputs)
            except Exception:
                # Catch ValidationError and any unexpected TypeError/ValueError
                # that can arise from comparing incomparable types (e.g. None).
                # validate() must never propagate an exception to the caller.
                return False
        return True
