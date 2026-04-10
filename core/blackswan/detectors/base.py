"""
Abstract base class and shared data types for all BlackSwan failure detectors.

A detector receives the output of one simulation iteration and decides whether
a failure condition is present. It knows nothing about line attribution — that
is handled by the attribution layer after findings are collected.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TriggerDisclosure:
    """
    Exact numerical threshold that caused a detector to fire.

    Provides full transparency about the detector's decision boundary so users
    can verify, tune, or override thresholds without reading source code.

    Examples:
        TriggerDisclosure(
            detector_name="MatrixPSDDetector",
            observed_value=-0.0034,
            threshold=-1e-10,
            comparison="<",
            explanation="Minimum eigenvalue -0.0034 fell below threshold -1e-10",
        )
    """
    detector_name: str
    """Name of the detector class that fired."""

    observed_value: float | str
    """The actual numerical value that crossed the threshold."""

    threshold: float | str
    """The configured threshold value."""

    comparison: str
    """Comparison operator that defines the trigger: '>' | '<' | '>=' | '<=' | '==' | '!='"""

    explanation: str
    """Human-readable explanation of why this value triggered the detector."""


@dataclass
class Finding:
    """
    A single failure observed in one simulation iteration.

    The runner aggregates Findings across all iterations to produce
    ShatterPoints in the final JSON response. Line and column attribution
    are filled in by the attribution layer; detectors leave them as None.
    """

    failure_type: str
    """Machine-readable category matching the failure_type enum in schema.json."""

    severity: str
    """'critical' | 'warning' | 'info'"""

    message: str
    """Human-readable explanation. Must state what failed and the observed value."""

    iteration: int
    """0-indexed iteration number in which this finding occurred."""

    line: int | None = None
    """Source line where failure manifests. Filled in by the attribution layer."""

    column: int | None = None
    """Column offset within that line. Filled in by the attribution layer."""

    exc_frames: list[tuple[str, int]] = field(default_factory=list)
    """
    Traceback frames captured when the target function raised an exception.
    Each entry is (filename, lineno). Populated by the runner; empty for
    detector-based findings where no exception was raised.
    The attribution layer uses these to resolve the proximate source line.
    """

    trigger_disclosure: TriggerDisclosure | None = None
    """
    Exact threshold details that caused this detector to fire.
    Populated by concrete detectors; None for exception-based findings.
    Provides full transparency into the detector's decision boundary.
    """


class FailureDetector(ABC):
    """
    Interface that all BlackSwan detectors must implement.

    Each concrete detector targets one class of numerical failure and is
    responsible only for recognising it — not for attribution or aggregation.
    """

    @abstractmethod
    def check(self, inputs: dict, output: Any, iteration: int) -> Finding | None:
        """
        Examine the output of one simulation iteration.

        Args:
            inputs:    The perturbed inputs passed to the target function this
                       iteration (provided for context; detectors may ignore it).
            output:    The value returned by the target function. May be a scalar,
                       NumPy array, Pandas DataFrame, or dict of any of the above.
            iteration: 0-indexed iteration counter.

        Returns:
            A Finding if a failure condition is detected, otherwise None.
            Return at most one Finding per call; the most severe one if multiple
            conditions are present simultaneously.
        """
