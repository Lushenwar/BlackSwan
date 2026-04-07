"""
Auto-Detector Tagging for BlackSwan (Phase 2, Step 2C).

AutoTagger reads a Python source file through ASTAnalyzer and produces:

  LineTagMap  — a dict-backed mapping of source line numbers to the names of
                the FailureDetector subclasses that should watch those lines.
                Used by the DAG visualiser (Phase 3) to highlight risky nodes
                and by the CLI to select the minimal detector suite.

  detector_suite() — the minimal list of FailureDetector instances for
                     StressRunner.  NaNInfDetector and BoundsDetector are
                     always present.  The three fragility detectors
                     (DivisionStabilityDetector, MatrixPSDDetector,
                     ConditionNumberDetector) are included only when their
                     corresponding AST pattern appears in the source.

Design principle (per CLAUDE.md):
  "No user configuration required."  The tagging is fully automatic and
  deterministic given a source file.  Expanding to new detector types only
  requires adding a new entry in _TAGGING_RULES and a new method on
  ASTAnalyzer (or reusing an existing one).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .ast_analyzer import ASTAnalyzer

# ---------------------------------------------------------------------------
# Detector name constants — single source of truth used by tests and callers
# ---------------------------------------------------------------------------

DETECTOR_DIVISION    = "DivisionStabilityDetector"
DETECTOR_MATRIX_PSD  = "MatrixPSDDetector"
DETECTOR_COND_NUMBER = "ConditionNumberDetector"
DETECTOR_NAN_INF     = "NaNInfDetector"
DETECTOR_BOUNDS      = "BoundsDetector"

# Map from detector name → callable that extracts the relevant line numbers
# from a SourceAnalysis.  Adding a new detector only requires a new entry here.
_TAGGING_RULES: dict[str, str] = {
    DETECTOR_DIVISION:    "division_lines",
    DETECTOR_MATRIX_PSD:  "matrix_construct_lines",
    DETECTOR_COND_NUMBER: "inv_call_lines",
}


# ---------------------------------------------------------------------------
# LineTagMap
# ---------------------------------------------------------------------------

@dataclass
class LineTagMap:
    """
    Immutable mapping from source line numbers to required detector names.

    Access patterns:
        tag_map.tags_for_line(32)      → ["MatrixPSDDetector"]
        tag_map.all_tags()             → {"MatrixPSDDetector", ...}
        tag_map.tagged_lines()         → {32: ["MatrixPSDDetector"], ...}
        tag_map.is_empty()             → True / False
    """

    _lines: dict[int, list[str]] = field(default_factory=dict)

    def tags_for_line(self, lineno: int) -> list[str]:
        """Return a copy of the tags list for *lineno* (empty list if untagged)."""
        return list(self._lines.get(lineno, []))

    def all_tags(self) -> set[str]:
        """Return the set of all distinct detector names across all tagged lines."""
        return {tag for tags in self._lines.values() for tag in tags}

    def tagged_lines(self) -> dict[int, list[str]]:
        """Return a shallow copy of the internal line→tags mapping."""
        return dict(self._lines)

    def is_empty(self) -> bool:
        """True when no lines have been tagged."""
        return not self._lines


# ---------------------------------------------------------------------------
# AutoTagger
# ---------------------------------------------------------------------------

class AutoTagger:
    """
    Derives a LineTagMap and an optimised detector suite from AST analysis.

    Usage:
        tagger   = AutoTagger(source_path)
        tag_map  = tagger.build()           # LineTagMap
        suite    = tagger.detector_suite()  # list[FailureDetector]

    Both methods are idempotent and reentrant; they re-derive from the cached
    SourceAnalysis on every call.
    """

    def __init__(self, source_path: str | Path) -> None:
        self._analysis = ASTAnalyzer(source_path).analyze()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> LineTagMap:
        """
        Return a LineTagMap mapping each relevant source line to the
        detector names required for it.

        Each line appears at most once per detector name (no duplicates).
        """
        lines: dict[int, list[str]] = {}

        for detector_name, attr in _TAGGING_RULES.items():
            for lineno in getattr(self._analysis, attr):
                bucket = lines.setdefault(lineno, [])
                if detector_name not in bucket:
                    bucket.append(detector_name)

        return LineTagMap(_lines=lines)

    def detector_suite(self) -> list:
        """
        Return the minimal list of FailureDetector instances for StressRunner.

        NaNInfDetector  — always active (catches any computation producing
                          NaN or Inf, including exceptions from the target).
        BoundsDetector  — always active (checks plausibility of outputs).
        Fragility detectors — included only when their AST pattern appears:
            DivisionStabilityDetector  if any division found in source
            MatrixPSDDetector          if any matrix-construction found
            ConditionNumberDetector    if any inversion call found
        """
        # Import here to avoid circular imports; detectors do not import parser.
        from ..detectors.numerical import NaNInfDetector, DivisionStabilityDetector
        from ..detectors.matrix import MatrixPSDDetector, ConditionNumberDetector
        from ..detectors.portfolio import BoundsDetector

        active_tags = self.build().all_tags()

        suite = [NaNInfDetector()]

        if DETECTOR_DIVISION in active_tags:
            suite.append(DivisionStabilityDetector())
        if DETECTOR_MATRIX_PSD in active_tags:
            suite.append(MatrixPSDDetector())
        if DETECTOR_COND_NUMBER in active_tags:
            suite.append(ConditionNumberDetector())

        suite.append(BoundsDetector())
        return suite
