"""
Traceback attribution: resolve a Finding's proximate source line.

TracebackResolver uses two strategies depending on how the failure was detected:

  1. Exception-based (exc_frames populated by the runner):
     Filter stored traceback frames to those in the user's source file.
     Take the last matching frame — it is the innermost user-code location
     where the exception propagated through.

  2. Detector-based (no exc_frames — detector inspected the output post-hoc):
     Walk the source file's AST to find the last assignment whose RHS
     contains a pattern that matches the failure type:
       non_psd_matrix / ill_conditioned_matrix → matrix construction
           (MatMult operator, or calls to np.outer/np.cov/np.corrcoef/np.dot)
       division_by_zero → Div or FloorDiv operator in an assignment RHS

     The "last" such assignment is returned because in a typical risk model
     the proximate failure site is the final matrix being constructed before
     it is used, not an earlier intermediate.

     Returns None when no matching assignment is found — the caller should
     treat an unresolved line as "attribution pending".

Assumption (documented per CLAUDE.md engineering guidelines):
  AST-based attribution is a Phase 1D heuristic. The full dependency-graph
  walk (Phase 2) will supersede this for multi-variable causal chains.
  This resolver is deliberately narrow: it finds the proximate site, not
  the root cause.
"""

import ast
import copy
import dataclasses
from pathlib import Path
from typing import Any

from ..detectors.base import Finding

# Failure types that use the matrix-construction heuristic
_MATRIX_FAILURE_TYPES = frozenset({"non_psd_matrix", "ill_conditioned_matrix"})

# Function names whose calls indicate matrix construction
_MATRIX_CALL_NAMES = frozenset({
    "cov", "corrcoef", "outer", "dot",           # numpy functions (short)
    "np.cov", "np.corrcoef", "np.outer", "np.dot",  # qualified
})


class TracebackResolver:
    """
    Resolves a Finding's proximate source line using stored traceback frames
    or AST pattern matching, depending on how the failure was detected.

    Safe to construct once and reuse for many findings against the same source.
    Never raises; returns the finding unchanged on any resolution failure.
    """

    def __init__(self, source_path: str | Path) -> None:
        self._path = Path(source_path).resolve()
        source = self._path.read_text(encoding="utf-8")
        self._lines = source.splitlines()
        self._tree = ast.parse(source, filename=str(self._path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, finding: Finding) -> Finding:
        """
        Return a copy of finding with .line filled in.

        If the finding already has a line set, it is returned unchanged
        (never overwrite prior attribution).  If resolution fails for any
        reason, the finding is returned with .line still None.
        """
        if finding.line is not None:
            return copy.copy(finding)

        line = self._resolve_line(finding)
        return dataclasses.replace(finding, line=line)

    # ------------------------------------------------------------------
    # Resolution strategies
    # ------------------------------------------------------------------

    def _resolve_line(self, finding: Finding) -> int | None:
        # Strategy 1: exception frames stored by the runner take precedence
        # over AST heuristics — they are the ground truth for exception-based
        # failures regardless of failure_type.
        if finding.exc_frames:
            return self._line_from_exc_frames(finding.exc_frames)

        # Strategy 2: AST heuristic by failure type
        if finding.failure_type in _MATRIX_FAILURE_TYPES:
            return self._last_matrix_construction_line()

        if finding.failure_type == "division_by_zero":
            return self._last_division_line()

        return None

    # ------------------------------------------------------------------
    # Strategy 1 — exception frames
    # ------------------------------------------------------------------

    def _line_from_exc_frames(
        self, frames: list[tuple[str, int]]
    ) -> int | None:
        """
        Return the lineno of the last frame whose filename resolves to our
        source file.  "Last" = most recent = innermost user-code frame.
        """
        result = None
        for filename, lineno in frames:
            try:
                if Path(filename).resolve() == self._path:
                    result = lineno
            except (ValueError, OSError):
                continue
        return result

    # ------------------------------------------------------------------
    # Strategy 2 — AST heuristics
    # ------------------------------------------------------------------

    def _last_matrix_construction_line(self) -> int | None:
        """
        Find the last assignment in the source whose RHS contains a
        matrix-construction pattern: the @ (MatMult) operator, or a call
        to one of the known covariance/correlation/outer-product functions.

        "Last" assignment is chosen because in a typical model the covariance
        matrix is the final intermediate computed before the risk metric —
        i.e. the proximate failure site.
        """
        candidate: int | None = None
        for node in ast.walk(self._tree):
            if not isinstance(node, ast.Assign):
                continue
            if _rhs_contains_matrix_pattern(node.value):
                candidate = node.lineno
        return candidate

    def _last_division_line(self) -> int | None:
        """
        Find the last assignment in the source whose RHS contains a division
        (Div or FloorDiv) operator.
        """
        candidate: int | None = None
        for node in ast.walk(self._tree):
            if not isinstance(node, ast.Assign):
                continue
            if _rhs_contains_division(node.value):
                candidate = node.lineno
        return candidate


# ---------------------------------------------------------------------------
# AST predicate helpers
# ---------------------------------------------------------------------------

def _rhs_contains_matrix_pattern(node: ast.expr) -> bool:
    """
    Return True if the expression subtree contains a matrix-construction
    pattern: MatMult operator or a call to a known matrix function.
    """
    for child in ast.walk(node):
        # @ operator (matrix multiply)
        if isinstance(child, ast.BinOp) and isinstance(child.op, ast.MatMult):
            return True
        # Known function calls: np.outer, np.cov, np.corrcoef, np.dot, etc.
        if isinstance(child, ast.Call):
            name = _call_name(child)
            if name in _MATRIX_CALL_NAMES:
                return True
    return False


def _rhs_contains_division(node: ast.expr) -> bool:
    """Return True if the expression subtree contains a Div or FloorDiv node."""
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp) and isinstance(
            child.op, (ast.Div, ast.FloorDiv)
        ):
            return True
    return False


def _call_name(call: ast.Call) -> str:
    """
    Extract a dotted name string from a Call node's func attribute.
    Returns '' if the call target is not a simple name or attribute.
    """
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return f"{func.value.id}.{func.attr}"
        # e.g. np.linalg.inv — return just the final attribute for matching
        return func.attr
    return ""
