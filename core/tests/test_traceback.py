"""
Tests for TracebackResolver.

Covers:
  - Exception-based attribution via stored exc_frames
  - Detector-based attribution via AST heuristic (matrix construction lines)
  - Division-by-zero attribution via AST
  - Graceful fallback when no line can be resolved
  - Integration: broken_covariance.py + MatrixPSDDetector + TracebackResolver
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blackswan.attribution.traceback import TracebackResolver
from blackswan.detectors.base import Finding
from blackswan.detectors.matrix import MatrixPSDDetector, ConditionNumberDetector

FIXTURE_DIR = Path(__file__).parent / "fixtures"
BROKEN_COV_PATH = FIXTURE_DIR / "broken_covariance.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finding(failure_type: str = "nan_inf", line: int | None = None,
             exc_frames: list | None = None) -> Finding:
    return Finding(
        failure_type=failure_type,
        severity="critical",
        message="test",
        iteration=0,
        line=line,
        exc_frames=exc_frames or [],
    )


# ---------------------------------------------------------------------------
# Basic resolver behaviour
# ---------------------------------------------------------------------------

class TestResolverBasics:
    def test_resolve_returns_a_finding(self):
        resolver = TracebackResolver(BROKEN_COV_PATH)
        result = resolver.resolve(_finding("non_psd_matrix"))
        assert isinstance(result, Finding)

    def test_resolve_returns_new_object_not_original(self):
        resolver = TracebackResolver(BROKEN_COV_PATH)
        original = _finding("non_psd_matrix")
        result = resolver.resolve(original)
        assert result is not original

    def test_finding_with_line_already_set_is_returned_unchanged(self):
        """If attribution is already done, resolver must not overwrite it."""
        resolver = TracebackResolver(BROKEN_COV_PATH)
        result = resolver.resolve(_finding("non_psd_matrix", line=5))
        assert result.line == 5

    def test_unknown_failure_type_returns_line_none(self):
        resolver = TracebackResolver(BROKEN_COV_PATH)
        result = resolver.resolve(_finding("bounds_exceeded"))
        assert result.line is None


# ---------------------------------------------------------------------------
# Exception-based attribution (exc_frames)
# ---------------------------------------------------------------------------

class TestExceptionAttribution:
    def test_resolves_line_from_exc_frames_matching_source_file(self):
        """Frame that matches the source file → its lineno is returned."""
        resolver = TracebackResolver(BROKEN_COV_PATH)
        finding = _finding(
            "nan_inf",
            exc_frames=[(str(BROKEN_COV_PATH), 18)],
        )
        result = resolver.resolve(finding)
        assert result.line == 18

    def test_ignores_frames_from_other_files(self):
        """Frames from numpy internals or other modules must be filtered out."""
        resolver = TracebackResolver(BROKEN_COV_PATH)
        finding = _finding(
            "nan_inf",
            exc_frames=[
                ("/usr/lib/python3.11/numpy/core.py", 99),
                ("/some/other/module.py", 42),
            ],
        )
        result = resolver.resolve(finding)
        assert result.line is None

    def test_picks_last_matching_frame_when_multiple_present(self):
        """The last (most recent) frame in the source file is the proximate location."""
        resolver = TracebackResolver(BROKEN_COV_PATH)
        finding = _finding(
            "nan_inf",
            exc_frames=[
                (str(BROKEN_COV_PATH), 10),
                ("/numpy/internals.py", 55),
                (str(BROKEN_COV_PATH), 25),  # ← last match in source file
            ],
        )
        result = resolver.resolve(finding)
        assert result.line == 25

    def test_exc_frames_take_precedence_over_ast_heuristic(self):
        """For a matrix finding that also has exc_frames, frames win."""
        resolver = TracebackResolver(BROKEN_COV_PATH)
        finding = _finding(
            "non_psd_matrix",
            exc_frames=[(str(BROKEN_COV_PATH), 7)],
        )
        result = resolver.resolve(finding)
        assert result.line == 7


# ---------------------------------------------------------------------------
# Detector-based attribution — matrix failures (AST heuristic)
# ---------------------------------------------------------------------------

class TestMatrixAttribution:
    def setup_method(self):
        self.resolver = TracebackResolver(BROKEN_COV_PATH)
        self.source_lines = BROKEN_COV_PATH.read_text().splitlines()

    def _cov_matrix_lineno(self) -> int:
        """Dynamically find the line that assigns cov_matrix in the fixture."""
        for i, line in enumerate(self.source_lines, start=1):
            stripped = line.strip()
            if stripped.startswith("cov_matrix") and "=" in stripped and not stripped.startswith("#"):
                return i
        pytest.fail("cov_matrix assignment not found in broken_covariance.py")

    def test_matrix_psd_resolved_to_cov_matrix_line(self):
        """Core test: MatrixPSDDetector finding → cov_matrix construction line."""
        result = self.resolver.resolve(_finding("non_psd_matrix"))
        assert result.line == self._cov_matrix_lineno()

    def test_resolved_line_contains_matrix_multiplication_operator(self):
        """The resolved line must use @ — confirming it's the matrix product."""
        result = self.resolver.resolve(_finding("non_psd_matrix"))
        assert result.line is not None
        assert "@" in self.source_lines[result.line - 1]

    def test_condition_number_resolved_to_same_line_as_psd(self):
        """ConditionNumberDetector uses the same AST heuristic as MatrixPSDDetector."""
        psd_result = self.resolver.resolve(_finding("non_psd_matrix"))
        cond_result = self.resolver.resolve(_finding("ill_conditioned_matrix"))
        assert psd_result.line == cond_result.line

    def test_resolved_line_is_inside_function_not_module_level(self):
        """Attribution must point inside the function body, not top-level imports."""
        result = self.resolver.resolve(_finding("non_psd_matrix"))
        assert result.line is not None
        def_line = next(
            i + 1 for i, l in enumerate(self.source_lines)
            if l.strip().startswith("def ")
        )
        assert result.line > def_line


# ---------------------------------------------------------------------------
# Detector-based attribution — division failures (AST heuristic)
# ---------------------------------------------------------------------------

class TestDivisionAttribution:
    def test_division_finding_resolved_to_division_line(self, tmp_path):
        """`division_by_zero` finding → the assignment line containing /."""
        source = tmp_path / "div_model.py"
        source.write_text(
            "import numpy as np\n"        # 1
            "\n"                           # 2
            "def compute(weights, vol):\n" # 3
            "    n = len(weights)\n"       # 4
            "    normalised = weights / vol\n"  # 5 ← expected
            "    return normalised\n"      # 6
        )
        resolver = TracebackResolver(source)
        result = resolver.resolve(_finding("division_by_zero"))
        assert result.line == 5

    def test_division_resolved_line_contains_slash(self, tmp_path):
        source = tmp_path / "div2.py"
        source.write_text(
            "def f(a, b):\n"   # 1
            "    x = a + b\n"  # 2
            "    y = x / b\n"  # 3 ← expected
            "    return y\n"   # 4
        )
        resolver = TracebackResolver(source)
        result = resolver.resolve(_finding("division_by_zero"))
        assert result.line is not None
        lines = source.read_text().splitlines()
        assert "/" in lines[result.line - 1]

    def test_no_division_in_source_returns_none(self, tmp_path):
        source = tmp_path / "no_div.py"
        source.write_text(
            "def f(a, b):\n"
            "    return a + b\n"
        )
        resolver = TracebackResolver(source)
        result = resolver.resolve(_finding("division_by_zero"))
        assert result.line is None


# ---------------------------------------------------------------------------
# Integration: broken_covariance + MatrixPSDDetector + TracebackResolver
# ---------------------------------------------------------------------------

class TestEndToEndIntegration:
    def test_full_pipeline_matrix_psd_fires_and_resolves(self):
        """
        Run broken_covariance.py, confirm MatrixPSDDetector fires, then
        confirm TracebackResolver points to the cov_matrix construction line.
        """
        from tests.fixtures.broken_covariance import calculate_portfolio_var

        weights = np.array([0.4, 0.3, 0.3])
        vol = np.array([0.2, 0.3, 0.25])

        # correlation=0.35 → pairwise corr = 0.8+0.35=1.15 > 1 → non-PSD
        output = calculate_portfolio_var(weights, vol, correlation=0.35)

        detector = MatrixPSDDetector()
        finding = detector.check(inputs={}, output=output, iteration=0)
        assert finding is not None, "MatrixPSDDetector must fire on this fixture"
        assert finding.failure_type == "non_psd_matrix"

        resolver = TracebackResolver(BROKEN_COV_PATH)
        resolved = resolver.resolve(finding)

        assert resolved.line is not None
        source_lines = BROKEN_COV_PATH.read_text().splitlines()
        resolved_text = source_lines[resolved.line - 1]
        assert "cov_matrix" in resolved_text

    def test_clean_inputs_produce_no_finding(self):
        """correlation=0.0 → PSD matrix → detector silent → nothing to resolve."""
        from tests.fixtures.broken_covariance import calculate_portfolio_var

        weights = np.array([0.4, 0.3, 0.3])
        vol = np.array([0.2, 0.3, 0.25])
        output = calculate_portfolio_var(weights, vol, correlation=0.0)

        finding = MatrixPSDDetector().check(inputs={}, output=output, iteration=0)
        assert finding is None
