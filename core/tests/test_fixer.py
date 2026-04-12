"""
Production-grade tests for blackswan.fixer.guards.

Covers all four fix patterns with realistic financial model code:
  1. division_instability / division_by_zero  — epsilon guard on denominator
  2. non_psd_matrix                           — eigenvalue clamping (Higham nearest-PSD)
  3. ill_conditioned_matrix                   — conditional pinv fallback
  4. nan_inf                                  — np.nan_to_num guard

Each pattern is tested for:
  - Happy path: correct fix applied, result is syntactically valid Python
  - Indentation preservation: 0-space, 4-space, 8-space, tab indentation
  - Guard content: the guard expression matches the expected pattern
  - Explanation text: non-empty and references the variable
  - Error cases: out-of-range line, wrong line type, no matching node
  - Unsupported failure type: returns status="unsupported"
  - CLI round-trip: `python -m blackswan fix` produces valid JSON

Also skips cleanly if libcst is not installed (developer without optional dep).
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

CORE_DIR = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# libcst availability guard — skip all tests if not installed
# ---------------------------------------------------------------------------

try:
    import libcst  # noqa: F401
    LIBCST_AVAILABLE = True
except ImportError:
    LIBCST_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not LIBCST_AVAILABLE,
    reason="libcst not installed — run: pip install 'blackswan[fixer]'",
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def fix(source: str, line: int, failure_type: str):
    """Call apply_fix and return the FixResult."""
    from blackswan.fixer.guards import apply_fix
    return apply_fix(source, line, failure_type)


def is_valid_python(code: str) -> bool:
    """Return True if `code` is syntactically valid Python.

    Strips leading indentation so a single indented line (e.g. the replacement
    from a fix) can be validated without wrapping it in a function first.
    """
    try:
        ast.parse(textwrap.dedent(code))
        return True
    except SyntaxError:
        return False


def apply_result_to_source(source: str, result) -> str:
    """
    Apply a FixResult to the source string, producing the fixed version.
    Mirrors the logic in fixer.ts _applyResultToContent.
    """
    lines = source.splitlines(keepends=True)
    line_idx = result.line - 1

    if result.replacement is not None and result.replacement != result.original:
        original_nl = lines[line_idx].endswith("\n")
        lines[line_idx] = result.replacement + ("\n" if original_nl else "")

    if result.extra_lines:
        insert_pos = line_idx + 1
        for i, extra in enumerate(result.extra_lines):
            lines.insert(insert_pos + i, extra + "\n")

    return "".join(lines)


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke `python -m blackswan <args>` from core/."""
    return subprocess.run(
        [sys.executable, "-m", "blackswan", *args],
        capture_output=True,
        text=True,
        cwd=str(CORE_DIR),
    )


# ---------------------------------------------------------------------------
# 1. Division guard
# ---------------------------------------------------------------------------

class TestDivisionGuard:
    """Tests for division_instability and division_by_zero fixes."""

    SOURCE_SIMPLE = textwrap.dedent("""\
        def compute_sharpe(returns, vol):
            sharpe = returns / vol
            return sharpe
    """)

    SOURCE_INDENTED = textwrap.dedent("""\
        def compute_sharpe(returns, vol):
            if vol is not None:
                sharpe = returns / vol
                return sharpe
    """)

    SOURCE_DEEPLY_INDENTED = textwrap.dedent("""\
        def compute_sharpe(returns, vol):
            for i in range(10):
                for j in range(10):
                    sharpe = returns / vol
        """)

    SOURCE_TAB_INDENTED = "def f(x, y):\n\tresult = x / y\n\treturn result\n"

    SOURCE_COMPLEX_DENOM = textwrap.dedent("""\
        def compute_vol_adjusted(r, sigma):
            result = r / sigma
            return result
    """)

    SOURCE_FLOOR_DIV = textwrap.dedent("""\
        def bucket(val, bucket_size):
            idx = val // bucket_size
            return idx
    """)

    SOURCE_MULTI_ASSIGN = textwrap.dedent("""\
        def metrics(ret, vol, spread):
            sharpe = ret / vol
            cost_ratio = spread / vol
            return sharpe, cost_ratio
    """)

    def test_returns_ok_status(self):
        r = fix(self.SOURCE_SIMPLE, 2, "division_instability")
        assert r.status == "ok"

    def test_result_line_echoed_back(self):
        r = fix(self.SOURCE_SIMPLE, 2, "division_instability")
        assert r.line == 2

    def test_original_field_matches_source_line(self):
        r = fix(self.SOURCE_SIMPLE, 2, "division_instability")
        assert r.original == "    sharpe = returns / vol"

    def test_replacement_contains_epsilon_guard(self):
        r = fix(self.SOURCE_SIMPLE, 2, "division_instability")
        assert r.replacement is not None
        assert "1e-10" in r.replacement, "Epsilon guard must be present"

    def test_replacement_preserves_variable_assignment(self):
        r = fix(self.SOURCE_SIMPLE, 2, "division_instability")
        assert r.replacement is not None
        assert "sharpe" in r.replacement
        assert "returns" in r.replacement

    def test_replacement_is_syntactically_valid(self):
        r = fix(self.SOURCE_SIMPLE, 2, "division_instability")
        assert is_valid_python(r.replacement or "")

    def test_fixed_source_is_syntactically_valid(self):
        r = fix(self.SOURCE_SIMPLE, 2, "division_instability")
        fixed = apply_result_to_source(self.SOURCE_SIMPLE, r)
        assert is_valid_python(fixed)

    def test_4space_indentation_preserved(self):
        r = fix(self.SOURCE_INDENTED, 3, "division_instability")
        assert r.replacement is not None
        assert r.replacement.startswith("        ")  # 8 spaces (nested)

    def test_deeply_nested_indentation_preserved(self):
        r = fix(self.SOURCE_DEEPLY_INDENTED, 4, "division_instability")
        assert r.replacement is not None
        assert r.replacement.startswith("            ")  # 12 spaces

    def test_tab_indentation_preserved(self):
        r = fix(self.SOURCE_TAB_INDENTED, 2, "division_instability")
        assert r.replacement is not None
        assert r.replacement.startswith("\t")

    def test_floor_division_also_fixed(self):
        r = fix(self.SOURCE_FLOOR_DIV, 2, "division_by_zero")
        assert r.status == "ok"
        assert "1e-10" in (r.replacement or "")

    def test_explanation_is_non_empty(self):
        r = fix(self.SOURCE_SIMPLE, 2, "division_instability")
        assert r.explanation and len(r.explanation) > 20

    def test_explanation_mentions_denominator_variable(self):
        r = fix(self.SOURCE_SIMPLE, 2, "division_instability")
        assert "vol" in (r.explanation or "")

    def test_sign_preservation_ternary_present(self):
        """Guard must preserve sign: negative denominators stay negative."""
        r = fix(self.SOURCE_SIMPLE, 2, "division_instability")
        replacement = r.replacement or ""
        # The sign-preserving branch uses a conditional
        assert ">= 0" in replacement or "abs(" in replacement

    def test_first_division_on_multi_division_line_is_fixed(self):
        """Only the denominator at the detected line should be wrapped."""
        r = fix(self.SOURCE_MULTI_ASSIGN, 2, "division_instability")
        assert r.status == "ok"

    def test_error_when_line_out_of_range(self):
        r = fix(self.SOURCE_SIMPLE, 999, "division_instability")
        assert r.status == "error"
        assert r.message is not None

    def test_error_when_no_division_on_line(self):
        source = "def f():\n    x = 1 + 2\n    return x\n"
        r = fix(source, 2, "division_instability")
        assert r.status == "error"

    def test_error_line_zero_is_out_of_range(self):
        r = fix(self.SOURCE_SIMPLE, 0, "division_instability")
        assert r.status == "error"


# ---------------------------------------------------------------------------
# 2. PSD guard (non_psd_matrix)
# ---------------------------------------------------------------------------

class TestPSDGuard:
    """Tests for non_psd_matrix eigenvalue clamping fix."""

    SOURCE = textwrap.dedent("""\
        import numpy as np

        def build_cov(returns):
            cov_matrix = np.cov(returns.T)
            return cov_matrix
    """)

    SOURCE_DEEP = textwrap.dedent("""\
        import numpy as np

        def stress_test(corr, vols):
            if corr is not None:
                cov = corr * np.outer(vols, vols)
                return cov
    """)

    def test_returns_ok_status(self):
        r = fix(self.SOURCE, 4, "non_psd_matrix")
        assert r.status == "ok"

    def test_extra_lines_not_empty(self):
        r = fix(self.SOURCE, 4, "non_psd_matrix")
        assert r.extra_lines and len(r.extra_lines) >= 2

    def test_original_line_unchanged(self):
        """PSD fix inserts lines after — original line is NOT modified."""
        r = fix(self.SOURCE, 4, "non_psd_matrix")
        assert r.replacement == r.original

    def test_guard_uses_eigh(self):
        r = fix(self.SOURCE, 4, "non_psd_matrix")
        extra = "\n".join(r.extra_lines or [])
        assert "np.linalg.eigh" in extra

    def test_guard_clamps_eigenvalues_with_maximum(self):
        r = fix(self.SOURCE, 4, "non_psd_matrix")
        extra = "\n".join(r.extra_lines or [])
        assert "np.maximum" in extra
        assert "0.0" in extra

    def test_guard_reconstructs_matrix(self):
        r = fix(self.SOURCE, 4, "non_psd_matrix")
        extra = "\n".join(r.extra_lines or [])
        assert "@" in extra  # matrix multiply reconstruction

    def test_variable_referenced_in_guard(self):
        r = fix(self.SOURCE, 4, "non_psd_matrix")
        extra = "\n".join(r.extra_lines or [])
        assert "cov_matrix" in extra

    def test_indentation_preserved_in_extra_lines(self):
        r = fix(self.SOURCE, 4, "non_psd_matrix")
        for line in (r.extra_lines or []):
            assert line.startswith("    "), f"Expected 4-space indent, got: {line!r}"

    def test_deep_indentation_preserved(self):
        r = fix(self.SOURCE_DEEP, 5, "non_psd_matrix")
        assert r.status == "ok"
        for line in (r.extra_lines or []):
            assert line.startswith("        "), f"Expected 8-space indent, got: {line!r}"

    def test_fixed_source_is_syntactically_valid(self):
        r = fix(self.SOURCE, 4, "non_psd_matrix")
        fixed = apply_result_to_source(self.SOURCE, r)
        assert is_valid_python(fixed)

    def test_explanation_mentions_psd(self):
        r = fix(self.SOURCE, 4, "non_psd_matrix")
        explanation = (r.explanation or "").lower()
        assert "psd" in explanation or "semi-definite" in explanation or "eigenvalue" in explanation

    def test_error_when_line_out_of_range(self):
        r = fix(self.SOURCE, 999, "non_psd_matrix")
        assert r.status == "error"


# ---------------------------------------------------------------------------
# 3. Condition number guard (ill_conditioned_matrix)
# ---------------------------------------------------------------------------

class TestConditionNumberGuard:
    """Tests for ill_conditioned_matrix conditional pinv fix."""

    SOURCE = textwrap.dedent("""\
        import numpy as np

        def compute_weights(cov_matrix, mu):
            weights = np.linalg.inv(cov_matrix) @ mu
            return weights
    """)

    SOURCE_INDENTED = textwrap.dedent("""\
        import numpy as np

        def compute_weights(cov, mu):
            if cov is not None:
                w = np.linalg.inv(cov) @ mu
                return w
    """)

    def test_returns_ok_status(self):
        r = fix(self.SOURCE, 4, "ill_conditioned_matrix")
        assert r.status == "ok"

    def test_replacement_contains_cond_check(self):
        r = fix(self.SOURCE, 4, "ill_conditioned_matrix")
        assert "np.linalg.cond" in (r.replacement or "")

    def test_replacement_contains_pinv_fallback(self):
        r = fix(self.SOURCE, 4, "ill_conditioned_matrix")
        assert "np.linalg.pinv" in (r.replacement or "")

    def test_replacement_contains_threshold(self):
        r = fix(self.SOURCE, 4, "ill_conditioned_matrix")
        assert "1e12" in (r.replacement or "")

    def test_matrix_argument_preserved(self):
        r = fix(self.SOURCE, 4, "ill_conditioned_matrix")
        assert "cov_matrix" in (r.replacement or "")

    def test_replacement_is_syntactically_valid(self):
        r = fix(self.SOURCE, 4, "ill_conditioned_matrix")
        assert is_valid_python(r.replacement or "")

    def test_fixed_source_is_syntactically_valid(self):
        r = fix(self.SOURCE, 4, "ill_conditioned_matrix")
        fixed = apply_result_to_source(self.SOURCE, r)
        assert is_valid_python(fixed)

    def test_indentation_preserved(self):
        r = fix(self.SOURCE_INDENTED, 5, "ill_conditioned_matrix")
        assert r.status == "ok"
        # SOURCE_INDENTED: top-level def → 0 spaces; if block → 4 spaces; body → 8 spaces
        assert (r.replacement or "").startswith("        ")  # 8 spaces

    def test_original_line_not_equal_to_replacement(self):
        r = fix(self.SOURCE, 4, "ill_conditioned_matrix")
        assert r.replacement != r.original

    def test_explanation_mentions_condition_number(self):
        r = fix(self.SOURCE, 4, "ill_conditioned_matrix")
        explanation = (r.explanation or "").lower()
        assert "condition" in explanation or "singular" in explanation

    def test_error_when_no_inv_on_line(self):
        source = "def f():\n    x = np.dot(a, b)\n    return x\n"
        r = fix(source, 2, "ill_conditioned_matrix")
        assert r.status == "error"


# ---------------------------------------------------------------------------
# 4. NaN/inf guard
# ---------------------------------------------------------------------------

class TestNaNInfGuard:
    """Tests for nan_inf np.nan_to_num guard fix."""

    SOURCE = textwrap.dedent("""\
        import numpy as np

        def compute_portfolio_value(weights, returns):
            portfolio_value = np.sum(weights * returns)
            return portfolio_value
    """)

    SOURCE_INDENTED = textwrap.dedent("""\
        import numpy as np

        def compute(w, r):
            if w is not None:
                value = np.dot(w, r)
                return value
    """)

    def test_returns_ok_status(self):
        r = fix(self.SOURCE, 4, "nan_inf")
        assert r.status == "ok"

    def test_extra_lines_not_empty(self):
        r = fix(self.SOURCE, 4, "nan_inf")
        assert r.extra_lines and len(r.extra_lines) >= 1

    def test_original_line_unchanged(self):
        r = fix(self.SOURCE, 4, "nan_inf")
        assert r.replacement == r.original

    def test_guard_uses_nan_to_num(self):
        r = fix(self.SOURCE, 4, "nan_inf")
        extra = "\n".join(r.extra_lines or [])
        assert "np.nan_to_num" in extra

    def test_guard_handles_nan(self):
        r = fix(self.SOURCE, 4, "nan_inf")
        extra = "\n".join(r.extra_lines or [])
        assert "nan=" in extra

    def test_guard_handles_inf(self):
        r = fix(self.SOURCE, 4, "nan_inf")
        extra = "\n".join(r.extra_lines or [])
        assert "posinf=" in extra and "neginf=" in extra

    def test_variable_referenced_in_guard(self):
        r = fix(self.SOURCE, 4, "nan_inf")
        extra = "\n".join(r.extra_lines or [])
        assert "portfolio_value" in extra

    def test_indentation_preserved(self):
        r = fix(self.SOURCE, 4, "nan_inf")
        for line in (r.extra_lines or []):
            assert line.startswith("    "), f"Expected 4-space indent, got: {line!r}"

    def test_deep_indentation_preserved(self):
        r = fix(self.SOURCE_INDENTED, 5, "nan_inf")
        assert r.status == "ok"
        for line in (r.extra_lines or []):
            assert line.startswith("        ")

    def test_fixed_source_is_syntactically_valid(self):
        r = fix(self.SOURCE, 4, "nan_inf")
        fixed = apply_result_to_source(self.SOURCE, r)
        assert is_valid_python(fixed)

    def test_explanation_is_non_empty(self):
        r = fix(self.SOURCE, 4, "nan_inf")
        assert r.explanation and len(r.explanation) > 20


# ---------------------------------------------------------------------------
# 5. Unsupported and error cases
# ---------------------------------------------------------------------------

class TestUnsupportedAndErrors:
    SOURCE = "def f(x):\n    return x * 2\n"

    def test_unsupported_failure_type_returns_unsupported_status(self):
        r = fix(self.SOURCE, 1, "bounds_exceeded")
        assert r.status == "unsupported"

    def test_unsupported_has_helpful_message(self):
        r = fix(self.SOURCE, 1, "regime_shift")
        assert r.message and len(r.message) > 10

    def test_unknown_failure_type_is_unsupported(self):
        r = fix(self.SOURCE, 1, "completely_made_up_type")
        assert r.status == "unsupported"

    def test_empty_source_returns_error_for_division(self):
        r = fix("", 1, "division_instability")
        assert r.status == "error"

    def test_syntactically_invalid_source_returns_error(self):
        bad_source = "def f(\n    x = 1 +\n"
        r = fix(bad_source, 2, "division_instability")
        assert r.status == "error"

    def test_fix_result_to_json_roundtrip(self):
        """FixResult.to_json() must produce parseable JSON with all fields."""
        r = fix("def f():\n    x = 1 / 2\n", 2, "division_instability")
        assert r.status == "ok"
        data = json.loads(r.to_json())
        assert data["status"] == "ok"
        assert data["line"] == 2
        assert "original" in data
        assert "replacement" in data
        assert "explanation" in data

    def test_fix_result_no_null_fields_in_json(self):
        """None fields must be omitted from the JSON output."""
        r = fix("def f():\n    x = 1 / 2\n", 2, "division_instability")
        data = json.loads(r.to_json())
        for value in data.values():
            assert value is not None, "None values must be omitted from JSON"


# ---------------------------------------------------------------------------
# 6. CLI round-trip (subprocess)
# ---------------------------------------------------------------------------

class TestFixerCLI:
    """End-to-end tests that invoke `python -m blackswan fix` via subprocess."""

    def _write_fixture(self, tmp_path: Path, content: str, name: str = "model.py") -> Path:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_division_fix_exits_zero(self, tmp_path):
        src = "def f(x, vol):\n    return x / vol\n"
        p = self._write_fixture(tmp_path, src)
        result = _run_cli("fix", str(p), "--line", "2", "--type", "division_instability")
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_division_fix_stdout_is_valid_json(self, tmp_path):
        src = "def f(x, vol):\n    return x / vol\n"
        p = self._write_fixture(tmp_path, src)
        result = _run_cli("fix", str(p), "--line", "2", "--type", "division_instability")
        data = json.loads(result.stdout)
        assert data["status"] == "ok"

    def test_psd_fix_via_cli(self, tmp_path):
        src = "import numpy as np\ndef f(r):\n    cov = np.cov(r.T)\n    return cov\n"
        p = self._write_fixture(tmp_path, src)
        result = _run_cli("fix", str(p), "--line", "3", "--type", "non_psd_matrix")
        data = json.loads(result.stdout)
        assert data["status"] == "ok"
        assert "extra_lines" in data

    def test_nan_fix_via_cli(self, tmp_path):
        src = "import numpy as np\ndef f(w, r):\n    val = np.dot(w, r)\n    return val\n"
        p = self._write_fixture(tmp_path, src)
        result = _run_cli("fix", str(p), "--line", "3", "--type", "nan_inf")
        data = json.loads(result.stdout)
        assert data["status"] == "ok"

    def test_unsupported_type_exits_nonzero(self, tmp_path):
        src = "def f(x):\n    return x\n"
        p = self._write_fixture(tmp_path, src)
        result = _run_cli("fix", str(p), "--line", "1", "--type", "bounds_exceeded")
        assert result.returncode != 0
        data = json.loads(result.stdout)
        assert data["status"] == "unsupported"

    def test_missing_file_exits_nonzero(self, tmp_path):
        result = _run_cli("fix", str(tmp_path / "nonexistent.py"), "--line", "1", "--type", "division_instability")
        assert result.returncode != 0

    def test_output_is_not_empty(self, tmp_path):
        src = "def f(x, y):\n    return x / y\n"
        p = self._write_fixture(tmp_path, src)
        result = _run_cli("fix", str(p), "--line", "2", "--type", "division_instability")
        assert result.stdout.strip() != ""

    def test_no_stderr_on_successful_fix(self, tmp_path):
        src = "def f(x, y):\n    return x / y\n"
        p = self._write_fixture(tmp_path, src)
        result = _run_cli("fix", str(p), "--line", "2", "--type", "division_instability")
        assert result.returncode == 0
        # stderr should be empty on success
        assert result.stderr.strip() == ""


# ---------------------------------------------------------------------------
# 7. Edge cases: realistic financial model snippets
# ---------------------------------------------------------------------------

class TestRealisticFinancialCode:
    """Fixer correctness on snippets that resemble actual quant finance code."""

    def test_sharpe_ratio_division(self):
        source = textwrap.dedent("""\
            import numpy as np

            def sharpe_ratio(returns, risk_free_rate, volatility):
                excess_returns = returns - risk_free_rate
                sharpe = np.mean(excess_returns) / volatility
                return sharpe
        """)
        r = fix(source, 5, "division_instability")
        assert r.status == "ok"
        assert is_valid_python(apply_result_to_source(source, r))

    def test_information_ratio_division(self):
        source = textwrap.dedent("""\
            def information_ratio(active_returns, tracking_error):
                return active_returns / tracking_error
        """)
        r = fix(source, 2, "division_instability")
        assert r.status == "ok"
        assert "tracking_error" in (r.explanation or "")

    def test_covariance_matrix_psd_fix(self):
        source = textwrap.dedent("""\
            import numpy as np

            def estimate_covariance(returns_matrix, shrinkage=0.1):
                sample_cov = np.cov(returns_matrix.T)
                shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * np.eye(sample_cov.shape[0])
                return shrunk_cov
        """)
        r = fix(source, 4, "non_psd_matrix")
        assert r.status == "ok"
        fixed = apply_result_to_source(source, r)
        assert is_valid_python(fixed)
        assert "np.linalg.eigh" in fixed

    def test_portfolio_optimisation_inv(self):
        source = textwrap.dedent("""\
            import numpy as np

            def minimum_variance_weights(cov_matrix):
                ones = np.ones(cov_matrix.shape[0])
                inv_cov = np.linalg.inv(cov_matrix)
                w = inv_cov @ ones
                return w / w.sum()
        """)
        r = fix(source, 5, "ill_conditioned_matrix")
        assert r.status == "ok"
        fixed = apply_result_to_source(source, r)
        assert is_valid_python(fixed)
        assert "np.linalg.cond" in fixed

    def test_portfolio_value_nan_guard(self):
        source = textwrap.dedent("""\
            import numpy as np

            def portfolio_pnl(weights, price_returns):
                pnl = np.dot(weights, price_returns)
                return pnl
        """)
        r = fix(source, 4, "nan_inf")
        assert r.status == "ok"
        fixed = apply_result_to_source(source, r)
        assert is_valid_python(fixed)
        assert "nan_to_num" in fixed

    def test_fixed_code_runs_without_import_error(self):
        """
        The fixed source must not introduce any new import requirements beyond numpy.
        This catches accidental introduction of `import math` etc.
        """
        source = textwrap.dedent("""\
            import numpy as np

            def f(x, vol):
                return x / vol
        """)
        r = fix(source, 4, "division_instability")
        assert r.status == "ok"
        fixed = apply_result_to_source(source, r)
        # Parse and check no new bare imports were added
        tree = ast.parse(fixed)
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        import_names = {
            alias.name
            for node in imports
            for alias in (node.names if isinstance(node, ast.Import) else [])
        }
        # math or other unexpected imports must not appear
        assert "math" not in import_names
