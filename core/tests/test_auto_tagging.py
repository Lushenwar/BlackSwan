"""
Tests for AutoTagger and LineTagMap (Phase 2, Step 2C).

Verifies that:
  - The LineTagMap dataclass API works correctly.
  - Each fixture file produces the expected detector tags on the expected lines.
  - AutoTagger.detector_suite() returns a minimal but complete detector set.
  - The auto-tagged detector suite integrates correctly with StressRunner.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blackswan.parser.auto_tagger import AutoTagger, LineTagMap

FIXTURE_DIR = Path(__file__).parent / "fixtures"
BROKEN_COV  = FIXTURE_DIR / "broken_covariance.py"
CLEAN_MODEL = FIXTURE_DIR / "clean_model.py"
SIMPLE_PORT = FIXTURE_DIR / "simple_portfolio.py"
OVERFLOW    = FIXTURE_DIR / "overflow_leverage.py"
DIV_EDGE    = FIXTURE_DIR / "division_edge.py"
INT_SHORT   = FIXTURE_DIR / "intentional_short.py"

_DIV  = "DivisionStabilityDetector"
_PSD  = "MatrixPSDDetector"
_COND = "ConditionNumberDetector"
_NAN  = "NaNInfDetector"
_BOUNDS = "BoundsDetector"


# ---------------------------------------------------------------------------
# LineTagMap dataclass API
# ---------------------------------------------------------------------------

class TestLineTagMap:
    def _make(self, mapping: dict) -> LineTagMap:
        return LineTagMap(_lines=mapping)

    def test_tags_for_line_returns_copy_of_list(self):
        tag_map = self._make({5: [_DIV]})
        result = tag_map.tags_for_line(5)
        result.append("extra")
        assert tag_map.tags_for_line(5) == [_DIV]

    def test_tags_for_unknown_line_returns_empty(self):
        tag_map = self._make({5: [_DIV]})
        assert tag_map.tags_for_line(99) == []

    def test_all_tags_aggregates_across_lines(self):
        tag_map = self._make({5: [_DIV], 10: [_PSD], 15: [_COND, _PSD]})
        assert tag_map.all_tags() == {_DIV, _PSD, _COND}

    def test_all_tags_empty_when_no_lines_tagged(self):
        assert LineTagMap(_lines={}).all_tags() == set()

    def test_tagged_lines_returns_dict_copy(self):
        tag_map = self._make({5: [_DIV]})
        d = tag_map.tagged_lines()
        d[99] = [_PSD]
        assert 99 not in tag_map.tagged_lines()

    def test_is_empty_true_when_no_tags(self):
        assert LineTagMap(_lines={}).is_empty()

    def test_is_empty_false_when_tags_present(self):
        assert not self._make({5: [_DIV]}).is_empty()


# ---------------------------------------------------------------------------
# Line-level tag correctness per fixture
# ---------------------------------------------------------------------------

class TestLineTagsPerFixture:
    def test_division_edge_has_division_tagged_line(self):
        tag_map = AutoTagger(DIV_EDGE).build()
        assert _DIV in tag_map.all_tags()

    def test_division_edge_tagged_line_contains_slash(self):
        tag_map = AutoTagger(DIV_EDGE).build()
        source = DIV_EDGE.read_text(encoding="utf-8").splitlines()
        for lineno, tags in tag_map.tagged_lines().items():
            if _DIV in tags:
                assert "/" in source[lineno - 1], (
                    f"Line {lineno} tagged DivisionStabilityDetector but contains no '/': "
                    f"{source[lineno - 1]!r}"
                )

    def test_broken_covariance_has_matrix_psd_tagged_line(self):
        tag_map = AutoTagger(BROKEN_COV).build()
        assert _PSD in tag_map.all_tags()

    def test_broken_covariance_psd_line_contains_matmult(self):
        tag_map = AutoTagger(BROKEN_COV).build()
        source = BROKEN_COV.read_text(encoding="utf-8").splitlines()
        for lineno, tags in tag_map.tagged_lines().items():
            if _PSD in tags:
                assert "@" in source[lineno - 1], (
                    f"Line {lineno} tagged MatrixPSDDetector but no '@': "
                    f"{source[lineno - 1]!r}"
                )

    def test_simple_portfolio_has_condition_number_tagged_line(self):
        tag_map = AutoTagger(SIMPLE_PORT).build()
        assert _COND in tag_map.all_tags()

    def test_simple_portfolio_condition_number_line_contains_inv(self):
        tag_map = AutoTagger(SIMPLE_PORT).build()
        source = SIMPLE_PORT.read_text(encoding="utf-8").splitlines()
        for lineno, tags in tag_map.tagged_lines().items():
            if _COND in tags:
                assert "inv" in source[lineno - 1], (
                    f"Line {lineno} tagged ConditionNumberDetector but no 'inv': "
                    f"{source[lineno - 1]!r}"
                )

    def test_simple_portfolio_has_division_tagged_line(self):
        tag_map = AutoTagger(SIMPLE_PORT).build()
        assert _DIV in tag_map.all_tags()

    def test_simple_portfolio_has_matrix_psd_tagged_line(self):
        tag_map = AutoTagger(SIMPLE_PORT).build()
        assert _PSD in tag_map.all_tags()

    def test_clean_model_has_no_fragility_tags(self):
        tag_map = AutoTagger(CLEAN_MODEL).build()
        fragility = {_DIV, _PSD, _COND}
        assert not (tag_map.all_tags() & fragility)

    def test_clean_model_tag_map_is_empty(self):
        assert AutoTagger(CLEAN_MODEL).build().is_empty()

    def test_overflow_has_no_fragility_tags(self):
        tag_map = AutoTagger(OVERFLOW).build()
        fragility = {_DIV, _PSD, _COND}
        assert not (tag_map.all_tags() & fragility)

    def test_intentional_short_has_no_fragility_tags(self):
        tag_map = AutoTagger(INT_SHORT).build()
        fragility = {_DIV, _PSD, _COND}
        assert not (tag_map.all_tags() & fragility)

    def test_floor_division_is_tagged(self, tmp_path):
        src = tmp_path / "fd.py"
        src.write_text("def f(a=10, b=3):\n    x = a // b\n    return x\n")
        tag_map = AutoTagger(src).build()
        assert _DIV in tag_map.all_tags()

    def test_np_cov_call_tagged_as_matrix_psd(self, tmp_path):
        src = tmp_path / "cov.py"
        src.write_text(
            "import numpy as np\ndef f(d=None):\n"
            "    if d is None:\n        d = np.ones((5,3))\n"
            "    c = np.cov(d.T)\n    return c\n"
        )
        tag_map = AutoTagger(src).build()
        assert _PSD in tag_map.all_tags()

    def test_np_linalg_inv_tagged_as_condition_number(self, tmp_path):
        src = tmp_path / "inv.py"
        src.write_text(
            "import numpy as np\ndef f(m=None):\n"
            "    if m is None:\n        m = np.eye(3)\n"
            "    inv_m = np.linalg.inv(m)\n    return inv_m\n"
        )
        tag_map = AutoTagger(src).build()
        assert _COND in tag_map.all_tags()

    def test_no_duplicate_tags_on_same_line(self, tmp_path):
        # A line with two matmult ops should not get the tag twice
        src = tmp_path / "dup.py"
        src.write_text(
            "import numpy as np\ndef f(a=None, b=None, c=None):\n"
            "    if a is None: a = np.eye(3)\n"
            "    if b is None: b = np.eye(3)\n"
            "    if c is None: c = np.eye(3)\n"
            "    x = a @ b @ c\n    return x\n"
        )
        tag_map = AutoTagger(src).build()
        for lineno, tags in tag_map.tagged_lines().items():
            assert tags.count(_PSD) <= 1


# ---------------------------------------------------------------------------
# Detector suite selection
# ---------------------------------------------------------------------------

class TestDetectorSuite:
    def _names(self, path: Path) -> list[str]:
        return [type(d).__name__ for d in AutoTagger(path).detector_suite()]

    def test_suite_always_contains_nan_inf_detector(self):
        for path in [BROKEN_COV, CLEAN_MODEL, SIMPLE_PORT, OVERFLOW, DIV_EDGE]:
            assert _NAN in self._names(path), f"NaNInfDetector missing for {path.name}"

    def test_suite_always_contains_bounds_detector(self):
        for path in [BROKEN_COV, CLEAN_MODEL, SIMPLE_PORT, OVERFLOW, DIV_EDGE]:
            assert _BOUNDS in self._names(path), f"BoundsDetector missing for {path.name}"

    def test_clean_model_suite_has_no_fragility_detectors(self):
        names = set(self._names(CLEAN_MODEL))
        assert _DIV not in names
        assert _PSD not in names
        assert _COND not in names

    def test_overflow_suite_has_no_fragility_detectors(self):
        names = set(self._names(OVERFLOW))
        assert _DIV not in names
        assert _PSD not in names
        assert _COND not in names

    def test_broken_covariance_suite_includes_matrix_psd(self):
        assert _PSD in self._names(BROKEN_COV)

    def test_division_edge_suite_includes_division_stability(self):
        assert _DIV in self._names(DIV_EDGE)

    def test_simple_portfolio_suite_includes_all_three_fragility_detectors(self):
        names = set(self._names(SIMPLE_PORT))
        assert _DIV in names
        assert _PSD in names
        assert _COND in names

    def test_no_duplicate_detector_types_in_suite(self):
        for path in [BROKEN_COV, CLEAN_MODEL, SIMPLE_PORT]:
            names = self._names(path)
            assert len(names) == len(set(names)), f"Duplicate detectors in suite for {path.name}"

    def test_suite_returns_detector_instances_not_classes(self):
        from blackswan.detectors.base import FailureDetector
        for d in AutoTagger(BROKEN_COV).detector_suite():
            assert isinstance(d, FailureDetector)


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------

class TestRunnerIntegration:
    def test_broken_covariance_psd_detector_fires_via_auto_suite(self):
        """MatrixPSDDetector from auto suite fires on non-PSD matrix output."""
        from tests.fixtures.broken_covariance import calculate_portfolio_var

        weights = np.array([0.4, 0.3, 0.3])
        vol = np.array([0.2, 0.3, 0.25])
        output = calculate_portfolio_var(weights, vol, correlation=0.35)

        suite = AutoTagger(BROKEN_COV).detector_suite()
        psd = next(d for d in suite if type(d).__name__ == _PSD)
        finding = psd.check(inputs={}, output=output, iteration=0)
        assert finding is not None
        assert finding.failure_type == "non_psd_matrix"

    def test_clean_model_auto_suite_produces_no_findings(self):
        """Auto suite for clean_model has no fragility detectors; output is safe."""
        from tests.fixtures.clean_model import compute_portfolio_value
        from blackswan.engine.runner import StressRunner

        class _PassThroughScenario:
            iterations = 10
            def apply(self, inputs, rng):
                return inputs

        suite = AutoTagger(CLEAN_MODEL).detector_suite()
        runner = StressRunner(
            fn=compute_portfolio_value,
            base_inputs={"weights": None, "prices": None},
            scenario=_PassThroughScenario(),
            detectors=suite,
            seed=42,
        )
        result = runner.run()
        assert result.findings == []

    def test_division_edge_division_detector_fires_via_auto_suite(self):
        """DivisionStabilityDetector from auto suite detects near-zero vol."""
        suite = AutoTagger(DIV_EDGE).detector_suite()
        div_det = next(d for d in suite if type(d).__name__ == _DIV)
        # denominator near zero triggers DivisionStabilityDetector
        finding = div_det.check(inputs={"denominator": 1e-12}, output=None, iteration=0)
        assert finding is not None
        assert finding.failure_type == "division_by_zero"

    def test_auto_suite_used_in_stress_runner_is_deterministic(self):
        """Two runs of the auto suite with the same seed produce identical findings."""
        from tests.fixtures.clean_model import compute_portfolio_value
        from blackswan.engine.runner import StressRunner

        class _Scenario:
            iterations = 20
            def apply(self, inputs, rng):
                return {"weights": rng.uniform(0.0, 1.0, 3),
                        "prices": rng.uniform(50.0, 200.0, 3)}

        kwargs = dict(
            fn=compute_portfolio_value,
            base_inputs={},
            detectors=AutoTagger(CLEAN_MODEL).detector_suite(),
            seed=7,
        )
        r1 = StressRunner(scenario=_Scenario(), **kwargs).run()
        r2 = StressRunner(scenario=_Scenario(), **kwargs).run()
        assert r1.findings == r2.findings


# ---------------------------------------------------------------------------
# CLI integration: auto-tagging wired into cli._cmd_test
# ---------------------------------------------------------------------------

class TestCLIAutoTaggingIntegration:
    def test_cli_uses_auto_tagging_clean_model_exits_zero(self):
        """Regression: CLI must still exit 0 for clean_model with auto-tagging."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "blackswan", "test",
             str(CLEAN_MODEL), "--scenario", "liquidity_crash", "--iterations", "5"],
            capture_output=True, text=True,
            cwd=str(FIXTURE_DIR.parent.parent),
        )
        assert result.returncode == 0

    def test_cli_auto_tagging_detector_names_absent_for_clean_model(self):
        """Auto suite for clean_model omits fragility detectors — output unchanged."""
        import subprocess, sys, json
        result = subprocess.run(
            [sys.executable, "-m", "blackswan", "test",
             str(CLEAN_MODEL), "--scenario", "liquidity_crash", "--iterations", "5"],
            capture_output=True, text=True,
            cwd=str(FIXTURE_DIR.parent.parent),
        )
        data = json.loads(result.stdout)
        assert data["status"] == "no_failures"
