"""
Tests for the BlackSwan CLI (Phase 1, Step 1E).

Tests the CLI mechanics end-to-end via subprocess so we exercise the real
argument parser, exit codes, JSON output, and error handling paths. Detector
and runner logic are covered by their own test suites.

Fixtures used:
  clean_model.py    — returns a safe scalar; no detector ever fires
  always_fails.py   — always raises; every iteration produces a finding
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CORE_DIR = Path(__file__).parent.parent          # core/
FIXTURE_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(*args: str) -> subprocess.CompletedProcess:
    """Invoke `python -m blackswan <args>` from the core/ directory."""
    return subprocess.run(
        [sys.executable, "-m", "blackswan", *args],
        capture_output=True,
        text=True,
        cwd=str(CORE_DIR),
    )


# ---------------------------------------------------------------------------
# --list-scenarios
# ---------------------------------------------------------------------------

class TestListScenarios:
    def test_exits_zero(self):
        result = _run("--list-scenarios")
        assert result.returncode == 0

    def test_outputs_all_five_preset_names(self):
        result = _run("--list-scenarios")
        for name in [
            "liquidity_crash",
            "vol_spike",
            "correlation_breakdown",
            "rate_shock",
            "missing_data",
        ]:
            assert name in result.stdout

    def test_output_has_at_least_five_lines(self):
        result = _run("--list-scenarios")
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        assert len(lines) >= 5

    def test_nothing_printed_to_stderr(self):
        result = _run("--list-scenarios")
        assert result.stderr.strip() == ""


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

class TestExitCodes:
    def test_exit_0_when_no_failures(self):
        result = _run(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "10",
        )
        assert result.returncode == 0

    def test_exit_1_when_failures_detected(self):
        result = _run(
            "test", str(FIXTURE_DIR / "always_fails.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        assert result.returncode == 1

    def test_exit_2_on_nonexistent_file(self):
        result = _run(
            "test", "/nonexistent/path/model.py",
            "--scenario", "liquidity_crash",
        )
        assert result.returncode == 2

    def test_exit_2_on_unknown_scenario(self):
        result = _run(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "no_such_scenario",
        )
        assert result.returncode == 2

    def test_exit_2_on_syntax_error_in_target_file(self, tmp_path):
        bad = tmp_path / "bad_syntax.py"
        bad.write_text("def f(:\n    pass\n")
        result = _run("test", str(bad), "--scenario", "liquidity_crash")
        assert result.returncode == 2

    def test_exit_2_on_unknown_function_name(self):
        result = _run(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--function", "nonexistent_function",
            "--iterations", "5",
        )
        assert result.returncode == 2


# ---------------------------------------------------------------------------
# JSON output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:
    def _parse(self, *args: str) -> dict:
        result = _run(*args)
        return json.loads(result.stdout)

    def test_stdout_is_valid_json(self):
        result = _run(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        json.loads(result.stdout)  # must not raise

    def test_has_all_top_level_keys(self):
        data = self._parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        for key in ["version", "status", "runtime_ms", "iterations_completed",
                    "summary", "shatter_points", "scenario_card"]:
            assert key in data, f"missing key: {key}"

    def test_version_is_1_0(self):
        data = self._parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        assert data["version"] == "1.0"

    def test_status_no_failures_on_clean_model(self):
        data = self._parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "10",
        )
        assert data["status"] == "no_failures"

    def test_status_failures_detected_on_always_fails(self):
        data = self._parse(
            "test", str(FIXTURE_DIR / "always_fails.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        assert data["status"] == "failures_detected"

    def test_iterations_completed_matches_override(self):
        data = self._parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "7",
        )
        assert data["iterations_completed"] == 7

    def test_shatter_points_empty_when_clean(self):
        data = self._parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        assert data["shatter_points"] == []

    def test_shatter_points_populated_when_failing(self):
        data = self._parse(
            "test", str(FIXTURE_DIR / "always_fails.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        assert len(data["shatter_points"]) > 0

    def test_shatter_point_has_required_fields(self):
        data = self._parse(
            "test", str(FIXTURE_DIR / "always_fails.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        sp = data["shatter_points"][0]
        for field in ["id", "line", "column", "severity", "failure_type",
                      "message", "frequency", "causal_chain", "fix_hint"]:
            assert field in sp, f"shatter_point missing field: {field}"

    def test_shatter_point_id_format(self):
        data = self._parse(
            "test", str(FIXTURE_DIR / "always_fails.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        assert data["shatter_points"][0]["id"].startswith("sp_")

    def test_frequency_string_format(self):
        data = self._parse(
            "test", str(FIXTURE_DIR / "always_fails.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        freq = data["shatter_points"][0]["frequency"]
        # Must match: "N / M iterations (P%)"
        assert " / " in freq
        assert "iterations" in freq
        assert "%" in freq

    def test_summary_fields_present(self):
        data = self._parse(
            "test", str(FIXTURE_DIR / "always_fails.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        s = data["summary"]
        assert "total_failures" in s
        assert "failure_rate" in s
        assert "unique_failure_types" in s

    def test_summary_total_failures_is_positive_when_failing(self):
        data = self._parse(
            "test", str(FIXTURE_DIR / "always_fails.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        assert data["summary"]["total_failures"] > 0

    def test_scenario_card_has_name_and_seed(self):
        data = self._parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        sc = data["scenario_card"]
        assert sc["name"] == "liquidity_crash"
        assert "seed" in sc
        assert sc["reproducible"] is True

    def test_runtime_ms_is_non_negative_integer(self):
        data = self._parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        assert isinstance(data["runtime_ms"], int)
        assert data["runtime_ms"] >= 0


# ---------------------------------------------------------------------------
# Error output
# ---------------------------------------------------------------------------

class TestErrorOutput:
    def test_error_goes_to_stderr_stdout_is_empty(self):
        result = _run("test", "/nonexistent/file.py", "--scenario", "liquidity_crash")
        assert result.returncode == 2
        assert result.stdout.strip() == ""
        assert result.stderr.strip() != ""

    def test_error_output_is_valid_json(self):
        result = _run("test", "/nonexistent/file.py", "--scenario", "liquidity_crash")
        json.loads(result.stderr)  # must not raise

    def test_error_json_has_status_error(self):
        result = _run("test", "/nonexistent/file.py", "--scenario", "liquidity_crash")
        data = json.loads(result.stderr)
        assert data["status"] == "error"

    def test_error_json_has_message(self):
        result = _run("test", "/nonexistent/file.py", "--scenario", "liquidity_crash")
        data = json.loads(result.stderr)
        assert "message" in data
        assert len(data["message"]) > 0

    def test_no_python_traceback_in_stderr_on_bad_scenario(self):
        result = _run(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "no_such_scenario",
        )
        # A raw Python traceback would contain "Traceback (most recent call last)"
        assert "Traceback" not in result.stderr


# ---------------------------------------------------------------------------
# Argument overrides
# ---------------------------------------------------------------------------

class TestArgumentOverrides:
    def test_iterations_override_is_respected(self):
        result = _run(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "13",
        )
        data = json.loads(result.stdout)
        assert data["iterations_completed"] == 13

    def test_seed_override_appears_in_scenario_card(self):
        result = _run(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
            "--seed", "99999",
        )
        data = json.loads(result.stdout)
        assert data["scenario_card"]["seed"] == 99999

    def test_same_seed_produces_identical_results(self):
        args = [
            "test", str(FIXTURE_DIR / "always_fails.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "20",
            "--seed", "7",
        ]
        r1 = json.loads(_run(*args).stdout)
        r2 = json.loads(_run(*args).stdout)
        assert r1["summary"]["total_failures"] == r2["summary"]["total_failures"]
        assert r1["summary"]["unique_failure_types"] == r2["summary"]["unique_failure_types"]

    def test_different_seeds_may_differ(self):
        """Sanity check: same fixture, different seeds → same deterministic function,
        same result (always_fails raises every time regardless of seed)."""
        r1 = json.loads(_run(
            "test", str(FIXTURE_DIR / "always_fails.py"),
            "--scenario", "liquidity_crash", "--iterations", "5", "--seed", "1",
        ).stdout)
        r2 = json.loads(_run(
            "test", str(FIXTURE_DIR / "always_fails.py"),
            "--scenario", "liquidity_crash", "--iterations", "5", "--seed", "2",
        ).stdout)
        # Both should detect failures — result structure must be valid
        assert r1["status"] == "failures_detected"
        assert r2["status"] == "failures_detected"


# ---------------------------------------------------------------------------
# --function flag
# ---------------------------------------------------------------------------

class TestFunctionFlag:
    def test_explicit_function_name_is_accepted(self):
        result = _run(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--function", "compute_portfolio_value",
            "--iterations", "5",
        )
        # Should succeed (exit 0 or 1), not error (exit 2)
        assert result.returncode in (0, 1)

    def test_unknown_function_name_exits_2(self):
        result = _run(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--function", "does_not_exist",
            "--iterations", "5",
        )
        assert result.returncode == 2

    def test_unknown_function_error_is_json_on_stderr(self):
        result = _run(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--function", "does_not_exist",
            "--iterations", "5",
        )
        data = json.loads(result.stderr)
        assert data["status"] == "error"


# ---------------------------------------------------------------------------
# --adversarial flag
# ---------------------------------------------------------------------------

class TestAdversarialFlag:
    def test_adversarial_flag_accepted(self):
        result = _run(
            "test", str(FIXTURE_DIR / "broken_covariance.py"),
            "--scenario", "liquidity_crash",
            "--adversarial",
            "--iterations", "3",
            "--population", "10",
        )
        assert result.returncode in (0, 1)

    def test_adversarial_output_is_valid_json(self):
        result = _run(
            "test", str(FIXTURE_DIR / "broken_covariance.py"),
            "--scenario", "liquidity_crash",
            "--adversarial",
            "--iterations", "3",
            "--population", "10",
        )
        data = json.loads(result.stdout)  # must not raise
        assert "shatter_points" in data

    def test_adversarial_finds_failures(self):
        result = _run(
            "test", str(FIXTURE_DIR / "broken_covariance.py"),
            "--scenario", "liquidity_crash",
            "--adversarial",
            "--iterations", "25",
            "--population", "20",
            "--seed", "42",
        )
        data = json.loads(result.stdout)
        assert data["status"] == "failures_detected"
        psd_failures = [
            sp for sp in data["shatter_points"]
            if sp["failure_type"] == "non_psd_matrix"
        ]
        assert len(psd_failures) >= 1

    def test_population_flag_accepted(self):
        result = _run(
            "test", str(FIXTURE_DIR / "broken_covariance.py"),
            "--scenario", "liquidity_crash",
            "--adversarial",
            "--population", "5",
            "--iterations", "2",
        )
        assert result.returncode in (0, 1)

    def test_adversarial_same_seed_reproducible(self):
        args = [
            "test", str(FIXTURE_DIR / "broken_covariance.py"),
            "--scenario", "liquidity_crash",
            "--adversarial",
            "--iterations", "5",
            "--population", "10",
            "--seed", "42",
        ]
        r1 = json.loads(_run(*args).stdout)
        r2 = json.loads(_run(*args).stdout)
        assert r1["summary"]["total_failures"] == r2["summary"]["total_failures"]
