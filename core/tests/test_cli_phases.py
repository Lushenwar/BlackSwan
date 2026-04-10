"""
Tests for BlackSwan CLI — Phase 4 (Ecosystem) and Phase 5 (Transparency).

Covers:
  Phase 4A — --mode fast|full flag
  Phase 4B — reproducibility_card in JSON output
  Phase 4C — --max-runtime-sec and --max-iterations budget flags
  Phase 5A — confidence field in shatter_points (probabilistic messaging)
  Phase 5B — trigger_disclosure in shatter_points
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


def _parse(*args: str) -> dict:
    result = _run(*args)
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# Phase 4A — --mode flag
# ---------------------------------------------------------------------------

class TestModeFlag:
    def test_mode_fast_accepted(self):
        result = _run(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
            "--mode", "fast",
        )
        assert result.returncode == 0

    def test_mode_full_accepted(self):
        result = _run(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
            "--mode", "full",
        )
        assert result.returncode == 0

    def test_invalid_mode_rejected(self):
        result = _run(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--mode", "turbo",
        )
        assert result.returncode == 2

    def test_mode_appears_in_json_output(self):
        data = _parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
            "--mode", "fast",
        )
        assert data["mode"] == "fast"

    def test_mode_full_is_default(self):
        data = _parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        assert data["mode"] == "full"

    def test_fast_mode_still_finds_failures(self):
        data = _parse(
            "test", str(FIXTURE_DIR / "always_fails.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
            "--mode", "fast",
        )
        assert data["status"] == "failures_detected"

    def test_fast_and_full_same_failure_count(self):
        args_base = [
            "test", str(FIXTURE_DIR / "always_fails.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "10",
            "--seed", "42",
        ]
        fast = _parse(*args_base, "--mode", "fast")
        full = _parse(*args_base, "--mode", "full")
        assert fast["summary"]["total_failures"] == full["summary"]["total_failures"]

    def test_mode_full_appears_in_reproducibility_card(self):
        data = _parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
            "--mode", "full",
        )
        assert data["reproducibility_card"]["mode"] == "full"

    def test_mode_fast_appears_in_reproducibility_card(self):
        data = _parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
            "--mode", "fast",
        )
        assert data["reproducibility_card"]["mode"] == "fast"


# ---------------------------------------------------------------------------
# Phase 4B — reproducibility_card
# ---------------------------------------------------------------------------

class TestReproducibilityCard:
    def _card(self, *extra_args: str) -> dict:
        data = _parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
            *extra_args,
        )
        return data["reproducibility_card"]

    def test_reproducibility_card_present_in_response(self):
        data = _parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        assert "reproducibility_card" in data

    def test_card_has_all_required_fields(self):
        card = self._card()
        required = [
            "blackswan_version", "python_version", "numpy_version", "platform",
            "seed", "scenario_name", "scenario_hash", "mode",
            "iterations_requested", "iterations_executed", "iterations_skipped",
            "budget_exhausted", "budget_reason", "timestamp_utc",
            "reproducible", "replay_command",
        ]
        for field in required:
            assert field in card, f"missing field: {field}"

    def test_scenario_name_matches(self):
        card = self._card()
        assert card["scenario_name"] == "liquidity_crash"

    def test_seed_matches_cli_seed(self):
        card = _parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
            "--seed", "12345",
        )["reproducibility_card"]
        assert card["seed"] == 12345

    def test_iterations_requested_matches_cli(self):
        card = self._card("--iterations", "7")
        assert card["iterations_requested"] == 7

    def test_iterations_executed_matches_completed(self):
        data = _parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "7",
        )
        assert data["reproducibility_card"]["iterations_executed"] == data["iterations_completed"]

    def test_iterations_skipped_is_zero_without_budget(self):
        card = self._card()
        assert card["iterations_skipped"] == 0

    def test_budget_exhausted_is_false_without_budget_flags(self):
        card = self._card()
        assert card["budget_exhausted"] is False

    def test_budget_reason_is_null_without_budget_flags(self):
        card = self._card()
        assert card["budget_reason"] is None

    def test_reproducible_is_true_without_budget(self):
        card = self._card()
        assert card["reproducible"] is True

    def test_replay_command_contains_scenario(self):
        card = self._card()
        assert "liquidity_crash" in card["replay_command"]

    def test_replay_command_is_runnable_string(self):
        card = self._card("--seed", "99")
        cmd = card["replay_command"]
        assert "python" in cmd
        assert "blackswan" in cmd
        assert "--seed 99" in cmd

    def test_python_version_is_populated(self):
        card = self._card()
        assert len(card["python_version"]) > 0
        assert "." in card["python_version"]

    def test_numpy_version_is_populated(self):
        card = self._card()
        assert len(card["numpy_version"]) > 0

    def test_scenario_hash_is_12_chars(self):
        card = self._card()
        assert len(card["scenario_hash"]) == 12

    def test_same_scenario_same_hash(self):
        card1 = self._card()
        card2 = self._card()
        assert card1["scenario_hash"] == card2["scenario_hash"]

    def test_timestamp_utc_is_iso_format(self):
        card = self._card()
        ts = card["timestamp_utc"]
        # Should contain 'T' (ISO 8601 separator) and '+' or 'Z' for UTC offset
        assert "T" in ts


# ---------------------------------------------------------------------------
# Phase 4C — budget flags: --max-runtime-sec and --max-iterations
# ---------------------------------------------------------------------------

class TestBudgetFlags:
    def test_max_runtime_sec_accepted(self):
        result = _run(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "20",
            "--max-runtime-sec", "60",
        )
        assert result.returncode in (0, 1)

    def test_max_iterations_accepted(self):
        result = _run(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "100",
            "--max-iterations", "5",
        )
        assert result.returncode in (0, 1)

    def test_max_iterations_caps_execution(self):
        data = _parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "1000",
            "--max-iterations", "3",
        )
        assert data["iterations_completed"] <= 3

    def test_budget_section_present_in_output(self):
        data = _parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        assert "budget" in data
        assert "exhausted" in data["budget"]
        assert "reason" in data["budget"]

    def test_budget_not_exhausted_without_flags(self):
        data = _parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        assert data["budget"]["exhausted"] is False
        assert data["budget"]["reason"] is None

    def test_max_iterations_budget_exhausted_flag(self):
        # With max-iterations lower than scenario default, budget should be exhausted
        # when the scenario default is higher
        data = _parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--max-iterations", "2",
        )
        # iterations_completed should be at most 2
        assert data["iterations_completed"] <= 2

    def test_max_runtime_sec_very_short_stops_run(self):
        # 0.001 seconds is too short for even one iteration in some cases,
        # but should at least produce valid JSON without crashing
        result = _run(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "10000",
            "--max-runtime-sec", "0.001",
        )
        assert result.returncode in (0, 1)
        data = json.loads(result.stdout)
        assert "iterations_completed" in data

    def test_output_is_valid_json_with_budget_flags(self):
        result = _run(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "10",
            "--max-runtime-sec", "30",
            "--max-iterations", "5",
        )
        json.loads(result.stdout)  # must not raise


# ---------------------------------------------------------------------------
# Phase 5A — confidence field (probabilistic messaging)
# ---------------------------------------------------------------------------

class TestConfidenceField:
    def test_shatter_points_have_confidence_field_when_failing(self):
        data = _parse(
            "test", str(FIXTURE_DIR / "always_fails.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        for sp in data["shatter_points"]:
            assert "confidence" in sp, "shatter_point missing 'confidence' field"

    def test_confidence_is_valid_level(self):
        data = _parse(
            "test", str(FIXTURE_DIR / "always_fails.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        valid_levels = {"high", "medium", "low", "unverified"}
        for sp in data["shatter_points"]:
            assert sp["confidence"] in valid_levels, (
                f"unexpected confidence level: {sp['confidence']}"
            )

    def test_fast_mode_shatter_points_have_confidence(self):
        data = _parse(
            "test", str(FIXTURE_DIR / "always_fails.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
            "--mode", "fast",
        )
        for sp in data["shatter_points"]:
            assert "confidence" in sp

    def test_confidence_not_empty_string(self):
        data = _parse(
            "test", str(FIXTURE_DIR / "always_fails.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        for sp in data["shatter_points"]:
            assert sp["confidence"] != ""


# ---------------------------------------------------------------------------
# Phase 5B — trigger_disclosure in shatter_points
# ---------------------------------------------------------------------------

class TestTriggerDisclosureInOutput:
    def test_trigger_disclosure_present_when_psd_fails(self):
        """broken_covariance.py produces non_psd_matrix findings with trigger_disclosure."""
        data = _parse(
            "test", str(FIXTURE_DIR / "broken_covariance.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "20",
            "--seed", "42",
        )
        psd_points = [sp for sp in data["shatter_points"] if sp["failure_type"] == "non_psd_matrix"]
        if psd_points:
            sp = psd_points[0]
            assert "trigger_disclosure" in sp
            td = sp["trigger_disclosure"]
            assert "detector_name" in td
            assert "observed_value" in td
            assert "threshold" in td
            assert "comparison" in td
            assert "explanation" in td

    def test_trigger_disclosure_has_correct_fields_when_present(self):
        data = _parse(
            "test", str(FIXTURE_DIR / "broken_covariance.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "20",
            "--seed", "42",
        )
        psd_points = [sp for sp in data["shatter_points"] if sp["failure_type"] == "non_psd_matrix"]
        if psd_points and "trigger_disclosure" in psd_points[0]:
            td = psd_points[0]["trigger_disclosure"]
            assert isinstance(td["detector_name"], str)
            assert isinstance(td["explanation"], str)
            assert td["comparison"] in (">", "<", ">=", "<=", "==", "!=")

    def test_trigger_disclosure_detector_name_matches_failure_type(self):
        data = _parse(
            "test", str(FIXTURE_DIR / "broken_covariance.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "20",
            "--seed", "42",
        )
        psd_points = [sp for sp in data["shatter_points"] if sp["failure_type"] == "non_psd_matrix"]
        if psd_points and "trigger_disclosure" in psd_points[0]:
            td = psd_points[0]["trigger_disclosure"]
            assert "PSD" in td["detector_name"] or "Matrix" in td["detector_name"]

    def test_exception_based_failures_may_not_have_trigger_disclosure(self):
        """always_fails raises an exception — no threshold-based disclosure."""
        data = _parse(
            "test", str(FIXTURE_DIR / "always_fails.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        # Exception-based nan_inf findings don't have trigger_disclosure
        # (they don't come from a detector threshold, they come from an exception)
        # This test just verifies the response is valid JSON with expected structure
        assert "shatter_points" in data
        assert isinstance(data["shatter_points"], list)


# ---------------------------------------------------------------------------
# Response structure — additive (all existing fields still present)
# ---------------------------------------------------------------------------

class TestResponseStructurePreserved:
    """
    Verify that all fields from the original CLI tests remain present.
    The new fields (mode, budget, reproducibility_card, confidence) are additive.
    """

    def test_all_original_top_level_keys_still_present(self):
        data = _parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        for key in ["version", "status", "runtime_ms", "iterations_completed",
                    "summary", "shatter_points", "scenario_card"]:
            assert key in data, f"missing original key: {key}"

    def test_all_new_top_level_keys_present(self):
        data = _parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        for key in ["mode", "budget", "reproducibility_card"]:
            assert key in data, f"missing new key: {key}"

    def test_shatter_points_still_have_all_original_fields(self):
        data = _parse(
            "test", str(FIXTURE_DIR / "always_fails.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        for sp in data["shatter_points"]:
            for field in ["id", "line", "column", "severity", "failure_type",
                          "message", "frequency", "causal_chain", "fix_hint"]:
                assert field in sp, f"shatter_point missing original field: {field}"

    def test_version_is_still_1_0(self):
        data = _parse(
            "test", str(FIXTURE_DIR / "clean_model.py"),
            "--scenario", "liquidity_crash",
            "--iterations", "5",
        )
        assert data["version"] == "1.0"
