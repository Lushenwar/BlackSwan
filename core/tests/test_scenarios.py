"""
Tests for the scenario registry.

Covers: loading a preset by name, full field parsing into Scenario/Perturbation,
validation errors, list_scenarios(), all 5 presets load cleanly, and the
Scenario.apply() integration with StressRunner protocol.
"""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from blackswan.scenarios.registry import Scenario, load_scenario, list_scenarios
from blackswan.engine.perturbation import Perturbation


# ---------------------------------------------------------------------------
# Load liquidity_crash — happy path
# ---------------------------------------------------------------------------

class TestLoadLiquidityCrash:
    def setup_method(self):
        self.scenario = load_scenario("liquidity_crash")

    def test_returns_scenario_instance(self):
        assert isinstance(self.scenario, Scenario)

    def test_name_is_liquidity_crash(self):
        assert self.scenario.name == "liquidity_crash"

    def test_display_name_is_present_and_non_empty(self):
        assert isinstance(self.scenario.display_name, str)
        assert len(self.scenario.display_name) > 0

    def test_description_is_present_and_non_empty(self):
        assert isinstance(self.scenario.description, str)
        assert len(self.scenario.description) > 0

    def test_has_exactly_four_perturbations(self):
        assert len(self.scenario.perturbations) == 4

    def test_all_perturbations_are_perturbation_instances(self):
        for p in self.scenario.perturbations:
            assert isinstance(p, Perturbation)

    def test_default_iterations_is_5000(self):
        assert self.scenario.default_iterations == 5000

    def test_default_seed_is_42(self):
        assert self.scenario.default_seed == 42


# ---------------------------------------------------------------------------
# Perturbation field values for liquidity_crash
# ---------------------------------------------------------------------------

class TestLiquidityCrashPerturbationFields:
    def setup_method(self):
        self.perturbs = load_scenario("liquidity_crash").perturbations

    def _by_target(self, target: str) -> Perturbation:
        matches = [p for p in self.perturbs if p.target == target]
        assert len(matches) == 1, f"Expected exactly one perturbation for target={target!r}"
        return matches[0]

    def test_spread_is_multiplicative_uniform(self):
        p = self._by_target("spread")
        assert p.type == "multiplicative"
        assert p.distribution == "uniform"

    def test_spread_range_is_1_5_to_3_5(self):
        p = self._by_target("spread")
        assert p.params["low"] == pytest.approx(1.5)
        assert p.params["high"] == pytest.approx(3.5)

    def test_vol_is_multiplicative_lognormal(self):
        p = self._by_target("vol")
        assert p.type == "multiplicative"
        assert p.distribution == "lognormal"

    def test_vol_lognormal_params_match_spec(self):
        p = self._by_target("vol")
        assert p.params["mu"] == pytest.approx(1.8)
        assert p.params["sigma"] == pytest.approx(0.3)

    def test_correlation_is_additive_uniform(self):
        p = self._by_target("correlation")
        assert p.type == "additive"
        assert p.distribution == "uniform"

    def test_correlation_shift_range(self):
        p = self._by_target("correlation")
        assert p.params["low"] == pytest.approx(0.10)
        assert p.params["high"] == pytest.approx(0.35)

    def test_correlation_has_clip_max_of_0_99(self):
        p = self._by_target("correlation")
        assert p.clamp is not None
        _lo, hi = p.clamp
        assert hi == pytest.approx(0.99)

    def test_turnover_is_multiplicative_uniform(self):
        p = self._by_target("turnover")
        assert p.type == "multiplicative"
        assert p.distribution == "uniform"

    def test_turnover_range_is_0_3_to_0_7(self):
        p = self._by_target("turnover")
        assert p.params["low"] == pytest.approx(0.30)
        assert p.params["high"] == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# Scenario.apply() — satisfies StressRunner duck-typed protocol
# ---------------------------------------------------------------------------

class TestScenarioApply:
    def test_apply_returns_dict(self):
        scenario = load_scenario("liquidity_crash")
        rng = np.random.default_rng(42)
        result = scenario.apply({"spread": 100.0, "vol": 0.2, "correlation": 0.5, "turnover": 1.0}, rng)
        assert isinstance(result, dict)

    def test_apply_perturbs_targeted_keys(self):
        scenario = load_scenario("liquidity_crash")
        rng = np.random.default_rng(42)
        base = {"spread": 100.0, "vol": 0.2, "correlation": 0.5, "turnover": 1.0}
        result = scenario.apply(base, rng)
        # spread is multiplicative [1.5, 3.5] → result must be in [150, 350]
        assert 100.0 * 1.5 <= result["spread"] <= 100.0 * 3.5

    def test_apply_is_deterministic_given_same_rng_state(self):
        scenario = load_scenario("liquidity_crash")
        base = {"spread": 100.0, "vol": 0.2, "correlation": 0.5, "turnover": 1.0}
        result_a = scenario.apply(base, np.random.default_rng(42))
        result_b = scenario.apply(base, np.random.default_rng(42))
        assert result_a == result_b

    def test_scenario_iterations_attribute_exists(self):
        """Scenario must expose .iterations so StressRunner can use it directly."""
        scenario = load_scenario("liquidity_crash")
        assert scenario.iterations == scenario.default_iterations


# ---------------------------------------------------------------------------
# list_scenarios()
# ---------------------------------------------------------------------------

class TestListScenarios:
    def test_returns_list_of_strings(self):
        names = list_scenarios()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_includes_all_five_presets(self):
        names = list_scenarios()
        expected = {
            "liquidity_crash",
            "vol_spike",
            "correlation_breakdown",
            "rate_shock",
            "missing_data",
        }
        assert expected.issubset(set(names))

    def test_returns_sorted_names(self):
        names = list_scenarios()
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# All 5 presets load without error
# ---------------------------------------------------------------------------

class TestAllPresetsLoad:
    @pytest.mark.parametrize("name", [
        "liquidity_crash",
        "vol_spike",
        "correlation_breakdown",
        "rate_shock",
        "missing_data",
    ])
    def test_preset_loads_successfully(self, name):
        scenario = load_scenario(name)
        assert isinstance(scenario, Scenario)
        assert scenario.name == name
        assert scenario.default_iterations > 0
        assert len(scenario.perturbations) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidationErrors:
    def _write_yaml(self, tmp_path: Path, content: str) -> Path:
        """Write a YAML string to a temp .yaml file and return its path."""
        p = tmp_path / "bad_scenario.yaml"
        p.write_text(textwrap.dedent(content))
        return p

    def test_missing_top_level_field_raises_valueerror(self, tmp_path):
        path = self._write_yaml(tmp_path, """
            name: bad
            display_name: Bad
            description: missing defaults and perturbations
        """)
        with pytest.raises(ValueError, match="missing required"):
            from blackswan.scenarios.registry import _load_from_path
            _load_from_path(path)

    def test_missing_perturbation_field_raises_valueerror(self, tmp_path):
        path = self._write_yaml(tmp_path, """
            name: bad
            display_name: Bad
            description: A scenario
            defaults:
              iterations: 100
              seed: 1
            perturbations:
              - target: vol
                type: multiplicative
                # missing distribution and params
        """)
        with pytest.raises(ValueError, match="missing required"):
            from blackswan.scenarios.registry import _load_from_path
            _load_from_path(path)

    def test_unknown_perturbation_type_raises_valueerror(self, tmp_path):
        path = self._write_yaml(tmp_path, """
            name: bad
            display_name: Bad
            description: A scenario
            defaults:
              iterations: 100
              seed: 1
            perturbations:
              - target: vol
                type: explode
                distribution: uniform
                params:
                  low: 1.0
                  high: 2.0
        """)
        with pytest.raises(ValueError, match="unknown type"):
            from blackswan.scenarios.registry import _load_from_path
            _load_from_path(path)

    def test_nonexistent_scenario_raises_filenotfounderror(self):
        with pytest.raises(FileNotFoundError, match="no preset"):
            load_scenario("does_not_exist")
