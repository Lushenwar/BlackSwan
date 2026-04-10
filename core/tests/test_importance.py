"""
Tests for ImportanceSampler (Phase 3.1).
"""

from __future__ import annotations

import numpy as np
import pytest

from blackswan.engine.importance import ImportanceSampler


_RANGES = {"vol": (1.0, 4.0), "correlation": (0.1, 0.45), "spread": (1.5, 3.5)}


class TestImportanceSamplerUniform:
    def test_uniform_when_no_failures(self):
        sampler = ImportanceSampler(_RANGES)
        rng = np.random.default_rng(0)
        samples = [sampler.sample(rng) for _ in range(200)]
        # All samples must be within ranges
        for s in samples:
            for name, (lo, hi) in _RANGES.items():
                assert lo <= s[name] <= hi

    def test_returns_all_param_keys(self):
        sampler = ImportanceSampler(_RANGES)
        rng = np.random.default_rng(0)
        s = sampler.sample(rng)
        assert set(s.keys()) == set(_RANGES.keys())

    def test_uniform_before_min_failures(self):
        sampler = ImportanceSampler(_RANGES, min_failures_before_bias=10)
        rng = np.random.default_rng(0)
        # Record only 3 failures (below threshold)
        for _ in range(3):
            sampler.record_failure({"vol": 3.9, "correlation": 0.44, "spread": 3.4})
        # Still should sample uniformly (not crash, not be all near failure point)
        samples = [sampler.sample(rng) for _ in range(50)]
        vols = [s["vol"] for s in samples]
        # With uniform sampling, values should be spread across [1.0, 4.0]
        assert min(vols) < 2.5  # some low values expected

    def test_failure_count_increments(self):
        sampler = ImportanceSampler(_RANGES)
        assert sampler.failure_count == 0
        sampler.record_failure({"vol": 2.0, "correlation": 0.2})
        assert sampler.failure_count == 1


class TestImportanceSamplerBiased:
    def _fill_failures(self, sampler: ImportanceSampler, n: int = 10):
        for _ in range(n):
            # All failures near high vol + high correlation
            sampler.record_failure({"vol": 3.8, "correlation": 0.43, "spread": 3.4})

    def test_tail_samples_stay_in_range(self):
        sampler = ImportanceSampler(_RANGES, bandwidth=0.2, min_failures_before_bias=5)
        self._fill_failures(sampler, 10)
        rng = np.random.default_rng(42)
        for _ in range(100):
            s = sampler.sample(rng)
            for name, (lo, hi) in _RANGES.items():
                assert lo <= s[name] <= hi, f"{name}={s[name]} out of [{lo},{hi}]"

    def test_tail_samples_biased_toward_failure_region(self):
        """After enough failures near high vol, the mean sample vol should be above midpoint."""
        sampler = ImportanceSampler(
            _RANGES,
            bandwidth=0.1,
            tail_fraction=0.95,  # almost always tail-sample
            min_failures_before_bias=5,
        )
        self._fill_failures(sampler, 20)
        rng = np.random.default_rng(99)
        vols = [sampler.sample(rng)["vol"] for _ in range(300)]
        mean_vol = sum(vols) / len(vols)
        # With failures at vol=3.8 and tail_fraction=0.95, mean should be well above midpoint 2.5
        assert mean_vol > 3.0

    def test_reset_clears_failure_points(self):
        sampler = ImportanceSampler(_RANGES)
        self._fill_failures(sampler, 10)
        assert sampler.failure_count == 10
        sampler.reset()
        assert sampler.failure_count == 0

    def test_record_failure_ignores_array_values(self):
        import numpy as np
        sampler = ImportanceSampler(_RANGES)
        # Array values should be silently ignored (not stored, not crash)
        sampler.record_failure({"vol": np.array([1.0, 2.0]), "correlation": 0.3})
        # Only scalar values in _RANGES are stored
        assert sampler.failure_count == 1  # point stored with just correlation
        # Subsequent samples should not crash
        rng = np.random.default_rng(0)
        s = sampler.sample(rng)
        assert isinstance(s, dict)

    def test_deterministic_with_same_seed(self):
        sampler_a = ImportanceSampler(_RANGES)
        sampler_b = ImportanceSampler(_RANGES)
        failure = {"vol": 3.5, "correlation": 0.4, "spread": 3.0}
        for _ in range(8):
            sampler_a.record_failure(failure)
            sampler_b.record_failure(failure)
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)
        samples_a = [sampler_a.sample(rng_a) for _ in range(20)]
        samples_b = [sampler_b.sample(rng_b) for _ in range(20)]
        for a, b in zip(samples_a, samples_b):
            for k in a:
                assert a[k] == pytest.approx(b[k])


class TestImportanceSamplerEdgeCases:
    def test_empty_param_ranges(self):
        sampler = ImportanceSampler({})
        rng = np.random.default_rng(0)
        assert sampler.sample(rng) == {}

    def test_single_param(self):
        sampler = ImportanceSampler({"x": (0.0, 1.0)})
        rng = np.random.default_rng(0)
        s = sampler.sample(rng)
        assert "x" in s
        assert 0.0 <= s["x"] <= 1.0

    def test_record_failure_with_unknown_keys_ignored(self):
        sampler = ImportanceSampler({"vol": (1.0, 4.0)})
        # extra keys not in param_ranges are silently dropped
        sampler.record_failure({"vol": 3.5, "unknown_param": 99.0})
        assert sampler.failure_count == 1
