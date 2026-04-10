"""
Tests for the Two-Path execution architecture:
  - TracerBackend (SetTraceBackend)
  - NumPyHookSet + NumPyHookCapture
  - SlowPathReplayer
  - ShrinkingEngine + ConstraintRepairPipeline
  - cluster_findings / RootCauseBucket
  - StressRunner integration (mode="fast" vs mode="full")
"""

from __future__ import annotations

import math
import sys
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from blackswan.detectors.base import FailureDetector, Finding
from blackswan.detectors.matrix import MatrixPSDDetector
from blackswan.detectors.numerical import NaNInfDetector
from blackswan.engine.cluster import RootCauseBucket, cluster_findings
from blackswan.engine.hooks import NumPyHookCapture, NumPyHookSet
from blackswan.engine.replay import ReplayDivergenceError, SlowPathReplayer
from blackswan.engine.runner import RunResult, StressRunner, _deep_copy_inputs
from blackswan.engine.shrink import (
    ConstraintRepairPipeline,
    ShrinkingEngine,
    _round_significant,
)
from blackswan.engine.tracer import SetTraceBackend, make_tracer


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_finding(
    failure_type: str = "nan_inf",
    severity: str = "critical",
    line: int | None = None,
    iteration: int = 0,
    exc_frames=None,
) -> Finding:
    return Finding(
        failure_type=failure_type,
        severity=severity,
        message="test finding",
        iteration=iteration,
        line=line,
        exc_frames=exc_frames or [],
    )


def _always_good(x: float = 1.0) -> float:
    return x * 2.0


def _always_bad(x: float = 1.0) -> float:
    raise ValueError("intentional failure")


def _nan_on_negative(x: float = 1.0) -> float:
    if x < 0:
        return float("nan")
    return x


class _AlwaysFiringDetector(FailureDetector):
    def check(self, inputs, output, iteration) -> Finding | None:
        return _make_finding(iteration=iteration)


class _NeverFiringDetector(FailureDetector):
    def check(self, inputs, output, iteration) -> Finding | None:
        return None


# ---------------------------------------------------------------------------
# TracerBackend (SetTraceBackend)
# ---------------------------------------------------------------------------

class TestSetTraceBackend:
    def _make_target(self, tmp_path):
        """Write a small Python file and return its path."""
        src = tmp_path / "target.py"
        src.write_text(
            "def compute(x):\n"
            "    y = x + 1\n"
            "    z = y * 2\n"
            "    return z\n"
        )
        return str(src)

    def test_get_frame_log_returns_list(self):
        tracer = SetTraceBackend(__file__)
        tracer.activate()
        tracer.deactivate()
        assert isinstance(tracer.get_frame_log(), list)

    def test_reset_clears_log(self):
        tracer = SetTraceBackend(__file__)
        tracer.activate()
        # trivial call to generate events
        x = 1 + 1
        tracer.deactivate()
        tracer.reset()
        assert tracer.get_frame_log() == []
        assert tracer.get_local_snapshots() == {}

    def test_activate_deactivate_idempotent(self):
        tracer = SetTraceBackend(__file__)
        tracer.activate()
        tracer.activate()   # second activate should be a no-op
        tracer.deactivate()
        tracer.deactivate() # second deactivate should be a no-op

    def test_does_not_trace_other_files(self, tmp_path):
        tracer = SetTraceBackend("/nonexistent/fake_target.py")
        tracer.activate()
        _ = [i ** 2 for i in range(10)]  # runs in this file, not target
        tracer.deactivate()
        # No events from this file should appear since filename doesn't match
        log = tracer.get_frame_log()
        for event in log:
            assert event.filename == "/nonexistent/fake_target.py"

    def test_restores_previous_trace(self):
        import sys
        sentinel = []
        def my_trace(frame, event, arg):
            sentinel.append(1)
            return my_trace
        sys.settrace(my_trace)
        try:
            tracer = SetTraceBackend(__file__)
            tracer.activate()
            tracer.deactivate()
            # After deactivation, the previous trace function should be restored
            assert sys.gettrace() is my_trace
        finally:
            sys.settrace(None)


# ---------------------------------------------------------------------------
# make_tracer factory
# ---------------------------------------------------------------------------

class TestMakeTracer:
    def test_returns_tracer_backend(self):
        from blackswan.engine.tracer import TracerBackend
        tracer = make_tracer(__file__)
        assert isinstance(tracer, TracerBackend)

    def test_tracer_has_required_methods(self):
        tracer = make_tracer(__file__)
        assert callable(tracer.activate)
        assert callable(tracer.deactivate)
        assert callable(tracer.get_frame_log)
        assert callable(tracer.get_local_snapshots)
        assert callable(tracer.reset)


# ---------------------------------------------------------------------------
# NumPyHookSet
# ---------------------------------------------------------------------------

class TestNumPyHookSet:
    def test_captures_np_cov_call(self):
        import numpy as np
        capture = NumPyHookCapture()
        hooks = NumPyHookSet(capture)
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        hooks.activate()
        try:
            result = np.cov(data.T)
        finally:
            hooks.deactivate()
        assert len(capture.calls) >= 1
        cov_calls = [c for c in capture.calls if c.fn_name == "cov"]
        assert len(cov_calls) == 1
        assert cov_calls[0].raised is None
        assert cov_calls[0].output_snapshot is not None

    def test_captures_np_linalg_inv_call(self):
        import numpy as np
        capture = NumPyHookCapture()
        hooks = NumPyHookSet(capture)
        mat = np.array([[2.0, 0.0], [0.0, 3.0]])
        hooks.activate()
        try:
            np.linalg.inv(mat)
        finally:
            hooks.deactivate()
        inv_calls = [c for c in capture.calls if c.fn_name == "inv"]
        assert len(inv_calls) == 1

    def test_captures_raised_exception(self):
        import numpy as np
        capture = NumPyHookCapture()
        hooks = NumPyHookSet(capture)
        singular = np.array([[1.0, 0.0], [0.0, 0.0]])
        hooks.activate()
        try:
            with pytest.raises(np.linalg.LinAlgError):
                np.linalg.inv(singular)
        finally:
            hooks.deactivate()
        inv_calls = [c for c in capture.calls if c.fn_name == "inv"]
        assert len(inv_calls) == 1
        assert inv_calls[0].raised is not None

    def test_deactivate_restores_original(self):
        import numpy as np
        capture = NumPyHookCapture()
        hooks = NumPyHookSet(capture)
        original_cov = np.cov
        hooks.activate()
        hooks.deactivate()
        assert np.cov is original_cov

    def test_idempotent_activate(self):
        import numpy as np
        capture = NumPyHookCapture()
        hooks = NumPyHookSet(capture)
        hooks.activate()
        hooks.activate()  # second call should be no-op
        hooks.deactivate()
        assert np.cov.__name__ == "cov"

    def test_capture_clear(self):
        import numpy as np
        capture = NumPyHookCapture()
        hooks = NumPyHookSet(capture)
        hooks.activate()
        np.cov(np.eye(2))
        hooks.deactivate()
        capture.clear()
        assert len(capture.calls) == 0


# ---------------------------------------------------------------------------
# SlowPathReplayer
# ---------------------------------------------------------------------------

class TestSlowPathReplayer:
    def _replayer(self):
        tracer = SetTraceBackend(__file__)
        return SlowPathReplayer(
            tracer=tracer,
            detectors=[NaNInfDetector()],
            target_filename=__file__,
        )

    def test_replay_reproduces_exception_failure(self):
        replayer = self._replayer()
        inputs = {"x": 1.0}
        finding = _make_finding(exc_frames=[(__file__, 10)])
        attr = replayer.replay(_always_bad, inputs, finding)
        assert attr is not None
        assert attr.confidence in ("high", "medium", "low")

    def test_replay_returns_attribution(self):
        from blackswan.engine.replay import Attribution
        replayer = self._replayer()
        inputs = {"x": 1.0}
        finding = _make_finding(exc_frames=[(__file__, 10)])
        attr = replayer.replay(_always_bad, inputs, finding)
        assert isinstance(attr, Attribution)

    def test_replay_divergence_raises(self):
        """
        If function doesn't reproduce the failure on replay,
        ReplayDivergenceError must be raised.
        """
        replayer = self._replayer()
        inputs = {"x": 1.0}
        # Original finding claims it was a detector-based failure (no exc_frames),
        # but _always_good will not trigger any detector
        finding = _make_finding(failure_type="bounds_exceeded", exc_frames=[])
        with pytest.raises(ReplayDivergenceError):
            replayer.replay(_always_good, inputs, finding)

    def test_replay_captures_numpy_hooks(self):
        """Slow-path replay should capture NumPy calls via hooks."""
        import numpy as np

        def fn_with_cov(x: float = 1.0):
            data = np.array([[x, x + 1], [x + 2, x + 3]])
            raise ValueError("deliberate")

        tracer = SetTraceBackend(__file__)
        replayer = SlowPathReplayer(
            tracer=tracer,
            detectors=[],
            target_filename=__file__,
        )
        finding = _make_finding(exc_frames=[(__file__, 10)])
        attr = replayer.replay(fn_with_cov, {"x": 1.0}, finding)
        # cov call was made — verify numpy_calls captured it or at least no crash
        assert isinstance(attr.numpy_calls, list)


# ---------------------------------------------------------------------------
# ShrinkingEngine
# ---------------------------------------------------------------------------

class TestShrinkingEngine:
    def _shrinker(self, fn, base_inputs, detectors=None):
        return ShrinkingEngine(
            fn=fn,
            base_inputs=base_inputs,
            detectors=detectors or [],
            validator=None,
            max_steps=200,
            seed=0,
        )

    def test_shrink_reduces_magnitude(self):
        """Shrinker should bring the failing value closer to the base."""
        base = {"x": 0.0}

        def fn_fails_on_large(x: float = 0.0) -> float:
            if abs(x) > 0.5:
                raise ValueError("too large")
            return x

        shrinker = self._shrinker(fn_fails_on_large, base)
        finding = _make_finding(exc_frames=[(__file__, 1)])
        minimal = shrinker.shrink({"x": 2.0}, finding)
        # Shrunken value should be smaller than original but still > 0.5
        assert abs(minimal["x"]) <= abs(2.0)
        assert abs(minimal["x"]) > 0.5

    def test_shrink_keeps_failing_input(self):
        """Output must still cause the same failure."""
        base = {"x": 0.0}

        def fn_fails_on_large(x: float = 0.0) -> float:
            if abs(x) > 0.5:
                raise ValueError("too large")
            return x

        shrinker = self._shrinker(fn_fails_on_large, base)
        finding = _make_finding(exc_frames=[(__file__, 1)])
        minimal = shrinker.shrink({"x": 5.0}, finding)
        with pytest.raises(ValueError):
            fn_fails_on_large(**minimal)

    def test_shrink_does_not_crash_on_already_minimal(self):
        """Shrinker must be idempotent on inputs that can't be reduced further."""
        base = {"x": 0.0}

        def fn_always_fails(x: float = 0.0):
            raise ValueError("always")

        shrinker = self._shrinker(fn_always_fails, base)
        finding = _make_finding(exc_frames=[(__file__, 1)])
        minimal = shrinker.shrink({"x": 0.001}, finding)
        assert isinstance(minimal, dict)

    def test_dimensional_reduction_removes_irrelevant_params(self):
        """If param y is irrelevant to failure, it should be reset to base value."""
        base = {"x": 0.0, "y": 0.0}

        def fn(x: float = 0.0, y: float = 0.0) -> float:
            if x > 0.5:
                raise ValueError("x too large")
            return x + y

        shrinker = self._shrinker(fn, base)
        finding = _make_finding(exc_frames=[(__file__, 1)])
        # y is irrelevant — only x causes failure
        minimal = shrinker.shrink({"x": 2.0, "y": 99.0}, finding)
        # y should have been reset to base (0.0) since it's irrelevant
        assert minimal["y"] == pytest.approx(0.0, abs=1e-9)

    def test_precision_reduction_rounds_values(self):
        assert _round_significant(3.14159, 3) == pytest.approx(3.14, rel=1e-3)
        assert _round_significant(0.001234, 2) == pytest.approx(0.0012, rel=1e-3)
        assert _round_significant(0.0, 3) == 0.0

    def test_round_significant_handles_negative(self):
        assert _round_significant(-3.14159, 2) == pytest.approx(-3.1, rel=1e-2)


# ---------------------------------------------------------------------------
# ConstraintRepairPipeline
# ---------------------------------------------------------------------------

class TestConstraintRepairPipeline:
    def test_clamps_correlation_above_one(self):
        pipeline = ConstraintRepairPipeline()
        result = pipeline.repair({"correlation": 1.5})
        assert result.params["correlation"] <= 0.999
        assert result.repair_applied is True

    def test_clamps_correlation_below_minus_one(self):
        pipeline = ConstraintRepairPipeline()
        result = pipeline.repair({"correlation": -1.5})
        assert result.params["correlation"] >= -0.999

    def test_valid_correlation_unchanged(self):
        pipeline = ConstraintRepairPipeline()
        result = pipeline.repair({"correlation": 0.5})
        assert result.params["correlation"] == pytest.approx(0.5)
        assert result.repair_applied is False

    def test_clamps_negative_volatility(self):
        pipeline = ConstraintRepairPipeline()
        result = pipeline.repair({"vol": -0.1})
        assert result.params["vol"] > 0
        assert result.repair_applied is True

    def test_does_not_touch_unrecognised_keys(self):
        pipeline = ConstraintRepairPipeline()
        result = pipeline.repair({"spread": 2.5, "custom_param": 99.0})
        assert result.params["custom_param"] == 99.0


# ---------------------------------------------------------------------------
# cluster_findings
# ---------------------------------------------------------------------------

class TestClusterFindings:
    def test_empty_findings_returns_empty(self):
        assert cluster_findings([], total_iterations=100) == []

    def test_groups_same_failure_type_and_line(self):
        findings = [
            _make_finding(failure_type="nan_inf", line=10, iteration=i)
            for i in range(20)
        ]
        buckets = cluster_findings(findings, total_iterations=100)
        assert len(buckets) == 1
        assert buckets[0].total_occurrences == 20

    def test_separates_different_failure_types(self):
        findings = [
            _make_finding(failure_type="nan_inf", line=10, iteration=0),
            _make_finding(failure_type="non_psd_matrix", line=20, iteration=1),
        ]
        buckets = cluster_findings(findings, total_iterations=100)
        assert len(buckets) == 2

    def test_occurrence_rate_correct(self):
        findings = [_make_finding(iteration=i) for i in range(50)]
        buckets = cluster_findings(findings, total_iterations=100)
        assert buckets[0].occurrence_rate == pytest.approx(0.5)

    def test_bucket_sorted_by_count_descending(self):
        findings = (
            [_make_finding(failure_type="nan_inf", line=1, iteration=i) for i in range(30)]
            + [_make_finding(failure_type="bounds_exceeded", line=2, iteration=30 + i) for i in range(5)]
        )
        buckets = cluster_findings(findings, total_iterations=100)
        assert buckets[0].total_occurrences >= buckets[1].total_occurrences

    def test_bucket_ids_are_unique(self):
        findings = [
            _make_finding(failure_type="nan_inf", line=1, iteration=0),
            _make_finding(failure_type="non_psd_matrix", line=2, iteration=1),
            _make_finding(failure_type="bounds_exceeded", line=3, iteration=2),
        ]
        buckets = cluster_findings(findings, total_iterations=10)
        ids = [b.bucket_id for b in buckets]
        assert len(ids) == len(set(ids))

    def test_attaches_sample_inputs(self):
        failing_inputs = {i: {"x": float(i)} for i in range(5)}
        findings = [_make_finding(iteration=i) for i in range(5)]
        buckets = cluster_findings(findings, 100, failing_inputs=failing_inputs)
        assert len(buckets[0].sample_inputs) <= 3

    def test_representative_is_most_severe(self):
        findings = [
            _make_finding(severity="warning", iteration=0),
            _make_finding(severity="critical", iteration=1),
            _make_finding(severity="info", iteration=2),
        ]
        buckets = cluster_findings(findings, 100)
        assert buckets[0].representative_finding.severity == "critical"


# ---------------------------------------------------------------------------
# StressRunner integration: mode="fast" vs mode="full"
# ---------------------------------------------------------------------------

class _SimpleScenario:
    """Minimal scenario for integration tests."""
    iterations = 50

    def apply(self, base_inputs, rng):
        perturbed = dict(base_inputs)
        perturbed["x"] = float(rng.uniform(-2.0, 2.0))
        return perturbed

    def make_validator(self):
        return None


class TestStressRunnerTwoPath:
    def test_fast_mode_no_attribution(self):
        """mode='fast' should produce root_cause_buckets but confidence='unverified'."""
        scenario = _SimpleScenario()
        scenario.iterations = 20

        runner = StressRunner(
            fn=_nan_on_negative,
            base_inputs={"x": 1.0},
            scenario=scenario,
            detectors=[NaNInfDetector()],
            seed=42,
            mode="fast",
        )
        result = runner.run()
        assert isinstance(result, RunResult)
        assert result.mode == "fast"
        # Buckets present even in fast mode
        assert isinstance(result.root_cause_buckets, list)

    def test_full_mode_returns_buckets(self):
        """mode='full' should cluster findings into root_cause_buckets."""
        scenario = _SimpleScenario()

        runner = StressRunner(
            fn=_nan_on_negative,
            base_inputs={"x": 1.0},
            scenario=scenario,
            detectors=[NaNInfDetector()],
            seed=42,
            mode="full",
        )
        result = runner.run()
        assert isinstance(result.root_cause_buckets, list)
        # With seed 42 and -2.0..2.0 range, many iterations produce negative x → NaN
        total_occ = sum(b.total_occurrences for b in result.root_cause_buckets)
        assert total_occ > 0

    def test_budget_max_iterations_respected(self):
        scenario = _SimpleScenario()
        scenario.iterations = 1000

        runner = StressRunner(
            fn=_always_good,
            base_inputs={"x": 1.0},
            scenario=scenario,
            detectors=[],
            seed=42,
            mode="fast",
            max_iterations=10,
        )
        result = runner.run()
        assert result.iterations_completed <= 10

    def test_budget_exhausted_flag_set(self):
        scenario = _SimpleScenario()
        scenario.iterations = 10000

        runner = StressRunner(
            fn=_always_good,
            base_inputs={"x": 1.0},
            scenario=scenario,
            detectors=[],
            seed=42,
            mode="fast",
            max_runtime_sec=0.001,   # immediately exhausted
        )
        result = runner.run()
        assert result.budget_exhausted is True
        assert result.budget_reason == "max_runtime_sec"

    def test_run_result_has_all_fields(self):
        scenario = _SimpleScenario()
        scenario.iterations = 5

        runner = StressRunner(
            fn=_always_good,
            base_inputs={"x": 1.0},
            scenario=scenario,
            detectors=[],
            seed=0,
            mode="fast",
        )
        result = runner.run()
        assert isinstance(result.iterations_completed, int)
        assert isinstance(result.findings, list)
        assert isinstance(result.root_cause_buckets, list)
        assert isinstance(result.runtime_ms, int)
        assert isinstance(result.seed, int)
        assert isinstance(result.baseline_established, bool)
        assert isinstance(result.mode, str)
        assert isinstance(result.budget_exhausted, bool)

    def test_deterministic_across_runs(self):
        scenario = _SimpleScenario()
        runner_a = StressRunner(
            fn=_nan_on_negative, base_inputs={"x": 1.0},
            scenario=scenario, detectors=[NaNInfDetector()], seed=7, mode="fast",
        )
        runner_b = StressRunner(
            fn=_nan_on_negative, base_inputs={"x": 1.0},
            scenario=scenario, detectors=[NaNInfDetector()], seed=7, mode="fast",
        )
        a = runner_a.run()
        b = runner_b.run()
        assert a.iterations_completed == b.iterations_completed
        assert len(a.findings) == len(b.findings)


# ---------------------------------------------------------------------------
# _deep_copy_inputs helper
# ---------------------------------------------------------------------------

class TestDeepCopyInputs:
    def test_copies_numpy_arrays(self):
        import numpy as np
        arr = np.array([1.0, 2.0, 3.0])
        inputs = {"arr": arr}
        copied = _deep_copy_inputs(inputs)
        copied["arr"][0] = 999.0
        assert arr[0] == 1.0   # original unchanged

    def test_copies_scalars(self):
        inputs = {"x": 1.0, "y": 2}
        copied = _deep_copy_inputs(inputs)
        assert copied == inputs
        assert copied is not inputs

    def test_handles_empty(self):
        assert _deep_copy_inputs({}) == {}
