"""
Simulation runner for BlackSwan stress tests.

Two-Path execution architecture:

  Fast-Path: run all N iterations at full NumPy speed with detectors only.
             No sys.settrace, no NumPy hooks. Captures which inputs caused
             failures and stores them for Slow-Path replay.

  Slow-Path: for the first occurrence of each unique failure type, re-run
             the exact failing inputs under TracerBackend + NumPyHookSet.
             Produces runtime attribution (source line, locals, causal chain).
             Skipped in mode="fast".

After the loop, findings are:
  1. Shrunk    — each failing input is reduced to its minimal reproducing form.
  2. Clustered — grouped into RootCauseBuckets by (failure_type, line, exc_type).

RunResult carries the clusters, not raw findings, as the primary output.
Raw findings are preserved for compatibility with existing callers.
"""

from __future__ import annotations

import copy
import os
import sys
import time
import traceback as tb
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..detectors.base import FailureDetector, Finding
from .cluster import RootCauseBucket, cluster_findings
from .hooks import NumPyHookCapture, NumPyHookSet
from .replay import Attribution, ReplayDivergenceError, SlowPathReplayer
from .shrink import ShrinkingEngine
from .tracer import make_tracer


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """
    Output of a single StressRunner.run() call.

    Primary output is root_cause_buckets — deduplicated, shrunk, attributed
    failures sorted by occurrence rate.

    findings is retained for backward compatibility with existing tests and
    the CLI serialisation layer. It contains all raw Fast-Path findings before
    clustering.

    iterations_completed counts only iterations where the target function
    was actually called (validator-skipped iterations are excluded).

    baseline_established is False when fn(**base_inputs) raised before the
    loop began — findings should be interpreted with caution.
    """
    iterations_completed: int
    findings: list[Finding]
    root_cause_buckets: list[RootCauseBucket]
    runtime_ms: int
    seed: int
    baseline_established: bool = True
    mode: str = "full"               # "fast" | "full" | "adversarial"
    budget_exhausted: bool = False
    budget_reason: str | None = None


# ---------------------------------------------------------------------------
# StressRunner
# ---------------------------------------------------------------------------

class StressRunner:
    """
    Runs a target function N times under perturbed inputs and collects failures.

    Execution is deterministic: same fn, base_inputs, scenario, detectors, and
    seed → identical RunResult every time.

    Modes:
        fast        — Fast-Path only (no attribution). CI-grade, <2 s.
        full        — Fast-Path sweep + Slow-Path replay for attribution.
        (adversarial is handled by EvolutionaryStressRunner)
    """

    def __init__(
        self,
        fn: Callable,
        base_inputs: dict,
        scenario: Any,
        detectors: list[FailureDetector],
        seed: int,
        mode: str = "full",
        max_runtime_sec: float | None = None,
        max_iterations: int | None = None,
    ) -> None:
        self.fn = fn
        self.base_inputs = base_inputs
        self.scenario = scenario
        self.detectors = detectors
        self.seed = seed
        self.mode = mode
        self.max_runtime_sec = max_runtime_sec
        self.max_iterations = max_iterations

        # Derive target filename for tracer — try to get it from fn's source file
        self._target_filename = _resolve_target_filename(fn)

    def run(self) -> RunResult:
        rng = np.random.default_rng(self.seed)
        start = time.monotonic()

        # ── Baseline pass ────────────────────────────────────────────────
        baseline_established = True
        try:
            base_output = self.fn(**self.base_inputs)
        except Exception:
            base_output = None
            baseline_established = False

        for detector in self.detectors:
            if hasattr(detector, "reset"):
                detector.reset()
            if base_output is not None and hasattr(detector, "set_baseline"):
                detector.set_baseline(self.base_inputs, base_output)

        validator = (
            self.scenario.make_validator()
            if hasattr(self.scenario, "make_validator")
            else None
        )

        # ── Fast-Path loop ───────────────────────────────────────────────
        all_findings: list[Finding] = []
        # Maps iteration index → deep-copied perturbed inputs (for Slow-Path + shrinker)
        failing_inputs: dict[int, dict[str, Any]] = {}
        executed = 0
        budget_exhausted = False
        budget_reason: str | None = None

        n_iterations = (
            self.max_iterations
            if self.max_iterations is not None
            else self.scenario.iterations
        )

        for i in range(n_iterations):
            # Budget: time
            if self.max_runtime_sec is not None:
                elapsed = time.monotonic() - start
                if elapsed >= self.max_runtime_sec:
                    budget_exhausted = True
                    budget_reason = "max_runtime_sec"
                    break

            perturbed = self.scenario.apply(self.base_inputs, rng)

            if validator is not None and not validator.validate(perturbed):
                continue

            # Deep-copy inputs before function call so we capture pre-call state
            inputs_copy = _deep_copy_inputs(perturbed)

            try:
                output = self.fn(**perturbed)
            except Exception as exc:
                frames = [
                    (fr.filename, fr.lineno)
                    for fr in tb.extract_tb(exc.__traceback__)
                ]
                finding = Finding(
                    failure_type="nan_inf",
                    severity="critical",
                    message=f"{type(exc).__name__}: {exc}",
                    iteration=i,
                    exc_frames=frames,
                )
                all_findings.append(finding)
                failing_inputs[i] = inputs_copy
                executed += 1
                continue

            executed += 1
            for detector in self.detectors:
                finding = detector.check(inputs=perturbed, output=output, iteration=i)
                if finding is not None:
                    all_findings.append(finding)
                    failing_inputs.setdefault(i, inputs_copy)

        runtime_ms = int((time.monotonic() - start) * 1000)

        # ── Slow-Path replay (skipped in fast mode) ──────────────────────
        attributions: dict[int, Attribution] = {}
        if self.mode != "fast" and self._target_filename and all_findings:
            attributions = self._run_slow_path(all_findings, failing_inputs)

        # ── Shrink failing inputs ────────────────────────────────────────
        if self.mode != "fast" and all_findings:
            failing_inputs = self._shrink_inputs(all_findings, failing_inputs)

        # ── Cluster findings ─────────────────────────────────────────────
        buckets = cluster_findings(
            findings=all_findings,
            total_iterations=executed,
            attributions=attributions,
            failing_inputs=failing_inputs,
        )

        return RunResult(
            iterations_completed=executed,
            findings=all_findings,
            root_cause_buckets=buckets,
            runtime_ms=runtime_ms,
            seed=self.seed,
            baseline_established=baseline_established,
            mode=self.mode,
            budget_exhausted=budget_exhausted,
            budget_reason=budget_reason,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Private: Slow-Path
    # ──────────────────────────────────────────────────────────────────────

    def _run_slow_path(
        self,
        findings: list[Finding],
        failing_inputs: dict[int, dict[str, Any]],
    ) -> dict[int, Attribution]:
        """
        Replay up to 3 occurrences per unique failure type under full tracing.
        Returns dict mapping iteration → Attribution.
        """
        tracer = make_tracer(self._target_filename)
        replayer = SlowPathReplayer(
            tracer=tracer,
            detectors=self.detectors,
            target_filename=self._target_filename,
        )
        attributions: dict[int, Attribution] = {}
        # Track how many replays we've done per (failure_type, line) key
        replay_counts: dict[tuple, int] = {}
        MAX_REPLAYS_PER_BUCKET = 3

        for finding in findings:
            if finding.iteration not in failing_inputs:
                continue
            key = (finding.failure_type, finding.line)
            if replay_counts.get(key, 0) >= MAX_REPLAYS_PER_BUCKET:
                continue
            try:
                attr = replayer.replay(
                    fn=self.fn,
                    inputs=failing_inputs[finding.iteration],
                    fast_path_finding=finding,
                )
                attributions[finding.iteration] = attr
                replay_counts[key] = replay_counts.get(key, 0) + 1
            except ReplayDivergenceError as e:
                # Non-determinism in user code or engine bug — exclude this finding
                # Log to stderr so the user knows something was dropped
                print(
                    f"[BlackSwan] ReplayDivergenceError at iteration {finding.iteration}: {e}",
                    file=sys.stderr,
                )
            except Exception:
                pass  # unexpected error in tracer — skip gracefully

        return attributions

    # ──────────────────────────────────────────────────────────────────────
    # Private: Shrinking
    # ──────────────────────────────────────────────────────────────────────

    def _shrink_inputs(
        self,
        findings: list[Finding],
        failing_inputs: dict[int, dict[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        """
        For each unique (failure_type, line) bucket, shrink the representative
        failing input to its minimal form.
        """
        validator = (
            self.scenario.make_validator()
            if hasattr(self.scenario, "make_validator")
            else None
        )
        shrinker = ShrinkingEngine(
            fn=self.fn,
            base_inputs=self.base_inputs,
            detectors=self.detectors,
            validator=validator,
            max_steps=300,
            seed=self.seed,
        )

        shrunken = dict(failing_inputs)
        shrunk_buckets: set[tuple] = set()

        for finding in findings:
            if finding.iteration not in failing_inputs:
                continue
            key = (finding.failure_type, finding.line)
            if key in shrunk_buckets:
                continue
            try:
                minimal = shrinker.shrink(failing_inputs[finding.iteration], finding)
                shrunken[finding.iteration] = minimal
                shrunk_buckets.add(key)
            except Exception:
                pass  # shrinking is best-effort

        return shrunken


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_target_filename(fn: Callable) -> str | None:
    """Get the absolute path of the source file where fn is defined."""
    try:
        import inspect
        path = inspect.getfile(fn)
        return os.path.abspath(path)
    except (TypeError, OSError):
        return None


def _deep_copy_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Copy an inputs dict. Uses np.copy for arrays, copy.copy for scalars.
    Never raises — falls back to reference on copy failure.
    """
    result = {}
    for k, v in inputs.items():
        try:
            if isinstance(v, np.ndarray):
                result[k] = np.copy(v)
            else:
                result[k] = copy.copy(v)
        except Exception:
            result[k] = v  # reference fallback
    return result
