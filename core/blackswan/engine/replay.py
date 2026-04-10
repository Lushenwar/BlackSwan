"""
SlowPathReplayer — focused re-execution of a single failed iteration under
full runtime instrumentation.

When the Fast-Path detects a failure, SlowPathReplayer re-runs the exact same
inputs with TracerBackend and NumPyHookSet active. If the failure reproduces,
attribution data (source line, local variable snapshot, NumPy call chain) is
extracted and returned as an Attribution object.

If the failure does not reproduce, ReplayDivergenceError is raised. This is
treated as a non-determinism bug in the user's code or the engine — the
finding is excluded from the report rather than silently attributed incorrectly.
"""

from __future__ import annotations

import traceback as tb
from dataclasses import dataclass, field
from typing import Any, Callable

from ..detectors.base import FailureDetector, Finding
from .hooks import NumPyHookCapture, NumPyHookSet
from .tracer import FrameEvent, TracerBackend


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ReplayDivergenceError(Exception):
    """
    Raised when the Slow-Path replay does not reproduce the Fast-Path failure.

    Indicates non-determinism in the target function or an engine bug.
    The finding that triggered the replay must be excluded from the report.
    """


# ---------------------------------------------------------------------------
# Attribution output
# ---------------------------------------------------------------------------

@dataclass
class CausalLink:
    """One node in the causal chain traced from root input to failure site."""
    line: int
    variable: str
    role: str              # "root_input" | "intermediate" | "failure_site"
    value_repr: str        # repr() of the variable value at execution time


@dataclass
class TriggerDisclosure:
    """Exact threshold that caused a detector to fire."""
    detector_name: str
    observed_value: float | str
    threshold: float | str
    comparison: str        # ">" | "<" | "==" | "!="
    explanation: str


@dataclass
class Attribution:
    """
    Full runtime attribution for one reproduced failure.

    Produced by SlowPathReplayer.replay() and attached to the corresponding
    RootCauseBucket (see cluster.py).
    """
    failure_line: int | None          # source line where failure manifested
    failure_variable: str | None      # name of the variable that held the bad value
    locals_at_failure: dict[str, Any] # all locals at the failure line
    causal_chain: list[CausalLink]    # ordered root → failure_site
    numpy_calls: list[Any]            # NumPyCall records from hooks.py
    frame_log: list[FrameEvent]       # complete frame event log
    confidence: str                   # "high" | "medium" | "low"
    attribution_method: str           # "slow_path_trace" | "exc_frames_only"
    trigger_disclosure: TriggerDisclosure | None = None


# ---------------------------------------------------------------------------
# SlowPathReplayer
# ---------------------------------------------------------------------------

class SlowPathReplayer:
    """
    Re-executes a target function under full instrumentation to attribute
    a previously detected failure to a specific source line.

    Thread-safety: instances are not thread-safe. Create one per worker.
    """

    def __init__(
        self,
        tracer: TracerBackend,
        detectors: list[FailureDetector],
        target_filename: str,
    ) -> None:
        self._tracer = tracer
        self._detectors = detectors
        self._target_filename = target_filename

    def replay(
        self,
        fn: Callable,
        inputs: dict[str, Any],
        fast_path_finding: Finding,
    ) -> Attribution:
        """
        Re-run fn(**inputs) under TracerBackend + NumPyHookSet.

        Returns Attribution if the failure reproduces.
        Raises ReplayDivergenceError if it does not.
        """
        self._tracer.reset()
        capture = NumPyHookCapture()
        hooks = NumPyHookSet(capture)

        exc_type_expected = fast_path_finding.failure_type
        exc_class_expected = (
            self._exc_class_from_frames(fast_path_finding.exc_frames)
            if fast_path_finding.exc_frames
            else None
        )

        reproduced_findings: list[Finding] = []
        replay_exc: Exception | None = None

        self._tracer.activate()
        hooks.activate()
        try:
            output = fn(**inputs)
        except Exception as exc:
            replay_exc = exc
            output = None
        finally:
            hooks.deactivate()
            self._tracer.deactivate()

        # Run detectors on output (if function didn't raise)
        if replay_exc is None and output is not None:
            for detector in self._detectors:
                f = detector.check(inputs=inputs, output=output, iteration=-1)
                if f is not None:
                    reproduced_findings.append(f)

        # Verify reproduction
        if not self._reproduced(
            fast_path_finding, reproduced_findings, replay_exc, exc_class_expected
        ):
            raise ReplayDivergenceError(
                f"Slow-Path replay did not reproduce failure_type='{exc_type_expected}' "
                f"with inputs {_summarise_inputs(inputs)}"
            )

        # Extract attribution from tracer + hooks
        return self._build_attribution(
            fast_path_finding=fast_path_finding,
            frame_log=self._tracer.get_frame_log(),
            local_snapshots=self._tracer.get_local_snapshots(),
            numpy_calls=capture.calls,
            replay_exc=replay_exc,
            reproduced_findings=reproduced_findings,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reproduced(
        self,
        original: Finding,
        replay_findings: list[Finding],
        replay_exc: Exception | None,
        exc_class_expected: type | None,
    ) -> bool:
        """
        Return True if the replay produced the same class of failure as the
        original Fast-Path finding.
        """
        # Case 1: original was an exception — replay must also raise same type
        if original.exc_frames:
            if replay_exc is None:
                return False
            if exc_class_expected is not None and not isinstance(replay_exc, exc_class_expected):
                return False
            return True

        # Case 2: original was a detector finding — replay must produce same failure_type
        for f in replay_findings:
            if f.failure_type == original.failure_type:
                return True
        return False

    def _build_attribution(
        self,
        fast_path_finding: Finding,
        frame_log: list[FrameEvent],
        local_snapshots: dict[int, dict[str, Any]],
        numpy_calls: list[Any],
        replay_exc: Exception | None,
        reproduced_findings: list[Finding],
    ) -> Attribution:
        """
        Derive failure_line, locals_at_failure, and causal_chain from the
        frame log and local snapshots.
        """
        # --- Identify failure line ---
        failure_line: int | None = None
        failure_variable: str | None = None
        confidence = "medium"
        method = "slow_path_trace"

        if replay_exc is not None:
            # Exception path: last frame in target file is failure site
            target_frames = [
                e for e in frame_log
                if e.filename == self._target_filename
            ]
            if target_frames:
                last = target_frames[-1]
                failure_line = last.lineno
                locals_at = last.locals_snapshot
                confidence = "high"
            else:
                # Fell back to exc_frames from Fast-Path
                for fname, lineno in reversed(fast_path_finding.exc_frames):
                    if fname == self._target_filename:
                        failure_line = lineno
                        break
                locals_at = local_snapshots.get(failure_line or 0, {})
                confidence = "medium"
                method = "exc_frames_only"
        else:
            # Detector path: find last line before detector fired
            if reproduced_findings:
                # Use the line from the reproduced finding if available
                rf = reproduced_findings[0]
                if rf.line is not None:
                    failure_line = rf.line
            if failure_line is None and frame_log:
                target_frames = [e for e in frame_log if e.filename == self._target_filename]
                if target_frames:
                    failure_line = target_frames[-1].lineno
            locals_at = local_snapshots.get(failure_line or 0, {})
            confidence = "high" if failure_line is not None else "low"

        # --- Build causal chain from frame log ---
        causal_chain = _build_causal_chain(
            frame_log=frame_log,
            failure_line=failure_line,
            target_filename=self._target_filename,
        )

        # --- Identify failure variable (last assigned before failure line) ---
        if locals_at and failure_line is not None:
            failure_variable = _identify_failure_variable(locals_at, numpy_calls)

        return Attribution(
            failure_line=failure_line,
            failure_variable=failure_variable,
            locals_at_failure=locals_at if 'locals_at' in dir() else {},
            causal_chain=causal_chain,
            numpy_calls=numpy_calls,
            frame_log=frame_log,
            confidence=confidence,
            attribution_method=method,
        )

    def _exc_class_from_frames(self, frames: list[tuple[str, int]]) -> type | None:
        """Not inferable from frames alone — return None (type check skipped)."""
        return None


# ---------------------------------------------------------------------------
# Causal chain construction
# ---------------------------------------------------------------------------

def _build_causal_chain(
    frame_log: list[FrameEvent],
    failure_line: int | None,
    target_filename: str,
) -> list[CausalLink]:
    """
    Walk the frame log to build an ordered list of CausalLinks.

    Strategy:
    - Collect all "line" events in the target file in execution order.
    - De-duplicate by lineno (keep last snapshot for each line).
    - Tag first-seen lines as "root_input" if they are early in the function,
      "failure_site" for the failure line, and "intermediate" for the rest.
    - Return the chain ordered from root → failure_site.
    """
    if failure_line is None:
        return []

    seen: dict[int, FrameEvent] = {}
    ordered_lines: list[int] = []

    for event in frame_log:
        if event.filename != target_filename or event.event not in ("line", "call"):
            continue
        if event.lineno not in seen:
            ordered_lines.append(event.lineno)
        seen[event.lineno] = event

    if not ordered_lines:
        return []

    chain: list[CausalLink] = []
    first_line = ordered_lines[0]
    last_line = ordered_lines[-1]

    for lineno in ordered_lines:
        event = seen[lineno]
        # Role assignment heuristic
        if lineno == failure_line:
            role = "failure_site"
        elif lineno == first_line:
            role = "root_input"
        else:
            role = "intermediate"

        # Pick the most "interesting" variable at this line
        var_name, var_repr = _most_relevant_local(event.locals_snapshot)

        chain.append(CausalLink(
            line=lineno,
            variable=var_name,
            role=role,
            value_repr=var_repr,
        ))

    # Trim to at most 10 links: keep first 3 + last 7 to bound JSON size
    if len(chain) > 10:
        chain = chain[:3] + chain[-7:]

    return chain


def _most_relevant_local(snapshot: dict[str, Any]) -> tuple[str, str]:
    """
    Pick the most informative variable from a locals snapshot.
    Prefers numpy arrays (likely computed quantities) over scalars.
    """
    if not snapshot:
        return ("?", "?")
    try:
        import numpy as np
        for name, val in snapshot.items():
            if isinstance(val, np.ndarray):
                return (name, f"ndarray{val.shape} dtype={val.dtype}")
    except ImportError:
        pass
    first_name, first_val = next(iter(snapshot.items()))
    return (first_name, repr(first_val)[:80])


def _identify_failure_variable(
    locals_at: dict[str, Any],
    numpy_calls: list[Any],
) -> str | None:
    """
    Attempt to identify which variable holds the failing value.
    If a NumPy call raised during replay, that call's output is the suspect.
    Otherwise, look for NaN/Inf in the locals snapshot.
    """
    # Check numpy calls for a raised exception
    for call in reversed(numpy_calls):
        if call.raised is not None:
            return call.fn_name

    # Scan locals for NaN/Inf
    try:
        import numpy as np
        for name, val in locals_at.items():
            if isinstance(val, np.ndarray):
                if not np.all(np.isfinite(val)):
                    return name
            elif isinstance(val, float) and not (val == val):  # NaN check
                return name
    except ImportError:
        pass

    return None


def _summarise_inputs(inputs: dict[str, Any]) -> str:
    """One-line summary of inputs for error messages."""
    parts = []
    for k, v in list(inputs.items())[:4]:
        try:
            import numpy as np
            if isinstance(v, np.ndarray):
                parts.append(f"{k}=ndarray{v.shape}")
                continue
        except ImportError:
            pass
        parts.append(f"{k}={repr(v)[:20]}")
    return "{" + ", ".join(parts) + ("..." if len(inputs) > 4 else "") + "}"
