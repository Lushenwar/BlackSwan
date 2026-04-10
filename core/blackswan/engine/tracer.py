"""
TracerBackend — runtime instrumentation for BlackSwan Slow-Path replay.

Two implementations, selected at import time based on Python version:
  - MonitoringBackend  (Python 3.12+)  uses sys.monitoring (PEP 669, low overhead)
  - SetTraceBackend    (Python 3.11)   uses sys.settrace   (universal fallback)

Only active during Slow-Path replay of a single detected failure. Never active
during the Fast-Path sweep. Overhead is acceptable because replay is rare.

Both backends emit FrameEvent objects and expose local variable snapshots keyed
by source line number, which the SlowPathReplayer uses for attribution.
"""

from __future__ import annotations

import copy
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------

@dataclass
class FrameEvent:
    """One recorded execution event from inside the target function."""
    event: str                         # "call" | "line" | "return" | "exception"
    filename: str
    lineno: int
    locals_snapshot: dict[str, Any]    # shallow copy of frame locals at this event


def _safe_copy(value: Any) -> Any:
    """
    Best-effort copy of an arbitrary value.
    - numpy arrays: np.copy (fast, avoids deepcopy overhead on large arrays)
    - everything else: copy.copy (shallow)
    Falls back to a sentinel on failure rather than raising.
    """
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return np.copy(value)
    except ImportError:
        pass
    try:
        return copy.copy(value)
    except Exception:
        return _CopyFailed(type(value).__name__)


@dataclass
class _CopyFailed:
    """Sentinel used when _safe_copy cannot copy a value."""
    type_name: str

    def __repr__(self) -> str:
        return f"<CopyFailed: {self.type_name}>"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TracerBackend(ABC):
    """
    Abstraction over sys.settrace / sys.monitoring.

    Usage:
        tracer = make_tracer(target_filename="/abs/path/to/model.py")
        tracer.activate()
        try:
            fn(**inputs)
        finally:
            tracer.deactivate()
        events = tracer.get_frame_log()
        snapshots = tracer.get_local_snapshots()
    """

    @abstractmethod
    def activate(self) -> None:
        """Install the trace hook. Idempotent."""

    @abstractmethod
    def deactivate(self) -> None:
        """Remove the trace hook and stop recording. Idempotent."""

    @abstractmethod
    def get_frame_log(self) -> list[FrameEvent]:
        """Return all recorded events in chronological order."""

    @abstractmethod
    def get_local_snapshots(self) -> dict[int, dict[str, Any]]:
        """
        Return a dict mapping source line number → local variable snapshot
        captured the last time that line was executed.
        Useful for identifying which variable held a bad value on a given line.
        """

    def reset(self) -> None:
        """Clear recorded state so the backend can be reused for another replay."""


# ---------------------------------------------------------------------------
# SetTraceBackend — Python 3.11 compatible
# ---------------------------------------------------------------------------

class SetTraceBackend(TracerBackend):
    """
    Tracer using sys.settrace. Activated only during Slow-Path replay.

    Traces only frames whose co_filename matches target_filename to avoid
    tracing into stdlib, NumPy C wrappers, or other user modules.
    """

    def __init__(self, target_filename: str) -> None:
        self._target = target_filename
        self._log: list[FrameEvent] = []
        self._snapshots: dict[int, dict[str, Any]] = {}
        self._prev_trace = None
        self._active = False

    def activate(self) -> None:
        if self._active:
            return
        self._prev_trace = sys.gettrace()
        sys.settrace(self._trace_fn)
        self._active = True

    def deactivate(self) -> None:
        if not self._active:
            return
        sys.settrace(self._prev_trace)
        self._prev_trace = None
        self._active = False

    def get_frame_log(self) -> list[FrameEvent]:
        return list(self._log)

    def get_local_snapshots(self) -> dict[int, dict[str, Any]]:
        return dict(self._snapshots)

    def reset(self) -> None:
        self._log.clear()
        self._snapshots.clear()

    def _trace_fn(self, frame, event, arg):
        if frame.f_code.co_filename != self._target:
            # Only trace the target file; returning None stops tracing this frame's children
            return self._trace_fn if event == "call" else None

        snapshot: dict[str, Any] = {
            k: _safe_copy(v) for k, v in frame.f_locals.items()
        }
        lineno = frame.f_lineno

        self._log.append(FrameEvent(
            event=event,
            filename=frame.f_code.co_filename,
            lineno=lineno,
            locals_snapshot=snapshot,
        ))
        # Keep only the most recent snapshot per line
        self._snapshots[lineno] = snapshot

        return self._trace_fn


# ---------------------------------------------------------------------------
# MonitoringBackend — Python 3.12+ (PEP 669)
# ---------------------------------------------------------------------------

class MonitoringBackend(TracerBackend):
    """
    Tracer using sys.monitoring (PEP 669). Lower overhead than sys.settrace
    because event callbacks are not invoked for C-extension frames.

    Falls back to raising NotImplementedError on Python < 3.12 (should never
    happen — make_tracer() selects the correct backend).
    """

    # Use DEBUGGER_ID slot so we don't collide with coverage tools
    _TOOL_ID: int

    def __init__(self, target_filename: str) -> None:
        if sys.version_info < (3, 12):
            raise NotImplementedError("MonitoringBackend requires Python 3.12+")
        self._target = target_filename
        self._log: list[FrameEvent] = []
        self._snapshots: dict[int, dict[str, Any]] = {}
        self._active = False
        # Assign tool ID lazily
        self._TOOL_ID = sys.monitoring.DEBUGGER_ID  # type: ignore[attr-defined]

    def activate(self) -> None:
        if self._active:
            return
        m = sys.monitoring  # type: ignore[attr-defined]
        m.set_tool(self._TOOL_ID, "BlackSwan")
        m.set_events(
            self._TOOL_ID,
            m.events.LINE | m.events.EXCEPTION_RAISED,
        )
        m.register_callback(self._TOOL_ID, m.events.LINE, self._on_line)
        m.register_callback(self._TOOL_ID, m.events.EXCEPTION_RAISED, self._on_exception)
        self._active = True

    def deactivate(self) -> None:
        if not self._active:
            return
        m = sys.monitoring  # type: ignore[attr-defined]
        m.set_events(self._TOOL_ID, m.events.NO_EVENTS)
        m.set_tool(self._TOOL_ID, None)
        self._active = False

    def get_frame_log(self) -> list[FrameEvent]:
        return list(self._log)

    def get_local_snapshots(self) -> dict[int, dict[str, Any]]:
        return dict(self._snapshots)

    def reset(self) -> None:
        self._log.clear()
        self._snapshots.clear()

    def _on_line(self, code, line_number):
        # sys.monitoring does not pass the frame — we can't get locals without
        # a settrace-style hook. Record the event with an empty snapshot;
        # the SlowPathReplayer uses exc_frames for attribution, not locals here.
        import sys
        frame = sys._getframe(1)  # caller's frame
        if frame.f_code.co_filename != self._target:
            return
        snapshot: dict[str, Any] = {
            k: _safe_copy(v) for k, v in frame.f_locals.items()
        }
        self._log.append(FrameEvent("line", frame.f_code.co_filename, line_number, snapshot))
        self._snapshots[line_number] = snapshot

    def _on_exception(self, code, offset, exception):
        import sys
        frame = sys._getframe(1)
        if frame.f_code.co_filename != self._target:
            return
        snapshot: dict[str, Any] = {
            k: _safe_copy(v) for k, v in frame.f_locals.items()
        }
        self._log.append(FrameEvent("exception", frame.f_code.co_filename, frame.f_lineno, snapshot))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_tracer(target_filename: str) -> TracerBackend:
    """
    Return the best available TracerBackend for the current Python version.
    MonitoringBackend on 3.12+, SetTraceBackend on 3.11.
    """
    if sys.version_info >= (3, 12):
        try:
            return MonitoringBackend(target_filename)
        except Exception:
            pass  # fall through to settrace
    return SetTraceBackend(target_filename)
