"""
NumPy hook layer for BlackSwan Slow-Path replay.

C-extension calls (np.linalg.inv, np.cov, etc.) are opaque to sys.settrace and
sys.monitoring. During Slow-Path replay, NumPyHookSet temporarily replaces these
functions with thin Python wrappers that capture inputs and outputs.

Only active during SlowPathReplayer.replay() — never during the Fast-Path sweep.

Captured data is stored in NumPyHookCapture and used by SlowPathReplayer to
attribute failures inside NumPy C functions to the correct source lines.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from functools import wraps
from typing import Any


# ---------------------------------------------------------------------------
# Capture types
# ---------------------------------------------------------------------------

@dataclass
class NumPyCall:
    """Record of one NumPy function call captured during Slow-Path replay."""
    fn_name: str
    args_snapshot: tuple           # deep copies of positional args
    kwargs_snapshot: dict          # deep copies of keyword args
    output_snapshot: Any           # copy of return value; None if call raised
    raised: Exception | None       # populated if the call raised


class NumPyHookCapture:
    """Accumulates NumPyCall records for one Slow-Path replay session."""

    def __init__(self) -> None:
        self.calls: list[NumPyCall] = []

    def clear(self) -> None:
        self.calls.clear()


# ---------------------------------------------------------------------------
# Hook targets
# ---------------------------------------------------------------------------

# (module_attr_path, function_name)
# These are the NumPy functions most relevant to financial/numerical fragility.
_HOOK_TARGETS: list[tuple[str, str]] = [
    ("numpy.linalg", "inv"),
    ("numpy.linalg", "solve"),
    ("numpy.linalg", "eig"),
    ("numpy.linalg", "eigvals"),
    ("numpy.linalg", "eigh"),
    ("numpy.linalg", "eigvalsh"),
    ("numpy.linalg", "cond"),
    ("numpy.linalg", "matrix_rank"),
    ("numpy",        "cov"),
    ("numpy",        "corrcoef"),
    ("numpy",        "dot"),
    ("numpy",        "matmul"),
    ("numpy",        "inner"),
    ("numpy",        "outer"),
]


def _safe_copy(value: Any) -> Any:
    """Best-effort non-raising copy. Returns a sentinel on failure."""
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return np.copy(value)
    except ImportError:
        pass
    try:
        return copy.copy(value)
    except Exception:
        return f"<CopyFailed:{type(value).__name__}>"


# ---------------------------------------------------------------------------
# NumPyHookSet
# ---------------------------------------------------------------------------

class NumPyHookSet:
    """
    Installs thin wrapper functions over NumPy C-extensions during Slow-Path replay.

    Usage:
        capture = NumPyHookCapture()
        hooks = NumPyHookSet(capture)
        hooks.activate()
        try:
            fn(**inputs)
        finally:
            hooks.deactivate()
        # capture.calls contains all NumPy calls made during fn
    """

    def __init__(self, capture: NumPyHookCapture) -> None:
        self._capture = capture
        self._original: dict[str, Any] = {}
        self._active = False

    def activate(self) -> None:
        if self._active:
            return
        import importlib
        for module_path, fn_name in _HOOK_TARGETS:
            try:
                module = sys.modules.get(module_path)
                if module is None:
                    module = importlib.import_module(module_path)
                orig = getattr(module, fn_name, None)
                if orig is None:
                    continue
                key = f"{module_path}.{fn_name}"
                self._original[key] = orig
                setattr(module, fn_name, self._make_wrapper(fn_name, orig))
            except Exception:
                pass  # skip targets that don't exist in this NumPy version
        self._active = True

    def deactivate(self) -> None:
        if not self._active:
            return
        import importlib
        for module_path, fn_name in _HOOK_TARGETS:
            key = f"{module_path}.{fn_name}"
            if key not in self._original:
                continue
            try:
                module = sys.modules.get(module_path)
                if module is None:
                    module = importlib.import_module(module_path)
                setattr(module, fn_name, self._original[key])
            except Exception:
                pass
        self._original.clear()
        self._active = False

    def _make_wrapper(self, name: str, orig_fn: Any):
        capture = self._capture

        @wraps(orig_fn)
        def wrapper(*args, **kwargs):
            args_snap = tuple(_safe_copy(a) for a in args)
            kwargs_snap = {k: _safe_copy(v) for k, v in kwargs.items()}
            exc_captured: Exception | None = None
            result = None
            try:
                result = orig_fn(*args, **kwargs)
                return result
            except Exception as e:
                exc_captured = e
                raise
            finally:
                capture.calls.append(NumPyCall(
                    fn_name=name,
                    args_snapshot=args_snap,
                    kwargs_snapshot=kwargs_snap,
                    output_snapshot=_safe_copy(result),
                    raised=exc_captured,
                ))

        return wrapper
