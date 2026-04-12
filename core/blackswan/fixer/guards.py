"""
BlackSwan mathematical guard insertion.

Given Python source, a 1-indexed line number, and a failure type, this module
inserts a minimal deterministic fix using libcst to locate and validate the
target expression while preserving all original whitespace and indentation.

Supported failure types and their fix patterns:
  division_instability / division_by_zero  — sign-preserving epsilon guard on denominator
  non_psd_matrix                           — eigenvalue clamping (Higham nearest-PSD)
  ill_conditioned_matrix                   — conditional pinv fallback on inversion
  nan_inf                                  — np.nan_to_num guard after assignment

Exit codes (when invoked via CLI):
  0  Fix found — JSON FixResult written to stdout
  1  Fix not applicable at this location — JSON error written to stdout
  2  Parse error or libcst not installed
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class FixResult:
    """Result returned as JSON to the TypeScript fixer.ts module."""
    status: str                           # "ok" | "error" | "unsupported"
    line: Optional[int] = None            # 1-indexed target line (echo back)
    original: Optional[str] = None        # original line text (no newline)
    replacement: Optional[str] = None     # in-place replacement (no newline); equals original for insert-only fixes
    extra_lines: Optional[list[str]] = None   # lines to INSERT AFTER `line` (indented already)
    explanation: Optional[str] = None     # human-readable explanation for the user
    message: Optional[str] = None         # error message when status != "ok"

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in asdict(self).items() if v is not None}, indent=2)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_fix(source: str, line: int, failure_type: str) -> FixResult:
    """
    Apply a deterministic mathematical guard at `line` (1-indexed) for `failure_type`.

    Returns a FixResult. Never raises — errors are returned as status="error".
    """
    try:
        import libcst as cst  # noqa: F401  (checked here for a clean error)
    except ImportError:
        return FixResult(
            status="error",
            message=(
                "libcst is not installed. Run: pip install 'libcst>=1.1' "
                "to enable deterministic mathematical guards."
            ),
        )

    if failure_type in ("division_instability", "division_by_zero"):
        return _fix_division(source, line)
    if failure_type == "non_psd_matrix":
        return _fix_psd(source, line)
    if failure_type == "ill_conditioned_matrix":
        return _fix_condition_number(source, line)
    if failure_type == "nan_inf":
        return _fix_nan_inf(source, line)

    return FixResult(
        status="unsupported",
        message=(
            f"No deterministic guard pattern is available for failure type "
            f"'{failure_type}'. You can still apply the fix hint manually."
        ),
    )


# ---------------------------------------------------------------------------
# Fix: division_instability / division_by_zero
# ---------------------------------------------------------------------------

def _fix_division(source: str, line: int) -> FixResult:
    """
    Wrap the denominator of a division at `line` with a sign-preserving
    epsilon guard:
        x / denom  →  x / (denom if abs(denom) > 1e-10 else (1e-10 if denom >= 0 else -1e-10))
    """
    import libcst as cst
    from libcst import metadata

    lines = source.splitlines()
    if line < 1 or line > len(lines):
        return FixResult(status="error", message=f"Line {line} is out of range (file has {len(lines)} lines)")

    try:
        module = cst.parse_module(source)
    except Exception as exc:
        return FixResult(status="error", message=f"Parse error: {exc}")

    wrapper = metadata.MetadataWrapper(module)
    finder = _DivisionFinder(module, line)
    finder.visit(wrapper)

    if not finder.found:
        return FixResult(
            status="error",
            message=f"No division operation found at line {line}. "
                    f"The failure may span multiple lines or use a function call (e.g. np.true_divide).",
        )

    denom = finder.denom_code
    # Sign-preserving guard: keeps model sign semantics intact.
    # Uses an inline conditional rather than abs() to avoid changing the magnitude.
    new_denom = (
        f"({denom} if abs({denom}) > 1e-10 "
        f"else (1e-10 if ({denom}) >= 0 else -1e-10))"
    )

    original_line_text = lines[line - 1]
    # Replace "/ denom" — try with a space first, then without
    fixed_line_text = _replace_after_slash(original_line_text, denom, new_denom)
    if fixed_line_text is None:
        return FixResult(
            status="error",
            message=f"Could not locate denominator '{denom}' after '/' on line {line}.",
        )

    return FixResult(
        status="ok",
        line=line,
        original=original_line_text,
        replacement=fixed_line_text,
        explanation=(
            f"Wrapped denominator `{denom}` with a sign-preserving epsilon guard (ε = 1e-10). "
            f"The guard prevents division by zero while preserving the algebraic sign of the "
            f"original denominator. "
            f"Note: if your model requires a different precision floor, "
            f"adjust the 1e-10 constant."
        ),
    )


class _DivisionFinder(object):
    """
    libcst visitor that finds the denominator of the first division
    operation on `target_line` and records its source text.
    """

    def __init__(self, module: object, target_line: int) -> None:
        import libcst as cst
        from libcst import metadata

        self._module = module
        self._target_line = target_line
        self.denom_code: Optional[str] = None
        self.found = False

        # We implement the visitor as an inner class so we can close over
        # `self` (the _DivisionFinder) for state mutation.
        outer = self

        class _Visitor(cst.CSTVisitor):
            METADATA_DEPENDENCIES = (metadata.PositionProvider,)

            def visit_BinaryOperation(self, node: cst.BinaryOperation) -> None:
                if outer.found:
                    return
                if not isinstance(node.operator, (cst.Divide, cst.FloorDivide)):
                    return
                pos = self.get_metadata(metadata.PositionProvider, node)
                if pos.start.line != outer._target_line:
                    return
                outer.denom_code = outer._module.code_for_node(node.right).strip()
                outer.found = True

        self._visitor_class = _Visitor

    def visit(self, wrapper: object) -> None:
        wrapper.visit(self._visitor_class())  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Fix: non_psd_matrix  (eigenvalue clamping)
# ---------------------------------------------------------------------------

def _fix_psd(source: str, line: int) -> FixResult:
    """
    Insert eigenvalue clamping after the matrix assignment at `line`:
        _bs_w, _bs_v = np.linalg.eigh(<var>)
        <var> = _bs_v @ np.diag(np.maximum(_bs_w, 0.0)) @ _bs_v.T
    """
    import libcst as cst
    from libcst import metadata

    lines = source.splitlines()
    if line < 1 or line > len(lines):
        return FixResult(status="error", message=f"Line {line} is out of range")

    try:
        module = cst.parse_module(source)
    except Exception as exc:
        return FixResult(status="error", message=f"Parse error: {exc}")

    wrapper = metadata.MetadataWrapper(module)
    finder = _AssignmentFinder(module, line)
    finder.visit(wrapper)

    original_line_text = lines[line - 1]
    indent_str = _leading_indent(original_line_text)

    var = finder.lhs_name or _guess_lhs(original_line_text)
    if var is None:
        return FixResult(
            status="error",
            message=f"Cannot determine the matrix variable name at line {line}.",
        )

    extra = [
        f"{indent_str}_bs_w, _bs_v = np.linalg.eigh({var})",
        f"{indent_str}{var} = _bs_v @ np.diag(np.maximum(_bs_w, 0.0)) @ _bs_v.T",
    ]

    return FixResult(
        status="ok",
        line=line,
        original=original_line_text,
        replacement=original_line_text,   # original line is not changed
        extra_lines=extra,
        explanation=(
            f"Added eigenvalue clamping after `{var}` is constructed. "
            f"Uses `np.linalg.eigh` (symmetric eigendecomposition) to factor the matrix, "
            f"clamps all negative eigenvalues to zero with `np.maximum`, then reconstructs. "
            f"This implements the nearest-PSD projection following Higham (2002). "
            f"Requires NumPy — already imported in BlackSwan-supported code."
        ),
    )


# ---------------------------------------------------------------------------
# Fix: ill_conditioned_matrix  (conditional pinv)
# ---------------------------------------------------------------------------

def _fix_condition_number(source: str, line: int) -> FixResult:
    """
    Replace np.linalg.inv(M) with a conditional that falls back to
    np.linalg.pinv(M) when the condition number exceeds 1e12.
    """
    import libcst as cst
    from libcst import metadata

    lines = source.splitlines()
    if line < 1 or line > len(lines):
        return FixResult(status="error", message=f"Line {line} is out of range")

    try:
        module = cst.parse_module(source)
    except Exception as exc:
        return FixResult(status="error", message=f"Parse error: {exc}")

    wrapper = metadata.MetadataWrapper(module)
    finder = _InvCallFinder(module, line)
    finder.visit(wrapper)

    if not finder.found:
        return FixResult(
            status="error",
            message=f"No np.linalg.inv call found at line {line}.",
        )

    original_line_text = lines[line - 1]
    m = finder.matrix_arg_code
    old_call = f"np.linalg.inv({m})"
    new_call = (
        f"(np.linalg.inv({m}) if np.linalg.cond({m}) < 1e12 "
        f"else np.linalg.pinv({m}))"
    )

    fixed_line_text = original_line_text.replace(old_call, new_call, 1)
    if fixed_line_text == original_line_text:
        return FixResult(
            status="error",
            message=(
                f"Could not rewrite the inversion call — the exact text "
                f"'np.linalg.inv({m})' was not found on line {line}. "
                f"The matrix may be aliased differently."
            ),
        )

    return FixResult(
        status="ok",
        line=line,
        original=original_line_text,
        replacement=fixed_line_text,
        explanation=(
            f"Replaced `np.linalg.inv({m})` with a conditional: "
            f"if the condition number is < 1e12, use exact inversion; "
            f"otherwise fall back to `np.linalg.pinv` (Moore-Penrose pseudoinverse). "
            f"This prevents numerical explosion when `{m}` is near-singular. "
            f"Note: the pinv fallback computes a minimum-norm solution, which may "
            f"differ from the exact inverse in well-posed cases — verify your model semantics."
        ),
    )


class _InvCallFinder(object):
    """Finds the first np.linalg.inv(M) call at `target_line` and records M's source text."""

    def __init__(self, module: object, target_line: int) -> None:
        import libcst as cst
        from libcst import metadata

        self._module = module
        self._target_line = target_line
        self.found = False
        self.matrix_arg_code: Optional[str] = None
        outer = self

        class _Visitor(cst.CSTVisitor):
            METADATA_DEPENDENCIES = (metadata.PositionProvider,)

            def visit_Call(self, node: cst.Call) -> None:
                if outer.found:
                    return
                pos = self.get_metadata(metadata.PositionProvider, node)
                if pos.start.line != outer._target_line:
                    return
                func_code = outer._module.code_for_node(node.func).strip()
                if "linalg.inv" not in func_code:
                    return
                if not node.args:
                    return
                outer.matrix_arg_code = outer._module.code_for_node(node.args[0].value).strip()
                outer.found = True

        self._visitor_class = _Visitor

    def visit(self, wrapper: object) -> None:
        wrapper.visit(self._visitor_class())  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Fix: nan_inf  (np.nan_to_num guard)
# ---------------------------------------------------------------------------

def _fix_nan_inf(source: str, line: int) -> FixResult:
    """
    Insert a np.nan_to_num guard after the assignment at `line`:
        <var> = np.nan_to_num(<var>, nan=0.0, posinf=np.finfo(float).max, neginf=np.finfo(float).min)
    """
    import libcst as cst
    from libcst import metadata

    lines = source.splitlines()
    if line < 1 or line > len(lines):
        return FixResult(status="error", message=f"Line {line} is out of range")

    try:
        module = cst.parse_module(source)
    except Exception as exc:
        return FixResult(status="error", message=f"Parse error: {exc}")

    wrapper = metadata.MetadataWrapper(module)
    finder = _AssignmentFinder(module, line)
    finder.visit(wrapper)

    original_line_text = lines[line - 1]
    indent_str = _leading_indent(original_line_text)

    var = finder.lhs_name or _guess_lhs(original_line_text)
    if var is None:
        return FixResult(
            status="error",
            message=f"Cannot determine the variable name at line {line}.",
        )

    extra = [
        f"{indent_str}{var} = np.nan_to_num("
        f"{var}, nan=0.0, "
        f"posinf=np.finfo(float).max, "
        f"neginf=np.finfo(float).min)",
    ]

    return FixResult(
        status="ok",
        line=line,
        original=original_line_text,
        replacement=original_line_text,
        extra_lines=extra,
        explanation=(
            f"Added `np.nan_to_num` guard after `{var}` is computed. "
            f"Replaces NaN → 0.0 and ±Inf → float64 max/min. "
            f"Review the `nan=0.0` default: for portfolio returns, 0.0 may be correct; "
            f"for weights or prices, a different sentinel (e.g. the last valid value) "
            f"may be more appropriate for your model."
        ),
    )


# ---------------------------------------------------------------------------
# Shared visitors and helpers
# ---------------------------------------------------------------------------

class _AssignmentFinder(object):
    """
    libcst visitor that records the LHS variable name of the first
    Assign or AnnAssign statement at `target_line`.
    """

    def __init__(self, module: object, target_line: int) -> None:
        import libcst as cst
        from libcst import metadata

        self._target_line = target_line
        self.lhs_name: Optional[str] = None
        self.found = False
        outer = self

        class _Visitor(cst.CSTVisitor):
            METADATA_DEPENDENCIES = (metadata.PositionProvider,)

            def visit_Assign(self, node: cst.Assign) -> None:
                if outer.found:
                    return
                pos = self.get_metadata(metadata.PositionProvider, node)
                if pos.start.line != outer._target_line:
                    return
                if node.targets and isinstance(node.targets[0].target, cst.Name):
                    outer.lhs_name = node.targets[0].target.value
                    outer.found = True

            def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
                if outer.found:
                    return
                pos = self.get_metadata(metadata.PositionProvider, node)
                if pos.start.line != outer._target_line:
                    return
                if isinstance(node.target, cst.Name):
                    outer.lhs_name = node.target.value
                    outer.found = True

        self._visitor_class = _Visitor

    def visit(self, wrapper: object) -> None:
        wrapper.visit(self._visitor_class())  # type: ignore[attr-defined]


def _replace_after_slash(line: str, denom: str, new_denom: str) -> Optional[str]:
    """Replace the first occurrence of `/ denom` or `/denom` in `line`."""
    for pattern in (f"/ {denom}", f"/{denom}"):
        idx = line.find(pattern)
        if idx != -1:
            return line[:idx] + f"/ {new_denom}" + line[idx + len(pattern):]
    return None


def _leading_indent(line: str) -> str:
    """Return the leading whitespace of `line`."""
    stripped = line.lstrip()
    return line[: len(line) - len(stripped)]


def _guess_lhs(line: str) -> Optional[str]:
    """
    Best-effort: extract the LHS variable name from a simple assignment line.
    Handles `var = ...` and `var: type = ...`.
    """
    m = re.match(r"\s*([a-zA-Z_]\w*)\s*(?::[^=]*)?\s*=", line)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# CLI entry point (used by `python -m blackswan fix ...`)
# ---------------------------------------------------------------------------

def cli_main(args: list[str]) -> int:
    """
    Called from cli.py when the user runs `python -m blackswan fix`.

    Returns the exit code:
      0 — fix found, JSON written to stdout
      1 — fix not applicable, JSON error written to stdout
      2 — usage error
    """
    import argparse

    p = argparse.ArgumentParser(prog="blackswan fix")
    p.add_argument("file", help="Python file to fix")
    p.add_argument("--line", type=int, required=True, metavar="N",
                   help="1-indexed line number where the failure was detected")
    p.add_argument("--type", dest="failure_type", required=True,
                   help="BlackSwan failure type (e.g. division_instability)")

    parsed = p.parse_args(args)

    from pathlib import Path
    path = Path(parsed.file)
    if not path.exists():
        result = FixResult(status="error", message=f"File not found: {parsed.file}")
        print(result.to_json())
        return 1

    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        result = FixResult(status="error", message=f"Cannot read file: {exc}")
        print(result.to_json())
        return 1

    result = apply_fix(source, parsed.line, parsed.failure_type)
    print(result.to_json())
    return 0 if result.status == "ok" else 1
