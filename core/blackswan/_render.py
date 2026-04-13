"""
_render.py — Rich terminal renderer for BlackSwan CLI output.

Converts the JSON response dict into a human-readable terminal report.
Uses ANSI escape codes directly — no external dependencies.

Auto-disables color when:
  - stdout is not a TTY
  - NO_COLOR environment variable is set
  - TERM=dumb
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import textwrap
from typing import Any

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

def _use_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return sys.stdout.isatty()


_COLOR = None  # resolved lazily on first use


def _c() -> bool:
    global _COLOR
    if _COLOR is None:
        _COLOR = _use_color()
    return _COLOR


# ANSI codes — only applied when _c() is True
def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if _c() else s

def _dim(s: str) -> str:
    return f"\033[2m{s}\033[0m" if _c() else s

def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m" if _c() else s

def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m" if _c() else s

def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m" if _c() else s

def _blue(s: str) -> str:
    return f"\033[34m{s}\033[0m" if _c() else s

def _cyan(s: str) -> str:
    return f"\033[36m{s}\033[0m" if _c() else s

def _bold_red(s: str) -> str:
    return f"\033[1;31m{s}\033[0m" if _c() else s

def _bold_green(s: str) -> str:
    return f"\033[1;32m{s}\033[0m" if _c() else s

def _bold_yellow(s: str) -> str:
    return f"\033[1;33m{s}\033[0m" if _c() else s

def _bold_white(s: str) -> str:
    return f"\033[1;37m{s}\033[0m" if _c() else s


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

FAILURE_TYPE_LABELS: dict[str, str] = {
    "nan_inf":                "NaN / Inf Detected",
    "division_by_zero":       "Division Instability",
    "non_psd_matrix":         "Non-PSD Matrix",
    "ill_conditioned_matrix": "Ill-Conditioned Matrix",
    "bounds_exceeded":        "Bounds Exceeded",
    "division_instability":   "Division Instability",
    "exploding_gradient":     "Exploding Gradient",
    "regime_shift":           "Regime Shift",
    "logical_invariant":      "Logical Invariant Violation",
}

ROLE_LABELS: dict[str, str] = {
    "root_input":   "root input",
    "intermediate": "intermediate",
    "failure_site": "FAILURE SITE",
}

SEVERITY_COLOR = {
    "critical": _bold_red,
    "warning":  _bold_yellow,
    "info":     _blue,
}


def _width() -> int:
    return min(shutil.get_terminal_size(fallback=(100, 24)).columns, 100)


def _rule(char: str = "─") -> str:
    return _dim(char * _width())


def _hrule(label: str = "", char: str = "─") -> str:
    """A horizontal rule with an optional left-aligned label."""
    w = _width()
    if label:
        gap = max(0, w - len(label) - 2)
        return _dim(f"{char}{char} ") + label + _dim(f" {char * gap}")
    return _dim(char * w)


def _bar(rate: float, width: int = 28) -> str:
    """Unicode block progress bar. rate is 0.0–1.0."""
    filled = round(rate * width)
    empty = width - filled
    if _c():
        return "\033[32m" + "█" * filled + "\033[2;37m" + "░" * empty + "\033[0m"
    return "#" * filled + "-" * empty


def _wrap(text: str, indent: int = 2, subsequent_indent: int | None = None) -> str:
    """Wrap text to terminal width with given indent."""
    w = _width() - indent
    si = " " * (subsequent_indent if subsequent_indent is not None else indent)
    prefix = " " * indent
    return textwrap.fill(text, width=w, initial_indent=prefix, subsequent_indent=si)


def _two_col(label: str, value: str, label_width: int = 12) -> str:
    """Print a two-column key: value line, wrapping value if needed."""
    indent = label_width + 2
    prefix = f"  {label:<{label_width}}"
    w = _width() - indent
    lines = textwrap.wrap(value, width=w)
    if not lines:
        return f"{prefix}"
    result = f"{prefix}{lines[0]}"
    for line in lines[1:]:
        result += f"\n{' ' * indent}{line}"
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render(response: dict[str, Any]) -> None:
    """
    Print a human-readable report of a BlackSwan engine response to stdout.

    Called only when stdout is a TTY (cli.py checks sys.stdout.isatty()).
    The caller is responsible for that guard.
    """
    # Ensure UTF-8 output on Windows where the default console encoding
    # may be cp1252 and cannot encode box-drawing / arrow characters.
    if sys.platform == "win32":
        try:
            import io
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
        except AttributeError:
            pass  # already wrapped or no buffer attribute

    out: list[str] = []

    _header(response, out)
    _body(response, out)
    _footer(response, out)

    print("\n".join(out))


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _header(resp: dict, out: list[str]) -> None:
    """One-line summary: tool · scenario · iterations · seed · runtime."""
    rc = resp.get("reproducibility_card") or {}
    version = rc.get("blackswan_version", "")
    scenario = resp.get("scenario_card", {}).get("name", "?")
    seed = resp.get("scenario_card", {}).get("seed", "?")
    iters = f"{resp.get('iterations_completed', 0):,}"
    runtime_ms = resp.get("runtime_ms", 0)
    runtime_str = f"{runtime_ms / 1000:.2f}s" if runtime_ms >= 10 else f"{runtime_ms}ms"
    mode = resp.get("mode", "full")

    parts = [f"BlackSwan {version}" if version else "BlackSwan"]
    parts.append(scenario)
    parts.append(f"{iters} iterations")
    parts.append(f"seed {seed}")
    parts.append(mode)
    title = _dim("  ·  ").join([_bold_white(parts[0])] + [_dim(p) for p in parts[1:]])

    # Runtime right-aligned
    w = _width()
    clean_title = "  ".join(parts)  # approx length without ANSI
    pad = max(1, w - len(clean_title) - len(runtime_str) - 2)
    out.append("")
    out.append(f"  {title}{' ' * pad}{_dim(runtime_str)}")
    out.append("")


def _body(resp: dict, out: list[str]) -> None:
    status = resp.get("status", "")
    shatter_points = resp.get("shatter_points", [])

    if status == "no_failures" or not shatter_points:
        out.append(_bold_green("  ✓  NO FAILURES DETECTED"))
        out.append("")
        out.append(_dim(_wrap(
            "All 5 detectors ran across all iterations — no mathematical "
            "fragility was found under this scenario and seed.",
            indent=5,
        )))
        out.append("")
        return

    # Count by severity
    n = len(shatter_points)
    label = f"FAILURE{'S' if n > 1 else ''} DETECTED  ({n} shatter point{'s' if n > 1 else ''})"
    out.append(_hrule(_bold_red(f"  {label}  ")))
    out.append("")

    for i, sp in enumerate(shatter_points, start=1):
        _shatter_point(sp, i, n, resp, out)


def _shatter_point(sp: dict, index: int, total: int, resp: dict, out: list[str]) -> None:
    severity = sp.get("severity", "critical")
    failure_type = sp.get("failure_type", "")
    type_label = FAILURE_TYPE_LABELS.get(failure_type, failure_type)
    line_num = sp.get("line")
    message = sp.get("message", "")
    frequency = sp.get("frequency", "")
    confidence = sp.get("confidence", "")
    causal_chain = sp.get("causal_chain", [])
    fix_hint = sp.get("fix_hint", "")
    td = sp.get("trigger_disclosure") or {}

    # Severity badge + failure type label + line number
    sev_fn = SEVERITY_COLOR.get(severity, _dim)
    sev_badge = sev_fn(f" {severity.upper()} ")
    line_str = _dim(f"line {line_num}") if line_num is not None else ""

    w = _width()
    label_block = f"  {sev_badge}  {_bold_white(type_label)}"
    clean_len = 2 + len(severity) + 4 + len(type_label)
    pad = max(1, w - clean_len - len(f"line {line_num or '?'}") - 2)
    out.append(f"{label_block}{' ' * pad}{line_str}")
    out.append("")

    # Message — wrapped
    out.append(_wrap(message, indent=2))
    out.append("")

    # Frequency bar
    rate = _parse_rate(frequency)
    bar = _bar(rate)
    pct = f"{rate * 100:.1f}%"
    count_str = frequency.split(" iterations")[0].strip() if "iterations" in frequency else frequency
    out.append(f"  {_dim('Frequency')}   {bar}  {_bold_white(pct)}  {_dim(count_str)}")

    if confidence:
        conf_color = _green if confidence == "high" else (_yellow if confidence == "medium" else _dim)
        out.append(f"  {_dim('Confidence')}  {conf_color(confidence)}")

    out.append("")

    # Causal chain
    if causal_chain:
        out.append(f"  {_dim('Causal Chain')} {_dim('─' * 24)}")
        for link in causal_chain:
            role = link.get("role", "")
            var = link.get("variable", "")
            lineno = link.get("line", "?")
            role_label = ROLE_LABELS.get(role, role)

            if role == "root_input":
                role_str = _blue(f"  {role_label:<14}")
                arrow = _dim("→")
            elif role == "failure_site":
                role_str = _bold_red(f"  {role_label:<14}")
                arrow = _red("►")
            else:
                role_str = _dim(f"  {role_label:<14}")
                arrow = _dim("·")

            out.append(f"    {arrow} {_dim(f'line {lineno:<5}')}{_dim(var):<28} {role_str}")

        out.append("")

    # Trigger disclosure
    if td:
        detector = td.get("detector_name", "")
        explanation = td.get("explanation", "")
        if detector:
            out.append(_two_col(_dim("Detector"), detector))
        if explanation:
            out.append(_two_col(_dim("Trigger"), _dim(explanation)))
        out.append("")

    # Fix hint
    if fix_hint:
        out.append(_two_col(_bold_yellow("Fix Hint"), fix_hint))
        out.append("")

    # Quick-fix command (only for guardable failure types)
    _GUARDABLE = {"division_instability", "division_by_zero", "non_psd_matrix",
                  "ill_conditioned_matrix", "nan_inf"}
    rc = resp.get("reproducibility_card") or {}
    # Extract file path from replay command (simplest approach)
    replay = rc.get("replay_command", "")
    file_arg = _extract_file_arg(replay)

    if failure_type in _GUARDABLE and line_num is not None and file_arg:
        fix_cmd = f"python -m blackswan fix {file_arg} --line {line_num} --type {failure_type}"
        out.append(f"  {_dim('Quick Fix')}   {_green('$')} {_cyan(fix_cmd)}")
        out.append("")

    out.append(_rule())
    out.append("")


def _footer(resp: dict, out: list[str]) -> None:
    """Summary line + replay command."""
    summary = resp.get("summary", {})
    total = summary.get("total_failures", 0)
    rate = summary.get("failure_rate", 0.0)
    unique = summary.get("unique_failure_types", 0)
    iters = resp.get("iterations_completed", 0)
    runtime_ms = resp.get("runtime_ms", 0)
    runtime_str = f"{runtime_ms / 1000:.2f}s"
    status = resp.get("status", "")

    if status == "no_failures":
        out.append(
            f"  {_bold_green('✓')}  {_dim(f'{iters:,} iterations completed')}  "
            f"{_dim('·')}  {_dim(runtime_str)}"
        )
    else:
        type_word = "types" if unique != 1 else "type"
        out.append(
            f"  {_bold_red(f'{total:,} failures')}  {_dim('·')}"
            f"  {_dim(f'{rate * 100:.1f}% rate')}  {_dim('·')}"
            f"  {_dim(f'{unique} unique {type_word}')}  {_dim('·')}"
            f"  {_dim(runtime_str)}"
        )

    rc = resp.get("reproducibility_card") or {}
    replay = rc.get("replay_command", "")
    budget = resp.get("budget", {})
    if budget.get("exhausted"):
        reason = budget.get("reason") or "budget limit reached"
        out.append(f"  {_bold_yellow('⚠')}  {_yellow(f'Budget: {reason}')}")

    if replay:
        out.append("")
        out.append(_two_col(_dim("Replay"), _dim(replay), label_width=8))

    out.append("")
    out.append(_dim("  " + "─" * (_width() - 2)))
    json_hint_cmd = replay.replace(" --mode full", "").replace(" --mode fast", "")
    # Build a concise hint — strip the full absolute path down to just the flags
    out.append(
        f"  {_dim('Full JSON')}  "
        f"{_dim('Add')} {_cyan('--format json')} {_dim('to the command above for the complete machine-readable output.')}"
    )
    out.append("")


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _parse_rate(frequency_str: str) -> float:
    """
    Extract the failure rate as a float 0.0–1.0 from a frequency string.
    e.g. '2981 / 5000 iterations (59.6%)' → 0.596
    """
    import re
    m = re.search(r"\((\d+\.?\d*)%\)", frequency_str)
    if m:
        return min(float(m.group(1)) / 100.0, 1.0)
    # Fallback: try N / M
    m2 = re.search(r"(\d+)\s*/\s*(\d+)", frequency_str)
    if m2:
        n, d = int(m2.group(1)), int(m2.group(2))
        return n / d if d else 0.0
    return 0.0


def _extract_file_arg(replay_command: str) -> str | None:
    """
    Pull the file path argument from a replay command string.
    e.g. 'python -m blackswan test /path/to/model.py --scenario ...'
    → 'model.py' (basename for the fix command display)
    """
    import re
    m = re.search(r"blackswan test\s+(\S+)", replay_command)
    if m:
        return os.path.basename(m.group(1))
    return None
