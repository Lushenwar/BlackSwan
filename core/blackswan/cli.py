"""
CLI entry point for BlackSwan.

Usage:
    python -m blackswan test <file> --scenario <name> [options]
    python -m blackswan --list-scenarios

Exit codes:
    0  All iterations completed with zero failures detected.
    1  One or more failure types were detected across the iterations.
    2  Error: bad file path, unknown scenario, parse/import failure, etc.
       Error details are printed as a JSON object to stderr — never a traceback.
"""

import argparse
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any

from .attribution.traceback import TracebackResolver
from .detectors.matrix import ConditionNumberDetector, MatrixPSDDetector
from .detectors.numerical import DivisionStabilityDetector, NaNInfDetector
from .detectors.portfolio import BoundsDetector
from .engine.runner import RunResult, StressRunner
from .scenarios.registry import list_scenarios, load_scenario


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_scenarios:
        _cmd_list_scenarios()
        return

    if args.command is None:
        parser.print_help(sys.stderr)
        sys.exit(2)

    _cmd_test(args)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blackswan",
        description="BlackSwan: stress-test mathematical and financial logic in Python code.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List all available preset scenarios with descriptions and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")

    test_p = subparsers.add_parser(
        "test",
        help="Stress-test a target function in a Python file.",
    )
    test_p.add_argument("file", help="Path to the Python file containing the target function.")
    test_p.add_argument(
        "--scenario", required=True,
        help="Preset scenario name (e.g. liquidity_crash). Use --list-scenarios to see all.",
    )
    test_p.add_argument(
        "--function", dest="function_name", default=None,
        help=(
            "Name of the function to stress-test. "
            "If omitted, the first public function in the file is used."
        ),
    )
    test_p.add_argument(
        "--iterations", type=int, default=None,
        help="Number of simulation iterations. Overrides the scenario's default.",
    )
    test_p.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for reproducibility. Overrides the scenario's default.",
    )

    return parser


# ---------------------------------------------------------------------------
# Subcommand: --list-scenarios
# ---------------------------------------------------------------------------

def _cmd_list_scenarios() -> None:
    """Print all available preset scenarios and their descriptions."""
    for name in list_scenarios():
        try:
            scenario = load_scenario(name)
            print(f"{name:<28} {scenario.description}")
        except Exception:
            print(name)


# ---------------------------------------------------------------------------
# Subcommand: test
# ---------------------------------------------------------------------------

def _cmd_test(args: argparse.Namespace) -> None:
    file_path = Path(args.file).resolve()

    # Validate file exists.
    if not file_path.exists():
        _error_exit(f"File not found: {args.file}", 2)

    # Load scenario from registry.
    try:
        scenario = load_scenario(args.scenario)
    except FileNotFoundError:
        _error_exit(
            f"Unknown scenario '{args.scenario}'. "
            "Run with --list-scenarios to see available presets.",
            2,
        )

    # Import the target module and locate the function.
    try:
        fn, base_inputs = _load_function(file_path, args.function_name)
    except SyntaxError as exc:
        _error_exit(f"Syntax error in {file_path.name}: {exc}", 2)
    except (ImportError, ValueError) as exc:
        _error_exit(str(exc), 2)

    # Apply CLI overrides for iterations and seed.
    iterations = args.iterations if args.iterations is not None else scenario.default_iterations
    seed = args.seed if args.seed is not None else scenario.default_seed

    # Run with all five detectors active.
    detectors = [
        NaNInfDetector(),
        DivisionStabilityDetector(),
        MatrixPSDDetector(),
        ConditionNumberDetector(),
        BoundsDetector(),
    ]
    runner = StressRunner(
        fn=fn,
        base_inputs=base_inputs,
        scenario=_IterationOverride(scenario, iterations),
        detectors=detectors,
        seed=seed,
    )
    result = runner.run()

    # Resolve line numbers for every finding.
    resolver = TracebackResolver(file_path)
    resolved = [resolver.resolve(f) for f in result.findings]

    # Emit the full JSON response to stdout.
    print(json.dumps(_build_response(resolved, result, scenario.name, seed), indent=2))

    sys.exit(1 if resolved else 0)


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

def _load_function(file_path: Path, function_name: str | None) -> tuple:
    """
    Import the target module and find the function to stress-test.

    Returns (fn, base_inputs) where base_inputs is a dict of parameter name
    → default value for every parameter that has a Python default.  Parameters
    without defaults are omitted; apply_perturbations() silently skips scenario
    targets not present in the inputs dict.

    Phase 1 note: functions with required parameters not covered by scenario
    perturbation targets will raise TypeError at call time, which the runner
    captures as a 'nan_inf' finding.  Phase 2 AST analysis will infer types
    for required parameters automatically.
    """
    # Add the file's directory to sys.path so relative imports in the target work.
    module_dir = str(file_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location("_blackswan_target", file_path)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except SyntaxError:
        raise
    except Exception as exc:
        raise ImportError(f"Failed to import {file_path.name}: {exc}") from exc

    # Locate the target function.
    if function_name is not None:
        fn = getattr(module, function_name, None)
        if fn is None or not callable(fn):
            raise ValueError(
                f"Function '{function_name}' not found in {file_path.name}. "
                "Check the spelling or omit --function to auto-detect."
            )
    else:
        # Auto-detect: collect non-private functions defined in this module
        # (not imported ones), in alphabetical order.
        candidates = [
            obj
            for name, obj in inspect.getmembers(module, inspect.isfunction)
            if not name.startswith("_") and obj.__module__ == module.__name__
        ]
        if not candidates:
            raise ValueError(f"No public functions found in {file_path.name}.")
        fn = candidates[0]

    # Build base inputs from parameter defaults.
    sig = inspect.signature(fn)
    base_inputs = {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }

    return fn, base_inputs


# ---------------------------------------------------------------------------
# JSON response builder
# ---------------------------------------------------------------------------

def _build_response(
    findings: list,
    result: RunResult,
    scenario_name: str,
    seed: int,
) -> dict:
    """
    Aggregate per-iteration findings into a contract-compliant response dict.

    Findings with the same failure_type are merged into one shatter_point;
    the line from the first (earliest) such finding is used, and the frequency
    string reports how many iterations produced that failure type.
    """
    from collections import defaultdict

    # Preserve first-seen insertion order while grouping by failure_type.
    groups: dict[str, list] = defaultdict(list)
    for f in findings:
        groups[f.failure_type].append(f)

    shatter_points = []
    for i, (failure_type, group) in enumerate(groups.items(), start=1):
        first = group[0]
        count = len(group)
        pct = (count / result.iterations_completed * 100) if result.iterations_completed > 0 else 0.0
        shatter_points.append({
            "id": f"sp_{i:03d}",
            "line": first.line,
            "column": first.column,
            "severity": first.severity,
            "failure_type": failure_type,
            "message": first.message,
            "frequency": f"{count} / {result.iterations_completed} iterations ({pct:.1f}%)",
            "causal_chain": [],   # Phase 2 will populate from AST dependency walk
            "fix_hint": "",       # Phase 2 will derive from failure type + causal chain
        })

    total = len(findings)
    rate = total / result.iterations_completed if result.iterations_completed > 0 else 0.0

    return {
        "version": "1.0",
        "status": "failures_detected" if total > 0 else "no_failures",
        "runtime_ms": result.runtime_ms,
        "iterations_completed": result.iterations_completed,
        "summary": {
            "total_failures": total,
            "failure_rate": round(rate, 4),
            "unique_failure_types": len(groups),
        },
        "shatter_points": shatter_points,
        "scenario_card": {
            "name": scenario_name,
            "parameters_applied": {},   # Phase 2 will record applied perturbation values
            "seed": seed,
            "reproducible": True,
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _IterationOverride:
    """
    Thin wrapper around a Scenario that substitutes a caller-supplied
    iteration count, allowing --iterations to override the YAML default
    without mutating the Scenario dataclass.
    """

    def __init__(self, scenario: Any, iterations: int) -> None:
        self._scenario = scenario
        self.iterations = iterations

    def apply(self, inputs: dict, rng: Any) -> dict:
        return self._scenario.apply(inputs, rng)


def _error_exit(message: str, code: int) -> None:
    """Emit a JSON error object to stderr and terminate with the given code."""
    print(json.dumps({"status": "error", "message": message}), file=sys.stderr)
    sys.exit(code)
