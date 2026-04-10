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
from .engine.runner import RunResult, StressRunner
from .parser.auto_tagger import AutoTagger
from .parser.graph import DependencyGraphBuilder
from .scenarios.registry import list_scenarios, load_scenario, Scenario


# ---------------------------------------------------------------------------
# Static fix hints indexed by failure_type
# ---------------------------------------------------------------------------

_FIX_HINTS: dict[str, str] = {
    "non_psd_matrix": (
        "Apply nearest-PSD correction (e.g. Higham 2002) after correlation "
        "perturbation, or clamp eigenvalues to a small positive epsilon before "
        "further use: eigvals, eigvecs = np.linalg.eigh(m); "
        "m_psd = eigvecs @ np.diag(np.maximum(eigvals, 1e-8)) @ eigvecs.T"
    ),
    "ill_conditioned_matrix": (
        "Check the condition number before inversion with np.linalg.cond(). "
        "Use np.linalg.lstsq or the pseudo-inverse (np.linalg.pinv) instead of "
        "direct inversion for ill-conditioned systems (cond > 1e12)."
    ),
    "division_by_zero": (
        "Guard the denominator against near-zero values before dividing: "
        "use np.where(np.abs(denom) > 1e-10, num / denom, 0.0) or assert "
        "np.all(np.abs(denom) > 1e-10) at the function boundary."
    ),
    "nan_inf": (
        "Add input validation at the function boundary and check intermediate "
        "results for NaN/Inf with np.isfinite(). Use bounded perturbation ranges "
        "in the scenario YAML to prevent extreme inputs."
    ),
    "bounds_exceeded": (
        "Review the output bounds configured in the scenario YAML or add a "
        "clamp after computation: np.clip(result, lower, upper). Verify that "
        "input perturbations remain within plausible financial ranges."
    ),
}


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
    test_p.add_argument(
        "--adversarial",
        action="store_true",
        default=False,
        help=(
            "Use the Evolutionary (Genetic Algorithm) stress runner instead of "
            "random Monte Carlo sampling. Finds shatter points faster by evolving "
            "perturbation sets toward maximum failure severity. "
            "--iterations controls the number of GA generations (default: 20)."
        ),
    )
    test_p.add_argument(
        "--population",
        type=int,
        default=100,
        help="Population size per generation when using --adversarial (default: 100).",
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

    # Select detector suite automatically from AST analysis of the target file.
    detectors = AutoTagger(file_path).detector_suite()
    if getattr(args, "adversarial", False):
        from .engine.adversarial import EvolutionaryStressRunner
        n_generations = iterations  # --iterations is repurposed as generation count
        population_size = getattr(args, "population", 100)
        runner = EvolutionaryStressRunner(
            fn=fn,
            base_inputs=base_inputs,
            scenario=_IterationOverride(scenario, n_generations),
            detectors=detectors,
            seed=seed,
            n_generations=n_generations,
            population_size=population_size,
            elite_fraction=0.2,
        )
    else:
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
    print(json.dumps(_build_response(resolved, result, scenario, seed, file_path), indent=2))

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
    scenario: Scenario,
    seed: int,
    file_path: Path,
) -> dict:
    """
    Aggregate per-iteration findings into a contract-compliant response dict.

    Findings with the same failure_type are merged into one shatter_point;
    the line from the first (earliest) such finding is used, and the frequency
    string reports how many iterations produced that failure type.

    Also populates:
      causal_chain     — walked backwards from the failure line via the
                         dependency graph (DependencyGraphBuilder).
      fix_hint         — static lookup from _FIX_HINTS by failure_type.
      parameters_applied — the perturbation spec from the scenario definition,
                           describing every target, type, distribution, and range
                           that was active during the run.
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
            "causal_chain": _build_causal_chain(first.line, file_path),
            "fix_hint": _FIX_HINTS.get(failure_type, ""),
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
            "name": scenario.name,
            "parameters_applied": _parameters_applied(scenario),
            "seed": seed,
            "reproducible": True,
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_causal_chain(line: int | None, file_path: Path) -> list[dict]:
    """
    Build a causal chain for a failure at *line* using the dependency graph.

    Walks the DAG backwards from the failure node to collect root inputs and
    intermediates, then appends the failure node itself as 'failure_site'.
    Returns an empty list if *line* is None or graph construction fails for
    any reason (graceful degradation — the rest of the response is still valid).

    The chain is ordered by source line number (ascending): root inputs appear
    first, the failure site last.  Unresolved (import / builtin) nodes are
    excluded since they don't map to user source lines.
    """
    if line is None:
        return []
    try:
        graph = DependencyGraphBuilder(file_path).build()
    except Exception:
        return []

    node_map: dict[str, Any] = {n.id: n for n in graph.nodes}

    # Reverse-adjacency: node_id → set of predecessor ids (edges flow src→tgt)
    preds: dict[str, set[str]] = {n.id: set() for n in graph.nodes}
    for edge in graph.edges:
        preds[edge.target].add(edge.source)

    # Find the failure node: the node whose lineno == line.
    # If multiple nodes share the same line, prefer one with detector tags.
    failure_node = None
    for node in graph.nodes:
        if node.lineno == line:
            if failure_node is None or (
                node.detector_tags and not failure_node.detector_tags
            ):
                failure_node = node

    if failure_node is None:
        return []

    # BFS backwards from the failure node's direct predecessors.
    ancestor_ids: set[str] = set()
    queue: list[str] = list(preds.get(failure_node.id, set()))
    while queue:
        current = queue.pop(0)
        if current in ancestor_ids:
            continue
        ancestor_ids.add(current)
        for pred_id in preds.get(current, set()):
            if pred_id not in ancestor_ids:
                queue.append(pred_id)

    # Collect ancestor nodes, excluding unresolved (no lineno) nodes.
    # Cap at 20 entries to keep the chain readable.
    chain_nodes = sorted(
        [
            node_map[nid]
            for nid in ancestor_ids
            if nid in node_map
            and node_map[nid].lineno is not None
            and node_map[nid].node_type != "unresolved_input"
        ],
        key=lambda n: n.lineno or 0,
    )[:20]

    result: list[dict] = []
    for node in chain_nodes:
        result.append({
            "line": node.lineno,
            "variable": node.label,
            "role": node.node_type,
        })

    # Failure site is always last.
    result.append({
        "line": failure_node.lineno,
        "variable": failure_node.label,
        "role": "failure_site",
    })

    return result


def _parameters_applied(scenario: Scenario) -> dict:
    """
    Summarise the perturbation specification from *scenario* as a plain dict.

    Keyed by perturbation target name; value describes the type, distribution,
    and numeric range so the scenario_card is self-documenting.
    """
    applied: dict[str, Any] = {}
    for p in scenario.perturbations:
        entry: dict[str, Any] = {
            "type": p.type,
            "distribution": p.distribution,
        }
        # Include distribution parameters verbatim so the card is reproducible.
        entry.update(p.params)
        if p.clamp is not None:
            lo, hi = p.clamp
            if lo is not None:
                entry["clip_min"] = lo
            if hi is not None:
                entry["clip_max"] = hi
        applied[p.target] = entry
    return applied


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

    def __getattr__(self, name: str) -> Any:
        # Forward any attribute not explicitly defined here to the wrapped scenario.
        return getattr(self._scenario, name)


def _error_exit(message: str, code: int) -> None:
    """Emit a JSON error object to stderr and terminate with the given code."""
    print(json.dumps({"status": "error", "message": message}), file=sys.stderr)
    sys.exit(code)
