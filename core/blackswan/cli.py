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

from .engine.repro import build_reproducibility_card
from .engine.runner import RunResult, StressRunner
from .parser.auto_tagger import AutoTagger
from .parser.graph import DependencyGraphBuilder
from .scenarios.registry import list_scenarios, load_scenario, Scenario


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
        "--mode",
        choices=["fast", "full"],
        default="full",
        help=(
            "Execution mode. "
            "'full' (default): Fast-Path sweep + Slow-Path attribution replay. "
            "'fast': Fast-Path only — no attribution tracing. Faster, less detail."
        ),
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
    test_p.add_argument(
        "--max-runtime-sec",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Hard time limit in seconds. The run stops after this many seconds "
            "even if not all iterations completed. The response includes "
            "budget_exhausted=true and iterations_skipped."
        ),
    )
    test_p.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Hard iteration cap. Stops after N iterations regardless of the "
            "scenario's default. Takes precedence over --iterations when both are set."
        ),
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

    # Budget overrides (--max-iterations takes precedence over --iterations).
    max_iterations = getattr(args, "max_iterations", None)
    max_runtime_sec = getattr(args, "max_runtime_sec", None)
    # When --max-iterations is set, it acts as the iteration cap.
    effective_iterations = max_iterations if max_iterations is not None else iterations

    # Execution mode.
    mode = getattr(args, "mode", "full") or "full"

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
        effective_mode = "adversarial"
    else:
        runner = StressRunner(
            fn=fn,
            base_inputs=base_inputs,
            scenario=_IterationOverride(scenario, effective_iterations),
            detectors=detectors,
            seed=seed,
            mode=mode,
            max_runtime_sec=max_runtime_sec,
            max_iterations=max_iterations,
        )
        effective_mode = mode

    result = runner.run()

    # Emit the full JSON response to stdout.
    print(json.dumps(
        _build_response(
            result=result,
            scenario=scenario,
            seed=seed,
            file_path=file_path,
            mode=effective_mode,
            iterations_requested=effective_iterations,
        ),
        indent=2,
    ))

    has_failures = any(b.total_occurrences > 0 for b in result.root_cause_buckets)
    sys.exit(1 if has_failures else 0)


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
    result: RunResult,
    scenario: Any,
    seed: int,
    file_path: Path,
    mode: str,
    iterations_requested: int,
) -> dict:
    """
    Build the contract-compliant JSON response from a completed RunResult.

    Uses root_cause_buckets as the primary source of truth (already clustered,
    shrunk, and attributed by the engine). Falls back to AST-based causal chain
    construction when Slow-Path attribution is unavailable (fast mode /
    adversarial mode).

    Includes:
      reproducibility_card  — exact provenance for replay.
      confidence            — per shatter_point confidence level.
      trigger_disclosure    — detector threshold that caused each finding.
      budget                — budget exhaustion status.
    """
    buckets = result.root_cause_buckets
    total_occurrences = sum(b.total_occurrences for b in buckets)
    rate = total_occurrences / result.iterations_completed if result.iterations_completed > 0 else 0.0

    shatter_points = []
    for i, bucket in enumerate(buckets, start=1):
        pct = bucket.occurrence_rate * 100

        # Causal chain: Slow-Path attribution first; AST graph walk as fallback.
        causal_chain: list[dict] = []
        if bucket.causal_chain:
            causal_chain = [
                {"line": link.line, "variable": link.variable, "role": link.role}
                for link in bucket.causal_chain
            ]
        elif bucket.line is not None:
            causal_chain = _build_causal_chain(bucket.line, file_path)

        # Trigger disclosure (populated by concrete detectors).
        trigger_disclosure = None
        td = bucket.representative_finding.trigger_disclosure
        if td is not None:
            trigger_disclosure = {
                "detector_name": td.detector_name,
                "observed_value": td.observed_value,
                "threshold": td.threshold,
                "comparison": td.comparison,
                "explanation": td.explanation,
            }

        sp: dict[str, Any] = {
            "id": f"sp_{i:03d}",
            "line": bucket.line,
            "column": None,
            "severity": bucket.severity,
            "failure_type": bucket.failure_type,
            "message": bucket.message,
            "frequency": (
                f"{bucket.total_occurrences} / {result.iterations_completed} "
                f"iterations ({pct:.1f}%)"
            ),
            "causal_chain": causal_chain,
            "fix_hint": bucket.fix_hint,
            "confidence": bucket.confidence,
        }
        if trigger_disclosure is not None:
            sp["trigger_disclosure"] = trigger_disclosure

        shatter_points.append(sp)

    repro_card = build_reproducibility_card(
        scenario=scenario,
        seed=seed,
        mode=mode,
        iterations_requested=iterations_requested,
        iterations_executed=result.iterations_completed,
        budget_exhausted=result.budget_exhausted,
        budget_reason=result.budget_reason,
        file_path=file_path,
    )

    return {
        "version": "1.0",
        "status": "failures_detected" if total_occurrences > 0 else "no_failures",
        "mode": mode,
        "runtime_ms": result.runtime_ms,
        "iterations_completed": result.iterations_completed,
        "summary": {
            "total_failures": total_occurrences,
            "failure_rate": round(rate, 4),
            "unique_failure_types": len(buckets),
        },
        "shatter_points": shatter_points,
        "scenario_card": {
            "name": scenario.name,
            "parameters_applied": _parameters_applied(scenario),
            "seed": seed,
            "reproducible": True,
        },
        "reproducibility_card": repro_card.to_dict(),
        "budget": {
            "exhausted": result.budget_exhausted,
            "reason": result.budget_reason,
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


def _parameters_applied(scenario: Any) -> dict:
    """
    Summarise the perturbation specification from *scenario* as a plain dict.

    Keyed by perturbation target name; value describes the type, distribution,
    and numeric range so the scenario_card is self-documenting.
    """
    applied: dict[str, Any] = {}
    for p in getattr(scenario, "perturbations", []):
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
