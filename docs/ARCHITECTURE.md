# BlackSwan — Architecture

## Overview

BlackSwan is two independent systems connected by a versioned JSON contract.

```
┌──────────────────────────────────┐        JSON over stdout       ┌──────────────────────────────────┐
│       blackswan-vscode           │ ─────────────────────────────►│       blackswan-core             │
│       (TypeScript)               │◄───────────────────────────── │       (Python)                   │
│                                  │        child_process           │                                  │
│  CodeLens  Diagnostics  Hover    │                                │  Detectors  Runner  Attribution  │
│  Progress  DAG Webview  Bridge   │                                │  AST Parser  Scenarios  CLI      │
└──────────────────────────────────┘                                └──────────────────────────────────┘
```

**Critical principle:** The engine is the product. The extension is a renderer. `python -m blackswan test model.py --scenario liquidity_crash` must produce a complete, actionable JSON report with no VS Code present.

---

## Python Engine (`blackswan-core`)

### Directory Structure

```
core/blackswan/
├── __init__.py
├── __main__.py
├── cli.py                    # argparse CLI, entry point (test + fix subcommands)
│
├── fixer/
│   ├── __init__.py
│   └── guards.py             # apply_fix() — libcst CST rewriter for 4 guard patterns
│
├── engine/
│   ├── runner.py             # StressRunner — Monte Carlo loop
│   ├── adversarial.py        # EvolutionaryStressRunner + HardnessAdaptor + GA primitives
│   ├── perturbation.py       # apply_perturbations() — scenario YAML → perturbed inputs
│   └── validator.py          # PlausibilityValidator + PlausibilityConstraint
│
├── detectors/
│   ├── base.py               # FailureDetector ABC, Finding dataclass
│   ├── numerical.py          # NaNInf, DivisionStability, ExplodingGradient, RegimeShift, LogicalInvariant
│   ├── matrix.py             # MatrixPSD, ConditionNumber
│   ├── portfolio.py          # BoundsDetector
│   └── sensitivity.py        # Root cause sensitivity analysis
│
├── parser/
│   ├── ast_analyzer.py       # ast.parse walk — extracts functions, assignments, calls, divisions
│   ├── variable_tracker.py   # Maps variable names → definition lines
│   └── graph.py              # DAG: nodes = variables, edges = data flow; JSON serialization
│
├── scenarios/
│   ├── registry.py           # Scenario, PlausibilityConstraint dataclasses; load_scenario()
│   └── presets/
│       ├── liquidity_crash.yaml
│       ├── vol_spike.yaml
│       ├── correlation_breakdown.yaml
│       ├── rate_shock.yaml
│       └── missing_data.yaml
│
└── attribution/
    ├── traceback.py           # Proximate failure location from Python traceback or AST lookup
    └── causal_chain.py        # Walk DAG backwards from failure variable → root input ranking
```

---

### Engine: StressRunner

`StressRunner` runs the standard Monte Carlo stress test.

**Execution flow:**

```
1. Establish baseline — call fn(**base_inputs) once
   • Calls reset() and set_baseline(output) on each stateful detector (hasattr guard)
   • Failure here sets RunResult.baseline_established = False

2. For each iteration i in range(scenario.iterations):
   a. Apply perturbations → perturbed_inputs
   b. PlausibilityValidator.validate(perturbed_inputs) → skip if invalid
   c. Call fn(**perturbed_inputs)
   d. Pass (perturbed_inputs, output, i) to each detector
   e. Collect Finding objects
   executed += 1   ← only increments here (skipped iterations excluded)

3. Return RunResult(findings, iterations_completed=executed, baseline_established)
```

Key invariant: `RunResult.iterations_completed` counts only executed calls, never validator-skipped ones. Same seed + scenario always produces identical results.

---

### Engine: EvolutionaryStressRunner

Drop-in replacement for `StressRunner`. Uses a genetic algorithm to actively search for worst-case parameter combinations instead of sampling randomly.

**GA loop:**

```
1. Extract param_ranges from scenario perturbation definitions
2. Initialise population of size population_size with random Individuals (params dict + fitness float)
3. For each generation g in range(n_generations):
   a. Evaluate each Individual:
      - Apply individual.params as perturbation overrides via _apply_individual()
      - Run fn, collect detector findings
      - compute_fitness(findings) → severity-weighted score
   b. Sort population by fitness descending
   c. Retain top elite_fraction as elites
   d. Fill remainder via crossover(elite_a, elite_b) + mutate(child)
   e. HardnessAdaptor.update(best_fitness) → increase range_scale if stalling
4. Return RunResult(all_findings, iterations_completed = n_generations * population_size)
```

**Fitness scoring:**

```python
_SEVERITY_WEIGHT = {"critical": 3.0, "warning": 1.5, "info": 0.5}
_TYPE_BONUS      = {"non_psd_matrix": 1.5, "ill_conditioned_matrix": 1.2}

fitness = sum(weight * bonus for each finding)
```

**HardnessAdaptor:** Increments `hardness` by `step=0.1` whenever a generation produces zero fitness. `range_scale = 1.0 + hardness` (max 2.0) is applied to expand perturbation ranges.

---

### Detectors

All detectors inherit from `FailureDetector` (ABC in `detectors/base.py`):

```python
class FailureDetector(ABC):
    @abstractmethod
    def check(self, inputs: dict, output: Any, iteration: int) -> Finding | None: ...
```

`Finding` is a frozen dataclass:

```python
@dataclass(frozen=True)
class Finding:
    failure_type: str
    severity: str          # "critical" | "warning" | "info"
    message: str
    iteration: int
    line: int
    column: int
    exc_frames: tuple
```

**Stateful detectors** (`RegimeShiftDetector`, `ExplodingGradientDetector`) implement:
- `reset()` — clear internal state between runs
- `set_baseline(output)` — record baseline output for comparison

**Detector auto-tagging:** `parser/ast_analyzer.py` walks the AST and tags lines:
- `np.linalg.inv` / `np.linalg.solve` → `ConditionNumberDetector`
- `ast.BinOp` with `Div` / `FloorDiv` → `DivisionStabilityDetector`
- `np.cov`, `np.corrcoef`, or explicit matrix construction → `MatrixPSDDetector`
- Return statements → `BoundsDetector`

---

### Perturbation System

`engine/perturbation.py` applies scenario-defined perturbations to a copy of `base_inputs`.

Supported perturbation types:

| type | distribution | params |
|---|---|---|
| `additive` | `uniform` | `low`, `high` |
| `additive` | `lognormal` | `mu`, `sigma` |
| `multiplicative` | `uniform` | `low`, `high` |
| `multiplicative` | `lognormal` | `mu`, `sigma` |

Each perturbation entry may define `constraints.min` / `constraints.max` to clamp individual parameter values after perturbation.

`PlausibilityValidator` applies `global_constraints` (cross-parameter bounds) and rejects the entire input set before calling the function, avoiding physically impossible scenarios.

---

### Attribution

**Proximate failure (`traceback.py`):** Python's `sys.exc_info()` traceback frames are captured when a detector fires due to an exception. For failures detected post-call (e.g., eigenvalue check after `cov_matrix` is built), the AST graph locates the relevant assignment line.

**Causal chain (`causal_chain.py`):** Walks the DAG backwards from the failure variable to all root inputs (parameters with no upstream dependencies). A quick sensitivity re-run (perturbing one input at a time) ranks inputs by their marginal contribution to failure frequency.

---

## VS Code Extension (`blackswan-vscode`)

### Directory Structure

```
extension/src/
├── extension.ts        # Activation, command registration, wires all components
├── bridge.ts           # Spawns Python child_process, stdin/stdout JSON protocol
├── diagnostics.ts      # DiagnosticsController — shatter_points → vscode.Diagnostic[] + 3 CodeActions
├── codelens.ts         # BlackSwanCodeLensProvider + scanForTargets() pure function
├── hover.ts            # BlackSwanHoverProvider — rich tooltips with causal chain
├── progress.ts         # ProgressSession wrapping vscode.withProgress
├── orchestrator.ts     # Run mutex, progress wiring, error routing
├── fixer.ts            # applyMathGuard() — diff preview, WorkspaceEdit; BlackSwanPreviewProvider
├── aiExplainer.ts      # explainFailure() — Gemini BYOK, rate limiter, privacy-safe prompts
├── dagPanel.ts         # DagPanelController — pure-SVG DAG, dark fintech theme, KPI strip, detail drawer
└── types.ts            # EngineRuntimeError, EngineFrameworkError, EngineProtocolError
```

### Data Flow

```
User clicks "▶ Run BlackSwan"
        │
        ▼
extension.ts: show scenario QuickPick
        │
        ▼
Orchestrator.run(document, functionName, scenarioName)
        │  mutex check — rejects concurrent runs for same file
        ▼
bridge.runBlackSwanEngine(fsPath, scenarioName, options)
        │  spawns Python child_process
        │  writes JSON request to stdin
        │  reads JSON response from stdout
        ▼
DiagnosticsController.apply(document, response)
        │  shatter_points → vscode.Diagnostic (red squiggles)
        │  causal_chain   → relatedInformation (clickable upstream lines)
        │  3 CodeActions per diagnostic (see Auto-Fixer below)
        ▼
BlackSwanHoverProvider.set(uri, shatter_points)
        │  stores tooltips keyed by line number
        ▼
DagPanelController.show(response, uri)
        │  opens/updates webview with pure-SVG DAG (manual layout, no D3/Dagre)
        │  failure_site = red (glow), intermediate = orange, root_input = yellow
```

---

### Hybrid Auto-Fixer

Every diagnostic exposes three CodeActions via `BlackSwanCodeActionProvider`:

```
User clicks lightbulb (Quick Fix)
        │
        ├─► "Apply Mathematical Guard"  (isPreferred: true)
        │         │
        │         ▼
        │   blackswan.applyGuard(uri, line, failureType)
        │         │
        │         ▼
        │   applyMathGuard() in fixer.ts
        │         │  spawns: python -m blackswan fix <file> --line N --type T
        │         │  parses JSON: { status, line, original, replacement, extra_lines }
        │         │  sets BlackSwanPreviewProvider content (blackswan-preview:// URI)
        │         │  vscode.diff — side-by-side preview
        │         │  user confirms → WorkspaceEdit.replace() + optional insert()
        │         └─ fully undoable via Ctrl+Z
        │
        ├─► "Explain with BlackSwan AI"
        │         │
        │         ▼
        │   blackswan.explainWithAI(explainPayload)
        │         │
        │         ▼
        │   explainFailure() in aiExplainer.ts
        │         │  reads API key from VS Code SecretStorage (never settings/source)
        │         │  rate-limit check: 15 RPM sliding window
        │         │  builds prompt from metadata only (no source code)
        │         │  POST to Gemini REST API (gemini-2.5-flash, thinkingBudget: 0)
        │         └─ opens Markdown virtual document beside active editor
        │
        └─► "Insert comment hint"
                  │  edit-based — indented # BlackSwan Fix Hint: comment
                  └─ no subprocess, no API call
```

**Fixer CLI** (standalone, no VS Code):

```bash
python -m blackswan fix models/risk.py --line 36 --type non_psd_matrix
```

Returns JSON: `{ "status": "ok", "line": 36, "original": "...", "replacement": "...", "extra_lines": [...], "explanation": "..." }`

**Guard patterns** (`core/blackswan/fixer/guards.py`, powered by `libcst`):

| failure_type | Strategy |
|---|---|
| `division_instability` | Replace denominator with `max(denom, 1e-10)` |
| `non_psd_matrix` | Append eigenvalue clamp via `np.linalg.eigh` + `np.maximum` |
| `ill_conditioned_matrix` | Replace `np.linalg.inv(M)` with conditional `pinv` when `cond > 1e12` |
| `nan_inf` | Append `np.nan_to_num(result, posinf=…, neginf=…)` |

**Privacy contract:** `aiExplainer.ts` sends only `ExplainPayload` fields to Gemini — failure type, engine message, frequency, fix hint, and causal chain variable names with line numbers. Raw source code is never transmitted.

### Python Path Resolution

`Orchestrator._resolvePythonPath()` checks in order:
1. `blackswan.pythonPath` VS Code setting (user override)
2. `python.defaultInterpreterPath` (Python extension)
3. Platform default: `"python"` on Windows, `"python3"` elsewhere

### Error Types

| Error class | Meaning |
|---|---|
| `EngineRuntimeError` | Python not found / process failed to start |
| `EngineFrameworkError` | Engine returned `status: "error"` with structured message |
| `EngineProtocolError` | stdout was not valid JSON (version mismatch) |

---

## JSON Contract

The contract (`contract/schema.json`) is the single source of truth for engine↔extension communication.

### Request (Extension → Engine)

```json
{
  "version": "1.0",
  "command": "stress_test",
  "file_path": "/absolute/path/to/model.py",
  "function_target": "calculate_portfolio_var",
  "scenario": {
    "name": "liquidity_crash",
    "parameters": {
      "spread_widening_bps": 250,
      "vol_regime_multiplier": 1.8,
      "correlation_shift": 0.25,
      "turnover_decay_pct": -40
    },
    "iterations": 5000,
    "seed": 42
  }
}
```

### Response (Engine → Extension)

```json
{
  "version": "1.0",
  "status": "failures_detected",
  "runtime_ms": 3420,
  "iterations_completed": 5000,
  "summary": {
    "total_failures": 847,
    "failure_rate": 0.1694,
    "unique_failure_types": 2
  },
  "shatter_points": [
    {
      "id": "sp_001",
      "line": 82,
      "column": 12,
      "severity": "critical",
      "failure_type": "non_psd_matrix",
      "message": "...",
      "frequency": "847 / 5000 iterations (16.9%)",
      "causal_chain": [
        { "line": 14, "variable": "corr_shift",        "role": "root_input" },
        { "line": 47, "variable": "adjusted_corr_matrix", "role": "intermediate" },
        { "line": 82, "variable": "cov_matrix",         "role": "failure_site" }
      ],
      "fix_hint": "..."
    }
  ],
  "scenario_card": {
    "name": "liquidity_crash",
    "parameters_applied": {},
    "seed": 42,
    "reproducible": true
  }
}
```

Every engine response is validated against `contract/schema.json` before being returned. Any change to the response structure requires a schema update.

---

## Design Decisions & Assumptions

| Decision | Rationale |
|---|---|
| Sequential NumPy loop, no multiprocessing | 5 000 iterations finish in <5 s. Parallelism adds complexity; profile first. |
| No AST shape inference | NumPy broadcasting rules are not statically verifiable; shape analysis is a runtime concern. |
| Pandas chaining treated as single DAG node | Method chains have no intermediate named variables to track. |
| child_process stdout (not WebSocket) | Sufficient for synchronous request/response; avoids server lifecycle complexity. |
| GA crossover is 50/50 per-param coin flip | Simple and sufficient. Arithmetic crossover adds complexity for marginal gain at this scale. |
| Validator skips (not raises) on invalid inputs | Invalid inputs are physically impossible, not failures. Skipping prevents false positives. |
| baseline_established in RunResult | Makes baseline failure observable to callers without raising from the runner constructor. |
| libcst for fixer (not ast + astor) | libcst is a Concrete Syntax Tree — it preserves formatting, comments, and whitespace exactly. ast round-trips lose indentation and blank lines. |
| Fixer sends replacement/extra_lines separately | The extension can apply replacement in-place and insert extra_lines after — avoids rewriting multi-line blocks from scratch. |
| Gemini BYOK via SecretStorage | API keys must never appear in settings.json (world-readable). SecretStorage is OS-encrypted. |
| thinkingBudget: 0 on Gemini 2.5 flash | Gemini 2.5 flash is a reasoning model. Without this flag, internal thinking consumes most of the output token budget, truncating the response. |
| Privacy: no source code in AI prompts | Source code may contain proprietary logic. Only structural metadata (failure type, variable names, line numbers) is transmitted. |
