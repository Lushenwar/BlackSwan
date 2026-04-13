# BlackSwan

[![PyPI version](https://img.shields.io/pypi/v/blackswan?color=black&label=blackswan)](https://pypi.org/project/blackswan/)
[![Python](https://img.shields.io/pypi/pyversions/blackswan)](https://pypi.org/project/blackswan/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/Lushenwar/BlackSwan/blackswan-ci.yml?label=build)](https://github.com/Lushenwar/BlackSwan/actions)

**A debugger for mathematical fragility.**

<img width="1280" height="720" alt="logo" src="https://github.com/user-attachments/assets/a1437241-bbd4-4134-ba24-a3cfa1421918" />

BlackSwan stress-tests financial and mathematical Python code to find the exact line where your model breaks under extreme conditions — before your clients do.

Standard linters tell you your code runs. BlackSwan tells you where the math fails.

---

## What It Does

Given a Python function containing numerical or financial logic, BlackSwan:

1. Applies thousands of perturbations drawn from realistic stress scenarios (liquidity crash, vol spike, correlation breakdown, etc.)
2. Runs your function under each scenario, watching for numerical failures
3. Reports the **exact source line** where the model breaks, how often, and which input caused it

```
$ python -m blackswan test models/risk.py --scenario liquidity_crash

  BlackSwan 0.4.0  ·  liquidity_crash  ·  5,000 iterations  ·  seed 42  ·  full

  ── FAILURES DETECTED  (1 shatter point) ─────────────────────────────────────────

   CRITICAL   Non-PSD Matrix                                            line 82

  Covariance matrix loses positive semi-definiteness when pairwise
  correlation exceeds 0.91. Smallest eigenvalue: -0.0034.

  Frequency   ████████░░░░░░░░░░░░░░░░░░░░  16.9%  847 / 5000 iterations
  Confidence  high

  Causal Chain ────────────────────────────
    →  line 14    corr_shift              root input
    ·  line 47    adjusted_corr_matrix    intermediate
    ►  line 82    cov_matrix              FAILURE SITE

  Fix Hint     Apply nearest-PSD correction (Higham 2002) after correlation
               perturbation, or clamp eigenvalues to epsilon.

  Quick Fix   $ python -m blackswan fix risk.py --line 82 --type non_psd_matrix
```

---

## Installation

### Python Engine (CLI + API)

```bash
pip install blackswan
```

With the mathematical auto-fixer (requires `libcst`):

```bash
pip install blackswan[fixer]
```

With the Claude MCP server:

```bash
pip install blackswan[mcp]
```

Requires Python 3.11+.

### VS Code Extension

<p align="center">
  <img src="extension/media/logo.png" width="120" alt="BlackSwan extension icon" />
</p>

**v0.4.0** — Download [`blackswan-vscode-0.4.0.vsix`](https://github.com/Lushenwar/BlackSwan/releases/latest) from Releases and install it manually:

```bash
code --install-extension blackswan-vscode-0.4.0.vsix
```

Or: **Extensions** panel → `···` menu → **Install from VSIX…**

Once installed, open any Python file containing numerical logic and click **▶ Run BlackSwan** above a function definition.

---

## Usage

### VS Code

1. Open a Python file containing financial or mathematical logic
2. Click **▶ Run BlackSwan** above any function
3. Select a preset stress scenario from the dropdown
4. Watch the progress bar — failures appear as red squiggles with hover tooltips
5. Click any squiggle to see the failure type, frequency, and a full causal chain in the hover tooltip
6. Use the lightbulb Quick Fix menu on any squiggle to access the **Hybrid Auto-Fixer**
7. Open the **BlackSwan DAG** panel to explore the dependency graph — failure site nodes glow red, propagation nodes orange, root inputs yellow

**Settings** (`Ctrl+,` → search "BlackSwan"):

| Setting | Default | Description |
|---|---|---|
| `blackswan.pythonPath` | _(auto)_ | Python executable path. Falls back to the Python extension's interpreter. |
| `blackswan.mode` | `fast` | `fast` for responsive IDE feedback; `full` for verified attribution. |
| `blackswan.maxRuntimeSec` | _(none)_ | Hard time cap in seconds. Engine stops early if exceeded. |
| `blackswan.claudeModel` | `claude-sonnet-4-20250514` | Claude model for AI failure explanations. Run **BlackSwan: Set Anthropic API Key** to configure. |

### Auto-Fixer

Every red squiggle has three Quick Fix options in the lightbulb menu:

| Action | What it does |
|---|---|
| **Apply Mathematical Guard** | Rewrites the failing line using a deterministic libcst guard (epsilon clamp, PSD correction, conditional `pinv`, or `nan_to_num`). Shows a side-by-side diff before applying — fully undoable. |
| **Explain with BlackSwan AI** | Sends the failure metadata (type, frequency, causal chain, fix hint — never source code) to Claude and displays a plain-English explanation in a side panel. |
| **Insert comment hint** | Adds an indented `# BlackSwan Fix Hint:` comment directly below the failing line — no subprocess, no API. |

**Supported guard types:**

| Failure type | Guard applied |
|---|---|
| `division_instability` | `max(denominator, 1e-10)` epsilon clamp |
| `non_psd_matrix` | Higham 2002 nearest-PSD via `np.linalg.eigh` + `np.maximum` |
| `ill_conditioned_matrix` | Conditional `np.linalg.pinv` fallback when `cond > 1e12` |
| `nan_inf` | `np.nan_to_num(result, posinf=…, neginf=…)` guard |

**Fixer CLI** (also available standalone):

```bash
# Install the fixer extra
pip install blackswan[fixer]

# Apply a mathematical guard at a detected failure line
python -m blackswan fix models/risk.py --line 82 --type non_psd_matrix
```

**AI explanations (BYOK):**

1. Run the command **BlackSwan: Set Anthropic API Key** from the Command Palette (`Ctrl+Shift+P`)
2. Paste your [Anthropic API key](https://console.anthropic.com/) — stored encrypted in VS Code's SecretStorage, never in settings or source
3. Click **Explain with BlackSwan AI** on any failure

---

### CLI

```bash
# Run standard Monte Carlo stress test (human-readable terminal output)
python -m blackswan test models/risk.py --scenario liquidity_crash

# JSON output (pipe-safe — auto-detected when stdout is not a TTY)
python -m blackswan test models/risk.py --scenario liquidity_crash | jq .

# Explicit format flags
python -m blackswan test models/risk.py --scenario liquidity_crash --format text
python -m blackswan test models/risk.py --scenario liquidity_crash --format json

# SARIF output for GitHub Code Scanning / any SARIF-aware CI tool
python -m blackswan test models/risk.py --scenario liquidity_crash --output sarif
python -m blackswan test models/risk.py --scenario liquidity_crash --output sarif --output-path results.sarif

# Specify a target function explicitly
python -m blackswan test models/risk.py --scenario vol_spike --function calculate_var

# Override iteration count and seed
python -m blackswan test models/risk.py --scenario correlation_breakdown --iterations 10000 --seed 123

# Fast mode — skips attribution replay (default in VS Code)
python -m blackswan test models/risk.py --scenario liquidity_crash --mode fast

# Full mode — Two-Path engine with Slow-Path attribution replay (highest confidence)
python -m blackswan test models/risk.py --scenario liquidity_crash --mode full

# Adversarial mode — genetic algorithm actively searches for worst-case inputs
python -m blackswan test models/risk.py --scenario liquidity_crash --adversarial --population 200

# Budget flags — stop early if a time or iteration limit is hit
python -m blackswan test models/risk.py --scenario liquidity_crash --max-runtime-sec 30
python -m blackswan test models/risk.py --scenario liquidity_crash --max-iterations 2000

# List available scenarios
python -m blackswan --list-scenarios
```

**Exit codes:** `0` = no failures, `1` = failures detected, `2` = engine error.

---

### SARIF and CI Integration

BlackSwan emits [SARIF 2.1.0](https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html) for GitHub Code Scanning and any SARIF-aware CI tool.

**Emit SARIF from CLI:**

```bash
python -m blackswan test models/risk.py \
  --scenario liquidity_crash \
  --output sarif \
  --output-path results.sarif
```

**Upload to GitHub Code Scanning** (`.github/workflows/blackswan-ci.yml` is pre-configured in this repo):

```yaml
- name: Run BlackSwan stress scan
  run: |
    python -m blackswan test models/risk.py \
      --scenario liquidity_crash --seed 42 \
      --output sarif --output-path results.sarif || true

- name: Upload SARIF
  uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: results.sarif
    category: blackswan
```

Each SARIF result includes:
- The exact source line as the primary location
- Causal chain links as clickable `relatedLocations` (GitHub shows them as annotation links)
- Fix hints embedded as `fixes` entries
- Severity mapped to SARIF levels (`critical` → `error`, `warning` → `warning`)

---

### MCP Server (Claude Code Integration)

BlackSwan exposes a [Model Context Protocol](https://modelcontextprotocol.io/) server so Claude can run stress tests directly from your IDE conversation.

**Install:**

```bash
pip install blackswan[mcp]
```

**Configure in `claude_desktop_config.json`:**

```json
{
  "mcpServers": {
    "blackswan": {
      "command": "blackswan-mcp"
    }
  }
}
```

Or without installing:

```json
{
  "mcpServers": {
    "blackswan": {
      "command": "python",
      "args": ["-m", "blackswan.mcp_server"]
    }
  }
}
```

**Available MCP tools:**

| Tool | Description |
|---|---|
| `run_blackswan` | Run a stress test on a Python file — returns the full findings dict |
| `list_scenarios` | List all preset scenarios with descriptions and perturbation targets |
| `get_finding_detail` | Extract a specific shatter point by id from a previous result |
| `explain_finding` | Return a Markdown explanation of what a failure type means and why it matters |

Once configured, you can ask Claude: *"Run BlackSwan on my portfolio model with the correlation_breakdown scenario"* and get findings, causal chains, and fix hints directly in the conversation.

---

### Python API

```python
from blackswan.engine.runner import StressRunner
from blackswan.scenarios.registry import load_scenario
from blackswan.parser.auto_tagger import AutoTagger
from pathlib import Path

file_path = Path("models/risk.py")
scenario = load_scenario("liquidity_crash")
detectors = AutoTagger(file_path).detector_suite()

runner = StressRunner(
    fn=calculate_portfolio_var,
    base_inputs={"weights": w, "vol": v, "correlation": 0.0},
    scenario=scenario,
    detectors=detectors,
    seed=42,
)
result = runner.run()

for bucket in result.root_cause_buckets:
    print(f"Line {bucket.line}: {bucket.message} ({bucket.occurrence_rate:.1%})")
```

---

## Execution Modes

| Mode | Speed | Attribution confidence | When to use |
|---|---|---|---|
| `fast` (default) | Fastest | `unverified` | Interactive IDE use — rapid feedback loop |
| `full` | Slower | `high` / `medium` / `low` | Final audits, CI pipelines, reproducible reports |
| `adversarial` | Slowest | `unverified` | Finding worst-case inputs via genetic search |

Every run emits a **ReproducibilityCard** — a machine-readable provenance record with the exact BlackSwan version, Python version, NumPy version, scenario hash, seed, and a ready-to-paste replay command:

```json
"reproducibility_card": {
  "blackswan_version": "0.4.0",
  "python_version": "3.11.9",
  "numpy_version": "1.26.4",
  "platform": "linux",
  "seed": 42,
  "scenario_name": "liquidity_crash",
  "scenario_hash": "a3f9c2e1d084",
  "mode": "full",
  "iterations_requested": 5000,
  "iterations_executed": 5000,
  "reproducible": true,
  "replay_command": "python -m blackswan test models/risk.py --scenario liquidity_crash --seed 42 --mode full"
}
```

---

## Preset Stress Scenarios

| Scenario | What it tests |
|---|---|
| `liquidity_crash` | Spread widening 1.5–3.5×, vol expansion, correlation shift +0.10–+0.35, turnover collapse 30–70% |
| `vol_spike` | Volatility multiplier 2–4×, mild correlation increase |
| `correlation_breakdown` | Pairwise correlation shift +0.20–+0.50, vol increase 1.2–1.5× |
| `rate_shock` | Interest rate shift +100–+300 bps, spread widening +50–+150 bps |
| `missing_data` | Random NaN injection 5–20% of data points, partial time series truncation |

All scenarios are reproducible YAML files — same seed always produces identical results. See [docs/SCENARIOS.md](docs/SCENARIOS.md) for full parameter tables and how to write custom scenarios.

---

## Failure Detectors

BlackSwan auto-selects detectors from AST analysis of the target file — no configuration required.

| Detector | What it catches |
|---|---|
| `NaNInfDetector` | Any computation producing NaN or Inf |
| `DivisionStabilityDetector` | Denominator approaching zero (default threshold: 1e-10) |
| `MatrixPSDDetector` | Covariance/correlation matrix losing positive semi-definiteness |
| `ConditionNumberDetector` | Ill-conditioned matrices before inversion (condition number > 1e12) |
| `BoundsDetector` | Outputs exceeding configurable plausible bounds |
| `ExplodingGradientDetector` | Output growth > 100× input perturbation magnitude |
| `RegimeShiftDetector` | Structural breaks in output distribution across iterations |
| `LogicalInvariantDetector` | User-defined assertion violations (e.g. weights must sum to 1) |

---

## Adversarial Mode

Standard Monte Carlo samples perturbations randomly. Adversarial mode uses a genetic algorithm to **evolve** stress parameters toward worst-case scenarios:

```bash
python -m blackswan test models/risk.py --scenario liquidity_crash --adversarial
```

The GA maintains a population of parameter sets, scores each by failure severity, and breeds the worst performers over successive generations. A `HardnessAdaptor` automatically increases perturbation intensity when no failures are found, preventing the search from stalling on robust code.

---

## Supported Code Patterns

BlackSwan works well on:

- Pure functions with NumPy/Pandas inputs and outputs
- Explicit variable assignments (not chained one-liners)
- NumPy array operations, `linalg`, random
- Pandas DataFrame column operations
- Single-file scripts and focused modules

BlackSwan is intentionally scoped to portfolio risk, covariance/correlation analysis, and VaR-style models. It does not attempt to support all Python. Unsupported files are rejected with an explanatory message.

---

## Architecture

```
blackswan/
├── engine/
│   ├── runner.py          # Monte Carlo StressRunner (Two-Path: Fast + Slow)
│   ├── adversarial.py     # Evolutionary EvolutionaryStressRunner + HardnessAdaptor
│   ├── perturbation.py    # Perturbation application from scenario YAML
│   └── validator.py       # PlausibilityValidator — filters impossible inputs
├── detectors/
│   ├── base.py            # FailureDetector ABC + Finding + TriggerDisclosure
│   ├── numerical.py       # NaNInf, DivisionStability, ExplodingGradient, RegimeShift, LogicalInvariant
│   ├── matrix.py          # MatrixPSD, ConditionNumber
│   ├── portfolio.py       # BoundsDetector
│   └── sensitivity.py     # Root cause sensitivity analysis
├── parser/
│   ├── ast_analyzer.py    # AST extraction of functions, variables, calls
│   ├── variable_tracker.py
│   └── graph.py           # DAG construction + JSON serialization
├── scenarios/
│   ├── registry.py        # Scenario + PlausibilityConstraint dataclasses, YAML loading
│   └── presets/           # 5 YAML scenario files
├── attribution/
│   ├── traceback.py       # Proximate failure location
│   └── causal_chain.py    # DAG walk → root cause ranking
├── fixer/
│   └── guards.py          # Deterministic libcst guards (PSD, epsilon, pinv, nan_to_num)
├── sarif.py               # SARIF 2.1.0 serializer for GitHub Code Scanning / CI
├── mcp_server.py          # Claude MCP server (stdio) — run_blackswan, list_scenarios, explain
└── cli.py                 # argparse CLI entry point
```

The VS Code extension (`extension/`) communicates with the engine via a versioned JSON contract (`contract/schema.json`). The engine is fully functional as a standalone CLI — the extension is a renderer, not the product.

For a detailed breakdown see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](LICENSE).
