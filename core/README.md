# blackswan

[![PyPI version](https://img.shields.io/pypi/v/blackswan?color=black&label=blackswan)](https://pypi.org/project/blackswan/)
[![Python](https://img.shields.io/pypi/pyversions/blackswan)](https://pypi.org/project/blackswan/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Lushenwar/BlackSwan/blob/main/LICENSE)

**A stress-testing engine for financial and mathematical Python code.**

BlackSwan finds the exact source line where your model breaks under extreme conditions — before production does. It is a debugger for numerical fragility, not a linter, not a simulator.

---

## Installation

```bash
pip install blackswan
```

Requires **Python 3.11+** and NumPy 1.26+.

---

## Quickstart

```bash
# Stress-test a function with the liquidity crash scenario
python -m blackswan test models/risk.py --scenario liquidity_crash
```

```json
{
  "status": "failures_detected",
  "runtime_ms": 2840,
  "iterations_completed": 5000,
  "summary": {
    "total_failures": 847,
    "failure_rate": 0.1694,
    "unique_failure_types": 1
  },
  "shatter_points": [
    {
      "line": 36,
      "severity": "critical",
      "failure_type": "non_psd_matrix",
      "message": "Covariance matrix loses positive semi-definiteness when pairwise correlation exceeds 0.91. Smallest eigenvalue: -0.0034.",
      "frequency": "847 / 5000 iterations (16.9%)",
      "causal_chain": [
        { "line": 8,  "variable": "correlation", "role": "root_input" },
        { "line": 31, "variable": "corr_matrix",  "role": "intermediate" },
        { "line": 36, "variable": "cov_matrix",   "role": "failure_site" }
      ],
      "fix_hint": "Apply nearest-PSD correction (Higham 2002) after correlation perturbation, or clamp eigenvalues to epsilon."
    }
  ]
}
```

---

## CLI Reference

### `test` — stress-test a model

```
python -m blackswan test <file> [options]

Options:
  --scenario     Preset scenario name (required)
  --function     Target function name (auto-detected if omitted)
  --iterations   Number of Monte Carlo iterations (default: 5000)
  --seed         Random seed for reproducibility (default: 42)
  --adversarial  Use genetic algorithm to search for worst-case inputs
  --population   GA population size (default: 100, requires --adversarial)
```

**Exit codes:** `0` = no failures, `1` = failures detected, `2` = engine error.

### `fix` — generate a deterministic guard for a failure line

```
python -m blackswan fix <file> --line N --type TYPE
```

Requires `pip install blackswan[fixer]` (adds `libcst`).

| `--type` | Guard applied |
|---|---|
| `division_instability` | `max(denominator, 1e-10)` epsilon clamp |
| `non_psd_matrix` | Nearest-PSD correction via `np.linalg.eigh` + `np.maximum` |
| `ill_conditioned_matrix` | Conditional `np.linalg.pinv` fallback when `cond > 1e12` |
| `nan_inf` | `np.nan_to_num(result, posinf=…, neginf=…)` guard |

Output JSON:

```json
{
  "status": "ok",
  "line": 36,
  "original": "    cov_matrix = np.cov(returns.T)",
  "replacement": "    cov_matrix = np.cov(returns.T)",
  "extra_lines": [
    "    vals, vecs = np.linalg.eigh(cov_matrix)",
    "    cov_matrix = vecs @ np.diag(np.maximum(vals, 1e-10)) @ vecs.T"
  ],
  "explanation": "Clamps negative eigenvalues to epsilon, restoring PSD property."
}
```

`status` is `"ok"`, `"error"`, or `"unsupported"`. Exit code is always `0` (the caller decides what to do with the result).

### Available Scenarios

| Name | Description |
|---|---|
| `liquidity_crash` | Spread widening, vol expansion, correlation stress, turnover collapse |
| `vol_spike` | 2–4× volatility multiplier, mild correlation increase |
| `correlation_breakdown` | Pairwise correlation shift +0.20–+0.50 |
| `rate_shock` | +100–+300 bps interest rate shock with spread widening |
| `missing_data` | Random NaN injection and partial time series truncation |

---

## Python API

```python
from blackswan.engine.runner import StressRunner
from blackswan.scenarios.registry import load_scenario

scenario = load_scenario("liquidity_crash")
runner = StressRunner(scenario)

result = runner.run(
    calculate_portfolio_var,
    base_inputs={"weights": weights, "vol": vols, "correlation": 0.0},
)

print(f"Failures: {len(result.findings)}")
for finding in result.findings:
    print(f"  Line {finding.line} [{finding.severity}]: {finding.message}")
```

### Adversarial API

```python
from blackswan.engine.adversarial import EvolutionaryStressRunner

runner = EvolutionaryStressRunner(
    scenario,
    n_generations=20,
    population_size=100,
    elite_fraction=0.2,
)
result = runner.run(fn, base_inputs)
```

### Custom Detectors

```python
from blackswan.detectors.numerical import LogicalInvariantDetector

# Weights must always sum to 1 (+/-1e-6)
invariant = LogicalInvariantDetector(
    assertion=lambda inputs, output: abs(output.sum() - 1.0) < 1e-6,
    name="weights_sum_to_one",
)
runner = StressRunner(scenario, extra_detectors=[invariant])
```

---

## Failure Detectors

| Detector | Catches | Always active? |
|---|---|---|
| `NaNInfDetector` | NaN or Inf in any output | Yes |
| `DivisionStabilityDetector` | Denominator approaching zero | Yes |
| `MatrixPSDDetector` | Covariance/correlation matrix non-PSD | Auto (on matrix code) |
| `ConditionNumberDetector` | Ill-conditioned matrices (cond > 1e12) | Auto (on `linalg.inv`) |
| `BoundsDetector` | Outputs outside configurable plausible ranges | Auto |
| `ExplodingGradientDetector` | Output growth > 100x input perturbation | Auto |
| `RegimeShiftDetector` | Structural breaks in output distribution | Auto |
| `LogicalInvariantDetector` | User-defined assertions | On demand |

Detectors are auto-tagged to relevant source lines via AST analysis.

---

## Custom Scenarios

Create a YAML file and pass its path as `--scenario`:

```yaml
name: my_scenario
description: "Custom stress test for credit portfolio"
perturbations:
  - target: spread
    type: multiplicative
    distribution: lognormal
    mu: 2.0
    sigma: 0.4
    constraints:
      min: 1.0
      max: 5.0
  - target: correlation
    type: additive
    distribution: uniform
    low: 0.15
    high: 0.40
global_constraints:
  - target: correlation
    min_value: -1.0
    max_value: 0.95
defaults:
  iterations: 5000
  seed: 42
```

```bash
python -m blackswan test models/credit.py --scenario path/to/my_scenario.yaml
```

---

## What BlackSwan Supports

**Works well on:**
- Pure functions with NumPy / Pandas inputs and outputs
- Explicit variable assignments
- NumPy array operations, `linalg`, random
- Pandas DataFrame column operations
- Single-file scripts and focused modules

**Out of scope for V1:**
- Deeply nested class hierarchies with side effects
- Config-driven logic loading parameters from databases
- Multi-threaded or async computation pipelines
- C extensions or Cython modules

---

## Links

- [GitHub](https://github.com/Lushenwar/BlackSwan)
- [Architecture](https://github.com/Lushenwar/BlackSwan/blob/main/docs/ARCHITECTURE.md)
- [Scenarios](https://github.com/Lushenwar/BlackSwan/blob/main/docs/SCENARIOS.md)
- [Changelog](https://github.com/Lushenwar/BlackSwan/blob/main/CHANGELOG.md)
- [Bug Tracker](https://github.com/Lushenwar/BlackSwan/issues)

---

## License

MIT
