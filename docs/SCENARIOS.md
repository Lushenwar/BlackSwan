# BlackSwan — Stress Scenarios

Scenarios are the core configuration unit in BlackSwan. Each scenario defines which parameters to perturb, how to perturb them, and what constraints to enforce. All scenarios are deterministic — same seed always produces identical results.

---

## Using a Preset Scenario

```bash
# CLI
python -m blackswan test models/risk.py --scenario liquidity_crash

# Adversarial (GA) mode — actively searches for worst-case inputs
python -m blackswan test models/risk.py --scenario liquidity_crash --adversarial
```

```python
# Python API
from blackswan.scenarios.registry import load_scenario
scenario = load_scenario("liquidity_crash")
```

---

## Preset Scenarios

### `liquidity_crash`

Models a severe market liquidity event: bid-ask spreads blow out, volatility spikes, assets become correlated, and turnover collapses.

**Targets:** `spread`, `vol`, `correlation`, `turnover`

| Parameter | Type | Distribution | Range |
|---|---|---|---|
| `spread` | multiplicative | lognormal | μ=2.5, σ=0.4 (effective: ~1.5×–3.5×) |
| `vol` | multiplicative | lognormal | μ=1.8, σ=0.3 |
| `correlation` | additive | uniform | +0.10 to +0.35 |
| `turnover` | multiplicative | uniform | 0.30× to 0.70× |

**Most likely to trigger:** `MatrixPSDDetector`, `NaNInfDetector`

**Why it's hard:** The correlation additive shift pushes pairwise correlations toward 1.0. When `base_correlation + shift > ~0.9`, covariance matrices systematically lose positive semi-definiteness.

---

### `vol_spike`

Models a pure volatility shock (e.g., VIX spike) with mild correlation contagion.

**Targets:** `vol`, `correlation`

| Parameter | Type | Distribution | Range |
|---|---|---|---|
| `vol` | multiplicative | uniform | 2.0× to 4.0× |
| `correlation` | additive | uniform | +0.05 to +0.15 |

**Most likely to trigger:** `NaNInfDetector`, `BoundsDetector`, `ExplodingGradientDetector`

**Why it's hard:** 4× vol multiplier causes VaR and expected shortfall estimates to overflow plausible bounds. Leverage calculations can spiral under squared-vol inputs.

---

### `correlation_breakdown`

Models a correlation regime shift — assets that were weakly correlated suddenly move together (e.g., 2008 financial crisis).

**Targets:** `correlation`, `vol`

| Parameter | Type | Distribution | Range |
|---|---|---|---|
| `correlation` | additive | uniform | +0.20 to +0.50 |
| `vol` | multiplicative | uniform | 1.2× to 1.5× |

**Most likely to trigger:** `MatrixPSDDetector`, `ConditionNumberDetector`

**Why it's hard:** Large positive correlation shifts are the most reliable way to break PSD constraints. Combined with vol increase, condition numbers of covariance matrices climb rapidly.

---

### `rate_shock`

Models a rapid interest rate move (e.g., central bank emergency hike) with associated spread widening.

**Targets:** `rate`, `spread`

| Parameter | Type | Distribution | Range |
|---|---|---|---|
| `rate` | additive | uniform | +0.01 to +0.03 (100–300 bps) |
| `spread` | additive | uniform | +0.005 to +0.015 (50–150 bps) |

**Most likely to trigger:** `DivisionStabilityDetector`, `BoundsDetector`

**Why it's hard:** Duration calculations often involve division by `(1 + rate)^n`. When the denominator approaches zero or models compute DV01 iteratively, small rate changes compound.

---

### `missing_data`

Models data quality failure: NaN injection and partial time series truncation.

**Targets:** `data` (array inputs)

| Parameter | Type | Range |
|---|---|---|
| NaN injection rate | uniform | 5% to 20% of array elements |
| Time series truncation | uniform | 60% to 95% of original length |

**Most likely to trigger:** `NaNInfDetector`, `DivisionStabilityDetector`

**Why it's hard:** Covariance estimators over NaN-contaminated arrays propagate NaN silently. Mean-variance calculations on truncated time series produce degenerate results when fewer observations than assets remain.

---

## YAML Scenario Format

Every scenario is a YAML file. Preset files live in `core/blackswan/scenarios/presets/`. You can pass a path to any YAML file as `--scenario`.

```yaml
name: my_scenario
description: "Credit portfolio stress under EM contagion"

perturbations:
  - target: spread
    type: multiplicative
    distribution: lognormal
    mu: 2.0
    sigma: 0.4
    constraints:
      min: 1.0    # clamp individual param value after perturbation
      max: 6.0

  - target: correlation
    type: additive
    distribution: uniform
    low: 0.15
    high: 0.45
    constraints:
      min: -1.0
      max: 0.99

  - target: vol
    type: multiplicative
    distribution: uniform
    low: 1.2
    high: 2.5

global_constraints:
  - target: correlation
    min_value: -1.0    # reject entire input set if constraint violated
    max_value: 0.95
  - target: spread
    min_value: 0.0

defaults:
  iterations: 5000
  seed: 42
```

### Field Reference

#### `perturbations[].type`

| Value | Effect |
|---|---|
| `additive` | `param += sample` |
| `multiplicative` | `param *= sample` |

#### `perturbations[].distribution`

| Value | Parameters |
|---|---|
| `uniform` | `low`, `high` |
| `lognormal` | `mu` (mean of log), `sigma` (std of log) |

#### `perturbations[].constraints`

Per-parameter clamp applied **after** the sample is drawn. Does not reject the entire input set — use `global_constraints` for cross-parameter validation.

#### `global_constraints`

A list of `{target, min_value, max_value}` entries. The `PlausibilityValidator` checks all constraints before calling the target function. If any constraint is violated, the iteration is **skipped** (not counted as a failure). This prevents physically impossible inputs (e.g., correlation > 1.0) from generating false positives.

---

## Adversarial Mode

Standard Monte Carlo samples perturbations independently and randomly. Adversarial mode uses a genetic algorithm to **evolve** parameter combinations toward maximum failure severity.

```bash
python -m blackswan test models/risk.py --scenario liquidity_crash --adversarial --population 200
```

### How It Works

1. Initialise a population of `--population` parameter sets sampled uniformly from the scenario's perturbation ranges
2. Evaluate each individual: run the function, collect detector findings, compute fitness
3. Fitness = `sum(severity_weight × type_bonus)` per finding (`critical` = 3.0, `warning` = 1.5, `info` = 0.5)
4. Retain the top `elite_fraction` (default 20%) as parents
5. Crossover: for each parameter, randomly inherit from one of two parents
6. Mutate: add Gaussian noise (`noise_scale=0.1`) to child parameters
7. Repeat for `--iterations` generations (default 20)
8. `HardnessAdaptor`: if a generation produces zero fitness, expand perturbation ranges by 10% (up to 2×) to break stalling

### When to Use Adversarial Mode

| Situation | Recommendation |
|---|---|
| Initial scan of a new model | Standard Monte Carlo (`--iterations 5000`) |
| Model passed Monte Carlo but you suspect fragility | `--adversarial --population 100` |
| Proving robustness for audit/compliance | `--adversarial --population 500 --iterations 50` |
| Tight deadline, quick check | Standard Monte Carlo |

Adversarial mode runs `n_generations × population_size` total evaluations. Default (20 × 100 = 2 000) is faster than the standard 5 000 iteration Monte Carlo but explores the failure surface more intelligently.

---

## Reproducibility

Every run that specifies a seed is fully reproducible:

```bash
python -m blackswan test models/risk.py --scenario liquidity_crash --seed 42
```

The seed initialises NumPy's `default_rng`, which is used for all perturbation sampling and GA operations. The same seed on the same code with the same scenario always produces byte-identical output.

The `scenario_card` field in the JSON response records the scenario name, all applied parameters, and the seed — sufficient to reproduce any run exactly.
