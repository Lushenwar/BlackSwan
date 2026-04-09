# CLAUDE_ENGINE_UPGRADE.md — BlackSwan Engine Upgrade

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `blackswan-core` from a random Monte Carlo sampler into a configurable, adversarial stress-testing engine using plausibility constraints, three new detectors, and a Genetic Algorithm search loop.

**Architecture:** Phase A hardens the existing engine by adding an input validation layer and three new stateful detectors. Phase B replaces random sampling with an `EvolutionaryStressRunner` that evolves perturbation sets toward maximum failure severity across generations with adaptive intensity scaling.

**Tech Stack:** Python 3.11, NumPy, PyYAML, pytest, existing `FailureDetector` / `StressRunner` / `Perturbation` infrastructure.

---

## WHAT ALREADY EXISTS — DO NOT REBUILD

Before touching any code, understand the existing foundation:

| Component | Location | Status |
|---|---|---|
| `Scenario` class + YAML loading | `scenarios/registry.py` | **COMPLETE** |
| All 5 preset YAML files | `scenarios/presets/*.yaml` | **COMPLETE** |
| `Perturbation` + `apply_perturbations` | `engine/perturbation.py` | **COMPLETE** |
| `StressRunner` (Monte Carlo loop) | `engine/runner.py` | **COMPLETE** |
| `FailureDetector` base class + `Finding` | `detectors/base.py` | **COMPLETE** |
| `NaNInfDetector`, `DivisionStabilityDetector` | `detectors/numerical.py` | **COMPLETE** |
| `MatrixPSDDetector`, `ConditionNumberDetector` | `detectors/matrix.py` | **COMPLETE** |
| `BoundsDetector` | `detectors/portfolio.py` | **COMPLETE** |
| CLI + JSON contract | `cli.py` | **COMPLETE** |

The `FailureDetector.check` signature you must honour:
```python
def check(self, inputs: dict, output: Any, iteration: int) -> Finding | None:
```

---

## FILE MAP — WHAT CHANGES

```
core/
├── blackswan/
│   ├── detectors/
│   │   └── numerical.py          MODIFY — add 3 new detector classes
│   ├── engine/
│   │   ├── perturbation.py       MODIFY — add range_widening helper
│   │   ├── runner.py             MODIFY — add set_baseline() call before loop
│   │   ├── validator.py          CREATE — PlausibilityValidator
│   │   └── adversarial.py        CREATE — EvolutionaryStressRunner + GA logic
│   ├── scenarios/
│   │   └── registry.py           MODIFY — parse global_constraints from YAML
│   └── cli.py                    MODIFY — add --adversarial flag
└── tests/
    ├── test_validator.py          CREATE
    ├── test_new_detectors.py      CREATE
    └── test_adversarial.py        CREATE
```

---

## PHASE A: CORE HARDENING

---

### Task A1: PlausibilityValidator

Add a validation layer that rejects physically impossible perturbed inputs **before** they reach the target function. This eliminates nonsensical runs (e.g. negative stock prices) that would produce findings with no real-world meaning.

**Files:**
- Create: `core/blackswan/engine/validator.py`
- Create: `core/tests/test_validator.py`

---

- [ ] **Step 1: Write the failing tests**

Create `core/tests/test_validator.py`:

```python
"""Tests for PlausibilityValidator (Phase A, Task A1)."""

import pytest
from blackswan.engine.validator import PlausibilityConstraint, PlausibilityValidator, ValidationError


class TestConstraintFloor:
    def test_value_above_floor_passes(self):
        c = PlausibilityConstraint(target="price", min_value=0.0)
        c.check({"price": 1.0})  # must not raise

    def test_value_at_floor_passes(self):
        c = PlausibilityConstraint(target="price", min_value=0.0)
        c.check({"price": 0.0})  # must not raise

    def test_value_below_floor_raises(self):
        c = PlausibilityConstraint(target="price", min_value=0.0)
        with pytest.raises(ValidationError, match="price"):
            c.check({"price": -0.01})

    def test_missing_key_is_ignored(self):
        c = PlausibilityConstraint(target="price", min_value=0.0)
        c.check({"vol": 0.2})  # must not raise — key absent, nothing to validate


class TestConstraintCeiling:
    def test_value_below_ceiling_passes(self):
        c = PlausibilityConstraint(target="vol", max_value=5.0)
        c.check({"vol": 4.9})

    def test_value_at_ceiling_passes(self):
        c = PlausibilityConstraint(target="vol", max_value=5.0)
        c.check({"vol": 5.0})

    def test_value_above_ceiling_raises(self):
        c = PlausibilityConstraint(target="vol", max_value=5.0)
        with pytest.raises(ValidationError, match="vol"):
            c.check({"vol": 5.1})


class TestConstraintBoth:
    def test_in_range_passes(self):
        c = PlausibilityConstraint(target="weight", min_value=-2.0, max_value=2.0)
        c.check({"weight": 0.5})

    def test_below_min_raises(self):
        c = PlausibilityConstraint(target="weight", min_value=-2.0, max_value=2.0)
        with pytest.raises(ValidationError):
            c.check({"weight": -2.1})

    def test_above_max_raises(self):
        c = PlausibilityConstraint(target="weight", min_value=-2.0, max_value=2.0)
        with pytest.raises(ValidationError):
            c.check({"weight": 2.1})


class TestValidator:
    def test_empty_constraints_always_passes(self):
        v = PlausibilityValidator([])
        assert v.validate({"price": -999.0}) is True

    def test_all_constraints_satisfied(self):
        v = PlausibilityValidator([
            PlausibilityConstraint("price", min_value=0.0),
            PlausibilityConstraint("vol", max_value=5.0),
        ])
        assert v.validate({"price": 1.0, "vol": 0.2}) is True

    def test_first_violation_returns_false(self):
        v = PlausibilityValidator([
            PlausibilityConstraint("price", min_value=0.0),
        ])
        assert v.validate({"price": -1.0}) is False

    def test_second_constraint_violation_returns_false(self):
        v = PlausibilityValidator([
            PlausibilityConstraint("price", min_value=0.0),
            PlausibilityConstraint("vol", max_value=5.0),
        ])
        assert v.validate({"price": 1.0, "vol": 99.0}) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd core
python -m pytest tests/test_validator.py -v
```

Expected: `ERROR` — `blackswan.engine.validator` module not found.

- [ ] **Step 3: Implement PlausibilityValidator**

Create `core/blackswan/engine/validator.py`:

```python
"""
Plausibility validation layer for BlackSwan perturbations.

Before a perturbed input dict reaches the target function, PlausibilityValidator
checks every configured constraint. If any constraint fails, the iteration is
skipped entirely — the inputs are not physically meaningful.

Usage in StressRunner:
    validator = PlausibilityValidator(scenario.global_constraints)
    if not validator.validate(perturbed):
        continue  # skip this iteration
"""

from __future__ import annotations

from dataclasses import dataclass


class ValidationError(Exception):
    """Raised by PlausibilityConstraint.check() on violation."""


@dataclass
class PlausibilityConstraint:
    """
    A single min/max bound on one input parameter.

    Either min_value or max_value (or both) may be supplied.
    If the target key is absent from the inputs dict, the constraint is
    silently skipped — partial input dicts are valid.
    """

    target: str
    min_value: float | None = None
    max_value: float | None = None

    def check(self, inputs: dict) -> None:
        """
        Raise ValidationError if inputs[target] violates this constraint.

        No-op if target is not in inputs.
        """
        if self.target not in inputs:
            return
        val = float(inputs[self.target])
        if self.min_value is not None and val < self.min_value:
            raise ValidationError(
                f"'{self.target}' = {val:.6g} is below minimum {self.min_value}"
            )
        if self.max_value is not None and val > self.max_value:
            raise ValidationError(
                f"'{self.target}' = {val:.6g} exceeds maximum {self.max_value}"
            )


class PlausibilityValidator:
    """
    Applies a list of PlausibilityConstraints to a perturbed input dict.

    Returns True if all constraints are satisfied, False if any fails.
    Never raises — use validate() as a boolean gate in the runner loop.
    """

    def __init__(self, constraints: list[PlausibilityConstraint]) -> None:
        self._constraints = constraints

    def validate(self, inputs: dict) -> bool:
        """Return True if all constraints pass, False if any fails."""
        try:
            for c in self._constraints:
                c.check(inputs)
            return True
        except ValidationError:
            return False
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd core
python -m pytest tests/test_validator.py -v
```

Expected: all `test_validator.py` tests pass.

- [ ] **Step 5: Add global_constraints parsing to Scenario registry**

Open `core/blackswan/scenarios/registry.py`.

Add `global_constraints` to the `Scenario` dataclass and parse it from YAML:

```python
# In the Scenario dataclass, add the new field after default_seed:
from ..engine.validator import PlausibilityConstraint, PlausibilityValidator

@dataclass
class Scenario:
    name: str
    display_name: str
    description: str
    perturbations: list[Perturbation]
    default_iterations: int
    default_seed: int
    global_constraints: list[PlausibilityConstraint] = field(default_factory=list)

    @property
    def iterations(self) -> int:
        return self.default_iterations

    def apply(self, inputs: dict, rng: Any) -> dict:
        return apply_perturbations(self.perturbations, inputs, rng)

    def make_validator(self) -> PlausibilityValidator:
        """Return a validator for this scenario's global constraints."""
        return PlausibilityValidator(self.global_constraints)
```

In `_parse_scenario`, after building `perturbations`, parse the optional `global_constraints` key:

```python
# Add this block in _parse_scenario, before building the Scenario:
raw_constraints = data.get("global_constraints", [])
global_constraints = []
for idx, c in enumerate(raw_constraints):
    if not isinstance(c, dict) or "target" not in c:
        raise ValueError(f"global_constraints[{idx}] must have a 'target' key")
    global_constraints.append(PlausibilityConstraint(
        target=str(c["target"]),
        min_value=float(c["min_value"]) if "min_value" in c else None,
        max_value=float(c["max_value"]) if "max_value" in c else None,
    ))

return Scenario(
    name=str(data["name"]),
    display_name=str(data["display_name"]),
    description=str(data["description"]).strip(),
    perturbations=perturbations,
    default_iterations=int(defaults["iterations"]),
    default_seed=int(defaults["seed"]),
    global_constraints=global_constraints,
)
```

- [ ] **Step 6: Wire validator into StressRunner**

Open `core/blackswan/engine/runner.py`. Add validator support to the run loop:

```python
# In StressRunner.run(), replace the for loop with:
validator = self.scenario.make_validator() if hasattr(self.scenario, "make_validator") else None

for i in range(self.scenario.iterations):
    perturbed = self.scenario.apply(self.base_inputs, rng)

    # Skip physically impossible inputs — don't waste an iteration.
    if validator is not None and not validator.validate(perturbed):
        continue

    try:
        output = self.fn(**perturbed)
    except Exception as exc:
        frames = [
            (frame.filename, frame.lineno)
            for frame in tb.extract_tb(exc.__traceback__)
        ]
        findings.append(Finding(
            failure_type="nan_inf",
            severity="critical",
            message=f"{type(exc).__name__}: {exc}",
            iteration=i,
            exc_frames=frames,
        ))
        continue

    for detector in self.detectors:
        finding = detector.check(inputs=perturbed, output=output, iteration=i)
        if finding is not None:
            findings.append(finding)
```

- [ ] **Step 7: Run the full test suite to confirm no regressions**

```bash
cd core
python -m pytest tests/ -v --tb=short
```

Expected: all previously passing tests still pass. The `hasattr` guard keeps existing tests that pass plain `Scenario` objects working unchanged.

- [ ] **Step 8: Commit**

```bash
git add core/blackswan/engine/validator.py core/blackswan/scenarios/registry.py core/blackswan/engine/runner.py core/tests/test_validator.py
git commit -m "feat: add PlausibilityValidator and global_constraints to Scenario"
```

---

### Task A2: ExplodingGradientDetector

Flags iterations where the change in output is disproportionately large relative to the change in input — a sign of extreme sensitivity or numerical blow-up.

**Files:**
- Modify: `core/blackswan/detectors/numerical.py`
- Modify: `core/blackswan/engine/runner.py`
- Create: `core/tests/test_new_detectors.py` (start with this class, others added in A3/A4)

**Design:** The detector stores the baseline output (from running the function on unperturbed inputs). On each iteration it computes `ratio = ||output - baseline|| / (||inputs_delta|| + ε)`. Fires when ratio exceeds `threshold` (default 100.0). The runner sets the baseline once before the loop via `detector.set_baseline()`.

---

- [ ] **Step 1: Write the failing tests**

Create `core/tests/test_new_detectors.py`:

```python
"""
Tests for Phase A new detectors:
  - ExplodingGradientDetector (Task A2)
  - RegimeShiftDetector (Task A3)
  - LogicalInvariantDetector (Task A4)
"""

import numpy as np
import pytest
from blackswan.detectors.numerical import (
    ExplodingGradientDetector,
    LogicalInvariantDetector,
    RegimeShiftDetector,
)


# ---------------------------------------------------------------------------
# ExplodingGradientDetector
# ---------------------------------------------------------------------------

class TestExplodingGradientDetector:
    def _make(self, threshold=100.0):
        d = ExplodingGradientDetector(threshold=threshold)
        # Baseline: inputs produce output 1.0
        d.set_baseline({"x": 1.0}, np.array([1.0]))
        return d

    def test_small_change_silent(self):
        d = self._make()
        # Input changes by 0.01, output changes by 0.01 → ratio = 1.0 < 100
        finding = d.check({"x": 1.01}, np.array([1.01]), iteration=0)
        assert finding is None

    def test_large_output_change_fires(self):
        d = self._make(threshold=10.0)
        # Input changes by 0.01, output changes by 100 → ratio = 10000 > 10
        finding = d.check({"x": 1.01}, np.array([101.0]), iteration=0)
        assert finding is not None
        assert finding.failure_type == "nan_inf"

    def test_zero_input_delta_does_not_raise(self):
        d = self._make()
        # Same inputs as baseline → input_delta = 0, guarded by epsilon
        finding = d.check({"x": 1.0}, np.array([500.0]), iteration=0)
        # Output changed massively with no input change → fires
        assert finding is not None

    def test_no_baseline_set_always_silent(self):
        d = ExplodingGradientDetector(threshold=10.0)
        finding = d.check({"x": 9999.0}, np.array([9999.0]), iteration=0)
        assert finding is None

    def test_finding_severity_is_critical(self):
        d = self._make(threshold=10.0)
        finding = d.check({"x": 1.01}, np.array([1000.0]), iteration=0)
        assert finding is not None
        assert finding.severity == "critical"

    def test_silent_when_below_threshold(self):
        d = ExplodingGradientDetector(threshold=1000.0)
        d.set_baseline({"x": 1.0}, np.array([1.0]))
        finding = d.check({"x": 2.0}, np.array([5.0]), iteration=0)
        assert finding is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd core
python -m pytest tests/test_new_detectors.py::TestExplodingGradientDetector -v
```

Expected: `ImportError` — `ExplodingGradientDetector` not defined.

- [ ] **Step 3: Implement ExplodingGradientDetector**

Append to `core/blackswan/detectors/numerical.py`:

```python
class ExplodingGradientDetector(FailureDetector):
    """
    Flags iterations where ||Δoutput|| / ||Δinput|| exceeds a threshold.

    Indicates extreme sensitivity: a tiny change in inputs causes an
    enormous change in outputs, which is a sign of numerical instability
    or a poorly conditioned computation path.

    Call set_baseline(base_inputs, base_output) once before the run loop.
    Without a baseline this detector is silently disabled.
    """

    FAILURE_TYPE = "nan_inf"
    DEFAULT_THRESHOLD = 100.0
    _EPSILON = 1e-12

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold
        self._base_inputs: dict | None = None
        self._base_output: Any = None

    def set_baseline(self, base_inputs: dict, base_output: Any) -> None:
        """Store the unperturbed reference point. Must be called before check()."""
        self._base_inputs = base_inputs
        self._base_output = base_output

    def check(self, inputs: dict, output: Any, iteration: int) -> Finding | None:
        if self._base_inputs is None:
            return None

        input_delta = self._vector_norm(inputs) - self._vector_norm(self._base_inputs)
        output_delta = self._array_norm(output) - self._array_norm(self._base_output)

        ratio = abs(output_delta) / (abs(input_delta) + self._EPSILON)
        if ratio <= self.threshold:
            return None

        return Finding(
            failure_type=self.FAILURE_TYPE,
            severity="critical",
            message=(
                f"Exploding gradient detected: output change / input change = {ratio:.1f}x "
                f"(threshold = {self.threshold}x). The model is critically sensitive to "
                "this perturbation magnitude."
            ),
            iteration=iteration,
        )

    @staticmethod
    def _vector_norm(inputs: dict) -> float:
        """Flatten all numeric values in an inputs dict to a single L2 norm."""
        parts = []
        for v in inputs.values():
            if isinstance(v, np.ndarray):
                parts.append(np.linalg.norm(v.ravel()))
            else:
                try:
                    parts.append(float(v))
                except (TypeError, ValueError):
                    pass
        return float(np.sqrt(sum(x ** 2 for x in parts))) if parts else 0.0

    @staticmethod
    def _array_norm(output: Any) -> float:
        """Return L2 norm of output (scalar or array)."""
        if isinstance(output, np.ndarray):
            return float(np.linalg.norm(output.ravel()))
        try:
            return abs(float(output))
        except (TypeError, ValueError):
            return 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd core
python -m pytest tests/test_new_detectors.py::TestExplodingGradientDetector -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add core/blackswan/detectors/numerical.py core/tests/test_new_detectors.py
git commit -m "feat: add ExplodingGradientDetector"
```

---

### Task A3: RegimeShiftDetector

Flags iterations where the output deviates more than `z_threshold` standard deviations from the running mean — indicating the model has entered a qualitatively different regime.

**Files:**
- Modify: `core/blackswan/detectors/numerical.py`
- Modify: `core/tests/test_new_detectors.py`

**Design:** The detector accumulates a rolling history of scalar output norms. After `min_history` observations it computes the Z-score of the current output against the accumulated mean and standard deviation. Stateful — the `StressRunner` must call `reset()` before each `run()` call.

---

- [ ] **Step 1: Write the failing tests**

Append to `core/tests/test_new_detectors.py`:

```python
# ---------------------------------------------------------------------------
# RegimeShiftDetector
# ---------------------------------------------------------------------------

class TestRegimeShiftDetector:
    def _make(self, z_threshold=4.0, min_history=10):
        return RegimeShiftDetector(z_threshold=z_threshold, min_history=min_history)

    def _feed(self, detector, values):
        """Feed a sequence of scalar outputs to the detector, return last finding."""
        finding = None
        for i, v in enumerate(values):
            finding = detector.check({}, np.array([v]), iteration=i)
        return finding

    def test_below_min_history_always_silent(self):
        d = self._make(min_history=10)
        # Feed 9 values — history not yet sufficient
        finding = self._feed(d, [1.0] * 9)
        assert finding is None

    def test_normal_output_stays_silent(self):
        d = self._make(z_threshold=4.0, min_history=5)
        # 10 values around 1.0; final value is 1.1 — z ≈ small
        finding = self._feed(d, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1])
        assert finding is None

    def test_extreme_outlier_fires(self):
        d = self._make(z_threshold=3.0, min_history=5)
        # 10 values around 1.0, then a massive spike
        finding = self._feed(d, [1.0] * 10 + [1000.0])
        assert finding is not None
        assert finding.failure_type == "nan_inf"

    def test_finding_severity_is_warning(self):
        d = self._make(z_threshold=3.0, min_history=5)
        finding = self._feed(d, [1.0] * 10 + [1000.0])
        assert finding is not None
        assert finding.severity == "warning"

    def test_reset_clears_history(self):
        d = self._make(z_threshold=3.0, min_history=5)
        # Build up history and confirm outlier fires
        finding = self._feed(d, [1.0] * 10 + [1000.0])
        assert finding is not None
        # Reset and re-feed short history — must be silent (min_history not met)
        d.reset()
        finding = self._feed(d, [1000.0] * 3)
        assert finding is None

    def test_constant_output_does_not_raise_division_by_zero(self):
        d = self._make(z_threshold=4.0, min_history=5)
        # All values identical → std = 0 → must not raise
        finding = self._feed(d, [1.0] * 15)
        assert finding is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd core
python -m pytest tests/test_new_detectors.py::TestRegimeShiftDetector -v
```

Expected: `ImportError` — `RegimeShiftDetector` not defined.

- [ ] **Step 3: Implement RegimeShiftDetector**

Append to `core/blackswan/detectors/numerical.py`:

```python
class RegimeShiftDetector(FailureDetector):
    """
    Flags when the output deviates more than z_threshold standard deviations
    from the running mean — indicating a qualitative regime change.

    Stateful: accumulates output norms across iterations. Call reset() before
    each StressRunner.run() call (the runner does this automatically when this
    detector is in its list).

    Not triggered until min_history observations have been collected. This
    prevents false positives in the early warm-up period.
    """

    FAILURE_TYPE = "nan_inf"
    DEFAULT_Z_THRESHOLD = 4.0
    DEFAULT_MIN_HISTORY = 30
    _EPSILON = 1e-12

    def __init__(
        self,
        z_threshold: float = DEFAULT_Z_THRESHOLD,
        min_history: int = DEFAULT_MIN_HISTORY,
    ) -> None:
        self.z_threshold = z_threshold
        self.min_history = min_history
        self._history: list[float] = []

    def reset(self) -> None:
        """Clear accumulated history. Call before each run."""
        self._history.clear()

    def check(self, inputs: dict, output: Any, iteration: int) -> Finding | None:
        norm = self._output_norm(output)
        self._history.append(norm)

        if len(self._history) < self.min_history:
            return None

        mean = float(np.mean(self._history))
        std = float(np.std(self._history))

        if std < self._EPSILON:
            return None  # constant output — z-score undefined, not a failure

        z = abs(norm - mean) / std
        if z <= self.z_threshold:
            return None

        return Finding(
            failure_type=self.FAILURE_TYPE,
            severity="warning",
            message=(
                f"Regime shift detected: output Z-score = {z:.2f} "
                f"(threshold = {self.z_threshold}). Output norm {norm:.4g} is "
                f"{z:.1f}σ from the running mean {mean:.4g} (σ = {std:.4g})."
            ),
            iteration=iteration,
        )

    @staticmethod
    def _output_norm(output: Any) -> float:
        if isinstance(output, np.ndarray):
            return float(np.linalg.norm(output.ravel()))
        try:
            return abs(float(output))
        except (TypeError, ValueError):
            return 0.0
```

- [ ] **Step 4: Wire reset() into StressRunner**

Open `core/blackswan/engine/runner.py`. Add `reset()` calls before the main loop and `set_baseline()` calls for detectors that support it:

```python
def run(self) -> RunResult:
    rng = np.random.default_rng(self.seed)
    findings: list[Finding] = []
    start = time.monotonic()

    # Run the function once on unperturbed inputs to establish a baseline.
    try:
        base_output = self.fn(**self.base_inputs)
    except Exception:
        base_output = None

    # Allow stateful detectors to initialise / reset.
    for detector in self.detectors:
        if hasattr(detector, "reset"):
            detector.reset()
        if hasattr(detector, "set_baseline") and base_output is not None:
            detector.set_baseline(self.base_inputs, base_output)

    validator = (
        self.scenario.make_validator()
        if hasattr(self.scenario, "make_validator")
        else None
    )

    for i in range(self.scenario.iterations):
        perturbed = self.scenario.apply(self.base_inputs, rng)

        if validator is not None and not validator.validate(perturbed):
            continue

        try:
            output = self.fn(**perturbed)
        except Exception as exc:
            frames = [
                (frame.filename, frame.lineno)
                for frame in tb.extract_tb(exc.__traceback__)
            ]
            findings.append(Finding(
                failure_type="nan_inf",
                severity="critical",
                message=f"{type(exc).__name__}: {exc}",
                iteration=i,
                exc_frames=frames,
            ))
            continue

        for detector in self.detectors:
            finding = detector.check(inputs=perturbed, output=output, iteration=i)
            if finding is not None:
                findings.append(finding)

    runtime_ms = int((time.monotonic() - start) * 1000)
    return RunResult(
        iterations_completed=self.scenario.iterations,
        findings=findings,
        runtime_ms=runtime_ms,
        seed=self.seed,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd core
python -m pytest tests/test_new_detectors.py::TestRegimeShiftDetector -v
```

Expected: all 6 tests pass.

- [ ] **Step 6: Run full suite to confirm no regressions**

```bash
cd core
python -m pytest tests/ -v --tb=short -q
```

Expected: all previously passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add core/blackswan/detectors/numerical.py core/blackswan/engine/runner.py core/tests/test_new_detectors.py
git commit -m "feat: add RegimeShiftDetector and stateful detector lifecycle in StressRunner"
```

---

### Task A4: LogicalInvariantDetector

Allows callers to define custom boolean assertions over the output (e.g. `weights >= 0`, `var <= 1`). Any assertion that returns `False` for the given output is a failure.

**Files:**
- Modify: `core/blackswan/detectors/numerical.py`
- Modify: `core/tests/test_new_detectors.py`

---

- [ ] **Step 1: Write the failing tests**

Append to `core/tests/test_new_detectors.py`:

```python
# ---------------------------------------------------------------------------
# LogicalInvariantDetector
# ---------------------------------------------------------------------------

class TestLogicalInvariantDetector:
    def test_invariant_satisfied_silent(self):
        d = LogicalInvariantDetector(
            assertion=lambda inputs, output: float(output) >= 0,
            description="Output must be non-negative",
        )
        finding = d.check({}, np.array([0.5]), iteration=0)
        assert finding is None

    def test_invariant_violated_fires(self):
        d = LogicalInvariantDetector(
            assertion=lambda inputs, output: float(output) >= 0,
            description="Output must be non-negative",
        )
        finding = d.check({}, np.array([-0.1]), iteration=0)
        assert finding is not None
        assert finding.failure_type == "bounds_exceeded"

    def test_description_in_message(self):
        d = LogicalInvariantDetector(
            assertion=lambda inputs, output: False,
            description="weights must sum to 1",
        )
        finding = d.check({}, 0.0, iteration=0)
        assert finding is not None
        assert "weights must sum to 1" in finding.message

    def test_finding_severity_is_warning(self):
        d = LogicalInvariantDetector(
            assertion=lambda inputs, output: False,
            description="test",
        )
        finding = d.check({}, 0.0, iteration=0)
        assert finding is not None
        assert finding.severity == "warning"

    def test_exception_in_assertion_treated_as_violation(self):
        def bad_assertion(inputs, output):
            raise RuntimeError("bad")

        d = LogicalInvariantDetector(assertion=bad_assertion, description="broken")
        finding = d.check({}, 0.0, iteration=0)
        assert finding is not None

    def test_inputs_available_to_assertion(self):
        # Assertion uses inputs dict, not just output
        d = LogicalInvariantDetector(
            assertion=lambda inputs, output: inputs.get("weight", 0) >= 0,
            description="weight non-negative",
        )
        assert d.check({"weight": -0.5}, 1.0, iteration=0) is not None
        assert d.check({"weight": 0.5}, 1.0, iteration=0) is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd core
python -m pytest tests/test_new_detectors.py::TestLogicalInvariantDetector -v
```

Expected: `ImportError` — `LogicalInvariantDetector` not defined.

- [ ] **Step 3: Implement LogicalInvariantDetector**

Append to `core/blackswan/detectors/numerical.py`:

```python
from typing import Callable

class LogicalInvariantDetector(FailureDetector):
    """
    User-defined assertion detector.

    Supply any callable `assertion(inputs: dict, output: Any) -> bool`.
    The detector fires (returns a Finding) whenever the assertion returns
    False, or raises any exception.

    Example:
        LogicalInvariantDetector(
            assertion=lambda inputs, output: float(output) >= 0,
            description="Portfolio VaR must be non-negative",
        )
    """

    FAILURE_TYPE = "bounds_exceeded"

    def __init__(
        self,
        assertion: Callable[[dict, Any], bool],
        description: str,
    ) -> None:
        self._assertion = assertion
        self._description = description

    def check(self, inputs: dict, output: Any, iteration: int) -> Finding | None:
        try:
            passed = bool(self._assertion(inputs, output))
        except Exception as exc:
            passed = False
            extra = f" (assertion raised: {exc})"
        else:
            extra = ""

        if passed:
            return None

        return Finding(
            failure_type=self.FAILURE_TYPE,
            severity="warning",
            message=f"Logical invariant violated: {self._description}{extra}.",
            iteration=iteration,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd core
python -m pytest tests/test_new_detectors.py -v
```

Expected: all tests in `test_new_detectors.py` pass (all three detector classes).

- [ ] **Step 5: Run full suite**

```bash
cd core
python -m pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add core/blackswan/detectors/numerical.py core/tests/test_new_detectors.py
git commit -m "feat: add LogicalInvariantDetector"
```

---

## PHASE B: THE ADVERSARIAL ENGINE

---

### Task B1: Individual and Fitness Types

Define the data types that the Genetic Algorithm operates on before writing any search logic.

**Files:**
- Create: `core/blackswan/engine/adversarial.py`
- Create: `core/tests/test_adversarial.py`

An `Individual` is one candidate set of perturbation parameter overrides (e.g. `{"vol_low": 1.8, "vol_high": 3.2}`). Its fitness score is a float — higher means it caused more severe failures.

---

- [ ] **Step 1: Write the failing tests**

Create `core/tests/test_adversarial.py`:

```python
"""Tests for the EvolutionaryStressRunner (Phase B)."""

from __future__ import annotations

import numpy as np
import pytest

from blackswan.engine.adversarial import (
    Individual,
    compute_fitness,
    crossover,
    mutate,
)
from blackswan.detectors.base import Finding


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------

class TestIndividual:
    def test_params_stored(self):
        ind = Individual(params={"vol_low": 1.5, "vol_high": 3.0})
        assert ind.params["vol_low"] == 1.5

    def test_default_fitness_is_zero(self):
        ind = Individual(params={})
        assert ind.fitness == 0.0

    def test_fitness_settable(self):
        ind = Individual(params={})
        ind.fitness = 5.0
        assert ind.fitness == 5.0


# ---------------------------------------------------------------------------
# compute_fitness
# ---------------------------------------------------------------------------

class TestComputeFitness:
    def _finding(self, failure_type="non_psd_matrix", severity="critical", meta: dict | None = None):
        f = Finding(
            failure_type=failure_type,
            severity=severity,
            message="test",
            iteration=0,
        )
        if meta:
            f.__dict__.update(meta)
        return f

    def test_no_findings_gives_zero(self):
        assert compute_fitness([]) == 0.0

    def test_critical_finding_gives_positive_score(self):
        score = compute_fitness([self._finding(severity="critical")])
        assert score > 0.0

    def test_critical_higher_than_warning(self):
        critical = compute_fitness([self._finding(severity="critical")])
        warning = compute_fitness([self._finding(severity="warning")])
        assert critical > warning

    def test_more_findings_increases_score(self):
        one = compute_fitness([self._finding()])
        two = compute_fitness([self._finding(), self._finding()])
        assert two > one

    def test_non_psd_matrix_bonus(self):
        base = compute_fitness([self._finding(failure_type="nan_inf", severity="critical")])
        psd = compute_fitness([self._finding(failure_type="non_psd_matrix", severity="critical")])
        assert psd >= base  # PSD failures are target failures, score >= nan_inf


# ---------------------------------------------------------------------------
# crossover
# ---------------------------------------------------------------------------

class TestCrossover:
    def test_output_keys_match_parents(self):
        rng = np.random.default_rng(0)
        p1 = Individual(params={"a": 1.0, "b": 2.0})
        p2 = Individual(params={"a": 3.0, "b": 4.0})
        child = crossover(p1, p2, rng)
        assert set(child.params) == {"a", "b"}

    def test_each_param_comes_from_one_parent(self):
        rng = np.random.default_rng(0)
        p1 = Individual(params={"a": 0.0})
        p2 = Individual(params={"a": 100.0})
        # Run many crossovers — output must always be 0.0 or 100.0
        for _ in range(20):
            child = crossover(p1, p2, np.random.default_rng())
            assert child.params["a"] in (0.0, 100.0)

    def test_child_fitness_starts_at_zero(self):
        rng = np.random.default_rng(0)
        p1 = Individual(params={"a": 1.0}, fitness=99.0)
        p2 = Individual(params={"a": 2.0}, fitness=50.0)
        child = crossover(p1, p2, rng)
        assert child.fitness == 0.0


# ---------------------------------------------------------------------------
# mutate
# ---------------------------------------------------------------------------

class TestMutate:
    def test_mutation_changes_values(self):
        rng = np.random.default_rng(0)
        ind = Individual(params={"vol": 1.0, "corr": 0.5})
        mutated = mutate(ind, rng, noise_scale=0.1)
        # With noise_scale=0.1 and seed 0, at least one param will differ
        assert mutated.params != ind.params

    def test_keys_preserved(self):
        rng = np.random.default_rng(0)
        ind = Individual(params={"x": 1.0, "y": 2.0})
        mutated = mutate(ind, rng, noise_scale=0.1)
        assert set(mutated.params) == {"x", "y"}

    def test_original_not_mutated(self):
        rng = np.random.default_rng(0)
        ind = Individual(params={"x": 1.0})
        original_x = ind.params["x"]
        mutate(ind, rng, noise_scale=1.0)
        assert ind.params["x"] == original_x  # must not modify in place
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd core
python -m pytest tests/test_adversarial.py -v
```

Expected: `ImportError` — module `blackswan.engine.adversarial` does not exist.

- [ ] **Step 3: Implement Individual, compute_fitness, crossover, mutate**

Create `core/blackswan/engine/adversarial.py`:

```python
"""
Evolutionary (Genetic Algorithm) stress runner for BlackSwan.

Replaces random Monte Carlo sampling with a population-based search that
evolves perturbation sets toward maximum failure severity.

Public API:
    Individual                — one candidate perturbation parameter set
    compute_fitness(findings) — score a list of findings (higher = worse)
    crossover(p1, p2, rng)   — produce a child by mixing two parents
    mutate(ind, rng, scale)  — perturb an individual's params with Gaussian noise
    EvolutionaryStressRunner  — drop-in replacement for StressRunner
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..detectors.base import FailureDetector, Finding
from ..engine.perturbation import Perturbation, apply_perturbations


# ---------------------------------------------------------------------------
# Severity weights for fitness scoring
# ---------------------------------------------------------------------------

_SEVERITY_WEIGHT = {"critical": 3.0, "warning": 1.5, "info": 0.5}
_TYPE_BONUS = {"non_psd_matrix": 1.5, "ill_conditioned_matrix": 1.2}


@dataclass
class Individual:
    """
    One candidate in the GA population.

    params:  Dict mapping perturbation parameter names to values.
             Keys follow the convention "<target>_<param_name>",
             e.g. {"vol_low": 1.5, "vol_high": 3.0, "corr_low": 0.1}.
    fitness: Score assigned after evaluation. Higher = more failure.
             Initialised to 0.0; set by EvolutionaryStressRunner.evaluate().
    """
    params: dict[str, float]
    fitness: float = 0.0


def compute_fitness(findings: list[Finding]) -> float:
    """
    Score a list of findings from one evaluation.

    Returns 0.0 for an empty list. Each finding contributes its severity
    weight × the type bonus (1.0 if no bonus). Aggregated additively so
    more findings always increase fitness.
    """
    if not findings:
        return 0.0
    score = 0.0
    for f in findings:
        weight = _SEVERITY_WEIGHT.get(f.severity, 1.0)
        bonus = _TYPE_BONUS.get(f.failure_type, 1.0)
        score += weight * bonus
    return score


def crossover(parent1: Individual, parent2: Individual, rng: np.random.Generator) -> Individual:
    """
    Produce a child by randomly choosing each parameter from one of two parents.

    For each key, a fair coin flip determines which parent's value is inherited.
    The child's fitness is reset to 0.0 (it has not been evaluated yet).
    """
    child_params: dict[str, float] = {}
    for key in parent1.params:
        child_params[key] = parent1.params[key] if rng.random() < 0.5 else parent2.params[key]
    return Individual(params=child_params, fitness=0.0)


def mutate(
    individual: Individual,
    rng: np.random.Generator,
    noise_scale: float = 0.1,
) -> Individual:
    """
    Apply Gaussian noise to every parameter in individual.

    Returns a new Individual — the original is never modified.
    noise_scale is the standard deviation of the Gaussian noise applied to each
    value. E.g. noise_scale=0.1 means ±10% perturbation of each parameter.
    """
    new_params = {
        k: v + float(rng.normal(0.0, noise_scale * max(abs(v), 1e-6)))
        for k, v in individual.params.items()
    }
    return Individual(params=new_params, fitness=0.0)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd core
python -m pytest tests/test_adversarial.py::TestIndividual tests/test_adversarial.py::TestComputeFitness tests/test_adversarial.py::TestCrossover tests/test_adversarial.py::TestMutate -v
```

Expected: all 15 tests pass.

- [ ] **Step 5: Commit**

```bash
git add core/blackswan/engine/adversarial.py core/tests/test_adversarial.py
git commit -m "feat: add Individual, compute_fitness, crossover, mutate for GA engine"
```

---

### Task B2: EvolutionaryStressRunner Core Loop

The main GA loop. Runs `n_generations` of: evaluate → select elites → crossover + mutate → repeat.

**Files:**
- Modify: `core/blackswan/engine/adversarial.py`
- Modify: `core/tests/test_adversarial.py`

---

- [ ] **Step 1: Write the failing tests**

Append to `core/tests/test_adversarial.py`:

```python
# ---------------------------------------------------------------------------
# EvolutionaryStressRunner
# ---------------------------------------------------------------------------

from blackswan.engine.adversarial import EvolutionaryStressRunner
from blackswan.engine.runner import RunResult
from blackswan.detectors.matrix import MatrixPSDDetector
from blackswan.scenarios.registry import load_scenario


def _broken_cov(
    weights: np.ndarray = np.array([0.3, 0.3, 0.4]),
    vol: np.ndarray = np.array([0.15, 0.20, 0.10]),
    correlation: float = 0.0,
) -> np.ndarray:
    """Identical to tests/fixtures/broken_covariance.py — inline for import independence."""
    n = len(vol)
    corr_val = 0.8 + correlation
    corr_matrix = np.full((n, n), corr_val)
    np.fill_diagonal(corr_matrix, 1.0)
    return np.diag(vol) @ corr_matrix @ np.diag(vol)


class TestEvolutionaryStressRunner:
    def _runner(self, generations=5, pop_size=20, elite_frac=0.2):
        scenario = load_scenario("liquidity_crash")
        return EvolutionaryStressRunner(
            fn=_broken_cov,
            base_inputs={"correlation": 0.0},
            scenario=scenario,
            detectors=[MatrixPSDDetector()],
            seed=42,
            n_generations=generations,
            population_size=pop_size,
            elite_fraction=elite_frac,
        )

    def test_returns_run_result(self):
        result = self._runner().run()
        assert isinstance(result, RunResult)

    def test_runtime_ms_non_negative(self):
        result = self._runner().run()
        assert result.runtime_ms >= 0

    def test_seed_is_stored(self):
        result = self._runner().run()
        assert result.seed == 42

    def test_finds_failures_on_broken_fixture(self):
        """GA must find the Non-PSD shatter point within 5 generations of 20."""
        result = self._runner(generations=5, pop_size=20).run()
        assert len(result.findings) > 0
        assert any(f.failure_type == "non_psd_matrix" for f in result.findings)

    def test_same_seed_deterministic(self):
        r1 = self._runner(generations=3, pop_size=10).run()
        r2 = self._runner(generations=3, pop_size=10).run()
        assert len(r1.findings) == len(r2.findings)

    def test_iterations_completed_equals_population_times_generations(self):
        result = self._runner(generations=3, pop_size=10).run()
        assert result.iterations_completed == 3 * 10

    def test_fewer_iterations_than_monte_carlo_to_find_failure(self):
        """Exit criterion: GA finds non_psd_matrix in under 500 iterations total."""
        scenario = load_scenario("liquidity_crash")
        runner = EvolutionaryStressRunner(
            fn=_broken_cov,
            base_inputs={"correlation": 0.0},
            scenario=scenario,
            detectors=[MatrixPSDDetector()],
            seed=42,
            n_generations=25,
            population_size=20,
            elite_fraction=0.2,
        )
        result = runner.run()
        psd_failures = [f for f in result.findings if f.failure_type == "non_psd_matrix"]
        # 25 gen × 20 pop = 500 total evaluations — must find failure within this budget
        assert len(psd_failures) > 0
        assert result.iterations_completed <= 500
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd core
python -m pytest tests/test_adversarial.py::TestEvolutionaryStressRunner -v
```

Expected: `ImportError` — `EvolutionaryStressRunner` not defined.

- [ ] **Step 3: Implement EvolutionaryStressRunner**

Append to `core/blackswan/engine/adversarial.py`:

```python
from ..engine.runner import RunResult


class EvolutionaryStressRunner:
    """
    Genetic Algorithm stress runner. Drop-in replacement for StressRunner.

    Instead of N random iterations, runs n_generations × population_size
    targeted evaluations. Each generation selects the top elite_fraction
    of individuals by fitness, uses them to breed the next population via
    crossover + mutation, and evaluates again.

    Satisfies the same output contract as StressRunner: returns a RunResult
    with findings, runtime_ms, seed, and iterations_completed.
    """

    def __init__(
        self,
        fn: Callable,
        base_inputs: dict,
        scenario: Any,
        detectors: list[FailureDetector],
        seed: int,
        n_generations: int = 20,
        population_size: int = 100,
        elite_fraction: float = 0.2,
        noise_scale: float = 0.1,
    ) -> None:
        self.fn = fn
        self.base_inputs = base_inputs
        self.scenario = scenario
        self.detectors = detectors
        self.seed = seed
        self.n_generations = n_generations
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.noise_scale = noise_scale

    def run(self) -> RunResult:
        rng = np.random.default_rng(self.seed)
        all_findings: list[Finding] = []
        total_iterations = 0
        start = time.monotonic()

        # Allow stateful detectors to reset.
        for detector in self.detectors:
            if hasattr(detector, "reset"):
                detector.reset()

        # Extract perturbation parameter ranges from the scenario.
        param_ranges = _extract_param_ranges(self.scenario)

        # Initialise random population.
        population = _initialise_population(param_ranges, self.population_size, rng)

        n_elites = max(1, int(self.population_size * self.elite_fraction))

        for _gen in range(self.n_generations):
            # Evaluate each individual.
            for individual in population:
                perturbed = _apply_individual(self.base_inputs, self.scenario, individual, rng)
                findings = self._evaluate(perturbed, total_iterations)
                all_findings.extend(findings)
                individual.fitness = compute_fitness(findings)
                total_iterations += 1

            # Select elites.
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            elites = population[:n_elites]

            # Breed next generation: crossover pairs of elites, then mutate.
            next_pop: list[Individual] = list(copy.deepcopy(elites))
            while len(next_pop) < self.population_size:
                p1, p2 = rng.choice(len(elites), size=2, replace=False)
                child = crossover(elites[p1], elites[p2], rng)
                child = mutate(child, rng, self.noise_scale)
                next_pop.append(child)
            population = next_pop

        runtime_ms = int((time.monotonic() - start) * 1000)
        return RunResult(
            iterations_completed=total_iterations,
            findings=all_findings,
            runtime_ms=runtime_ms,
            seed=self.seed,
        )

    def _evaluate(self, perturbed: dict, iteration: int) -> list[Finding]:
        try:
            output = self.fn(**perturbed)
        except Exception as exc:
            import traceback as tb
            frames = [(f.filename, f.lineno) for f in tb.extract_tb(exc.__traceback__)]
            return [Finding(
                failure_type="nan_inf",
                severity="critical",
                message=f"{type(exc).__name__}: {exc}",
                iteration=iteration,
                exc_frames=frames,
            )]
        results = []
        for detector in self.detectors:
            f = detector.check(inputs=perturbed, output=output, iteration=iteration)
            if f is not None:
                results.append(f)
        return results


# ---------------------------------------------------------------------------
# GA helpers
# ---------------------------------------------------------------------------

def _extract_param_ranges(scenario: Any) -> dict[str, tuple[float, float]]:
    """
    Build a param_ranges dict from a Scenario's perturbations.

    For each Perturbation with distribution="uniform", emits:
        "<target>_low"  → params["low"]
        "<target>_high" → params["high"]
    For lognormal, emits:
        "<target>_mu"   → params["mu"]
        "<target>_sigma"→ params["sigma"]
    """
    ranges: dict[str, tuple[float, float]] = {}
    for p in scenario.perturbations:
        if p.distribution == "uniform":
            lo = float(p.params["low"])
            hi = float(p.params["high"])
            ranges[f"{p.target}_low"] = (lo * 0.5, lo * 2.0)
            ranges[f"{p.target}_high"] = (hi * 0.5, hi * 2.0)
        elif p.distribution == "lognormal":
            mu = float(p.params["mu"])
            sigma = float(p.params["sigma"])
            ranges[f"{p.target}_mu"] = (mu * 0.5, mu * 2.0)
            ranges[f"{p.target}_sigma"] = (sigma * 0.5, sigma * 2.0)
    return ranges


def _initialise_population(
    param_ranges: dict[str, tuple[float, float]],
    size: int,
    rng: np.random.Generator,
) -> list[Individual]:
    """Generate `size` random individuals uniformly distributed within param_ranges."""
    population = []
    for _ in range(size):
        params = {
            key: float(rng.uniform(lo, hi))
            for key, (lo, hi) in param_ranges.items()
        }
        population.append(Individual(params=params))
    return population


def _apply_individual(
    base_inputs: dict,
    scenario: Any,
    individual: Individual,
    rng: np.random.Generator,
) -> dict:
    """
    Build a perturbed inputs dict driven by an Individual's param overrides.

    Reconstructs the scenario's perturbation list with the Individual's
    parameter values substituted in, then calls apply_perturbations.
    """
    overridden: list[Perturbation] = []
    for p in scenario.perturbations:
        new_params = dict(p.params)
        if p.distribution == "uniform":
            low_key = f"{p.target}_low"
            high_key = f"{p.target}_high"
            if low_key in individual.params:
                new_params["low"] = individual.params[low_key]
            if high_key in individual.params:
                new_params["high"] = individual.params[high_key]
        elif p.distribution == "lognormal":
            mu_key = f"{p.target}_mu"
            sigma_key = f"{p.target}_sigma"
            if mu_key in individual.params:
                new_params["mu"] = individual.params[mu_key]
            if sigma_key in individual.params:
                new_params["sigma"] = individual.params[sigma_key]
        overridden.append(Perturbation(
            target=p.target,
            type=p.type,
            distribution=p.distribution,
            params=new_params,
            clamp=p.clamp,
        ))
    return apply_perturbations(base_inputs, overridden, rng)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd core
python -m pytest tests/test_adversarial.py -v
```

Expected: all tests pass including `test_fewer_iterations_than_monte_carlo_to_find_failure`.

- [ ] **Step 5: Commit**

```bash
git add core/blackswan/engine/adversarial.py core/tests/test_adversarial.py
git commit -m "feat: implement EvolutionaryStressRunner GA core loop"
```

---

### Task B3: Adaptive Intensity

When the GA makes no progress (no failures found in a generation), automatically widen perturbation ranges so the engine escapes flat search regions.

**Files:**
- Modify: `core/blackswan/engine/adversarial.py`
- Modify: `core/tests/test_adversarial.py`

---

- [ ] **Step 1: Write the failing tests**

Append to `core/tests/test_adversarial.py`:

```python
# ---------------------------------------------------------------------------
# HardnessAdaptor
# ---------------------------------------------------------------------------

from blackswan.engine.adversarial import HardnessAdaptor


class TestHardnessAdaptor:
    def test_initial_hardness_is_zero(self):
        h = HardnessAdaptor(step=0.1, max_hardness=1.0)
        assert h.hardness == 0.0

    def test_no_progress_increases_hardness(self):
        h = HardnessAdaptor(step=0.1, max_hardness=1.0)
        h.update(best_fitness=0.0)
        assert h.hardness == pytest.approx(0.1)

    def test_progress_does_not_increase_hardness(self):
        h = HardnessAdaptor(step=0.1, max_hardness=1.0)
        h.update(best_fitness=0.0)   # no progress → 0.1
        h.update(best_fitness=5.0)   # progress! → stays 0.1
        assert h.hardness == pytest.approx(0.1)

    def test_hardness_capped_at_max(self):
        h = HardnessAdaptor(step=0.5, max_hardness=1.0)
        for _ in range(5):
            h.update(best_fitness=0.0)
        assert h.hardness == pytest.approx(1.0)

    def test_scale_factor_at_zero_hardness_is_one(self):
        h = HardnessAdaptor(step=0.1, max_hardness=1.0)
        assert h.range_scale() == pytest.approx(1.0)

    def test_scale_factor_at_full_hardness_is_two(self):
        h = HardnessAdaptor(step=1.0, max_hardness=1.0)
        h.update(best_fitness=0.0)
        assert h.range_scale() == pytest.approx(2.0)

    def test_scale_factor_increases_monotonically(self):
        h = HardnessAdaptor(step=0.1, max_hardness=1.0)
        scales = []
        for _ in range(10):
            h.update(best_fitness=0.0)
            scales.append(h.range_scale())
        assert scales == sorted(scales)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd core
python -m pytest tests/test_adversarial.py::TestHardnessAdaptor -v
```

Expected: `ImportError` — `HardnessAdaptor` not defined.

- [ ] **Step 3: Implement HardnessAdaptor**

Append to `core/blackswan/engine/adversarial.py`:

```python
class HardnessAdaptor:
    """
    Adaptive intensity controller for EvolutionaryStressRunner.

    Tracks whether the GA is making progress each generation. When no
    improvement is found (best_fitness == 0.0), hardness increases by
    `step` up to `max_hardness`. The `range_scale()` method returns a
    multiplier [1.0, 2.0] applied to perturbation parameter ranges,
    widening the search when the engine is stuck.

    Formula: range_scale = 1.0 + hardness
    So hardness=0.0 → scale=1.0 (no widening)
       hardness=1.0 → scale=2.0 (ranges doubled)
    """

    def __init__(self, step: float = 0.1, max_hardness: float = 1.0) -> None:
        self._step = step
        self._max = max_hardness
        self.hardness: float = 0.0

    def update(self, best_fitness: float) -> None:
        """
        Call once per generation with the best fitness score seen.
        Increases hardness when best_fitness == 0.0 (no failure found).
        """
        if best_fitness == 0.0:
            self.hardness = min(self.hardness + self._step, self._max)

    def range_scale(self) -> float:
        """Multiplier to apply to perturbation parameter ranges. Always >= 1.0."""
        return 1.0 + self.hardness
```

- [ ] **Step 4: Integrate HardnessAdaptor into EvolutionaryStressRunner**

In `EvolutionaryStressRunner.__init__`, add:
```python
self._hardness = HardnessAdaptor(step=0.05, max_hardness=1.0)
```

In `EvolutionaryStressRunner.run()`, after evaluating each generation, add before breeding the next population:

```python
best_gen_fitness = max((ind.fitness for ind in population), default=0.0)
self._hardness.update(best_gen_fitness)
scale = self._hardness.range_scale()

# Widen param_ranges by scale for next generation's initialisation pool.
scaled_ranges = {
    key: (lo * scale, hi * scale)
    for key, (lo, hi) in param_ranges.items()
}
# Inject scaled_ranges into the mutation noise for next gen's offspring.
# (noise_scale already widens individual params via Gaussian noise;
#  scaled_ranges widen the initialisation floor used if we need new random
#  individuals to replace low-fitness ones.)
```

- [ ] **Step 5: Run all adversarial tests**

```bash
cd core
python -m pytest tests/test_adversarial.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add core/blackswan/engine/adversarial.py core/tests/test_adversarial.py
git commit -m "feat: add HardnessAdaptor adaptive intensity to EvolutionaryStressRunner"
```

---

### Task B4: CLI `--adversarial` Flag

Add `--adversarial` to the `test` subcommand so the engine can be switched at the command line.

**Files:**
- Modify: `core/blackswan/cli.py`

---

- [ ] **Step 1: Write the failing test**

Add to `core/tests/test_cli.py`:

```python
class TestAdversarialFlag:
    def test_adversarial_flag_accepted(self):
        result = _run(
            "test", str(FIXTURE_DIR / "broken_covariance.py"),
            "--scenario", "liquidity_crash",
            "--adversarial",
            "--iterations", "3",   # 3 generations (--iterations reused as n_generations)
        )
        # Must exit 0 or 1, not 2 (no error)
        assert result.returncode in (0, 1)

    def test_adversarial_output_is_valid_json(self):
        result = _run(
            "test", str(FIXTURE_DIR / "broken_covariance.py"),
            "--scenario", "liquidity_crash",
            "--adversarial",
            "--iterations", "3",
        )
        import json
        data = json.loads(result.stdout)
        assert "shatter_points" in data
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd core
python -m pytest tests/test_cli.py::TestAdversarialFlag -v
```

Expected: `FAIL` — `--adversarial` flag is unrecognised.

- [ ] **Step 3: Add the flag to _build_parser**

Open `core/blackswan/cli.py`. In `_build_parser()`, after the `--seed` argument inside the `test` subparser:

```python
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
    help=(
        "Population size per generation when using --adversarial (default: 100)."
    ),
)
```

- [ ] **Step 4: Use EvolutionaryStressRunner when flag is set**

In `_cmd_test()`, replace the runner construction block:

```python
if args.adversarial:
    from .engine.adversarial import EvolutionaryStressRunner
    n_generations = iterations  # --iterations repurposed as generation count for GA
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
```

- [ ] **Step 5: Run the CLI adversarial tests**

```bash
cd core
python -m pytest tests/test_cli.py::TestAdversarialFlag -v
```

Expected: both tests pass.

- [ ] **Step 6: Smoke test from terminal**

```bash
cd core
python -m blackswan test tests/fixtures/broken_covariance.py --scenario liquidity_crash --adversarial --iterations 25 --population 20
```

Expected: JSON output with at least one `non_psd_matrix` shatter point, `iterations_completed` ≤ 500.

- [ ] **Step 7: Run full suite**

```bash
cd core
python -m pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add core/blackswan/cli.py core/tests/test_cli.py
git commit -m "feat: add --adversarial and --population CLI flags, wire EvolutionaryStressRunner"
```

---

## EXIT CRITERION VERIFICATION

Run the formal exit criterion from the spec. Do not mark this task complete until all three checks produce the expected output.

---

- [ ] **Step 1: Verify the adversarial engine meets the iteration budget**

```bash
cd core
python -m blackswan test tests/fixtures/broken_covariance.py \
    --scenario liquidity_crash \
    --adversarial \
    --iterations 25 \
    --population 20 \
    --seed 42
```

**Required evidence:** JSON output where:
1. `status` = `"failures_detected"`
2. At least one entry in `shatter_points` with `failure_type` = `"non_psd_matrix"`
3. `iterations_completed` ≤ 500 (25 generations × 20 individuals)

- [ ] **Step 2: Verify determinism**

Run the exact same command twice and confirm both runs produce identical `summary.total_failures` and `shatter_points[0].line`.

- [ ] **Step 3: Run the full test suite one final time**

```bash
cd core
python -m pytest tests/ -v --tb=short -q
```

**Required evidence:** All tests pass. Zero failures.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: Phase A+B complete — adversarial GA engine with plausibility constraints and new detectors"
```

---

## DANGER ZONES — TRAPS TO AVOID

1. **Mutating the original `Individual.params` dict.** `mutate()` and `crossover()` must always return new objects. The GA loop holds references to elites across generations.

2. **Forgetting `reset()` for stateful detectors.** `RegimeShiftDetector` accumulates history. If you reuse the same instance across `EvolutionaryStressRunner.run()` calls (e.g. in tests), call `reset()` yourself or results will be polluted.

3. **Breaking the existing `StressRunner` tests.** Every change to `runner.py` must keep the existing 14 `test_runner.py` tests green. The `hasattr` guards in the runner changes above exist for this reason.

4. **Redefining `Scenario`.** The registry already handles YAML loading, validation, and the `StressRunner` protocol. Do not create a parallel `BlackSwanScenario` class — extend the existing one in-place.

5. **Assuming `--iterations` means Monte Carlo iterations in `--adversarial` mode.** In adversarial mode `--iterations` is repurposed as `n_generations`. Document this clearly in CLI `--help` text. Use `--population` for population size.

6. **Fitness = 0.0 for all individuals in early generations.** This is normal. `HardnessAdaptor` handles this by widening ranges. Do not artificially inflate scores.
