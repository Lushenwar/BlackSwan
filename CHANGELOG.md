# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.2] — 2026-04-09

### Added

- **Adversarial stress engine** (`blackswan/engine/adversarial.py`): genetic-algorithm-based runner (`EvolutionaryStressRunner`) that actively evolves stress parameters to maximise failure fitness, rather than sampling randomly. Exposes the same `RunResult` interface as `StressRunner`.
- **`HardnessAdaptor`**: automatically scales perturbation intensity when the GA finds no failures, preventing the search from stalling on robust code.
- **`ExplodingGradientDetector`**: detects runaway output growth relative to input perturbation magnitude (ratio > 100× default threshold).
- **`RegimeShiftDetector`**: stateful detector that flags structural breaks in output distributions across iterations via z-score analysis.
- **`LogicalInvariantDetector`**: user-supplied callable assertion checked on every iteration — e.g., `lambda inputs, out: out.sum() <= 1.0` for weight constraints.
- **`PlausibilityValidator`** (`blackswan/engine/validator.py`): validates perturbed inputs against configurable min/max constraints before each function call, skipping physically impossible scenarios.
- **`global_constraints` in scenario YAML**: per-scenario input validation bounds, e.g. `correlation_shift: {min: -1.0, max: 0.2}`.
- **`--adversarial` CLI flag**: `python -m blackswan test model.py --scenario liquidity_crash --adversarial` switches to evolutionary search.
- **`--population` CLI flag**: control GA population size (default 100).
- **`baseline_established`** field in `RunResult`: signals whether the baseline function call succeeded before perturbation began.

### Changed

- `RunResult.iterations_completed` now counts only *executed* iterations (validator-skipped iterations excluded).
- Version bumped 0.1.1 → 0.1.2 to reflect the new engine capabilities.
- `pyproject.toml`: added `Documentation` and `Changelog` URLs; added `genetic-algorithm`, `adversarial-testing`, `portfolio-risk` keywords; added `Topic :: Software Development :: Testing` and `Typing :: Typed` classifiers.

### Fixed

- `StressRunner` now records `baseline_established: False` when the baseline function call raises, rather than silently swallowing the exception.

---

## [0.1.1] — 2026-03-15

### Fixed

- Removed unsupported GitHub Packages publishing step from CI workflow.

---

## [0.1.0] — 2026-03-10

### Added

- Initial release of `blackswan-core`.
- Five failure detectors: `NaNInfDetector`, `DivisionStabilityDetector`, `MatrixPSDDetector`, `ConditionNumberDetector`, `BoundsDetector`.
- Monte Carlo `StressRunner` with deterministic seeding.
- Five preset stress scenarios: `liquidity_crash`, `vol_spike`, `correlation_breakdown`, `rate_shock`, `missing_data`.
- YAML-driven perturbation system with uniform and lognormal distributions.
- AST-based dependency graph construction and auto-detector tagging.
- Traceback attribution with proximate failure site, causal chain, and root cause ranking.
- CLI: `python -m blackswan test <file> --scenario <name>`.
- Versioned JSON contract (`contract/schema.json`) validated on every response.
- VS Code extension with CodeLens, Diagnostics, HoverProvider, progress bar, and DAG webview panel.

[0.1.2]: https://github.com/Lushenwar/BlackSwan/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Lushenwar/BlackSwan/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Lushenwar/BlackSwan/releases/tag/v0.1.0
