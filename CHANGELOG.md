# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Extension 0.4.0 / Engine 0.4.0] — 2026-04-12

### Added

- **Hybrid Auto-Fixer** — every red squiggle now has three Quick Fix CodeActions:
  - **Apply Mathematical Guard**: rewrites the failing line using a deterministic libcst guard; shows a side-by-side diff before applying; fully undoable via VS Code's undo stack.
  - **Explain with BlackSwan AI**: sends failure metadata (never source code) to Gemini Flash and opens a plain-English explanation in a side panel.
  - **Insert comment hint**: adds an indented `# BlackSwan Fix Hint:` comment inline — no subprocess, no API.
- **`blackswan.fixer` Python module** (`core/blackswan/fixer/guards.py`) — libcst-based CST rewriter implementing four guard patterns:
  - `division_instability` → `max(denominator, 1e-10)` epsilon clamp
  - `non_psd_matrix` → Higham 2002 nearest-PSD correction via `eigh` + `np.maximum`
  - `ill_conditioned_matrix` → conditional `np.linalg.pinv` fallback when `cond > 1e12`
  - `nan_inf` → `np.nan_to_num(result, posinf=…, neginf=…)` guard
- **`python -m blackswan fix` CLI subcommand** — generates a JSON fix proposal for any line/type combination. Available standalone without VS Code. Requires `pip install blackswan[fixer]`.
- **`aiExplainer.ts`** — Gemini Flash BYOK integration: API key stored encrypted in VS Code SecretStorage; 15 RPM sliding-window rate limiter; privacy-safe prompt (metadata only).
- **`fixer.ts`** — diff preview via `blackswan-preview://` virtual document scheme; `WorkspaceEdit`-based application for undo support.
- **New VS Code commands**: `blackswan.setApiKey`, `blackswan.applyGuard`, `blackswan.explainWithAI`.
- **New VS Code setting**: `blackswan.geminiModel` (default `gemini-2.5-flash`).
- **`.env.example`** — template for local API key configuration; `.env` is gitignored.
- **73 new Python tests** covering all four guard types, CLI contract, error/unsupported cases, and realistic financial code fixtures.
- **290+ TypeScript tests** covering unit tests for fixer/aiExplainer/diagnostics and integration tests for the Python subprocess round-trip and live Gemini API.

### Changed

- `BlackSwanCodeActionProvider` now returns 3 CodeActions per diagnostic (was 1 comment-only action).
- Default Gemini model updated to `gemini-2.5-flash` with `thinkingBudget: 0` (previous `gemini-2.0-flash` no longer available for new API keys).
- `DiagnosticFixData` interface extended with `failureType`, `line`, and `explainPayload` fields.
- Engine version bumped `0.1.2 → 0.2.0`; extension version bumped `0.3.0 → 0.4.0`.

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

[Extension 0.4.0 / Engine 0.4.0]: https://github.com/Lushenwar/BlackSwan/compare/v0.3.0...v0.4.0
[0.1.2]: https://github.com/Lushenwar/BlackSwan/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Lushenwar/BlackSwan/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Lushenwar/BlackSwan/releases/tag/v0.1.0
