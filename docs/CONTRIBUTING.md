# Contributing to BlackSwan

Thank you for your interest. This document covers how to set up a development environment, run the test suite, and submit changes.

---

## Prerequisites

- **Python 3.11+**
- **Node.js 18+** and **npm** (for the VS Code extension)
- **VS Code** (for extension development and manual testing)
- **Git**

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Lushenwar/BlackSwan.git
cd BlackSwan
```

### 2. Install the Python engine in development mode

```bash
cd core
pip install -e ".[dev]"
```

This installs `blackswan` as an editable package plus `pytest`, `pytest-cov`, and `libcst` (required for the mathematical auto-fixer).

### 3a. (Optional) Configure AI explanations

Copy the env template and add your Gemini key:

```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY=AIza...
```

The `.env` file is in `.gitignore` and will never be committed. The key is only used by integration tests — in the VS Code extension it is stored via the **BlackSwan: Set Gemini API Key** command.

### 3. Install the extension dependencies

```bash
cd ../extension
npm install
```

---

## Building the Extension Package

To produce a `.vsix` for local installation or release:

```bash
cd extension
npm run compile
npx @vscode/vsce package --no-dependencies
```

This outputs `blackswan-vscode-<version>.vsix` in the `extension/` directory. Install it in VS Code with:

```bash
code --install-extension blackswan-vscode-0.3.0.vsix
```

Or via the Extensions panel: `···` menu → **Install from VSIX…**

---

## Running Tests

### Python Engine

```bash
cd core
pytest
```

This runs all tests with coverage reporting. Target: **>90% coverage** on the engine.

Run a specific test file:

```bash
pytest tests/test_detectors.py -v
```

Run a specific test:

```bash
pytest tests/test_adversarial.py::TestEvolutionaryStressRunner::test_run_returns_run_result -v
```

### VS Code Extension

```bash
cd extension
npm test
```

The extension test suite uses Jest with a `vscode` module mock (`__mocks__/vscode.ts`). No VS Code instance is required.

---

## Project Structure

```
BlackSwan/
├── contract/              # Versioned JSON schema (engine ↔ extension)
│   └── schema.json
├── core/                  # Python engine (pip install blackswan)
│   ├── blackswan/
│   │   ├── engine/        # StressRunner, EvolutionaryStressRunner, perturbation, validator
│   │   ├── detectors/     # FailureDetector subclasses
│   │   ├── fixer/         # libcst-based mathematical guard rewriter
│   │   ├── parser/        # AST analysis, dependency graph
│   │   ├── scenarios/     # Scenario dataclasses, YAML loading, presets
│   │   └── attribution/   # Traceback + causal chain
│   ├── tests/
│   │   └── fixtures/      # Python fixture files for integration tests
│   └── pyproject.toml
├── extension/             # VS Code extension (TypeScript)
│   ├── src/
│   │   ├── fixer.ts       # Diff preview + WorkspaceEdit guard application
│   │   └── aiExplainer.ts # Gemini BYOK AI explanations
│   └── package.json
├── docs/                  # Architecture, scenarios, contributing
├── .env.example           # Copy to .env and add GEMINI_API_KEY for integration tests
├── CHANGELOG.md
└── CLAUDE.md              # Authoritative engineering guide
```

---

## Development Guidelines

### Engine (Python)

- **Test first.** Every detector needs ≥10 test cases: 5 true positives, 5 true negatives including tricky edge cases. A noisy detector is worse than a missing one.
- **Determinism is mandatory.** Any randomness that is not seeded via `numpy.default_rng(seed)` is a bug. Perturbation determinism tests must pass.
- **Maintain the JSON contract.** If you change engine output fields, update `contract/schema.json` first. The contract is the source of truth.
- **Keep RunResult backward-compatible.** The VS Code extension depends on it. Additive fields are fine; removing or renaming fields is a breaking change.
- **No parallelism without profiling.** The sequential NumPy loop is fast enough for 5 000 iterations. Do not add `multiprocessing` without a benchmark proving it's needed.

### Extension (TypeScript)

- **The extension renders; the engine computes.** Never hardcode or mock shatter points in the extension.
- **Keep the bridge thin.** `bridge.ts` spawns a process and parses JSON. Business logic belongs in the engine.
- **Every diagnostic must answer five questions:** what failed, how often, under what conditions, on which line, caused by which input. If it can't, it's not ready to ship.

### Auto-Fixer (Python + TypeScript)

- **libcst only for the fixer.** AST + astor lose formatting; libcst preserves it exactly.
- **Never transmit source code to the AI.** `aiExplainer.ts` sends only `ExplainPayload` metadata — failure type, message, frequency, fix hint, and causal chain variable names. Raw source stays local.
- **Keys via SecretStorage only.** Never read or write a Gemini API key from `settings.json` or any file that could be committed.

### Both

- **Small commits.** One logical change per commit. Prefix messages with `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`.
- **No Co-Authored-By trailers** in commits.
- **Do not advance phases** without meeting the exit criterion defined in `CLAUDE.md`.

---

## Adding a New Detector

1. Create or modify a file in `core/blackswan/detectors/`
2. Subclass `FailureDetector` from `detectors/base.py`
3. Implement `check(self, inputs, output, iteration) -> Finding | None`
4. If stateful, implement `reset()` and `set_baseline(output)`
5. Register the detector in the runner's default detector list
6. Write ≥10 test cases in `core/tests/test_detectors.py`
7. Document in `docs/ARCHITECTURE.md` and `README.md`

```python
from blackswan.detectors.base import FailureDetector, Finding

class MyDetector(FailureDetector):
    FAILURE_TYPE = "my_failure"

    def check(self, inputs: dict, output, iteration: int) -> Finding | None:
        if <failure_condition>:
            return Finding(
                failure_type=self.FAILURE_TYPE,
                severity="critical",
                message="Description of what failed and why.",
                iteration=iteration,
                line=0,
                column=0,
                exc_frames=(),
            )
        return None
```

---

## Adding a New Scenario

1. Create a YAML file in `core/blackswan/scenarios/presets/`
2. Follow the format documented in [docs/SCENARIOS.md](SCENARIOS.md)
3. Add a test that runs the scenario against an appropriate fixture and verifies it fires the expected detector
4. Document the scenario in `docs/SCENARIOS.md`

---

## Submitting a Pull Request

1. Fork the repository and create a branch from `main`
2. Make your changes following the guidelines above
3. Ensure all tests pass: `pytest` (Python) and `npm test` (extension)
4. Open a pull request against `main` with a clear description of what changed and why
5. Reference any related issues with `Closes #<number>`

---

## Reporting Bugs

Open an issue at [github.com/Lushenwar/BlackSwan/issues](https://github.com/Lushenwar/BlackSwan/issues). Include:

- BlackSwan version (`pip show blackswan`)
- Python version (`python --version`)
- The command you ran
- The full output (stdout + stderr)
- A minimal reproducing fixture if possible

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
