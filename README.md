# BlackSwan 

**A debugger for mathematical and financial fragility.**

A black swan in business is an unpredictable, extremely rare event with a massive, often catastrophic impact, which is inappropriately rationalized in hindsight as having been foreseeable.
<img width="1280" height="720" alt="logo" src="https://github.com/user-attachments/assets/68c066a6-2846-459c-9364-01afd02a00e9" />

Standard linters (Pylint, Flake8, Pyright) catch syntax and type errors—they make sure your code runs. **BlackSwan catches logical and numerical fragility**—it makes sure your math survives under pressure. 

BlackSwan is a VS Code extension backed by a decoupled Python engine. It stress-tests mathematical logic in Python code using Monte Carlo simulations to find the exact line where a model breaks under extreme conditions, and explains why.

---

## 🚀 Features

- **Line-Level Attribution:** Draws red squiggly lines on the exact line of code responsible for a mathematical failure.
- **Causal Chains:** Hover over a failure to see the exact chain of variables that led from the perturbed input to the shatter point.
- **Built-in Financial Scenarios:** Stress test using 5 preset scenarios: *Liquidity Crash, Volatility Spike, Correlation Breakdown, Rate Shock, and Missing Data.*
- **5 Core Failure Detectors:**
  - `NaNInfDetector`: Catches any computation producing NaN or Inf.
  - `DivisionStabilityDetector`: Flags divisions where the denominator approaches zero.
  - `MatrixPSDDetector`: Verifies positive semi-definiteness of covariance/correlation matrices.
  - `ConditionNumberDetector`: Flags ill-conditioned matrices before inversion.
  - `BoundsDetector`: Catches outputs exceeding plausible bounds.
- **Dependency Graph (DAG):** Visualize your model's data flow and pinpoint root causes in the BlackSwan sidebar panel.

---

## 📦 Installation

BlackSwan consists of two independent systems: the **Python Engine** (which does the heavy lifting) and the **VS Code Extension** (which renders the results). **You must install both.**

### Step 1: Install the Python Engine
The core engine must be installed in the Python environment you use to run your models.

Open your terminal and activate your project's virtual environment, then run:
```bash
pip install blackswan-core
```

### Step 2: Install the VS Code Extension
Currently, BlackSwan is distributed as a `.vsix` package. 

1. Download the latest `blackswan-vscode.vsix` file from the [Releases](#) page.
2. Open Visual Studio Code.
3. Open the Extensions view (`Ctrl+Shift+X` or `Cmd+Shift+X` on Mac).
4. Click the `...` (Views and More Actions) menu in the top right of the Extensions panel.
5. Select **"Install from VSIX..."**
6. Locate and select the `blackswan-vscode.vsix` file you downloaded.

*Note: Ensure your VS Code Python interpreter is set to the same environment where you installed `blackswan-core`.*

---

## 🛠️ Usage

### Using the VS Code Extension (Recommended)
1. Open a Python file containing financial or mathematical functions.
2. Look for the **"▶ Run BlackSwan"** CodeLens button hovering above your function definitions.
3. Click it and select a stress scenario from the dropdown (e.g., `liquidity_crash`).
4. Watch the progress bar as the engine runs thousands of iterations.
5. **Review Failures:** Any lines that fail under stress will be underlined in red. Hover over them to see the failure frequency, explanation, and causal chain.

### Using the CLI Engine
The Python engine is fully functional as a standalone CLI tool. You can run it directly in your terminal, making it perfect for CI/CD pipelines.

```bash
# Run a specific scenario on a model
python -m blackswan test risk_model.py --scenario liquidity_crash

# List all available stress scenarios
python -m blackswan --list-scenarios
```
The CLI outputs a structured JSON report detailing all shatter points, causal chains, and scenario cards for exact reproducibility.

---

## 🎯 Supported Code Patterns
BlackSwan V1 is deliberately focused on portfolio risk, covariance/correlation analysis, and VaR-style risk models.

**Supported:**
- Pure functions with NumPy/Pandas inputs and outputs
- Explicit variable assignments and NumPy array operations
- Pandas DataFrame column operations
- Single-file scripts and Jupyter notebook cells (exported to script)

**Not Supported in V1:**
- Deeply nested class hierarchies with complex state/side-effects
- Multi-threaded or async computation pipelines
- Sprawling multi-package codebases with circular imports

---

## 🤝 Contributing
BlackSwan's engine and VS Code extension are deeply decoupled. If you are interested in adding new numerical detectors or contributing to the AST parser, please see our [ARCHITECTURE.md](docs/ARCHITECTURE.md) and [CONTRIBUTING.md](docs/CONTRIBUTING.md) guides.

## 📄 License
[MIT License](LICENSE)
