# BlackSwan Engineering Roadmap

Complete implementation guide for the next phase of development.

---

## What BlackSwan Is

BlackSwan is a mathematical fragility debugger for Python. It stress-tests financial and numerical functions by applying thousands of perturbations drawn from realistic market scenarios (liquidity crash, vol spike, correlation breakdown, etc.), then pinpoints the exact source line where the model breaks, how often it breaks, and which input caused it. It ships as a pip package, a CLI tool, a Python API, and a VS Code extension with red-squiggle annotations, a DAG dependency panel, an auto-fixer, and optional Gemini-powered explanations.

---

## Priority Overview

Five initiatives ordered by impact-to-effort ratio. Complete them in this sequence unless a specific reason exists to reorder.

---

## Step 1 — Public Landing Page

### What It Is

A single static webpage that markets BlackSwan to its target audience: quantitative developers, risk engineers, and financial software teams. Its sole job is to answer "what is this and why should I care?" and funnel visitors to `pip install blackswan` and the VS Code extension download.

### Why It Matters

The README is thorough and technically excellent but written for people who have already decided to look. A landing page is written for people who have not decided yet. Without it, discovery is limited to direct GitHub links. With it, there is a shareable URL, a demo to post on Hacker News, LinkedIn, or Reddit, and a professional face for the project.

### Where to Host

GitHub Pages at `lushenwar.github.io/BlackSwan` or a custom domain. Enable in repository Settings > Pages pointing at the `docs/` folder of the main branch.

### Sections to Build (in order)

1. **Hero section**: bold headline naming the pain ("Your risk model passes every test. Then it blows up in prod."), one subheadline explaining BlackSwan in one sentence, two CTAs: `pip install blackswan` and Download VS Code Extension.
2. **Animated terminal demo**: looping animation showing a real BlackSwan run. Use exact JSON output from the README. Use `typed.js` or CSS keyframe animation. Loop under 30 seconds or make scroll-triggered.
3. **Three-feature strip**: finds the exact line, shows the full causal chain, auto-fixes with a deterministic guard.
4. **Scenarios section**: visual grid of five preset scenarios with one-line descriptions. Signals domain expertise.
5. **Install section**: `pip install blackswan` and VSIX install command in code blocks with copy-to-clipboard buttons.
6. **Footer**: GitHub link, PyPI badge, license badge, build status badge.

### Technical Implementation

Plain HTML, CSS, and minimal vanilla JavaScript. No React, no bundler, no build step. GitHub Pages serves static files directly.

For the terminal animation: use a `pre` element with monospace font and blinking cursor. Append characters one at a time using `setInterval` or `requestAnimationFrame`. Use a single text node and update `nodeValue` — NOT `innerHTML` on every character (causes layout reflow).

### Watch Out For

- Mobile viewport: terminal animation must not overflow horizontally
- `og:image` and meta description tags required for social sharing
- CNAME file for custom domain if used
- Copy-to-clipboard requires `navigator.clipboard.writeText()` with HTTPS fallback

---

## Step 2 — SARIF and CI Integration

### What SARIF Is

SARIF (Static Analysis Results Interchange Format) is a JSON schema adopted by GitHub, Azure DevOps, GitLab as the standard for code analysis findings. When a SARIF file is uploaded to GitHub via `upload-sarif`, GitHub renders each finding as an inline annotation on the PR diff at the exact line, adds to the Security > Code Scanning tab, and can block merging if branch protection rules require resolved alerts.

### What to Build

Add `--output sarif` flag to CLI. When set, write a `blackswan-results.sarif` file to `--output-path` (default: `./blackswan-results.sarif`). SARIF is purely a serialization of the existing `Finding` dataclass — nothing in the engine changes.

### SARIF File Structure (minimal valid 2.1.0)

```json
{
  "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
  "version": "2.1.0",
  "runs": [{
    "tool": {
      "driver": {
        "name": "blackswan",
        "version": "0.4.0",
        "rules": [
          {
            "id": "non_psd_matrix",
            "name": "NonPSDMatrix",
            "shortDescription": { "text": "Covariance matrix loses positive semi-definiteness" },
            "helpUri": "https://github.com/Lushenwar/BlackSwan"
          }
        ]
      }
    },
    "results": [
      {
        "ruleId": "non_psd_matrix",
        "level": "error",
        "message": { "text": "Covariance matrix loses PSD at correlation > 0.91. Freq: 16.9%" },
        "locations": [{
          "physicalLocation": {
            "artifactLocation": { "uri": "models/risk.py", "uriBaseId": "%SRCROOT%" },
            "region": { "startLine": 36 }
          }
        }]
      }
    ]
  }]
}
```

### Severity Mapping

- `critical` → `"error"`
- `medium` → `"warning"`
- `low` → `"note"`

### GitHub Actions Workflow

```yaml
name: BlackSwan stress test
on: [push, pull_request]
jobs:
  blackswan:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install blackswan
      - name: Run BlackSwan
        run: |
          python -m blackswan test models/risk.py \
            --scenario liquidity_crash \
            --output sarif \
            --output-path blackswan-results.sarif
        continue-on-error: true
      - name: Upload SARIF to GitHub
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: blackswan-results.sarif
```

### Watch Out For

- `uriBaseId: "%SRCROOT%"` is required on every artifactLocation or GitHub annotations won't render
- `tool.driver.rules` must be built dynamically from unique `failure_types` present in findings
- `continue-on-error: true` on the CLI step so SARIF uploads even when findings exist (exit code 1)
- Validate output against official SARIF 2.1.0 JSON Schema before shipping

---

## Step 3 — Claude Code MCP Server

### What MCP Is

Model Context Protocol (MCP) is an open standard by Anthropic that lets AI assistants call external tools in a structured, typed way. A MCP server exposes named tools with JSON Schema definitions. When Claude Code runs and a user asks "stress test my risk.py against a liquidity crash scenario", Claude calls the BlackSwan MCP tool, receives structured JSON output, and reasons over findings in full context — suggesting fixes, explaining causal chains, asking follow-up questions.

### Tools to Expose

- `run_blackswan` — stress-test a Python file with a named scenario
- `list_scenarios` — return all available preset scenarios
- `get_finding_detail` — return full causal chain for a specific shatter point
- `explain_finding` — call Anthropic API to explain a finding in plain English

### Implementation

```python
# blackswan/mcp_server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from blackswan.engine.runner import StressRunner
from blackswan.scenarios.registry import load_scenario
import asyncio, json, os

server = Server("blackswan")

@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="run_blackswan",
            description="Stress-test a Python function for mathematical fragility under market scenarios.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "Path to the Python file"},
                    "scenario": {"type": "string", "description": "Scenario name, e.g. liquidity_crash"},
                    "iterations": {"type": "integer", "default": 5000}
                },
                "required": ["file", "scenario"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "run_blackswan":
        file_path = os.path.abspath(os.path.join(os.getcwd(), arguments["file"]))
        scenario = load_scenario(arguments["scenario"])
        runner = StressRunner(scenario)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, runner.run_file, file_path)
        return [types.TextContent(type="text", text=json.dumps(result.to_dict()))]

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())
```

### User Setup (~/.claude/claude_desktop_config.json)

```json
{
  "mcpServers": {
    "blackswan": {
      "command": "python",
      "args": ["-m", "blackswan.mcp_server"],
      "env": {}
    }
  }
}
```

### Replacing the Gemini Dependency

The VS Code extension currently calls Gemini API for "Explain with AI". Replace with Anthropic API using `claude-sonnet-4-20250514`. System prompt includes failure type, frequency, causal chain, fix hint. Removes external Gemini dependency; users need only one API key.

### Watch Out For

- All synchronous runner calls must be wrapped in `asyncio.get_event_loop().run_in_executor(None, ...)` to avoid blocking
- Resolve all file path arguments to absolute paths with `os.path.abspath` before passing to runner
- Add entry point in `pyproject.toml`: `blackswan-mcp = 'blackswan.mcp_server:main'`
- Test end-to-end in Claude Code before shipping

---

## Step 4 — Agentic Fix Loop

### What It Is

The agentic fix loop closes the feedback cycle that currently requires human intervention. Current workflow: BlackSwan finds failure → user reads → user applies fixer → user re-runs manually. The agentic loop does this automatically: find failures → apply deterministic guard fixer → re-run stress test → verify fix held → repeat until stable or max rounds reached.

### What to Build

Add `--agentic` flag and `--max-agentic-rounds N` flag (default 3) to CLI.

Loop logic:
1. Round 1: Run stress test on original file. Collect all findings.
2. For each critical finding, call Fixer module to generate guard patch and apply to working copy.
3. Round 2: Re-run stress test on patched file using exact same scenario and seed.
4. Compare findings: finding from Round 1 not in Round 2 at same line = "resolved". Appears in both = "fix ineffective", not retried.
5. Repeat up to `--max-agentic-rounds` or until no critical findings remain.
6. Write final `AgentReport` showing original, resolved, and unresolvable findings, plus unified diff.
7. Show diff to user and require explicit confirmation before overwriting original file.

The loop always operates on a copy in `.blackswan_patch/`, never the original, until user approves.

### Pseudocode

```python
def agentic_loop(file, scenario, seed, max_rounds=3):
    working = copy_to_patch_dir(file)  # .blackswan_patch/risk.py
    history = []
    unresolvable = set()

    for round_num in range(max_rounds):
        result = run_stress_test(working, scenario, seed=seed)
        critical = [
            f for f in result.findings
            if f.severity == "critical"
            and (f.line, f.failure_type) not in unresolvable
        ]

        if not critical:
            break

        history.append({"round": round_num, "findings": critical})

        # Re-parse AST after EACH fix — never batch
        for finding in critical:
            patch = generate_fix(working, finding)
            if patch:
                apply_patch(working, patch)
            else:
                unresolvable.add((finding.line, finding.failure_type))

    diff = unified_diff(file, working)
    print(diff)
    if confirm_user("Apply these changes?"):
        overwrite(file, working)

    return AgentReport(history, unresolvable, diff)
```

### Watch Out For

- Re-parse AST after every individual fix application — never batch fixes from a single parse (line numbers shift)
- Mark finding unresolvable when same `(line, failure_type)` pair appears across two consecutive rounds
- Enforce seed consistency: pass same `--seed` to every round
- Add `--dry-run` flag that prints diff without writing any files

---

## Step 5 — PyCharm Integration

### Two Options

**Option A: External Tool (Low Effort — Do This Now)**

JetBrains IDEs support External Tools: arbitrary command-line programs triggered from Tools menu or keyboard shortcut. Output appears in Run panel with clickable file paths and line numbers. Requires zero plugin development.

Setup guide fields for users (Settings > Tools > External Tools > +):
- Name: `BlackSwan`
- Program: `python`
- Arguments: `-m blackswan test $FilePath$ --scenario liquidity_crash`
- Working directory: `$ProjectFileDir$`

Add this guide to `docs/` and link from README.

**Option B: Full IntelliJ Platform Plugin (High Effort — Defer)**

A proper IntelliJ plugin with inline red squiggles, hover tooltips, and DAG panel would match the VS Code extension in quality. Written in Kotlin, built with Gradle. The engine-extension contract in `contract/schema.json` means the plugin is purely a renderer. Defer until user research provides clear signal that a significant portion of users are on JetBrains IDEs.

---

## Master Implementation Checklist

### Step 1 — Landing Page
- [ ] Create `docs/` folder in BlackSwan repo
- [ ] Enable GitHub Pages in repository Settings > Pages, pointing to `docs/` of main branch
- [ ] Create `docs/index.html` with sections: hero, animated terminal, three-feature strip, scenarios grid, install section, footer
- [ ] Implement animated terminal using `typed.js` or `requestAnimationFrame` with a text node (NOT innerHTML)
- [ ] Add copy-to-clipboard buttons on all code blocks using the Clipboard API
- [ ] Add `og:image` and meta description tags to HTML head
- [ ] Add CNAME file to repo root if using custom domain
- [ ] Test on mobile viewport — terminal animation must not overflow horizontally

### Step 2 — SARIF / CI
- [ ] Add `sarif` as valid value for `--output` flag in `cli.py`
- [ ] Add `--output-path` argument to `cli.py` (default: `./blackswan-results.sarif`)
- [ ] Write `SARIFSerializer` class that accepts `StressResult` and produces valid SARIF 2.1.0 dict
- [ ] Map severity: critical → 'error', medium → 'warning', low → 'note'
- [ ] Build `tool.driver.rules` dynamically from unique `failure_types` in findings
- [ ] Add `uriBaseId: '%SRCROOT%'` to every `artifactLocation` object
- [ ] Validate output against official SARIF 2.1.0 JSON Schema before shipping
- [ ] Create `.github/workflows/blackswan.yml` with `permissions: security-events: write`
- [ ] Add `continue-on-error: true` on CLI run step
- [ ] Test full pipeline end-to-end on fork with known-failing model file

### Step 3 — Claude Code MCP Server
- [ ] Add `mcp` to project dependencies
- [ ] Create `blackswan/mcp_server.py` exposing: `run_blackswan`, `list_scenarios`, `get_finding_detail`, `explain_finding`
- [ ] Wrap all synchronous runner calls in `asyncio.get_event_loop().run_in_executor(None, ...)`
- [ ] Resolve all file path arguments to absolute paths
- [ ] Add entry point in `pyproject.toml`: `blackswan-mcp = 'blackswan.mcp_server:main'`
- [ ] Write user setup docs with exact `claude_desktop_config.json` snippet
- [ ] Replace Gemini API call in VS Code extension with Anthropic API using `claude-sonnet-4-20250514`
- [ ] Test end-to-end in Claude Code

### Step 4 — Agentic Fix Loop
- [ ] Add `--agentic` and `--max-agentic-rounds N` flags to `cli.py`
- [ ] Implement `copy_to_patch_dir()` creating `.blackswan_patch/` directory
- [ ] Re-parse AST after every individual fix application
- [ ] Implement unresolvable detection: same `(line, failure_type)` across two consecutive rounds
- [ ] Enforce seed consistency across all rounds
- [ ] Show unified diff and require explicit user confirmation before overwriting
- [ ] Implement `AgentReport` dataclass with per-round findings, resolved, and unresolvable
- [ ] Add `--dry-run` flag

### Step 5 — PyCharm Integration
- [ ] Write External Tool setup guide with exact field values
- [ ] Include keyboard shortcut configuration instructions
- [ ] Add guide to `docs/` and link from README under 'Editor Integrations' section
- [ ] Add note that full IntelliJ plugin is on roadmap — do not build it yet
