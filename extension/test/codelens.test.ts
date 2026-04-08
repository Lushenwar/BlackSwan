/**
 * Unit tests for codelens.ts.
 *
 * Covers:
 *   scanForTargets  — the pure line-scanner (no VS Code dependency)
 *   BlackSwanCodeLensProvider — document → CodeLens[] conversion
 */

import * as vscode from "vscode";
import {
  CODELENS_COMMAND,
  BlackSwanCodeLensProvider,
  FunctionTarget,
  scanForTargets,
} from "../src/codelens";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function mockDoc(
  lines: string[],
  languageId = "python",
): vscode.TextDocument {
  return {
    uri:         vscode.Uri.file("/workspace/model.py"),
    languageId,
    lineCount:   lines.length,
    lineAt:      jest.fn((i: number) => ({ text: lines[i] ?? "" })),
  } as unknown as vscode.TextDocument;
}

// ---------------------------------------------------------------------------
// scanForTargets — pure function tests
// ---------------------------------------------------------------------------

describe("scanForTargets — function detection", () => {
  test("detects a top-level def", () => {
    const targets = scanForTargets(["def calculate_var(weights, cov):","    pass"]);
    expect(targets).toHaveLength(1);
    expect(targets[0]).toMatchObject({ name: "calculate_var", line: 0, kind: "function" });
  });

  test("detects async def", () => {
    const targets = scanForTargets(["async def fetch_data():", "    pass"]);
    expect(targets).toHaveLength(1);
    expect(targets[0]).toMatchObject({ name: "fetch_data", kind: "function" });
  });

  test("detects indented method inside a class", () => {
    const lines = [
      "class RiskModel:",
      "    def compute(self):",
      "        pass",
    ];
    const targets = scanForTargets(lines);
    const methods = targets.filter((t) => t.kind === "function");
    expect(methods).toHaveLength(1);
    expect(methods[0].name).toBe("compute");
    expect(methods[0].line).toBe(1);
  });

  test("detects dunder methods (__init__, __call__)", () => {
    const lines = ["    def __init__(self):", "        pass"];
    const targets = scanForTargets(lines);
    expect(targets.map((t) => t.name)).toContain("__init__");
  });

  test("detects private functions (_helper)", () => {
    const targets = scanForTargets(["def _helper(x):", "    pass"]);
    expect(targets).toHaveLength(1);
    expect(targets[0].name).toBe("_helper");
  });

  test("detects multiple functions in order", () => {
    const lines = [
      "def alpha():", "    pass",
      "def beta():",  "    pass",
      "def gamma():", "    pass",
    ];
    const targets = scanForTargets(lines);
    expect(targets.map((t) => t.name)).toEqual(["alpha", "beta", "gamma"]);
  });

  test("reports correct 0-indexed line numbers", () => {
    const lines = [
      "# comment",
      "import numpy as np",
      "",
      "def my_func():",
      "    pass",
    ];
    const targets = scanForTargets(lines);
    expect(targets[0].line).toBe(3);
  });

  test("handles async with extra whitespace (async  def)", () => {
    // "async  def" (double space) — regex uses \s+ so this must match.
    const targets = scanForTargets(["async  def edge_case():", "    pass"]);
    expect(targets).toHaveLength(1);
    expect(targets[0].name).toBe("edge_case");
  });
});

describe("scanForTargets — class detection", () => {
  test("detects a top-level class", () => {
    const targets = scanForTargets(["class PortfolioModel:", "    pass"]);
    expect(targets).toHaveLength(1);
    expect(targets[0]).toMatchObject({ name: "PortfolioModel", line: 0, kind: "class" });
  });

  test("detects a class with a base class", () => {
    const targets = scanForTargets(["class VaRModel(BaseModel):", "    pass"]);
    expect(targets).toHaveLength(1);
    expect(targets[0].name).toBe("VaRModel");
  });

  test("detects indented inner class", () => {
    const lines = ["class Outer:", "    class Inner:", "        pass"];
    const targets = scanForTargets(lines);
    const classes = targets.filter((t) => t.kind === "class");
    expect(classes.map((c) => c.name)).toEqual(["Outer", "Inner"]);
  });
});

describe("scanForTargets — mixed content", () => {
  test("returns empty array for a file with no functions or classes", () => {
    const lines = [
      "# Just imports",
      "import numpy as np",
      "import pandas as pd",
      "",
      "CONSTANT = 42",
    ];
    expect(scanForTargets(lines)).toHaveLength(0);
  });

  test("returns empty array for empty input", () => {
    expect(scanForTargets([])).toHaveLength(0);
  });

  test("does not produce targets for string literals containing 'def'", () => {
    const lines = [
      'description = "def is a keyword"',
      "x = 1",
    ];
    // The string literal starts with `description =`, not `def`
    expect(scanForTargets(lines)).toHaveLength(0);
  });

  test("detects both classes and functions, preserving source order", () => {
    const lines = [
      "class Alpha:",
      "    def method(self):",
      "        pass",
      "",
      "def standalone():",
      "    pass",
    ];
    const targets = scanForTargets(lines);
    expect(targets.map((t) => t.name)).toEqual(["Alpha", "method", "standalone"]);
    expect(targets.map((t) => t.kind)).toEqual(["class", "function", "function"]);
  });

  test("realistic financial model file", () => {
    const lines = [
      "import numpy as np",
      "",
      "class PortfolioRiskModel:",
      "    def __init__(self, weights, cov_matrix):",
      "        self.weights = weights",
      "        self.cov = cov_matrix",
      "",
      "    def calculate_var(self, confidence=0.95):",
      "        portfolio_vol = np.sqrt(self.weights @ self.cov @ self.weights)",
      "        return portfolio_vol",
      "",
      "    def stress_test(self, scenarios):",
      "        pass",
      "",
      "def quick_var(weights, cov):",
      "    return np.sqrt(weights @ cov @ weights)",
    ];
    const targets = scanForTargets(lines);
    expect(targets.map((t) => t.name)).toEqual([
      "PortfolioRiskModel",
      "__init__",
      "calculate_var",
      "stress_test",
      "quick_var",
    ]);
  });
});

// ---------------------------------------------------------------------------
// BlackSwanCodeLensProvider
// ---------------------------------------------------------------------------

describe("BlackSwanCodeLensProvider", () => {
  let provider: BlackSwanCodeLensProvider;

  beforeEach(() => {
    provider = new BlackSwanCodeLensProvider();
  });

  afterEach(() => {
    provider.dispose();
  });

  test("returns one CodeLens per detected target", () => {
    const doc = mockDoc([
      "def alpha():", "    pass",
      "def beta():",  "    pass",
    ]);
    const lenses = provider.provideCodeLenses(doc);
    expect(lenses).toHaveLength(2);
  });

  test("returns empty array for non-Python documents", () => {
    const doc = mockDoc(["def foo():", "    pass"], "javascript");
    expect(provider.provideCodeLenses(doc)).toHaveLength(0);
  });

  test("each CodeLens range starts on the definition line (0-indexed)", () => {
    const doc = mockDoc([
      "# header",
      "import numpy",
      "",
      "def my_func():",
      "    pass",
    ]);
    const [lens] = provider.provideCodeLenses(doc);
    expect(lens.range.start.line).toBe(3);
  });

  test("CodeLens command is CODELENS_COMMAND", () => {
    const doc = mockDoc(["def f():", "    pass"]);
    const [lens] = provider.provideCodeLenses(doc);
    expect(lens.command?.command).toBe(CODELENS_COMMAND);
    expect(lens.command?.command).toBe("blackswan.runStressTest");
  });

  test("CodeLens title is '▶ Run BlackSwan'", () => {
    const doc = mockDoc(["def f():", "    pass"]);
    const [lens] = provider.provideCodeLenses(doc);
    expect(lens.command?.title).toBe("▶ Run BlackSwan");
  });

  test("CodeLens arguments contain [uri, functionName]", () => {
    const doc = mockDoc(["def calculate_var():", "    pass"]);
    const [lens] = provider.provideCodeLenses(doc);
    const [uri, name] = lens.command?.arguments as [vscode.Uri, string];
    expect(uri.toString()).toBe(doc.uri.toString());
    expect(name).toBe("calculate_var");
  });

  test("CodeLens tooltip mentions the function name", () => {
    const doc = mockDoc(["def calculate_var():", "    pass"]);
    const [lens] = provider.provideCodeLenses(doc);
    expect(lens.command?.tooltip).toContain("calculate_var");
  });

  test("returns empty array for a document with no functions or classes", () => {
    const doc = mockDoc(["import numpy as np", "", "X = 42"]);
    expect(provider.provideCodeLenses(doc)).toHaveLength(0);
  });

  test("refresh() fires onDidChangeCodeLenses event", () => {
    const listener = jest.fn();
    provider.onDidChangeCodeLenses(listener);
    provider.refresh();
    expect(listener).toHaveBeenCalledTimes(1);
  });

  test("event listener is not called after provider is disposed", () => {
    const listener = jest.fn();
    provider.onDidChangeCodeLenses(listener);
    provider.dispose();
    // EventEmitter.dispose() clears all listeners — fire should be a no-op.
    provider.refresh(); // fires on (now-disposed) EventEmitter
    expect(listener).not.toHaveBeenCalled();
  });
});
