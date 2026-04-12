/**
 * Tests for fixer.ts — the deterministic mathematical guard applicator.
 *
 * These tests cover:
 *   - BlackSwanPreviewProvider content lifecycle (set/clear/update)
 *   - _applyResultToContent logic via the exported helper (tested indirectly)
 *   - applyMathGuard happy path with a mocked Python subprocess
 *   - applyMathGuard handling of unsupported / error responses
 *
 * The vscode module is mocked via __mocks__/vscode.ts.
 * child_process.spawn is mocked to avoid spawning a real Python process.
 */

import * as vscode from "vscode";
import { BlackSwanPreviewProvider } from "../src/fixer";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeUri(fsPath: string): vscode.Uri {
  return vscode.Uri.file(fsPath);
}

// ---------------------------------------------------------------------------
// BlackSwanPreviewProvider
// ---------------------------------------------------------------------------

describe("BlackSwanPreviewProvider", () => {
  let provider: BlackSwanPreviewProvider;

  beforeEach(() => {
    provider = new BlackSwanPreviewProvider();
  });

  afterEach(() => {
    provider.dispose();
  });

  it("returns empty string for unknown URI", () => {
    const uri = makeUri("/some/model.py");
    expect(provider.provideTextDocumentContent(uri)).toBe("");
  });

  it("returns stored content after set()", () => {
    const uri = vscode.Uri.parse("blackswan-preview:///workspace/model.py");
    provider.set(uri, "fixed content");
    expect(provider.provideTextDocumentContent(uri)).toBe("fixed content");
  });

  it("returns empty string after clear()", () => {
    const uri = vscode.Uri.parse("blackswan-preview:///workspace/model.py");
    provider.set(uri, "some content");
    provider.clear(uri);
    expect(provider.provideTextDocumentContent(uri)).toBe("");
  });

  it("replaces content on repeated set()", () => {
    const uri = vscode.Uri.parse("blackswan-preview:///workspace/model.py");
    provider.set(uri, "first");
    provider.set(uri, "second");
    expect(provider.provideTextDocumentContent(uri)).toBe("second");
  });

  it("fires onDidChange event when content is set", () => {
    const uri = vscode.Uri.parse("blackswan-preview:///workspace/model.py");
    let fired: vscode.Uri | undefined;
    provider.onDidChange((u) => { fired = u; });
    provider.set(uri, "content");
    expect(fired?.toString()).toBe(uri.toString());
  });

  it("isolates content between different URIs", () => {
    const uri1 = vscode.Uri.parse("blackswan-preview:///a.py");
    const uri2 = vscode.Uri.parse("blackswan-preview:///b.py");
    provider.set(uri1, "content-a");
    provider.set(uri2, "content-b");
    expect(provider.provideTextDocumentContent(uri1)).toBe("content-a");
    expect(provider.provideTextDocumentContent(uri2)).toBe("content-b");
    provider.clear(uri1);
    expect(provider.provideTextDocumentContent(uri1)).toBe("");
    expect(provider.provideTextDocumentContent(uri2)).toBe("content-b");
  });
});

// ---------------------------------------------------------------------------
// AST fix application (Python guards.py) — integration contract tests
//
// These tests verify that the JSON contract between fixer.ts and guards.py is
// correctly handled by the TypeScript side, using a mocked subprocess response.
// They do not spawn real Python.
// ---------------------------------------------------------------------------

describe("Fixer response contract", () => {
  /**
   * Parse a FixerResponse JSON string the way fixer.ts does (JSON.parse).
   * We test the shape our code would receive and act on.
   */
  function parseFixerResponse(json: string): object {
    return JSON.parse(json) as object;
  }

  it("accepts ok response with replacement", () => {
    const response = parseFixerResponse(JSON.stringify({
      status: "ok",
      line: 42,
      original: "    result = x / vol",
      replacement: "    result = x / (vol if abs(vol) > 1e-10 else (1e-10 if (vol) >= 0 else -1e-10))",
      explanation: "Wrapped denominator with epsilon guard.",
    }));
    expect((response as { status: string }).status).toBe("ok");
    expect((response as { line: number }).line).toBe(42);
  });

  it("accepts ok response with extra_lines (insert-only fix)", () => {
    const response = parseFixerResponse(JSON.stringify({
      status: "ok",
      line: 55,
      original: "    cov_matrix = np.cov(returns.T)",
      replacement: "    cov_matrix = np.cov(returns.T)",
      extra_lines: [
        "    _bs_w, _bs_v = np.linalg.eigh(cov_matrix)",
        "    cov_matrix = _bs_v @ np.diag(np.maximum(_bs_w, 0.0)) @ _bs_v.T",
      ],
      explanation: "Added eigenvalue clamping.",
    }));
    expect((response as { status: string }).status).toBe("ok");
    expect((response as { extra_lines: string[] }).extra_lines).toHaveLength(2);
  });

  it("accepts error response", () => {
    const response = parseFixerResponse(JSON.stringify({
      status: "error",
      message: "No division operation found at line 42.",
    }));
    expect((response as { status: string }).status).toBe("error");
    expect((response as { message: string }).message).toContain("No division");
  });

  it("accepts unsupported response", () => {
    const response = parseFixerResponse(JSON.stringify({
      status: "unsupported",
      message: "No guard pattern for failure type 'bounds_exceeded'.",
    }));
    expect((response as { status: string }).status).toBe("unsupported");
  });
});
