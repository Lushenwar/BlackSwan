/**
 * Unit tests for hover.ts.
 *
 * Tests exercise:
 *   - buildHoverContent  — pure content builder (all five failure types,
 *                          causal chain roles, fix hint presence/absence,
 *                          command link format, trusted MarkdownString)
 *   - BlackSwanHoverProvider — store management (set/clear/clearAll) and
 *                              provideHover line matching
 *
 * The vscode module is resolved to __mocks__/vscode.ts by Jest's
 * moduleNameMapper — no live extension host needed.
 */

import * as vscode from "vscode";
import { buildHoverContent, BlackSwanHoverProvider } from "../src/hover";
import { ShatterPoint } from "../src/types";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const DOC_URI = vscode.Uri.file("/workspace/risk_model.py");

function makeShatterPoint(overrides: Partial<ShatterPoint> = {}): ShatterPoint {
  return {
    id:           "sp_001",
    line:         82,
    column:       4,
    severity:     "critical",
    failure_type: "non_psd_matrix",
    message:      "Covariance matrix loses positive semi-definiteness when pairwise correlation exceeds 0.91.",
    frequency:    "847 / 5000 iterations (16.9%)",
    causal_chain: [
      { line: 14, variable: "corr_shift",           role: "root_input"   },
      { line: 47, variable: "adjusted_corr_matrix", role: "intermediate" },
      { line: 82, variable: "cov_matrix",           role: "failure_site" },
    ],
    fix_hint: "Apply nearest-PSD correction (Higham 2002) after correlation perturbation.",
    ...overrides,
  };
}

function mockDocument(
  uri = DOC_URI,
  lineCount = 200,
): vscode.TextDocument {
  return {
    uri,
    languageId: "python",
    lineCount,
    lineAt: jest.fn((_line: number) => ({ text: "" })),
  } as unknown as vscode.TextDocument;
}

function mockPosition(line: number, character = 0): vscode.Position {
  return new vscode.Position(line, character);
}

// ---------------------------------------------------------------------------
// buildHoverContent — returns a MarkdownString
// ---------------------------------------------------------------------------

describe("buildHoverContent — MarkdownString properties", () => {
  test("returns a MarkdownString", () => {
    const md = buildHoverContent(makeShatterPoint(), DOC_URI);
    expect(md).toBeInstanceOf(vscode.MarkdownString);
  });

  test("isTrusted is true (required for command:// links)", () => {
    const md = buildHoverContent(makeShatterPoint(), DOC_URI);
    expect(md.isTrusted).toBe(true);
  });

  test("supportHtml is false (no raw HTML in hover)", () => {
    const md = buildHoverContent(makeShatterPoint(), DOC_URI);
    expect(md.supportHtml).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// buildHoverContent — header
// ---------------------------------------------------------------------------

describe("buildHoverContent — failure type labels", () => {
  const cases: Array<[ShatterPoint["failure_type"], string]> = [
    ["non_psd_matrix",         "Non-PSD Matrix"],
    ["nan_inf",                "NaN / Inf Detected"],
    ["division_by_zero",       "Division Instability"],
    ["ill_conditioned_matrix", "Ill-Conditioned Matrix"],
    ["bounds_exceeded",        "Bounds Exceeded"],
  ];

  test.each(cases)("%s → label '%s' appears in header", (failureType, label) => {
    const sp = makeShatterPoint({ failure_type: failureType });
    const md = buildHoverContent(sp, DOC_URI);
    expect(md.value).toContain(label);
  });

  test("header line starts with '## ⚠ BlackSwan:'", () => {
    const md = buildHoverContent(makeShatterPoint(), DOC_URI);
    expect(md.value).toMatch(/^## ⚠ BlackSwan:/);
  });
});

// ---------------------------------------------------------------------------
// buildHoverContent — message and frequency
// ---------------------------------------------------------------------------

describe("buildHoverContent — message and frequency", () => {
  test("contains the engine failure message", () => {
    const sp = makeShatterPoint({
      message: "Covariance matrix loses positive semi-definiteness when pairwise correlation exceeds 0.91.",
    });
    const md = buildHoverContent(sp, DOC_URI);
    expect(md.value).toContain(
      "Covariance matrix loses positive semi-definiteness when pairwise correlation exceeds 0.91.",
    );
  });

  test("contains the frequency string verbatim", () => {
    const sp = makeShatterPoint({ frequency: "847 / 5000 iterations (16.9%)" });
    const md = buildHoverContent(sp, DOC_URI);
    expect(md.value).toContain("847 / 5000 iterations (16.9%)");
  });

  test("frequency is preceded by bold '**Frequency:**' label", () => {
    const md = buildHoverContent(makeShatterPoint(), DOC_URI);
    expect(md.value).toMatch(/\*\*Frequency:\*\*/);
  });
});

// ---------------------------------------------------------------------------
// buildHoverContent — causal chain
// ---------------------------------------------------------------------------

describe("buildHoverContent — causal chain", () => {
  test("contains '**Causal Chain:**' heading when chain is non-empty", () => {
    const md = buildHoverContent(makeShatterPoint(), DOC_URI);
    expect(md.value).toContain("**Causal Chain:**");
  });

  test("does NOT contain 'Causal Chain' when chain is empty", () => {
    const sp = makeShatterPoint({ causal_chain: [] });
    const md = buildHoverContent(sp, DOC_URI);
    expect(md.value).not.toContain("Causal Chain");
  });

  test("root_input link is labelled 'Root input'", () => {
    const md = buildHoverContent(makeShatterPoint(), DOC_URI);
    expect(md.value).toContain("Root input");
  });

  test("intermediate link is labelled 'Intermediate'", () => {
    const md = buildHoverContent(makeShatterPoint(), DOC_URI);
    expect(md.value).toContain("Intermediate");
  });

  test("failure_site link is labelled '**Failure site**'", () => {
    const md = buildHoverContent(makeShatterPoint(), DOC_URI);
    expect(md.value).toContain("**Failure site**");
  });

  test("each chain variable name appears in the content", () => {
    const md = buildHoverContent(makeShatterPoint(), DOC_URI);
    expect(md.value).toContain("corr_shift");
    expect(md.value).toContain("adjusted_corr_matrix");
    expect(md.value).toContain("cov_matrix");
  });

  test("each chain line number appears in the content", () => {
    const md = buildHoverContent(makeShatterPoint(), DOC_URI);
    expect(md.value).toContain("line 14");
    expect(md.value).toContain("line 47");
    expect(md.value).toContain("line 82");
  });

  test("chain links use command:vscode.open (clickable navigation)", () => {
    const md = buildHoverContent(makeShatterPoint(), DOC_URI);
    expect(md.value).toContain("command:vscode.open");
  });

  test("command link args include the document URI", () => {
    const md = buildHoverContent(makeShatterPoint(), DOC_URI);
    // The URI string should appear somewhere in the encoded args
    expect(md.value).toContain(encodeURIComponent(DOC_URI.toString()).slice(0, 10));
  });

  test("chain with only one entry renders without error", () => {
    const sp = makeShatterPoint({
      causal_chain: [{ line: 10, variable: "x", role: "failure_site" }],
    });
    const md = buildHoverContent(sp, DOC_URI);
    expect(md.value).toContain("**Failure site**");
    expect(md.value).toContain("x");
    expect(md.value).toContain("line 10");
  });
});

// ---------------------------------------------------------------------------
// buildHoverContent — fix hint
// ---------------------------------------------------------------------------

describe("buildHoverContent — fix hint", () => {
  test("contains '**Fix Hint:**' when fix_hint is non-empty", () => {
    const sp = makeShatterPoint({
      fix_hint: "Apply Higham 2002 nearest-PSD correction",
    });
    const md = buildHoverContent(sp, DOC_URI);
    expect(md.value).toContain("**Fix Hint:**");
  });

  test("contains the fix hint text when non-empty", () => {
    const sp = makeShatterPoint({
      fix_hint: "Apply Higham 2002 nearest-PSD correction",
    });
    const md = buildHoverContent(sp, DOC_URI);
    expect(md.value).toContain("Apply Higham 2002 nearest-PSD correction");
  });

  test("contains the 💡 emoji before Fix Hint when non-empty", () => {
    const sp = makeShatterPoint({ fix_hint: "Some fix" });
    const md = buildHoverContent(sp, DOC_URI);
    expect(md.value).toContain("💡");
  });

  test("does NOT contain 'Fix Hint' when fix_hint is empty string", () => {
    const sp = makeShatterPoint({ fix_hint: "" });
    const md = buildHoverContent(sp, DOC_URI);
    expect(md.value).not.toContain("Fix Hint");
  });

  test("does NOT contain '💡' when fix_hint is empty string", () => {
    const sp = makeShatterPoint({ fix_hint: "" });
    const md = buildHoverContent(sp, DOC_URI);
    expect(md.value).not.toContain("💡");
  });
});

// ---------------------------------------------------------------------------
// BlackSwanHoverProvider — store management
// ---------------------------------------------------------------------------

describe("BlackSwanHoverProvider — store management", () => {
  test("set() stores shatter points for a URI", () => {
    const provider = new BlackSwanHoverProvider();
    const sp = makeShatterPoint({ line: 82 });
    provider.set(DOC_URI, [sp]);

    const doc = mockDocument(DOC_URI);
    // Line 82 (1-indexed) = position line 81 (0-indexed)
    const hover = provider.provideHover(doc, mockPosition(81));
    expect(hover).toBeDefined();
    provider.dispose();
  });

  test("set() replaces previous shatter points (upsert)", () => {
    const provider = new BlackSwanHoverProvider();
    const sp1 = makeShatterPoint({ id: "sp_001", line: 10 });
    const sp2 = makeShatterPoint({ id: "sp_002", line: 20 });

    provider.set(DOC_URI, [sp1]);
    provider.set(DOC_URI, [sp2]); // replaces sp1

    const doc = mockDocument(DOC_URI);
    expect(provider.provideHover(doc, mockPosition(9))).toBeUndefined();  // old line gone
    expect(provider.provideHover(doc, mockPosition(19))).toBeDefined();   // new line present
    provider.dispose();
  });

  test("clear() removes shatter points for that URI", () => {
    const provider = new BlackSwanHoverProvider();
    const sp = makeShatterPoint({ line: 50 });
    provider.set(DOC_URI, [sp]);
    provider.clear(DOC_URI);

    const hover = provider.provideHover(mockDocument(DOC_URI), mockPosition(49));
    expect(hover).toBeUndefined();
    provider.dispose();
  });

  test("clear() on a URI with no data is a no-op", () => {
    const provider = new BlackSwanHoverProvider();
    expect(() => provider.clear(DOC_URI)).not.toThrow();
    provider.dispose();
  });

  test("clearAll() removes data for all URIs", () => {
    const provider = new BlackSwanHoverProvider();
    const uriA = vscode.Uri.file("/workspace/model_a.py");
    const uriB = vscode.Uri.file("/workspace/model_b.py");
    provider.set(uriA, [makeShatterPoint({ line: 10 })]);
    provider.set(uriB, [makeShatterPoint({ line: 20 })]);

    provider.clearAll();

    expect(provider.provideHover(mockDocument(uriA), mockPosition(9))).toBeUndefined();
    expect(provider.provideHover(mockDocument(uriB), mockPosition(19))).toBeUndefined();
    provider.dispose();
  });

  test("dispose() clears the store without throwing", () => {
    const provider = new BlackSwanHoverProvider();
    provider.set(DOC_URI, [makeShatterPoint()]);
    expect(() => provider.dispose()).not.toThrow();
    // After dispose, no hover should be returned.
    expect(provider.provideHover(mockDocument(DOC_URI), mockPosition(81))).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// BlackSwanHoverProvider — provideHover line matching
// ---------------------------------------------------------------------------

describe("BlackSwanHoverProvider.provideHover", () => {
  test("returns undefined when no data has been stored for the document", () => {
    const provider = new BlackSwanHoverProvider();
    const hover = provider.provideHover(mockDocument(DOC_URI), mockPosition(81));
    expect(hover).toBeUndefined();
    provider.dispose();
  });

  test("returns undefined when cursor is on a line with no shatter point", () => {
    const provider = new BlackSwanHoverProvider();
    provider.set(DOC_URI, [makeShatterPoint({ line: 82 })]);

    // Cursor on line 10 (0-indexed) — no shatter point there.
    const hover = provider.provideHover(mockDocument(DOC_URI), mockPosition(10));
    expect(hover).toBeUndefined();
    provider.dispose();
  });

  test("returns a Hover when cursor is on the exact failing line", () => {
    const provider = new BlackSwanHoverProvider();
    // Engine reports line 82 (1-indexed) → VS Code position.line 81 (0-indexed).
    provider.set(DOC_URI, [makeShatterPoint({ line: 82 })]);

    const hover = provider.provideHover(mockDocument(DOC_URI), mockPosition(81));
    expect(hover).toBeInstanceOf(vscode.Hover);
    provider.dispose();
  });

  test("hover contents include the shatter point's failure type label", () => {
    const provider = new BlackSwanHoverProvider();
    provider.set(DOC_URI, [makeShatterPoint({ line: 82, failure_type: "nan_inf" })]);

    const hover = provider.provideHover(mockDocument(DOC_URI), mockPosition(81))!;
    const md = hover.contents as unknown as vscode.MarkdownString;
    expect(md.value).toContain("NaN / Inf Detected");
    provider.dispose();
  });

  test("hover contents include the shatter point's frequency", () => {
    const provider = new BlackSwanHoverProvider();
    provider.set(DOC_URI, [makeShatterPoint({ line: 10, frequency: "12 / 100 iterations (12%)" })]);

    const hover = provider.provideHover(mockDocument(DOC_URI), mockPosition(9))!;
    const md = hover.contents as unknown as vscode.MarkdownString;
    expect(md.value).toContain("12 / 100 iterations (12%)");
    provider.dispose();
  });

  test("returns first matching shatter point when multiple share the same line", () => {
    const provider = new BlackSwanHoverProvider();
    const sp1 = makeShatterPoint({ id: "sp_001", line: 30, failure_type: "non_psd_matrix" });
    const sp2 = makeShatterPoint({ id: "sp_002", line: 30, failure_type: "nan_inf" });
    provider.set(DOC_URI, [sp1, sp2]);

    const hover = provider.provideHover(mockDocument(DOC_URI), mockPosition(29))!;
    const md = hover.contents as unknown as vscode.MarkdownString;
    // First entry wins.
    expect(md.value).toContain("Non-PSD Matrix");
    provider.dispose();
  });

  test("shatter point with null line never matches any cursor position", () => {
    const provider = new BlackSwanHoverProvider();
    provider.set(DOC_URI, [makeShatterPoint({ line: null })]);

    // Even position 0 should not match — null means no source attribution.
    const hover = provider.provideHover(mockDocument(DOC_URI), mockPosition(0));
    expect(hover).toBeUndefined();
    provider.dispose();
  });

  test("returns undefined after clear() is called for that URI", () => {
    const provider = new BlackSwanHoverProvider();
    provider.set(DOC_URI, [makeShatterPoint({ line: 82 })]);
    provider.clear(DOC_URI);

    const hover = provider.provideHover(mockDocument(DOC_URI), mockPosition(81));
    expect(hover).toBeUndefined();
    provider.dispose();
  });

  test("provides hover for line 1 (minimum 1-indexed line → position 0)", () => {
    const provider = new BlackSwanHoverProvider();
    provider.set(DOC_URI, [makeShatterPoint({ line: 1 })]);

    const hover = provider.provideHover(mockDocument(DOC_URI), mockPosition(0));
    expect(hover).toBeInstanceOf(vscode.Hover);
    provider.dispose();
  });

  test("stores for different URIs are independent", () => {
    const provider = new BlackSwanHoverProvider();
    const uriA = vscode.Uri.file("/workspace/model_a.py");
    const uriB = vscode.Uri.file("/workspace/model_b.py");

    provider.set(uriA, [makeShatterPoint({ line: 10 })]);
    // uriB has no data

    expect(provider.provideHover(mockDocument(uriA), mockPosition(9))).toBeDefined();
    expect(provider.provideHover(mockDocument(uriB), mockPosition(9))).toBeUndefined();
    provider.dispose();
  });
});
