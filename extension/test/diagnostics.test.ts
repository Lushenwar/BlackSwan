/**
 * Unit tests for diagnostics.ts.
 *
 * These tests exercise:
 *   - mapSeverity        — pure severity mapping
 *   - shatterPointToDiagnostic — full field conversion
 *   - buildDiagnostics   — response → Diagnostic[] conversion
 *   - BlackSwanCodeActionProvider — Quick Fix code actions
 *
 * The vscode module is resolved to __mocks__/vscode.ts by the Jest
 * moduleNameMapper in package.json. No live extension host is needed.
 */

import * as vscode from "vscode";
import {
  DIAGNOSTIC_SOURCE,
  DiagnosticFixData,
  BlackSwanCodeActionProvider,
  buildDiagnostics,
  mapSeverity,
  shatterPointToDiagnostic,
} from "../src/diagnostics";
import { BlackSwanResponse, ShatterPoint } from "../src/types";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const TEST_URI = vscode.Uri.file("/workspace/model.py");

/** Minimal valid shatter point with all optional fields populated. */
function makeShatterPoint(overrides: Partial<ShatterPoint> = {}): ShatterPoint {
  return {
    id: "sp_001",
    line: 82,
    column: 4,
    severity: "critical",
    failure_type: "non_psd_matrix",
    message: "Covariance matrix loses positive semi-definiteness",
    frequency: "847 / 5000 iterations (16.9%)",
    causal_chain: [
      { line: 14, variable: "corr_shift",           role: "root_input"   },
      { line: 47, variable: "adjusted_corr_matrix", role: "intermediate" },
      { line: 82, variable: "cov_matrix",           role: "failure_site" },
    ],
    fix_hint: "Apply nearest-PSD correction (Higham 2002) after correlation perturbation",
    ...overrides,
  };
}

/** Minimal valid BlackSwanResponse with one shatter point. */
function makeResponse(
  shatterPoints: ShatterPoint[] = [makeShatterPoint()],
  overrides: Partial<BlackSwanResponse> = {},
): BlackSwanResponse {
  return {
    version: "1.0",
    status: "failures_detected",
    runtime_ms: 3420,
    iterations_completed: 5000,
    summary: {
      total_failures: 847,
      failure_rate: 0.1694,
      unique_failure_types: 1,
    },
    shatter_points: shatterPoints,
    scenario_card: {
      name: "liquidity_crash",
      parameters_applied: { spread_widening_bps: 250 },
      seed: 42,
      reproducible: true,
    },
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// mapSeverity
// ---------------------------------------------------------------------------

describe("mapSeverity", () => {
  test("maps critical → DiagnosticSeverity.Error (0)", () => {
    expect(mapSeverity("critical")).toBe(vscode.DiagnosticSeverity.Error);
  });

  test("maps warning → DiagnosticSeverity.Warning (1)", () => {
    expect(mapSeverity("warning")).toBe(vscode.DiagnosticSeverity.Warning);
  });

  test("maps info → DiagnosticSeverity.Information (2)", () => {
    expect(mapSeverity("info")).toBe(vscode.DiagnosticSeverity.Information);
  });
});

// ---------------------------------------------------------------------------
// shatterPointToDiagnostic — range construction
// ---------------------------------------------------------------------------

describe("shatterPointToDiagnostic — range", () => {
  test("converts 1-indexed line to 0-indexed range start", () => {
    const sp = makeShatterPoint({ line: 82, column: 4 });
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 200, "liquidity_crash");
    expect(diag.range.start.line).toBe(81);
    expect(diag.range.start.character).toBe(4);
  });

  test("range ends on the same line (single-line underline)", () => {
    const sp = makeShatterPoint({ line: 10, column: 0 });
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 100, "liquidity_crash");
    expect(diag.range.end.line).toBe(9);
  });

  test("range end character is Number.MAX_SAFE_INTEGER (underlines to EOL)", () => {
    const sp = makeShatterPoint({ line: 1, column: 0 });
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 50, "liquidity_crash");
    expect(diag.range.end.character).toBe(Number.MAX_SAFE_INTEGER);
  });

  test("null line falls back to line 0", () => {
    const sp = makeShatterPoint({ line: null });
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 100, "liquidity_crash");
    expect(diag.range.start.line).toBe(0);
  });

  test("null column falls back to column 0", () => {
    const sp = makeShatterPoint({ column: null });
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 100, "liquidity_crash");
    expect(diag.range.start.character).toBe(0);
  });

  test("out-of-range line is clamped to last line of document", () => {
    // Engine reports line 1000 but document only has 50 lines.
    const sp = makeShatterPoint({ line: 1000 });
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 50, "liquidity_crash");
    expect(diag.range.start.line).toBe(49);
  });

  test("line 0 from engine (off-by-one guard) is clamped to 0", () => {
    const sp = makeShatterPoint({ line: 0 });
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 100, "liquidity_crash");
    expect(diag.range.start.line).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// shatterPointToDiagnostic — message and metadata
// ---------------------------------------------------------------------------

describe("shatterPointToDiagnostic — message and metadata", () => {
  test("message includes the [BlackSwan] prefix", () => {
    const sp = makeShatterPoint();
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 200, "liquidity_crash");
    expect(diag.message).toMatch(/^\[BlackSwan\]/);
  });

  test("message includes the human-readable failure type label", () => {
    const sp = makeShatterPoint({ failure_type: "non_psd_matrix" });
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 200, "liquidity_crash");
    expect(diag.message).toContain("Non-PSD matrix");
  });

  test("message includes the engine's failure message text", () => {
    const sp = makeShatterPoint({
      message: "Covariance matrix loses positive semi-definiteness",
    });
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 200, "liquidity_crash");
    expect(diag.message).toContain(
      "Covariance matrix loses positive semi-definiteness",
    );
  });

  test("message includes the frequency string", () => {
    const sp = makeShatterPoint({ frequency: "847 / 5000 iterations (16.9%)" });
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 200, "liquidity_crash");
    expect(diag.message).toContain("847 / 5000 iterations (16.9%)");
  });

  test("message includes the scenario name", () => {
    const sp = makeShatterPoint();
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 200, "vol_spike");
    expect(diag.message).toContain("vol_spike");
  });

  test("source is set to DIAGNOSTIC_SOURCE constant", () => {
    const sp = makeShatterPoint();
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 200, "liquidity_crash");
    expect(diag.source).toBe(DIAGNOSTIC_SOURCE);
    expect(diag.source).toBe("BlackSwan");
  });

  test("code is set to the failure_type string", () => {
    const sp = makeShatterPoint({ failure_type: "ill_conditioned_matrix" });
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 200, "liquidity_crash");
    expect(diag.code).toBe("ill_conditioned_matrix");
  });

  test("severity is Error for critical shatter points", () => {
    const sp = makeShatterPoint({ severity: "critical" });
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 200, "liquidity_crash");
    expect(diag.severity).toBe(vscode.DiagnosticSeverity.Error);
  });

  test("severity is Warning for warning shatter points", () => {
    const sp = makeShatterPoint({ severity: "warning" });
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 200, "liquidity_crash");
    expect(diag.severity).toBe(vscode.DiagnosticSeverity.Warning);
  });

  test("severity is Information for info shatter points", () => {
    const sp = makeShatterPoint({ severity: "info" });
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 200, "liquidity_crash");
    expect(diag.severity).toBe(vscode.DiagnosticSeverity.Information);
  });

  test("message contains labels for all 5 failure types", () => {
    const types: Array<ShatterPoint["failure_type"]> = [
      "nan_inf",
      "division_by_zero",
      "non_psd_matrix",
      "ill_conditioned_matrix",
      "bounds_exceeded",
    ];
    const expectedLabels = [
      "NaN / Inf detected",
      "Division instability",
      "Non-PSD matrix",
      "Ill-conditioned matrix",
      "Bounds exceeded",
    ];
    for (let i = 0; i < types.length; i++) {
      const sp = makeShatterPoint({ failure_type: types[i] });
      const diag = shatterPointToDiagnostic(sp, TEST_URI, 200, "liquidity_crash");
      expect(diag.message).toContain(expectedLabels[i]);
    }
  });
});

// ---------------------------------------------------------------------------
// shatterPointToDiagnostic — causal chain → RelatedInformation
// ---------------------------------------------------------------------------

describe("shatterPointToDiagnostic — causal chain", () => {
  test("produces relatedInformation for each causal chain link", () => {
    const sp = makeShatterPoint();
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 200, "liquidity_crash");
    expect(diag.relatedInformation).toHaveLength(3);
  });

  test("root_input link has 'Root input' label", () => {
    const sp = makeShatterPoint();
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 200, "liquidity_crash");
    const rootLink = diag.relatedInformation![0];
    expect(rootLink.message).toContain("Root input");
    expect(rootLink.message).toContain("corr_shift");
  });

  test("intermediate link has 'Intermediate' label", () => {
    const sp = makeShatterPoint();
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 200, "liquidity_crash");
    const midLink = diag.relatedInformation![1];
    expect(midLink.message).toContain("Intermediate");
    expect(midLink.message).toContain("adjusted_corr_matrix");
  });

  test("failure_site link has 'Failure site' label", () => {
    const sp = makeShatterPoint();
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 200, "liquidity_crash");
    const siteLink = diag.relatedInformation![2];
    expect(siteLink.message).toContain("Failure site");
    expect(siteLink.message).toContain("cov_matrix");
  });

  test("causal chain links point to the correct (0-indexed) lines", () => {
    const sp = makeShatterPoint(); // chain lines: 14, 47, 82
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 200, "liquidity_crash");
    expect(diag.relatedInformation![0].location.range.start.line).toBe(13);
    expect(diag.relatedInformation![1].location.range.start.line).toBe(46);
    expect(diag.relatedInformation![2].location.range.start.line).toBe(81);
  });

  test("causal chain links reference the document URI", () => {
    const sp = makeShatterPoint();
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 200, "liquidity_crash");
    for (const rel of diag.relatedInformation!) {
      expect(rel.location.uri.toString()).toBe(TEST_URI.toString());
    }
  });

  test("empty causal chain produces no relatedInformation", () => {
    const sp = makeShatterPoint({ causal_chain: [] });
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 200, "liquidity_crash");
    expect(diag.relatedInformation).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// shatterPointToDiagnostic — fix_hint → data payload
// ---------------------------------------------------------------------------

describe("shatterPointToDiagnostic — fix hint data", () => {
  test("data.fixHint is set when fix_hint is non-empty", () => {
    const sp = makeShatterPoint({
      fix_hint: "Apply Higham 2002 nearest-PSD correction",
    });
    const diag = shatterPointToDiagnostic(
      sp, TEST_URI, 200, "liquidity_crash",
    ) as vscode.Diagnostic & { data?: DiagnosticFixData };
    expect(diag.data?.fixHint).toBe("Apply Higham 2002 nearest-PSD correction");
  });

  test("data.shatterPointId matches the shatter point id", () => {
    const sp = makeShatterPoint({ id: "sp_003", fix_hint: "Some hint" });
    const diag = shatterPointToDiagnostic(
      sp, TEST_URI, 200, "liquidity_crash",
    ) as vscode.Diagnostic & { data?: DiagnosticFixData };
    expect(diag.data?.shatterPointId).toBe("sp_003");
  });

  test("data is undefined when fix_hint is empty string", () => {
    const sp = makeShatterPoint({ fix_hint: "" });
    const diag = shatterPointToDiagnostic(
      sp, TEST_URI, 200, "liquidity_crash",
    ) as vscode.Diagnostic & { data?: DiagnosticFixData };
    expect(diag.data).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// buildDiagnostics
// ---------------------------------------------------------------------------

describe("buildDiagnostics", () => {
  test("returns one Diagnostic per shatter point", () => {
    const response = makeResponse([
      makeShatterPoint({ id: "sp_001" }),
      makeShatterPoint({ id: "sp_002", failure_type: "nan_inf" }),
    ]);
    const diags = buildDiagnostics(TEST_URI, response, 200);
    expect(diags).toHaveLength(2);
  });

  test("returns empty array when there are no shatter points", () => {
    const response = makeResponse([], {
      status: "no_failures",
      summary: { total_failures: 0, failure_rate: 0, unique_failure_types: 0 },
    });
    const diags = buildDiagnostics(TEST_URI, response, 200);
    expect(diags).toHaveLength(0);
  });

  test("all diagnostics have source set to BlackSwan", () => {
    const response = makeResponse([
      makeShatterPoint({ id: "sp_001" }),
      makeShatterPoint({ id: "sp_002" }),
    ]);
    const diags = buildDiagnostics(TEST_URI, response, 200);
    for (const d of diags) {
      expect(d.source).toBe(DIAGNOSTIC_SOURCE);
    }
  });

  test("scenario name in each diagnostic message matches scenario_card.name", () => {
    const response = makeResponse(
      [makeShatterPoint()],
      { scenario_card: { name: "rate_shock", parameters_applied: {}, seed: 7, reproducible: true } },
    );
    const diags = buildDiagnostics(TEST_URI, response, 200);
    expect(diags[0].message).toContain("rate_shock");
  });

  test("preserves shatter point order (most severe first)", () => {
    const sp1 = makeShatterPoint({ id: "sp_001", failure_type: "non_psd_matrix" });
    const sp2 = makeShatterPoint({ id: "sp_002", failure_type: "nan_inf", severity: "warning" });
    const response = makeResponse([sp1, sp2]);
    const diags = buildDiagnostics(TEST_URI, response, 200);
    expect(diags[0].code).toBe("non_psd_matrix");
    expect(diags[1].code).toBe("nan_inf");
  });
});

// ---------------------------------------------------------------------------
// BlackSwanCodeActionProvider
// ---------------------------------------------------------------------------

describe("BlackSwanCodeActionProvider", () => {
  /** Create a minimal mock TextDocument. */
  function mockDocument(lineText = "    cov_matrix = np.dot(adjusted_corr, adjusted_corr.T)"): vscode.TextDocument {
    return {
      uri: TEST_URI,
      lineCount: 100,
      lineAt: jest.fn((_line: number) => ({ text: lineText })),
    } as unknown as vscode.TextDocument;
  }

  /** Build a Diagnostic that carries fix hint data. */
  function diagWithHint(hint: string): vscode.Diagnostic & { data?: DiagnosticFixData } {
    const sp = makeShatterPoint({ fix_hint: hint });
    return shatterPointToDiagnostic(sp, TEST_URI, 100, "liquidity_crash") as
      vscode.Diagnostic & { data?: DiagnosticFixData };
  }

  const provider = new BlackSwanCodeActionProvider();
  const dummyRange = new vscode.Range(81, 0, 81, 9999);

  test("returns a code action for a diagnostic with a fix hint", () => {
    const diag = diagWithHint("Apply Higham 2002 correction");
    const actions = provider.provideCodeActions(
      mockDocument(),
      dummyRange,
      { diagnostics: [diag], only: undefined, triggerKind: 1 } as unknown as vscode.CodeActionContext,
    );
    expect(actions).toHaveLength(1);
  });

  test("action title contains the fix hint text", () => {
    const diag = diagWithHint("Clamp eigenvalues to epsilon");
    const [action] = provider.provideCodeActions(
      mockDocument(),
      dummyRange,
      { diagnostics: [diag], only: undefined, triggerKind: 1 } as unknown as vscode.CodeActionContext,
    );
    expect(action.title).toContain("Clamp eigenvalues to epsilon");
  });

  test("action title starts with 'BlackSwan Fix Hint:'", () => {
    const diag = diagWithHint("Some hint");
    const [action] = provider.provideCodeActions(
      mockDocument(),
      dummyRange,
      { diagnostics: [diag], only: undefined, triggerKind: 1 } as unknown as vscode.CodeActionContext,
    );
    expect(action.title).toMatch(/^BlackSwan Fix Hint:/);
  });

  test("action.edit inserts a comment at the start of the failing line", () => {
    const diag = diagWithHint("Apply nearest-PSD correction");
    const [action] = provider.provideCodeActions(
      mockDocument("    cov_matrix = x"),
      dummyRange,
      { diagnostics: [diag], only: undefined, triggerKind: 1 } as unknown as vscode.CodeActionContext,
    );
    const edit = action.edit as unknown as {
      getInserts(): Array<{ uri: vscode.Uri; position: vscode.Position; text: string }>;
    };
    const inserts = edit.getInserts();
    expect(inserts).toHaveLength(1);
    expect(inserts[0].text).toMatch(/^    # BlackSwan Fix Hint:/);
    expect(inserts[0].text).toContain("Apply nearest-PSD correction");
    expect(inserts[0].position.line).toBe(diag.range.start.line);
    expect(inserts[0].position.character).toBe(0);
  });

  test("inserted comment preserves line indentation", () => {
    const diag = diagWithHint("Some fix");
    const [action] = provider.provideCodeActions(
      mockDocument("        deeply_indented = True"),
      dummyRange,
      { diagnostics: [diag], only: undefined, triggerKind: 1 } as unknown as vscode.CodeActionContext,
    );
    const edit = action.edit as unknown as {
      getInserts(): Array<{ uri: vscode.Uri; position: vscode.Position; text: string }>;
    };
    const [insert] = edit.getInserts();
    expect(insert.text).toMatch(/^        # BlackSwan/);
  });

  test("action.isPreferred is false (hint requires human judgment)", () => {
    const diag = diagWithHint("Some fix");
    const [action] = provider.provideCodeActions(
      mockDocument(),
      dummyRange,
      { diagnostics: [diag], only: undefined, triggerKind: 1 } as unknown as vscode.CodeActionContext,
    );
    expect(action.isPreferred).toBe(false);
  });

  test("action.diagnostics references the source diagnostic", () => {
    const diag = diagWithHint("Some fix");
    const [action] = provider.provideCodeActions(
      mockDocument(),
      dummyRange,
      { diagnostics: [diag], only: undefined, triggerKind: 1 } as unknown as vscode.CodeActionContext,
    );
    expect(action.diagnostics).toContain(diag);
  });

  test("returns no actions when diagnostic has no fix hint", () => {
    const sp = makeShatterPoint({ fix_hint: "" });
    const diag = shatterPointToDiagnostic(sp, TEST_URI, 100, "liquidity_crash");
    const actions = provider.provideCodeActions(
      mockDocument(),
      dummyRange,
      { diagnostics: [diag], only: undefined, triggerKind: 1 } as unknown as vscode.CodeActionContext,
    );
    expect(actions).toHaveLength(0);
  });

  test("ignores diagnostics from other sources", () => {
    const foreignDiag = new vscode.Diagnostic(
      new vscode.Range(0, 0, 0, 10),
      "Some other tool's error",
      vscode.DiagnosticSeverity.Error,
    );
    foreignDiag.source = "Pylint";
    (foreignDiag as vscode.Diagnostic & { data?: DiagnosticFixData }).data = {
      shatterPointId: "sp_999",
      fixHint: "This should be ignored",
    };
    const actions = provider.provideCodeActions(
      mockDocument(),
      dummyRange,
      { diagnostics: [foreignDiag], only: undefined, triggerKind: 1 } as unknown as vscode.CodeActionContext,
    );
    expect(actions).toHaveLength(0);
  });

  test("handles mixed diagnostics — only BlackSwan ones get actions", () => {
    const blackSwanDiag = diagWithHint("Fix PSD");
    const pylintDiag = new vscode.Diagnostic(
      new vscode.Range(0, 0, 0, 10),
      "pylint: undefined variable",
      vscode.DiagnosticSeverity.Warning,
    );
    pylintDiag.source = "Pylint";
    const actions = provider.provideCodeActions(
      mockDocument(),
      dummyRange,
      { diagnostics: [blackSwanDiag, pylintDiag], only: undefined, triggerKind: 1 } as unknown as vscode.CodeActionContext,
    );
    expect(actions).toHaveLength(1);
  });

  test("providedCodeActionKinds declares QuickFix", () => {
    expect(BlackSwanCodeActionProvider.providedCodeActionKinds).toContain(
      vscode.CodeActionKind.QuickFix,
    );
  });
});
