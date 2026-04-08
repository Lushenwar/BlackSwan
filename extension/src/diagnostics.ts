/**
 * diagnostics.ts — VS Code Diagnostics integration for BlackSwan.
 *
 * Converts BlackSwanResponse.shatter_points into VS Code Diagnostics,
 * manages the DiagnosticCollection lifecycle, and registers a
 * CodeActionProvider that surfaces fix_hint as a Quick Fix.
 *
 * The pure conversion functions (mapSeverity, shatterPointToDiagnostic,
 * buildDiagnostics) carry no side effects and are exported for unit testing
 * without a live extension host.
 */

import * as vscode from "vscode";
import {
  BlackSwanResponse,
  FailureType,
  Severity,
  ShatterPoint,
} from "./types";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Source tag shown in the Problems panel and on inline squiggles. */
export const DIAGNOSTIC_SOURCE = "BlackSwan";

/** Human-readable labels for each failure type — shown in the Problems panel. */
const FAILURE_TYPE_LABELS: Record<FailureType, string> = {
  nan_inf:                "NaN / Inf detected",
  division_by_zero:       "Division instability",
  non_psd_matrix:         "Non-PSD matrix",
  ill_conditioned_matrix: "Ill-conditioned matrix",
  bounds_exceeded:        "Bounds exceeded",
};

/**
 * Data payload attached to diagnostics that have a fix_hint.
 * Read by BlackSwanCodeActionProvider to surface the Quick Fix.
 */
export interface DiagnosticFixData {
  shatterPointId: string;
  fixHint: string;
}

// Internal type alias so we can attach data without widening the public API.
type DiagnosticWithHint = vscode.Diagnostic & { data?: DiagnosticFixData };

// ---------------------------------------------------------------------------
// Pure conversion helpers
// ---------------------------------------------------------------------------

/**
 * Map a BlackSwan Severity string to the matching VS Code DiagnosticSeverity.
 *
 * critical → Error   (model produces incorrect results under stress)
 * warning  → Warning (potential instability, not a definite break)
 * info     → Information (advisory note)
 */
export function mapSeverity(severity: Severity): vscode.DiagnosticSeverity {
  switch (severity) {
    case "critical": return vscode.DiagnosticSeverity.Error;
    case "warning":  return vscode.DiagnosticSeverity.Warning;
    case "info":     return vscode.DiagnosticSeverity.Information;
  }
}

/**
 * Convert a single ShatterPoint to a VS Code Diagnostic.
 *
 * @param sp           Shatter point from the engine.
 * @param uri          Document URI — used for causal-chain RelatedInformation links.
 * @param lineCount    Total lines in the document; out-of-range engine lines are clamped.
 *                     Pass Infinity in tests to disable clamping.
 * @param scenarioName Scenario that produced this failure (included in the message).
 */
export function shatterPointToDiagnostic(
  sp: ShatterPoint,
  uri: vscode.Uri,
  lineCount: number,
  scenarioName: string,
): vscode.Diagnostic {
  // Engine lines are 1-indexed; VS Code Ranges are 0-indexed.
  // Clamp to [0, lineCount - 1] to guard against off-by-one or attribution gaps.
  const maxLine = Math.max(lineCount - 1, 0);
  const zeroLine = sp.line !== null
    ? Math.min(Math.max(sp.line - 1, 0), maxLine)
    : 0;
  const zeroCol = sp.column !== null ? sp.column : 0;

  // Underline from the attributed column to end-of-line.
  // Using Number.MAX_SAFE_INTEGER so VS Code clips to the actual line length.
  const range = new vscode.Range(
    zeroLine, zeroCol,
    zeroLine, Number.MAX_SAFE_INTEGER,
  );

  // Build the Problems-panel message.
  //   [BlackSwan] <TypeLabel>: <message>  •  <frequency>  [scenario: <name>]
  //
  // The [BlackSwan] prefix is intentional: VS Code shows `source` in the
  // Problems panel but NOT in inline squiggle annotations, so the prefix
  // lets users identify the origin at a glance in the editor gutter.
  const typeLabel = FAILURE_TYPE_LABELS[sp.failure_type] ?? sp.failure_type;
  const message = `[BlackSwan] ${typeLabel}: ${sp.message}  •  ${sp.frequency}  [scenario: ${scenarioName}]`;

  const diagnostic: DiagnosticWithHint = new vscode.Diagnostic(
    range,
    message,
    mapSeverity(sp.severity),
  );
  diagnostic.source = DIAGNOSTIC_SOURCE;
  // The failure_type code enables filtering in the Problems panel and lets
  // third-party tools (e.g., CI reporters) machine-read the failure category.
  diagnostic.code = sp.failure_type;

  // Causal chain → clickable RelatedInformation links in the Problems panel.
  // Each entry navigates to the exact line of the contributing variable.
  if (sp.causal_chain.length > 0) {
    diagnostic.relatedInformation = sp.causal_chain.map((link) => {
      const chainLine = Math.min(Math.max(link.line - 1, 0), maxLine);
      const chainRange = new vscode.Range(
        chainLine, 0,
        chainLine, Number.MAX_SAFE_INTEGER,
      );
      const roleLabel =
        link.role === "root_input"   ? "Root input"    :
        link.role === "intermediate" ? "Intermediate"  :
                                       "Failure site";
      return new vscode.DiagnosticRelatedInformation(
        new vscode.Location(uri, chainRange),
        `${roleLabel}: ${link.variable}`,
      );
    });
  }

  // Attach fix_hint as a structured data payload.
  // BlackSwanCodeActionProvider reads this to build the Quick Fix action.
  if (sp.fix_hint) {
    diagnostic.data = {
      shatterPointId: sp.id,
      fixHint: sp.fix_hint,
    };
  }

  return diagnostic;
}

/**
 * Build the full Diagnostic[] for a document from a BlackSwanResponse.
 * Every shatter point in the response produces exactly one Diagnostic.
 */
export function buildDiagnostics(
  uri: vscode.Uri,
  response: BlackSwanResponse,
  lineCount: number,
): vscode.Diagnostic[] {
  return response.shatter_points.map((sp) =>
    shatterPointToDiagnostic(sp, uri, lineCount, response.scenario_card.name),
  );
}

// ---------------------------------------------------------------------------
// CodeActionProvider — surfaces fix_hint as a Quick Fix comment.
// ---------------------------------------------------------------------------

/**
 * Provides a Quick Fix code action for every BlackSwan diagnostic that
 * carries a non-empty fix_hint.
 *
 * The action inserts a comment above the failing line:
 *   # BlackSwan Fix Hint: <hint text>
 *
 * A comment is used intentionally: fix hints require human judgment (e.g.,
 * "apply Higham 2002 nearest-PSD correction") — auto-editing the code would
 * be unsafe. The comment prompts the developer without making assumptions.
 */
export class BlackSwanCodeActionProvider implements vscode.CodeActionProvider {
  static readonly providedCodeActionKinds = [vscode.CodeActionKind.QuickFix];

  provideCodeActions(
    document: vscode.TextDocument,
    _range: vscode.Range | vscode.Selection,
    context: vscode.CodeActionContext,
  ): vscode.CodeAction[] {
    const actions: vscode.CodeAction[] = [];

    for (const diag of context.diagnostics) {
      if (diag.source !== DIAGNOSTIC_SOURCE) continue;
      const withHint = diag as DiagnosticWithHint;
      if (!withHint.data?.fixHint) continue;

      const hint = withHint.data.fixHint;
      const action = new vscode.CodeAction(
        `BlackSwan Fix Hint: ${hint}`,
        vscode.CodeActionKind.QuickFix,
      );
      action.diagnostics = [diag];
      // isPreferred = false: this is a suggestion, not a safe auto-fix.
      action.isPreferred = false;

      // Detect indentation of the failing line and match it in the comment.
      const targetLine = diag.range.start.line;
      const lineText = document.lineAt(targetLine).text;
      const indentMatch = lineText.match(/^(\s*)/);
      const indent = indentMatch ? indentMatch[1] : "";
      const commentText = `${indent}# BlackSwan Fix Hint: ${hint}\n`;

      action.edit = new vscode.WorkspaceEdit();
      action.edit.insert(
        document.uri,
        new vscode.Position(targetLine, 0),
        commentText,
      );

      actions.push(action);
    }

    return actions;
  }
}

// ---------------------------------------------------------------------------
// DiagnosticsController — owns the DiagnosticCollection lifetime.
// ---------------------------------------------------------------------------

/**
 * Central controller for BlackSwan diagnostics. One instance per extension
 * activation.
 *
 * Usage pattern in extension.ts:
 *
 *   const diagnostics = new DiagnosticsController(context);
 *   // ...after a successful engine run:
 *   diagnostics.apply(document, response);
 *   // ...on cancel or close:
 *   diagnostics.clear(document.uri);
 *
 * Passing `context` is optional but recommended: it registers the internal
 * disposables with VS Code's extension lifecycle so cleanup is automatic.
 */
export class DiagnosticsController implements vscode.Disposable {
  private readonly _collection: vscode.DiagnosticCollection;
  private readonly _disposables: vscode.Disposable[] = [];

  constructor(context?: vscode.ExtensionContext) {
    this._collection = vscode.languages.createDiagnosticCollection(DIAGNOSTIC_SOURCE);

    const provider = vscode.languages.registerCodeActionsProvider(
      { language: "python", scheme: "file" },
      new BlackSwanCodeActionProvider(),
      {
        providedCodeActionKinds: BlackSwanCodeActionProvider.providedCodeActionKinds,
      },
    );

    this._disposables.push(this._collection, provider);

    if (context) {
      context.subscriptions.push(...this._disposables);
    }
  }

  /**
   * Apply engine results to a document, replacing any previous diagnostics
   * for that URI. Safe to call multiple times — each call is a full replace.
   */
  apply(document: vscode.TextDocument, response: BlackSwanResponse): void {
    const diagnostics = buildDiagnostics(
      document.uri,
      response,
      document.lineCount,
    );
    this._collection.set(document.uri, diagnostics);
  }

  /** Remove all BlackSwan diagnostics for a single document URI. */
  clear(uri: vscode.Uri): void {
    this._collection.delete(uri);
  }

  /** Remove all BlackSwan diagnostics across all open documents. */
  clearAll(): void {
    this._collection.clear();
  }

  /** Release the DiagnosticCollection and CodeActionProvider registration. */
  dispose(): void {
    for (const d of this._disposables) {
      d.dispose();
    }
  }
}
