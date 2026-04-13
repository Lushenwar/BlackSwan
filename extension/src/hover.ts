/**
 * hover.ts — Hover provider for BlackSwan.
 *
 * Shows a rich MarkdownString tooltip on lines that have BlackSwan shatter
 * points. Content includes: failure type label, engine message, frequency,
 * causal chain with clickable line navigation links, and fix hint.
 *
 * BlackSwanHoverProvider maintains its own shatter-point store (keyed by
 * URI string) because the DiagnosticCollection only carries flattened message
 * strings — it cannot reconstruct the full causal chain needed for a rich
 * tooltip.
 *
 * Integration points:
 *   - Orchestrator calls hover.set()   after each successful engine run.
 *   - Orchestrator calls hover.clear() on cancellation.
 *   - extension.ts registers this provider via vscode.languages.registerHoverProvider.
 *
 * Exported pure function: buildHoverContent — zero VS Code side effects,
 * fully unit-testable without a live extension host.
 */

import * as vscode from "vscode";
import { CausalRole, FailureType, ShatterPoint } from "./types";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Human-readable labels for each failure type — shown in the hover header. */
const FAILURE_TYPE_LABELS: Record<FailureType, string> = {
  nan_inf:                "NaN / Inf Detected",
  division_by_zero:       "Division Instability",
  non_psd_matrix:         "Non-PSD Matrix",
  ill_conditioned_matrix: "Ill-Conditioned Matrix",
  bounds_exceeded:        "Bounds Exceeded",
  division_instability:   "Division Instability",
  exploding_gradient:     "Exploding Gradient",
  regime_shift:           "Regime Shift",
  logical_invariant:      "Logical Invariant Violation",
};

/** Causal role labels — failure_site is bold to make it stand out. */
const ROLE_LABELS: Record<CausalRole, string> = {
  root_input:   "Root input",
  intermediate: "Intermediate",
  failure_site: "**Failure site**",
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Build a markdown command link that navigates to a specific 1-indexed line.
 *
 * Uses the built-in `vscode.open` command with a `selection` argument so the
 * editor scrolls to and highlights the target line when the user clicks.
 *
 * @param uri        Document URI — tells `vscode.open` which file to navigate.
 * @param line1Based Engine line number (1-indexed).
 * @param label      Visible link text (the variable name).
 */
function makeLineLink(
  uri: vscode.Uri,
  line1Based: number,
  label: string,
): string {
  const zero = line1Based - 1;
  const args = encodeURIComponent(
    JSON.stringify([
      uri.toString(),
      {
        selection: {
          start: { line: zero, character: 0 },
          end:   { line: zero, character: 0 },
        },
      },
    ]),
  );
  return `[\`${label}\` (line ${line1Based})](command:vscode.open?${args} "Go to line ${line1Based}")`;
}

// ---------------------------------------------------------------------------
// Pure content builder
// ---------------------------------------------------------------------------

/**
 * Build the MarkdownString hover content for a single shatter point.
 *
 * Produces a trusted MarkdownString so `command://` URIs in causal-chain links
 * are clickable (VS Code requires `isTrusted = true` for command links).
 *
 * @param sp           Shatter point from the engine.
 * @param documentUri  URI of the document that contains the failure — used to
 *                     build `vscode.open` command links in the causal chain.
 * @param getLineText  Optional: given a 1-indexed line number, returns the raw
 *                     source text for that line.  When provided, each causal-
 *                     chain node shows the actual code at that line as an
 *                     inline snippet so the user can trace the propagation path
 *                     without jumping between lines.  Safe to omit in tests.
 */
export function buildHoverContent(
  sp: ShatterPoint,
  documentUri: vscode.Uri,
  getLineText?: (line1Based: number) => string | undefined,
): vscode.MarkdownString {
  const md = new vscode.MarkdownString("");
  md.isTrusted = true;   // Required for command:// URIs in links.
  md.supportHtml = false;

  const typeLabel = FAILURE_TYPE_LABELS[sp.failure_type] ?? sp.failure_type;

  // ── Header ──────────────────────────────────────────────────────────────
  md.appendMarkdown(`## ⚠ BlackSwan: ${typeLabel}\n\n`);

  // ── Engine failure message ───────────────────────────────────────────────
  md.appendMarkdown(`${sp.message}\n\n`);

  // ── Frequency ───────────────────────────────────────────────────────────
  md.appendMarkdown(`**Frequency:** ${sp.frequency}\n\n`);

  // ── Causal chain ────────────────────────────────────────────────────────
  if (sp.causal_chain.length > 0) {
    md.appendMarkdown(`**Causal Chain:**\n\n`);
    for (const link of sp.causal_chain) {
      const roleLabel = ROLE_LABELS[link.role] ?? link.role;
      const lineLink  = makeLineLink(documentUri, link.line, link.variable);

      // Pull the raw source for this line so the user can see exactly what
      // code is at each propagation step without leaving the hover.
      // Trim leading/trailing whitespace and cap at 80 chars so long
      // expressions don't overflow the hover card.
      const rawText  = getLineText?.(link.line)?.trim() ?? "";
      const snippet  = rawText
        ? ` — \`${rawText.length > 80 ? rawText.slice(0, 80) + "…" : rawText}\``
        : "";

      md.appendMarkdown(`- ${roleLabel} → ${lineLink}${snippet}\n`);
    }
    md.appendMarkdown("\n");
  }

  // ── Fix hint + clickable guard action ────────────────────────────────────
  if (sp.fix_hint) {
    md.appendMarkdown(`---\n\n💡 **Fix Hint:** ${sp.fix_hint}\n`);

    // Only show the apply button when we have a concrete line to fix.
    // Failure types that have no deterministic guard (e.g. bounds_exceeded,
    // regime_shift, logical_invariant) will still show the hint text above.
    const GUARDABLE: ReadonlySet<string> = new Set([
      "division_instability",
      "division_by_zero",
      "non_psd_matrix",
      "ill_conditioned_matrix",
      "nan_inf",
    ]);

    if (sp.line !== null && GUARDABLE.has(sp.failure_type)) {
      // Encode args as a JSON array for the command: URI scheme.
      // VS Code deserialises these as plain JS values, so vscode.Uri objects
      // must be passed as strings — the command handler accepts both.
      const args = encodeURIComponent(
        JSON.stringify([documentUri.toString(), sp.line, sp.failure_type]),
      );
      md.appendMarkdown(
        `\n\n[$(shield) Apply Mathematical Guard](command:blackswan.applyGuard?${args}` +
          ` "Run the deterministic fixer and preview the diff")`,
      );
    }
  }

  return md;
}

// ---------------------------------------------------------------------------
// BlackSwanHoverProvider
// ---------------------------------------------------------------------------

/**
 * Hover provider that surfaces BlackSwan results as rich tooltips.
 *
 * Owns a shatter-point store keyed by `uri.toString()`. Orchestrator updates
 * it via `set()` / `clear()` as runs complete or are cancelled.
 *
 * Registered in extension.ts for `{ language: "python", scheme: "file" }`.
 */
export class BlackSwanHoverProvider
  implements vscode.HoverProvider, vscode.Disposable
{
  /** Shatter points keyed by `uri.toString()`. */
  private readonly _store = new Map<string, ShatterPoint[]>();

  // ---------------------------------------------------------------------------
  // Store management — called by Orchestrator
  // ---------------------------------------------------------------------------

  /**
   * Replace the shatter points for a document.
   * Each call is a full replace — safe to call repeatedly for the same URI.
   */
  set(uri: vscode.Uri, points: ShatterPoint[]): void {
    this._store.set(uri.toString(), points);
  }

  /**
   * Remove shatter points for a single document.
   * Called on run cancellation so stale hovers don't linger.
   */
  clear(uri: vscode.Uri): void {
    this._store.delete(uri.toString());
  }

  /** Remove all stored shatter points across all documents. */
  clearAll(): void {
    this._store.clear();
  }

  // ---------------------------------------------------------------------------
  // HoverProvider
  // ---------------------------------------------------------------------------

  /**
   * Return a hover for the first shatter point whose line matches the cursor.
   *
   * Engine lines are 1-indexed; VS Code `position.line` is 0-indexed.
   * Returns `undefined` (no hover shown) when no shatter point covers the
   * cursor position or when no data has been stored for the document.
   */
  provideHover(
    document: vscode.TextDocument,
    position: vscode.Position,
  ): vscode.Hover | undefined {
    const points = this._store.get(document.uri.toString());
    if (!points || points.length === 0) return undefined;

    // Match cursor line (0-indexed) to shatter point line (1-indexed).
    // Skip shatter points with null lines — they have no source attribution.
    const match = points.find(
      (sp) => sp.line !== null && sp.line - 1 === position.line,
    );
    if (!match) return undefined;

    // Pass a live line reader so each causal-chain node shows its source text.
    // The callback is defined inline here to keep buildHoverContent free of
    // any direct dependency on vscode.TextDocument (it stays pure / testable).
    const getLineText = (line1Based: number): string | undefined => {
      const idx = line1Based - 1;
      if (idx < 0 || idx >= document.lineCount) return undefined;
      return document.lineAt(idx).text;
    };

    return new vscode.Hover(buildHoverContent(match, document.uri, getLineText));
  }

  dispose(): void {
    this._store.clear();
  }
}
