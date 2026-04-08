/**
 * extension.ts — BlackSwan VS Code extension entry point.
 *
 * Activated on `onLanguage:python`. Wires together:
 *   • BlackSwanCodeLensProvider  — injects "▶ Run BlackSwan" buttons
 *   • DiagnosticsController      — owns the red-squiggle DiagnosticCollection
 *   • Orchestrator               — runs the engine, applies results
 *   • blackswan.runStressTest    — command triggered by CodeLens clicks
 *
 * All disposables are pushed to context.subscriptions so VS Code cleans them
 * up automatically on deactivation (no manual teardown needed).
 */

import * as vscode from "vscode";
import {
  BlackSwanCodeLensProvider,
  CODELENS_COMMAND,
} from "./codelens";
import { DiagnosticsController } from "./diagnostics";
import { Orchestrator } from "./orchestrator";

// ---------------------------------------------------------------------------
// Scenario picker items — one entry per preset YAML in core/scenarios/presets/
// ---------------------------------------------------------------------------

const SCENARIO_ITEMS: vscode.QuickPickItem[] = [
  {
    label:       "Liquidity Crash",
    description: "liquidity_crash",
    detail:
      "Spread widening ×1.5–3.5 · Vol expansion (lognormal) · Correlation shift +0.10–0.35 · Turnover –30–70%",
  },
  {
    label:       "Vol Spike",
    description: "vol_spike",
    detail:      "Volatility multiplier ×2–4 · Mild correlation increase +0.05–0.15",
  },
  {
    label:       "Correlation Breakdown",
    description: "correlation_breakdown",
    detail:      "Pairwise correlation shift +0.20–0.50 · Vol increase ×1.2–1.5",
  },
  {
    label:       "Rate Shock",
    description: "rate_shock",
    detail:      "Interest rate shift +100–300 bps · Spread widening +50–150 bps",
  },
  {
    label:       "Missing Data",
    description: "missing_data",
    detail:      "Random NaN injection 5–20% · Partial time-series truncation",
  },
];

// ---------------------------------------------------------------------------
// Activation
// ---------------------------------------------------------------------------

export function activate(context: vscode.ExtensionContext): void {
  // ── Build the component graph ─────────────────────────────────────────────
  const diagnostics  = new DiagnosticsController(context);
  const orchestrator = new Orchestrator(diagnostics);
  const codeLens     = new BlackSwanCodeLensProvider();

  // ── Register CodeLens provider ───────────────────────────────────────────
  context.subscriptions.push(
    vscode.languages.registerCodeLensProvider(
      { language: "python", scheme: "file" },
      codeLens,
    ),
    codeLens,
    orchestrator,
    // DiagnosticsController registers its own disposables via context in its
    // constructor, so we don't push it again here.
  );

  // ── Register the main command ────────────────────────────────────────────
  context.subscriptions.push(
    vscode.commands.registerCommand(
      CODELENS_COMMAND,
      async (uri: vscode.Uri, functionName: string) => {
        // Resolve the TextDocument — openTextDocument is idempotent for already-
        // open files, so this is safe to call even if the file is already open.
        let document: vscode.TextDocument;
        try {
          document = await vscode.workspace.openTextDocument(uri);
        } catch {
          void vscode.window.showErrorMessage(
            `BlackSwan: Could not open document at ${uri.fsPath}`,
          );
          return;
        }

        // Show scenario picker.
        const picked = await vscode.window.showQuickPick(SCENARIO_ITEMS, {
          title:           `BlackSwan — Stress test: ${functionName}`,
          placeHolder:     "Select a stress scenario",
          matchOnDetail:   true,
          matchOnDescription: true,
        });

        if (!picked) return; // user dismissed without selecting

        // `description` is the machine-readable scenario name (e.g. "liquidity_crash").
        const scenarioName = picked.description!;
        await orchestrator.run(document, functionName, scenarioName);
      },
    ),
  );
}

export function deactivate(): void {
  // All resources registered in context.subscriptions are disposed automatically
  // by VS Code — nothing manual required here.
}
