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
import { DagPanelController } from "./dagPanel";
import { DiagnosticsController } from "./diagnostics";
import { BlackSwanHoverProvider } from "./hover";
import { Orchestrator } from "./orchestrator";
import { explainFailure, ExplainPayload, setGeminiApiKey } from "./aiExplainer";
import { applyMathGuard, BlackSwanPreviewProvider } from "./fixer";
import { FailureType } from "./types";

// ---------------------------------------------------------------------------
// Scenario picker items — one entry per preset YAML in core/scenarios/presets/
// ---------------------------------------------------------------------------

const SCENARIO_ITEMS: vscode.QuickPickItem[] = [
  {
    label:       "$(flame)  Liquidity Crash",
    description: "liquidity_crash",
    detail:      "Spread widening ×1.5–3.5  ·  Vol expansion (lognormal)  ·  Correlation shift +0.10–0.35  ·  Turnover –30–70%",
  },
  {
    label:       "$(graph)  Vol Spike",
    description: "vol_spike",
    detail:      "Volatility multiplier ×2–4  ·  Mild correlation increase +0.05–0.15",
  },
  {
    label:       "$(git-pull-request-closed)  Correlation Breakdown",
    description: "correlation_breakdown",
    detail:      "Pairwise correlation shift +0.20–0.50  ·  Vol increase ×1.2–1.5",
  },
  {
    label:       "$(pulse)  Rate Shock",
    description: "rate_shock",
    detail:      "Interest rate shift +100–300 bps  ·  Spread widening +50–150 bps",
  },
  {
    label:       "$(warning)  Missing Data",
    description: "missing_data",
    detail:      "Random NaN injection 5–20%  ·  Partial time-series truncation",
  },
];

// ---------------------------------------------------------------------------
// Activation
// ---------------------------------------------------------------------------

export function activate(context: vscode.ExtensionContext): void {
  // ── Build the component graph ─────────────────────────────────────────────
  const diagnostics      = new DiagnosticsController(context);
  const hover            = new BlackSwanHoverProvider();
  const dag              = new DagPanelController();
  const orchestrator     = new Orchestrator(diagnostics, hover, dag);
  const codeLens         = new BlackSwanCodeLensProvider();
  const previewProvider  = new BlackSwanPreviewProvider();

  // ── Status bar item ──────────────────────────────────────────────────────
  // Shown in the right side of the status bar while a run is active.
  const statusBar = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100,
  );
  statusBar.name    = "BlackSwan";
  statusBar.command = "blackswan.showDag";
  context.subscriptions.push(statusBar);

  function setStatusRunning(fileName: string): void {
    statusBar.text        = "$(sync~spin) BlackSwan";
    statusBar.tooltip     = `Running stress test on ${fileName}…`;
    statusBar.color       = undefined;
    statusBar.backgroundColor = undefined;
    statusBar.show();
  }

  function setStatusResult(failures: number, scenarioLabel: string): void {
    if (failures === 0) {
      statusBar.text            = "$(check) BlackSwan: clean";
      statusBar.tooltip         = `No failures detected — ${scenarioLabel}`;
      statusBar.color           = new vscode.ThemeColor("statusBarItem.prominentForeground");
      statusBar.backgroundColor = undefined;
    } else {
      statusBar.text            = `$(warning) BlackSwan: ${failures} failure${failures === 1 ? "" : "s"}`;
      statusBar.tooltip         = `${failures} failure${failures === 1 ? "" : "s"} detected — ${scenarioLabel}. Click to open graph.`;
      statusBar.backgroundColor = new vscode.ThemeColor("statusBarItem.warningBackground");
      statusBar.color           = undefined;
    }
    statusBar.show();
  }

  function clearStatus(): void {
    statusBar.hide();
  }

  // ── Register providers ───────────────────────────────────────────────────
  context.subscriptions.push(
    vscode.languages.registerCodeLensProvider(
      { language: "python", scheme: "file" },
      codeLens,
    ),
    vscode.languages.registerHoverProvider(
      { language: "python", scheme: "file" },
      hover,
    ),
    vscode.workspace.registerTextDocumentContentProvider(
      "blackswan-preview",
      previewProvider,
    ),
    codeLens,
    hover,
    dag,
    orchestrator,
    previewProvider,
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

        // Show scenario picker with a separator between groups.
        const picked = await vscode.window.showQuickPick(SCENARIO_ITEMS, {
          title:              `$(beaker) BlackSwan — Stress test: ${functionName}`,
          placeHolder:        "Pick a stress scenario  ·  Results open in the DAG panel",
          matchOnDetail:      true,
          matchOnDescription: true,
        });

        if (!picked) return; // user dismissed without selecting

        // `description` is the machine-readable scenario name (e.g. "liquidity_crash").
        const scenarioName  = picked.description!;
        // Strip codicon prefix from the label to get the human-readable name.
        const scenarioLabel = picked.label.replace(/^\$\([^)]+\)\s+/, "");

        setStatusRunning(vscode.workspace.asRelativePath(uri));
        try {
          await orchestrator.run(document, functionName, scenarioName);
          // Read the result from diagnostics to update the status bar.
          const diags = vscode.languages.getDiagnostics(uri);
          const bsDiags = diags.filter((d) => d.source === "BlackSwan");
          setStatusResult(bsDiags.length, scenarioLabel);
        } catch {
          clearStatus();
        }
      },
    ),

    // ── Show-DAG command (triggered by status bar click) ──────────────────
    vscode.commands.registerCommand("blackswan.showDag", () => {
      // Attempt to open the DAG panel for the active Python editor.
      const editor = vscode.window.activeTextEditor;
      if (editor && editor.document.languageId === "python") {
        void vscode.commands.executeCommand("workbench.action.focusSecondEditorGroup");
      }
    }),

    // ── BYOK: store Gemini API key in SecretStorage ───────────────────────
    vscode.commands.registerCommand("blackswan.setApiKey", async () => {
      await setGeminiApiKey(context.secrets);
    }),

    // ── Apply deterministic mathematical guard ────────────────────────────
    //
    // The `uriOrString` argument can be either a proper vscode.Uri (when
    // called from the Problems-panel Quick Fix action, which passes
    // document.uri directly) OR a plain string (when called from a hover
    // tooltip command link, which serialises args through JSON and therefore
    // loses the Uri class).  Both paths are handled below.
    vscode.commands.registerCommand(
      "blackswan.applyGuard",
      async (uriOrString: vscode.Uri | string, line: number, failureType: FailureType) => {
        const uri =
          typeof uriOrString === "string"
            ? vscode.Uri.parse(uriOrString)
            : uriOrString;

        let document: vscode.TextDocument;
        try {
          document = await vscode.workspace.openTextDocument(uri);
        } catch {
          void vscode.window.showErrorMessage(
            `BlackSwan Fix: Could not open document at ${uri.fsPath}`,
          );
          return;
        }

        const pythonPath = _resolvePythonPathForFixer();
        await applyMathGuard(document, line, failureType, pythonPath, previewProvider);
      },
    ),

    // ── Explain failure with Gemini AI ────────────────────────────────────
    vscode.commands.registerCommand(
      "blackswan.explainWithAI",
      async (payload: ExplainPayload) => {
        await explainFailure(context.secrets, payload);
      },
    ),
  );
}

/**
 * Resolve Python executable for the fixer (same logic as Orchestrator._resolvePythonPath).
 * Duplicated here to avoid coupling extension.ts to orchestrator internals.
 */
function _resolvePythonPathForFixer(): string {
  const bsCfg = vscode.workspace.getConfiguration("blackswan");
  const bsPath = bsCfg.get<string>("pythonPath");
  if (bsPath?.trim()) return bsPath.trim();

  const pyCfg = vscode.workspace.getConfiguration("python");
  const interp = pyCfg.get<string>("defaultInterpreterPath");
  if (interp?.trim()) return interp.trim();

  return process.platform === "win32" ? "python" : "python3";
}

export function deactivate(): void {
  // All resources registered in context.subscriptions are disposed automatically
  // by VS Code — nothing manual required here.
}
