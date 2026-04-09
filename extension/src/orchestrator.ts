/**
 * orchestrator.ts — Execution orchestrator for BlackSwan stress tests.
 *
 * Owns:
 *   • The per-document running-state mutex (prevents concurrent runs).
 *   • Progress reporting via VS Code's withProgress API.
 *   • Bridge invocation and typed error routing.
 *   • Result dispatch to DiagnosticsController.
 *
 * Does NOT own:
 *   • Scenario selection UI (lives in extension.ts).
 *   • CodeLens registration (lives in codelens.ts).
 *   • Diagnostics collection lifetime (owned by DiagnosticsController).
 */

import * as path from "path";
import * as vscode from "vscode";
import { runBlackSwanEngine } from "./bridge";
import { DagPanelController } from "./dagPanel";
import { DiagnosticsController } from "./diagnostics";
import { BlackSwanHoverProvider } from "./hover";
import {
  formatProgressTitle,
  ProgressSession,
  SCENARIO_LABELS,
} from "./progress";
import {
  EngineFrameworkError,
  EngineProtocolError,
  EngineRuntimeError,
} from "./types";

export class Orchestrator implements vscode.Disposable {
  /**
   * Active runs, keyed by document fsPath.
   * One entry = one in-progress stress test for that file.
   */
  private readonly _running = new Map<string, AbortController>();
  private readonly _diagnostics: DiagnosticsController;
  private readonly _hover?: BlackSwanHoverProvider;
  private readonly _dag?: DagPanelController;

  constructor(
    diagnostics: DiagnosticsController,
    hover?: BlackSwanHoverProvider,
    dag?: DagPanelController,
  ) {
    this._diagnostics = diagnostics;
    this._hover = hover;
    this._dag = dag;
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /** True if a stress test is currently running for the given document URI. */
  isRunning(uri: vscode.Uri): boolean {
    return this._running.has(uri.fsPath);
  }

  /**
   * Start a stress test for `document`, targeting `functionName` under
   * `scenarioName`.
   *
   * Returns `false` immediately (without running) if a test is already in
   * progress for this document — the double-trigger guard. Shows a VS Code
   * warning message in that case.
   *
   * Returns `true` after the run finishes (success, failure, error, or cancel).
   */
  async run(
    document: vscode.TextDocument,
    functionName: string,
    scenarioName: string,
  ): Promise<boolean> {
    const fsPath = document.uri.fsPath;

    // ── Mutex check ────────────────────────────────────────────────────────
    if (this._running.has(fsPath)) {
      void vscode.window.showWarningMessage(
        `BlackSwan is already running for "${path.basename(fsPath)}". ` +
          `Wait for the current run to complete, or cancel it first.`,
      );
      return false;
    }

    // ── Register this run ───────────────────────────────────────────────────
    const controller = new AbortController();
    this._running.set(fsPath, controller);

    const scenarioLabel = SCENARIO_LABELS[scenarioName] ?? scenarioName;
    const fileName = path.basename(fsPath);

    try {
      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: formatProgressTitle(fileName, scenarioLabel),
          cancellable: true,
        },
        async (progress, token) => {
          // Wire VS Code's cancellation token → AbortController signal so
          // the bridge can kill the Python process when the user hits ✕.
          token.onCancellationRequested(() => controller.abort());

          // ProgressSession starts the notification in indeterminate mode
          // ("Running N iterations…"). The bridge calls onProgress() with the
          // final count on completion, which advances the bar to 100%.
          const session = new ProgressSession(progress, 5000);

          const response = await runBlackSwanEngine(fsPath, scenarioName, {
            pythonPath: this._resolvePythonPath(),
            functionName,
            cwd: path.dirname(fsPath),
            signal: controller.signal,
            onProgress: (completed, total) =>
              session.reportIterations(completed, total),
          });

          // Push results to the Diagnostics API — red squiggles appear here.
          this._diagnostics.apply(document, response);
          // Update the hover store so tooltips reflect the latest run.
          this._hover?.set(document.uri, response.shatter_points);
          // Open / update the dependency graph panel.
          this._dag?.show(response, document.uri);

          const { total_failures } = response.summary;
          if (total_failures === 0) {
            void vscode.window.showInformationMessage(
              `BlackSwan: No failures detected in "${fileName}" under ${scenarioLabel}.`,
            );
          } else {
            void vscode.window.showWarningMessage(
              `BlackSwan: ${total_failures} failure${total_failures === 1 ? "" : "s"} ` +
                `detected in "${fileName}". See the Problems panel for details.`,
            );
          }
        },
      );
    } catch (err) {
      this._handleError(err, document.uri, fileName);
    } finally {
      // Always release the mutex, even on error or cancellation.
      this._running.delete(fsPath);
    }

    return true;
  }

  /**
   * Abort the in-progress run for the given URI, if any.
   * The run's promise will reject with a cancellation error, which
   * `run()` catches silently.
   */
  cancel(uri: vscode.Uri): void {
    this._running.get(uri.fsPath)?.abort();
  }

  /** Abort all running tests and release resources. */
  dispose(): void {
    for (const controller of this._running.values()) {
      controller.abort();
    }
    this._running.clear();
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  /**
   * Route a caught error to the appropriate VS Code message.
   * Cancellations are silent — the user asked for them.
   */
  private _handleError(err: unknown, uri: vscode.Uri, fileName: string): void {
    if (err instanceof Error && /cancelled/i.test(err.message)) {
      // User hit ✕ on the progress bar — clear stale diagnostics and hovers, be quiet.
      this._diagnostics.clear(uri);
      this._hover?.clear(uri);
      return;
    }

    if (err instanceof EngineRuntimeError) {
      void vscode.window.showErrorMessage(
        `BlackSwan: Python not found. ` +
          `Check your Python installation and the "blackswan.pythonPath" setting.\n` +
          `Details: ${err.message}`,
      );
      return;
    }

    if (err instanceof EngineFrameworkError) {
      void vscode.window.showErrorMessage(
        `BlackSwan engine error for "${fileName}": ${err.message}`,
      );
      return;
    }

    if (err instanceof EngineProtocolError) {
      void vscode.window.showErrorMessage(
        `BlackSwan: Unexpected engine output — this is likely a version mismatch. ` +
          `Please file a bug at https://github.com/Lushenwar/BlackSwan/issues.\n` +
          `Details: ${err.message}`,
      );
      return;
    }

    // Catch-all for truly unexpected failures.
    void vscode.window.showErrorMessage(
      `BlackSwan: Unexpected error — ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  /**
   * Resolve the Python executable to use for spawning the engine.
   *
   * Priority:
   *   1. blackswan.pythonPath (user-set in VS Code settings)
   *   2. python.defaultInterpreterPath (from the Python extension, if installed)
   *   3. Platform default: "python" on Windows, "python3" elsewhere
   */
  private _resolvePythonPath(): string {
    const bsConfig = vscode.workspace.getConfiguration("blackswan");
    const bsPath = bsConfig.get<string>("pythonPath");
    if (bsPath?.trim()) return bsPath.trim();

    const pyConfig = vscode.workspace.getConfiguration("python");
    const interpPath = pyConfig.get<string>("defaultInterpreterPath");
    if (interpPath?.trim()) return interpPath.trim();

    return process.platform === "win32" ? "python" : "python3";
  }
}
