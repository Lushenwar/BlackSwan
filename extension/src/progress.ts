/**
 * progress.ts — Progress UI utilities for BlackSwan stress-test runs.
 *
 * Provides:
 *   SCENARIO_LABELS      — human-readable names for the 5 preset scenarios.
 *   formatProgressTitle  — produces the withProgress notification title.
 *   formatProgressMessage — formats "Running N iterations… M complete (P%)".
 *   ProgressSession      — wraps VS Code's Progress object, tracks the running
 *                          percentage, and translates iteration counts into the
 *                          `increment` values VS Code needs to move the bar.
 *
 * All three exported functions are pure (no VS Code dependency) and are
 * therefore fully unit-testable without a live extension host.
 *
 * ProgressSession is designed to work with both:
 *   • Bulk mode (current): bridge calls reportIterations() once with the
 *     final count after the run completes → bar jumps to 100%.
 *   • Streaming mode (future): bridge calls reportIterations() on each NDJSON
 *     progress line → bar moves smoothly to 100%.
 *
 * The Orchestrator owns the AbortController / withProgress call; this module
 * only contains the formatting logic and the progress-reporting session.
 */

// ---------------------------------------------------------------------------
// Scenario labels (source of truth for human-readable scenario names)
// ---------------------------------------------------------------------------

/**
 * Display labels for the 5 preset scenarios.
 * Keyed by the machine-readable scenario name used in engine requests.
 * Used by: orchestrator.ts (notification title) and extension.ts (Quick Pick).
 */
export const SCENARIO_LABELS: Record<string, string> = {
  liquidity_crash:       "Liquidity Crash",
  vol_spike:             "Vol Spike",
  correlation_breakdown: "Correlation Breakdown",
  rate_shock:            "Rate Shock",
  missing_data:          "Missing Data",
};

// ---------------------------------------------------------------------------
// Pure formatting helpers
// ---------------------------------------------------------------------------

/**
 * Format the `withProgress` notification title.
 *
 * @param fileName      Basename of the file being stress-tested (e.g. "model.py").
 * @param scenarioLabel Human-readable scenario label (e.g. "Liquidity Crash").
 *
 * @example
 *   formatProgressTitle("risk_model.py", "Liquidity Crash")
 *   // → 'BlackSwan: Stress testing "risk_model.py" [Liquidity Crash]'
 */
export function formatProgressTitle(
  fileName: string,
  scenarioLabel: string,
): string {
  return `BlackSwan: Stress testing "${fileName}" [${scenarioLabel}]`;
}

/**
 * Format the running progress message shown beneath the notification title.
 *
 * @param completed Iterations completed so far.
 * @param total     Total iterations planned.
 *
 * @example
 *   formatProgressMessage(2340, 5000)
 *   // → "Running 5,000 iterations… 2,340 complete (47%)"
 */
export function formatProgressMessage(completed: number, total: number): string {
  const pct = total > 0 ? Math.round((completed / total) * 100) : 0;
  const fmtTotal     = total.toLocaleString("en-US");
  const fmtCompleted = completed.toLocaleString("en-US");
  return `Running ${fmtTotal} iterations… ${fmtCompleted} complete (${pct}%)`;
}

// ---------------------------------------------------------------------------
// ProgressSession
// ---------------------------------------------------------------------------

/**
 * Narrow interface for the VS Code Progress object.
 *
 * Declared here (rather than importing directly from vscode) so tests can
 * pass a plain object stub without the vscode mock being involved.
 */
export interface ProgressReporter {
  report(value: { message?: string; increment?: number }): void;
}

/**
 * Wraps a VS Code progress reporter for a single BlackSwan run.
 *
 * Tracks the cumulative percentage reported so far and translates each
 * `reportIterations()` call into an `increment` that VS Code uses to advance
 * a determinate progress bar. Starting from 0, each call adds the delta.
 *
 * Usage in orchestrator.ts:
 *
 *   await vscode.window.withProgress({ ..., cancellable: true }, async (progress) => {
 *     const session = new ProgressSession(progress, 5000);
 *     const response = await runBlackSwanEngine(file, scenario, {
 *       ...,
 *       onProgress: (c, t) => session.reportIterations(c, t),
 *     });
 *   });
 */
export class ProgressSession {
  /** Cumulative percentage reported to VS Code so far (0–100). */
  private _reportedPct = 0;

  /**
   * @param reporter      VS Code Progress object (or stub in tests).
   * @param defaultTotal  Expected iteration count — shown in the initial message
   *                      before any streaming updates arrive.
   */
  constructor(
    private readonly _reporter: ProgressReporter,
    readonly defaultTotal: number,
  ) {
    // Display the initial "N iterations" message with no increment so VS Code
    // shows an indeterminate spinner. The bar becomes determinate once the first
    // reportIterations() call arrives with an increment > 0.
    this._reporter.report({
      message: `Running ${defaultTotal.toLocaleString("en-US")} iterations…`,
    });
  }

  /**
   * Update the notification with the current iteration count.
   *
   * Calculates the percentage increment since the last call and reports both
   * the formatted message and the delta so VS Code advances the bar correctly.
   *
   * Safe to call with `total = 0` (shows 0% without dividing by zero).
   * Safe to call multiple times — each call reports only the delta.
   *
   * @param completed Iterations completed so far (from the engine).
   * @param total     Total iterations for this run.
   */
  reportIterations(completed: number, total: number): void {
    const pct = total > 0 ? Math.round((completed / total) * 100) : 0;
    const increment = pct - this._reportedPct;
    this._reportedPct = pct;

    this._reporter.report({
      message: formatProgressMessage(completed, total),
      // Only pass increment when it's positive — a zero or negative increment
      // would move the bar backwards, which VS Code ignores but is still noisy.
      increment: increment > 0 ? increment : undefined,
    });
  }
}

