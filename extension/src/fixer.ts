/**
 * fixer.ts — Deterministic mathematical guard applicator for BlackSwan.
 *
 * Responsibilities:
 *   • Spawn `python -m blackswan fix <file> --line <n> --type <t>` and parse the JSON response.
 *   • Register a virtual document provider (scheme: blackswan-preview) so we can
 *     show a side-by-side diff before the user commits to the change.
 *   • Apply confirmed fixes via WorkspaceEdit (fully undo-able).
 *
 * Design principles:
 *   • The AI never writes code — this module always drives the actual edit.
 *   • The Python engine (guards.py) is the source of truth for what changes.
 *   • No edits are applied without the user seeing a diff and clicking "Apply".
 */

import * as path from "path";
import { spawn } from "child_process";
import * as vscode from "vscode";
import { FailureType } from "./types";

// ---------------------------------------------------------------------------
// Virtual document provider for the diff preview
// ---------------------------------------------------------------------------

const PREVIEW_SCHEME = "blackswan-preview";

/**
 * Provides in-memory content for `blackswan-preview://` URIs.
 *
 * Register once in extension.ts:
 *   context.subscriptions.push(
 *     vscode.workspace.registerTextDocumentContentProvider(
 *       PREVIEW_SCHEME, blackSwanPreviewProvider,
 *     ),
 *   );
 */
export class BlackSwanPreviewProvider
  implements vscode.TextDocumentContentProvider
{
  private readonly _contents = new Map<string, string>();
  private readonly _onDidChange = new vscode.EventEmitter<vscode.Uri>();

  /** Fires when a preview document's content is updated. */
  readonly onDidChange = this._onDidChange.event;

  /** Store (or replace) content for `uri`. */
  set(uri: vscode.Uri, content: string): void {
    this._contents.set(uri.toString(), content);
    this._onDidChange.fire(uri);
  }

  /** Release stored content for `uri`. */
  clear(uri: vscode.Uri): void {
    this._contents.delete(uri.toString());
  }

  provideTextDocumentContent(uri: vscode.Uri): string {
    return this._contents.get(uri.toString()) ?? "";
  }

  dispose(): void {
    this._onDidChange.dispose();
  }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/**
 * Apply a mathematical guard to `line` (1-indexed) in `document`.
 *
 * Flow:
 *   1. Spawn the Python fixer engine.
 *   2. Show a diff of the proposed change.
 *   3. Ask the user to confirm.
 *   4. Apply via WorkspaceEdit (undo-able with Ctrl+Z).
 */
export async function applyMathGuard(
  document: vscode.TextDocument,
  line: number,
  failureType: FailureType,
  pythonPath: string,
  previewProvider: BlackSwanPreviewProvider,
): Promise<void> {
  // ── Step 1: Ask the Python engine for the fix ───────────────────────────
  let result: FixerResponse;
  try {
    result = await runPythonFixer(
      document.uri.fsPath,
      line,
      failureType,
      pythonPath,
    );
  } catch (err) {
    void vscode.window.showErrorMessage(
      `BlackSwan Fix: Failed to spawn fixer — ${err instanceof Error ? err.message : String(err)}`,
    );
    return;
  }

  if (result.status === "unsupported") {
    void vscode.window.showInformationMessage(
      `BlackSwan Fix: No deterministic guard pattern available for '${failureType}'. ` +
        `Check the fix hint in the Problems panel for manual guidance.`,
    );
    return;
  }

  if (result.status === "error") {
    void vscode.window.showWarningMessage(
      `BlackSwan Fix: ${result.message ?? "Unknown error from fixer engine."}`,
    );
    return;
  }

  // ── Step 2: Build the proposed document content ─────────────────────────
  const originalContent = document.getText();
  const fixedContent = _applyResultToContent(originalContent, result);

  // ── Step 3: Show diff ────────────────────────────────────────────────────
  const previewUri = vscode.Uri.parse(
    `${PREVIEW_SCHEME}://${encodeURIComponent(document.uri.fsPath)}`,
  );
  previewProvider.set(previewUri, fixedContent);

  await vscode.commands.executeCommand(
    "vscode.diff",
    document.uri,
    previewUri,
    `BlackSwan Fix Preview — ${path.basename(document.uri.fsPath)}`,
    { preview: true },
  );

  // ── Step 4: Confirm ──────────────────────────────────────────────────────
  const hint = result.explanation ?? `Apply guard for ${failureType} at line ${line}.`;
  const choice = await vscode.window.showInformationMessage(
    `Apply mathematical guard to line ${line}?`,
    {
      modal: false,
      detail: hint,
    },
    "Apply Fix",
    "Cancel",
  );

  previewProvider.clear(previewUri);

  if (choice !== "Apply Fix") return;

  // ── Step 5: Apply via WorkspaceEdit (undo-able) ─────────────────────────
  const edit = new vscode.WorkspaceEdit();
  const zeroLine = line - 1;

  if (
    result.replacement !== undefined &&
    result.replacement !== result.original
  ) {
    const lineRange = document.lineAt(zeroLine).rangeIncludingLineBreak;
    edit.replace(document.uri, lineRange, result.replacement + "\n");
  }

  if (result.extra_lines && result.extra_lines.length > 0) {
    // Insert extra lines (guards) AFTER the target line.
    const insertPos = new vscode.Position(zeroLine + 1, 0);
    edit.insert(
      document.uri,
      insertPos,
      result.extra_lines.join("\n") + "\n",
    );
  }

  const success = await vscode.workspace.applyEdit(edit);
  if (success) {
    void vscode.window.showInformationMessage(
      "BlackSwan: Mathematical guard applied. Use Ctrl+Z to undo.",
    );
  } else {
    void vscode.window.showErrorMessage(
      "BlackSwan Fix: Failed to apply edit — the document may have changed.",
    );
  }
}

// ---------------------------------------------------------------------------
// Python subprocess
// ---------------------------------------------------------------------------

interface FixerResponse {
  status: "ok" | "error" | "unsupported";
  line?: number;
  original?: string;
  replacement?: string;
  extra_lines?: string[];
  explanation?: string;
  message?: string;
}

function runPythonFixer(
  filePath: string,
  line: number,
  failureType: FailureType,
  pythonPath: string,
): Promise<FixerResponse> {
  return new Promise((resolve, reject) => {
    const args = [
      "-m", "blackswan", "fix", filePath,
      "--line", String(line),
      "--type", failureType,
    ];

    const proc = spawn(pythonPath, args, {
      cwd: path.dirname(filePath),
      stdio: ["ignore", "pipe", "pipe"],
      env: process.env,
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf-8");
    });
    proc.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf-8");
    });

    proc.on("error", (err) => {
      reject(new Error(`Failed to spawn '${pythonPath}': ${err.message}`));
    });

    proc.on("close", () => {
      try {
        resolve(JSON.parse(stdout) as FixerResponse);
      } catch {
        reject(
          new Error(
            `Fixer returned non-JSON output.\n` +
              `stdout: ${stdout.slice(0, 300)}\n` +
              `stderr: ${stderr.slice(0, 300)}`,
          ),
        );
      }
    });
  });
}

// ---------------------------------------------------------------------------
// Content builder
// ---------------------------------------------------------------------------

/**
 * Apply a FixerResponse to the full document source string and return the
 * modified string.  Used to populate the diff preview virtual document.
 */
function _applyResultToContent(
  source: string,
  result: FixerResponse,
): string {
  if (result.line === undefined) return source;

  const lines = source.split("\n");
  const zeroLine = result.line - 1;

  if (zeroLine < 0 || zeroLine >= lines.length) return source;

  const output = [...lines];

  // Replace the target line if it changed.
  if (
    result.replacement !== undefined &&
    result.replacement !== result.original
  ) {
    output[zeroLine] = result.replacement;
  }

  // Insert extra_lines after the (possibly replaced) target line.
  if (result.extra_lines && result.extra_lines.length > 0) {
    output.splice(zeroLine + 1, 0, ...result.extra_lines);
  }

  return output.join("\n");
}
