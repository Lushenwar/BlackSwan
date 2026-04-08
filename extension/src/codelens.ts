/**
 * codelens.ts — CodeLens provider for BlackSwan.
 *
 * Injects a "▶ Run BlackSwan" button above every Python function and class
 * definition in the active document. Clicking the button fires the
 * blackswan.runStressTest command with the document URI and function name.
 *
 * The scan is intentionally simple: a line-by-line regex pass. This avoids
 * a full AST parse in the extension host (keeping CodeLens fast) and defers
 * AST work to the engine where it belongs.
 *
 * Exported pure function: scanForTargets — zero VS Code dependency, fully
 * unit-testable.
 */

import * as vscode from "vscode";

export const CODELENS_COMMAND = "blackswan.runStressTest";

/** Matches `def` and `async def` at any indentation level. */
const FUNCTION_DEF_RE = /^\s*(async\s+)?def\s+(\w+)/;
/** Matches `class` at any indentation level. */
const CLASS_DEF_RE = /^\s*class\s+(\w+)/;

export type TargetKind = "function" | "class";

export interface FunctionTarget {
  /** The function or class name extracted from the definition line. */
  name: string;
  /** 0-indexed line number of the definition. */
  line: number;
  kind: TargetKind;
}

/**
 * Scan an array of source lines and return all function/class definitions.
 *
 * Pure function — no VS Code API dependency, safe to call in tests and in
 * Node.js worker threads.
 */
export function scanForTargets(lines: readonly string[]): FunctionTarget[] {
  const targets: FunctionTarget[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    const fnMatch = FUNCTION_DEF_RE.exec(line);
    if (fnMatch) {
      targets.push({ name: fnMatch[2], line: i, kind: "function" });
      continue; // a `def` line can't also be a `class` line
    }

    const clsMatch = CLASS_DEF_RE.exec(line);
    if (clsMatch) {
      targets.push({ name: clsMatch[1], line: i, kind: "class" });
    }
  }

  return targets;
}

/**
 * VS Code CodeLens provider. Registered in extension.ts for
 * `{ language: "python", scheme: "file" }` documents.
 */
export class BlackSwanCodeLensProvider
  implements vscode.CodeLensProvider, vscode.Disposable
{
  private readonly _onDidChangeCodeLenses = new vscode.EventEmitter<void>();
  /** Fired when the extension needs VS Code to re-query CodeLens items. */
  readonly onDidChangeCodeLenses: vscode.Event<void> =
    this._onDidChangeCodeLenses.event;

  provideCodeLenses(document: vscode.TextDocument): vscode.CodeLens[] {
    if (document.languageId !== "python") return [];

    const lines: string[] = [];
    for (let i = 0; i < document.lineCount; i++) {
      lines.push(document.lineAt(i).text);
    }

    const targets = scanForTargets(lines);

    return targets.map((target) => {
      const range = new vscode.Range(target.line, 0, target.line, 0);
      return new vscode.CodeLens(range, {
        title: "▶ Run BlackSwan",
        command: CODELENS_COMMAND,
        arguments: [document.uri, target.name],
        tooltip: `Stress-test ${target.kind} '${target.name}' with BlackSwan`,
      });
    });
  }

  /**
   * Trigger a CodeLens refresh (e.g., after a run completes so the button
   * can be re-enabled in a future version that grays it out while running).
   */
  refresh(): void {
    this._onDidChangeCodeLenses.fire();
  }

  dispose(): void {
    this._onDidChangeCodeLenses.dispose();
  }
}
