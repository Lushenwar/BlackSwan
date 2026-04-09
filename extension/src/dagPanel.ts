/**
 * dagPanel.ts — Dependency Graph webview panel for BlackSwan.
 *
 * Opens a VS Code WebviewPanel (beside the active editor) that renders the
 * causal dependency graph derived from a stress-test result as an interactive
 * SVG diagram.
 *
 * Node colours match the CLAUDE.md spec:
 *   yellow  (#f5c842) — root_input   (the upstream cause)
 *   orange  (#f57c42) — intermediate  (the propagation path)
 *   red     (#e03030) — failure_site  (where the math breaks)
 *
 * Clicking any node sends a `navigate` message back to the extension host,
 * which opens the source file at the corresponding line.
 *
 * Exported pure functions:
 *   buildDagData — converts ShatterPoint[] to a DagData structure (testable).
 *   buildDagHtml — produces the full webview HTML string from DagData (testable).
 *
 * DagPanelController manages the WebviewPanel lifecycle (create / reveal /
 * update / dispose) and is registered in extension.ts.
 */

import * as vscode from "vscode";
import { CausalRole, ShatterPoint } from "./types";

// ---------------------------------------------------------------------------
// DAG data types
// ---------------------------------------------------------------------------

export interface DagNode {
  /** Stable ID derived from line and variable name. */
  id: string;
  line: number;
  variable: string;
  role: CausalRole;
}

export interface DagEdge {
  /** ID of the upstream node. */
  source: string;
  /** ID of the downstream node. */
  target: string;
}

export interface DagData {
  nodes: DagNode[];
  edges: DagEdge[];
  /** fsPath of the document — sent back to the extension on node click. */
  documentUri: string;
}

// ---------------------------------------------------------------------------
// Pure builder — ShatterPoint[] → DagData
// ---------------------------------------------------------------------------

/**
 * Build a DagData structure from the shatter points of a stress-test result.
 *
 * Nodes are deduplicated by (line, variable) across all causal chains so a
 * shared upstream variable appears as a single node even when it contributes
 * to multiple failures. Edges are deduplicated the same way.
 *
 * Pure function — no VS Code dependency. Safe to call in unit tests.
 */
export function buildDagData(
  shatterPoints: ShatterPoint[],
  documentUri: vscode.Uri,
): DagData {
  const nodeMap  = new Map<string, DagNode>();
  const edgeKeys = new Set<string>();
  const edges: DagEdge[] = [];

  for (const sp of shatterPoints) {
    const chain = sp.causal_chain;
    for (let i = 0; i < chain.length; i++) {
      const link = chain[i];
      // Stable node ID: "L<line>_<variable>" — survives across runs as long as
      // line numbers and variable names don't change.
      const nodeId = `L${link.line}_${link.variable}`;

      if (!nodeMap.has(nodeId)) {
        nodeMap.set(nodeId, {
          id:       nodeId,
          line:     link.line,
          variable: link.variable,
          role:     link.role,
        });
      }

      if (i > 0) {
        const prev   = chain[i - 1];
        const prevId = `L${prev.line}_${prev.variable}`;
        const eKey   = `${prevId}→${nodeId}`;
        if (!edgeKeys.has(eKey)) {
          edgeKeys.add(eKey);
          edges.push({ source: prevId, target: nodeId });
        }
      }
    }
  }

  return {
    nodes:       Array.from(nodeMap.values()),
    edges,
    documentUri: documentUri.fsPath,
  };
}

// ---------------------------------------------------------------------------
// Pure HTML builder — DagData → webview HTML string
// ---------------------------------------------------------------------------

/**
 * Node colours for each causal role.
 * yellow = root cause, orange = propagation, red = failure site.
 */
const ROLE_COLORS: Record<CausalRole, string> = {
  root_input:   "#f5c842",
  intermediate: "#f57c42",
  failure_site: "#e03030",
};

/**
 * Escape a value for safe embedding inside a `<script>` tag.
 *
 * JSON.stringify is safe for values, but the resulting string can contain
 * `</script>` which would break out of the script block. Replace the
 * dangerous characters with Unicode escapes.
 */
function safeJson(value: unknown): string {
  return JSON.stringify(value)
    .replace(/</g,  "\\u003c")
    .replace(/>/g,  "\\u003e")
    .replace(/&/g,  "\\u0026")
    .replace(/'/g,  "\\u0027");
}

/**
 * Build the full webview HTML string for the DAG panel.
 *
 * @param data  Pre-built DagData to embed as the initial render payload.
 * @param nonce Random nonce string used in the Content-Security-Policy and
 *              in the `<script nonce="…">` tag. Must be unique per panel load.
 *
 * Pure function — deterministic given the same inputs, no side effects.
 */
export function buildDagHtml(data: DagData, nonce: string): string {
  const dataJson = safeJson(data);

  return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
  <title>BlackSwan — Dependency Graph</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: var(--vscode-editor-background, #1e1e1e);
      color: var(--vscode-editor-foreground, #d4d4d4);
      font-family: var(--vscode-font-family, system-ui, -apple-system, sans-serif);
      font-size: 12px;
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }

    #bs-header {
      padding: 10px 14px 8px;
      border-bottom: 1px solid var(--vscode-panel-border, #3c3c3c);
      flex-shrink: 0;
    }
    #bs-header h2 {
      font-size: 12px;
      font-weight: 600;
      color: var(--vscode-panelTitle-activeForeground, #d4d4d4);
      margin-bottom: 2px;
    }
    #bs-header p {
      font-size: 10px;
      color: var(--vscode-descriptionForeground, #858585);
    }

    #bs-legend {
      display: flex;
      gap: 14px;
      align-items: center;
      padding: 6px 14px;
      border-bottom: 1px solid var(--vscode-panel-border, #3c3c3c);
      flex-shrink: 0;
    }
    .bs-legend-item {
      display: flex;
      align-items: center;
      gap: 5px;
      font-size: 10px;
      color: var(--vscode-descriptionForeground, #858585);
    }
    .bs-legend-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      border: 1px solid rgba(0,0,0,0.25);
      flex-shrink: 0;
    }

    #bs-dag-container {
      flex: 1;
      overflow: auto;
      padding: 16px 8px;
    }

    svg#bs-dag {
      display: block;
      width: 100%;
      overflow: visible;
    }

    #bs-empty {
      display: none;
      padding: 40px 16px;
      text-align: center;
      color: var(--vscode-descriptionForeground, #858585);
      font-size: 11px;
    }

    .bs-node { cursor: pointer; }
    .bs-node:focus { outline: none; }
    .bs-node:focus circle,
    .bs-node:hover circle { filter: brightness(1.15); }

    .bs-var-name {
      fill: #111;
      font-size: 8.5px;
      font-weight: 700;
      pointer-events: none;
      dominant-baseline: middle;
      text-anchor: middle;
    }
    .bs-line-num {
      fill: #222;
      font-size: 7.5px;
      pointer-events: none;
      dominant-baseline: middle;
      text-anchor: middle;
      opacity: 0.75;
    }
    .bs-edge {
      fill: none;
      stroke: var(--vscode-editor-foreground, #858585);
      stroke-width: 1.5;
      stroke-opacity: 0.4;
      marker-end: url(#bs-arrow);
    }
  </style>
</head>
<body>
  <div id="bs-header">
    <h2>&#9671; BlackSwan — Dependency Graph</h2>
    <p>Click any node to navigate to its source line</p>
  </div>
  <div id="bs-legend">
    <div class="bs-legend-item">
      <div class="bs-legend-dot" style="background:#f5c842;"></div>Root input
    </div>
    <div class="bs-legend-item">
      <div class="bs-legend-dot" style="background:#f57c42;"></div>Intermediate
    </div>
    <div class="bs-legend-item">
      <div class="bs-legend-dot" style="background:#e03030;"></div>Failure site
    </div>
  </div>
  <div id="bs-dag-container">
    <svg id="bs-dag" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="bs-arrow" markerWidth="7" markerHeight="7"
                refX="5" refY="3.5" orient="auto">
          <path d="M0,0 L0,7 L7,3.5 z"
                fill="#858585" opacity="0.45"/>
        </marker>
      </defs>
    </svg>
    <div id="bs-empty">No dependency data available for this run.</div>
  </div>

  <script nonce="${nonce}">
  (function () {
    'use strict';

    /* ── Constants ─────────────────────────────────────────────────────── */
    var COLORS = {
      root_input:   '${ROLE_COLORS.root_input}',
      intermediate: '${ROLE_COLORS.intermediate}',
      failure_site: '${ROLE_COLORS.failure_site}'
    };
    var ROLE_LABELS = {
      root_input:   'Root input',
      intermediate: 'Intermediate',
      failure_site: 'Failure site'
    };
    /* Column x-centre positions for each role */
    var COL_X = { root_input: 90, intermediate: 300, failure_site: 510 };
    var NODE_R  = 30;   /* node circle radius */
    var ROW_H   = 100;  /* vertical spacing between nodes in the same column */
    var SVG_W   = 620;
    var SVG_MIN_H = 220;

    /* ── VS Code API ───────────────────────────────────────────────────── */
    var vscode = acquireVsCodeApi();

    /* ── SVG helpers ───────────────────────────────────────────────────── */
    var NS = 'http://www.w3.org/2000/svg';
    function el(tag, attrs) {
      var e = document.createElementNS(NS, tag);
      for (var k in attrs) e.setAttribute(k, attrs[k]);
      return e;
    }
    function txt(tag, attrs, text) {
      var e = el(tag, attrs);
      e.textContent = text;
      return e;
    }

    /* ── Layout ────────────────────────────────────────────────────────── */
    function computeLayout(nodes) {
      var byRole = { root_input: [], intermediate: [], failure_site: [] };
      nodes.forEach(function (n) {
        if (byRole[n.role]) byRole[n.role].push(n);
      });

      var maxCount = Math.max(
        byRole.root_input.length,
        byRole.intermediate.length,
        byRole.failure_site.length,
        1
      );
      var svgH = Math.max(maxCount * ROW_H + ROW_H, SVG_MIN_H);

      var pos = {};
      Object.keys(byRole).forEach(function (role) {
        var col = byRole[role];
        var totalH = col.length * ROW_H;
        var startY = (svgH - totalH) / 2 + ROW_H / 2;
        col.forEach(function (n, i) {
          pos[n.id] = { x: COL_X[role], y: startY + i * ROW_H };
        });
      });

      return { pos: pos, svgH: svgH };
    }

    /* ── Render ────────────────────────────────────────────────────────── */
    function render(data) {
      var svg   = document.getElementById('bs-dag');
      var empty = document.getElementById('bs-empty');

      /* Clear previous content (keep <defs>) */
      var defs = svg.querySelector('defs');
      svg.innerHTML = '';
      if (defs) svg.appendChild(defs);

      if (!data || !data.nodes || data.nodes.length === 0) {
        empty.style.display = 'block';
        svg.style.display   = 'none';
        return;
      }
      empty.style.display = 'none';
      svg.style.display   = 'block';

      var layout = computeLayout(data.nodes);
      var pos    = layout.pos;
      var svgH   = layout.svgH;

      svg.setAttribute('viewBox', '0 0 ' + SVG_W + ' ' + svgH);
      svg.setAttribute('height', String(svgH));

      /* ── Draw edges (behind nodes) ─────────────────────────────────── */
      data.edges.forEach(function (e) {
        var s = pos[e.source], t = pos[e.target];
        if (!s || !t) return;
        var cx1 = s.x + (t.x - s.x) * 0.45;
        var cx2 = s.x + (t.x - s.x) * 0.55;
        var d = 'M ' + (s.x + NODE_R) + ' ' + s.y +
                ' C ' + cx1 + ' ' + s.y +
                ', '  + cx2 + ' ' + t.y +
                ', '  + (t.x - NODE_R) + ' ' + t.y;
        svg.appendChild(el('path', { d: d, 'class': 'bs-edge' }));
      });

      /* ── Draw nodes ────────────────────────────────────────────────── */
      data.nodes.forEach(function (n) {
        var p = pos[n.id];
        if (!p) return;
        var color = COLORS[n.role] || '#888';
        var label = ROLE_LABELS[n.role] || n.role;

        var g = el('g', {
          'class':     'bs-node',
          transform:   'translate(' + p.x + ',' + p.y + ')',
          tabindex:    '0',
          'aria-label': n.variable + ' — line ' + n.line + ' — ' + label
        });

        /* Navigate on click */
        g.addEventListener('click', function () {
          vscode.postMessage({ type: 'navigate', line: n.line, uri: data.documentUri });
        });
        g.addEventListener('keydown', function (ev) {
          if (ev.key === 'Enter' || ev.key === ' ') {
            ev.preventDefault();
            vscode.postMessage({ type: 'navigate', line: n.line, uri: data.documentUri });
          }
        });

        /* Circle */
        g.appendChild(el('circle', {
          r:            String(NODE_R),
          fill:         color,
          stroke:       '#1a1a1a',
          'stroke-width': '1.5'
        }));

        /* Variable name — truncate if needed */
        var varLabel = n.variable.length > 11
          ? n.variable.slice(0, 10) + '\u2026'
          : n.variable;
        g.appendChild(txt('text', { 'class': 'bs-var-name', dy: '-7' }, varLabel));

        /* Line number */
        g.appendChild(txt('text', { 'class': 'bs-line-num', dy: '10' }, 'L' + n.line));

        svg.appendChild(g);
      });
    }

    /* ── Initial render ─────────────────────────────────────────────── */
    var initialData = ${dataJson};
    render(initialData);

    /* ── Live updates from the extension host ───────────────────────── */
    window.addEventListener('message', function (event) {
      if (event.data && event.data.type === 'update') {
        render(event.data.data);
      }
    });
  })();
  </script>
</body>
</html>`;
}

// ---------------------------------------------------------------------------
// Nonce generator
// ---------------------------------------------------------------------------

/**
 * Generate a cryptographically-adequate random nonce for the webview CSP.
 * 32 alphanumeric characters gives ~190 bits of entropy — sufficient for a
 * per-session, non-secret CSP nonce.
 */
function getNonce(): string {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let nonce = "";
  for (let i = 0; i < 32; i++) {
    nonce += chars[Math.floor(Math.random() * chars.length)];
  }
  return nonce;
}

// ---------------------------------------------------------------------------
// DagPanelController
// ---------------------------------------------------------------------------

/**
 * Manages the lifecycle of the BlackSwan DAG WebviewPanel.
 *
 * - First call to `show()` creates the panel beside the active editor.
 * - Subsequent calls reveal the existing panel and push an "update" message
 *   so the webview re-renders with fresh data — no panel flicker.
 * - Navigation messages from the webview (node click) are translated into
 *   `vscode.open` commands that scroll the editor to the target line.
 *
 * Registered and disposed via `extension.ts`. Passed to `Orchestrator` so
 * the panel opens automatically after each successful stress-test run.
 */
export class DagPanelController implements vscode.Disposable {
  static readonly viewType = "blackswan.dagPanel";

  private _panel?: vscode.WebviewPanel;
  private readonly _disposables: vscode.Disposable[] = [];

  /** True when the panel has been created and not yet disposed. */
  get isVisible(): boolean {
    return this._panel !== undefined;
  }

  /**
   * Show or update the DAG panel for a completed stress-test result.
   *
   * @param response    The engine response — provides the shatter points whose
   *                    causal chains form the graph.
   * @param documentUri URI of the Python file that was stress-tested — used
   *                    for navigation when a node is clicked.
   */
  show(response: { shatter_points: ShatterPoint[] }, documentUri: vscode.Uri): void {
    const dagData = buildDagData(response.shatter_points, documentUri);

    if (this._panel) {
      // Panel already open — reveal it and push a data update.
      this._panel.reveal(vscode.ViewColumn.Beside);
      void this._panel.webview.postMessage({ type: "update", data: dagData });
      return;
    }

    this._panel = vscode.window.createWebviewPanel(
      DagPanelController.viewType,
      "BlackSwan — Dependency Graph",
      vscode.ViewColumn.Beside,
      {
        enableScripts:          true,
        retainContextWhenHidden: true,
        // No local resources needed — everything is inlined.
        localResourceRoots: [],
      },
    );

    this._panel.webview.html = buildDagHtml(dagData, getNonce());

    // Handle messages posted by the webview script.
    this._panel.webview.onDidReceiveMessage(
      (message: { type: string; line: number; uri: string }) => {
        if (message.type !== "navigate") return;
        // Engine lines are 1-indexed; VS Code Range is 0-indexed.
        const lineIdx = Math.max(message.line - 1, 0);
        const fileUri = vscode.Uri.file(message.uri);
        void vscode.commands.executeCommand(
          "vscode.open",
          fileUri,
          { selection: new vscode.Range(lineIdx, 0, lineIdx, 0) },
        );
      },
      undefined,
      this._disposables,
    );

    // Release the panel reference when the user closes it manually.
    this._panel.onDidDispose(
      () => { this._panel = undefined; },
      undefined,
      this._disposables,
    );
  }

  dispose(): void {
    this._panel?.dispose();
    this._panel = undefined;
    for (const d of this._disposables) d.dispose();
    this._disposables.length = 0;
  }
}
