/**
 * dagPanel.ts — Dependency Graph webview panel for BlackSwan.
 *
 * Professional fintech dark theme featuring:
 *   • Summary header with scenario, seed, mode, and runtime metadata
 *   • KPI strip with severity-breakdown failure counts and failure rate
 *   • Interactive SVG DAG — rounded-rect nodes, SVG glow filters, cubic bezier edges
 *   • Slide-in detail panel showing full shatter point data on node click
 *   • Zoom in / zoom out / reset controls
 *   • Keyboard accessible throughout (Tab + Enter/Space navigation)
 *
 * Node colours (test-contractual — these exact hex values MUST remain):
 *   #f5c842 — root_input   (upstream root cause)
 *   #f57c42 — intermediate  (propagation path)
 *   #e03030 — failure_site  (where the math breaks)
 *
 * Exported pure functions (no VS Code dependency, safe for unit tests):
 *   buildDagData — ShatterPoint[] → DagData
 *   buildDagHtml — DagData + nonce → full webview HTML string
 *
 * DagPanelController manages the WebviewPanel lifecycle.
 */

import * as vscode from "vscode";
import { BlackSwanResponse, CausalRole, ShatterPoint } from "./types";

// ---------------------------------------------------------------------------
// DAG data types
// ---------------------------------------------------------------------------

export interface DagNode {
  /** Stable ID: "L<line>_<variable>" */
  id: string;
  line: number;
  variable: string;
  role: CausalRole;
}

export interface DagEdge {
  source: string;
  target: string;
}

/** Shatter-point detail for a failure_site node, embedded for the detail panel. */
export interface NodeDetail {
  severity: string;
  failure_type: string;
  message: string;
  frequency: string;
  fix_hint: string;
  confidence: string;
}

/** Optional run-level summary embedded in DagData for the KPI strip. */
export interface DagSummary {
  total_failures: number;
  critical_count: number;
  warning_count: number;
  failure_rate: number;
  iterations_completed: number;
  runtime_ms: number;
  scenario_name: string;
  mode: string;
  seed: number;
}

export interface DagData {
  nodes: DagNode[];
  edges: DagEdge[];
  /** fsPath of the document — sent back to extension on node click. */
  documentUri: string;
  /** Run-level metadata for the header and KPI strip. Optional for test compat. */
  summary?: DagSummary;
  /** Per-node detail data keyed by node ID. Optional for test compat. */
  nodeDetails?: Record<string, NodeDetail>;
}

// ---------------------------------------------------------------------------
// Pure builder — ShatterPoint[] → DagData
// ---------------------------------------------------------------------------

/**
 * Build a DagData structure from shatter points.
 *
 * Nodes are deduplicated by (line, variable) across all causal chains.
 * Edges are deduplicated the same way.
 *
 * Pass the full BlackSwanResponse as the optional third argument to populate
 * the `summary` and `nodeDetails` fields used by the rich UI.
 *
 * Pure function — no VS Code dependency. Safe to call in unit tests.
 */
export function buildDagData(
  shatterPoints: ShatterPoint[],
  documentUri: vscode.Uri,
  response?: BlackSwanResponse,
): DagData {
  const nodeMap   = new Map<string, DagNode>();
  const edgeKeys  = new Set<string>();
  const edges: DagEdge[] = [];
  const nodeDetails: Record<string, NodeDetail> = {};

  for (const sp of shatterPoints) {
    const chain = sp.causal_chain;
    for (let i = 0; i < chain.length; i++) {
      const link   = chain[i];
      const nodeId = `L${link.line}_${link.variable}`;

      if (!nodeMap.has(nodeId)) {
        nodeMap.set(nodeId, {
          id:       nodeId,
          line:     link.line,
          variable: link.variable,
          role:     link.role,
        });
      }

      // For failure_site nodes, attach the first associated ShatterPoint's details.
      if (link.role === "failure_site" && !nodeDetails[nodeId]) {
        nodeDetails[nodeId] = {
          severity:     sp.severity,
          failure_type: sp.failure_type,
          message:      sp.message,
          frequency:    sp.frequency,
          fix_hint:     sp.fix_hint,
          confidence:   sp.confidence,
        };
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

  // Build optional summary for the KPI strip.
  let summary: DagSummary | undefined;
  if (response) {
    const criticalCount = shatterPoints.filter((s) => s.severity === "critical").length;
    const warningCount  = shatterPoints.filter((s) => s.severity === "warning").length;
    summary = {
      total_failures:       response.summary.total_failures,
      critical_count:       criticalCount,
      warning_count:        warningCount,
      failure_rate:         response.summary.failure_rate,
      iterations_completed: response.iterations_completed,
      runtime_ms:           response.runtime_ms,
      scenario_name:        response.scenario_card.name,
      mode:                 response.mode,
      seed:                 response.scenario_card.seed,
    };
  }

  return {
    nodes:       Array.from(nodeMap.values()),
    edges,
    documentUri: documentUri.fsPath,
    summary:     summary,
    nodeDetails: Object.keys(nodeDetails).length > 0 ? nodeDetails : undefined,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Escape a value for safe embedding inside a `<script>` tag.
 * Replaces dangerous HTML characters with Unicode escapes.
 */
function safeJson(value: unknown): string {
  return JSON.stringify(value)
    .replace(/</g,  "\\u003c")
    .replace(/>/g,  "\\u003e")
    .replace(/&/g,  "\\u0026")
    .replace(/'/g,  "\\u0027");
}

/**
 * Node fill colours for each causal role.
 * These exact hex values are contractual — test suite checks for them.
 * yellow = root cause, orange = propagation, red = failure site.
 */
const ROLE_COLORS: Record<CausalRole, string> = {
  root_input:   "#f5c842",
  intermediate: "#f57c42",
  failure_site: "#e03030",
};

// ---------------------------------------------------------------------------
// Pure HTML builder — DagData → webview HTML string
// ---------------------------------------------------------------------------

/**
 * Build the full webview HTML for the DAG panel.
 *
 * Pure function — deterministic given the same inputs, no side effects.
 *
 * @param data   Pre-built DagData to embed as the initial render payload.
 * @param nonce  Random nonce for the Content-Security-Policy and script tag.
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

    @media (prefers-reduced-motion: reduce) {
      * { transition-duration: 0.01ms !important; animation-duration: 0.01ms !important; }
    }

    :root {
      --bs-bg:          var(--vscode-editor-background, #0d1117);
      --bs-surface:     var(--vscode-sideBar-background, #161b22);
      --bs-border:      var(--vscode-panel-border, #30363d);
      --bs-text:        var(--vscode-editor-foreground, #e6edf3);
      --bs-muted:       var(--vscode-descriptionForeground, #8b949e);
      --bs-accent:      var(--vscode-focusBorder, #388bfd);
      --bs-critical:    #f85149;
      --bs-warning:     #e3b341;
      --bs-info:        #79c0ff;
      --bs-node-yellow: #f5c842;
      --bs-node-orange: #f57c42;
      --bs-node-red:    #e03030;
      --bs-radius:      6px;
    }

    html, body {
      height: 100%;
      overflow: hidden;
    }

    body {
      background: var(--bs-bg);
      color: var(--bs-text);
      font-family: var(--vscode-font-family, 'Segoe UI', system-ui, -apple-system, sans-serif);
      font-size: 12px;
      display: flex;
      flex-direction: column;
    }

    /* ── Header ──────────────────────────────────────────────────────────── */
    #bs-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 9px 14px 8px;
      background: var(--bs-surface);
      border-bottom: 1px solid var(--bs-border);
      flex-shrink: 0;
      gap: 12px;
      min-height: 40px;
    }

    #bs-brand {
      display: flex;
      align-items: center;
      gap: 7px;
      font-size: 12px;
      font-weight: 700;
      color: var(--bs-text);
      letter-spacing: 0.03em;
      flex-shrink: 0;
    }

    #bs-brand svg {
      flex-shrink: 0;
    }

    #bs-run-meta {
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .bs-meta-pill {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 2px 7px;
      border-radius: 20px;
      border: 1px solid var(--bs-border);
      font-size: 10px;
      color: var(--bs-muted);
      background: transparent;
      white-space: nowrap;
    }

    .bs-meta-pill .bs-meta-label {
      opacity: 0.65;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-size: 9px;
    }

    .bs-meta-pill .bs-meta-value {
      color: var(--bs-text);
      font-weight: 600;
    }

    /* ── KPI strip ───────────────────────────────────────────────────────── */
    #bs-kpi-bar {
      display: flex;
      gap: 1px;
      background: var(--bs-border);
      border-bottom: 1px solid var(--bs-border);
      flex-shrink: 0;
    }

    .bs-kpi {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 6px 8px;
      background: var(--bs-bg);
      gap: 1px;
    }

    .bs-kpi-label {
      font-size: 9px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      color: var(--bs-muted);
    }

    .bs-kpi-value {
      font-size: 16px;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
      line-height: 1.1;
    }

    .bs-kpi-sub {
      font-size: 9px;
      color: var(--bs-muted);
    }

    .kpi-critical { color: var(--bs-critical); }
    .kpi-warning  { color: var(--bs-warning);  }
    .kpi-rate     { color: var(--bs-node-orange); }
    .kpi-neutral  { color: var(--bs-text); }

    /* ── Main split ──────────────────────────────────────────────────────── */
    #bs-main {
      display: flex;
      flex: 1;
      overflow: hidden;
    }

    /* ── Graph area ──────────────────────────────────────────────────────── */
    #bs-graph-area {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      position: relative;
    }

    #bs-graph-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 6px 10px;
      border-bottom: 1px solid var(--bs-border);
      flex-shrink: 0;
      gap: 8px;
    }

    .bs-legend {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .bs-legend-item {
      display: flex;
      align-items: center;
      gap: 4px;
      font-size: 10px;
      color: var(--bs-muted);
    }

    .bs-legend-dot {
      width: 9px;
      height: 9px;
      border-radius: 2px;
      flex-shrink: 0;
    }

    .bs-zoom-controls {
      display: flex;
      align-items: center;
      gap: 4px;
    }

    .bs-ctrl-btn {
      padding: 2px 8px;
      border: 1px solid var(--bs-border);
      border-radius: var(--bs-radius);
      background: var(--bs-surface);
      color: var(--bs-text);
      font-size: 11px;
      cursor: pointer;
      line-height: 1.5;
      transition: background 150ms, border-color 150ms;
    }

    .bs-ctrl-btn:hover {
      background: var(--bs-border);
      border-color: var(--bs-accent);
    }

    .bs-ctrl-btn:focus {
      outline: 2px solid var(--bs-accent);
      outline-offset: 1px;
    }

    #bs-zoom-label {
      font-size: 10px;
      color: var(--bs-muted);
      min-width: 32px;
      text-align: center;
      font-variant-numeric: tabular-nums;
    }

    #bs-graph-scroll {
      flex: 1;
      overflow: auto;
      padding: 16px 12px 24px;
    }

    #bs-zoom-wrapper {
      transform-origin: top left;
      transition: transform 150ms ease;
      display: inline-block;
    }

    svg#bs-dag {
      display: block;
      overflow: visible;
    }

    #bs-empty {
      display: none;
      padding: 48px 20px;
      text-align: center;
      color: var(--bs-muted);
      font-size: 11px;
    }

    /* ── SVG node interaction ────────────────────────────────────────────── */
    .bs-node {
      cursor: pointer;
    }

    .bs-node:focus {
      outline: none;
    }

    .bs-node:hover .bs-node-rect,
    .bs-node:focus .bs-node-rect {
      stroke-width: 2.5;
    }

    .bs-node.selected .bs-node-rect {
      stroke-width: 3;
    }

    .bs-node-text {
      pointer-events: none;
      dominant-baseline: middle;
      text-anchor: middle;
      font-size: 9.5px;
      font-weight: 700;
    }

    .bs-node-line {
      pointer-events: none;
      dominant-baseline: middle;
      text-anchor: middle;
      font-size: 8px;
      opacity: 0.7;
    }

    .bs-col-header-text {
      font-size: 9px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      text-anchor: middle;
      fill: var(--vscode-descriptionForeground, #8b949e);
    }

    .bs-separator {
      stroke: var(--vscode-panel-border, #30363d);
      stroke-width: 1;
      stroke-dasharray: 4 4;
      opacity: 0.4;
    }

    .bs-edge {
      fill: none;
      stroke-width: 1.5;
      stroke-linecap: round;
      opacity: 0.55;
    }

    /* When a node is selected, dim unrelated nodes */
    svg.has-selection .bs-node:not(.selected):not(.connected) {
      opacity: 0.3;
    }

    svg.has-selection .bs-edge:not(.active-edge) {
      opacity: 0.1;
    }

    /* ── Detail panel ────────────────────────────────────────────────────── */
    #bs-detail {
      width: 264px;
      flex-shrink: 0;
      border-left: 1px solid var(--bs-border);
      background: var(--bs-surface);
      display: flex;
      flex-direction: column;
      transform: translateX(100%);
      transition: transform 220ms cubic-bezier(0.2, 0, 0, 1);
      overflow: hidden;
    }

    #bs-detail.open {
      transform: translateX(0);
    }

    #bs-detail-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 9px 10px 8px;
      border-bottom: 1px solid var(--bs-border);
      flex-shrink: 0;
    }

    #bs-detail-title {
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--bs-muted);
    }

    #bs-detail-close {
      padding: 2px 6px;
      border: none;
      border-radius: var(--bs-radius);
      background: transparent;
      color: var(--bs-muted);
      cursor: pointer;
      font-size: 12px;
      line-height: 1;
      transition: background 150ms, color 150ms;
    }

    #bs-detail-close:hover {
      background: var(--bs-border);
      color: var(--bs-text);
    }

    #bs-detail-close:focus {
      outline: 2px solid var(--bs-accent);
      outline-offset: 1px;
    }

    #bs-detail-body {
      flex: 1;
      overflow-y: auto;
      padding: 12px 12px 20px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    #bs-detail-placeholder {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 4px;
      color: var(--bs-muted);
      font-size: 11px;
      text-align: center;
      opacity: 0.6;
      padding: 20px;
    }

    .bs-detail-var {
      font-family: var(--vscode-editor-font-family, 'Consolas', 'Courier New', monospace);
      font-size: 15px;
      font-weight: 700;
      color: var(--bs-text);
      word-break: break-all;
      line-height: 1.2;
    }

    .bs-detail-badges {
      display: flex;
      gap: 5px;
      flex-wrap: wrap;
    }

    .bs-badge {
      display: inline-flex;
      align-items: center;
      padding: 2px 7px;
      border-radius: 3px;
      font-size: 9.5px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .bs-badge-critical { background: rgba(248,81,73,0.2);  color: var(--bs-critical); border: 1px solid rgba(248,81,73,0.4); }
    .bs-badge-warning  { background: rgba(227,179,65,0.2); color: var(--bs-warning);  border: 1px solid rgba(227,179,65,0.4); }
    .bs-badge-info     { background: rgba(121,192,255,0.15); color: var(--bs-info);   border: 1px solid rgba(121,192,255,0.35); }

    .bs-badge-role-root        { background: rgba(245,200,66,0.15); color: #f5c842; border: 1px solid rgba(245,200,66,0.35); }
    .bs-badge-role-intermediate { background: rgba(245,124,66,0.15); color: #f57c42; border: 1px solid rgba(245,124,66,0.35); }
    .bs-badge-role-failure      { background: rgba(224,48,48,0.15);  color: #e03030; border: 1px solid rgba(224,48,48,0.35); }

    .bs-detail-section {
      display: flex;
      flex-direction: column;
      gap: 3px;
    }

    .bs-detail-section-label {
      font-size: 9px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: var(--bs-muted);
    }

    .bs-detail-section-value {
      font-size: 11px;
      color: var(--bs-text);
      line-height: 1.5;
    }

    .bs-go-to-line {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 4px 9px;
      border: 1px solid var(--bs-border);
      border-radius: var(--bs-radius);
      background: transparent;
      color: var(--bs-accent, #79c0ff);
      font-size: 11px;
      font-family: var(--vscode-editor-font-family, monospace);
      cursor: pointer;
      transition: background 150ms, border-color 150ms;
      width: fit-content;
    }

    .bs-go-to-line:hover {
      background: var(--bs-border);
      border-color: var(--bs-accent, #79c0ff);
    }

    .bs-go-to-line:focus {
      outline: 2px solid var(--bs-accent);
      outline-offset: 1px;
    }

    .bs-fix-hint-box {
      background: rgba(88,166,255,0.07);
      border: 1px solid rgba(88,166,255,0.2);
      border-radius: var(--bs-radius);
      padding: 7px 9px;
      font-size: 10.5px;
      color: var(--bs-text);
      line-height: 1.5;
    }

    .bs-divider {
      height: 1px;
      background: var(--bs-border);
      margin: 2px 0;
    }
  </style>
</head>
<body>

  <!-- Header bar -->
  <div id="bs-header">
    <div id="bs-brand">
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <path d="M7 1L13 4V10L7 13L1 10V4L7 1Z" stroke="#f5c842" stroke-width="1.4" fill="rgba(245,200,66,0.15)"/>
        <circle cx="7" cy="7" r="2" fill="#f5c842"/>
      </svg>
      BlackSwan
    </div>
    <div id="bs-run-meta" aria-label="Run metadata"></div>
  </div>

  <!-- KPI strip -->
  <div id="bs-kpi-bar" role="region" aria-label="Run statistics">
    <div class="bs-kpi">
      <div class="bs-kpi-label">Critical</div>
      <div class="bs-kpi-value kpi-critical" id="kpi-critical">—</div>
    </div>
    <div class="bs-kpi">
      <div class="bs-kpi-label">Warning</div>
      <div class="bs-kpi-value kpi-warning" id="kpi-warning">—</div>
    </div>
    <div class="bs-kpi">
      <div class="bs-kpi-label">Failure Rate</div>
      <div class="bs-kpi-value kpi-rate" id="kpi-rate">—</div>
    </div>
    <div class="bs-kpi">
      <div class="bs-kpi-label">Iterations</div>
      <div class="bs-kpi-value kpi-neutral" id="kpi-iters">—</div>
      <div class="bs-kpi-sub" id="kpi-runtime"></div>
    </div>
  </div>

  <!-- Main split -->
  <div id="bs-main">

    <!-- Graph area -->
    <div id="bs-graph-area">
      <!-- Toolbar: legend + zoom -->
      <div id="bs-graph-toolbar">
        <div class="bs-legend" aria-label="Node legend">
          <div class="bs-legend-item">
            <div class="bs-legend-dot" style="background:#f5c842;"></div>Root Cause
          </div>
          <div class="bs-legend-item">
            <div class="bs-legend-dot" style="background:#f57c42;"></div>Propagation
          </div>
          <div class="bs-legend-item">
            <div class="bs-legend-dot" style="background:#e03030;"></div>Failure Site
          </div>
        </div>
        <div class="bs-zoom-controls" role="group" aria-label="Zoom controls">
          <button class="bs-ctrl-btn" id="bs-zoom-out"  title="Zoom out"  aria-label="Zoom out">−</button>
          <span id="bs-zoom-label" aria-live="polite">100%</span>
          <button class="bs-ctrl-btn" id="bs-zoom-in"   title="Zoom in"   aria-label="Zoom in">+</button>
          <button class="bs-ctrl-btn" id="bs-zoom-reset" title="Reset zoom" aria-label="Reset zoom">Reset</button>
        </div>
      </div>

      <!-- Scrollable graph canvas -->
      <div id="bs-graph-scroll">
        <div id="bs-zoom-wrapper">
          <svg id="bs-dag" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <!-- Glow filter: root_input (yellow) -->
              <filter id="glow-yellow" x="-60%" y="-60%" width="220%" height="220%">
                <feGaussianBlur in="SourceGraphic" stdDeviation="5" result="blur"/>
                <feMerge>
                  <feMergeNode in="blur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
              <!-- Glow filter: intermediate (orange) -->
              <filter id="glow-orange" x="-60%" y="-60%" width="220%" height="220%">
                <feGaussianBlur in="SourceGraphic" stdDeviation="5" result="blur"/>
                <feMerge>
                  <feMergeNode in="blur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
              <!-- Glow filter: failure_site (red) -->
              <filter id="glow-red" x="-60%" y="-60%" width="220%" height="220%">
                <feGaussianBlur in="SourceGraphic" stdDeviation="6" result="blur"/>
                <feMerge>
                  <feMergeNode in="blur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
              <!-- Glow filter: selected node (bright white halo) -->
              <filter id="glow-selected" x="-60%" y="-60%" width="220%" height="220%">
                <feGaussianBlur in="SourceGraphic" stdDeviation="4" result="blur"/>
                <feMerge>
                  <feMergeNode in="blur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
              <!-- Arrow marker: yellow edge -->
              <marker id="arrow-yellow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
                <path d="M0,0 L0,8 L8,4 Z" fill="#f5c842" opacity="0.7"/>
              </marker>
              <!-- Arrow marker: orange edge -->
              <marker id="arrow-orange" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
                <path d="M0,0 L0,8 L8,4 Z" fill="#f57c42" opacity="0.7"/>
              </marker>
              <!-- Arrow marker: red edge -->
              <marker id="arrow-red" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
                <path d="M0,0 L0,8 L8,4 Z" fill="#e03030" opacity="0.7"/>
              </marker>
            </defs>
          </svg>
          <div id="bs-empty" aria-live="polite">No dependency data available for this run.</div>
        </div>
      </div>
    </div>

    <!-- Detail panel (slides in from the right) -->
    <div id="bs-detail" role="complementary" aria-label="Node detail">
      <div id="bs-detail-header">
        <span id="bs-detail-title">Node Detail</span>
        <button id="bs-detail-close" aria-label="Close detail panel" title="Close">&#x2715;</button>
      </div>

      <!-- Shown when a node is selected -->
      <div id="bs-detail-body" style="display:none;">
        <div class="bs-detail-var" id="d-variable"></div>

        <div class="bs-detail-badges" id="d-badges"></div>

        <div class="bs-divider"></div>

        <div class="bs-detail-section">
          <div class="bs-detail-section-label">Source Line</div>
          <button class="bs-go-to-line" id="d-goto" aria-label="Navigate to source line">
            <svg width="11" height="11" viewBox="0 0 11 11" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
              <path d="M1 10L10 1M10 1H4M10 1V7" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span id="d-line-text">L—</span>
          </button>
        </div>

        <div class="bs-detail-section" id="d-type-section" style="display:none;">
          <div class="bs-detail-section-label">Failure Type</div>
          <div class="bs-detail-section-value" id="d-failure-type"></div>
        </div>

        <div class="bs-detail-section" id="d-msg-section" style="display:none;">
          <div class="bs-detail-section-label">Description</div>
          <div class="bs-detail-section-value" id="d-message"></div>
        </div>

        <div class="bs-detail-section" id="d-freq-section" style="display:none;">
          <div class="bs-detail-section-label">Frequency</div>
          <div class="bs-detail-section-value" id="d-frequency"></div>
        </div>

        <div class="bs-detail-section" id="d-conf-section" style="display:none;">
          <div class="bs-detail-section-label">Attribution</div>
          <div class="bs-detail-section-value" id="d-confidence"></div>
        </div>

        <div class="bs-detail-section" id="d-fix-section" style="display:none;">
          <div class="bs-detail-section-label">Fix Hint</div>
          <div class="bs-fix-hint-box" id="d-fix-hint"></div>
        </div>
      </div>

      <!-- Shown when no node is selected -->
      <div id="bs-detail-placeholder">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" style="opacity:0.4; margin-bottom:6px;">
          <circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.5"/>
          <path d="M12 8V12L14 14" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
        <div>Click any node</div>
        <div style="opacity:0.6; font-size:10px; margin-top:2px;">to inspect details</div>
      </div>
    </div>

  </div><!-- #bs-main -->

  <script nonce="${nonce}">
  (function () {
    'use strict';

    /* ── Constants ──────────────────────────────────────────────────────────── */
    var COLORS = {
      root_input:   '#f5c842',
      intermediate: '#f57c42',
      failure_site: '#e03030'
    };

    var STROKE_COLORS = {
      root_input:   '#c9a21a',
      intermediate: '#c95f20',
      failure_site: '#a51f1f'
    };

    var FILL_BG = {
      root_input:   '#1e1600',
      intermediate: '#1e0c00',
      failure_site: '#1e0808'
    };

    var TEXT_COLORS = {
      root_input:   '#f5c842',
      intermediate: '#f57c42',
      failure_site: '#e03030'
    };

    var GLOW_FILTERS = {
      root_input:   'glow-yellow',
      intermediate: 'glow-orange',
      failure_site: 'glow-red'
    };

    var ARROW_MARKERS = {
      root_input:   'arrow-yellow',
      intermediate: 'arrow-orange',
      failure_site: 'arrow-red'
    };

    var ROLE_LABELS = {
      root_input:   'Root Cause',
      intermediate: 'Propagation',
      failure_site: 'Failure Site'
    };

    var COL_HEADERS = {
      root_input:   'Root Cause',
      intermediate: 'Propagation Path',
      failure_site: 'Failure Site'
    };

    var FAILURE_TYPE_LABELS = {
      nan_inf:               'NaN / Inf',
      division_by_zero:      'Division by Zero',
      non_psd_matrix:        'Non-PSD Matrix',
      ill_conditioned_matrix: 'Ill-Conditioned Matrix',
      bounds_exceeded:       'Bounds Exceeded',
      division_instability:  'Division Instability',
      exploding_gradient:    'Exploding Gradient',
      regime_shift:          'Regime Shift',
      logical_invariant:     'Logical Invariant'
    };

    var COL_X = { root_input: 110, intermediate: 330, failure_site: 550 };
    var NODE_W  = 144;
    var NODE_H  = 54;
    var NODE_RX = 8;
    var ROW_H   = 108;
    var SVG_W   = 680;
    var SVG_MIN_H = 260;
    var COL_HEADER_Y = 26;
    var NODES_START_Y = 62; /* vertical offset where nodes begin (below col headers) */

    /* ── State ─────────────────────────────────────────────────────────────── */
    var vscode         = acquireVsCodeApi();
    var currentData    = null;
    var selectedNodeId = null;
    var scale          = 1.0;
    var SCALE_STEP     = 0.15;
    var MIN_SCALE      = 0.35;
    var MAX_SCALE      = 2.5;

    /* ── SVG namespace helpers ─────────────────────────────────────────────── */
    var NS = 'http://www.w3.org/2000/svg';

    function svgEl(tag, attrs) {
      var e = document.createElementNS(NS, tag);
      if (attrs) {
        for (var k in attrs) {
          if (Object.prototype.hasOwnProperty.call(attrs, k)) {
            e.setAttribute(k, attrs[k]);
          }
        }
      }
      return e;
    }

    function svgTxt(tag, attrs, text) {
      var e = svgEl(tag, attrs);
      e.textContent = text;
      return e;
    }

    /* ── Layout ──────────────────────────────────────────────────────────── */
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

      var nodesH = maxCount * ROW_H;
      var svgH   = Math.max(nodesH + NODES_START_Y + ROW_H / 2, SVG_MIN_H);

      var pos = {};
      Object.keys(byRole).forEach(function (role) {
        var col    = byRole[role];
        var totalH = col.length * ROW_H;
        /* Centre the column vertically within the available node area */
        var availH = svgH - NODES_START_Y;
        var startY = NODES_START_Y + (availH - totalH) / 2 + ROW_H / 2;
        col.forEach(function (n, i) {
          pos[n.id] = { x: COL_X[role], y: startY + i * ROW_H };
        });
      });

      return { pos: pos, svgH: svgH };
    }

    /* ── Edge colour ─────────────────────────────────────────────────────── */
    function edgeColorForTarget(targetRole) {
      return COLORS[targetRole] || '#888';
    }

    function arrowMarkerForTarget(targetRole) {
      return ARROW_MARKERS[targetRole] || 'arrow-orange';
    }

    /* ── Render ──────────────────────────────────────────────────────────── */
    function render(data) {
      currentData    = data;
      selectedNodeId = null;

      updateHeader(data);
      updateKPIs(data);
      renderGraph(data);
    }

    function renderGraph(data) {
      var svg   = document.getElementById('bs-dag');
      var empty = document.getElementById('bs-empty');

      /* Clear previous content but keep <defs> */
      var defs = svg.querySelector('defs');
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      if (defs) svg.appendChild(defs);

      if (!data || !data.nodes || data.nodes.length === 0) {
        empty.style.display = 'block';
        svg.style.display   = 'none';
        return;
      }

      empty.style.display = 'none';
      svg.style.display   = 'block';
      svg.classList.remove('has-selection');

      var layout = computeLayout(data.nodes);
      var pos    = layout.pos;
      var svgH   = layout.svgH;

      svg.setAttribute('viewBox', '0 0 ' + SVG_W + ' ' + svgH);
      svg.setAttribute('width',   String(SVG_W));
      svg.setAttribute('height',  String(svgH));

      /* ── Column headers ──────────────────────────────────────────────── */
      ['root_input', 'intermediate', 'failure_site'].forEach(function (role) {
        var hdr = svgTxt('text', {
          'class': 'bs-col-header-text',
          x:       String(COL_X[role]),
          y:       String(COL_HEADER_Y)
        }, COL_HEADERS[role]);
        svg.appendChild(hdr);
      });

      /* ── Column separator lines ──────────────────────────────────────── */
      var sep1X = (COL_X.root_input + COL_X.intermediate) / 2;
      var sep2X = (COL_X.intermediate + COL_X.failure_site) / 2;
      [sep1X, sep2X].forEach(function (x) {
        svg.appendChild(svgEl('line', {
          'class': 'bs-separator',
          x1: String(x), y1: String(NODES_START_Y - 12),
          x2: String(x), y2: String(svgH - 8)
        }));
      });

      /* ── Edges (behind nodes) ────────────────────────────────────────── */
      data.edges.forEach(function (e) {
        var s = pos[e.source], t = pos[e.target];
        if (!s || !t) return;

        /* Find target node to determine color */
        var targetNode = null;
        for (var i = 0; i < data.nodes.length; i++) {
          if (data.nodes[i].id === e.target) { targetNode = data.nodes[i]; break; }
        }
        var tRole  = targetNode ? targetNode.role : 'intermediate';
        var eColor = edgeColorForTarget(tRole);
        var marker = arrowMarkerForTarget(tRole);

        /* Cubic bezier from right edge of source to left edge of target */
        var sx  = s.x + NODE_W / 2;
        var sy  = s.y;
        var tx  = t.x - NODE_W / 2 - 6; /* 6px gap before arrowhead */
        var ty  = t.y;
        var cpX = (sx + tx) / 2;

        var d = 'M ' + sx + ' ' + sy +
                ' C ' + cpX + ' ' + sy +
                ', '  + cpX + ' ' + ty +
                ', '  + tx  + ' ' + ty;

        var path = svgEl('path', {
          d:                  d,
          'class':            'bs-edge',
          stroke:             eColor,
          'marker-end':       'url(#' + marker + ')',
          'data-source':      e.source,
          'data-target':      e.target
        });
        svg.appendChild(path);
      });

      /* ── Nodes ───────────────────────────────────────────────────────── */
      data.nodes.forEach(function (n) {
        var p = pos[n.id];
        if (!p) return;

        var fillColor   = COLORS[n.role]      || '#888';
        var strokeColor = STROKE_COLORS[n.role] || '#555';
        var bgColor     = FILL_BG[n.role]      || '#111';
        var textColor   = TEXT_COLORS[n.role]   || '#fff';
        var glowFilter  = GLOW_FILTERS[n.role]  || '';
        var roleLabel   = ROLE_LABELS[n.role]   || n.role;

        var halfW = NODE_W / 2;
        var halfH = NODE_H / 2;

        var g = svgEl('g', {
          'class':     'bs-node',
          transform:   'translate(' + p.x + ',' + p.y + ')',
          tabindex:    '0',
          role:        'button',
          'data-id':   n.id,
          'aria-label': n.variable + ', line ' + n.line + ', ' + roleLabel
        });

        /* Dark background fill rect (no glow) */
        g.appendChild(svgEl('rect', {
          x:      String(-halfW),
          y:      String(-halfH),
          width:  String(NODE_W),
          height: String(NODE_H),
          rx:     String(NODE_RX),
          fill:   bgColor,
          stroke: strokeColor,
          'stroke-width': '1.5',
          'class': 'bs-node-rect'
        }));

        /* Coloured glow halo behind the fill (separate rect so glow doesn't bleed into text) */
        var glowRect = svgEl('rect', {
          x:      String(-halfW),
          y:      String(-halfH),
          width:  String(NODE_W),
          height: String(NODE_H),
          rx:     String(NODE_RX),
          fill:   'none',
          stroke: fillColor,
          'stroke-width': '1.5',
          filter: 'url(#' + glowFilter + ')',
          opacity: '0.6',
          'pointer-events': 'none'
        });
        g.appendChild(glowRect);

        /* Variable name — truncate if needed */
        var varLabel = n.variable.length > 15
          ? n.variable.slice(0, 14) + '\u2026'
          : n.variable;

        g.appendChild(svgTxt('text', {
          'class': 'bs-node-text',
          x:       '0',
          y:       '-7',
          fill:    textColor
        }, varLabel));

        /* Line number label */
        g.appendChild(svgTxt('text', {
          'class': 'bs-node-line',
          x:       '0',
          y:       '11',
          fill:    textColor
        }, 'L' + n.line));

        /* Click / keyboard navigation */
        g.addEventListener('click', function () {
          handleNodeSelect(n);
        });
        g.addEventListener('keydown', function (ev) {
          if (ev.key === 'Enter' || ev.key === ' ') {
            ev.preventDefault();
            handleNodeSelect(n);
          }
        });

        svg.appendChild(g);
      });
    }

    /* ── Node selection ──────────────────────────────────────────────────── */
    function handleNodeSelect(node) {
      var svg = document.getElementById('bs-dag');

      /* Toggle off if already selected */
      if (selectedNodeId === node.id) {
        deselectAll(svg);
        closeDetail();
        return;
      }

      deselectAll(svg);
      selectedNodeId = node.id;

      /* Mark the selected node */
      var nodeEls = svg.querySelectorAll('.bs-node');
      var connectedIds = new Set();
      connectedIds.add(node.id);

      /* Find connected nodes via edges */
      if (currentData && currentData.edges) {
        currentData.edges.forEach(function (e) {
          if (e.source === node.id) connectedIds.add(e.target);
          if (e.target === node.id) connectedIds.add(e.source);
        });
      }

      nodeEls.forEach(function (el) {
        var nid = el.getAttribute('data-id');
        if (nid === node.id) {
          el.classList.add('selected');
        } else if (connectedIds.has(nid)) {
          el.classList.add('connected');
        }
      });

      /* Highlight connected edges */
      var edgeEls = svg.querySelectorAll('.bs-edge');
      edgeEls.forEach(function (el) {
        var src = el.getAttribute('data-source');
        var tgt = el.getAttribute('data-target');
        if (src === node.id || tgt === node.id) {
          el.classList.add('active-edge');
        }
      });

      svg.classList.add('has-selection');

      /* Show detail panel */
      showDetail(node);
    }

    function deselectAll(svg) {
      selectedNodeId = null;
      svg.classList.remove('has-selection');
      svg.querySelectorAll('.bs-node').forEach(function (e) {
        e.classList.remove('selected', 'connected');
      });
      svg.querySelectorAll('.bs-edge').forEach(function (e) {
        e.classList.remove('active-edge');
      });
    }

    /* ── Detail panel ────────────────────────────────────────────────────── */
    function showDetail(node) {
      var panel       = document.getElementById('bs-detail');
      var body        = document.getElementById('bs-detail-body');
      var placeholder = document.getElementById('bs-detail-placeholder');

      /* Variable name */
      document.getElementById('d-variable').textContent = node.variable;

      /* Badges: role */
      var badges    = document.getElementById('d-badges');
      badges.innerHTML = '';
      var roleCls = {
        root_input:   'bs-badge-role-root',
        intermediate: 'bs-badge-role-intermediate',
        failure_site: 'bs-badge-role-failure'
      }[node.role] || '';

      var roleBadge = document.createElement('span');
      roleBadge.className = 'bs-badge ' + roleCls;
      roleBadge.textContent = ROLE_LABELS[node.role] || node.role;
      badges.appendChild(roleBadge);

      /* Severity badge (only for failure_site nodes with detail) */
      var detail = currentData && currentData.nodeDetails && currentData.nodeDetails[node.id];
      if (detail && detail.severity) {
        var sevBadge = document.createElement('span');
        var sevCls   = { critical: 'bs-badge-critical', warning: 'bs-badge-warning', info: 'bs-badge-info' }[detail.severity] || '';
        sevBadge.className   = 'bs-badge ' + sevCls;
        sevBadge.textContent = detail.severity.toUpperCase();
        badges.appendChild(sevBadge);
      }

      /* Go-to-line button */
      var gotoBtn  = document.getElementById('d-goto');
      var lineText = document.getElementById('d-line-text');
      lineText.textContent = 'Line ' + node.line;
      gotoBtn.onclick = function () {
        vscode.postMessage({
          type: 'navigate',
          line: node.line,
          uri:  currentData ? currentData.documentUri : ''
        });
      };

      /* Failure-site specific sections */
      function setSection(id, value, label) {
        var section = document.getElementById(id + '-section');
        var el      = document.getElementById(id);
        if (value) {
          section.style.display = '';
          el.textContent        = label ? label : value;
        } else {
          section.style.display = 'none';
          el.textContent        = '';
        }
      }

      if (detail) {
        setSection('d-failure-type', detail.failure_type,
          FAILURE_TYPE_LABELS[detail.failure_type] || detail.failure_type);
        setSection('d-message',   detail.message,   null);
        setSection('d-frequency', detail.frequency, null);

        var confEl    = document.getElementById('d-conf-section');
        var confValue = document.getElementById('d-confidence');
        if (detail.confidence) {
          confEl.style.display    = '';
          confValue.textContent   = detail.confidence.charAt(0).toUpperCase() +
                                    detail.confidence.slice(1) + ' confidence';
        } else {
          confEl.style.display = 'none';
        }

        setSection('d-fix-hint', detail.fix_hint, null);
      } else {
        ['d-type', 'd-msg', 'd-freq', 'd-conf', 'd-fix'].forEach(function (p) {
          var s = document.getElementById(p.replace('d-', 'd-') + '-section');
          if (s) s.style.display = 'none';
        });
        /* Hide individually */
        ['d-type-section','d-msg-section','d-freq-section','d-conf-section','d-fix-section']
          .forEach(function (id) {
            var s = document.getElementById(id);
            if (s) s.style.display = 'none';
          });
      }

      body.style.display        = '';
      placeholder.style.display = 'none';
      panel.classList.add('open');
    }

    function closeDetail() {
      var svg = document.getElementById('bs-dag');
      deselectAll(svg);

      var panel       = document.getElementById('bs-detail');
      var body        = document.getElementById('bs-detail-body');
      var placeholder = document.getElementById('bs-detail-placeholder');
      panel.classList.remove('open');
      body.style.display        = 'none';
      placeholder.style.display = '';
    }

    document.getElementById('bs-detail-close').addEventListener('click', closeDetail);

    /* ── Header & KPI updates ────────────────────────────────────────────── */
    function updateHeader(data) {
      var metaEl = document.getElementById('bs-run-meta');
      metaEl.innerHTML = '';
      if (!data || !data.summary) return;
      var s = data.summary;

      var pills = [
        { label: 'Scenario', value: s.scenario_name.replace(/_/g, ' ') },
        { label: 'Seed',     value: String(s.seed) },
        { label: 'Mode',     value: s.mode }
      ];

      pills.forEach(function (p) {
        var pill = document.createElement('div');
        pill.className = 'bs-meta-pill';
        pill.innerHTML =
          '<span class="bs-meta-label">' + escHtml(p.label) + '</span>' +
          '<span class="bs-meta-value">' + escHtml(p.value) + '</span>';
        metaEl.appendChild(pill);
      });
    }

    function updateKPIs(data) {
      if (!data || !data.summary) {
        setText('kpi-critical', '—');
        setText('kpi-warning',  '—');
        setText('kpi-rate',     '—');
        setText('kpi-iters',    '—');
        setText('kpi-runtime',  '');
        return;
      }
      var s = data.summary;
      setText('kpi-critical', String(s.critical_count));
      setText('kpi-warning',  String(s.warning_count));
      setText('kpi-rate',     (s.failure_rate * 100).toFixed(1) + '%');
      setText('kpi-iters',    s.iterations_completed.toLocaleString());
      setText('kpi-runtime',  (s.runtime_ms / 1000).toFixed(2) + 's');
    }

    function setText(id, text) {
      var el = document.getElementById(id);
      if (el) el.textContent = text;
    }

    function escHtml(str) {
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    }

    /* ── Zoom ────────────────────────────────────────────────────────────── */
    function setZoom(newScale) {
      scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, newScale));
      var wrapper = document.getElementById('bs-zoom-wrapper');
      wrapper.style.transform = 'scale(' + scale.toFixed(2) + ')';
      var label = document.getElementById('bs-zoom-label');
      label.textContent = Math.round(scale * 100) + '%';
    }

    document.getElementById('bs-zoom-in').addEventListener('click', function () {
      setZoom(scale + SCALE_STEP);
    });
    document.getElementById('bs-zoom-out').addEventListener('click', function () {
      setZoom(scale - SCALE_STEP);
    });
    document.getElementById('bs-zoom-reset').addEventListener('click', function () {
      setZoom(1.0);
    });

    /* ── Initial render ──────────────────────────────────────────────────── */
    var initialData = ${dataJson};
    render(initialData);

    /* ── Live updates from the extension host ────────────────────────────── */
    window.addEventListener('message', function (event) {
      if (event.data && event.data.type === 'update') {
        closeDetail();
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
 * 32 alphanumeric characters gives ~190 bits of entropy.
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
 * - Navigation messages from the webview (node click → Go to Line) are
 *   translated into `vscode.open` commands that scroll the editor.
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
   * @param response    The full engine response — provides shatter points,
   *                    scenario card, and summary for the rich UI.
   * @param documentUri URI of the Python file that was stress-tested.
   */
  show(response: BlackSwanResponse, documentUri: vscode.Uri): void {
    const dagData = buildDagData(response.shatter_points, documentUri, response);

    if (this._panel) {
      this._panel.reveal(vscode.ViewColumn.Beside);
      void this._panel.webview.postMessage({ type: "update", data: dagData });
      return;
    }

    this._panel = vscode.window.createWebviewPanel(
      DagPanelController.viewType,
      "BlackSwan — Dependency Graph",
      vscode.ViewColumn.Beside,
      {
        enableScripts:           true,
        retainContextWhenHidden: true,
        localResourceRoots:      [],
      },
    );

    this._panel.webview.html = buildDagHtml(dagData, getNonce());

    this._panel.webview.onDidReceiveMessage(
      (message: { type: string; line: number; uri: string }) => {
        if (message.type !== "navigate") return;
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
