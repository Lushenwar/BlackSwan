/**
 * dagPanel.ts — Dependency Graph webview panel for BlackSwan.
 *
 * Professional fintech dark theme featuring:
 *   • Summary header with scenario, seed, mode, and runtime metadata
 *   • KPI strip with severity-breakdown failure counts and failure rate
 *   • Interactive SVG DAG — rounded-rect nodes, failure-site glow, cubic bezier edges
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
    /* ── RESET ──────────────────────────────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
      }
    }

    /* ── ANIMATIONS ─────────────────────────────────────────────────────── */
    @keyframes fadeSlideIn {
      from { opacity: 0; transform: translateY(6px); }
      to   { opacity: 1; transform: translateY(0);   }
    }

    @keyframes kpiIn {
      from { opacity: 0; transform: translateY(4px); }
      to   { opacity: 1; transform: translateY(0);   }
    }

    @keyframes pulseSite {
      0%, 100% { transform: scale(1);   opacity: 0.3; }
      50%       { transform: scale(2.6); opacity: 0;   }
    }

    /* ── DESIGN TOKENS ──────────────────────────────────────────────────── */
    :root {
      /* Theme-aware — falls back to GitHub-dark values when not inside VS Code */
      --bs-bg:       var(--vscode-editor-background,       #0d1117);
      --bs-surface:  var(--vscode-sideBar-background,      #161b22);
      --bs-surface2: var(--vscode-editorWidget-background, #1e2942);
      --bs-border:   var(--vscode-panel-border,            #21262d);
      --bs-text:     var(--vscode-editor-foreground,       #e6edf3);
      --bs-muted:    var(--vscode-descriptionForeground,   #7a8899);
      --bs-accent:   var(--vscode-focusBorder,             #388bfd);
      --bs-mono:     var(--vscode-editor-font-family, 'Consolas', 'SF Mono', 'Monaco',
                         'Fira Mono', 'DejaVu Sans Mono', monospace);
      --bs-sans:     var(--vscode-font-family, 'Segoe UI', system-ui, -apple-system, sans-serif);

      /* Type scale — 4:5 ratio anchored at 12px, 11px label floor */
      --bs-sz-display: 20px;   /* KPI numbers */
      --bs-sz-title:   14px;   /* Variable names, state titles */
      --bs-sz-body:    12px;   /* Field values, fix hints, prose */
      --bs-sz-label:   11px;   /* Badges, card labels, pills, column headers */

      /* Leading */
      --bs-lh-body:    1.4;
      --bs-lh-tight:   1.2;
      --bs-lh-display: 1.0;

      /* Semantic colours */
      --bs-critical: #f85149;
      --bs-warning:  #e3b341;
      --bs-success:  #3fb950;
      --bs-info:     #79c0ff;

      /* Node colours — CONTRACTUAL: test suite asserts these exact hex values */
      --bs-root:  #f5c842;
      --bs-inter: #f57c42;
      --bs-site:  #e03030;

      --bs-radius: 6px;
      --bs-radius-lg: 10px;
    }

    /* ── BASE ───────────────────────────────────────────────────────────── */
    html, body {
      height: 100%;
      overflow: hidden;
      background: var(--bs-bg);
      color: var(--bs-text);
      font-family: var(--bs-sans);
      font-size: 12px;
      display: flex;
      flex-direction: column;
    }

    /* ── HEADER ─────────────────────────────────────────────────────────── */
    #bs-header {
      position: relative;
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 0 14px 0 17px;
      height: 38px;
      background: var(--bs-surface);
      border-bottom: 1px solid var(--bs-border);
      flex-shrink: 0;
    }

    /* Three-stop gradient accent line on the left edge */
    #bs-header::before {
      content: '';
      position: absolute;
      left: 0; top: 0; bottom: 0;
      width: 3px;
      background: linear-gradient(180deg, var(--bs-root) 0%, var(--bs-inter) 50%, var(--bs-site) 100%);
    }

    #bs-logo {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.09em;
      text-transform: uppercase;
      color: var(--bs-text);
      flex-shrink: 0;
      user-select: none;
    }

    #bs-meta-pills {
      display: flex;
      align-items: center;
      gap: 5px;
      flex: 1;
      flex-wrap: wrap;
      overflow: hidden;
    }

    .bs-pill {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      height: 20px;
      padding: 0 8px;
      border-radius: 10px;
      border: 1px solid var(--bs-border);
      font-size: var(--bs-sz-label);
      color: var(--bs-muted);
      white-space: nowrap;
    }

    .bs-pill-lbl {
      font-size: var(--bs-sz-label);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      opacity: 0.6;
    }

    .bs-pill-val {
      font-weight: 600;
      color: var(--bs-text);
    }

    .bs-pill-fast { border-color: rgba(56,139,253,0.35); }
    .bs-pill-fast .bs-pill-val { color: var(--bs-info); }
    .bs-pill-full { border-color: rgba(63,185,80,0.35); }
    .bs-pill-full .bs-pill-val { color: var(--bs-success); }

    /* ── KPI STRIP ──────────────────────────────────────────────────────── */
    #bs-kpi {
      display: flex;
      flex-shrink: 0;
      gap: 1px;
      background: var(--bs-border);
      border-bottom: 1px solid var(--bs-border);
    }

    .bs-kcard {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 2px;
      padding: 8px 12px 7px;
      background: var(--bs-surface);
      position: relative;
      overflow: hidden;
      min-width: 0;
      animation: kpiIn 0.35s cubic-bezier(0.16, 1, 0.3, 1) both;
    }

    .kc-critical { animation-delay: 0ms;   }
    .kc-warning  { animation-delay: 60ms;  }
    .kc-rate     { animation-delay: 120ms; flex: 2; }
    .kc-iters    { animation-delay: 180ms; }

    /* Coloured top-border accent per card */
    .bs-kcard::after {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2px;
      border-radius: 0 0 2px 2px;
    }
    .kc-critical::after { background: var(--bs-critical); }
    .kc-warning::after  { background: var(--bs-warning); }
    .kc-rate::after     { background: var(--bs-info); }
    .kc-iters::after    { background: var(--bs-accent); }

    .bs-kcard-lbl {
      font-size: var(--bs-sz-label);
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--bs-muted);
    }

    .bs-kcard-num {
      font-size: var(--bs-sz-display);
      font-weight: 700;
      line-height: var(--bs-lh-display);
      font-variant-numeric: tabular-nums;
    }

    .kn-critical { color: var(--bs-critical); }
    .kn-warning  { color: var(--bs-warning);  }
    .kn-rate     { color: var(--bs-info);      }
    .kn-iters    { color: var(--bs-text);      }

    .bs-kcard-sub {
      font-size: var(--bs-sz-label);
      color: var(--bs-muted);
      font-variant-numeric: tabular-nums;
    }

    /* Failure rate mini progress bar */
    #kpi-rate-bar-wrap {
      margin-top: 4px;
      height: 3px;
      border-radius: 2px;
      background: var(--bs-border);
      overflow: hidden;
      display: none;
    }

    #kpi-rate-bar {
      height: 100%;
      border-radius: 2px;
      background: linear-gradient(90deg, var(--bs-warning) 0%, var(--bs-critical) 100%);
      width: 0%;
      transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── TOOLBAR ────────────────────────────────────────────────────────── */
    #bs-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 5px 10px;
      background: var(--bs-surface);
      border-bottom: 1px solid var(--bs-border);
      flex-shrink: 0;
      gap: 8px;
    }

    #bs-legend {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .bs-leg-item {
      display: flex;
      align-items: center;
      gap: 5px;
      font-size: var(--bs-sz-label);
      color: var(--bs-muted);
      cursor: pointer;
      border-radius: 3px;
      padding: 1px 4px;
      transition: color 140ms, opacity 140ms;
      user-select: none;
    }

    .bs-leg-item:hover        { color: var(--bs-text); }
    .bs-leg-item:focus-visible { outline: 2px solid var(--bs-accent); outline-offset: 2px; border-radius: 3px; }
    .bs-leg-item:active        { opacity: 0.7; }

    .bs-leg-item.dimmed { opacity: 0.35; }

    .bs-leg-dot {
      width: 8px;
      height: 8px;
      border-radius: 2px;
      flex-shrink: 0;
    }

    #bs-zoom-grp {
      display: flex;
      align-items: center;
      gap: 3px;
    }

    .bs-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 2px 8px;
      border: 1px solid var(--bs-border);
      border-radius: var(--bs-radius);
      background: transparent;
      color: var(--bs-text);
      font-size: var(--bs-sz-label);
      cursor: pointer;
      line-height: 1.6;
      transition: background 140ms, border-color 140ms;
    }

    .bs-btn:hover        { background: var(--bs-border); border-color: var(--bs-accent); }
    .bs-btn:active       { background: var(--bs-surface2); transform: scale(0.97); }
    .bs-btn:focus-visible { outline: 2px solid var(--bs-accent); outline-offset: 1px; }

    #bs-zoom-pct {
      font-size: var(--bs-sz-label);
      color: var(--bs-muted);
      min-width: 34px;
      text-align: center;
      font-variant-numeric: tabular-nums;
    }

    /* ── CONTENT WRAPPER ────────────────────────────────────────────────── */
    #bs-body {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    /* ── EMPTY / SUCCESS STATES ─────────────────────────────────────────── */
    #bs-state {
      flex: 1;
      display: none;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 10px;
      padding: 40px 20px;
      text-align: center;
      color: var(--bs-muted);
      animation: fadeSlideIn 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
    }

    #bs-state.show { display: flex; }

    .bs-state-icon { margin-bottom: 2px; opacity: 0.45; }
    #bs-state.success .bs-state-icon { opacity: 1; }

    .bs-state-title {
      font-size: var(--bs-sz-title);
      font-weight: 600;
      color: var(--bs-text);
    }

    #bs-state.success .bs-state-title { color: var(--bs-success); }

    .bs-state-sub {
      font-size: var(--bs-sz-body);
      line-height: var(--bs-lh-body);
      max-width: 300px;
    }

    /* ── GRAPH SCROLL ───────────────────────────────────────────────────── */
    #bs-graph-scroll {
      flex: 1;
      overflow: auto;
      padding: 20px 18px 28px;
      display: none;
    }

    #bs-graph-scroll.show { display: block; }

    #bs-zoom-wrapper {
      transform-origin: top left;
      transition: transform 120ms ease;
      display: inline-block;
    }

    svg#bs-dag {
      display: block;
      overflow: visible;
    }

    /* ── SVG NODE STYLES ────────────────────────────────────────────────── */
    .bs-node { cursor: pointer; }
    .bs-node:focus { outline: none; }

    .bs-node-border { transition: stroke-width 100ms, opacity 160ms; }

    .bs-node:hover .bs-node-border,
    .bs-node:focus .bs-node-border { stroke-width: 2.5; }
    .bs-node.selected .bs-node-border { stroke-width: 3; }

    /* Pulse ring — only on failure_site nodes */
    .bs-pulse {
      pointer-events: none;
      transform-box: fill-box;
      transform-origin: center;
      animation: pulseSite 2.8s cubic-bezier(0.45, 0, 0.55, 1) 2;
    }

    .bs-node-var {
      pointer-events: none;
      dominant-baseline: middle;
      text-anchor: middle;
      font-weight: 700;
    }

    .bs-node-lineno {
      pointer-events: none;
      dominant-baseline: middle;
      text-anchor: middle;
      opacity: 0.62;
    }

    .bs-col-hdr {
      text-anchor: middle;
      font-size: 11px;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      opacity: 0.45;
    }

    .bs-sep { stroke-dasharray: 3 5; }

    .bs-edge {
      fill: none;
      stroke-width: 1.5;
      stroke-linecap: round;
    }

    /* Dim non-selected nodes/edges when selection is active */
    svg.has-sel .bs-node:not(.selected):not(.connected) .bs-node-bg     { opacity: 0.12; }
    svg.has-sel .bs-node:not(.selected):not(.connected) .bs-node-border { opacity: 0.12; }
    svg.has-sel .bs-node:not(.selected):not(.connected) .bs-pulse       { display: none; }
    svg.has-sel .bs-edge:not(.edge-on) { opacity: 0.07; }

    /* ── BOTTOM DETAIL DRAWER ───────────────────────────────────────────── */
    #bs-drawer {
      flex-shrink: 0;
      background: var(--bs-surface);
      border-top: 2px solid var(--bs-border);
      display: grid;
      grid-template-rows: 0fr;
      transition: grid-template-rows 230ms cubic-bezier(0.2, 0, 0, 1);
    }

    #bs-drawer.open { grid-template-rows: 1fr; }

    #bs-drawer-inner {
      overflow: hidden;
      min-height: 0;
      padding: 11px 16px 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    #bs-drawer-top {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }

    #bs-drawer-var {
      font-family: var(--bs-mono);
      font-size: var(--bs-sz-title);
      font-weight: 700;
      color: var(--bs-text);
      flex: 1;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    #bs-drawer-badges { display: flex; gap: 5px; flex-shrink: 0; flex-wrap: wrap; }

    #bs-drawer-close {
      flex-shrink: 0;
      width: 22px;
      height: 22px;
      border: none;
      border-radius: var(--bs-radius);
      background: transparent;
      color: var(--bs-muted);
      cursor: pointer;
      font-size: 13px;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background 140ms, color 140ms;
    }

    #bs-drawer-close:hover        { background: var(--bs-border); color: var(--bs-text); }
    #bs-drawer-close:active       { background: var(--bs-surface2); transform: scale(0.93); }
    #bs-drawer-close:focus-visible { outline: 2px solid var(--bs-accent); outline-offset: 1px; }

    #bs-drawer-grid {
      display: grid;
      grid-template-columns: auto auto 1fr;
      gap: 6px 18px;
      align-items: start;
    }

    .bs-field { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
    .bs-field.span2 { grid-column: span 2; }
    .bs-field.span3 { grid-column: 1 / -1; }

    .bs-field-lbl {
      font-size: var(--bs-sz-label);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--bs-muted);
      white-space: nowrap;
    }

    .bs-field-val {
      font-size: var(--bs-sz-body);
      color: var(--bs-text);
      line-height: var(--bs-lh-body);
      overflow: hidden;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
    }

    .bs-goto {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      height: 22px;
      padding: 0 8px;
      border: 1px solid var(--bs-border);
      border-radius: var(--bs-radius);
      background: transparent;
      color: var(--bs-accent, #79c0ff);
      font-family: var(--bs-mono);
      font-size: var(--bs-sz-label);
      cursor: pointer;
      width: fit-content;
      transition: background 140ms, border-color 140ms;
    }

    .bs-goto:hover        { background: var(--bs-border); border-color: var(--bs-accent); }
    .bs-goto:active       { background: var(--bs-surface2); transform: scale(0.97); }
    .bs-goto:focus-visible { outline: 2px solid var(--bs-accent); outline-offset: 1px; }

    .bs-fix-box {
      position: relative;
      background: var(--bs-surface2);
      border: 1px solid rgba(88,166,255,0.18);
      border-radius: var(--bs-radius);
      padding: 6px 52px 6px 8px;
      font-size: var(--bs-sz-body);
      color: var(--bs-text);
      line-height: var(--bs-lh-body);
    }

    .bs-copy {
      position: absolute;
      top: 4px; right: 5px;
      padding: 1px 6px;
      border: 1px solid var(--bs-border);
      border-radius: 3px;
      background: var(--bs-surface2);
      color: var(--bs-muted);
      font-size: var(--bs-sz-label);
      cursor: pointer;
      transition: color 140ms, border-color 140ms;
    }

    .bs-copy:hover        { color: var(--bs-text); border-color: var(--bs-accent); }
    .bs-copy:active       { background: var(--bs-border); transform: scale(0.97); }
    .bs-copy:focus-visible { outline: 2px solid var(--bs-accent); outline-offset: 1px; }

    /* ── BADGES ─────────────────────────────────────────────────────────── */
    .bs-badge {
      display: inline-flex;
      align-items: center;
      height: 18px;
      padding: 0 6px;
      border-radius: 3px;
      font-size: var(--bs-sz-label);
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border: 1px solid transparent;
    }

    .bd-critical { background: rgba(248,81,73,.15);  color: var(--bs-critical); border-color: rgba(248,81,73,.3); }
    .bd-warning  { background: rgba(227,179,65,.15); color: var(--bs-warning);  border-color: rgba(227,179,65,.3); }
    .bd-info     { background: rgba(121,192,255,.1); color: var(--bs-info);     border-color: rgba(121,192,255,.28); }
    .bd-root     { background: rgba(245,200,66,.12); color: #f5c842;            border-color: rgba(245,200,66,.3); }
    .bd-inter    { background: rgba(245,124,66,.12); color: #f57c42;            border-color: rgba(245,124,66,.3); }
    .bd-site     { background: rgba(224,48,48,.12);  color: #e03030;            border-color: rgba(224,48,48,.3); }
  </style>
</head>
<body>

  <!-- ── HEADER ────────────────────────────────────────────────────────── -->
  <header id="bs-header">
    <div id="bs-logo">
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
        <path d="M7 1L13 4V10L7 13L1 10V4L7 1Z"
              stroke="#f5c842" stroke-width="1.3" fill="rgba(245,200,66,0.1)"/>
        <circle cx="7" cy="7" r="1.8" fill="#f5c842"/>
      </svg>
      BlackSwan
    </div>
    <div id="bs-meta-pills" aria-label="Run metadata"></div>
  </header>

  <!-- ── KPI STRIP ─────────────────────────────────────────────────────── -->
  <div id="bs-kpi" role="region" aria-label="Run statistics">
    <div class="bs-kcard kc-critical">
      <div class="bs-kcard-lbl">Critical</div>
      <div class="bs-kcard-num kn-critical" id="kpi-critical">—</div>
    </div>
    <div class="bs-kcard kc-warning">
      <div class="bs-kcard-lbl">Warning</div>
      <div class="bs-kcard-num kn-warning" id="kpi-warning">—</div>
    </div>
    <div class="bs-kcard kc-rate">
      <div class="bs-kcard-lbl">Failure Rate</div>
      <div class="bs-kcard-num kn-rate" id="kpi-rate">—</div>
      <div id="kpi-rate-bar-wrap" role="progressbar" aria-valuemin="0" aria-valuemax="100">
        <div id="kpi-rate-bar"></div>
      </div>
    </div>
    <div class="bs-kcard kc-iters">
      <div class="bs-kcard-lbl">Iterations</div>
      <div class="bs-kcard-num kn-iters" id="kpi-iters">—</div>
      <div class="bs-kcard-sub" id="kpi-runtime"></div>
    </div>
  </div>

  <!-- ── TOOLBAR ───────────────────────────────────────────────────────── -->
  <div id="bs-toolbar">
    <div id="bs-legend" role="group" aria-label="Filter by node type">
      <div class="bs-leg-item" data-role="root_input"   title="Toggle root cause nodes" tabindex="0" role="checkbox" aria-checked="true">
        <div class="bs-leg-dot" style="background:#f5c842;"></div>Root Cause
      </div>
      <div class="bs-leg-item" data-role="intermediate"  title="Toggle propagation nodes" tabindex="0" role="checkbox" aria-checked="true">
        <div class="bs-leg-dot" style="background:#f57c42;"></div>Propagation
      </div>
      <div class="bs-leg-item" data-role="failure_site"  title="Toggle failure site nodes" tabindex="0" role="checkbox" aria-checked="true">
        <div class="bs-leg-dot" style="background:#e03030;"></div>Failure Site
      </div>
    </div>
    <div id="bs-zoom-grp" role="group" aria-label="Zoom controls">
      <button class="bs-btn" id="bs-zoom-out"   title="Zoom out (−)"  aria-label="Zoom out">−</button>
      <span id="bs-zoom-pct" aria-live="polite">100%</span>
      <button class="bs-btn" id="bs-zoom-in"    title="Zoom in (+)"   aria-label="Zoom in">+</button>
      <button class="bs-btn" id="bs-zoom-reset" title="Reset zoom"    aria-label="Reset zoom">Reset</button>
    </div>
  </div>

  <!-- ── MAIN BODY ─────────────────────────────────────────────────────── -->
  <div id="bs-body">

    <!-- Empty / success state (shown when graph has no nodes) -->
    <div id="bs-state" role="status" aria-live="polite">
      <div class="bs-state-icon">
        <svg id="bs-state-svg" width="38" height="38" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M12 2L21 6.8V17.2L12 22L3 17.2V6.8L12 2Z" stroke="currentColor" stroke-width="1.2"/>
          <path d="M9.5 6L11.5 11L9.5 13L13.5 18" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div class="bs-state-title" id="bs-state-title">Waiting for results</div>
      <div class="bs-state-sub"   id="bs-state-sub">
        Click &#9654; Run BlackSwan above a function to start a stress test.
      </div>
    </div>

    <!-- Graph canvas -->
    <div id="bs-graph-scroll" aria-label="Dependency graph">
      <div id="bs-zoom-wrapper">
        <svg id="bs-dag" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <!-- Glow filter — failure_site only; reserved for where the math breaks -->
            <filter id="gf-r" x="-80%" y="-80%" width="260%" height="260%">
              <feGaussianBlur in="SourceGraphic" stdDeviation="5.5" result="b"/>
              <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <!-- Arrow markers -->
            <marker id="ar-y" markerWidth="7" markerHeight="7" refX="5.5" refY="3.5" orient="auto">
              <path d="M0,0 L0,7 L7,3.5 Z" fill="#f5c842" opacity="0.6"/>
            </marker>
            <marker id="ar-o" markerWidth="7" markerHeight="7" refX="5.5" refY="3.5" orient="auto">
              <path d="M0,0 L0,7 L7,3.5 Z" fill="#f57c42" opacity="0.6"/>
            </marker>
            <marker id="ar-r" markerWidth="7" markerHeight="7" refX="5.5" refY="3.5" orient="auto">
              <path d="M0,0 L0,7 L7,3.5 Z" fill="#e03030" opacity="0.6"/>
            </marker>
          </defs>
        </svg>
      </div>
    </div>

  </div><!-- #bs-body -->

  <!-- ── BOTTOM DETAIL DRAWER ──────────────────────────────────────────── -->
  <div id="bs-drawer" role="complementary" aria-label="Node detail" aria-expanded="false">
    <div id="bs-drawer-inner">
      <div id="bs-drawer-top">
        <div id="bs-drawer-var"></div>
        <div id="bs-drawer-badges"></div>
        <button id="bs-drawer-close" aria-label="Close detail panel">&#x2715;</button>
      </div>
      <div id="bs-drawer-grid"></div>
    </div>
  </div>

  <script nonce="${nonce}">
  (function () {
    'use strict';

    /* ── Contractual node colours (test-asserted) ─────────────────────── */
    var C = {
      root_input:   '#f5c842',
      intermediate: '#f57c42',
      failure_site: '#e84040'   /* brightened from #e03030 for 5.2:1 contrast on #180606 */
    };
    var STROKE = {
      root_input:   '#c9a21a',
      intermediate: '#c95f20',
      failure_site: '#a51f1f'
    };
    var FILL_BG = {
      root_input:   '#1e1600',
      intermediate: '#1e0c00',
      failure_site: '#180606'
    };
    var GLOWS  = { root_input: '', intermediate: '', failure_site: 'gf-r' };
    var ARROWS = { root_input: 'ar-y', intermediate: 'ar-o', failure_site: 'ar-r' };

    var ROLE_LABEL = {
      root_input:   'Root Cause',
      intermediate: 'Propagation',
      failure_site: 'Failure Site'
    };

    var ROLE_BADGE = {
      root_input:   'bd-root',
      intermediate: 'bd-inter',
      failure_site: 'bd-site'
    };

    var COL_HDR = {
      root_input:   'Root Cause',
      intermediate: 'Propagation Path',
      failure_site: 'Failure Site'
    };

    var FT_LBL = {
      nan_inf:               'NaN / Inf',
      division_by_zero:      'Division by Zero',
      non_psd_matrix:        'Non-PSD Matrix',
      ill_conditioned_matrix:'Ill-Conditioned Matrix',
      bounds_exceeded:       'Bounds Exceeded',
      division_instability:  'Division Instability',
      exploding_gradient:    'Exploding Gradient',
      regime_shift:          'Regime Shift',
      logical_invariant:     'Logical Invariant'
    };

    /* ── Layout constants ─────────────────────────────────────────────── */
    var COL_X    = { root_input: 110, intermediate: 330, failure_site: 550 };
    var NODE_W   = 150, NODE_H = 58, NODE_RX = 9;
    var ROW_H    = 114, SVG_W = 695, SVG_MIN_H = 260;
    var COL_HDR_Y = 22, NODES_Y = 62;

    /* ── State ────────────────────────────────────────────────────────── */
    var vscode  = acquireVsCodeApi();
    var current = null;
    var selId   = null;
    var scale   = 1.0;
    var STEP = 0.15, SMIN = 0.3, SMAX = 2.5;

    /* ── SVG helpers ──────────────────────────────────────────────────── */
    var NS = 'http://www.w3.org/2000/svg';

    function svgEl(tag, attrs) {
      var e = document.createElementNS(NS, tag);
      for (var k in (attrs || {})) {
        if (Object.prototype.hasOwnProperty.call(attrs, k)) e.setAttribute(k, String(attrs[k]));
      }
      return e;
    }

    function svgTxt(tag, attrs, text) {
      var e = svgEl(tag, attrs);
      e.textContent = text;
      return e;
    }

    /* ── Layout ───────────────────────────────────────────────────────── */
    function computeLayout(nodes) {
      var by = { root_input: [], intermediate: [], failure_site: [] };
      nodes.forEach(function (n) { if (by[n.role]) by[n.role].push(n); });

      var maxC = Math.max(by.root_input.length, by.intermediate.length, by.failure_site.length, 1);
      var svgH = Math.max(maxC * ROW_H + NODES_Y + ROW_H / 2, SVG_MIN_H);
      var pos  = {};

      ['root_input', 'intermediate', 'failure_site'].forEach(function (role) {
        var col   = by[role];
        var total = col.length * ROW_H;
        var avail = svgH - NODES_Y;
        var start = NODES_Y + (avail - total) / 2 + ROW_H / 2;
        col.forEach(function (n, i) {
          pos[n.id] = { x: COL_X[role], y: start + i * ROW_H };
        });
      });

      return { pos: pos, svgH: svgH };
    }

    /* ── Main render ──────────────────────────────────────────────────── */
    function render(data) {
      current = data;
      selId   = null;

      updateHeader(data);
      updateKPIs(data);

      var hasNodes = data && data.nodes && data.nodes.length > 0;
      var noFails  = data && data.summary && data.summary.total_failures === 0;

      var stateEl  = document.getElementById('bs-state');
      var graphEl  = document.getElementById('bs-graph-scroll');

      if (hasNodes) {
        stateEl.className = '';
        graphEl.className = 'show';
        renderGraph(data);
      } else if (noFails) {
        graphEl.className = '';
        showSuccess();
      } else {
        graphEl.className = '';
        showWaiting();
      }
    }

    function showWaiting() {
      var s = document.getElementById('bs-state');
      s.className = 'show';
      document.getElementById('bs-state-svg').innerHTML =
        '<path d="M12 2L21 6.8V17.2L12 22L3 17.2V6.8L12 2Z" stroke="currentColor" stroke-width="1.2"/>' +
        '<path d="M9.5 6L11.5 11L9.5 13L13.5 18" stroke="currentColor" stroke-width="1.6"' +
        ' stroke-linecap="round" stroke-linejoin="round"/>';
      document.getElementById('bs-state-title').textContent = 'Waiting for results';
      document.getElementById('bs-state-sub').textContent =
        'Click \u25b6 Run BlackSwan above a function to start a stress test.';
    }

    function showSuccess() {
      var s = document.getElementById('bs-state');
      s.className = 'show success';
      document.getElementById('bs-state-svg').innerHTML =
        '<path d="M12 2L21 6.8V17.2L12 22L3 17.2V6.8L12 2Z" stroke="#3fb950" stroke-width="1.2" fill="rgba(63,185,80,0.08)"/>' +
        '<path d="M8.5 12L11 14.5L15.5 9.5" stroke="#3fb950" stroke-width="1.6"' +
        ' stroke-linecap="round" stroke-linejoin="round"/>';
      document.getElementById('bs-state-title').textContent = 'No failures detected';
      document.getElementById('bs-state-sub').textContent =
        'Your model survived all iterations of this stress scenario. ' +
        'Try a more severe scenario or a different function.';
    }

    /* ── Graph rendering ──────────────────────────────────────────────── */
    function renderGraph(data) {
      var svg  = document.getElementById('bs-dag');
      var defs = svg.querySelector('defs');

      while (svg.firstChild) svg.removeChild(svg.firstChild);
      if (defs) svg.appendChild(defs);
      svg.classList.remove('has-sel');

      var lay  = computeLayout(data.nodes);
      var pos  = lay.pos;
      var svgH = lay.svgH;

      svg.setAttribute('viewBox', '0 0 ' + SVG_W + ' ' + svgH);
      svg.setAttribute('width',   String(SVG_W));
      svg.setAttribute('height',  String(svgH));

      /* Column headers */
      ['root_input', 'intermediate', 'failure_site'].forEach(function (role) {
        svg.appendChild(svgTxt('text', {
          'class': 'bs-col-hdr',
          x: String(COL_X[role]),
          y: String(COL_HDR_Y),
          fill: C[role] + '70'
        }, COL_HDR[role]));
      });

      /* Column separator lines */
      var sx1 = (COL_X.root_input + COL_X.intermediate) / 2;
      var sx2 = (COL_X.intermediate + COL_X.failure_site) / 2;
      [sx1, sx2].forEach(function (x) {
        svg.appendChild(svgEl('line', {
          'class': 'bs-sep',
          stroke:  'var(--bs-border)',
          'stroke-width': '1',
          x1: String(x), y1: String(NODES_Y - 14),
          x2: String(x), y2: String(svgH - 10)
        }));
      });

      /* Edges (rendered behind nodes) */
      data.edges.forEach(function (e) {
        var s = pos[e.source], t = pos[e.target];
        if (!s || !t) return;

        var tn    = null;
        for (var i = 0; i < data.nodes.length; i++) {
          if (data.nodes[i].id === e.target) { tn = data.nodes[i]; break; }
        }
        var role  = tn ? tn.role : 'intermediate';
        var color = C[role];
        var arrow = ARROWS[role];

        var sx  = s.x + NODE_W / 2;
        var sy  = s.y;
        var tx  = t.x - NODE_W / 2 - 5;
        var ty  = t.y;
        var cpx = (sx + tx) / 2;

        svg.appendChild(svgEl('path', {
          d: 'M ' + sx + ' ' + sy +
             ' C ' + cpx + ' ' + sy +
             ', ' + cpx + ' ' + ty +
             ', ' + tx + ' ' + ty,
          'class':       'bs-edge',
          stroke:        color,
          opacity:       '0.48',
          'marker-end':  'url(#' + arrow + ')',
          'data-source': e.source,
          'data-target': e.target
        }));
      });

      /* Nodes */
      data.nodes.forEach(function (n) {
        var p = pos[n.id];
        if (!p) return;

        var color  = C[n.role]      || '#888';
        var stroke = STROKE[n.role] || '#555';
        var bg     = FILL_BG[n.role]|| '#111';
        var glow   = GLOWS[n.role]  || '';
        var isSite = n.role === 'failure_site';
        var hw = NODE_W / 2, hh = NODE_H / 2;

        var g = svgEl('g', {
          'class':      'bs-node',
          transform:    'translate(' + p.x + ',' + p.y + ')',
          tabindex:     '0',
          role:         'button',
          'data-id':    n.id,
          'aria-label': n.variable + ', line ' + n.line + ', ' + (ROLE_LABEL[n.role] || n.role)
        });

        /* Animated pulse ring for failure_site nodes only */
        if (isSite) {
          g.appendChild(svgEl('circle', {
            cx: '0', cy: '0', r: '10',
            fill: 'none',
            stroke: color,
            'stroke-width': '2',
            'class': 'bs-pulse',
            opacity: '0.3'
          }));
        }

        /* Dark background fill */
        g.appendChild(svgEl('rect', {
          x: String(-hw), y: String(-hh),
          width:  String(NODE_W),
          height: String(NODE_H),
          rx: String(NODE_RX),
          fill: bg,
          'class': 'bs-node-bg'
        }));

        /* Glow halo — failure_site only (reserved; signals where math breaks) */
        if (glow) {
          g.appendChild(svgEl('rect', {
            x: String(-hw), y: String(-hh),
            width:  String(NODE_W),
            height: String(NODE_H),
            rx: String(NODE_RX),
            fill: 'none',
            stroke: color,
            'stroke-width': '1.5',
            filter: 'url(#' + glow + ')',
            opacity: '0.65',
            'pointer-events': 'none'
          }));
        }

        /* Crisp visible border */
        g.appendChild(svgEl('rect', {
          x: String(-hw), y: String(-hh),
          width:  String(NODE_W),
          height: String(NODE_H),
          rx: String(NODE_RX),
          fill: 'none',
          stroke: stroke,
          'stroke-width': '1.5',
          'class': 'bs-node-border',
          'pointer-events': 'none'
        }));

        /* Variable name (truncated if needed) */
        var label = n.variable.length > 17
          ? n.variable.slice(0, 16) + '\u2026'
          : n.variable;

        g.appendChild(svgTxt('text', {
          'class': 'bs-node-var',
          x: '0', y: '-8',
          fill: color,
          'font-size': '11'
        }, label));

        /* Line number */
        g.appendChild(svgTxt('text', {
          'class': 'bs-node-lineno',
          x: '0', y: '11',
          fill: color,
          'font-size': '10'
        }, 'L\u2009' + n.line));

        /* Interaction */
        g.addEventListener('click', function () { handleSelect(n); });
        g.addEventListener('keydown', function (ev) {
          if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); handleSelect(n); }
        });

        svg.appendChild(g);
      });
    }

    /* ── Node selection & dimming ─────────────────────────────────────── */
    function handleSelect(node) {
      var svg = document.getElementById('bs-dag');
      if (selId === node.id) { deselect(svg); closeDrawer(); return; }
      deselect(svg);
      selId = node.id;

      var conn = new Set([node.id]);
      if (current && current.edges) {
        current.edges.forEach(function (e) {
          if (e.source === node.id) conn.add(e.target);
          if (e.target === node.id) conn.add(e.source);
        });
      }

      svg.querySelectorAll('.bs-node').forEach(function (el) {
        var id = el.getAttribute('data-id');
        if (id === node.id) el.classList.add('selected');
        else if (conn.has(id)) el.classList.add('connected');
      });

      svg.querySelectorAll('.bs-edge').forEach(function (el) {
        var src = el.getAttribute('data-source');
        var tgt = el.getAttribute('data-target');
        if (src === node.id || tgt === node.id) el.classList.add('edge-on');
      });

      svg.classList.add('has-sel');
      openDrawer(node);
    }

    function deselect(svg) {
      selId = null;
      svg.classList.remove('has-sel');
      svg.querySelectorAll('.bs-node').forEach(function (e) {
        e.classList.remove('selected', 'connected');
      });
      svg.querySelectorAll('.bs-edge').forEach(function (e) {
        e.classList.remove('edge-on');
      });
    }

    /* ── Bottom drawer ────────────────────────────────────────────────── */
    function openDrawer(node) {
      var drawer  = document.getElementById('bs-drawer');
      var detail  = current && current.nodeDetails && current.nodeDetails[node.id];

      /* Variable name */
      document.getElementById('bs-drawer-var').textContent = node.variable;

      /* Role + severity badges */
      var badges = document.getElementById('bs-drawer-badges');
      badges.innerHTML = '';
      makeBadge(badges, ROLE_LABEL[node.role] || node.role, ROLE_BADGE[node.role] || '');
      if (detail && detail.severity) {
        makeBadge(badges, detail.severity.toUpperCase(), 'bd-' + detail.severity);
      }

      /* Grid body */
      var grid = document.getElementById('bs-drawer-grid');
      grid.innerHTML = '';

      /* Source line goto */
      addField(grid, 'Source', makeGoto(node), false);

      if (detail) {
        if (detail.failure_type) {
          addField(grid, 'Type', makeVal(FT_LBL[detail.failure_type] || detail.failure_type), false);
        }
        if (detail.frequency) {
          addField(grid, 'Frequency', makeVal(detail.frequency), false);
        }
        if (detail.message) {
          addField(grid, 'Description', makeVal(detail.message), true);
        }
        if (detail.confidence) {
          addField(grid, 'Attribution', makeVal(cap(detail.confidence) + ' confidence'), false);
        }
        if (detail.fix_hint) {
          addField(grid, 'Fix Hint', makeFixBox(detail.fix_hint), true);
        }
      }

      drawer.classList.add('open');
      drawer.setAttribute('aria-expanded', 'true');
    }

    function closeDrawer() {
      var drawer = document.getElementById('bs-drawer');
      drawer.classList.remove('open');
      drawer.setAttribute('aria-expanded', 'false');
      var svg = document.getElementById('bs-dag');
      deselect(svg);
    }

    document.getElementById('bs-drawer-close').addEventListener('click', closeDrawer);

    /* ── Drawer field builders ────────────────────────────────────────── */
    function makeBadge(parent, text, cls) {
      var b = document.createElement('span');
      b.className = 'bs-badge ' + (cls || '');
      b.textContent = text;
      parent.appendChild(b);
    }

    function addField(grid, label, content, fullWidth) {
      var wrap = document.createElement('div');
      wrap.className = 'bs-field' + (fullWidth ? ' span3' : '');
      var lbl = document.createElement('div');
      lbl.className = 'bs-field-lbl';
      lbl.textContent = label;
      wrap.appendChild(lbl);
      wrap.appendChild(content);
      grid.appendChild(wrap);
    }

    function makeVal(str) {
      var d = document.createElement('div');
      d.className = 'bs-field-val';
      d.textContent = str;
      return d;
    }

    function makeGoto(node) {
      var btn = document.createElement('button');
      btn.className = 'bs-goto';
      btn.setAttribute('aria-label', 'Go to line ' + node.line);
      btn.innerHTML =
        '<svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden="true">' +
        '<path d="M1 9L9 1M9 1H3M9 1V7" stroke="currentColor" stroke-width="1.5"' +
        ' stroke-linecap="round" stroke-linejoin="round"/></svg>' +
        ' Line\u00a0' + node.line;
      btn.addEventListener('click', function () {
        vscode.postMessage({
          type: 'navigate',
          line: node.line,
          uri:  current ? current.documentUri : ''
        });
      });
      return btn;
    }

    function makeFixBox(hint) {
      var div = document.createElement('div');
      div.className = 'bs-fix-box';
      div.textContent = hint;
      var copy = document.createElement('button');
      copy.className = 'bs-copy';
      copy.textContent = 'Copy';
      copy.addEventListener('click', function () {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(hint).then(function () {
            copy.textContent = 'Copied!';
            setTimeout(function () { copy.textContent = 'Copy'; }, 1600);
          });
        }
      });
      div.appendChild(copy);
      return div;
    }

    /* ── Header & KPI updates ─────────────────────────────────────────── */
    function updateHeader(data) {
      var meta = document.getElementById('bs-meta-pills');
      meta.innerHTML = '';
      if (!data || !data.summary) return;
      var s = data.summary;
      [
        { lbl: 'Scenario', val: s.scenario_name.replace(/_/g, '\u00a0') },
        { lbl: 'Seed',     val: String(s.seed) },
        { lbl: 'Mode',     val: s.mode, cls: 'bs-pill-' + (s.mode || 'fast') }
      ].forEach(function (p) {
        var pill = document.createElement('div');
        pill.className = 'bs-pill ' + (p.cls || '');
        pill.innerHTML =
          '<span class="bs-pill-lbl">' + escH(p.lbl) + '</span>' +
          '<span class="bs-pill-val">' + escH(p.val) + '</span>';
        meta.appendChild(pill);
      });
    }

    function updateKPIs(data) {
      if (!data || !data.summary) {
        ['kpi-critical','kpi-warning','kpi-rate','kpi-iters'].forEach(function (id) { setT(id,'—'); });
        setT('kpi-runtime', '');
        document.getElementById('kpi-rate-bar-wrap').style.display = 'none';
        return;
      }
      var s = data.summary;
      setT('kpi-critical', String(s.critical_count));
      setT('kpi-warning',  String(s.warning_count));

      var pct = s.failure_rate * 100;
      setT('kpi-rate', pct.toFixed(1) + '%');
      document.getElementById('kpi-rate-bar-wrap').style.display = '';
      document.getElementById('kpi-rate-bar').style.width = Math.min(pct, 100) + '%';
      document.getElementById('kpi-rate-bar-wrap').setAttribute('aria-valuenow', String(Math.round(pct)));

      setT('kpi-iters',   s.iterations_completed.toLocaleString());
      setT('kpi-runtime', (s.runtime_ms / 1000).toFixed(2) + 's');
    }

    function setT(id, t) {
      var e = document.getElementById(id);
      if (e) e.textContent = t;
    }

    function escH(str) {
      return String(str)
        .replace(/&/g,'&amp;').replace(/</g,'&lt;')
        .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }

    function cap(str) {
      return str ? str.charAt(0).toUpperCase() + str.slice(1) : str;
    }

    /* ── Legend filter — click to toggle node-type visibility ────────── */
    document.getElementById('bs-legend').addEventListener('click', function (ev) {
      var item = ev.target.closest('.bs-leg-item');
      if (!item) return;
      var role   = item.getAttribute('data-role');
      var dimmed = item.classList.toggle('dimmed');
      item.setAttribute('aria-checked', String(!dimmed));

      var svg = document.getElementById('bs-dag');
      if (!svg) return;

      svg.querySelectorAll('.bs-node').forEach(function (nodeEl) {
        var id = nodeEl.getAttribute('data-id');
        var nd = current && current.nodes &&
                 current.nodes.find(function (x) { return x.id === id; });
        if (nd && nd.role === role) {
          nodeEl.style.opacity = dimmed ? '0.12' : '';
        }
      });
    });

    /* Keyboard activation for legend items */
    document.getElementById('bs-legend').addEventListener('keydown', function (ev) {
      var item = ev.target.closest('.bs-leg-item');
      if (!item) return;
      if (ev.key === 'Enter' || ev.key === ' ') {
        ev.preventDefault();
        item.click();
      }
    });

    /* ── Zoom ─────────────────────────────────────────────────────────── */
    function setZoom(v) {
      scale = Math.min(SMAX, Math.max(SMIN, v));
      document.getElementById('bs-zoom-wrapper').style.transform = 'scale(' + scale.toFixed(2) + ')';
      setT('bs-zoom-pct', Math.round(scale * 100) + '%');
    }

    document.getElementById('bs-zoom-in').addEventListener('click',    function () { setZoom(scale + STEP); });
    document.getElementById('bs-zoom-out').addEventListener('click',   function () { setZoom(scale - STEP); });
    document.getElementById('bs-zoom-reset').addEventListener('click', function () { setZoom(1.0); });

    /* ── Initial render ───────────────────────────────────────────────── */
    var initialData = ${dataJson};
    render(initialData);

    /* ── Live updates from extension host ────────────────────────────── */
    window.addEventListener('message', function (ev) {
      if (ev.data && ev.data.type === 'update') {
        closeDrawer();
        render(ev.data.data);
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
