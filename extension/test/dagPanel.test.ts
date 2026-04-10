/**
 * Unit tests for dagPanel.ts.
 *
 * Tests cover:
 *   buildDagData  — pure conversion of ShatterPoint[] to DagData
 *   buildDagHtml  — pure HTML generation from DagData
 *   DagPanelController — WebviewPanel lifecycle, message handling,
 *                        show/reveal/update, dispose
 *
 * The vscode module is resolved to __mocks__/vscode.ts by Jest — no live
 * extension host. FakeWebviewPanel.simulateMessage() drives the navigate path.
 */

import * as vscode from "vscode";
import { buildDagData, buildDagHtml, DagPanelController, DagData } from "../src/dagPanel";
import { ShatterPoint } from "../src/types";
import { FakeWebviewPanel } from "../__mocks__/vscode";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const DOC_URI = vscode.Uri.file("/workspace/risk_model.py");

function makeShatterPoint(overrides: Partial<ShatterPoint> = {}): ShatterPoint {
  return {
    id:           "sp_001",
    line:         82,
    column:       4,
    severity:     "critical",
    failure_type: "non_psd_matrix",
    message:      "Covariance matrix loses PSD",
    frequency:    "847 / 5000 iterations (16.9%)",
    causal_chain: [
      { line: 14, variable: "corr_shift",           role: "root_input"   },
      { line: 47, variable: "adjusted_corr_matrix", role: "intermediate" },
      { line: 82, variable: "cov_matrix",           role: "failure_site" },
    ],
    fix_hint:   "Apply Higham 2002 correction",
    confidence: "high",
    ...overrides,
  };
}

function makeResponse(shatterPoints: ShatterPoint[] = [makeShatterPoint()]): import("../src/types").BlackSwanResponse {
  return {
    version:              "1.0",
    status:               "failures_detected",
    mode:                 "full",
    runtime_ms:           200,
    iterations_completed: 5000,
    summary: { total_failures: shatterPoints.length, failure_rate: 0.17, unique_failure_types: 1 },
    shatter_points: shatterPoints,
    scenario_card:  { name: "liquidity_crash", parameters_applied: {}, seed: 42, reproducible: true },
    reproducibility_card: {
      blackswan_version:    "0.3.0",
      python_version:       "3.11.0",
      numpy_version:        "1.26.0",
      platform:             "linux",
      seed:                 42,
      scenario_name:        "liquidity_crash",
      scenario_hash:        "abc123def456",
      mode:                 "full",
      iterations_requested: 5000,
      iterations_executed:  5000,
      iterations_skipped:   0,
      budget_exhausted:     false,
      budget_reason:        null,
      timestamp_utc:        "2026-04-10T00:00:00Z",
      reproducible:         true,
      replay_command:       "python -m blackswan test model.py --scenario liquidity_crash --seed 42",
    },
    budget: { exhausted: false, reason: null },
  };
}

/** Helper to get the FakeWebviewPanel created by the last createWebviewPanel call. */
function getCreatedPanel(): FakeWebviewPanel {
  const mock = vscode.window.createWebviewPanel as jest.Mock;
  const result = mock.mock.results[mock.mock.results.length - 1];
  return result.value as FakeWebviewPanel;
}

// ---------------------------------------------------------------------------
// buildDagData — empty inputs
// ---------------------------------------------------------------------------

describe("buildDagData — empty / no shatter points", () => {
  test("empty shatter points → empty nodes array", () => {
    const data = buildDagData([], DOC_URI);
    expect(data.nodes).toHaveLength(0);
  });

  test("empty shatter points → empty edges array", () => {
    const data = buildDagData([], DOC_URI);
    expect(data.edges).toHaveLength(0);
  });

  test("documentUri.fsPath is stored in result", () => {
    const data = buildDagData([], DOC_URI);
    expect(data.documentUri).toBe(DOC_URI.fsPath);
  });
});

// ---------------------------------------------------------------------------
// buildDagData — single shatter point
// ---------------------------------------------------------------------------

describe("buildDagData — single shatter point", () => {
  test("3-link causal chain → 3 nodes", () => {
    const data = buildDagData([makeShatterPoint()], DOC_URI);
    expect(data.nodes).toHaveLength(3);
  });

  test("3-link causal chain → 2 edges", () => {
    const data = buildDagData([makeShatterPoint()], DOC_URI);
    expect(data.edges).toHaveLength(2);
  });

  test("single-link chain → 1 node, 0 edges", () => {
    const sp = makeShatterPoint({
      causal_chain: [{ line: 10, variable: "x", role: "failure_site" }],
    });
    const data = buildDagData([sp], DOC_URI);
    expect(data.nodes).toHaveLength(1);
    expect(data.edges).toHaveLength(0);
  });

  test("empty causal chain → no nodes from that shatter point", () => {
    const sp = makeShatterPoint({ causal_chain: [] });
    const data = buildDagData([sp], DOC_URI);
    expect(data.nodes).toHaveLength(0);
  });

  test("node IDs are derived from line and variable", () => {
    const data = buildDagData([makeShatterPoint()], DOC_URI);
    const ids  = data.nodes.map((n) => n.id);
    expect(ids).toContain("L14_corr_shift");
    expect(ids).toContain("L47_adjusted_corr_matrix");
    expect(ids).toContain("L82_cov_matrix");
  });

  test("node roles are preserved from the causal chain", () => {
    const data = buildDagData([makeShatterPoint()], DOC_URI);
    const byId = Object.fromEntries(data.nodes.map((n) => [n.id, n]));
    expect(byId["L14_corr_shift"].role).toBe("root_input");
    expect(byId["L47_adjusted_corr_matrix"].role).toBe("intermediate");
    expect(byId["L82_cov_matrix"].role).toBe("failure_site");
  });

  test("node variable names are preserved exactly", () => {
    const data = buildDagData([makeShatterPoint()], DOC_URI);
    const vars = data.nodes.map((n) => n.variable);
    expect(vars).toContain("corr_shift");
    expect(vars).toContain("adjusted_corr_matrix");
    expect(vars).toContain("cov_matrix");
  });

  test("node line numbers are preserved", () => {
    const data = buildDagData([makeShatterPoint()], DOC_URI);
    const lines = data.nodes.map((n) => n.line);
    expect(lines).toContain(14);
    expect(lines).toContain(47);
    expect(lines).toContain(82);
  });

  test("edge direction follows the causal chain order", () => {
    const data = buildDagData([makeShatterPoint()], DOC_URI);
    expect(data.edges[0]).toEqual({ source: "L14_corr_shift", target: "L47_adjusted_corr_matrix" });
    expect(data.edges[1]).toEqual({ source: "L47_adjusted_corr_matrix", target: "L82_cov_matrix" });
  });
});

// ---------------------------------------------------------------------------
// buildDagData — deduplication
// ---------------------------------------------------------------------------

describe("buildDagData — deduplication across shatter points", () => {
  test("shared node across two shatter points is deduplicated", () => {
    // Both chains share the same root input on line 14.
    const sp1 = makeShatterPoint({
      id: "sp_001",
      causal_chain: [
        { line: 14, variable: "corr_shift", role: "root_input" },
        { line: 82, variable: "cov_matrix", role: "failure_site" },
      ],
    });
    const sp2 = makeShatterPoint({
      id: "sp_002",
      causal_chain: [
        { line: 14, variable: "corr_shift", role: "root_input" },
        { line: 99, variable: "vol_matrix",  role: "failure_site" },
      ],
    });
    const data = buildDagData([sp1, sp2], DOC_URI);
    const nodeIds = data.nodes.map((n) => n.id);
    const corrShiftCount = nodeIds.filter((id) => id === "L14_corr_shift").length;
    expect(corrShiftCount).toBe(1);
    expect(data.nodes).toHaveLength(3); // corr_shift, cov_matrix, vol_matrix
  });

  test("shared edge across two chains is deduplicated", () => {
    const sp1 = makeShatterPoint({
      id: "sp_001",
      causal_chain: [
        { line: 14, variable: "corr_shift", role: "root_input" },
        { line: 82, variable: "cov_matrix", role: "failure_site" },
      ],
    });
    // Same chain again — edge should not be duplicated.
    const sp2 = makeShatterPoint({
      id: "sp_002",
      causal_chain: [
        { line: 14, variable: "corr_shift", role: "root_input" },
        { line: 82, variable: "cov_matrix", role: "failure_site" },
      ],
    });
    const data = buildDagData([sp1, sp2], DOC_URI);
    expect(data.edges).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// buildDagHtml — HTML structure
// ---------------------------------------------------------------------------

describe("buildDagHtml — HTML structure", () => {
  const NONCE = "testNonce12345678901234567890ab";
  let html: string;
  let data: DagData;

  beforeAll(() => {
    data = buildDagData([makeShatterPoint()], DOC_URI);
    html = buildDagHtml(data, NONCE);
  });

  test("returns a string beginning with '<!DOCTYPE html>'", () => {
    expect(html).toMatch(/^<!DOCTYPE html>/);
  });

  test("CSP meta tag includes the nonce", () => {
    expect(html).toContain(`nonce-${NONCE}`);
  });

  test("script tag carries the nonce attribute", () => {
    expect(html).toContain(`nonce="${NONCE}"`);
  });

  test("CSP does not allow unsafe-eval", () => {
    expect(html).not.toContain("unsafe-eval");
  });

  test("CSP default-src is 'none'", () => {
    expect(html).toContain("default-src 'none'");
  });

  test("contains 'acquireVsCodeApi'", () => {
    expect(html).toContain("acquireVsCodeApi");
  });

  test("contains 'navigate' message type for node click", () => {
    expect(html).toContain("navigate");
  });

  test("embeds DAG data as JSON in the script", () => {
    // Each variable name should appear in the JSON-embedded data.
    expect(html).toContain("corr_shift");
    expect(html).toContain("adjusted_corr_matrix");
    expect(html).toContain("cov_matrix");
  });

  test("contains role colours for root_input (yellow)", () => {
    expect(html).toContain("#f5c842");
  });

  test("contains role colour for intermediate (orange)", () => {
    expect(html).toContain("#f57c42");
  });

  test("contains role colour for failure_site (red)", () => {
    expect(html).toContain("#e03030");
  });
});

// ---------------------------------------------------------------------------
// buildDagHtml — safe JSON encoding (XSS prevention)
// ---------------------------------------------------------------------------

describe("buildDagHtml — safe JSON encoding", () => {
  test("variable names containing '<' are escaped in embedded JSON", () => {
    const sp = makeShatterPoint({
      causal_chain: [
        { line: 1, variable: "a<b", role: "root_input" },
      ],
    });
    const dagData = buildDagData([sp], DOC_URI);
    const html    = buildDagHtml(dagData, "nonce");
    // Raw '<' must not appear inside the embedded JSON (could break out of <script>).
    // The safe encoder replaces < with \u003c.
    expect(html).toContain("\\u003c");
    expect(html).not.toMatch(/a<b/);
  });

  test("variable names containing '>' are escaped in embedded JSON", () => {
    const sp = makeShatterPoint({
      causal_chain: [{ line: 1, variable: "x>y", role: "failure_site" }],
    });
    const dagData = buildDagData([sp], DOC_URI);
    const html    = buildDagHtml(dagData, "nonce");
    expect(html).toContain("\\u003e");
  });

  test("'</script>' cannot appear in the embedded JSON data", () => {
    const sp = makeShatterPoint({
      causal_chain: [
        { line: 1, variable: "</script><script>alert(1)", role: "root_input" },
      ],
    });
    const dagData = buildDagData([sp], DOC_URI);
    const html    = buildDagHtml(dagData, "nonce");
    expect(html).not.toContain("</script><script>");
  });
});

// ---------------------------------------------------------------------------
// DagPanelController — isVisible
// ---------------------------------------------------------------------------

describe("DagPanelController.isVisible", () => {
  beforeEach(() => jest.clearAllMocks());

  test("isVisible is false before show() is called", () => {
    const ctrl = new DagPanelController();
    expect(ctrl.isVisible).toBe(false);
    ctrl.dispose();
  });

  test("isVisible is true after show() is called", () => {
    const ctrl = new DagPanelController();
    ctrl.show(makeResponse(), DOC_URI);
    expect(ctrl.isVisible).toBe(true);
    ctrl.dispose();
  });

  test("isVisible is false after dispose()", () => {
    const ctrl = new DagPanelController();
    ctrl.show(makeResponse(), DOC_URI);
    ctrl.dispose();
    expect(ctrl.isVisible).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// DagPanelController — first show()
// ---------------------------------------------------------------------------

describe("DagPanelController.show — first call", () => {
  beforeEach(() => jest.clearAllMocks());

  test("calls createWebviewPanel with the correct viewType", () => {
    const ctrl = new DagPanelController();
    ctrl.show(makeResponse(), DOC_URI);
    expect(vscode.window.createWebviewPanel).toHaveBeenCalledWith(
      "blackswan.dagPanel",
      expect.any(String),
      expect.anything(),
      expect.anything(),
    );
    ctrl.dispose();
  });

  test("panel title contains 'BlackSwan'", () => {
    const ctrl = new DagPanelController();
    ctrl.show(makeResponse(), DOC_URI);
    const [, title] = (vscode.window.createWebviewPanel as jest.Mock).mock.calls[0];
    expect(title).toContain("BlackSwan");
    ctrl.dispose();
  });

  test("sets webview.html after creating the panel", () => {
    const ctrl  = new DagPanelController();
    ctrl.show(makeResponse(), DOC_URI);
    const panel = getCreatedPanel();
    expect(panel.webview.html).toMatch(/<!DOCTYPE html>/);
    ctrl.dispose();
  });

  test("registers a message handler on the webview", () => {
    const ctrl  = new DagPanelController();
    ctrl.show(makeResponse(), DOC_URI);
    const panel = getCreatedPanel();
    expect(panel.webview.onDidReceiveMessage).toHaveBeenCalled();
    ctrl.dispose();
  });

  test("registers an onDidDispose handler", () => {
    const ctrl  = new DagPanelController();
    ctrl.show(makeResponse(), DOC_URI);
    const panel = getCreatedPanel();
    expect(panel.onDidDispose).toHaveBeenCalled();
    ctrl.dispose();
  });

  test("does NOT call postMessage on first show", () => {
    const ctrl  = new DagPanelController();
    ctrl.show(makeResponse(), DOC_URI);
    const panel = getCreatedPanel();
    expect(panel.webview.postMessage).not.toHaveBeenCalled();
    ctrl.dispose();
  });
});

// ---------------------------------------------------------------------------
// DagPanelController — subsequent show() calls
// ---------------------------------------------------------------------------

describe("DagPanelController.show — subsequent calls", () => {
  beforeEach(() => jest.clearAllMocks());

  test("second call does NOT create a new panel", () => {
    const ctrl = new DagPanelController();
    ctrl.show(makeResponse(), DOC_URI);
    ctrl.show(makeResponse(), DOC_URI);
    expect(vscode.window.createWebviewPanel).toHaveBeenCalledTimes(1);
    ctrl.dispose();
  });

  test("second call reveals the existing panel", () => {
    const ctrl  = new DagPanelController();
    ctrl.show(makeResponse(), DOC_URI);
    const panel = getCreatedPanel();
    ctrl.show(makeResponse(), DOC_URI);
    expect(panel.reveal).toHaveBeenCalled();
    ctrl.dispose();
  });

  test("second call posts an 'update' message to the webview", () => {
    const ctrl  = new DagPanelController();
    ctrl.show(makeResponse(), DOC_URI);
    const panel = getCreatedPanel();
    ctrl.show(makeResponse(), DOC_URI);
    expect(panel.webview.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: "update" }),
    );
    ctrl.dispose();
  });

  test("update message data contains nodes", () => {
    const ctrl  = new DagPanelController();
    ctrl.show(makeResponse(), DOC_URI);
    const panel = getCreatedPanel();
    ctrl.show(makeResponse(), DOC_URI);
    const [msg] = (panel.webview.postMessage as jest.Mock).mock.calls[0];
    expect(msg.data.nodes).toBeDefined();
    expect(Array.isArray(msg.data.nodes)).toBe(true);
    ctrl.dispose();
  });
});

// ---------------------------------------------------------------------------
// DagPanelController — navigate message handling
// ---------------------------------------------------------------------------

describe("DagPanelController — navigate message from webview", () => {
  beforeEach(() => jest.clearAllMocks());

  test("navigate message triggers vscode.commands.executeCommand('vscode.open')", () => {
    const ctrl  = new DagPanelController();
    ctrl.show(makeResponse(), DOC_URI);
    const panel = getCreatedPanel();

    panel.simulateMessage({ type: "navigate", line: 82, uri: "/workspace/risk_model.py" });

    expect(vscode.commands.executeCommand).toHaveBeenCalledWith(
      "vscode.open",
      expect.anything(),
      expect.anything(),
    );
    ctrl.dispose();
  });

  test("navigate message converts 1-indexed line to 0-indexed Range", () => {
    const ctrl  = new DagPanelController();
    ctrl.show(makeResponse(), DOC_URI);
    const panel = getCreatedPanel();

    panel.simulateMessage({ type: "navigate", line: 14, uri: "/workspace/risk_model.py" });

    const [, , opts] = (vscode.commands.executeCommand as jest.Mock).mock.calls[0];
    // line 14 (1-indexed) → 13 (0-indexed)
    expect(opts.selection.start.line).toBe(13);
    ctrl.dispose();
  });

  test("navigate line 1 converts to Range(0, 0, 0, 0) — no negative lines", () => {
    const ctrl  = new DagPanelController();
    ctrl.show(makeResponse(), DOC_URI);
    const panel = getCreatedPanel();

    panel.simulateMessage({ type: "navigate", line: 1, uri: "/workspace/risk_model.py" });

    const [, , opts] = (vscode.commands.executeCommand as jest.Mock).mock.calls[0];
    expect(opts.selection.start.line).toBe(0);
    ctrl.dispose();
  });

  test("unknown message type is silently ignored", () => {
    const ctrl  = new DagPanelController();
    ctrl.show(makeResponse(), DOC_URI);
    const panel = getCreatedPanel();

    expect(() => panel.simulateMessage({ type: "unknown" })).not.toThrow();
    expect(vscode.commands.executeCommand).not.toHaveBeenCalled();
    ctrl.dispose();
  });
});

// ---------------------------------------------------------------------------
// DagPanelController — panel dispose lifecycle
// ---------------------------------------------------------------------------

describe("DagPanelController — panel disposal", () => {
  beforeEach(() => jest.clearAllMocks());

  test("when panel is closed by user, isVisible becomes false", () => {
    const ctrl  = new DagPanelController();
    ctrl.show(makeResponse(), DOC_URI);
    const panel = getCreatedPanel();
    // Simulate user closing the panel tab — fires onDidDispose handlers.
    panel.dispose();
    expect(ctrl.isVisible).toBe(false);
    ctrl.dispose();
  });

  test("after user closes panel, next show() creates a new panel", () => {
    const ctrl  = new DagPanelController();
    ctrl.show(makeResponse(), DOC_URI);
    const panel = getCreatedPanel();
    panel.dispose();

    ctrl.show(makeResponse(), DOC_URI);
    expect(vscode.window.createWebviewPanel).toHaveBeenCalledTimes(2);
    ctrl.dispose();
  });

  test("dispose() on an idle controller does not throw", () => {
    const ctrl = new DagPanelController();
    expect(() => ctrl.dispose()).not.toThrow();
  });

  test("dispose() calls the panel's dispose method", () => {
    const ctrl  = new DagPanelController();
    ctrl.show(makeResponse(), DOC_URI);
    const panel = getCreatedPanel();
    ctrl.dispose();
    expect(panel.dispose).toHaveBeenCalled();
  });
});
