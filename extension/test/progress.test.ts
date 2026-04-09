/**
 * Unit tests for progress.ts.
 *
 * Tests cover:
 *   - SCENARIO_LABELS     — all 5 presets are present and correctly labelled
 *   - formatProgressTitle — notification title format
 *   - formatProgressMessage — iteration count message format
 *   - ProgressSession     — constructor initial report, reportIterations
 *                           increment calculation, edge cases
 *
 * No VS Code dependency — ProgressSession accepts a plain ProgressReporter
 * stub so these run in a plain Node.js Jest environment.
 */

import {
  SCENARIO_LABELS,
  formatProgressTitle,
  formatProgressMessage,
  ProgressSession,
  ProgressReporter,
} from "../src/progress";

// ---------------------------------------------------------------------------
// Test stub for the VS Code Progress object.
// ---------------------------------------------------------------------------

interface ProgressCall {
  message?: string;
  increment?: number;
}

function makeReporter(): ProgressReporter & { calls: ProgressCall[] } {
  const calls: ProgressCall[] = [];
  return {
    calls,
    report(value: ProgressCall) { calls.push({ ...value }); },
  };
}

// ---------------------------------------------------------------------------
// SCENARIO_LABELS
// ---------------------------------------------------------------------------

describe("SCENARIO_LABELS", () => {
  test("contains all 5 preset scenario keys", () => {
    const keys = Object.keys(SCENARIO_LABELS);
    expect(keys).toContain("liquidity_crash");
    expect(keys).toContain("vol_spike");
    expect(keys).toContain("correlation_breakdown");
    expect(keys).toContain("rate_shock");
    expect(keys).toContain("missing_data");
  });

  test("liquidity_crash maps to 'Liquidity Crash'", () => {
    expect(SCENARIO_LABELS["liquidity_crash"]).toBe("Liquidity Crash");
  });

  test("vol_spike maps to 'Vol Spike'", () => {
    expect(SCENARIO_LABELS["vol_spike"]).toBe("Vol Spike");
  });

  test("correlation_breakdown maps to 'Correlation Breakdown'", () => {
    expect(SCENARIO_LABELS["correlation_breakdown"]).toBe("Correlation Breakdown");
  });

  test("rate_shock maps to 'Rate Shock'", () => {
    expect(SCENARIO_LABELS["rate_shock"]).toBe("Rate Shock");
  });

  test("missing_data maps to 'Missing Data'", () => {
    expect(SCENARIO_LABELS["missing_data"]).toBe("Missing Data");
  });
});

// ---------------------------------------------------------------------------
// formatProgressTitle
// ---------------------------------------------------------------------------

describe("formatProgressTitle", () => {
  test("includes 'BlackSwan:' prefix", () => {
    const title = formatProgressTitle("model.py", "Liquidity Crash");
    expect(title).toMatch(/^BlackSwan:/);
  });

  test("wraps the file name in double quotes", () => {
    const title = formatProgressTitle("risk_model.py", "Vol Spike");
    expect(title).toContain('"risk_model.py"');
  });

  test("wraps the scenario label in square brackets", () => {
    const title = formatProgressTitle("model.py", "Liquidity Crash");
    expect(title).toContain("[Liquidity Crash]");
  });

  test("produces the exact expected format", () => {
    const title = formatProgressTitle("risk_model.py", "Liquidity Crash");
    expect(title).toBe('BlackSwan: Stress testing "risk_model.py" [Liquidity Crash]');
  });

  test("uses the scenario label string verbatim (no lookup)", () => {
    // formatProgressTitle takes the already-resolved label, not the key.
    const title = formatProgressTitle("model.py", "Unknown Custom Scenario");
    expect(title).toContain("Unknown Custom Scenario");
  });

  test("handles file names with spaces", () => {
    const title = formatProgressTitle("my risk model.py", "Rate Shock");
    expect(title).toContain('"my risk model.py"');
  });
});

// ---------------------------------------------------------------------------
// formatProgressMessage
// ---------------------------------------------------------------------------

describe("formatProgressMessage", () => {
  test("includes the total iteration count", () => {
    const msg = formatProgressMessage(0, 5000);
    expect(msg).toContain("5,000");
  });

  test("includes the completed iteration count", () => {
    const msg = formatProgressMessage(2340, 5000);
    expect(msg).toContain("2,340");
  });

  test("includes the calculated percentage", () => {
    const msg = formatProgressMessage(2500, 5000);
    expect(msg).toContain("50%");
  });

  test("rounds percentage to nearest integer", () => {
    // 1/3 = 33.33…% → rounds to 33%
    const msg = formatProgressMessage(1, 3);
    expect(msg).toContain("33%");
  });

  test("reports 0% when completed is 0", () => {
    const msg = formatProgressMessage(0, 5000);
    expect(msg).toContain("0%");
  });

  test("reports 100% when completed equals total", () => {
    const msg = formatProgressMessage(5000, 5000);
    expect(msg).toContain("100%");
  });

  test("does not throw when total is 0 (division by zero guard)", () => {
    expect(() => formatProgressMessage(0, 0)).not.toThrow();
  });

  test("shows 0% when total is 0", () => {
    const msg = formatProgressMessage(0, 0);
    expect(msg).toContain("0%");
  });

  test("contains 'Running' and 'complete' keywords", () => {
    const msg = formatProgressMessage(100, 200);
    expect(msg).toMatch(/Running/);
    expect(msg).toMatch(/complete/);
  });
});

// ---------------------------------------------------------------------------
// ProgressSession — constructor
// ---------------------------------------------------------------------------

describe("ProgressSession — constructor", () => {
  test("calls reporter.report once on construction", () => {
    const reporter = makeReporter();
    new ProgressSession(reporter, 5000);
    expect(reporter.calls).toHaveLength(1);
  });

  test("initial report message contains the total iteration count", () => {
    const reporter = makeReporter();
    new ProgressSession(reporter, 5000);
    expect(reporter.calls[0].message).toContain("5,000");
  });

  test("initial report message contains 'Running'", () => {
    const reporter = makeReporter();
    new ProgressSession(reporter, 5000);
    expect(reporter.calls[0].message).toMatch(/Running/);
  });

  test("initial report has no increment (indeterminate spinner)", () => {
    const reporter = makeReporter();
    new ProgressSession(reporter, 5000);
    // No increment = VS Code shows a spinner, not a percentage bar.
    expect(reporter.calls[0].increment).toBeUndefined();
  });

  test("exposes defaultTotal as a readonly property", () => {
    const reporter = makeReporter();
    const session = new ProgressSession(reporter, 1234);
    expect(session.defaultTotal).toBe(1234);
  });
});

// ---------------------------------------------------------------------------
// ProgressSession — reportIterations
// ---------------------------------------------------------------------------

describe("ProgressSession.reportIterations", () => {
  test("calls reporter.report with the formatted message", () => {
    const reporter = makeReporter();
    const session = new ProgressSession(reporter, 5000);
    session.reportIterations(2500, 5000);

    const lastCall = reporter.calls[reporter.calls.length - 1];
    expect(lastCall.message).toContain("2,500");
    expect(lastCall.message).toContain("5,000");
    expect(lastCall.message).toContain("50%");
  });

  test("provides a positive increment when percentage advances", () => {
    const reporter = makeReporter();
    const session = new ProgressSession(reporter, 5000);
    session.reportIterations(2500, 5000); // 50%

    const lastCall = reporter.calls[reporter.calls.length - 1];
    expect(lastCall.increment).toBe(50);
  });

  test("increment is the DELTA since the previous call, not absolute %", () => {
    const reporter = makeReporter();
    const session = new ProgressSession(reporter, 100);

    session.reportIterations(50, 100);  // 50% — increment should be 50
    session.reportIterations(75, 100);  // 75% — increment should be 25

    const calls = reporter.calls.slice(1); // skip constructor call
    expect(calls[0].increment).toBe(50);
    expect(calls[1].increment).toBe(25);
  });

  test("increment is undefined (not 0) when percentage does not advance", () => {
    const reporter = makeReporter();
    const session = new ProgressSession(reporter, 100);

    session.reportIterations(33, 100); // 33%
    session.reportIterations(33, 100); // still 33% — no change

    const lastCall = reporter.calls[reporter.calls.length - 1];
    // Passing increment: 0 to VS Code is noisy — we pass undefined instead.
    expect(lastCall.increment).toBeUndefined();
  });

  test("reports 100% correctly when completed equals total", () => {
    const reporter = makeReporter();
    const session = new ProgressSession(reporter, 5000);
    session.reportIterations(5000, 5000);

    const lastCall = reporter.calls[reporter.calls.length - 1];
    expect(lastCall.message).toContain("100%");
    expect(lastCall.increment).toBe(100); // jumped from 0% to 100%
  });

  test("does not throw when total is 0", () => {
    const reporter = makeReporter();
    const session = new ProgressSession(reporter, 0);
    expect(() => session.reportIterations(0, 0)).not.toThrow();
  });

  test("message matches formatProgressMessage output", () => {
    const reporter = makeReporter();
    const session = new ProgressSession(reporter, 5000);
    session.reportIterations(1000, 5000);

    const lastCall = reporter.calls[reporter.calls.length - 1];
    expect(lastCall.message).toBe(formatProgressMessage(1000, 5000));
  });

  test("three sequential calls accumulate the percentage correctly", () => {
    const reporter = makeReporter();
    const session = new ProgressSession(reporter, 100);

    session.reportIterations(25, 100);  // 25%
    session.reportIterations(50, 100);  // 50%
    session.reportIterations(100, 100); // 100%

    const calls = reporter.calls.slice(1); // skip constructor
    expect(calls[0].increment).toBe(25);
    expect(calls[1].increment).toBe(25);
    expect(calls[2].increment).toBe(50);
  });
});
