/**
 * Unit tests for orchestrator.ts.
 *
 * Strategy: mock `../src/bridge` so no Python process is spawned, and rely on
 * the vscode mock's withProgress that actually executes the task callback so
 * orchestrator logic runs inside tests.
 *
 * Covers:
 *   - isRunning state tracking
 *   - Double-trigger mutex guard
 *   - Successful run: no failures → showInformationMessage
 *   - Successful run: failures found → showWarningMessage
 *   - Error routing: EngineRuntimeError / EngineFrameworkError / EngineProtocolError
 *   - Cancellation: silent, clears diagnostics
 *   - cancel() aborts the in-progress run
 *   - Python path resolution priority
 */

import * as vscode from "vscode";
import { Orchestrator } from "../src/orchestrator";
import {
  EngineFrameworkError,
  EngineProtocolError,
  EngineRuntimeError,
} from "../src/types";

// ---------------------------------------------------------------------------
// Mock the bridge module — no Python processes in unit tests.
// ---------------------------------------------------------------------------

jest.mock("../src/bridge");
import { runBlackSwanEngine } from "../src/bridge";
const mockRun = jest.mocked(runBlackSwanEngine);

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const FILE_PATH = "/workspace/risk_model.py";
const FILE_URI  = vscode.Uri.file(FILE_PATH);

function mockDocument(fsPath = FILE_PATH): vscode.TextDocument {
  return {
    uri:         vscode.Uri.file(fsPath),
    languageId:  "python",
    lineCount:   100,
    lineAt:      jest.fn((i: number) => ({ text: "" })),
  } as unknown as vscode.TextDocument;
}

function makeResponse(totalFailures = 0) {
  return {
    version:              "1.0",
    status:               totalFailures === 0 ? ("no_failures" as const) : ("failures_detected" as const),
    runtime_ms:           200,
    iterations_completed: 50,
    summary: {
      total_failures:      totalFailures,
      failure_rate:        totalFailures / 50,
      unique_failure_types: totalFailures > 0 ? 1 : 0,
    },
    shatter_points: [],
    scenario_card: {
      name:                "liquidity_crash",
      parameters_applied:  {},
      seed:                42,
      reproducible:        true,
    },
  };
}

// Minimal DiagnosticsController mock — we only care that apply() and clear()
// are called correctly.
function mockDiagnostics() {
  return {
    apply: jest.fn(),
    clear: jest.fn(),
    clearAll: jest.fn(),
    dispose: jest.fn(),
  } as unknown as import("../src/diagnostics").DiagnosticsController;
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

beforeEach(() => {
  jest.clearAllMocks();
  // Default: withProgress executes the task callback (from vscode mock).
  // Default: runBlackSwanEngine returns a clean response.
  mockRun.mockResolvedValue(makeResponse(0));
});

// ---------------------------------------------------------------------------
// isRunning
// ---------------------------------------------------------------------------

describe("Orchestrator.isRunning", () => {
  test("returns false before any run", () => {
    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);
    expect(orch.isRunning(FILE_URI)).toBe(false);
    orch.dispose();
  });

  test("returns false after a run completes", async () => {
    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);
    await orch.run(mockDocument(), "my_func", "liquidity_crash");
    expect(orch.isRunning(FILE_URI)).toBe(false);
    orch.dispose();
  });

  test("returns false after a run that throws an error", async () => {
    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);
    mockRun.mockRejectedValueOnce(new EngineFrameworkError("engine error"));
    await orch.run(mockDocument(), "my_func", "liquidity_crash");
    expect(orch.isRunning(FILE_URI)).toBe(false);
    orch.dispose();
  });
});

// ---------------------------------------------------------------------------
// Double-trigger guard
// ---------------------------------------------------------------------------

describe("Orchestrator — double-trigger mutex", () => {
  test("second call while first is running returns false", async () => {
    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);

    // Make withProgress hang until we resolve it manually.
    let releaseFirst!: () => void;
    const firstCompleted = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });

    (vscode.window.withProgress as jest.Mock).mockImplementationOnce(
      async (
        _opts: unknown,
        task: (p: unknown, t: unknown) => Promise<unknown>,
      ) => {
        // Start task but never await it from withProgress's perspective —
        // the task runs but withProgress resolves only when we call releaseFirst.
        await firstCompleted;
      },
    );

    const doc = mockDocument();

    // Start first run without awaiting.
    const firstRun = orch.run(doc, "func", "liquidity_crash");

    // Yield one microtask tick so the mutex entry is set.
    await Promise.resolve();

    // Second run should be rejected immediately.
    const secondResult = await orch.run(doc, "func", "liquidity_crash");
    expect(secondResult).toBe(false);
    expect(vscode.window.showWarningMessage).toHaveBeenCalledWith(
      expect.stringContaining("already running"),
    );

    // Let first run finish so we don't leave dangling promises.
    releaseFirst();
    await firstRun;
    orch.dispose();
  });

  test("second call on a DIFFERENT file is allowed (mutex is per-file)", async () => {
    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);

    let releaseFirst!: () => void;
    const firstCompleted = new Promise<void>((r) => { releaseFirst = r; });

    (vscode.window.withProgress as jest.Mock).mockImplementationOnce(
      async () => { await firstCompleted; },
    );

    const docA = mockDocument("/workspace/model_a.py");
    const docB = mockDocument("/workspace/model_b.py");

    // First run on docA (pending).
    const firstRun = orch.run(docA, "func", "liquidity_crash");
    await Promise.resolve();

    // Second run on docB — different key, should proceed.
    const secondResult = await orch.run(docB, "func", "liquidity_crash");
    expect(secondResult).toBe(true);
    expect(vscode.window.showWarningMessage).not.toHaveBeenCalledWith(
      expect.stringContaining("already running"),
    );

    releaseFirst();
    await firstRun;
    orch.dispose();
  });
});

// ---------------------------------------------------------------------------
// Successful run — result dispatch
// ---------------------------------------------------------------------------

describe("Orchestrator.run — successful runs", () => {
  test("calls runBlackSwanEngine with correct file path and scenario", async () => {
    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);
    const doc  = mockDocument(FILE_PATH);

    await orch.run(doc, "calculate_var", "vol_spike");

    expect(mockRun).toHaveBeenCalledWith(
      FILE_PATH,
      "vol_spike",
      expect.objectContaining({ functionName: "calculate_var" }),
    );
    orch.dispose();
  });

  test("calls diagnostics.apply with the document and response on success", async () => {
    const diag     = mockDiagnostics();
    const orch     = new Orchestrator(diag);
    const doc      = mockDocument();
    const response = makeResponse(0);
    mockRun.mockResolvedValueOnce(response);

    await orch.run(doc, "f", "liquidity_crash");

    expect(diag.apply).toHaveBeenCalledWith(doc, response);
    orch.dispose();
  });

  test("shows informationMessage when no failures are detected", async () => {
    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);
    mockRun.mockResolvedValueOnce(makeResponse(0));

    await orch.run(mockDocument(), "f", "liquidity_crash");

    expect(vscode.window.showInformationMessage).toHaveBeenCalledWith(
      expect.stringContaining("No failures detected"),
    );
    orch.dispose();
  });

  test("shows warningMessage when failures are detected", async () => {
    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);
    mockRun.mockResolvedValueOnce(makeResponse(5));

    await orch.run(mockDocument(), "f", "liquidity_crash");

    expect(vscode.window.showWarningMessage).toHaveBeenCalledWith(
      expect.stringContaining("5 failures"),
    );
    orch.dispose();
  });

  test("failure count in message is singular for exactly 1 failure", async () => {
    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);
    mockRun.mockResolvedValueOnce(makeResponse(1));

    await orch.run(mockDocument(), "f", "liquidity_crash");

    expect(vscode.window.showWarningMessage).toHaveBeenCalledWith(
      expect.stringMatching(/1 failure[^s]/),
    );
    orch.dispose();
  });

  test("returns true on a successful run", async () => {
    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);
    const result = await orch.run(mockDocument(), "f", "liquidity_crash");
    expect(result).toBe(true);
    orch.dispose();
  });
});

// ---------------------------------------------------------------------------
// Error routing
// ---------------------------------------------------------------------------

describe("Orchestrator.run — error routing", () => {
  test("EngineRuntimeError → showErrorMessage mentioning Python not found", async () => {
    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);
    mockRun.mockRejectedValueOnce(new EngineRuntimeError("spawn ENOENT"));

    await orch.run(mockDocument(), "f", "liquidity_crash");

    expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
      expect.stringContaining("Python not found"),
    );
    orch.dispose();
  });

  test("EngineFrameworkError → showErrorMessage with engine message", async () => {
    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);
    mockRun.mockRejectedValueOnce(
      new EngineFrameworkError("Unknown scenario 'bad_scenario'"),
    );

    await orch.run(mockDocument(), "f", "bad_scenario");

    expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
      expect.stringContaining("Unknown scenario 'bad_scenario'"),
    );
    orch.dispose();
  });

  test("EngineProtocolError → showErrorMessage with bug report link", async () => {
    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);
    mockRun.mockRejectedValueOnce(
      new EngineProtocolError("Missing field: shatter_points"),
    );

    await orch.run(mockDocument(), "f", "liquidity_crash");

    const call = (vscode.window.showErrorMessage as jest.Mock).mock.calls[0][0] as string;
    expect(call).toContain("github.com");
    orch.dispose();
  });

  test("cancellation error → silent (no showErrorMessage)", async () => {
    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);
    mockRun.mockRejectedValueOnce(new Error("BlackSwan run cancelled"));

    await orch.run(mockDocument(), "f", "liquidity_crash");

    expect(vscode.window.showErrorMessage).not.toHaveBeenCalled();
    orch.dispose();
  });

  test("cancellation clears diagnostics for the document URI", async () => {
    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);
    const doc  = mockDocument();
    mockRun.mockRejectedValueOnce(new Error("BlackSwan run cancelled"));

    await orch.run(doc, "f", "liquidity_crash");

    expect(diag.clear).toHaveBeenCalledWith(doc.uri);
    orch.dispose();
  });

  test("unknown error → showErrorMessage with error message text", async () => {
    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);
    mockRun.mockRejectedValueOnce(new Error("something totally unexpected"));

    await orch.run(mockDocument(), "f", "liquidity_crash");

    expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
      expect.stringContaining("something totally unexpected"),
    );
    orch.dispose();
  });

  test("mutex is released even when the run throws", async () => {
    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);
    mockRun.mockRejectedValueOnce(new EngineFrameworkError("error"));

    await orch.run(mockDocument(), "f", "liquidity_crash");

    expect(orch.isRunning(FILE_URI)).toBe(false);
    orch.dispose();
  });
});

// ---------------------------------------------------------------------------
// cancel()
// ---------------------------------------------------------------------------

describe("Orchestrator.cancel", () => {
  test("cancel() on a non-running URI is a no-op (no error thrown)", () => {
    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);
    expect(() => orch.cancel(FILE_URI)).not.toThrow();
    orch.dispose();
  });

  test("cancel() while running causes the run to reject with a cancellation error", async () => {
    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);

    // Make runBlackSwanEngine return a promise that we resolve manually.
    let cancelBridge!: (err: Error) => void;
    mockRun.mockReturnValueOnce(
      new Promise((_res, rej) => { cancelBridge = rej; }),
    );

    const doc = mockDocument();
    const runPromise = orch.run(doc, "f", "liquidity_crash");

    // Yield so the run starts and enters withProgress.
    await Promise.resolve();
    await Promise.resolve();

    // Invoke cancel — this should abort the AbortController.
    orch.cancel(doc.uri);

    // Simulate bridge honouring the abort signal.
    cancelBridge(new Error("BlackSwan run cancelled"));

    await runPromise;

    // Cancellation is silent — no error dialog.
    expect(vscode.window.showErrorMessage).not.toHaveBeenCalled();
    orch.dispose();
  });
});

// ---------------------------------------------------------------------------
// dispose()
// ---------------------------------------------------------------------------

describe("Orchestrator.dispose", () => {
  test("dispose() on an idle orchestrator does not throw", () => {
    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);
    expect(() => orch.dispose()).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// Python path resolution
// ---------------------------------------------------------------------------

describe("Orchestrator — Python path resolution", () => {
  test("uses blackswan.pythonPath when configured", async () => {
    // Override workspace.getConfiguration to return a custom python path.
    (vscode.workspace.getConfiguration as jest.Mock).mockImplementation(
      (section: string) => ({
        get: (key: string) => {
          if (section === "blackswan" && key === "pythonPath") return "/custom/python";
          return undefined;
        },
      }),
    );

    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);
    await orch.run(mockDocument(), "f", "liquidity_crash");

    expect(mockRun).toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
      expect.objectContaining({ pythonPath: "/custom/python" }),
    );
    orch.dispose();
  });

  test("falls back to python.defaultInterpreterPath when blackswan.pythonPath is empty", async () => {
    (vscode.workspace.getConfiguration as jest.Mock).mockImplementation(
      (section: string) => ({
        get: (key: string) => {
          if (section === "python" && key === "defaultInterpreterPath") {
            return "/usr/local/bin/python3.11";
          }
          return undefined;
        },
      }),
    );

    const diag = mockDiagnostics();
    const orch = new Orchestrator(diag);
    await orch.run(mockDocument(), "f", "liquidity_crash");

    expect(mockRun).toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
      expect.objectContaining({ pythonPath: "/usr/local/bin/python3.11" }),
    );
    orch.dispose();
  });
});
