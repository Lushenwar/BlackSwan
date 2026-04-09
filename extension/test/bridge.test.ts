/**
 * Tests for bridge.ts — the TypeScript ↔ Python engine bridge.
 *
 * These are integration tests: they spawn the real Python engine against
 * real fixture files. No mocks. If the Python engine isn't installed or
 * the fixtures are missing, the tests fail loudly.
 *
 * Run from extension/: npm test
 */

import * as path from "path";
import { runBlackSwanEngine, validateEngineResponse } from "../src/bridge";
import {
  EngineFrameworkError,
  EngineProtocolError,
  EngineRuntimeError,
} from "../src/types";

// Resolve paths relative to the repo root so tests work regardless of cwd.
const REPO_ROOT = path.resolve(__dirname, "..", "..");
const CORE_DIR = path.join(REPO_ROOT, "core");
const FIXTURES_DIR = path.join(CORE_DIR, "tests", "fixtures");
/**
 * Resolve the real absolute Python executable path.
 *
 * Jest test environments often have modified PATHs, making bare `spawn("py")`
 * fail with ENOENT. By asking Python for its `sys.executable`, we get the
 * absolute path (e.g., C:\Users\...\Python313\python.exe) which Node can
 * spawn directly.
 */
function resolvePython(): string {
  if (process.platform === "win32") {
    try {
      return require("child_process").execSync(
        'py -3.11 -c "import sys; print(sys.executable)"',
        { encoding: "utf-8" }
      ).trim();
    } catch {
      return "python";
    }
  }
  return "python3";
}

const PYTHON = resolvePython();
const PYTHON_ARGS: string[] = [];

const CLEAN_MODEL = path.join(FIXTURES_DIR, "clean_model.py");
const BROKEN_COV = path.join(FIXTURES_DIR, "broken_covariance.py");

// Shared options shorthand
function opts(extra: object = {}) {
  return { pythonPath: PYTHON, pythonArgs: PYTHON_ARGS, cwd: CORE_DIR, ...extra };
}

// ---------------------------------------------------------------------------
// Successful runs
// ---------------------------------------------------------------------------

describe("runBlackSwanEngine — successful runs", () => {
  test("clean model with liquidity_crash returns status no_failures", async () => {
    const response = await runBlackSwanEngine(
      CLEAN_MODEL,
      "liquidity_crash",
      opts({ iterations: 10, seed: 42 })
    );

    expect(response.status).toBe("no_failures");
    expect(response.shatter_points).toHaveLength(0);
  });

  test("clean model response has correct top-level contract fields", async () => {
    const response = await runBlackSwanEngine(
      CLEAN_MODEL,
      "liquidity_crash",
      opts({ iterations: 5, seed: 1 })
    );

    expect(response.version).toBe("1.0");
    expect(typeof response.runtime_ms).toBe("number");
    expect(response.iterations_completed).toBe(5);
    expect(response.summary.total_failures).toBe(0);
    expect(response.summary.failure_rate).toBe(0);
    expect(response.summary.unique_failure_types).toBe(0);
  });

  test("clean model scenario_card matches requested scenario and seed", async () => {
    const response = await runBlackSwanEngine(
      CLEAN_MODEL,
      "liquidity_crash",
      opts({ iterations: 5, seed: 99 })
    );

    expect(response.scenario_card.name).toBe("liquidity_crash");
    expect(response.scenario_card.seed).toBe(99);
    expect(response.scenario_card.reproducible).toBe(true);
  });

  test("broken_covariance with liquidity_crash returns failures_detected", async () => {
    const response = await runBlackSwanEngine(
      BROKEN_COV,
      "liquidity_crash",
      opts({ iterations: 50, seed: 42 })
    );

    expect(response.status).toBe("failures_detected");
    expect(response.shatter_points.length).toBeGreaterThan(0);
  });

  test("broken_covariance shatter_points contain non_psd_matrix failure", async () => {
    const response = await runBlackSwanEngine(BROKEN_COV, "liquidity_crash", opts());
    const types = response.shatter_points.map((sp) => sp.failure_type);
    // liquidity_crash perturbs `correlation` (additive), pushing pairwise
    // correlations above 1.0 → covariance matrix loses PSD.
    expect(types).toContain("non_psd_matrix");
  });

  test("shatter_point fields conform to contract shape", async () => {
    const response = await runBlackSwanEngine(
      BROKEN_COV,
      "liquidity_crash",
      opts({ iterations: 50, seed: 42 })
    );

    expect(response.shatter_points.length).toBeGreaterThan(0);
    const sp = response.shatter_points[0];

    expect(sp.id).toMatch(/^sp_\d+$/);
    expect(["critical", "warning", "info"]).toContain(sp.severity);
    expect(typeof sp.message).toBe("string");
    expect(sp.message.length).toBeGreaterThan(0);
    expect(sp.frequency).toMatch(/^\d+ \/ \d+ iterations \(\d+(\.\d+)?%\)$/);
    expect(Array.isArray(sp.causal_chain)).toBe(true);
    expect(typeof sp.fix_hint).toBe("string");
  });

  test("same seed produces identical results across two runs", async () => {
    const runOpts = opts({ iterations: 20, seed: 77 });
    const [r1, r2] = await Promise.all([
      runBlackSwanEngine(BROKEN_COV, "liquidity_crash", runOpts),
      runBlackSwanEngine(BROKEN_COV, "liquidity_crash", runOpts),
    ]);

    expect(r1.summary).toEqual(r2.summary);
    expect(r1.shatter_points.length).toBe(r2.shatter_points.length);
  });
});

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

describe("runBlackSwanEngine — error cases", () => {
  test("rejects when file path does not exist", async () => {
    await expect(
      runBlackSwanEngine(
        "/nonexistent/path/model.py",
        "liquidity_crash",
        opts()
      )
    ).rejects.toThrow();
  });

  test("error message mentions the missing file path", async () => {
    await expect(
      runBlackSwanEngine(
        "/nonexistent/path/model.py",
        "liquidity_crash",
        opts()
      )
    ).rejects.toThrow(/model\.py/);
  });

  test("rejects when scenario name is unknown", async () => {
    await expect(
      runBlackSwanEngine(
        CLEAN_MODEL,
        "nonexistent_scenario_xyz",
        opts()
      )
    ).rejects.toThrow();
  });

  test("error message mentions the unknown scenario name", async () => {
    await expect(
      runBlackSwanEngine(
        CLEAN_MODEL,
        "nonexistent_scenario_xyz",
        opts()
      )
    ).rejects.toThrow(/nonexistent_scenario_xyz/);
  });
});

// ---------------------------------------------------------------------------
// Standard 3: Error discrimination — each failure path throws a distinct type
// ---------------------------------------------------------------------------

describe("runBlackSwanEngine — error discrimination", () => {
  test("bad file path throws EngineFrameworkError (exit code 2)", async () => {
    await expect(
      runBlackSwanEngine("/nonexistent/path/model.py", "liquidity_crash", opts())
    ).rejects.toBeInstanceOf(EngineFrameworkError);
  });

  test("unknown scenario throws EngineFrameworkError (exit code 2)", async () => {
    await expect(
      runBlackSwanEngine(CLEAN_MODEL, "nonexistent_scenario_xyz", opts())
    ).rejects.toBeInstanceOf(EngineFrameworkError);
  });

  test("invalid Python executable throws EngineRuntimeError", async () => {
    await expect(
      runBlackSwanEngine(CLEAN_MODEL, "liquidity_crash", {
        ...opts(),
        pythonPath: "/nonexistent/python_exe_that_cannot_exist",
        pythonArgs: [],
      })
    ).rejects.toBeInstanceOf(EngineRuntimeError);
  });
});

// ---------------------------------------------------------------------------
// Standard 2: Protocol validation — validateEngineResponse catches bad shapes
// ---------------------------------------------------------------------------

describe("validateEngineResponse — schema enforcement", () => {
  // Build a minimal valid response so individual field corruption is isolated.
  function validResponse(): unknown {
    return {
      version: "1.0",
      status: "no_failures",
      runtime_ms: 100,
      iterations_completed: 10,
      summary: { total_failures: 0, failure_rate: 0, unique_failure_types: 0 },
      shatter_points: [],
      scenario_card: {
        name: "liquidity_crash",
        parameters_applied: {},
        seed: 42,
        reproducible: true,
      },
    };
  }

  test("accepts a fully valid response without throwing", () => {
    expect(() => validateEngineResponse(validResponse())).not.toThrow();
  });

  test("throws EngineProtocolError when root is not an object", () => {
    expect(() => validateEngineResponse("not an object")).toThrow(EngineProtocolError);
  });

  test("throws EngineProtocolError when status is an unknown enum value", () => {
    const r = validResponse() as Record<string, unknown>;
    r.status = "definitely_not_valid";
    expect(() => validateEngineResponse(r)).toThrow(EngineProtocolError);
  });

  test("throws EngineProtocolError when shatter_points is missing", () => {
    const r = validResponse() as Record<string, unknown>;
    delete r.shatter_points;
    expect(() => validateEngineResponse(r)).toThrow(EngineProtocolError);
  });

  test("throws EngineProtocolError when a shatter_point has an invalid severity", () => {
    const r = validResponse() as Record<string, unknown>;
    r.shatter_points = [
      {
        id: "sp_001",
        line: 10,
        column: null,
        severity: "catastrophic", // invalid
        failure_type: "nan_inf",
        message: "test",
        frequency: "1 / 10 iterations (10%)",
        causal_chain: [],
        fix_hint: "",
      },
    ];
    expect(() => validateEngineResponse(r)).toThrow(EngineProtocolError);
  });

  test("throws EngineProtocolError when a shatter_point has an invalid failure_type", () => {
    const r = validResponse() as Record<string, unknown>;
    r.shatter_points = [
      {
        id: "sp_001",
        line: 10,
        column: null,
        severity: "critical",
        failure_type: "unknown_detector", // invalid
        message: "test",
        frequency: "1 / 10 iterations (10%)",
        causal_chain: [],
        fix_hint: "",
      },
    ];
    expect(() => validateEngineResponse(r)).toThrow(EngineProtocolError);
  });

  test("throws EngineProtocolError when a causal_chain link has an invalid role", () => {
    const r = validResponse() as Record<string, unknown>;
    r.shatter_points = [
      {
        id: "sp_001",
        line: 10,
        column: null,
        severity: "critical",
        failure_type: "nan_inf",
        message: "test",
        frequency: "1 / 10 iterations (10%)",
        causal_chain: [
          { line: 5, variable: "x", role: "not_a_real_role" }, // invalid
        ],
        fix_hint: "",
      },
    ];
    expect(() => validateEngineResponse(r)).toThrow(EngineProtocolError);
  });

  test("throws EngineProtocolError when summary.failure_rate is missing", () => {
    const r = validResponse() as Record<string, unknown>;
    r.summary = { total_failures: 0, unique_failure_types: 0 }; // missing failure_rate
    expect(() => validateEngineResponse(r)).toThrow(EngineProtocolError);
  });
});

// ---------------------------------------------------------------------------
// Standard 1: Process resilience — AbortSignal cancellation kills the process
// ---------------------------------------------------------------------------

describe("runBlackSwanEngine — cancellation", () => {
  test("rejects with a cancellation message when signal is aborted before spawn", async () => {
    const controller = new AbortController();
    controller.abort(); // aborted before the call

    await expect(
      runBlackSwanEngine(CLEAN_MODEL, "liquidity_crash", {
        ...opts({ iterations: 5000 }),
        signal: controller.signal,
      })
    ).rejects.toThrow(/cancelled/i);
  });

  test("rejects with a cancellation message when signal is aborted mid-run", async () => {
    const controller = new AbortController();

    // Abort after a short delay while the engine is running 5000 iterations.
    setTimeout(() => controller.abort(), 100);

    await expect(
      runBlackSwanEngine(BROKEN_COV, "liquidity_crash", {
        ...opts({ iterations: 5000 }),
        signal: controller.signal,
      })
    ).rejects.toThrow(/cancelled/i);
  });
});
