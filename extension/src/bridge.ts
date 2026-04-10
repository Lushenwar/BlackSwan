/**
 * Bridge between the TypeScript extension and the Python blackswan-core engine.
 *
 * Spawns `python -m blackswan test <file> --scenario <name> [options]`
 * as a child process, collects stdout, and parses the JSON response.
 *
 * Exit codes from the engine:
 *   0 — no failures detected
 *   1 — failures detected (shatter_points array is non-empty)
 *   2 — error (bad path, unknown scenario, parse error, etc.)
 *
 * This module has NO dependency on the VS Code API so it can be tested
 * in a plain Node.js environment with Jest.
 */

import { spawn } from "child_process";
import * as path from "path";
import {
  BlackSwanResponse,
  BridgeOptions,
  CausalRole,
  Confidence,
  EngineFrameworkError,
  EngineProtocolError,
  EngineRuntimeError,
  FailureType,
  ResponseStatus,
  Severity,
} from "./types";

const DEFAULT_PYTHON = process.platform === "win32" ? "python" : "python3";

const VALID_STATUSES: ResponseStatus[] = ["failures_detected", "no_failures", "error"];
const VALID_SEVERITIES: Severity[] = ["critical", "warning", "info"];
const VALID_FAILURE_TYPES: FailureType[] = [
  "nan_inf",
  "division_by_zero",
  "non_psd_matrix",
  "ill_conditioned_matrix",
  "bounds_exceeded",
  "division_instability",
  "exploding_gradient",
  "regime_shift",
  "logical_invariant",
];
const VALID_ROLES: CausalRole[] = ["root_input", "intermediate", "failure_site"];
const VALID_CONFIDENCE_LEVELS: Confidence[] = ["high", "medium", "low", "unverified"];

/**
 * Validate raw parsed JSON against the BlackSwan response contract.
 *
 * Exported so tests can exercise the validator directly without spawning a
 * Python process.
 *
 * Throws EngineProtocolError with a precise field path on any violation.
 */
export function validateEngineResponse(data: unknown): BlackSwanResponse {
  if (typeof data !== "object" || data === null) {
    throw new EngineProtocolError("Response root is not an object");
  }
  const r = data as Record<string, unknown>;

  if (typeof r.version !== "string") {
    throw new EngineProtocolError("Missing or invalid field: 'version' (expected string)");
  }
  if (!VALID_STATUSES.includes(r.status as ResponseStatus)) {
    throw new EngineProtocolError(`Invalid 'status' value: ${JSON.stringify(r.status)}`);
  }
  if (typeof r.runtime_ms !== "number") {
    throw new EngineProtocolError("Missing or invalid field: 'runtime_ms' (expected number)");
  }
  if (typeof r.iterations_completed !== "number") {
    throw new EngineProtocolError("Missing or invalid field: 'iterations_completed' (expected number)");
  }

  // summary
  if (typeof r.summary !== "object" || r.summary === null) {
    throw new EngineProtocolError("Missing or invalid field: 'summary' (expected object)");
  }
  const s = r.summary as Record<string, unknown>;
  for (const key of ["total_failures", "failure_rate", "unique_failure_types"] as const) {
    if (typeof s[key] !== "number") {
      throw new EngineProtocolError(`Missing or invalid field: 'summary.${key}' (expected number)`);
    }
  }

  // shatter_points
  if (!Array.isArray(r.shatter_points)) {
    throw new EngineProtocolError("Missing or invalid field: 'shatter_points' (expected array)");
  }
  for (let i = 0; i < r.shatter_points.length; i++) {
    _validateShatterPoint(r.shatter_points[i], i);
  }

  // scenario_card
  if (typeof r.scenario_card !== "object" || r.scenario_card === null) {
    throw new EngineProtocolError("Missing or invalid field: 'scenario_card' (expected object)");
  }
  const card = r.scenario_card as Record<string, unknown>;
  if (typeof card.name !== "string") {
    throw new EngineProtocolError("Missing or invalid field: 'scenario_card.name' (expected string)");
  }
  if (typeof card.seed !== "number") {
    throw new EngineProtocolError("Missing or invalid field: 'scenario_card.seed' (expected number)");
  }
  if (typeof card.reproducible !== "boolean") {
    throw new EngineProtocolError("Missing or invalid field: 'scenario_card.reproducible' (expected boolean)");
  }

  // New fields — validated when present but not required for backward compat
  // with older engine versions that pre-date these fields.
  if (r.budget !== undefined) {
    _validateBudget(r.budget);
  }
  if (r.reproducibility_card !== undefined) {
    _validateReproducibilityCard(r.reproducibility_card);
  }

  return data as BlackSwanResponse;
}

/**
 * Run the BlackSwan stress-test engine against a Python file.
 *
 * Resolves with a validated BlackSwanResponse on exit codes 0 and 1.
 * Rejects with a typed error on any failure:
 *   EngineRuntimeError   — Python not found / could not be spawned
 *   EngineFrameworkError — Engine exited with code 2 (bad path, unknown scenario, etc.)
 *   EngineProtocolError  — Engine output failed schema validation
 *   Error("cancelled")   — options.signal was aborted before the run completed
 */
export async function runBlackSwanEngine(
  filePath: string,
  scenarioName: string,
  options: BridgeOptions = {}
): Promise<BlackSwanResponse> {
  const pythonPath = options.pythonPath ?? DEFAULT_PYTHON;
  const cwd = options.cwd ?? path.dirname(filePath);
  const args = _buildArgs(filePath, scenarioName, options);

  return new Promise((resolve, reject) => {
    // Guard against double-settlement (cancel races close, error races close, etc.)
    let settled = false;
    const settle = (fn: () => void) => {
      if (!settled) { settled = true; fn(); }
    };

    const proc = spawn(pythonPath, args, {
      cwd,
      stdio: ["ignore", "pipe", "pipe"],
      env: process.env,
    });

    // Cancellation — kill the process and reject immediately.
    if (options.signal) {
      if (options.signal.aborted) {
        proc.kill();
        settle(() => reject(new Error("BlackSwan run cancelled")));
        return;
      }
      options.signal.addEventListener(
        "abort",
        () => {
          proc.kill();
          settle(() => reject(new Error("BlackSwan run cancelled")));
        },
        { once: true }
      );
    }

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf-8");
    });

    proc.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf-8");
    });

    // EngineRuntimeError — Python not found, ENOENT, EACCES, etc.
    proc.on("error", (err) => {
      settle(() =>
        reject(
          new EngineRuntimeError(
            `Failed to spawn Python process '${pythonPath}': ${err.message}`
          )
        )
      );
    });

    proc.on("close", (code) => {
      // EngineFrameworkError — engine's own error handling (exit 2).
      if (code === 2) {
        const message = _extractErrorMessage(stderr, filePath, scenarioName);
        settle(() => reject(new EngineFrameworkError(message)));
        return;
      }

      // Exit codes 0 and 1 both produce valid JSON on stdout.
      let parsed: unknown;
      try {
        parsed = JSON.parse(stdout);
      } catch {
        settle(() =>
          reject(
            new EngineProtocolError(
              `Engine output is not valid JSON.\n` +
              `stdout: ${stdout.slice(0, 500)}\n` +
              `stderr: ${stderr.slice(0, 500)}`
            )
          )
        );
        return;
      }

      // EngineProtocolError — valid JSON but wrong shape.
      try {
        const validated = validateEngineResponse(parsed);
        // Report final iteration count so the progress bar reaches 100% in
        // bulk mode (single-shot engine output, no streaming). In streaming
        // mode this is a redundant but harmless call.
        options.onProgress?.(
          validated.iterations_completed,
          validated.iterations_completed,
        );
        settle(() => resolve(validated));
      } catch (err) {
        settle(() => reject(err));
      }
    });
  });
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function _validateShatterPoint(sp: unknown, index: number): void {
  const ctx = `shatter_points[${index}]`;
  if (typeof sp !== "object" || sp === null) {
    throw new EngineProtocolError(`${ctx} is not an object`);
  }
  const p = sp as Record<string, unknown>;

  if (typeof p.id !== "string") throw new EngineProtocolError(`${ctx}.id is not a string`);
  if (p.line !== null && typeof p.line !== "number") throw new EngineProtocolError(`${ctx}.line must be number or null`);
  if (p.column !== null && typeof p.column !== "number") throw new EngineProtocolError(`${ctx}.column must be number or null`);
  if (!VALID_SEVERITIES.includes(p.severity as Severity)) throw new EngineProtocolError(`${ctx}.severity is invalid: ${JSON.stringify(p.severity)}`);
  if (!VALID_FAILURE_TYPES.includes(p.failure_type as FailureType)) throw new EngineProtocolError(`${ctx}.failure_type is invalid: ${JSON.stringify(p.failure_type)}`);
  if (typeof p.message !== "string") throw new EngineProtocolError(`${ctx}.message is not a string`);
  if (typeof p.frequency !== "string") throw new EngineProtocolError(`${ctx}.frequency is not a string`);
  if (!Array.isArray(p.causal_chain)) throw new EngineProtocolError(`${ctx}.causal_chain is not an array`);
  if (typeof p.fix_hint !== "string") throw new EngineProtocolError(`${ctx}.fix_hint is not a string`);

  // confidence — required field (present in all engine responses >= v3 rewrite)
  if (p.confidence !== undefined && !VALID_CONFIDENCE_LEVELS.includes(p.confidence as Confidence)) {
    throw new EngineProtocolError(`${ctx}.confidence is invalid: ${JSON.stringify(p.confidence)}`);
  }

  // trigger_disclosure — optional; validate shape when present
  if (p.trigger_disclosure !== undefined) {
    _validateTriggerDisclosure(p.trigger_disclosure, ctx);
  }

  for (let j = 0; j < (p.causal_chain as unknown[]).length; j++) {
    const link = (p.causal_chain as unknown[])[j];
    const lctx = `${ctx}.causal_chain[${j}]`;
    if (typeof link !== "object" || link === null) throw new EngineProtocolError(`${lctx} is not an object`);
    const l = link as Record<string, unknown>;
    if (typeof l.line !== "number") throw new EngineProtocolError(`${lctx}.line is not a number`);
    if (typeof l.variable !== "string") throw new EngineProtocolError(`${lctx}.variable is not a string`);
    if (!VALID_ROLES.includes(l.role as CausalRole)) throw new EngineProtocolError(`${lctx}.role is invalid: ${JSON.stringify(l.role)}`);
  }
}

function _validateTriggerDisclosure(td: unknown, parentCtx: string): void {
  const ctx = `${parentCtx}.trigger_disclosure`;
  if (typeof td !== "object" || td === null) {
    throw new EngineProtocolError(`${ctx} is not an object`);
  }
  const d = td as Record<string, unknown>;
  if (typeof d.detector_name !== "string") throw new EngineProtocolError(`${ctx}.detector_name is not a string`);
  if (typeof d.explanation !== "string") throw new EngineProtocolError(`${ctx}.explanation is not a string`);
  const validComparisons = [">", "<", ">=", "<=", "==", "!="];
  if (!validComparisons.includes(d.comparison as string)) {
    throw new EngineProtocolError(`${ctx}.comparison is invalid: ${JSON.stringify(d.comparison)}`);
  }
}

function _validateBudget(budget: unknown): void {
  if (typeof budget !== "object" || budget === null) {
    throw new EngineProtocolError("'budget' is not an object");
  }
  const b = budget as Record<string, unknown>;
  if (typeof b.exhausted !== "boolean") {
    throw new EngineProtocolError("'budget.exhausted' is not a boolean");
  }
  if (b.reason !== null && typeof b.reason !== "string") {
    throw new EngineProtocolError("'budget.reason' must be string or null");
  }
}

function _validateReproducibilityCard(card: unknown): void {
  if (typeof card !== "object" || card === null) {
    throw new EngineProtocolError("'reproducibility_card' is not an object");
  }
  const c = card as Record<string, unknown>;
  const stringFields = [
    "blackswan_version", "python_version", "numpy_version", "platform",
    "scenario_name", "scenario_hash", "mode", "timestamp_utc", "replay_command",
  ];
  for (const field of stringFields) {
    if (typeof c[field] !== "string") {
      throw new EngineProtocolError(`'reproducibility_card.${field}' is not a string`);
    }
  }
  if (typeof c.seed !== "number") throw new EngineProtocolError("'reproducibility_card.seed' is not a number");
  if (typeof c.reproducible !== "boolean") throw new EngineProtocolError("'reproducibility_card.reproducible' is not a boolean");
}

function _buildArgs(
  filePath: string,
  scenarioName: string,
  options: BridgeOptions
): string[] {
  const args = [
    ...(options.pythonArgs ?? []),
    "-m", "blackswan", "test", filePath, "--scenario", scenarioName,
  ];

  if (options.functionName) {
    args.push("--function", options.functionName);
  }
  if (options.iterations !== undefined) {
    args.push("--iterations", String(options.iterations));
  }
  if (options.seed !== undefined) {
    args.push("--seed", String(options.seed));
  }
  if (options.mode) {
    args.push("--mode", options.mode);
  }
  if (options.maxRuntimeSec !== undefined) {
    args.push("--max-runtime-sec", String(options.maxRuntimeSec));
  }
  if (options.maxIterations !== undefined) {
    args.push("--max-iterations", String(options.maxIterations));
  }

  return args;
}

function _extractErrorMessage(
  stderr: string,
  filePath: string,
  scenarioName: string
): string {
  // Engine writes { "status": "error", "message": "..." } to stderr on exit 2.
  try {
    const errObj = JSON.parse(stderr.trim()) as { message?: string };
    if (errObj.message) {
      return errObj.message;
    }
  } catch {
    // stderr wasn't JSON — fall through to a generic message that includes
    // the original inputs so the error is traceable.
  }
  return (
    `BlackSwan engine error for file '${path.basename(filePath)}' ` +
    `with scenario '${scenarioName}'.\n${stderr.trim()}`
  );
}
