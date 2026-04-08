/**
 * TypeScript interfaces matching contract/schema.json v1.0.
 *
 * These are the ONLY types the extension uses to communicate with the
 * Python engine. Never access raw JSON — always go through these types.
 * If the contract changes, update schema.json first, then update here.
 */

// ---------------------------------------------------------------------------
// Typed error hierarchy — lets callers switch on `instanceof` instead of
// parsing error messages.
// ---------------------------------------------------------------------------

/**
 * Python executable was not found or could not be spawned.
 * Cause: wrong pythonPath, Python not installed, ENOENT, EACCES.
 * UI action: show "Python not found — check your settings" message.
 */
export class EngineRuntimeError extends Error {
  readonly kind = "runtime" as const;
  constructor(message: string) {
    super(message);
    this.name = "EngineRuntimeError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * The Python engine launched but exited with code 2 (engine-level error).
 * Cause: bad file path, unknown scenario name, parse error inside engine.
 * UI action: show the engine's own error message to the user.
 */
export class EngineFrameworkError extends Error {
  readonly kind = "framework" as const;
  constructor(message: string) {
    super(message);
    this.name = "EngineFrameworkError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * The engine exited cleanly (code 0 or 1) but its output failed schema
 * validation — a contract violation between engine and extension.
 * Cause: engine version mismatch, partial write, corrupt stdout.
 * UI action: show "Unexpected engine output — please file a bug" message.
 */
export class EngineProtocolError extends Error {
  readonly kind = "protocol" as const;
  constructor(message: string) {
    super(message);
    this.name = "EngineProtocolError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export type FailureType =
  | "nan_inf"
  | "division_by_zero"
  | "non_psd_matrix"
  | "ill_conditioned_matrix"
  | "bounds_exceeded";

export type Severity = "critical" | "warning" | "info";

export type ResponseStatus = "failures_detected" | "no_failures" | "error";

export type CausalRole = "root_input" | "intermediate" | "failure_site";

export interface CausalChainLink {
  line: number;
  variable: string;
  role: CausalRole;
}

export interface ShatterPoint {
  id: string;
  line: number | null;
  column: number | null;
  severity: Severity;
  failure_type: FailureType;
  message: string;
  frequency: string;
  causal_chain: CausalChainLink[];
  fix_hint: string;
}

export interface ResponseSummary {
  total_failures: number;
  failure_rate: number;
  unique_failure_types: number;
}

export interface ScenarioCard {
  name: string;
  parameters_applied: Record<string, unknown>;
  seed: number;
  reproducible: boolean;
}

export interface BlackSwanResponse {
  version: string;
  status: ResponseStatus;
  runtime_ms: number;
  iterations_completed: number;
  summary: ResponseSummary;
  shatter_points: ShatterPoint[];
  scenario_card: ScenarioCard;
}

export interface BridgeOptions {
  /** Name of function to stress-test. Omit to auto-detect. */
  functionName?: string;
  /** Number of iterations. Overrides the scenario default. */
  iterations?: number;
  /** RNG seed for reproducibility. Overrides the scenario default. */
  seed?: number;
  /** Python executable to use. Defaults to "python3" (or "python" on Windows). */
  pythonPath?: string;
  /**
   * Extra arguments inserted between pythonPath and "-m blackswan".
   * Useful for the Windows `py` launcher: pythonPath="py", pythonArgs=["-3.11"].
   */
  pythonArgs?: string[];
  /** Working directory for the Python subprocess. Defaults to the directory of filePath. */
  cwd?: string;
  /**
   * Cancellation signal. Abort this signal to kill the engine process and
   * reject the returned promise. Wire a VS Code CancellationToken to this
   * via AbortController in progress.ts — keeps this module VS Code-free.
   */
  signal?: AbortSignal;
}
