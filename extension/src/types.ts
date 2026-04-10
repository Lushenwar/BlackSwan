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

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

export type FailureType =
  | "nan_inf"
  | "division_by_zero"
  | "non_psd_matrix"
  | "ill_conditioned_matrix"
  | "bounds_exceeded"
  | "division_instability"
  | "exploding_gradient"
  | "regime_shift"
  | "logical_invariant";

export type Severity = "critical" | "warning" | "info";

export type ResponseStatus = "failures_detected" | "no_failures" | "error";

export type CausalRole = "root_input" | "intermediate" | "failure_site";

/** Attribution confidence level. 'unverified' = fast/adversarial mode, no Slow-Path replay. */
export type Confidence = "high" | "medium" | "low" | "unverified";

// ---------------------------------------------------------------------------
// Nested contract types
// ---------------------------------------------------------------------------

export interface CausalChainLink {
  line: number;
  variable: string;
  role: CausalRole;
}

/**
 * Exact threshold that caused a threshold-based detector to fire.
 * Present on shatter_points where a concrete detector (MatrixPSDDetector,
 * ConditionNumberDetector, DivisionStabilityDetector) fired on a numeric threshold.
 * Absent for exception-based findings (nan_inf from a raised exception).
 */
export interface TriggerDisclosure {
  detector_name: string;
  observed_value: number | string;
  threshold: number | string;
  comparison: ">" | "<" | ">=" | "<=" | "==" | "!=";
  explanation: string;
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
  /** Attribution confidence. 'unverified' when Slow-Path replay was skipped (fast/adversarial mode). */
  confidence: Confidence;
  /** Populated by threshold-based detectors. Absent for exception-based findings. */
  trigger_disclosure?: TriggerDisclosure;
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

/**
 * Full provenance record for a BlackSwan run.
 * Enables exact reproduction of any run: same command, same output.
 */
export interface ReproducibilityCard {
  blackswan_version: string;
  python_version: string;
  numpy_version: string;
  platform: string;
  seed: number;
  scenario_name: string;
  scenario_hash: string;
  mode: string;
  iterations_requested: number;
  iterations_executed: number;
  iterations_skipped: number;
  budget_exhausted: boolean;
  budget_reason: string | null;
  timestamp_utc: string;
  reproducible: boolean;
  replay_command: string;
}

/** Budget exhaustion status for a run. */
export interface Budget {
  exhausted: boolean;
  reason: string | null;
}

export interface BlackSwanResponse {
  version: string;
  status: ResponseStatus;
  /** Execution mode: "fast" (no attribution) | "full" (Two-Path) | "adversarial" (GA). */
  mode: string;
  runtime_ms: number;
  iterations_completed: number;
  summary: ResponseSummary;
  shatter_points: ShatterPoint[];
  scenario_card: ScenarioCard;
  reproducibility_card: ReproducibilityCard;
  budget: Budget;
}

// ---------------------------------------------------------------------------
// Bridge options
// ---------------------------------------------------------------------------

export interface BridgeOptions {
  /** Name of function to stress-test. Omit to auto-detect. */
  functionName?: string;
  /** Number of iterations. Overrides the scenario default. */
  iterations?: number;
  /** RNG seed for reproducibility. Overrides the scenario default. */
  seed?: number;
  /**
   * Execution mode. 'fast' = no attribution tracing (faster).
   * 'full' = Two-Path with Slow-Path attribution replay.
   * Defaults to 'fast' in the extension for responsive IDE feedback.
   */
  mode?: "fast" | "full";
  /**
   * Hard time limit in seconds. The engine stops after this many seconds
   * even if not all iterations completed.
   */
  maxRuntimeSec?: number;
  /**
   * Hard iteration cap. Stops after N iterations regardless of scenario default.
   */
  maxIterations?: number;
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
   * via AbortController in the orchestrator — keeps this module VS Code-free.
   */
  signal?: AbortSignal;
  /**
   * Streaming progress callback. Called with (completed, total) on each
   * intermediate progress event when the engine supports NDJSON streaming.
   * Also called once with the final iteration count after a successful run
   * so the progress bar advances to 100% in bulk mode.
   * Safe to omit — the bridge is a no-op if this is not set.
   */
  onProgress?: (completed: number, total: number) => void;
}
