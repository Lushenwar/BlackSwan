/**
 * aiExplainer.ts — Claude (Anthropic) integration for BlackSwan failure explanations.
 *
 * Responsibilities:
 *   • BYOK: store and retrieve the Anthropic API key via VS Code SecretStorage (encrypted at rest).
 *   • Rate limiting: enforce a conservative 15 RPM cap in-process.
 *   • Prompt construction: send ONLY the failure metadata (type, message, causal chain, fix_hint)
 *     — never the user's source code — to the API.
 *   • Error routing: distinguish invalid key, rate limit, and network failures.
 *
 * The AI only explains — it never suggests code changes. All code modifications
 * are handled deterministically by fixer.ts + guards.py.
 */

import * as vscode from "vscode";
import { CausalChainLink, FailureType } from "./types";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SECRET_KEY = "blackswan.anthropicApiKey";
const ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_API_VERSION = "2023-06-01";
const DEFAULT_MODEL = "claude-sonnet-4-20250514";

/** Conservative rate limit — Anthropic free tier allows more, but this prevents abuse. */
const RPM_LIMIT = 15;
const RPM_WINDOW_MS = 60_000;

// ---------------------------------------------------------------------------
// In-process rate limiter (circular timestamp buffer)
// ---------------------------------------------------------------------------

const _timestamps: number[] = [];

function _isRateLimited(): boolean {
  const now = Date.now();
  // Evict timestamps older than 1 minute.
  while (_timestamps.length > 0 && _timestamps[0] < now - RPM_WINDOW_MS) {
    _timestamps.shift();
  }
  return _timestamps.length >= RPM_LIMIT;
}

function _recordRequest(): void {
  _timestamps.push(Date.now());
}

// ---------------------------------------------------------------------------
// Payload type — what we pass to the command and include in the prompt
// ---------------------------------------------------------------------------

/** Minimal shatter-point data sent to the AI.  Source code is never included. */
export interface ExplainPayload {
  failure_type: FailureType;
  message: string;
  frequency: string;
  fix_hint: string;
  causal_chain: CausalChainLink[];
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Prompt the user for an Anthropic API key and persist it in SecretStorage.
 * Safe to call from the command palette even when no Python file is open.
 */
export async function setGeminiApiKey(
  secrets: vscode.SecretStorage,
): Promise<void> {
  // Function name kept for backward compat with extension.ts command wiring.
  return setAnthropicApiKey(secrets);
}

export async function setAnthropicApiKey(
  secrets: vscode.SecretStorage,
): Promise<void> {
  const existing = await secrets.get(SECRET_KEY);

  const key = await vscode.window.showInputBox({
    title: "BlackSwan: Set Anthropic API Key",
    prompt:
      "Enter your Anthropic API key. " +
      "It is stored encrypted by VS Code SecretStorage and never sent anywhere except the Anthropic API.",
    password: true,
    placeHolder: "sk-ant-...",
    value: existing ? "••••••••••••••••••••••••••••••••••••••" : undefined,
    ignoreFocusOut: true,
  });

  if (!key?.trim() || key.includes("•")) {
    // User dismissed or left the masked placeholder — nothing to do.
    return;
  }

  await secrets.store(SECRET_KEY, key.trim());
  void vscode.window.showInformationMessage(
    "BlackSwan: Anthropic API key saved. Your key is encrypted at rest by VS Code.",
  );
}

/**
 * Fetch a natural-language explanation of a detected mathematical failure
 * from the Claude API and display it in a side panel.
 *
 * The function is intentionally fire-and-forget: all errors are shown as
 * VS Code messages so the caller does not need to handle them.
 */
export async function explainFailure(
  secrets: vscode.SecretStorage,
  payload: ExplainPayload,
): Promise<void> {
  // ── API key check ──────────────────────────────────────────────────────────
  const apiKey = await secrets.get(SECRET_KEY);
  if (!apiKey) {
    const action = await vscode.window.showWarningMessage(
      "BlackSwan AI: No Anthropic API key configured.",
      "Set API Key",
      "Dismiss",
    );
    if (action === "Set API Key") {
      await vscode.commands.executeCommand("blackswan.setApiKey");
    }
    return;
  }

  // ── Rate limit check ────────────────────────────────────────────────────────
  if (_isRateLimited()) {
    void vscode.window.showWarningMessage(
      "BlackSwan AI: Rate limit reached (15 req/min). " +
        "You can still apply the deterministic fix via the lightbulb menu.",
    );
    return;
  }

  // ── Fetch with progress ─────────────────────────────────────────────────────
  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "BlackSwan AI: Generating explanation…",
      cancellable: false,
    },
    async () => {
      const model = _resolveModel();
      const prompt = _buildPrompt(payload);

      try {
        _recordRequest();
        const explanation = await _callClaude(apiKey, model, prompt);
        await _showExplanation(payload.failure_type, explanation);
      } catch (err) {
        _handleApiError(err);
      }
    },
  );
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Read the configured model name from VS Code settings.
 * Falls back to claude-sonnet-4-20250514.
 */
function _resolveModel(): string {
  const cfg = vscode.workspace.getConfiguration("blackswan");
  // Support legacy geminiModel setting for a graceful migration period.
  return (
    cfg.get<string>("claudeModel") ??
    cfg.get<string>("geminiModel") ??
    DEFAULT_MODEL
  );
}

/**
 * Build the Claude prompt from a shatter point.
 *
 * Safety constraint: only failure metadata is included — never the user's
 * source code. The causal chain variables and line numbers are sufficient
 * for the model to reason about the numerical issue.
 */
function _buildPrompt(p: ExplainPayload): string {
  const chain = p.causal_chain
    .map((l) => `  • Line ${l.line}: \`${l.variable}\` (${l.role})`)
    .join("\n");

  return (
    `You are a quantitative finance and numerical computing expert. ` +
    `A stress-testing engine detected a mathematical failure in a Python financial model.\n\n` +
    `Failure type: ${p.failure_type}\n` +
    `Engine message: ${p.message}\n` +
    `Observed frequency: ${p.frequency}\n` +
    `Causal chain:\n${chain}\n` +
    `Engine-suggested fix: ${p.fix_hint}\n\n` +
    `Explain in plain English (under 200 words):\n` +
    `1. The mathematical invariant that is violated and why it matters in financial models.\n` +
    `2. The market conditions under which this failure would manifest in production.\n` +
    `3. The trade-offs of the suggested fix regarding precision, performance, and model semantics.\n\n` +
    `Be precise. Avoid generalities. ` +
    `Do not suggest code changes — those are handled deterministically by the fix engine.`
  );
}

/** Raw Anthropic Messages API call. Throws on HTTP error or empty response. */
async function _callClaude(
  apiKey: string,
  model: string,
  prompt: string,
): Promise<string> {
  const response = await fetch(ANTHROPIC_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": ANTHROPIC_API_VERSION,
    },
    body: JSON.stringify({
      model,
      max_tokens: 1024,
      temperature: 0.2,
      messages: [
        {
          role: "user",
          content: prompt,
        },
      ],
    }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new AnthropicApiError(response.status, body);
  }

  const data = (await response.json()) as AnthropicResponse;
  const text = data.content?.[0]?.text;
  if (!text) {
    throw new Error("Claude returned an empty response.");
  }
  return text;
}

/** Show the AI explanation in a read-only Markdown document beside the editor. */
async function _showExplanation(
  failureType: FailureType,
  explanation: string,
): Promise<void> {
  const heading = _formatFailureTypeLabel(failureType);
  const content =
    `# BlackSwan AI: ${heading}\n\n` +
    `${explanation}\n\n` +
    `---\n` +
    `*Explanation generated by Claude. ` +
    `All code fixes are applied deterministically — not by AI.*`;

  const doc = await vscode.workspace.openTextDocument({
    content,
    language: "markdown",
  });
  await vscode.window.showTextDocument(doc, {
    viewColumn: vscode.ViewColumn.Beside,
    preserveFocus: true,
    preview: true,
  });
}

function _formatFailureTypeLabel(type: FailureType): string {
  const labels: Record<FailureType, string> = {
    nan_inf: "NaN / Inf Detected",
    division_by_zero: "Division Instability",
    non_psd_matrix: "Non-PSD Matrix",
    ill_conditioned_matrix: "Ill-Conditioned Matrix",
    bounds_exceeded: "Bounds Exceeded",
    division_instability: "Division Instability",
    exploding_gradient: "Exploding Gradient",
    regime_shift: "Regime Shift",
    logical_invariant: "Logical Invariant Violation",
  };
  return labels[type] ?? type;
}

function _handleApiError(err: unknown): void {
  if (err instanceof AnthropicApiError) {
    if (err.status === 400) {
      void vscode.window.showErrorMessage(
        "BlackSwan AI: Bad request to Claude API. " +
          "The prompt may have exceeded the token limit.",
      );
    } else if (err.status === 401) {
      void vscode.window.showErrorMessage(
        "BlackSwan AI: Invalid or expired API key. " +
          "Run 'BlackSwan: Set Anthropic API Key' from the command palette to update it.",
      );
    } else if (err.status === 429) {
      void vscode.window.showWarningMessage(
        "BlackSwan AI: Claude API rate limit exceeded. " +
          "You can still apply the deterministic fix via the lightbulb menu.",
      );
    } else {
      void vscode.window.showErrorMessage(
        `BlackSwan AI: Claude API error ${err.status}. ${err.shortBody}`,
      );
    }
    return;
  }

  const msg = err instanceof Error ? err.message : String(err);
  void vscode.window.showErrorMessage(`BlackSwan AI: ${msg}`);
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

class AnthropicApiError extends Error {
  readonly status: number;
  readonly shortBody: string;

  constructor(status: number, body: string) {
    super(`Anthropic API error ${status}`);
    this.name = "AnthropicApiError";
    this.status = status;
    this.shortBody = body.slice(0, 300);
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

// ---------------------------------------------------------------------------
// Anthropic Messages API response shape (partial)
// ---------------------------------------------------------------------------

interface AnthropicResponse {
  id?: string;
  type?: string;
  role?: string;
  content?: Array<{
    type?: string;
    text?: string;
  }>;
  model?: string;
  stop_reason?: string;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
  };
}
