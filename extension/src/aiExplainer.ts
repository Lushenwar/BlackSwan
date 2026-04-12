/**
 * aiExplainer.ts — Gemini Flash integration for BlackSwan failure explanations.
 *
 * Responsibilities:
 *   • BYOK: store and retrieve the Gemini API key via VS Code SecretStorage (encrypted at rest).
 *   • Rate limiting: enforce the free-tier 15 RPM cap in-process.
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

const SECRET_KEY = "blackswan.geminiApiKey";
const GEMINI_ENDPOINT =
  "https://generativelanguage.googleapis.com/v1beta/models";

/** Matches the 15 RPM free-tier quota. */
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
 * Prompt the user for a Gemini API key and persist it in SecretStorage.
 * Safe to call from the command palette even when no Python file is open.
 */
export async function setGeminiApiKey(
  secrets: vscode.SecretStorage,
): Promise<void> {
  const existing = await secrets.get(SECRET_KEY);

  const key = await vscode.window.showInputBox({
    title: "BlackSwan: Set Gemini API Key",
    prompt:
      "Enter your Google Gemini API key. " +
      "It is stored encrypted by VS Code SecretStorage and never sent anywhere except the Gemini API.",
    password: true,
    placeHolder: "AIza...",
    value: existing ? "••••••••••••••••••••••••••••••••••••••" : undefined,
    ignoreFocusOut: true,
  });

  if (!key?.trim() || key.includes("•")) {
    // User dismissed or left the masked placeholder — nothing to do.
    return;
  }

  await secrets.store(SECRET_KEY, key.trim());
  void vscode.window.showInformationMessage(
    "BlackSwan: Gemini API key saved. Your key is encrypted at rest by VS Code.",
  );
}

/**
 * Fetch a natural-language explanation of a detected mathematical failure
 * from the Gemini API and display it in a side panel.
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
      "BlackSwan AI: No Gemini API key configured.",
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
      "BlackSwan AI: Gemini rate limit reached (15 req/min). " +
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
        const explanation = await _callGemini(apiKey, model, prompt);
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
 * Read the configured Gemini model name from VS Code settings.
 * Falls back to gemini-2.5-flash (15 RPM free tier).
 */
function _resolveModel(): string {
  const cfg = vscode.workspace.getConfiguration("blackswan");
  return cfg.get<string>("geminiModel") ?? "gemini-2.5-flash";
}

/**
 * Build the Gemini prompt from a shatter point.
 *
 * Safety constraint: only failure metadata is included — never the user's
 * source code.  The causal chain variables and line numbers are sufficient
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

/** Raw Gemini REST call. Throws on HTTP error or empty response. */
async function _callGemini(
  apiKey: string,
  model: string,
  prompt: string,
): Promise<string> {
  const url = `${GEMINI_ENDPOINT}/${model}:generateContent?key=${apiKey}`;

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: {
        maxOutputTokens: 1024,
        temperature: 0.2,
        thinkingConfig: { thinkingBudget: 0 },
      },
    }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new GeminiApiError(response.status, body);
  }

  const data = (await response.json()) as GeminiResponse;
  const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) {
    throw new Error("Gemini returned an empty response.");
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
    `*Explanation generated by Gemini. ` +
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
    nan_inf:                "NaN / Inf Detected",
    division_by_zero:       "Division Instability",
    non_psd_matrix:         "Non-PSD Matrix",
    ill_conditioned_matrix: "Ill-Conditioned Matrix",
    bounds_exceeded:        "Bounds Exceeded",
    division_instability:   "Division Instability",
    exploding_gradient:     "Exploding Gradient",
    regime_shift:           "Regime Shift",
    logical_invariant:      "Logical Invariant Violation",
  };
  return labels[type] ?? type;
}

function _handleApiError(err: unknown): void {
  if (err instanceof GeminiApiError) {
    if (err.status === 400 || err.status === 401) {
      void vscode.window.showErrorMessage(
        "BlackSwan AI: Invalid or expired API key. " +
          "Run 'BlackSwan: Set Gemini API Key' from the command palette to update it.",
      );
    } else if (err.status === 429) {
      void vscode.window.showWarningMessage(
        "BlackSwan AI: Gemini API quota exceeded. " +
          "You can still apply the deterministic fix via the lightbulb menu.",
      );
    } else {
      void vscode.window.showErrorMessage(
        `BlackSwan AI: Gemini API error ${err.status}. ${err.shortBody}`,
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

class GeminiApiError extends Error {
  readonly status: number;
  readonly shortBody: string;

  constructor(status: number, body: string) {
    super(`Gemini API error ${status}`);
    this.name = "GeminiApiError";
    this.status = status;
    this.shortBody = body.slice(0, 300);
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

// ---------------------------------------------------------------------------
// Gemini response shape (partial)
// ---------------------------------------------------------------------------

interface GeminiResponse {
  candidates?: Array<{
    content?: {
      parts?: Array<{ text?: string }>;
    };
  }>;
}
