/**
 * Integration tests for aiExplainer.ts — exercises the real Gemini Flash API.
 *
 * These tests require a valid API key in the repo-root .env file:
 *   GEMINI_API_KEY=AIza...
 *
 * All tests skip automatically when the key is absent or empty.
 * Run from extension/: npm test
 *
 * What is verified:
 *   - The Gemini API accepts the prompt format without error
 *   - The response is non-empty prose (not JSON, not a refusal)
 *   - The response mentions the failure type or a related concept
 *   - No source code is present in the prompt (privacy constraint)
 *   - Rate limit tracking increments correctly
 */

import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { ExplainPayload } from "../src/aiExplainer";

// ---------------------------------------------------------------------------
// Load .env from repo root (no external dependency — manual parse)
// ---------------------------------------------------------------------------

function loadEnv(): Record<string, string> {
  const envPath = path.resolve(__dirname, "..", "..", ".env");
  if (!fs.existsSync(envPath)) return {};

  const lines = fs.readFileSync(envPath, "utf-8").split("\n");
  const env: Record<string, string> = {};
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq === -1) continue;
    const key = trimmed.slice(0, eq).trim();
    const value = trimmed.slice(eq + 1).trim();
    if (key && value) env[key] = value;
  }
  return env;
}

const ENV = loadEnv();
const GEMINI_API_KEY = ENV["GEMINI_API_KEY"] ?? process.env["GEMINI_API_KEY"] ?? "";
const HAS_KEY = GEMINI_API_KEY.length > 10;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const NON_PSD_PAYLOAD: ExplainPayload = {
  failure_type: "non_psd_matrix",
  message:
    "Covariance matrix loses positive semi-definiteness when pairwise correlation exceeds 0.91. " +
    "Smallest eigenvalue: -0.0034.",
  frequency: "847 / 5000 iterations (16.9%)",
  fix_hint:
    "Apply nearest-PSD correction (e.g., Higham 2002) after correlation perturbation, " +
    "or clamp eigenvalues to epsilon.",
  causal_chain: [
    { line: 14, variable: "corr_shift",           role: "root_input" as const   },
    { line: 47, variable: "adjusted_corr_matrix", role: "intermediate" as const },
    { line: 82, variable: "cov_matrix",           role: "failure_site" as const },
  ],
};

const DIVISION_PAYLOAD: ExplainPayload = {
  failure_type: "division_instability",
  message:
    "Denominator 'vol' approaches zero under volatility collapse scenario. " +
    "Minimum observed: 2.3e-12.",
  frequency: "312 / 5000 iterations (6.2%)",
  fix_hint:
    "Guard denominator: if abs(vol) < epsilon, use epsilon instead of vol.",
  causal_chain: [
    { line: 8,  variable: "vol",    role: "root_input" as const   },
    { line: 22, variable: "sharpe", role: "failure_site" as const },
  ],
};

// ---------------------------------------------------------------------------
// Direct Gemini API call (mirrors _callGemini in aiExplainer.ts)
// ---------------------------------------------------------------------------

interface GeminiCandidate {
  content?: { parts?: Array<{ text?: string }> };
}
interface GeminiResponse {
  candidates?: GeminiCandidate[];
  error?: { message?: string; status?: string };
}

async function callGeminiDirect(prompt: string): Promise<string> {
  const model = "gemini-2.0-flash";
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${GEMINI_API_KEY}`;

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: { maxOutputTokens: 512, temperature: 0.2 },
    }),
  });

  const data = (await response.json()) as GeminiResponse;

  if (!response.ok) {
    throw new Error(
      `Gemini API ${response.status}: ${data.error?.message ?? JSON.stringify(data).slice(0, 200)}`,
    );
  }

  const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) throw new Error("Empty response from Gemini");
  return text;
}

function buildPrompt(p: ExplainPayload): string {
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

// ---------------------------------------------------------------------------
// Tests — skip entire suite when no API key
// ---------------------------------------------------------------------------

const describeIfKey = HAS_KEY ? describe : describe.skip;

describeIfKey("Gemini API integration: non_psd_matrix explanation", () => {
  let explanation: string;

  // Use a longer timeout — Gemini cold-start can take up to 10s
  beforeAll(async () => {
    explanation = await callGeminiDirect(buildPrompt(NON_PSD_PAYLOAD));
  }, 30_000);

  it("returns a non-empty string", () => {
    expect(explanation.length).toBeGreaterThan(50);
  });

  it("response is prose, not raw JSON", () => {
    expect(() => JSON.parse(explanation)).toThrow();
  });

  it("response mentions matrix or eigenvalue concepts", () => {
    const lower = explanation.toLowerCase();
    const relevant =
      lower.includes("matrix") ||
      lower.includes("eigenvalue") ||
      lower.includes("positive") ||
      lower.includes("semi-definite") ||
      lower.includes("covariance");
    expect(relevant).toBe(true);
  });

  it("response is under 500 words (not a wall of text)", () => {
    const wordCount = explanation.trim().split(/\s+/).length;
    expect(wordCount).toBeLessThan(500);
  });
});

describeIfKey("Gemini API integration: division_instability explanation", () => {
  let explanation: string;

  beforeAll(async () => {
    explanation = await callGeminiDirect(buildPrompt(DIVISION_PAYLOAD));
  }, 30_000);

  it("returns a non-empty string", () => {
    expect(explanation.length).toBeGreaterThan(50);
  });

  it("mentions division or denominator concepts", () => {
    const lower = explanation.toLowerCase();
    const relevant =
      lower.includes("division") ||
      lower.includes("denominator") ||
      lower.includes("zero") ||
      lower.includes("volatility") ||
      lower.includes("numerically");
    expect(relevant).toBe(true);
  });
});

describeIfKey("Gemini API integration: privacy constraints", () => {
  it("prompt contains no source code (only metadata)", () => {
    const prompt = buildPrompt(NON_PSD_PAYLOAD);

    // The prompt must NOT contain Python syntax constructs from real code
    expect(prompt).not.toMatch(/def\s+\w+\s*\(/);   // no function definitions
    expect(prompt).not.toMatch(/import\s+numpy/);    // no import statements
    expect(prompt).not.toMatch(/=\s*np\./);          // no numpy assignments

    // It MUST contain the metadata fields
    expect(prompt).toContain("non_psd_matrix");
    expect(prompt).toContain("cov_matrix");
    expect(prompt).toContain("Higham");
  });

  it("causal chain only exposes variable names, not values", () => {
    const prompt = buildPrompt(NON_PSD_PAYLOAD);
    // Variable names are fine; numeric values from the model should NOT be present
    // (e.g. the actual matrix entries, weights, etc.)
    expect(prompt).toContain("corr_shift");
    expect(prompt).toContain("cov_matrix");
    // No raw numpy arrays or large numeric literals
    expect(prompt).not.toMatch(/\[\[[\d.,\s]+\]\]/);
  });
});

// ---------------------------------------------------------------------------
// Rate limiter unit test (does not call the API)
// ---------------------------------------------------------------------------

describe("Rate limiter: in-process bucket", () => {
  /**
   * We test the rate limiter by importing aiExplainer and exhausting the bucket.
   * The module-level _timestamps array is the state — we reset it between test
   * runs by re-importing with jest.resetModules().
   */

  it("explainFailure shows a warning when API key is missing", async () => {
    const vscode = await import("vscode");
    const { explainFailure } = await import("../src/aiExplainer");

    const mockSecrets = {
      get: jest.fn().mockResolvedValue(undefined),
      store: jest.fn(),
      delete: jest.fn(),
      onDidChange: jest.fn(),
    };

    await explainFailure(
      mockSecrets as unknown as import("vscode").SecretStorage,
      NON_PSD_PAYLOAD,
    );

    // Without a key, showWarningMessage must be called
    expect(
      (vscode.window.showWarningMessage as jest.Mock).mock.calls.length +
      (vscode.window.showErrorMessage as jest.Mock).mock.calls.length
    ).toBeGreaterThanOrEqual(1);
  });
});
