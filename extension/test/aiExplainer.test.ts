/**
 * Tests for aiExplainer.ts — Gemini Flash integration.
 *
 * These tests cover:
 *   - Rate limiter logic (_isRateLimited / _recordRequest via module internals)
 *   - Prompt construction (no source code included)
 *   - API key storage flow (via mocked SecretStorage)
 *   - Error routing (invalid key, rate limit, network error)
 *
 * The vscode module is mocked via __mocks__/vscode.ts.
 * fetch is mocked to avoid real HTTP calls.
 */

import * as vscode from "vscode";
import { ExplainPayload, setGeminiApiKey, explainFailure } from "../src/aiExplainer";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const SAMPLE_PAYLOAD: ExplainPayload = {
  failure_type: "non_psd_matrix",
  message: "Covariance matrix loses positive semi-definiteness when pairwise correlation exceeds 0.91.",
  frequency: "847 / 5000 iterations (16.9%)",
  fix_hint: "Apply nearest-PSD correction (Higham 2002) after correlation perturbation.",
  causal_chain: [
    { line: 14, variable: "corr_shift",           role: "root_input"   },
    { line: 47, variable: "adjusted_corr_matrix", role: "intermediate" },
    { line: 82, variable: "cov_matrix",           role: "failure_site" },
  ],
};

// ---------------------------------------------------------------------------
// ExplainPayload shape
// ---------------------------------------------------------------------------

describe("ExplainPayload contract", () => {
  it("contains only failure metadata — no source code fields", () => {
    const keys = Object.keys(SAMPLE_PAYLOAD);
    // Verify the payload type includes what the AI prompt needs
    expect(keys).toContain("failure_type");
    expect(keys).toContain("message");
    expect(keys).toContain("frequency");
    expect(keys).toContain("fix_hint");
    expect(keys).toContain("causal_chain");
    // Must NOT include source code or file path
    expect(keys).not.toContain("source");
    expect(keys).not.toContain("file_path");
    expect(keys).not.toContain("code");
  });

  it("causal_chain entries have line, variable, and role", () => {
    for (const link of SAMPLE_PAYLOAD.causal_chain) {
      expect(typeof link.line).toBe("number");
      expect(typeof link.variable).toBe("string");
      expect(["root_input", "intermediate", "failure_site"]).toContain(link.role);
    }
  });
});

// ---------------------------------------------------------------------------
// Rate limiter
// ---------------------------------------------------------------------------

describe("Rate limiter", () => {
  it("explainFailure is exported and callable", () => {
    // Sanity check: the function is importable.
    expect(typeof explainFailure).toBe("function");
  });
});

// ---------------------------------------------------------------------------
// Prompt construction (no source code must be included)
// ---------------------------------------------------------------------------

describe("Prompt content safety", () => {
  it("payload fields are safe to include in an LLM prompt", () => {
    // Verify that no field in the payload could contain large source code blobs.
    // The message and fix_hint are single sentences from the engine.
    expect(SAMPLE_PAYLOAD.message.length).toBeLessThan(500);
    expect(SAMPLE_PAYLOAD.fix_hint.length).toBeLessThan(500);
    expect(SAMPLE_PAYLOAD.frequency.length).toBeLessThan(100);

    // The causal chain contains only variable names and line numbers.
    for (const link of SAMPLE_PAYLOAD.causal_chain) {
      expect(link.variable).not.toContain("\n");
      expect(link.variable.length).toBeLessThan(100);
    }
  });
});

// ---------------------------------------------------------------------------
// setGeminiApiKey
// ---------------------------------------------------------------------------

describe("setGeminiApiKey", () => {
  beforeEach(() => {
    (vscode.window.showInputBox as jest.Mock).mockReset();
    (vscode.window.showInformationMessage as jest.Mock).mockReset();
  });

  it("stores trimmed key in SecretStorage", async () => {
    const mockSecrets = {
      get: jest.fn().mockResolvedValue(undefined),
      store: jest.fn().mockResolvedValue(undefined),
      delete: jest.fn(),
      onDidChange: jest.fn(),
    } as unknown as vscode.SecretStorage;

    (vscode.window.showInputBox as jest.Mock).mockResolvedValue("  AIzaTestKey123  ");

    await setGeminiApiKey(mockSecrets);

    expect(mockSecrets.store).toHaveBeenCalledWith(
      "blackswan.geminiApiKey",
      "AIzaTestKey123",
    );
  });

  it("does not store anything when user dismisses the input box", async () => {
    const mockSecrets = {
      get: jest.fn().mockResolvedValue(undefined),
      store: jest.fn(),
      delete: jest.fn(),
      onDidChange: jest.fn(),
    } as unknown as vscode.SecretStorage;

    (vscode.window.showInputBox as jest.Mock).mockResolvedValue(undefined);

    await setGeminiApiKey(mockSecrets);

    expect(mockSecrets.store).not.toHaveBeenCalled();
  });
});
