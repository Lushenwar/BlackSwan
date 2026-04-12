/**
 * Integration tests for fixer.ts — spawns the real Python fixer subprocess.
 *
 * These tests verify the full TypeScript → Python → JSON round-trip.
 * They require:
 *   - Python installed with `blackswan[fixer]` (`pip install libcst>=1.1`)
 *   - The blackswan package installed in editable mode (`pip install -e core/`)
 *
 * Run from extension/:  npm test
 * The tests skip gracefully if Python / blackswan is unavailable.
 */

import * as path from "path";
import * as fs from "fs";
import * as os from "os";
import { execSync, spawn } from "child_process";

// ---------------------------------------------------------------------------
// Resolve Python executable (same logic as bridge.test.ts)
// ---------------------------------------------------------------------------

function resolvePython(): string {
  if (process.platform === "win32") {
    try {
      return execSync('py -3.11 -c "import sys; print(sys.executable)"', {
        encoding: "utf-8",
      }).trim();
    } catch {
      return "python";
    }
  }
  return "python3";
}

const PYTHON = resolvePython();
const REPO_ROOT = path.resolve(__dirname, "..", "..");
const CORE_DIR = path.join(REPO_ROOT, "core");

// ---------------------------------------------------------------------------
// Check prerequisites
// ---------------------------------------------------------------------------

function canRunFixer(): boolean {
  try {
    execSync(`"${PYTHON}" -c "import blackswan.fixer.guards; import libcst"`, {
      encoding: "utf-8",
      cwd: CORE_DIR,
    });
    return true;
  } catch {
    return false;
  }
}

const FIXER_AVAILABLE = canRunFixer();

// ---------------------------------------------------------------------------
// Subprocess helper
// ---------------------------------------------------------------------------

interface FixerJson {
  status: "ok" | "error" | "unsupported";
  line?: number;
  original?: string;
  replacement?: string;
  extra_lines?: string[];
  explanation?: string;
  message?: string;
}

function runFixer(
  filePath: string,
  line: number,
  failureType: string,
): Promise<FixerJson> {
  return new Promise((resolve, reject) => {
    const args = [
      "-m", "blackswan", "fix", filePath,
      "--line", String(line),
      "--type", failureType,
    ];
    const proc = spawn(PYTHON, args, {
      cwd: path.dirname(filePath),
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    proc.stdout.on("data", (c: Buffer) => { stdout += c; });
    proc.stderr.on("data", (c: Buffer) => { stderr += c; });

    proc.on("error", (err) => reject(new Error(`spawn failed: ${err.message}`)));
    proc.on("close", () => {
      try {
        resolve(JSON.parse(stdout) as FixerJson);
      } catch {
        reject(new Error(`Non-JSON output.\nstdout: ${stdout.slice(0, 300)}\nstderr: ${stderr.slice(0, 300)}`));
      }
    });
  });
}

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

let _tmpDir: string | null = null;

function getTmpDir(): string {
  if (!_tmpDir) {
    _tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "bsfix-"));
  }
  return _tmpDir;
}

function writeFixture(name: string, content: string): string {
  const p = path.join(getTmpDir(), name);
  fs.writeFileSync(p, content, "utf-8");
  return p;
}

afterAll(() => {
  if (_tmpDir) {
    fs.rmSync(_tmpDir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// Skip guard
// ---------------------------------------------------------------------------

const describeIfFixer = FIXER_AVAILABLE ? describe : describe.skip;

// ---------------------------------------------------------------------------
// Division guard integration
// ---------------------------------------------------------------------------

describeIfFixer("Fixer integration: division_instability", () => {
  const SRC = [
    "import numpy as np",
    "",
    "def compute_sharpe(returns, vol):",
    "    sharpe = returns / vol",
    "    return sharpe",
    "",
  ].join("\n");

  let result: FixerJson;

  beforeAll(async () => {
    const p = writeFixture("sharpe.py", SRC);
    result = await runFixer(p, 4, "division_instability");
  });

  it("status is ok", () => {
    expect(result.status).toBe("ok");
  });

  it("line is echoed back as 4", () => {
    expect(result.line).toBe(4);
  });

  it("replacement contains epsilon guard", () => {
    expect(result.replacement).toContain("1e-10");
  });

  it("replacement preserves 4-space indentation", () => {
    expect(result.replacement).toMatch(/^    /);
  });

  it("fixed source (with replacement applied) is syntactically valid Python", () => {
    // Apply the replacement to the full source and compile that.
    const lines = SRC.split("\n");
    const zeroLine = (result.line ?? 4) - 1;
    lines[zeroLine] = result.replacement ?? lines[zeroLine];
    const fixedSrc = lines.join("\n");

    const tmpFile = path.join(os.tmpdir(), "bsfix_syntax_check.py");
    fs.writeFileSync(tmpFile, fixedSrc, "utf-8");
    try {
      execSync(`"${PYTHON}" -m py_compile "${tmpFile}"`, {
        encoding: "utf-8",
        stdio: "pipe",
      });
    } finally {
      fs.rmSync(tmpFile, { force: true });
    }
    expect(true).toBe(true);
  });

  it("explanation references the denominator variable", () => {
    expect(result.explanation).toContain("vol");
  });
});

// ---------------------------------------------------------------------------
// PSD guard integration
// ---------------------------------------------------------------------------

describeIfFixer("Fixer integration: non_psd_matrix", () => {
  const SRC = [
    "import numpy as np",
    "",
    "def build_cov(returns):",
    "    cov_matrix = np.cov(returns.T)",
    "    return cov_matrix",
    "",
  ].join("\n");

  let result: FixerJson;

  beforeAll(async () => {
    const p = writeFixture("cov.py", SRC);
    result = await runFixer(p, 4, "non_psd_matrix");
  });

  it("status is ok", () => {
    expect(result.status).toBe("ok");
  });

  it("original line unchanged", () => {
    expect(result.replacement).toBe(result.original);
  });

  it("extra_lines has 2 guard lines", () => {
    expect(result.extra_lines).toHaveLength(2);
  });

  it("first extra line uses np.linalg.eigh", () => {
    expect(result.extra_lines?.[0]).toContain("np.linalg.eigh");
  });

  it("second extra line uses np.maximum", () => {
    expect(result.extra_lines?.[1]).toContain("np.maximum");
  });

  it("guard lines reference cov_matrix", () => {
    const extra = (result.extra_lines ?? []).join("\n");
    expect(extra).toContain("cov_matrix");
  });
});

// ---------------------------------------------------------------------------
// Condition number guard integration
// ---------------------------------------------------------------------------

describeIfFixer("Fixer integration: ill_conditioned_matrix", () => {
  const SRC = [
    "import numpy as np",
    "",
    "def min_var_weights(cov_matrix, mu):",
    "    weights = np.linalg.inv(cov_matrix) @ mu",
    "    return weights",
    "",
  ].join("\n");

  let result: FixerJson;

  beforeAll(async () => {
    const p = writeFixture("weights.py", SRC);
    result = await runFixer(p, 4, "ill_conditioned_matrix");
  });

  it("status is ok", () => {
    expect(result.status).toBe("ok");
  });

  it("replacement contains np.linalg.cond", () => {
    expect(result.replacement).toContain("np.linalg.cond");
  });

  it("replacement contains np.linalg.pinv", () => {
    expect(result.replacement).toContain("np.linalg.pinv");
  });

  it("replacement contains threshold 1e12", () => {
    expect(result.replacement).toContain("1e12");
  });

  it("original and replacement are different", () => {
    expect(result.replacement).not.toBe(result.original);
  });
});

// ---------------------------------------------------------------------------
// NaN/Inf guard integration
// ---------------------------------------------------------------------------

describeIfFixer("Fixer integration: nan_inf", () => {
  const SRC = [
    "import numpy as np",
    "",
    "def portfolio_pnl(weights, returns):",
    "    pnl = np.dot(weights, returns)",
    "    return pnl",
    "",
  ].join("\n");

  let result: FixerJson;

  beforeAll(async () => {
    const p = writeFixture("pnl.py", SRC);
    result = await runFixer(p, 4, "nan_inf");
  });

  it("status is ok", () => {
    expect(result.status).toBe("ok");
  });

  it("extra_lines includes nan_to_num call", () => {
    const extra = (result.extra_lines ?? []).join("\n");
    expect(extra).toContain("np.nan_to_num");
  });

  it("extra_lines handles posinf", () => {
    const extra = (result.extra_lines ?? []).join("\n");
    expect(extra).toContain("posinf=");
  });
});

// ---------------------------------------------------------------------------
// Error and unsupported cases integration
// ---------------------------------------------------------------------------

describeIfFixer("Fixer integration: error and unsupported cases", () => {
  it("returns error for out-of-range line", async () => {
    const src = "def f(x, y):\n    return x / y\n";
    const p = writeFixture("range_err.py", src);
    const result = await runFixer(p, 999, "division_instability");
    expect(result.status).toBe("error");
  });

  it("returns error when no division on target line", async () => {
    const src = "def f(x):\n    y = x + 1\n    return y\n";
    const p = writeFixture("no_div.py", src);
    const result = await runFixer(p, 2, "division_instability");
    expect(result.status).toBe("error");
  });

  it("returns unsupported for bounds_exceeded", async () => {
    const src = "def f(x):\n    return x\n";
    const p = writeFixture("unsupported.py", src);
    const result = await runFixer(p, 1, "bounds_exceeded");
    expect(result.status).toBe("unsupported");
  });

  it("exit code is 0 for ok fix", async () => {
    const src = "def f(x, y):\n    return x / y\n";
    const p = writeFixture("exit0.py", src);
    return new Promise<void>((resolve, reject) => {
      const proc = spawn(PYTHON, ["-m", "blackswan", "fix", p, "--line", "2", "--type", "division_instability"], {
        cwd: path.dirname(p),
      });
      proc.on("close", (code) => {
        try { expect(code).toBe(0); resolve(); }
        catch (e) { reject(e); }
      });
    });
  });
});
