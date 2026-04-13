"""
sarif.py — SARIF 2.1.0 serializer for BlackSwan results.

Converts the CLI response dict (built by cli._build_response) into a
SARIF 2.1.0 document suitable for GitHub Code Scanning / any SARIF-aware CI tool.

Usage (from CLI):
    python -m blackswan test model.py --scenario liquidity_crash --output sarif
    python -m blackswan test model.py --scenario liquidity_crash --output sarif --output-path results.sarif

Reference:
    https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html
    https://docs.github.com/en/code-security/code-scanning/integrating-with-code-scanning/sarif-support-for-code-scanning
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Rule catalogue — stable mapping of BlackSwan failure types to SARIF rule IDs
# ---------------------------------------------------------------------------

_RULE_META: dict[str, dict[str, str]] = {
    "nan_inf": {
        "id": "BS001",
        "name": "NaNInfDetected",
        "short": "NaN or Inf detected in computation",
        "full": (
            "A floating-point operation produced NaN (Not a Number) or Inf (Infinity). "
            "This indicates numerical overflow, division by zero, or an invalid operation "
            "such as sqrt of a negative number. In production financial models this silently "
            "propagates, corrupting downstream calculations without raising an exception."
        ),
        "tags": ["numerical", "floating-point"],
    },
    "division_by_zero": {
        "id": "BS002",
        "name": "DivisionInstability",
        "short": "Denominator approaches zero under stress",
        "full": (
            "A division operation has a denominator that approaches zero under the stress "
            "scenario. This is a pre-crash signal: the actual ZeroDivisionError may not fire "
            "in normal operation, but the result becomes numerically meaningless (very large "
            "magnitude) long before the denominator reaches exactly zero."
        ),
        "tags": ["numerical", "division"],
    },
    "division_instability": {
        "id": "BS002",
        "name": "DivisionInstability",
        "short": "Denominator approaches zero under stress",
        "full": (
            "A division operation has a denominator that approaches zero under the stress "
            "scenario. This is a pre-crash signal: the actual ZeroDivisionError may not fire "
            "in normal operation, but the result becomes numerically meaningless (very large "
            "magnitude) long before the denominator reaches exactly zero."
        ),
        "tags": ["numerical", "division"],
    },
    "non_psd_matrix": {
        "id": "BS003",
        "name": "NonPsdMatrix",
        "short": "Covariance or correlation matrix loses positive semi-definiteness",
        "full": (
            "A matrix that must be positive semi-definite (PSD) — typically a covariance or "
            "correlation matrix — acquires negative eigenvalues under stress perturbation. "
            "This violates a fundamental mathematical constraint: PSD matrices represent valid "
            "probability distributions over returns. A non-PSD covariance matrix makes VaR, "
            "portfolio optimisation, and risk decomposition undefined or nonsensical."
        ),
        "tags": ["numerical", "matrix", "finance"],
    },
    "ill_conditioned_matrix": {
        "id": "BS004",
        "name": "IllConditionedMatrix",
        "short": "Matrix condition number exceeds safe threshold before inversion",
        "full": (
            "The condition number of a matrix exceeds 1e12, meaning the matrix is nearly "
            "singular. Inverting an ill-conditioned matrix amplifies numerical errors by up to "
            "12 orders of magnitude. In Cholesky decomposition or portfolio optimisation, this "
            "produces weights or risk estimates that are entirely determined by floating-point "
            "rounding, not by the underlying financial model."
        ),
        "tags": ["numerical", "matrix", "linear-algebra"],
    },
    "bounds_exceeded": {
        "id": "BS005",
        "name": "BoundsExceeded",
        "short": "Output value exceeds plausible financial bounds",
        "full": (
            "A model output (portfolio weight, VaR estimate, Sharpe ratio, etc.) exceeded "
            "the configured bounds under stress. This is a model sanity check: even under "
            "extreme market conditions, certain quantities should remain within physically "
            "meaningful ranges. Bounds violations often indicate upstream instability."
        ),
        "tags": ["bounds", "finance"],
    },
    "exploding_gradient": {
        "id": "BS006",
        "name": "ExplodingGradient",
        "short": "Gradient magnitude explodes during iterative computation",
        "full": (
            "An iterative computation (gradient descent, Newton-Raphson, etc.) produces "
            "gradient magnitudes that grow without bound under stress. This prevents "
            "convergence and typically causes the final result to be NaN or Inf."
        ),
        "tags": ["numerical", "optimisation"],
    },
    "regime_shift": {
        "id": "BS007",
        "name": "RegimeShift",
        "short": "Statistical regime shift detected in output distribution",
        "full": (
            "The output distribution undergoes a sharp discontinuity under the stress scenario, "
            "indicating the model is operating near a regime boundary. Small parameter changes "
            "near this boundary produce disproportionately large output changes — a classic "
            "fragility signature in risk models."
        ),
        "tags": ["statistical", "finance"],
    },
    "logical_invariant": {
        "id": "BS008",
        "name": "LogicalInvariantViolation",
        "short": "A logical invariant or contract is violated under stress",
        "full": (
            "A logical constraint that must always hold (weights sum to 1, probabilities in "
            "[0,1], durations positive, etc.) is violated under stress. This indicates the "
            "model does not handle extreme inputs gracefully and may produce nonsensical "
            "results without any exception being raised."
        ),
        "tags": ["logic", "invariant"],
    },
}

# Fallback for unknown failure types
_FALLBACK_RULE: dict[str, str] = {
    "id": "BS999",
    "name": "UnknownFailure",
    "short": "Unknown failure type detected",
    "full": "BlackSwan detected a failure of an unrecognised type.",
    "tags": ["numerical"],
}


# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------

_SEVERITY_MAP: dict[str, str] = {
    "critical": "error",
    "warning": "warning",
    "info": "note",
    "note": "note",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def to_sarif(response: dict[str, Any], file_path: str | Path) -> dict[str, Any]:
    """
    Convert a BlackSwan CLI response dict to a SARIF 2.1.0 document.

    Args:
        response:  The dict returned by cli._build_response (or loaded from JSON).
        file_path: Absolute path to the analysed Python file.  Used to build
                   artifactLocation URIs relative to the repository root.

    Returns:
        A dict representing the SARIF document, ready for json.dumps().
    """
    file_path = Path(file_path).resolve()
    shatter_points = response.get("shatter_points", [])

    # Collect unique failure types to build the rules array.
    seen_rule_ids: dict[str, dict] = {}
    for sp in shatter_points:
        ft = sp.get("failure_type", "")
        meta = _RULE_META.get(ft, _FALLBACK_RULE)
        rid = meta["id"]
        if rid not in seen_rule_ids:
            seen_rule_ids[rid] = _build_rule(meta)

    rules = list(seen_rule_ids.values())

    # Resolve file URI relative to repo root (best-effort: walk up to find .git).
    repo_root = _find_repo_root(file_path) or file_path.parent
    rel_uri = _to_posix_uri(file_path, repo_root)

    rc = response.get("reproducibility_card") or {}
    version = rc.get("blackswan_version", "")

    results = [
        _build_result(sp, rel_uri)
        for sp in shatter_points
    ]

    return {
        "version": "2.1.0",
        "$schema": (
            "https://raw.githubusercontent.com/oasis-open.org/sarif-spec/"
            "master/Schemata/sarif-schema-2.1.0.json"
        ),
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "BlackSwan",
                        "version": version or "0.4.0",
                        "informationUri": "https://github.com/Lushenwar/BlackSwan",
                        "rules": rules,
                    }
                },
                "originalUriBaseIds": {
                    "%SRCROOT%": {
                        "uri": _dir_uri(repo_root),
                        "description": {
                            "text": "Repository root — set by BlackSwan at scan time."
                        },
                    }
                },
                "artifacts": [
                    {
                        "location": {
                            "uri": rel_uri,
                            "uriBaseId": "%SRCROOT%",
                        }
                    }
                ],
                "results": results,
                "properties": {
                    "scenario": response.get("scenario_card", {}).get("name", ""),
                    "seed": response.get("scenario_card", {}).get("seed"),
                    "iterations_completed": response.get("iterations_completed", 0),
                    "failure_rate": response.get("summary", {}).get("failure_rate", 0.0),
                    "runtime_ms": response.get("runtime_ms", 0),
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def _build_rule(meta: dict[str, str]) -> dict[str, Any]:
    return {
        "id": meta["id"],
        "name": meta["name"],
        "shortDescription": {"text": meta["short"]},
        "fullDescription": {"text": meta["full"]},
        "helpUri": (
            "https://github.com/Lushenwar/BlackSwan/blob/main/docs/SCENARIOS.md"
        ),
        "properties": {
            "tags": list(meta.get("tags", [])),
            "precision": "high",
            "problem.severity": "error",
        },
        "defaultConfiguration": {
            "level": "error",
        },
    }


def _build_result(sp: dict[str, Any], rel_uri: str) -> dict[str, Any]:
    ft = sp.get("failure_type", "")
    meta = _RULE_META.get(ft, _FALLBACK_RULE)
    rule_id = meta["id"]
    severity = sp.get("severity", "critical")
    level = _SEVERITY_MAP.get(severity, "error")
    line = sp.get("line") or 1

    result: dict[str, Any] = {
        "ruleId": rule_id,
        "level": level,
        "message": {
            "text": sp.get("message", ""),
        },
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": rel_uri,
                        "uriBaseId": "%SRCROOT%",
                    },
                    "region": {
                        "startLine": line,
                    },
                }
            }
        ],
        "properties": {
            "blackswan/severity": severity,
            "blackswan/failure_type": ft,
            "blackswan/frequency": sp.get("frequency", ""),
            "blackswan/confidence": sp.get("confidence", ""),
            "blackswan/fix_hint": sp.get("fix_hint", ""),
        },
    }

    # Attach causal chain as related locations so GitHub shows them as clickable links.
    causal = sp.get("causal_chain", [])
    if causal:
        related: list[dict] = []
        for link in causal:
            link_line = link.get("line")
            if link_line is None:
                continue
            role = link.get("role", "intermediate")
            var = link.get("variable", "?")
            related.append({
                "message": {"text": f"{var} ({role})"},
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": rel_uri,
                        "uriBaseId": "%SRCROOT%",
                    },
                    "region": {"startLine": link_line},
                },
            })
        if related:
            result["relatedLocations"] = related

    # Fix hint → suppression suggestion (informational, not a suppression).
    fix_hint = sp.get("fix_hint", "")
    if fix_hint:
        result["fixes"] = [
            {
                "description": {"text": fix_hint},
                "artifactChanges": [],
            }
        ]

    return result


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def _find_repo_root(start: Path) -> Path | None:
    """Walk up from start until we find a .git directory."""
    current = start if start.is_dir() else start.parent
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _to_posix_uri(file_path: Path, repo_root: Path) -> str:
    """
    Return the POSIX-style path of file_path relative to repo_root.

    SARIF requires forward slashes for URIs even on Windows.
    """
    try:
        return file_path.relative_to(repo_root).as_posix()
    except ValueError:
        # file_path is outside the repo root — use absolute URI.
        return file_path.as_posix()


def _dir_uri(directory: Path) -> str:
    """
    Convert a directory path to a file:// URI string with a trailing slash.

    The trailing slash is required by the SARIF spec for directory URIs.
    """
    posix = directory.as_posix()
    if not posix.startswith("/"):
        # Windows absolute path (e.g. C:/Users/...) — prepend slash.
        posix = "/" + posix
    uri = f"file://{posix}"
    if not uri.endswith("/"):
        uri += "/"
    return uri
