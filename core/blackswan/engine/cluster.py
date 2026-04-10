"""
Failure clustering — turns raw findings into actionable Root Cause Buckets.

Instead of returning 800 raw findings to the user, BlackSwan groups structurally
identical failures and reports the unique root causes with one minimal
representative each.

Clustering key: (failure_type, line_number, exc_type_name)
Two findings are in the same bucket iff they share the same key.

RootCauseBucket is the primary output unit of the engine after clustering.
It replaces the raw findings list in RunResult.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..detectors.base import Finding, TriggerDisclosure
from .replay import Attribution, CausalLink


# ---------------------------------------------------------------------------
# RootCauseBucket
# ---------------------------------------------------------------------------

@dataclass
class RootCauseBucket:
    """
    One unique failure mode observed across all simulation iterations.

    Carries the occurrence count, rate, a minimal representative example,
    and full attribution from the Slow-Path replay.
    """
    bucket_id: str                          # "rcb_001", "rcb_002", ...
    failure_type: str
    line: int | None
    severity: str
    total_occurrences: int
    occurrence_rate: float                  # total_occurrences / total_iterations
    representative_finding: Finding         # highest-severity example
    message: str
    fix_hint: str
    causal_chain: list[CausalLink]
    attribution: Attribution | None         # None when mode=fast (no Slow-Path)
    confidence: str                         # "high" | "medium" | "low" | "unverified"
    sample_inputs: list[dict[str, Any]]     # up to 3 distinct failing input dicts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cluster_findings(
    findings: list[Finding],
    total_iterations: int,
    attributions: dict[int, Attribution] | None = None,
    failing_inputs: dict[int, dict[str, Any]] | None = None,
) -> list[RootCauseBucket]:
    """
    Group findings into Root Cause Buckets sorted by occurrence rate descending.

    Args:
        findings:         All raw findings from Fast-Path sweep.
        total_iterations: Total executed iteration count (for rate calculation).
        attributions:     Optional dict mapping iteration → Attribution from Slow-Path.
        failing_inputs:   Optional dict mapping iteration → perturbed inputs dict.

    Returns:
        List of RootCauseBucket ordered by total_occurrences descending.
    """
    if not findings:
        return []

    attributions = attributions or {}
    failing_inputs = failing_inputs or {}

    # Group by clustering key
    buckets: dict[tuple, list[Finding]] = defaultdict(list)
    for f in findings:
        key = _clustering_key(f)
        buckets[key].append(f)

    result: list[RootCauseBucket] = []
    for i, (key, group) in enumerate(
        sorted(buckets.items(), key=lambda x: -len(x[1]))
    ):
        representative = _pick_representative(group)
        attribution = _best_attribution(group, attributions)
        samples = _collect_sample_inputs(group, failing_inputs, max_samples=3)

        line = attribution.failure_line if attribution else representative.line
        confidence = attribution.confidence if attribution else "unverified"
        causal_chain = attribution.causal_chain if attribution else []

        result.append(RootCauseBucket(
            bucket_id=f"rcb_{i + 1:03d}",
            failure_type=representative.failure_type,
            line=line,
            severity=representative.severity,
            total_occurrences=len(group),
            occurrence_rate=len(group) / max(total_iterations, 1),
            representative_finding=representative,
            message=_build_message(representative, len(group), total_iterations),
            fix_hint=_fix_hint(representative.failure_type),
            causal_chain=causal_chain,
            attribution=attribution,
            confidence=confidence,
            sample_inputs=samples,
        ))

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _clustering_key(finding: Finding) -> tuple:
    """
    Two findings are the same root cause iff they share:
    (failure_type, line_number_or_None, exc_class_from_first_frame)
    """
    exc_type = None
    if finding.exc_frames:
        # Use the innermost frame filename to distinguish cross-file failures
        exc_type = finding.exc_frames[-1][0] if finding.exc_frames else None
    return (finding.failure_type, finding.line, exc_type)


def _pick_representative(group: list[Finding]) -> Finding:
    """
    Choose the most informative finding from a bucket.
    Prefers: critical > warning > info; then longest exc_frames.
    """
    _sev_order = {"critical": 0, "warning": 1, "info": 2}
    return min(
        group,
        key=lambda f: (_sev_order.get(f.severity, 9), -len(f.exc_frames)),
    )


def _best_attribution(
    group: list[Finding],
    attributions: dict[int, Attribution],
) -> Attribution | None:
    """
    Return the attribution (from Slow-Path replay) for the most severe
    finding in this bucket, if available.
    """
    _sev_order = {"critical": 0, "warning": 1, "info": 2}
    sorted_group = sorted(group, key=lambda f: _sev_order.get(f.severity, 9))
    for finding in sorted_group:
        if finding.iteration in attributions:
            return attributions[finding.iteration]
    return None


def _collect_sample_inputs(
    group: list[Finding],
    failing_inputs: dict[int, dict[str, Any]],
    max_samples: int,
) -> list[dict[str, Any]]:
    """Collect up to max_samples distinct input dicts from the bucket."""
    samples: list[dict[str, Any]] = []
    seen_iters: set[int] = set()
    for finding in group:
        if len(samples) >= max_samples:
            break
        if finding.iteration in failing_inputs and finding.iteration not in seen_iters:
            samples.append(failing_inputs[finding.iteration])
            seen_iters.add(finding.iteration)
    return samples


def _build_message(finding: Finding, count: int, total: int) -> str:
    rate_pct = count / max(total, 1) * 100
    return (
        f"{finding.message} "
        f"[{count}/{total} iterations, {rate_pct:.1f}%]"
    )


_FIX_HINTS: dict[str, str] = {
    "non_psd_matrix": (
        "Apply nearest-PSD correction (e.g. Higham 2002) after correlation perturbation, "
        "or clamp eigenvalues to a small positive epsilon before use."
    ),
    "ill_conditioned_matrix": (
        "Add Tikhonov regularisation (ridge) before matrix inversion, "
        "or use np.linalg.lstsq instead of np.linalg.inv."
    ),
    "nan_inf": (
        "Trace the first NaN/Inf back to its source using np.seterr(all='raise'). "
        "Common causes: division by zero, overflow in exponentiation, log of zero."
    ),
    "division_instability": (
        "Guard denominators with a minimum absolute value (epsilon clamp) "
        "before dividing, e.g. denominator = max(abs(denom), 1e-10) * sign(denom)."
    ),
    "bounds_exceeded": (
        "Review the plausible range assumptions in your scenario YAML. "
        "If the bounds are correct, the function needs input validation or output clamping."
    ),
    "exploding_gradient": (
        "Check for recursive or iterative accumulation without a convergence check. "
        "Add gradient clipping or a stability bound on the output."
    ),
    "regime_shift": (
        "The output distribution changed structure across iterations. "
        "Investigate whether a threshold or conditional branch is causing discontinuous behaviour."
    ),
    "logical_invariant": (
        "An expected mathematical relationship was violated under stress. "
        "Add a runtime assertion with a repair step (e.g. re-normalise weights after perturbation)."
    ),
}


def _fix_hint(failure_type: str) -> str:
    return _FIX_HINTS.get(
        failure_type,
        "Review the flagged line and its upstream inputs for numerical instability.",
    )
