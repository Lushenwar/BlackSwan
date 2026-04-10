"""
ReproducibilityCard — captures the exact conditions needed to reproduce a run.

Every BlackSwan response includes a reproducibility_card that records the
engine version, Python/NumPy versions, seed, scenario fingerprint, and the
CLI command needed to replay the run identically.

Design goal: a user sharing a reproducibility_card with a colleague should
be able to get byte-for-byte identical findings by running the replay_command
on the same code. The card makes hidden variables (platform, library versions)
explicit so discrepancies are diagnosable.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ReproducibilityCard:
    """
    Full provenance record for a single BlackSwan run.

    Immutable once created. Use to_dict() for JSON serialisation.
    """
    blackswan_version: str
    python_version: str
    numpy_version: str
    platform: str
    seed: int
    scenario_name: str
    scenario_hash: str          # SHA-256[:12] of serialised scenario parameters
    mode: str                   # "fast" | "full" | "adversarial"
    iterations_requested: int
    iterations_executed: int
    iterations_skipped: int
    budget_exhausted: bool
    budget_reason: str | None
    timestamp_utc: str          # ISO 8601 UTC timestamp
    reproducible: bool          # False when budget was exhausted (run cut short)
    replay_command: str         # CLI command to reproduce this exact run

    def to_dict(self) -> dict[str, Any]:
        return {
            "blackswan_version": self.blackswan_version,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "platform": self.platform,
            "seed": self.seed,
            "scenario_name": self.scenario_name,
            "scenario_hash": self.scenario_hash,
            "mode": self.mode,
            "iterations_requested": self.iterations_requested,
            "iterations_executed": self.iterations_executed,
            "iterations_skipped": self.iterations_skipped,
            "budget_exhausted": self.budget_exhausted,
            "budget_reason": self.budget_reason,
            "timestamp_utc": self.timestamp_utc,
            "reproducible": self.reproducible,
            "replay_command": self.replay_command,
        }


def build_reproducibility_card(
    scenario: Any,
    seed: int,
    mode: str,
    iterations_requested: int,
    iterations_executed: int,
    budget_exhausted: bool,
    budget_reason: str | None,
    file_path: Path,
) -> ReproducibilityCard:
    """
    Build a ReproducibilityCard for the completed run.

    Args:
        scenario:             The scenario (or _IterationOverride wrapper) used.
        seed:                 RNG seed used for this run.
        mode:                 Execution mode: "fast" | "full" | "adversarial".
        iterations_requested: Total iterations requested (scenario + CLI override).
        iterations_executed:  Iterations actually executed (may be < requested if budget hit).
        budget_exhausted:     True if a budget limit stopped the run early.
        budget_reason:        "max_runtime_sec" | "max_iterations" | None.
        file_path:            Absolute path to the target Python file.
    """
    version = _blackswan_version()
    scenario_name = getattr(scenario, "name", "unknown")
    scenario_hash = _scenario_hash(scenario)

    replay_command = _build_replay_command(
        file_path=file_path,
        scenario_name=scenario_name,
        seed=seed,
        iterations=iterations_requested,
        mode=mode,
    )

    return ReproducibilityCard(
        blackswan_version=version,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        numpy_version=str(np.__version__),
        platform=platform.platform(aliased=True),
        seed=seed,
        scenario_name=scenario_name,
        scenario_hash=scenario_hash,
        mode=mode,
        iterations_requested=iterations_requested,
        iterations_executed=iterations_executed,
        iterations_skipped=max(0, iterations_requested - iterations_executed),
        budget_exhausted=budget_exhausted,
        budget_reason=budget_reason,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        reproducible=not budget_exhausted,
        replay_command=replay_command,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _blackswan_version() -> str:
    """Return blackswan package version, or 'unknown' if not installed."""
    try:
        from importlib.metadata import version
        return version("blackswan")
    except Exception:
        try:
            from importlib.metadata import version
            return version("blackswan-core")
        except Exception:
            return "unknown"


def _scenario_hash(scenario: Any) -> str:
    """
    Compute a short SHA-256 fingerprint of scenario parameters.
    Used to detect if two runs used the same effective scenario config.
    """
    try:
        parts: dict[str, Any] = {
            "name": getattr(scenario, "name", "unknown"),
            "perturbations": [],
        }
        for p in getattr(scenario, "perturbations", []):
            parts["perturbations"].append({
                "target": getattr(p, "target", ""),
                "type": getattr(p, "type", ""),
                "distribution": getattr(p, "distribution", ""),
                "params": getattr(p, "params", {}),
            })
        encoded = json.dumps(parts, sort_keys=True, default=str).encode()
        return hashlib.sha256(encoded).hexdigest()[:12]
    except Exception:
        return "unknown"


def _build_replay_command(
    file_path: Path,
    scenario_name: str,
    seed: int,
    iterations: int,
    mode: str,
) -> str:
    """Construct the CLI command that reproduces this run exactly."""
    cmd = (
        f"python -m blackswan test {file_path} "
        f"--scenario {scenario_name} "
        f"--seed {seed} "
        f"--iterations {iterations}"
    )
    if mode in ("fast", "full"):
        cmd += f" --mode {mode}"
    return cmd
