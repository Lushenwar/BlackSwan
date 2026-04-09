"""
Scenario registry: loads YAML preset files and exposes them as Scenario objects.

Public API:
    load_scenario(name)  → Scenario
    list_scenarios()     → list[str]
    _load_from_path(path) → Scenario   (exported for error-path testing)

Scenario satisfies the StressRunner duck-typed protocol:
    scenario.iterations         int
    scenario.apply(inputs, rng) → dict
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..engine.perturbation import Perturbation, apply_perturbations
from ..engine.validator import PlausibilityConstraint, PlausibilityValidator

PRESETS_DIR = Path(__file__).parent / "presets"

# All perturbation type strings the registry accepts.
# "nan_injection" and "truncation" are structural types for missing_data;
# apply_perturbations will raise at runtime if asked to execute them —
# that is intentional until the full type set is implemented in Step 1C.
_VALID_TYPES = frozenset({
    "multiplicative",
    "additive",
    "nan_injection",
    "truncation",
})


@dataclass
class Scenario:
    """
    A loaded, validated stress scenario ready to be used by StressRunner.

    The .iterations attribute is an alias for default_iterations so that
    StressRunner can use this directly without any adapter.
    """

    name: str
    display_name: str
    description: str
    perturbations: list[Perturbation]
    default_iterations: int
    default_seed: int
    global_constraints: list[PlausibilityConstraint] = field(default_factory=list)

    @property
    def iterations(self) -> int:
        """StressRunner protocol: number of iterations to run."""
        return self.default_iterations

    def apply(self, inputs: dict, rng: Any) -> dict:
        """StressRunner protocol: return perturbed copy of inputs for one iteration."""
        return apply_perturbations(inputs, self.perturbations, rng)

    def make_validator(self) -> PlausibilityValidator:
        """Return a PlausibilityValidator built from this scenario's global_constraints."""
        return PlausibilityValidator(self.global_constraints)


# ---------------------------------------------------------------------------
# Public loading functions
# ---------------------------------------------------------------------------

def load_scenario(name: str) -> Scenario:
    """
    Load a preset scenario by name.

    Args:
        name: Stem of the YAML file (e.g. "liquidity_crash").

    Raises:
        FileNotFoundError: No preset with that name exists.
        ValueError:        YAML structure is invalid.
    """
    path = PRESETS_DIR / f"{name}.yaml"
    if not path.exists():
        available = list_scenarios()
        raise FileNotFoundError(
            f"no preset scenario named {name!r}. "
            f"Available presets: {available}"
        )
    return _load_from_path(path)


def list_scenarios() -> list[str]:
    """Return sorted list of all available preset scenario names."""
    return sorted(p.stem for p in PRESETS_DIR.glob("*.yaml"))


def _load_from_path(path: Path) -> Scenario:
    """
    Load and parse a scenario YAML from an explicit path.

    Exported so tests can point at temporary bad YAML files for error-path
    testing without touching the presets directory.

    Raises:
        ValueError: YAML is structurally invalid.
    """
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"Scenario YAML must be a mapping, got {type(data).__name__}")

    return _parse_scenario(data)


# ---------------------------------------------------------------------------
# Internal parsers
# ---------------------------------------------------------------------------

def _parse_scenario(data: dict) -> Scenario:
    _require_keys(data, {"name", "display_name", "description", "defaults", "perturbations"},
                  context="scenario")

    defaults = data["defaults"]
    if not isinstance(defaults, dict):
        raise ValueError("Scenario 'defaults' must be a mapping")
    _require_keys(defaults, {"iterations", "seed"}, context="defaults")

    raw_perturbs = data["perturbations"]
    if not isinstance(raw_perturbs, list) or len(raw_perturbs) == 0:
        raise ValueError("Scenario 'perturbations' must be a non-empty list")

    perturbations = [
        _parse_perturbation(p, idx) for idx, p in enumerate(raw_perturbs)
    ]

    raw_constraints = data.get("global_constraints", [])
    global_constraints = []
    for idx, c in enumerate(raw_constraints):
        if not isinstance(c, dict) or "target" not in c:
            raise ValueError(f"global_constraints[{idx}] must have a 'target' key")
        global_constraints.append(PlausibilityConstraint(
            target=str(c["target"]),
            min_value=float(c["min_value"]) if "min_value" in c else None,
            max_value=float(c["max_value"]) if "max_value" in c else None,
        ))

    return Scenario(
        name=str(data["name"]),
        display_name=str(data["display_name"]),
        description=str(data["description"]).strip(),
        perturbations=perturbations,
        default_iterations=int(defaults["iterations"]),
        default_seed=int(defaults["seed"]),
        global_constraints=global_constraints,
    )


def _parse_perturbation(raw: dict, index: int) -> Perturbation:
    if not isinstance(raw, dict):
        raise ValueError(f"Perturbation[{index}] must be a mapping, got {type(raw).__name__}")

    _require_keys(raw, {"target", "type", "distribution", "params"},
                  context=f"perturbation[{index}]")

    ptype = raw["type"]
    if ptype not in _VALID_TYPES:
        raise ValueError(
            f"Perturbation[{index}] has unknown type {ptype!r}. "
            f"Expected one of {sorted(_VALID_TYPES)}"
        )

    # Map optional constraints block → Perturbation.clamp tuple
    clamp: tuple[float | None, float | None] | None = None
    if "constraints" in raw:
        c = raw["constraints"]
        lo = float(c["clip_min"]) if "clip_min" in c else None
        hi = float(c["clip_max"]) if "clip_max" in c else None
        clamp = (lo, hi)

    return Perturbation(
        target=str(raw["target"]),
        type=ptype,
        distribution=str(raw["distribution"]),
        params=dict(raw["params"]),
        clamp=clamp,
    )


def _require_keys(mapping: dict, required: set[str], context: str) -> None:
    missing = required - set(mapping)
    if missing:
        raise ValueError(
            f"Scenario YAML {context} is missing required fields: {sorted(missing)}"
        )
