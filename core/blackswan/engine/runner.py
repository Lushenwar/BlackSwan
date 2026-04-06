"""
Simulation runner for BlackSwan stress tests.

StressRunner is the core loop: it perturbs inputs, calls the target function,
runs all detectors on the output, and collects findings across N iterations.

Scenario protocol (duck-typed — Step 1C will provide YAML-backed implementations):
    scenario.iterations: int
    scenario.apply(inputs: dict, rng: np.random.Generator) -> dict
        Returns a new dict of perturbed inputs for one iteration.
        Must use rng exclusively for any randomness so that results are
        fully reproducible given the same seed.
"""

import time
import traceback as tb
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..detectors.base import FailureDetector, Finding


@dataclass
class RunResult:
    """
    Output of a single StressRunner.run() call.

    findings contains every Finding produced across all iterations,
    in the order they were detected. The attribution layer (Phase 1D)
    will later enrich these with line numbers and causal chains.
    """

    iterations_completed: int
    findings: list[Finding]
    runtime_ms: int
    seed: int


class StressRunner:
    """
    Runs a target function N times under perturbed inputs and collects failures.

    Each run is fully deterministic: given the same fn, base_inputs, scenario,
    detectors, and seed, run() will always produce the same findings list.
    No global state is mutated. The numpy Generator is created fresh from the
    seed on every run() call.
    """

    def __init__(
        self,
        fn: Callable,
        base_inputs: dict,
        scenario: Any,
        detectors: list[FailureDetector],
        seed: int,
    ) -> None:
        """
        Args:
            fn:          Target function to stress-test. Called as fn(**perturbed_inputs).
            base_inputs: Baseline input dict passed to scenario.apply() each iteration.
            scenario:    Duck-typed scenario object. Must have .iterations (int) and
                         .apply(inputs, rng) -> dict.
            detectors:   List of FailureDetector instances to run on every output.
            seed:        Integer seed for np.random.default_rng(). Determines the
                         complete perturbation sequence.
        """
        self.fn = fn
        self.base_inputs = base_inputs
        self.scenario = scenario
        self.detectors = detectors
        self.seed = seed

    def run(self) -> RunResult:
        """
        Execute the full simulation loop and return aggregated results.

        The RNG is created fresh here from self.seed so that calling run()
        twice on the same instance produces identical results.
        """
        rng = np.random.default_rng(self.seed)
        findings: list[Finding] = []
        start = time.monotonic()

        for i in range(self.scenario.iterations):
            perturbed = self.scenario.apply(self.base_inputs, rng)

            try:
                output = self.fn(**perturbed)
            except Exception as exc:
                # Target function raised — treat as a critical failure.
                # Capture traceback frames now, before the exception context
                # is cleared, so TracebackResolver can identify the source line.
                frames = [
                    (frame.filename, frame.lineno)
                    for frame in tb.extract_tb(exc.__traceback__)
                ]
                findings.append(Finding(
                    failure_type="nan_inf",
                    severity="critical",
                    message=f"{type(exc).__name__}: {exc}",
                    iteration=i,
                    exc_frames=frames,
                ))
                continue

            for detector in self.detectors:
                finding = detector.check(
                    inputs=perturbed,
                    output=output,
                    iteration=i,
                )
                if finding is not None:
                    findings.append(finding)

        runtime_ms = int((time.monotonic() - start) * 1000)

        return RunResult(
            iterations_completed=self.scenario.iterations,
            findings=findings,
            runtime_ms=runtime_ms,
            seed=self.seed,
        )
