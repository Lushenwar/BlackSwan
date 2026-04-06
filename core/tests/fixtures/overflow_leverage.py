"""
Test fixture: compounded leverage that overflows float64 at extreme multipliers.

NaNInfDetector should fire when leverage is very high (e.g. > 1e150 per step).
Silent at realistic leverage levels (1x – 5x).
"""
import numpy as np


def compute_compounded_leverage(base_return=0.01, leverage=2.0, steps=252):
    """
    Compound a levered return over `steps` periods.

    Fragile: when leverage is extremely large (as injected by a stress scenario),
    the iterated product overflows float64 to Inf and NaNInfDetector fires.
    """
    result = float(base_return)
    for _ in range(steps):
        result = result * leverage      # repeated multiply — overflows with huge leverage
    return result
