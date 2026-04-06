"""
Test fixture: volatility-adjusted return where the denominator (vol) can approach zero.

DivisionStabilityDetector should fire when vol is perturbed toward zero.
Silent when vol stays above epsilon.
"""
import numpy as np


def compute_vol_adjusted_return(returns=None, vol=0.2):
    """
    Normalise returns by volatility.

    Fragile: when vol is perturbed toward zero by a scenario, the division
    produces Inf or NaN and DivisionStabilityDetector fires.
    """
    if returns is None:
        returns = np.array([0.05, -0.02, 0.08, 0.01, -0.03])

    normalised = returns / vol          # ← division — proximate failure site
    excess = normalised - np.mean(normalised)
    return float(np.mean(excess))
