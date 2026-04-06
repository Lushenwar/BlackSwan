"""
Test fixture: mean-variance portfolio with covariance inversion.

Exercises np.linalg.inv (ConditionNumberDetector target) and the @
matrix-multiply operator (MatrixPSDDetector target).
Robust at default inputs; fragile when the covariance matrix is perturbed
toward singularity.
"""
import numpy as np


def compute_mv_weights(mu=None, cov=None):
    """
    Compute minimum-variance weights via the inverse covariance matrix.

    Returns a dict so BoundsDetector can inspect the 'weights' key.
    """
    if mu is None:
        mu = np.array([0.05, 0.07, 0.04])
    if cov is None:
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.002],
            [0.005, 0.002, 0.06],
        ])

    inv_cov = np.linalg.inv(cov)                          # ← linalg.inv call
    ones = np.ones(len(mu))
    raw_weights = inv_cov @ ones                          # ← @ operator
    weights = raw_weights / raw_weights.sum()             # ← division

    portfolio_var = float(weights @ cov @ weights)        # ← @ operator
    return {"weights": weights, "var": portfolio_var}
