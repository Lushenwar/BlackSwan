"""
Test fixture: clean portfolio computation.

Returns a finite scalar; no detector should fire under any preset scenario.
Used as the negative test case in CLI and integration tests.
"""
import numpy as np


def compute_portfolio_value(weights=None, prices=None):
    """
    Compute a simple dot-product portfolio value.

    All parameters have defaults so the CLI can auto-detect base inputs.
    No division, no matrix ops — this function is unconditionally safe.
    """
    if weights is None:
        weights = np.array([0.4, 0.3, 0.3])
    if prices is None:
        prices = np.array([100.0, 150.0, 200.0])
    return float(np.dot(weights, prices))
