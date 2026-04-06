"""
Test fixture: portfolio variance calculation with a fragile covariance matrix.

When correlation_shift >= 0.21, pairwise correlations exceed 1.0 (base 0.8 + shift),
causing the covariance matrix to lose positive semi-definiteness.
MatrixPSDDetector should resolve to the cov_matrix construction line.
"""
import numpy as np


def calculate_portfolio_var(
    weights: np.ndarray,
    vol: np.ndarray,
    correlation_shift: float = 0.0,
) -> np.ndarray:
    """
    Compute portfolio covariance matrix under a perturbed correlation regime.

    Returns the covariance matrix directly so detectors can inspect it.
    Fragile: when correlation_shift pushes pairwise correlations above 1.0,
    the matrix becomes non-PSD.
    """
    n = len(vol)

    # Perturbed correlation matrix — uniform pairwise correlation
    corr_val = 0.8 + correlation_shift
    corr_matrix = np.full((n, n), corr_val)
    np.fill_diagonal(corr_matrix, 1.0)

    # Covariance matrix — this is the proximate failure site.
    # Uses @ (matrix multiplication) so TracebackResolver can identify it.
    cov_matrix = np.diag(vol) @ corr_matrix @ np.diag(vol)

    return cov_matrix
