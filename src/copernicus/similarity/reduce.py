"""Reduction of a text-by-reference matrix to one score per text.

scikit-learn is imported by the first principal-component reduction, not with the package,
so that importing copernicus stays fast.
"""

from __future__ import annotations

import logging

import numpy as np

from copernicus.errors import ConfigurationError

logger = logging.getLogger(__name__)

__all__ = ["REDUCTIONS", "reduce_matrix"]

REDUCTIONS = ("mean", "max", "pca")
"""The reductions of the cosines to the references of a pole: ``mean``, ``max`` and ``pca``."""
MINIMUM_ROWS_FOR_PCA = 2


def _first_component(matrix: np.ndarray, label: str) -> np.ndarray:
    """Return the first principal component of the standardised columns, signed.

    The sign of a principal component is arbitrary, so it is anchored: the scores are
    flipped when they correlate negatively with the row means, which makes a higher
    similarity to the references mean a higher score.

    Raises:
        ConfigurationError: If the matrix has fewer than two rows, no column varies, or
            the decomposition fails or yields non-finite scores.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n_rows = matrix.shape[0]
    if n_rows < MINIMUM_ROWS_FOR_PCA:
        raise ConfigurationError(
            f"reduction of {label}: pca needs at least {MINIMUM_ROWS_FOR_PCA} texts, got {n_rows}"
        )
    if not np.any(matrix.var(axis=0) > 0):
        raise ConfigurationError(
            f"reduction of {label}: pca is undefined because every reference column is constant"
        )
    try:
        scores = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(matrix))[:, 0]
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        raise ConfigurationError(f"reduction of {label}: pca failed: {error}") from error
    if not np.all(np.isfinite(scores)):
        raise ConfigurationError(f"reduction of {label}: pca produced non-finite scores")
    row_means = matrix.mean(axis=1)
    if np.std(scores) > 0 and np.std(row_means) > 0 and np.corrcoef(scores, row_means)[0, 1] < 0:
        scores = -scores
    return scores


def reduce_matrix(mat: np.ndarray, method: str, label: str) -> np.ndarray:
    """Reduce a matrix with one row per text and one column per reference to one score per text.

    A matrix with a single column reduces to that column whatever the method; that is the
    correct reduction of one reference, not a fallback.

    Args:
        mat: Matrix of shape ``[N, K]``: cosines to ``K`` references, or the scores of
            ``K`` attributes in a roll-up.
        method: ``"mean"``, ``"max"``, or ``"pca"`` (first principal component of the
            standardised columns, signed to correlate positively with the row means).
        label: What is being reduced, named in every error message.

    Returns:
        Array of shape ``[N]``.

    Raises:
        ConfigurationError: If the method is unknown, the matrix is not two-dimensional,
            has no column, or holds a NaN or infinite value, or pca is asked of fewer than
            two texts or of columns without any variance.
    """
    if method not in REDUCTIONS:
        raise ConfigurationError(
            f"reduction of {label}: unknown reduction {method!r}; use one of {REDUCTIONS}"
        )
    if mat.ndim != 2:
        raise ConfigurationError(
            f"reduction of {label}: expected a two-dimensional matrix, got shape {mat.shape}"
        )
    if mat.shape[1] == 0:
        raise ConfigurationError(f"reduction of {label}: the matrix has no column")
    if not np.all(np.isfinite(mat)):
        raise ConfigurationError(f"reduction of {label}: the matrix holds a NaN or infinite value")
    if mat.shape[1] == 1:
        return mat[:, 0]
    if method == "mean":
        return mat.mean(axis=1)
    if method == "max":
        return mat.max(axis=1)
    return _first_component(mat, label)
