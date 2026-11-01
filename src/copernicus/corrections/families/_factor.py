"""One-factor helpers shared by the pooling families.

A single principal-axis factor is extracted from a correlation matrix of standardised score
columns, its sign is aligned with the row mean of the standardised scores, and the
loadings turn a standardised row into a factor score. The standardisation uses the
population standard deviation of the training rows, with a standard deviation of 1 for a
constant column (which therefore standardises to zeros and carries no weight).
"""

from __future__ import annotations

import logging

import numpy as np

from copernicus.errors import ConfigurationError
from copernicus.validity.metrics import is_constant

logger = logging.getLogger(__name__)

__all__ = [
    "align_loadings",
    "paf_one_factor",
    "regression_factor_score",
    "relation_sign",
    "sign_align",
    "standardize_apply",
    "standardize_fit",
    "weighted_factor_score",
]

PAF_MAX_ITER = 50
PAF_TOLERANCE = 1e-6
COMMUNALITY_FLOOR = 1e-3
LOADING_COMMUNALITY_FLOOR = 1e-6
REGRESSION_RIDGE = 1e-3


def standardize_fit(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column means and population standard deviations of a score matrix.

    Args:
        matrix: Rows by columns.

    Returns:
        ``(mean, sd)``; the standard deviation of a constant column is replaced by 1.
    """
    values = np.asarray(matrix, dtype=float)
    mean = values.mean(axis=0)
    sd = values.std(axis=0)
    constant = np.array([is_constant(values[:, index]) for index in range(values.shape[1])])
    return mean, np.where(constant, 1.0, sd)


def standardize_apply(matrix: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Standardise a score matrix with fitted means and standard deviations.

    Args:
        matrix: Rows by columns.
        mean: Column means from ``standardize_fit``.
        sd: Column standard deviations from ``standardize_fit``.

    Returns:
        ``(matrix - mean) / sd``.
    """
    return (np.asarray(matrix, dtype=float) - mean) / sd


def paf_one_factor(
    correlation: np.ndarray, n_iter: int = PAF_MAX_ITER, tol: float = PAF_TOLERANCE
) -> np.ndarray:
    """First-factor loadings by iterated principal-axis factoring.

    The diagonal of the correlation matrix is replaced by communalities that are updated
    until they change by less than ``tol`` (or ``n_iter`` rounds); the loadings are the top
    eigenvector scaled by the square root of its eigenvalue. The initial communalities are
    the squared multiple correlations ``1 - 1 / diag(R^-1)``; when the matrix is singular
    (more columns than rows, or a constant column) they are the largest absolute
    off-diagonal correlation of each column. The sign is arbitrary.

    Args:
        correlation: A square correlation matrix without NaN.
        n_iter: Maximum number of rounds.
        tol: Stopping tolerance on the change of the communalities.

    Returns:
        The loadings, one per column.

    Raises:
        ConfigurationError: If the matrix is not square.
    """
    matrix = np.asarray(correlation, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ConfigurationError(
            f"the correlation matrix must be square; got shape {tuple(matrix.shape)}"
        )
    size = matrix.shape[0]
    if np.linalg.matrix_rank(matrix) == size:
        communality = 1.0 - 1.0 / np.diag(np.linalg.inv(matrix))
    else:
        communality = np.max(np.abs(matrix - np.diag(np.diag(matrix))), axis=1)
    communality = np.clip(communality, COMMUNALITY_FLOOR, 1.0)
    loadings = np.zeros(size)
    for _ in range(n_iter):
        reduced = matrix.copy()
        np.fill_diagonal(reduced, communality)
        values, vectors = np.linalg.eigh(reduced)
        top = float(values[-1])
        if top <= 0.0:
            break
        loadings = vectors[:, -1] * np.sqrt(top)
        updated = np.clip(loadings**2, LOADING_COMMUNALITY_FLOOR, 1.0)
        converged = bool(np.max(np.abs(updated - communality)) < tol)
        communality = updated
        if converged:
            break
    return loadings


def relation_sign(scores: np.ndarray, reference: np.ndarray) -> float:
    """The sign that makes the scores correlate non-negatively with the reference.

    Args:
        scores: Values to orient.
        reference: Values that fix the orientation.

    Returns:
        -1.0 when the two correlate negatively, otherwise 1.0 (also when either is constant).
    """
    if is_constant(scores) or is_constant(reference):
        return 1.0
    return -1.0 if float(np.corrcoef(scores, reference)[0, 1]) < 0.0 else 1.0


def sign_align(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip the scores when they correlate negatively with the reference.

    Args:
        scores: Values to orient.
        reference: Values that fix the orientation.

    Returns:
        ``scores`` or ``-scores``; unchanged when either input is constant.
    """
    values = np.asarray(scores, dtype=float)
    return relation_sign(values, np.asarray(reference, dtype=float)) * values


def align_loadings(
    standardised: np.ndarray, loadings: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Flip the loadings so that the factor score does not correlate negatively with the reference.

    Args:
        standardised: The standardised score matrix the loadings are applied to.
        loadings: Loadings from ``paf_one_factor``.
        reference: Values that fix the orientation, such as the row mean of the matrix.

    Returns:
        ``loadings`` or ``-loadings``.
    """
    score = weighted_factor_score(standardised, loadings)
    return relation_sign(score, np.asarray(reference, dtype=float)) * np.asarray(
        loadings, dtype=float
    )


def weighted_factor_score(standardised: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """The weighted-sum factor score ``standardised @ loadings``.

    It needs no matrix inverse, so it also works when there are more columns than rows.

    Args:
        standardised: Standardised score matrix.
        loadings: Loadings, one per column.

    Returns:
        One score per row.
    """
    return np.asarray(standardised, dtype=float) @ np.asarray(loadings, dtype=float)


def regression_factor_score(
    standardised: np.ndarray,
    correlation: np.ndarray,
    loadings: np.ndarray,
    ridge: float = REGRESSION_RIDGE,
) -> np.ndarray:
    """The regression factor score ``standardised @ (R + ridge I)^-1 @ loadings``.

    A small ridge on the diagonal keeps the solve stable. Use it for a few well-conditioned
    columns.

    Args:
        standardised: Standardised score matrix.
        correlation: The correlation matrix of the columns.
        loadings: Loadings, one per column.
        ridge: Value added to the diagonal.

    Returns:
        One score per row.
    """
    matrix = np.asarray(correlation, dtype=float)
    weights = np.linalg.solve(
        matrix + ridge * np.eye(matrix.shape[0]), np.asarray(loadings, dtype=float)
    )
    return np.asarray(standardised, dtype=float) @ weights
