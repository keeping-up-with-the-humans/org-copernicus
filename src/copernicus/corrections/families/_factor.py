"""One-factor helpers shared by the pooling families.

A single principal-axis factor is extracted from a correlation matrix of standardised score
columns, its sign is aligned with the row mean of the standardised scores, and the
loadings turn a standardised row into a factor score. The standardisation subtracts each
column's computed mean on the training rows and divides by its population standard deviation
there, and by 1 when the column's range is at most ``1e-12`` times the larger of 1 and its
largest absolute value. An exactly constant column then has no variation, though rounding can
leave it at a constant value other than zero; a column that varies within that tolerance keeps
its variation; and such a column can still receive a loading.

Three choices that the data can leave open are recorded, not changed: the sign, when the scores
or the reference are constant or their computed correlation is at most ``1e-12`` in absolute
value (``orientation_identified``); the direction, when the two largest
eigenvalues of a round of the iteration are tied (``top_two_tied``); and the end of the
iteration, when it reaches its cap of rounds without meeting its tolerance (``paf_fit``).
``_open_cases`` holds the warning of each, which the callers log through their own loggers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from copernicus.errors import ConfigurationError
from copernicus.validity.metrics import is_constant

logger = logging.getLogger(__name__)

__all__ = [
    "ORIENTATION_TOLERANCE",
    "PAF_MAX_ITER",
    "PAF_TOLERANCE",
    "TIE_RELATIVE_TOLERANCE",
    "FactorFit",
    "align_loadings",
    "orientation_correlation",
    "orientation_identified",
    "paf_fit",
    "paf_one_factor",
    "regression_factor_score",
    "relation_sign",
    "sign_align",
    "standardize_apply",
    "standardize_fit",
    "top_two_tied",
    "weighted_factor_score",
]

PAF_MAX_ITER = 50
PAF_TOLERANCE = 1e-6
COMMUNALITY_FLOOR = 1e-3
LOADING_COMMUNALITY_FLOOR = 1e-6
REGRESSION_RIDGE = 1e-3
TIE_RELATIVE_TOLERANCE = 1e-12
"""Two largest values that differ by at most this share of the largest count as tied."""
ORIENTATION_TOLERANCE = 1e-12
"""A computed correlation of at most this size is treated as too small to identify a sign."""


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


def top_two_tied(values: np.ndarray) -> bool:
    """Whether the two largest of the values are equal up to a relative ``1e-12``.

    The rule: the largest minus the second largest is at most ``TIE_RELATIVE_TOLERANCE``
    (``1e-12``) times the absolute value of the largest. Applied to the eigenvalues of a
    symmetric matrix, or to the singular values of a matrix, it marks the direction of the
    largest as not unique: for two equal values every unit vector in the plane of their two
    directions is a direction of the largest, and numpy returns one of them; values closer than
    the threshold are marked the same way.

    Args:
        values: The eigenvalues or singular values, in any order.

    Returns:
        True for a tie; False with fewer than two values.
    """
    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.size < 2:
        return False
    top, second = float(ordered[-1]), float(ordered[-2])
    return top - second <= TIE_RELATIVE_TOLERANCE * abs(top)


@dataclass(frozen=True, eq=False)
class FactorFit:
    """The loadings of the principal-axis iteration and how the iteration ended.

    Attributes:
        loadings: The first-factor loadings, one per column; the sign is arbitrary.
        rounds: The rounds run, each one eigendecomposition of the reduced matrix.
        converged: Whether the iteration stopped because the communalities changed by less
            than the tolerance in its last round; False when it reached its cap of rounds
            without meeting the tolerance, True when it met it in its last allowed round.
        tied_rounds: The rounds, counted from 1, in which the two largest eigenvalues of the
            reduced matrix were tied by ``top_two_tied``.
    """

    loadings: np.ndarray
    rounds: int
    converged: bool
    tied_rounds: tuple[int, ...]

    @property
    def direction_unique(self) -> bool:
        """Whether no round had two tied largest eigenvalues."""
        return not self.tied_rounds


def paf_fit(
    correlation: np.ndarray, n_iter: int = PAF_MAX_ITER, tol: float = PAF_TOLERANCE
) -> FactorFit:
    """First-factor loadings by iterated principal-axis factoring, with how it ended.

    The diagonal of the correlation matrix is replaced by communalities that are updated
    until they change by less than ``tol`` (or ``n_iter`` rounds); the loadings are the top
    eigenvector scaled by the square root of its eigenvalue. The initial communalities are
    the squared multiple correlations ``1 - 1 / diag(R^-1)``; when the matrix is singular
    (more columns than rows, or a constant column) they are the largest absolute
    off-diagonal correlation of each column. The starting communalities are kept between
    ``1e-3`` and 1, and those of every later round, the squared loadings, between ``1e-6`` and
    1. The loadings themselves are not bounded: an improper solution (a Heywood case) returns a
    loading above 1 in absolute value while its communality on the diagonal stays at 1. The
    sign is arbitrary. A round whose two
    largest eigenvalues are tied (``top_two_tied``) takes the eigenvector that numpy returns,
    and an iteration that reaches ``n_iter`` rounds returns the loadings of its last round;
    both are recorded in the result.

    Args:
        correlation: A square correlation matrix without NaN.
        n_iter: Maximum number of rounds.
        tol: Stopping tolerance on the change of the communalities.

    Returns:
        The loadings, the rounds run, whether they converged and the rounds with a tie.

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
    rounds, converged, tied = 0, False, []
    for _ in range(n_iter):
        reduced = matrix.copy()
        np.fill_diagonal(reduced, communality)
        values, vectors = np.linalg.eigh(reduced)
        rounds += 1
        top = float(values[-1])
        if top <= 0.0:
            break
        if top_two_tied(values):
            tied.append(rounds)
        loadings = vectors[:, -1] * np.sqrt(top)
        updated = np.clip(loadings**2, LOADING_COMMUNALITY_FLOOR, 1.0)
        converged = bool(np.max(np.abs(updated - communality)) < tol)
        communality = updated
        if converged:
            break
    return FactorFit(loadings=loadings, rounds=rounds, converged=converged, tied_rounds=tuple(tied))


def paf_one_factor(
    correlation: np.ndarray, n_iter: int = PAF_MAX_ITER, tol: float = PAF_TOLERANCE
) -> np.ndarray:
    """First-factor loadings by iterated principal-axis factoring (the loadings of ``paf_fit``).

    Args:
        correlation: A square correlation matrix without NaN.
        n_iter: Maximum number of rounds.
        tol: Stopping tolerance on the change of the communalities.

    Returns:
        The loadings, one per column.

    Raises:
        ConfigurationError: If the matrix is not square.
    """
    return paf_fit(correlation, n_iter, tol).loadings


def orientation_correlation(scores: np.ndarray, reference: np.ndarray) -> float:
    """The correlation whose sign orients the scores, NaN when either input counts as constant.

    Args:
        scores: Values to orient.
        reference: Values that fix the orientation.

    Returns:
        The Pearson correlation, or NaN when the scores or the reference are treated as
        constant (``is_constant``): the correlation is then not consulted, also for values
        that vary within that rule's tolerance and have one.
    """
    if is_constant(scores) or is_constant(reference):
        return float("nan")
    return float(np.corrcoef(scores, reference)[0, 1])


def orientation_identified(scores: np.ndarray, reference: np.ndarray) -> bool:
    """Whether the correlation with the reference fixes the sign of the scores.

    It does not when the scores or the reference are treated as constant (``is_constant``):
    no correlation is consulted, and ``relation_sign`` returns 1.0, a convention. The computed
    correlation is also treated as too small to identify the sign when it is at most
    ``ORIENTATION_TOLERANCE`` (``1e-12``) in absolute value; ``relation_sign`` still returns
    the sign of the computed number. The threshold is a convention: a correlation below it
    can be genuine, and rounding can leave one above it, as for values that are large beside
    their spread. It covers two columns that are both scaled to the variance 1 and correlate
    negatively: their first component is their difference, which does not correlate with
    their equally weighted mean in exact arithmetic, so the computed correlation is of the
    size of rounding errors.

    Args:
        scores: Values to orient.
        reference: Values that fix the orientation.

    Returns:
        False when either is constant or the two correlate at most ``1e-12`` in absolute
        value, otherwise True.
    """
    return bool(abs(orientation_correlation(scores, reference)) > ORIENTATION_TOLERANCE)


def relation_sign(scores: np.ndarray, reference: np.ndarray) -> float:
    """The sign by which the scores are multiplied to orient them on the reference.

    Args:
        scores: Values to orient.
        reference: Values that fix the orientation.

    Returns:
        -1.0 when the computed correlation of the two is negative, otherwise 1.0, also when
        either is treated as constant (``is_constant``): no correlation is then consulted,
        so values that vary within that rule's tolerance keep a negative correlation. Where
        ``orientation_identified`` is False the sign is a convention, or the sign of a
        correlation of at most ``1e-12`` in absolute value.
    """
    return -1.0 if orientation_correlation(scores, reference) < 0.0 else 1.0


def sign_align(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip the scores when their computed correlation with the reference is negative.

    Args:
        scores: Values to orient.
        reference: Values that fix the orientation.

    Returns:
        ``scores`` or ``-scores``; unchanged when either input is treated as constant
        (``is_constant``), whatever their correlation (see `relation_sign`).
    """
    values = np.asarray(scores, dtype=float)
    return relation_sign(values, np.asarray(reference, dtype=float)) * values


def align_loadings(
    standardised: np.ndarray, loadings: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Flip the loadings so that the factor score does not correlate negatively with the reference.

    The loadings are left as they are when the factor score or the reference is treated as
    constant (``is_constant``), whatever their correlation (see `relation_sign`).

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
        ridge: Value added to the diagonal, ``1e-3`` by default.

    Returns:
        One score per row.
    """
    matrix = np.asarray(correlation, dtype=float)
    weights = np.linalg.solve(
        matrix + ridge * np.eye(matrix.shape[0]), np.asarray(loadings, dtype=float)
    )
    return np.asarray(standardised, dtype=float) @ weights
