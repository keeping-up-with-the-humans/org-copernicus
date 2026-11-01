"""Association between a score and a gold value, finite pairing and min-max scaling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr

from copernicus.errors import ConfigurationError, EstimatorError

logger = logging.getLogger(__name__)

__all__ = [
    "Association",
    "AssociationStatus",
    "association",
    "auc_from_ranks",
    "finite_pairs",
    "is_binary",
    "is_constant",
    "minmax01",
]

MIN_PAIRS = 3
CONSTANT_RELATIVE_TOLERANCE = 1e-12

AssociationStatus = Literal["ok", "too_few", "constant_score", "constant_gold"]
"""The status of an association: ``ok``, or why its metrics are missing.

The reasons are too few finite pairs (``too_few``), a constant score and a constant gold.
"""


@dataclass(frozen=True)
class Association:
    """Association metrics of a score with a gold value.

    Attributes:
        n: Number of rows where both score and gold are finite.
        spearman: Signed Spearman rank correlation.
        rank_validity: Absolute Spearman rank correlation.
        pearson: Pearson correlation.
        kendall: Kendall's tau-b.
        auc: Area under the ROC curve of the score for the larger gold value; NaN unless
            gold has exactly two distinct values.
        somers_d: ``2 * auc - 1``; NaN unless gold has exactly two distinct values.
        status: ``ok``, or why the metrics are NaN: ``too_few`` (fewer than three finite
            pairs), ``constant_score`` or ``constant_gold`` (numerically zero range).
    """

    n: int
    spearman: float
    rank_validity: float
    pearson: float
    kendall: float
    auc: float
    somers_d: float
    status: AssociationStatus


def finite_pairs(score: np.ndarray, gold: np.ndarray, *extra: np.ndarray) -> tuple[np.ndarray, ...]:
    """Keep the rows where every array is finite.

    Args:
        score: Score values.
        gold: Gold values, of the same length.
        *extra: Further arrays of the same length (a covariate, for example).

    Returns:
        The float arrays restricted to the rows where all of them are finite, in the
        order of the call and in the original row order.

    Raises:
        ConfigurationError: If an array is not one-dimensional or the lengths differ.
    """
    arrays = [np.asarray(values, dtype=float) for values in (score, gold, *extra)]
    shapes = [values.shape for values in arrays]
    if any(values.ndim != 1 for values in arrays) or len({shape[0] for shape in shapes}) != 1:
        raise ConfigurationError(
            f"score, gold and the extra arrays must be one-dimensional and of equal length; "
            f"got shapes {shapes}"
        )
    keep = np.isfinite(arrays[0])
    for values in arrays[1:]:
        keep &= np.isfinite(values)
    return tuple(values[keep] for values in arrays)


def is_constant(x: np.ndarray) -> bool:
    """Whether the values are constant up to floating-point noise.

    True when the range is at most ``1e-12`` times the larger of 1 and the largest absolute
    value, so a column that differs only in the last bits counts as constant. The floor is
    absolute below a scale of 1: values whose whole range is at most ``1e-12`` (a score on a
    scale below ``1e-12``) are treated as constant. The values must be finite; an empty
    array is constant.

    Args:
        x: Finite values.

    Returns:
        True for a constant or numerically constant input.
    """
    values = np.asarray(x, dtype=float)
    if values.size == 0:
        return True
    scale = max(1.0, float(np.max(np.abs(values))))
    return bool(np.ptp(values) <= CONSTANT_RELATIVE_TOLERANCE * scale)


def is_binary(gold: np.ndarray) -> bool:
    """Whether the finite values of ``gold`` take exactly two distinct values."""
    values = np.asarray(gold, dtype=float)
    return int(np.unique(values[np.isfinite(values)]).size) == 2


def minmax01(x: np.ndarray) -> np.ndarray:
    """Scale to the unit interval with the observed minimum and maximum.

    Non-finite entries stay NaN and do not enter the range.

    Args:
        x: Values to scale.

    Returns:
        ``(x - min) / (max - min)`` over the finite values.

    Raises:
        EstimatorError: If no finite value exists or the finite values have numerically zero range.
    """
    values = np.asarray(x, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise EstimatorError("min-max scaling needs at least one finite value")
    low, high = float(finite.min()), float(finite.max())
    if is_constant(finite):
        raise EstimatorError(f"min-max scaling is undefined: every finite value equals {low!r}")
    scaled = np.full(values.shape, np.nan)
    mask = np.isfinite(values)
    scaled[mask] = (values[mask] - low) / (high - low)
    return scaled


def auc_from_ranks(score: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve from average ranks.

    The Mann-Whitney U statistic over ``n_positive * n_negative``, so ties between a
    positive and a negative score count one half and a constant score gives 0.5. Both
    classes must be present.

    Args:
        score: Score values, finite.
        positive: Boolean mask of the positive class, of the same length.

    Returns:
        The probability that a random positive outranks a random negative.
    """
    ranks = rankdata(score)
    n_positive = int(positive.sum())
    n_negative = positive.size - n_positive
    u_statistic = float(ranks[positive].sum()) - n_positive * (n_positive + 1) / 2.0
    return u_statistic / (n_positive * n_negative)


def association(score: np.ndarray, gold: np.ndarray) -> Association:
    """Association metrics of ``score`` with ``gold`` over their finite pairs.

    Rank validity is the absolute Spearman correlation. AUC and Somers' D are computed
    only when gold has exactly two distinct finite values, with the larger value as the
    positive class. Statuses are checked in the order too few pairs, constant score,
    constant gold; the metric fields are NaN for any status other than ``ok``.

    Args:
        score: Score values.
        gold: Gold values, of the same length.

    Returns:
        The metrics, the number of finite pairs and the status.

    Raises:
        ConfigurationError: If the arrays differ in length.
    """
    pair_score, pair_gold = finite_pairs(score, gold)
    n = int(pair_score.size)
    nan = float("nan")
    status: AssociationStatus = "ok"
    if n < MIN_PAIRS:
        status = "too_few"
    elif is_constant(pair_score):
        status = "constant_score"
    elif is_constant(pair_gold):
        status = "constant_gold"
    if status != "ok":
        logger.debug("association not computed: status %s on %d finite pairs", status, n)
        return Association(n, nan, nan, nan, nan, nan, nan, status)
    rho = float(spearmanr(pair_score, pair_gold).statistic)
    auc = nan
    somers_d = nan
    if is_binary(pair_gold):
        auc = auc_from_ranks(pair_score, pair_gold == pair_gold.max())
        somers_d = 2.0 * auc - 1.0
    return Association(
        n=n,
        spearman=rho,
        rank_validity=abs(rho),
        pearson=float(pearsonr(pair_score, pair_gold).statistic),
        kendall=float(kendalltau(pair_score, pair_gold).statistic),
        auc=auc,
        somers_d=somers_d,
        status=status,
    )
