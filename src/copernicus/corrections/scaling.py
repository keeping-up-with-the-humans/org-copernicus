"""Choosing the raw score column and scaling held-out values with training ranges."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from copernicus.corrections._messages import list_first_ten
from copernicus.errors import EstimatorError
from copernicus.validity.metrics import finite_pairs, is_constant

logger = logging.getLogger(__name__)

__all__ = ["best_score_column", "minmax_apply", "minmax_fit", "scale_test"]

MIN_PAIRS = 3


def best_score_column(scores: pd.DataFrame, gold: np.ndarray) -> str:
    """The score column with the highest signed Spearman correlation with the gold values.

    Only the rows where both the score and the gold value are finite count. A column that
    is constant over those rows, or has fewer than three such rows, is skipped; on a tie
    the first column wins.

    Args:
        scores: Score columns, one row per text.
        gold: Gold values aligned with the rows, NaN where unlabelled.

    Returns:
        The name of the best column.

    Raises:
        ConfigurationError: If the lengths differ.
        EstimatorError: If no column is usable (all constant, too few pairs, or constant gold).
    """
    gold_values = np.asarray(gold, dtype=float)
    best_name: str | None = None
    best_value = -np.inf
    skipped: list[str] = []
    for name in scores.columns:
        score, paired_gold = finite_pairs(scores[name].to_numpy(dtype=float), gold_values)
        if score.size < MIN_PAIRS or is_constant(score) or is_constant(paired_gold):
            skipped.append(str(name))
            continue
        value = float(spearmanr(score, paired_gold).statistic)
        if value > best_value:
            best_name, best_value = str(name), value
    if best_name is None:
        raise EstimatorError(
            "no usable score column: each of "
            f"{list_first_ten(skipped)} is constant, has fewer than {MIN_PAIRS} labelled "
            "rows, or meets a constant gold value"
        )
    return best_name


def minmax_fit(x: np.ndarray) -> tuple[float, float]:
    """The minimum and maximum of the finite values.

    Args:
        x: Values; non-finite entries are ignored.

    Returns:
        ``(minimum, maximum)``.

    Raises:
        EstimatorError: If no value is finite.
    """
    values = np.asarray(x, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise EstimatorError("min-max range needs at least one finite value")
    return float(finite.min()), float(finite.max())


def minmax_apply(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Scale values with a fixed range: ``(x - lo) / (hi - lo)``.

    Values outside the range map outside the unit interval and NaN stays NaN. A training
    range of zero is refused rather than mapped to zeros, because a score or gold column
    that does not vary in the training rows has no scale to apply.

    Args:
        x: Values to scale.
        lo: Lower end of the range, fitted on training values.
        hi: Upper end of the range.

    Returns:
        The scaled values.

    Raises:
        EstimatorError: If the range is zero (numerically constant) or reversed.
    """
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi < lo or is_constant(np.array([lo, hi])):
        raise EstimatorError(
            f"min-max scaling is undefined for the training range [{lo!r}, {hi!r}]: "
            "the range must be positive"
        )
    return (np.asarray(x, dtype=float) - lo) / (hi - lo)


def scale_test(
    train_pred: np.ndarray,
    train_gold: np.ndarray,
    test_pred: np.ndarray,
    test_gold: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale held-out predictions and gold values with the ranges of the training values.

    The prediction and the gold value each use their own training minimum and maximum, so
    a held-out value outside the training range falls outside the unit interval and no
    information from the held-out rows enters the scaling.

    Args:
        train_pred: Predictions on the training rows.
        train_gold: Gold values on the training rows.
        test_pred: Predictions on the held-out rows.
        test_gold: Gold values on the held-out rows.

    Returns:
        ``(scaled test predictions, scaled test gold values)``.

    Raises:
        EstimatorError: If a training range is empty or zero.
    """
    pred_low, pred_high = minmax_fit(train_pred)
    gold_low, gold_high = minmax_fit(train_gold)
    return (
        minmax_apply(test_pred, pred_low, pred_high),
        minmax_apply(test_gold, gold_low, gold_high),
    )
